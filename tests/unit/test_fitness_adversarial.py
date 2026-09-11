"""Degenerate paths to a high score, kept permanently closed.

Every test here constructs a *specific exploit* — a trade ledger or an equity
curve that scored well without corresponding to an edge — and asserts it is
refused. They are written as adversarial fixtures rather than as one-off
regression checks: each states the property that makes the exploit impossible,
so a future change that reopens the hole by another route fails here too.

The three exploits recorded so far were all found by asking what an optimiser
would reach for, not by asking what a bug looks like:

1. an economically meaningless loss bought a perfect risk-adjusted score, so the
   search was paid to inject noise;
2. a do-nothing candidate read as maximally diverse, propping up the population
   diversity measure and suppressing the immigrant boost;
3. a ledger whose percentage expectancy and absolute total disagree in sign
   passed every gate with the trade-removal penalty left inert.

None of them is fixed by moving a threshold. Raising a threshold moves an
exploit; each fix here changes what the measure *means* at its degenerate edge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.core.config import load_config
from quantlab.core.fitness import InnerFoldReport, compute_fitness
from quantlab.core.metrics import compute_metrics
from quantlab.core.types import BacktestResult, Trade
from quantlab.core.validation.concentration import trade_removal_report
from quantlab.evolution.diversity import (
    CandidateView,
    Signature,
    agreement,
    population_diversity,
    select_survivors,
)

BAR_MS = 3_600_000


@pytest.fixture(scope="module")
def settings():
    return load_config().evolution


def a_trade(pnl: float, pnl_pct: float, i: int = 0) -> Trade:
    return Trade(
        trade_no=i,
        side="long",
        entry_ts=i * BAR_MS,
        entry_px=100.0,
        exit_ts=(i + 1) * BAR_MS,
        exit_px=100.0 * (1.0 + pnl_pct),
        qty=1.0,
        fees=0.0,
        slippage_cost=0.0,
        pnl=pnl,
        pnl_pct=pnl_pct,
        bars_held=1,
        exit_reason="signal",
    )


def a_result(equity, trades) -> BacktestResult:
    n = len(equity)
    return BacktestResult(
        equity=pd.Series(equity, dtype="float64"),
        position_frac=pd.Series(np.zeros(n), dtype="float64"),
        signals=pd.Series(["flat"] * n),
        trades=tuple(trades),
        bars_per_year=8760,
        warmup_bars=0,
        n_bars=n,
        cost_summary={"turnover": 0.0},
    )


def score(equity, trades, settings):
    metrics = compute_metrics(a_result(equity, trades))
    return metrics, compute_fitness(metrics, trades, InnerFoldReport(), None, settings.fitness)


# ---------------------------------------------------------------------------
# 1. buying a risk-adjusted score with a loss that is not economically real
# ---------------------------------------------------------------------------
#: A rising curve, and dips small enough to be nothing at the scale of the
#: returns it is made of. Per-bar returns here are ~1e-3.
NEGLIGIBLE_DIPS = (1e-9, 1e-6, 1e-3, 1e-1)

RISING = [10_000.0 + 10.0 * i for i in range(500)]
THIRTY_WINNERS = [a_trade(10.0, 0.001, i) for i in range(30)]


def _with_dip(equity: list[float], size: float, at: int = 250) -> list[float]:
    dipped = list(equity)
    dipped[at] = dipped[at - 1] - size
    return dipped


@pytest.mark.parametrize("dip", NEGLIGIBLE_DIPS)
def test_a_negligible_loss_cannot_buy_a_risk_adjusted_score(dip, settings) -> None:
    """Sortino must not become measurable because of one economically null bar.

    The exploit: a perfectly rising curve scored ``risk_adjusted = 0`` because
    Sortino was undefined with no downside at all, while a single bar dipping by
    1e-6 produced a Sortino of 2.1e10 that clipped to a *perfect* 1.0. Fitness
    rose 31 % for a loss of one hundred-thousandth of a currency unit.
    """
    _clean_metrics, clean = score(RISING, THIRTY_WINNERS, settings)
    metrics, dipped = score(_with_dip(RISING, dip), THIRTY_WINNERS, settings)

    assert metrics.sortino is None, (
        f"a dip of {dip} is noise at this return scale and must not define Sortino"
    )
    assert dipped.components["risk_adjusted"] == 0.0
    # Not equality: a dip still lowers the final equity a little, so fitness may
    # fall. What must never happen is that it *rises* — that was the exploit.
    assert dipped.fitness <= clean.fitness + 1e-12


def test_adding_a_loss_never_increases_fitness(settings) -> None:
    """The general property, not just the instance above.

    Any exploit of this shape — whatever route it arrives by — makes fitness
    non-monotonic in the losses added to an otherwise unchanged run. Pinning
    monotonicity catches the next one as well as this one.
    """
    _m, baseline = score(RISING, THIRTY_WINNERS, settings)
    for dip in NEGLIGIBLE_DIPS:
        _m, worse = score(_with_dip(RISING, dip), THIRTY_WINNERS, settings)
        assert worse.fitness <= baseline.fitness + 1e-12, (
            f"a dip of {dip} raised fitness from {baseline.fitness} to {worse.fitness}"
        )


def test_a_real_loss_is_still_measured(settings) -> None:
    """The fix must not blind the metric to downside that genuinely happened.

    A guard that made Sortino undefined for *every* well-behaved curve would
    close the exploit by breaking the measure, which is the failure mode this
    test exists to refuse.
    """
    equity = [10_000.0 + 10.0 * i for i in range(500)]
    for i in range(100, 400, 7):
        equity[i] = equity[i - 1] - 25.0
    metrics, result = score(equity, THIRTY_WINNERS, settings)
    assert metrics.sortino is not None
    assert result.components["risk_adjusted"] > 0.0


# ---------------------------------------------------------------------------
# 2. scoring as diverse by doing nothing
# ---------------------------------------------------------------------------
N_BARS = 1_000
ACTIVE = np.where(np.arange(N_BARS) % 4 < 2, 1.0, 0.0)
OTHER_ACTIVE = np.where(np.arange(N_BARS) % 7 < 3, 1.0, 0.0)
NEVER_TRADES = np.zeros(N_BARS)

SHARED = Signature(triples=(("indicator", "sma", ">"),))


def _distinct(i: int) -> Signature:
    return Signature(triples=(("indicator", f"ind{i}", ">"),))


def test_a_candidate_that_never_trades_is_not_diverse() -> None:
    """It has shown no behaviour, which is not the same as a different one."""
    assert agreement(ACTIVE, NEVER_TRADES) == 1.0
    assert agreement(NEVER_TRADES, ACTIVE) == 1.0
    assert agreement(NEVER_TRADES, NEVER_TRADES) == 1.0


def test_two_active_candidates_that_never_overlap_are_still_diverse() -> None:
    """The fix must not swallow the case the measure is actually for.

    Two strategies that genuinely traded, and never at the same time, are
    different. Only *absent* behaviour is unmeasurable.
    """
    early = np.zeros(N_BARS)
    early[: N_BARS // 2] = 1.0
    late = np.zeros(N_BARS)
    late[N_BARS // 2 :] = 1.0
    assert agreement(early, late) == 0.0


def test_dead_candidates_cannot_prop_up_a_collapsed_population(settings) -> None:
    """A population of behavioural clones must read as collapsed regardless.

    The exploit: eight clones score 0.0000 — a total collapse — but adding two
    structurally distinct do-nothing candidates lifted the measure to 0.3644,
    above the 0.35 floor, so the immigrant boost stopped firing on exactly the
    population it exists to rescue.
    """
    floor = settings.diversity.min_population_diversity
    clones = [CandidateView(f"c{i}", 0.5, SHARED, ACTIVE) for i in range(8)]
    assert population_diversity(clones, settings.diversity) < floor

    for n_dead in range(1, 9):
        padded = clones + [
            CandidateView(f"d{i}", -1e9, _distinct(i), NEVER_TRADES) for i in range(n_dead)
        ]
        diversity = population_diversity(padded, settings.diversity)
        assert diversity < floor, (
            f"{n_dead} do-nothing candidate(s) lifted a collapsed population to "
            f"{diversity:.4f}, at or above the {floor} floor"
        )


def test_a_do_nothing_candidate_does_not_survive_niching(settings) -> None:
    """Nothing was similar to it, so niching used to accept it as a survivor —
    and survivors are the parents the next generation is mutated from."""
    ranked = [CandidateView(f"c{i}", 0.5, SHARED, ACTIVE) for i in range(8)] + [
        CandidateView(f"d{i}", -1e9, SHARED, NEVER_TRADES) for i in range(8)
    ]
    survivors = select_survivors(
        ranked, n_survivors=settings.n_survivors, settings=settings.diversity
    )
    assert not any(view.candidate_id.startswith("d") for view in survivors)


# ---------------------------------------------------------------------------
# 3. evading the trade-removal test by losing money
# ---------------------------------------------------------------------------
def a_ledger_whose_percent_and_total_disagree() -> list[Trade]:
    """Positive percentage expectancy, negative absolute total.

    Twenty small-notional winners at +5 % and ten large-notional losers at -1 %.
    Expectancy is the mean of ``pnl_pct`` (+0.03, so ``F_EXPECTANCY`` passes)
    while retention is a ratio of absolute ``pnl`` (total -20, so retention is
    undefined). Both ``F_CONCENTRATION`` and ``p_removal`` used to skip an
    undefined retention, so the one shape that defeats the invariant also
    disabled the two defences resting on it.
    """
    winners = [a_trade(1.0, 0.05, i) for i in range(20)]
    losers = [a_trade(-4.0, -0.01, 20 + i) for i in range(10)]
    return winners + losers


def test_the_ledger_that_defeats_the_retention_invariant_is_rejected(settings) -> None:
    trades = a_ledger_whose_percent_and_total_disagree()
    metrics, result = score([10_000.0] * 500, trades, settings)

    # the premise: it really does pass the expectancy gate and really does
    # leave retention undefined, or the test is no longer adversarial
    assert metrics.expectancy_pct is not None and metrics.expectancy_pct > 0.0
    assert sum(trade.pnl for trade in trades) < 0.0
    report = trade_removal_report(trades, settings.fitness.penalties.removal_k)
    assert not report.is_defined

    assert result.rejected
    assert result.gate_failure == "F_CONCENTRATION"


def test_an_undefined_retention_never_scores_as_a_clean_removal_test(settings) -> None:
    """Defence in depth: the penalty must not read 1.0 if the gate is reordered."""
    from quantlab.core.fitness import _removal_penalty

    report = trade_removal_report(
        a_ledger_whose_percent_and_total_disagree(), settings.fitness.penalties.removal_k
    )
    assert _removal_penalty(report, settings.fitness) == 0.0


def test_a_profitable_concentrated_ledger_is_still_measured_normally(settings) -> None:
    """The fix must not turn every undefined-looking ledger into a rejection."""
    trades = [a_trade(10.0, 0.001, i) for i in range(30)]
    report = trade_removal_report(trades, settings.fitness.penalties.removal_k)
    assert report.is_defined
    _metrics, result = score([10_000.0 + 10.0 * i for i in range(500)], trades, settings)
    assert not result.rejected


# ---------------------------------------------------------------------------
# 4. selecting the scoring period through the declared warm-up
# ---------------------------------------------------------------------------
def a_curve_that_falls_then_rises(n: int = 2000) -> list[float]:
    """The period-selection target: a bad first half and a good second half."""
    equity = [10_000.0]
    for i in range(1, n):
        equity.append(equity[-1] * (0.9995 if i < n // 2 else 1.0015))
    return equity


def test_a_genome_cannot_declare_warm_up_it_does_not_need() -> None:
    """CLOSED (ADR 0007). Warm-up is derived from the structure, both ways.

    Warm-up bars are excluded from every metric, so declaring more of them
    deletes the bars a candidate would be judged on. Rule 6 used to bound it
    from below only, and the excess was free score.
    """
    from pydantic import ValidationError as PydanticValidationError
    from tests.unit.test_compiler import rsi_reversion_genome

    from quantlab.core.genome import StrategyGenome

    honest = rsi_reversion_genome()
    needed = honest.max_lookback()
    assert honest.warmup_bars == needed

    with pytest.raises(PydanticValidationError, match="need only"):
        StrategyGenome.model_validate({**honest.model_dump(mode="json"), "warmup_bars": needed + 9})
    with pytest.raises(PydanticValidationError, match="indicators need"):
        StrategyGenome.model_validate({**honest.model_dump(mode="json"), "warmup_bars": needed - 1})


def test_the_metric_window_is_still_sensitive_to_warm_up(settings) -> None:
    """Why the bound above has to be exact, kept measurable.

    The sensitivity itself is not a bug — warm-up must be excluded, or a
    strategy is scored on bars where its indicators were NaN. What was a bug is
    that a candidate could choose the number. This records how much the choice
    was worth, so that if a future change reintroduces a way to set it, the size
    of the prize is already written down.
    """
    equity = a_curve_that_falls_then_rises()
    trades = [a_trade(10.0, 0.001, i) for i in range(40)]

    def at(warmup: int):
        result = a_result(equity, trades).model_copy(update={"warmup_bars": warmup})
        metrics = compute_metrics(result)
        return metrics, compute_fitness(
            metrics,
            trades,
            InnerFoldReport(),
            None,
            settings.fitness,
            # Pinned at the ceiling this fixture's recorded numbers were measured
            # under, so ADR 0012 making the ceiling relative does not silently
            # rewrite the size of the warm-up prize. The prize is the point; the
            # ceiling is scenery, and scenery that moves would make the two
            # constants below incomparable with the ones ADR 0007 recorded.
            benchmark_drawdown=0.50,
        )

    honest_metrics, honest = at(0)
    trimmed_metrics, trimmed = at(900)

    assert trimmed.fitness > honest.fitness
    assert trimmed_metrics.max_drawdown < honest_metrics.max_drawdown
    assert honest.fitness == pytest.approx(0.078220, abs=1e-5)
    assert trimmed.fitness == pytest.approx(0.295525, abs=1e-5)


def test_mutation_does_not_ratchet_warm_up_upward() -> None:
    """The route that made the hole reachable without intent is closed.

    `_compile` used `max(existing, required)`, so a lineage that once held a
    long-lookback indicator kept the long warm-up after mutating that indicator
    away — and was scored on a shorter, later window than its competitors ever
    after. Warm-up now follows the structure that is actually present.
    """
    import inspect

    from quantlab.evolution import mutation

    source = inspect.getsource(mutation)
    assert "max(\n                    self.genome.warmup_bars" not in source
    assert '"warmup_bars": required_warmup(conditions, self.schema)' in source


# ---------------------------------------------------------------------------
# 5. the invariant checker must survive the corruption it looks for
# ---------------------------------------------------------------------------
def test_a_stored_genome_that_will_not_validate_is_a_verdict_not_a_crash() -> None:
    """INV-10 has to be able to *report* corruption, not die of it.

    `genome_of` let a pydantic ValidationError escape, so a single unparseable
    stored genome aborted the whole run's audit instead of being recorded as one
    mismatched candidate — the verifier crashed on exactly what it exists to
    detect.
    """
    from quantlab.core.errors import StrategyError
    from quantlab.evolution.lineage import genome_of

    class Row:
        candidate_id = "c1"
        kind = "genome"
        genome_json = '{"name": "x", "warmup_bars": -1}'

    with pytest.raises(StrategyError, match="not a valid genome"):
        genome_of(Row())
