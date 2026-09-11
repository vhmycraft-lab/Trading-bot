"""``F_DRAWDOWN`` is relative to buy-and-hold over the same window (ADR 0012).

The gate used to be an absolute 0.50. Measured on the first real campaign,
buy-and-hold on BTC/USDT 1h drew down 0.7041-0.7720 across all eight training
windows — so the asset itself failed, in 8 windows out of 8, the gate applied to
strategies trading it. And an absolute limit cannot distinguish "55% drawdown
while the market fell 80%" from "45% while it fell 30%".

The property that matters is **window-locality**. With a per-generation
``TrainingEnvironment`` the evaluation window moves while the symbol and the
segment name stay put, so a benchmark taken from anything but the candidate's own
bars is stale without looking wrong. These tests fix that by construction: one
unchanged candidate is judged against two different windows and must get two
different answers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.core.config import load_config
from quantlab.core.fitness import FITNESS_REJECTED, InnerFoldReport, compute_fitness
from quantlab.core.metrics import benchmark_drawdown, compute_metrics
from quantlab.core.types import BacktestResult, Trade

BAR_MS = 3_600_000


@pytest.fixture(scope="module")
def settings():
    return load_config().evolution


def a_trade(pnl_pct: float, i: int) -> Trade:
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
        pnl=100.0 * pnl_pct,
        pnl_pct=pnl_pct,
        bars_held=1,
        exit_reason="signal",
    )


#: Forty winners: clears expectancy, clears the 30-trade floor. Drawdown is the
#: only gate left with anything to say.
WINNERS = [a_trade(0.004, i) for i in range(40)]


def _curve(depth: float, n: int = 600) -> list[float]:
    """An equity curve that rises, falls by ``depth``, and recovers past its peak."""
    peak, trough = 10_000.0, 10_000.0 * (1.0 - depth)
    up = np.linspace(10_000.0, peak, n // 3)
    down = np.linspace(peak, trough, n // 3)
    back = np.linspace(trough, peak * 1.6, n - 2 * (n // 3))
    return [float(v) for v in np.concatenate([up, down, back])]


def score(equity, settings, *, benchmark):
    result = BacktestResult(
        equity=pd.Series(equity, dtype="float64"),
        position_frac=pd.Series(np.zeros(len(equity)), dtype="float64"),
        signals=pd.Series(["flat"] * len(equity)),
        trades=tuple(WINNERS),
        bars_per_year=8760,
        warmup_bars=0,
        n_bars=len(equity),
        cost_summary={"turnover": 0.0},
    )
    metrics = compute_metrics(result)
    return compute_fitness(
        metrics,
        WINNERS,
        InnerFoldReport(),
        None,
        settings.fitness,
        benchmark_drawdown=benchmark,
    )


# ---------------------------------------------------------------------------
# window-locality: the same candidate, two windows, two verdicts
# ---------------------------------------------------------------------------
def test_the_same_candidate_passes_in_a_harsh_window_and_fails_in_a_calm_one(
    settings,
) -> None:
    """The whole point of ADR 0012, and the thing a global benchmark would break.

    A 60% drawdown is respectable when the asset fell 77% and indefensible when it
    fell 30%. Nothing about the candidate changes between these two calls — only
    the window it is being judged against. A benchmark cached by symbol or by
    segment name would return the same number for both and this test would fail.
    """
    candidate = _curve(0.60)

    harsh = score(candidate, settings, benchmark=0.77)
    calm = score(candidate, settings, benchmark=0.30)

    assert harsh.gate_failure is None, (
        "a 60% drawdown against a 77% benchmark is better than holding the asset"
    )
    assert calm.gate_failure == "F_DRAWDOWN"
    assert calm.fitness == FITNESS_REJECTED


def test_buy_and_hold_itself_is_exactly_at_the_line(settings) -> None:
    """At 1.00x, a candidate that tracks the asset's drawdown is admitted and
    anything worse is not. The boundary is asserted on both sides rather than
    assumed from the multiplier."""
    assert settings.fitness.gates.max_drawdown_vs_benchmark == 1.00

    at_the_line = score(_curve(0.7041), settings, benchmark=0.7041)
    just_over = score(_curve(0.7100), settings, benchmark=0.7041)

    assert at_the_line.gate_failure is None
    assert just_over.gate_failure == "F_DRAWDOWN"


def test_the_old_absolute_limit_would_have_rejected_all_of_these(settings) -> None:
    """The measured campaign finding, kept as a fixture.

    Of the 26 candidates that cleared expectancy and the trade floor, exactly one
    was under 0.50 and all 26 were under 1.25x buy-and-hold. This asserts the
    direction of that change on a constructed candidate rather than on the stored
    campaign, so it keeps meaning something after the database is gone.
    """
    realistic = _curve(0.7657)  # the campaign's median for that population
    assert score(realistic, settings, benchmark=0.7041).gate_failure == "F_DRAWDOWN"
    assert score(realistic, settings, benchmark=0.80).gate_failure is None
    # and the old absolute rule would have rejected it against any benchmark:
    assert 0.7657 > 0.50


# ---------------------------------------------------------------------------
# the fallback, and what it is not
# ---------------------------------------------------------------------------
def test_no_benchmark_falls_back_to_the_absolute_ceiling_rather_than_rejecting(
    settings,
) -> None:
    """A window too short to have a benchmark is our failure, not the candidate's.

    ``None`` must not read as "a benchmark of zero", which would reject every
    candidate that ever lost a cent.
    """
    modest = score(_curve(0.20), settings, benchmark=None)
    assert modest.gate_failure is None


def test_the_fallback_is_not_a_second_gate(settings) -> None:
    """Exactly one ceiling applies. A candidate under the relative limit is not
    then re-checked against the absolute one, or the relative gate would be
    decorative wherever the absolute is tighter."""
    gates = settings.fitness.gates
    generous_benchmark = 0.90
    assert gates.max_drawdown_vs_benchmark * generous_benchmark > 0.50
    deep = score(_curve(0.85), settings, benchmark=generous_benchmark)
    assert deep.gate_failure is None, (
        "a candidate inside the relative ceiling was rejected anyway; "
        "the absolute limit is still being applied alongside it"
    )


# ---------------------------------------------------------------------------
# the benchmark itself
# ---------------------------------------------------------------------------
def test_benchmark_drawdown_reads_the_window_it_is_handed() -> None:
    falling = np.array([100.0, 90.0, 80.0, 50.0, 60.0])
    assert benchmark_drawdown(falling) == pytest.approx(0.50)
    rising = np.array([100.0, 110.0, 120.0])
    assert benchmark_drawdown(rising) == pytest.approx(0.0)


@pytest.mark.parametrize("closes", [np.array([]), np.array([100.0])])
def test_a_window_too_short_to_have_a_drawdown_reports_none(closes) -> None:
    """Not 0.0. A benchmark of zero would make the relative ceiling zero and
    reject everything, which is the opposite of "we could not measure it"."""
    assert benchmark_drawdown(closes) is None


def test_the_benchmark_is_invariant_to_the_capital_it_is_scaled_by() -> None:
    """Buy-and-hold is the close series up to a constant, and drawdown does not
    see constants — which is why the close series can be used directly instead of
    simulating a position."""
    closes = np.array([100.0, 130.0, 70.0, 90.0])
    assert benchmark_drawdown(closes) == pytest.approx(benchmark_drawdown(closes * 37.5))
