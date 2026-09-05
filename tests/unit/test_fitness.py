"""Multi-objective fitness (master spec section 13.3, task T47).

The acceptance criteria are specific, and each has a test named after it: a
hand-computed toy example matching to 1e-12; a 90 %-win-rate strategy with
negative expectancy scoring ``FITNESS_REJECTED``; a 60 % drawdown scoring
``FITNESS_REJECTED`` regardless of every other component; and weights that do not
sum to 1 raising ``ConfigError``.

The last two are the ones that matter most. Section 13's central claim is that a
candidate cannot buy its way past a robustness problem with a good number
elsewhere, and a test that only checked the arithmetic would not notice if that
stopped being true.
"""

from __future__ import annotations

import math

import pytest
from tests.unit.test_concentration import trade

from quantlab.core.config import (
    FitnessGates,
    FitnessPenalties,
    FitnessSettings,
    FitnessTargets,
    FitnessWeights,
)
from quantlab.core.errors import ConfigError
from quantlab.core.fitness import (
    FITNESS_REJECTED,
    GATE_IDS,
    InnerFold,
    InnerFoldReport,
    SensitivityReport,
    compute_fitness,
)
from quantlab.core.metrics import MetricSet

SETTINGS = FitnessSettings()

#: A ledger that clears every gate: 100 net pnl, well spread, 40 round trips.
GOOD_TRADES = [
    *[trade(i, 5.0) for i in range(1, 31)],
    *[trade(i, -5.0) for i in range(31, 41)],
]


def metrics(**overrides: object) -> MetricSet:
    """A metric set that passes every gate, with fields replaced per test."""
    base: dict[str, object] = {
        "expectancy_pct": 0.002,
        "n_trades": 200.0,
        "max_drawdown": 0.10,
        "sortino": 1.5,
        "profit_factor": 1.5,
        "consistency": 1.0,
        "win_rate": 0.55,
        "cagr": 0.20,
        "top5_profit_share": 0.0,
    }
    base.update(overrides)
    return MetricSet(**base)  # type: ignore[arg-type]


def perfect_inner() -> InnerFoldReport:
    """Folds that generalise perfectly and identically, so no penalty fires."""
    return InnerFoldReport(
        folds=tuple(InnerFold(is_sortino=1.0, oos_sortino=1.0, oos_return=0.1) for _ in range(4))
    )


# ---------------------------------------------------------------------------
# stage 1 — the gates are absolute
# ---------------------------------------------------------------------------
def test_a_ninety_percent_win_rate_cannot_rescue_negative_expectancy() -> None:
    """Section 13.3's required property, stated as its own test.

    Many small wins and a few enormous losses is the classic shape of a strategy
    that looks wonderful and loses money. It must be rejected *before* the
    weighted score exists, or a high win-rate component could offset it.
    """
    result = compute_fitness(
        metrics(expectancy_pct=-0.001, win_rate=0.90, profit_factor=0.8),
        GOOD_TRADES,
        perfect_inner(),
        None,
        SETTINGS,
    )
    assert result.fitness == FITNESS_REJECTED
    assert result.gate_failure == "F_EXPECTANCY"
    assert result.components == {}
    assert result.rejected


def test_a_sixty_percent_drawdown_is_rejected_whatever_else_is_true() -> None:
    """Every other component at its best still scores ``FITNESS_REJECTED``."""
    result = compute_fitness(
        metrics(max_drawdown=0.60, sortino=10.0, cagr=5.0, consistency=1.0),
        GOOD_TRADES,
        perfect_inner(),
        None,
        SETTINGS,
    )
    assert result.fitness == FITNESS_REJECTED
    assert result.gate_failure == "F_DRAWDOWN"


def test_too_few_trades_is_rejected() -> None:
    result = compute_fitness(metrics(n_trades=5.0), GOOD_TRADES, perfect_inner(), None, SETTINGS)
    assert result.gate_failure == "F_TRADES"


def test_a_strategy_that_needs_its_best_trade_is_rejected() -> None:
    """``F_CONCENTRATION``: retention with the single best trade removed must
    stay above the gate. This ledger's profit *is* one trade."""
    ledger = [trade(1, 100.0), *[trade(i, -1.0) for i in range(2, 12)], trade(12, 10.0)]
    settings = SETTINGS.model_copy(
        update={"gates": FitnessGates(min_retention_top1=0.10, min_trades=0)}
    )
    result = compute_fitness(metrics(n_trades=12.0), ledger, perfect_inner(), None, settings)
    assert result.gate_failure == "F_CONCENTRATION"


def test_an_unmeasured_expectancy_is_a_rejection_not_a_pass() -> None:
    """``None`` means it was never established; treating that as passing would
    admit every candidate whose metrics failed to compute."""
    result = compute_fitness(
        metrics(expectancy_pct=None), GOOD_TRADES, perfect_inner(), None, SETTINGS
    )
    assert result.gate_failure == "F_EXPECTANCY"


def test_the_first_gate_in_the_specification_s_order_is_the_one_reported() -> None:
    """A rejected candidate is discarded, so one reason is enough — and it must
    be the same reason every time, or two runs disagree about why."""
    failing = metrics(expectancy_pct=-1.0, n_trades=0.0, max_drawdown=0.99)
    assert compute_fitness(failing, [], perfect_inner(), None, SETTINGS).gate_failure == GATE_IDS[0]


def test_the_removal_report_survives_a_rejection() -> None:
    """The evidence is kept even when the verdict is no: section 6 stores it."""
    result = compute_fitness(
        metrics(expectancy_pct=-0.001), GOOD_TRADES, perfect_inner(), None, SETTINGS
    )
    assert result.removal is not None
    assert result.removal.n_trades == len(GOOD_TRADES)


# ---------------------------------------------------------------------------
# stage 2 — the weighted score
# ---------------------------------------------------------------------------
def test_a_candidate_at_every_target_scores_one() -> None:
    """Hand-computed: each component clips to 1.0, the weights sum to 1, and no
    penalty applies — so fitness is exactly 1.0, to 1e-12.

    ``max_drawdown`` is zero rather than the fixture's 0.10: the drawdown
    component is ``(ceiling - drawdown) / ceiling``, so only a curve that never
    gave anything back scores the full mark.
    """
    result = compute_fitness(
        metrics(max_drawdown=0.0), GOOD_TRADES, perfect_inner(), None, SETTINGS
    )
    assert result.base_score == pytest.approx(1.0, abs=1e-12)
    assert result.fitness == pytest.approx(1.0, abs=1e-12)
    assert set(result.components) == set(FitnessWeights().as_dict())
    assert all(value == pytest.approx(1.0) for value in result.components.values())
    assert result.penalty_product == pytest.approx(1.0, abs=1e-12)


def test_every_component_is_computed_from_the_specification_s_formula() -> None:
    """One toy candidate, every component worked out by hand to 1e-12."""
    targets = FitnessTargets()
    result = compute_fitness(
        metrics(
            expectancy_pct=0.001,  # half the target
            sortino=0.75,  # half the target
            max_drawdown=0.175,  # half the ceiling
            profit_factor=1.25,  # (1.25 - 1) / (1.5 - 1)
            consistency=0.5,
            win_rate=0.45,  # (0.45 - 0.35) / (0.55 - 0.35)
            cagr=0.10,  # half the target
            top5_profit_share=0.25,
            n_trades=50.0,
        ),
        GOOD_TRADES,
        InnerFoldReport(folds=(InnerFold(is_sortino=2.0, oos_sortino=1.0, oos_return=0.1),) * 4),
        None,
        SETTINGS,
    )
    expected = {
        "expectancy": 0.5,
        "risk_adjusted": 0.5,
        "drawdown": 0.5,
        "profit_factor": 0.5,
        "consistency": 0.5,
        "inner_oos": 0.5,
        "trades": math.log1p(50) / math.log1p(targets.trades),
        "win_rate": 0.5,
        "net_return": 0.5,
        "concentration": 0.75,
    }
    for name, value in expected.items():
        assert result.components[name] == pytest.approx(value, abs=1e-12), name

    weights = FitnessWeights().as_dict()
    assert result.base_score == pytest.approx(
        sum(weights[name] * expected[name] for name in expected), abs=1e-12
    )


def test_an_unmeasured_component_scores_zero() -> None:
    """ "Not demonstrated" and "no problem" must not be the same number."""
    result = compute_fitness(
        metrics(sortino=None, cagr=None, consistency=None, win_rate=None),
        GOOD_TRADES,
        perfect_inner(),
        None,
        SETTINGS,
    )
    for name in ("risk_adjusted", "net_return", "consistency", "win_rate"):
        assert result.components[name] == 0.0


def test_no_losing_trades_scores_the_profit_factor_component_at_one() -> None:
    """Section 13.3: ``None`` profit factor is 1.0 *and* forces the low-trade
    warning — the ratio is undefined, not infinite."""
    result = compute_fitness(
        metrics(profit_factor=None), GOOD_TRADES, perfect_inner(), None, SETTINGS
    )
    assert result.components["profit_factor"] == 1.0
    assert MetricSet(profit_factor=None).low_trade_warning is False  # set by compute_metrics


def test_generalisation_that_was_never_measured_scores_zero() -> None:
    """An empty inner report is the state before the inner walk-forward runs."""
    result = compute_fitness(metrics(), GOOD_TRADES, InnerFoldReport(), None, SETTINGS)
    assert result.components["inner_oos"] == 0.0


def test_the_score_never_leaves_the_unit_interval() -> None:
    """Every component is clipped, so an absurd metric cannot inflate the score
    past 1 and make two configurations incomparable."""
    result = compute_fitness(
        metrics(sortino=1e9, cagr=1e9, expectancy_pct=1e9, profit_factor=1e9, max_drawdown=0.0),
        GOOD_TRADES,
        perfect_inner(),
        None,
        SETTINGS,
    )
    assert 0.0 <= result.fitness <= 1.0
    assert result.base_score == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# stage 3 — penalties compound
# ---------------------------------------------------------------------------
def test_a_penalty_that_does_not_apply_is_reported_as_one() -> None:
    """Reported, not omitted: a missing penalty should be visibly absent."""
    result = compute_fitness(metrics(), GOOD_TRADES, perfect_inner(), None, SETTINGS)
    assert set(result.penalties) == {
        "p_drawdown",
        "p_trades",
        "p_instability",
        "p_sensitivity",
        "p_removal",
        "p_divergence",
        "p_complexity",
    }
    assert all(value == 1.0 for value in result.penalties.values())


def test_the_drawdown_penalty_ramps_between_the_soft_and_hard_thresholds() -> None:
    """Halfway from ``drawdown_soft`` (0.20) to ``gates.max_drawdown`` (0.50)."""
    result = compute_fitness(
        metrics(max_drawdown=0.35), GOOD_TRADES, perfect_inner(), None, SETTINGS
    )
    assert result.penalties["p_drawdown"] == pytest.approx(0.5, abs=1e-12)


def test_the_trades_penalty_is_the_ratio_to_the_soft_floor() -> None:
    result = compute_fitness(metrics(n_trades=50.0), GOOD_TRADES, perfect_inner(), None, SETTINGS)
    assert result.penalties["p_trades"] == pytest.approx(0.5, abs=1e-12)


def test_the_divergence_penalty_fires_when_inner_oos_falls_away() -> None:
    """Sortino a quarter of in-sample, against a floor of half."""
    result = compute_fitness(
        metrics(),
        GOOD_TRADES,
        InnerFoldReport(folds=(InnerFold(is_sortino=4.0, oos_sortino=1.0, oos_return=0.1),) * 4),
        None,
        SETTINGS,
    )
    assert result.penalties["p_divergence"] == pytest.approx(0.5, abs=1e-12)


def test_an_unmeasured_inner_walkforward_is_not_charged_twice() -> None:
    """It already scores zero in ``inner_oos``; charging ``p_divergence`` as well
    would make an unmeasured candidate rank below a demonstrably divergent one."""
    unmeasured = compute_fitness(metrics(), GOOD_TRADES, InnerFoldReport(), None, SETTINGS)
    assert unmeasured.penalties["p_divergence"] == 1.0
    assert unmeasured.penalties["p_instability"] == 1.0


def test_the_instability_penalty_reads_the_spread_of_fold_returns() -> None:
    """Four folds whose returns vary far more than ``instability_max_cv``."""
    folds = tuple(
        InnerFold(is_sortino=1.0, oos_sortino=1.0, oos_return=value)
        for value in (0.01, 0.4, -0.2, 0.5)
    )
    result = compute_fitness(metrics(), GOOD_TRADES, InnerFoldReport(folds=folds), None, SETTINGS)
    assert result.penalties["p_instability"] < 1.0


def test_the_sensitivity_penalty_reads_the_neighbourhood_report() -> None:
    """A median neighbour drop at the threshold does not fire; beyond it does."""
    at_threshold = compute_fitness(
        metrics(), GOOD_TRADES, perfect_inner(), SensitivityReport(median_drop=0.5), SETTINGS
    )
    assert at_threshold.penalties["p_sensitivity"] == 1.0

    beyond = compute_fitness(
        metrics(), GOOD_TRADES, perfect_inner(), SensitivityReport(median_drop=0.75), SETTINGS
    )
    assert beyond.penalties["p_sensitivity"] == pytest.approx(0.5, abs=1e-12)


def test_the_removal_penalty_takes_the_worst_depth_not_the_average() -> None:
    """A strategy that survives losing one trade but collapses on five is
    concentrated; averaging that away is the mistake the test exists to prevent."""
    # Net 100: five winners of 10, fifteen of 4, ten losers of -1. Losing the
    # single best trade costs 10 % of the profit; losing the best five costs half.
    ledger = [
        *[trade(i, 10.0) for i in range(1, 6)],
        *[trade(i, 4.0) for i in range(6, 21)],
        *[trade(i, -1.0) for i in range(21, 31)],
    ]
    result = compute_fitness(metrics(n_trades=30.0), ledger, perfect_inner(), None, SETTINGS)
    removal = result.removal
    assert removal is not None
    assert removal.retention[1] == pytest.approx(0.90)
    assert removal.retention[5] == pytest.approx(0.50)

    # Depths 1 and 3 are comfortably above their targets and score 1.0; depth 5
    # is not, and the minimum is what survives.
    assert result.penalties["p_removal"] == pytest.approx((0.50 - 0.10) / (0.55 - 0.10), abs=1e-12)
    assert result.penalties["p_removal"] < 1.0


def test_the_complexity_penalty_compounds_per_excess_parameter() -> None:
    base = compute_fitness(metrics(), GOOD_TRADES, perfect_inner(), None, SETTINGS)
    two_over = compute_fitness(
        metrics(), GOOD_TRADES, perfect_inner(), None, SETTINGS, n_free_params=8
    )
    long_body = compute_fitness(
        metrics(), GOOD_TRADES, perfect_inner(), None, SETTINGS, logic_lines=500
    )
    assert base.penalties["p_complexity"] == 1.0
    assert two_over.penalties["p_complexity"] == pytest.approx(0.95**2, abs=1e-12)
    assert long_body.penalties["p_complexity"] == pytest.approx(0.95, abs=1e-12)


def test_penalties_multiply_so_two_problems_compound() -> None:
    """Section 13.3's stated reason for multiplying rather than subtracting."""
    one = compute_fitness(metrics(max_drawdown=0.35), GOOD_TRADES, perfect_inner(), None, SETTINGS)
    two = compute_fitness(
        metrics(max_drawdown=0.35, n_trades=50.0), GOOD_TRADES, perfect_inner(), None, SETTINGS
    )
    assert two.penalties["p_drawdown"] == pytest.approx(0.5, abs=1e-12)
    assert two.penalties["p_trades"] == pytest.approx(0.5, abs=1e-12)
    # 0.25 of the base score, not 0.5 twice over.
    assert two.penalty_product == pytest.approx(0.25, abs=1e-12)
    assert two.fitness < one.fitness


def test_fitness_is_the_base_score_times_the_penalties() -> None:
    result = compute_fitness(
        metrics(max_drawdown=0.35, n_trades=50.0), GOOD_TRADES, perfect_inner(), None, SETTINGS
    )
    assert result.fitness == pytest.approx(result.base_score * result.penalty_product, abs=1e-12)
    assert 0.0 <= result.fitness <= 1.0
    assert not result.rejected


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
def test_weights_that_do_not_sum_to_one_are_refused_at_config_load() -> None:
    """Scores must stay comparable across configurations, so the check is at
    construction rather than at scoring time."""
    with pytest.raises(ConfigError, match=r"must sum to 1\.0"):
        FitnessWeights(expectancy=0.9)


def test_a_rejected_candidate_sorts_below_every_real_one() -> None:
    """``FITNESS_REJECTED`` is far below the attainable range so a plain
    descending sort needs no special case."""
    rejected = compute_fitness(
        metrics(expectancy_pct=-1.0), GOOD_TRADES, perfect_inner(), None, SETTINGS
    )
    scored = compute_fitness(metrics(), GOOD_TRADES, perfect_inner(), None, SETTINGS)
    assert sorted([scored, rejected], key=lambda r: -r.fitness)[0] is scored
    assert FITNESS_REJECTED < 0.0


def test_a_tightened_configuration_is_honoured() -> None:
    """Fitness reads its thresholds from the settings it is handed, never from
    module constants that a campaign could not override."""
    strict = FitnessSettings(
        gates=FitnessGates(min_trades=500),
        penalties=FitnessPenalties(trades_soft=500),
    )
    assert compute_fitness(metrics(), GOOD_TRADES, perfect_inner(), None, strict).gate_failure == (
        "F_TRADES"
    )
