"""The hard gates (master spec section 14.3, task T27).

Section 20 asks for "each gate pass/fail", and that is the spine of this file:
every gate has a test that passes it and a test that fails it, so a gate that
became unreachable — always true, always false, or reading the wrong field —
cannot pass the suite.

The gates are absolute. No soft check, no score and no good performance elsewhere
compensates for one, and several tests below exist to hold that: a strategy at
the top of every other measure still fails on a single impossible trade.
"""

from __future__ import annotations

from typing import Any

import pytest

from quantlab.core.config import PermutationSettings, ValidationSettings
from quantlab.core.metrics import MetricSet
from quantlab.core.types import Trade
from quantlab.core.validation.gates import (
    GATE_IDS,
    GateInputs,
    evaluate_gates,
    gate_thresholds,
)

SETTINGS = ValidationSettings()


def trade(no: int, pnl_pct: float) -> Trade:
    return Trade(
        trade_no=no,
        side="long",
        entry_ts=1_600_000_000_000 + no * 3_600_000,
        entry_px=100.0,
        exit_ts=1_600_000_000_000 + (no + 1) * 3_600_000,
        exit_px=100.0 * (1 + pnl_pct),
        qty=1.0,
        fees=0.0,
        slippage_cost=0.0,
        pnl=100.0 * pnl_pct,
        pnl_pct=pnl_pct,
        bars_held=1,
        exit_reason="signal",
    )


def passing(**overrides: Any) -> GateInputs:
    """Inputs that clear all seven gates, with fields replaced per test."""
    base: dict[str, Any] = {
        "probe_passed": True,
        "metrics_train": MetricSet(n_trades=150.0, sharpe=1.2, sortino=1.5),
        "metrics_val": MetricSet(
            n_trades=60.0, sharpe=0.9, sortino=1.1, net_return=0.2, max_drawdown=0.10
        ),
        "metrics_val_stressed": MetricSet(net_return=0.05),
        "trades_train": [trade(i, 0.01) for i in range(1, 4)],
        "trades_val": [trade(i, 0.02) for i in range(1, 4)],
        "position_frac_val": [0.0, 0.5, 1.0, 0.8],
        "permutation_p": 0.01,
        "metrics_buyhold_val": MetricSet(sortino=0.4, max_drawdown=0.40, net_return=0.15),
        "max_position_fraction": 1.0,
    }
    base.update(overrides)
    return GateInputs(**base)


def verdict(gate_id: str, **overrides: Any) -> bool:
    return evaluate_gates(passing(**overrides), SETTINGS)[gate_id].passed


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------
def test_a_clean_strategy_passes_every_gate() -> None:
    """Guards every negative test below against passing for the wrong reason."""
    report = evaluate_gates(passing(), SETTINGS)
    assert report.passed
    assert report.failed == ()
    assert tuple(result.gate_id for result in report.results) == GATE_IDS


def test_every_gate_runs_even_after_one_has_failed() -> None:
    """A rejection naming only the first problem would send an author round the
    loop once per gate, and the rest cost nothing next to the runs behind them."""
    report = evaluate_gates(
        passing(probe_passed=False, metrics_val=MetricSet(n_trades=1.0)), SETTINGS
    )
    assert len(report.results) == len(GATE_IDS)
    assert {"G_LEAK", "G_MIN_TRADES"} <= set(report.failed)


def test_every_gate_reports_its_observation_and_threshold() -> None:
    """Section 14.3's stated output shape. A rejection that could not be read
    could only be obeyed."""
    document = evaluate_gates(passing(), SETTINGS).as_dict()
    assert set(document) == set(GATE_IDS)
    for gate_id, entry in document.items():
        assert set(entry) == {"passed", "observed", "threshold", "reason"}
        assert entry["reason"], gate_id


def test_one_failure_is_enough_to_fail_the_whole_report() -> None:
    """Section 14.4: any hard-gate failure is a REJECT, whatever else is true."""
    report = evaluate_gates(passing(probe_passed=False), SETTINGS)
    assert not report.passed
    assert report.failed == ("G_LEAK",)


def test_an_unknown_gate_is_an_error_not_a_silent_pass() -> None:
    with pytest.raises(KeyError):
        evaluate_gates(passing(), SETTINGS)["G_INVENTED"]


# ---------------------------------------------------------------------------
# G_LEAK
# ---------------------------------------------------------------------------
def test_a_leaky_strategy_is_rejected() -> None:
    assert not verdict("G_LEAK", probe_passed=False)


def test_a_probe_that_never_ran_is_not_a_pass() -> None:
    """ "Not checked" and "checked and clean" are different states, and only one
    of them is evidence."""
    assert not verdict("G_LEAK", probe_passed=None)


# ---------------------------------------------------------------------------
# G_MIN_TRADES
# ---------------------------------------------------------------------------
def test_too_few_validation_trades_is_rejected() -> None:
    assert not verdict("G_MIN_TRADES", metrics_val=MetricSet(n_trades=5.0, sharpe=0.9))


def test_too_few_training_trades_is_rejected() -> None:
    """Both segments, not either: a strategy observed four times in validation
    has not been validated."""
    assert not verdict("G_MIN_TRADES", metrics_train=MetricSet(n_trades=10.0, sharpe=1.2))


def test_exactly_the_minimum_passes() -> None:
    assert verdict(
        "G_MIN_TRADES",
        metrics_train=MetricSet(n_trades=float(SETTINGS.min_trades_train), sharpe=1.2),
        metrics_val=MetricSet(
            n_trades=float(SETTINGS.min_trades_val), sharpe=0.9, sortino=1.1, net_return=0.2
        ),
    )


# ---------------------------------------------------------------------------
# G_SANITY
# ---------------------------------------------------------------------------
def test_an_impossible_single_trade_is_rejected() -> None:
    """Usually a data error or a fill at a price that never traded."""
    assert not verdict("G_SANITY", trades_val=[trade(1, 0.40)])


def test_an_impossible_loss_is_rejected_too() -> None:
    """The check is on the magnitude: a 40 % loss on one trade is as suspect as
    a 40 % gain."""
    assert not verdict("G_SANITY", trades_val=[trade(1, -0.40)])


def test_a_training_trade_is_checked_as_well() -> None:
    assert not verdict("G_SANITY", trades_train=[trade(1, 0.40)])


def test_a_position_beyond_the_cap_and_its_drift_is_rejected() -> None:
    assert not verdict("G_SANITY", position_frac_val=[0.0, 2.5], max_position_fraction=1.0)


def test_the_drift_allowance_is_honoured() -> None:
    """The cap bounds the *target* fraction at the deciding bar; the realised
    fraction drifts with the market until the next rebalance, so a near-zero
    tolerance would fail correctly-behaved strategies."""
    at_ceiling = 1.0 * (1.0 + SETTINGS.sanity_drift_allowance)
    assert verdict("G_SANITY", position_frac_val=[0.0, at_ceiling], max_position_fraction=1.0)
    assert not verdict(
        "G_SANITY", position_frac_val=[0.0, at_ceiling + 0.01], max_position_fraction=1.0
    )


def test_a_short_position_is_measured_by_its_magnitude() -> None:
    assert not verdict("G_SANITY", position_frac_val=[0.0, -2.5], max_position_fraction=1.0)


def test_a_tighter_allowance_is_stricter() -> None:
    strict = ValidationSettings(sanity_drift_allowance=0.0)
    report = evaluate_gates(passing(position_frac_val=[0.0, 1.2]), strict)
    assert not report["G_SANITY"].passed


# ---------------------------------------------------------------------------
# G_COST
# ---------------------------------------------------------------------------
def test_an_edge_that_dies_under_doubled_costs_is_rejected() -> None:
    """An edge that disappears when costs double was never an edge; it was an
    estimate of the spread."""
    assert not verdict("G_COST", metrics_val_stressed=MetricSet(net_return=-0.01))


def test_breaking_even_under_stress_is_not_surviving_it() -> None:
    assert not verdict("G_COST", metrics_val_stressed=MetricSet(net_return=0.0))


def test_a_cost_stress_that_was_never_run_is_not_a_pass() -> None:
    assert not verdict("G_COST", metrics_val_stressed=None)


# ---------------------------------------------------------------------------
# G_PERM
# ---------------------------------------------------------------------------
def test_a_result_indistinguishable_from_noise_is_rejected() -> None:
    assert not verdict("G_PERM", permutation_p=0.30)


def test_exactly_alpha_passes() -> None:
    """Section 14.3 writes the rule as ``p <= alpha``."""
    assert verdict("G_PERM", permutation_p=SETTINGS.permutation.alpha)


def test_a_permutation_test_that_never_ran_is_not_a_pass() -> None:
    assert not verdict("G_PERM", permutation_p=None)


def test_a_tighter_alpha_is_stricter() -> None:
    strict = ValidationSettings(permutation=PermutationSettings(alpha=0.001))
    assert not evaluate_gates(passing(permutation_p=0.01), strict)["G_PERM"].passed


# ---------------------------------------------------------------------------
# G_DEGRADE
# ---------------------------------------------------------------------------
def test_a_negative_validation_sharpe_is_rejected() -> None:
    assert not verdict("G_DEGRADE", metrics_val=MetricSet(n_trades=60.0, sharpe=-0.2))


def test_a_collapse_against_train_is_rejected() -> None:
    """Some degradation is expected — that is what out-of-sample means — but a
    fall to a tenth of the training Sharpe is not degradation."""
    assert not verdict(
        "G_DEGRADE",
        metrics_train=MetricSet(n_trades=150.0, sharpe=2.0),
        metrics_val=MetricSet(n_trades=60.0, sharpe=0.1, sortino=1.1, net_return=0.2),
    )


def test_expected_degradation_still_passes() -> None:
    assert verdict(
        "G_DEGRADE",
        metrics_train=MetricSet(n_trades=150.0, sharpe=2.0),
        metrics_val=MetricSet(n_trades=60.0, sharpe=1.0, sortino=1.1, net_return=0.2),
    )


def test_a_non_positive_training_sharpe_leaves_only_the_sign_test() -> None:
    """Dividing by it would let a strategy that lost money on both segments pass
    by losing slightly less on one."""
    assert verdict(
        "G_DEGRADE",
        metrics_train=MetricSet(n_trades=150.0, sharpe=-0.5),
        metrics_val=MetricSet(n_trades=60.0, sharpe=0.3, sortino=1.1, net_return=0.2),
    )
    assert not verdict(
        "G_DEGRADE",
        metrics_train=MetricSet(n_trades=150.0, sharpe=-0.5),
        metrics_val=MetricSet(n_trades=60.0, sharpe=-0.1),
    )


def test_an_unmeasured_validation_sharpe_is_not_a_pass() -> None:
    assert not verdict("G_DEGRADE", metrics_val=MetricSet(n_trades=60.0, sharpe=None))


# ---------------------------------------------------------------------------
# G_BENCH
# ---------------------------------------------------------------------------
def test_beating_buy_and_hold_on_risk_adjusted_return_passes() -> None:
    assert verdict("G_BENCH", metrics_buyhold_val=MetricSet(sortino=0.4, max_drawdown=0.40))


def test_halving_the_drawdown_while_making_money_also_passes() -> None:
    """The disjunction is the point: a strategy that earns less than buy-and-hold
    but halves the drawdown is a different and legitimate product."""
    assert verdict(
        "G_BENCH",
        metrics_val=MetricSet(
            n_trades=60.0, sharpe=0.9, sortino=0.2, net_return=0.05, max_drawdown=0.10
        ),
        metrics_buyhold_val=MetricSet(sortino=0.9, max_drawdown=0.40, net_return=0.30),
    )


def test_losing_on_both_counts_is_rejected() -> None:
    assert not verdict(
        "G_BENCH",
        metrics_val=MetricSet(
            n_trades=60.0, sharpe=0.9, sortino=0.2, net_return=0.05, max_drawdown=0.35
        ),
        metrics_buyhold_val=MetricSet(sortino=0.9, max_drawdown=0.40, net_return=0.30),
    )


def test_a_smaller_drawdown_while_losing_money_is_not_enough() -> None:
    """Holding cash also halves the drawdown."""
    assert not verdict(
        "G_BENCH",
        metrics_val=MetricSet(
            n_trades=60.0, sharpe=0.9, sortino=0.2, net_return=-0.05, max_drawdown=0.05
        ),
        metrics_buyhold_val=MetricSet(sortino=0.9, max_drawdown=0.40, net_return=0.30),
    )


def test_a_missing_baseline_is_not_a_pass() -> None:
    """Section 14.1 step 5 runs the baselines; a gate that passed without one
    would be comparing against nothing."""
    assert not verdict("G_BENCH", metrics_buyhold_val=None)


# ---------------------------------------------------------------------------
# thresholds travel with the verdict
# ---------------------------------------------------------------------------
def test_the_thresholds_are_snapshotted_with_the_verdict() -> None:
    """Section 14.4's ``thresholds_json``. Thresholds change; a verdict read a
    year later must be interpretable against the rules that produced it."""
    snapshot = gate_thresholds(SETTINGS)
    assert snapshot["min_trades_val"] == SETTINGS.min_trades_val
    assert snapshot["permutation_alpha"] == SETTINGS.permutation.alpha
    assert snapshot["sanity_drift_allowance"] == SETTINGS.sanity_drift_allowance

    tightened = ValidationSettings(min_trades_val=99)
    assert gate_thresholds(tightened)["min_trades_val"] == 99
