"""Soft checks, the overfit score and the verdict (spec section 14.4, task T31).

Section 20 names three things: the scoring table, the verdict thresholds, and the
thresholds snapshot. Every one of the eleven checks has a test that charges it
and a test that does not, so a check that became unreachable — always charging,
never charging, or reading the wrong field — cannot pass the suite.

Two rules run through the file and are worth stating before the tests do:

* Points are **charges against** a strategy, so a higher score is worse. Every
  other number in the platform runs the other way.
* A check whose input was never measured charges **nothing** but is recorded as
  unmeasured. That differs from the hard gates on purpose, and the last section
  tests the difference: a ``CANDIDATE`` assembled from half the evidence must be
  distinguishable from a clean one.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from quantlab.core.config import PermutationSettings, ScoreSettings, ValidationSettings
from quantlab.core.metrics import MetricSet
from quantlab.core.types import Trade
from quantlab.core.validation.concentration import trade_removal_report
from quantlab.core.validation.gates import GateReport, GateResult
from quantlab.core.validation.score import (
    CHECK_IDS,
    MAX_SCORE,
    VERDICTS,
    ScoreInputs,
    score_strategy,
    score_thresholds,
)
from quantlab.core.validation.sensitivity import SensitivityReport

SETTINGS = ValidationSettings()
HOUR = 3_600_000


def gates(passed: bool = True) -> GateReport:
    return GateReport(
        results=(GateResult(gate_id="G_LEAK", passed=passed, observed=passed, threshold=True),)
    )


def trade(no: int, pnl: float, *, exit_ts: int | None = None) -> Trade:
    stamp = exit_ts if exit_ts is not None else 1_600_000_000_000 + no * HOUR
    return Trade(
        trade_no=no,
        side="long",
        entry_ts=stamp - HOUR,
        entry_px=100.0,
        exit_ts=stamp,
        exit_px=100.0 + pnl,
        qty=1.0,
        fees=0.0,
        slippage_cost=0.0,
        pnl=pnl,
        pnl_pct=pnl / 100.0,
        bars_held=1,
        exit_reason="signal",
    )


def quarter(year: int, q: int, day: int = 1) -> int:
    """Milliseconds at the start of a calendar quarter, UTC."""
    month = 1 + (q - 1) * 3
    return int(dt.datetime(year, month, day, tzinfo=dt.UTC).timestamp() * 1000)


#: A ledger whose profit survives losing its best trades and spreads over time.
HEALTHY_TRADES = [
    trade(index, 5.0, exit_ts=quarter(2024, 1 + index % 4, 1 + index % 20))
    for index in range(1, 41)
] + [trade(index, -1.0, exit_ts=quarter(2024, 1 + index % 4, 2)) for index in range(41, 61)]


def clean(**overrides: Any) -> ScoreInputs:
    """Inputs that charge nothing, with fields replaced per test."""
    base: dict[str, Any] = {
        "deflated_sharpe": 0.99,
        "pbo": 0.10,
        "permutation_p": 0.001,
        "sensitivity": SensitivityReport(median_drop=0.05, n_neighbours=24),
        "metrics_val": MetricSet(sortino=1.4, top5_profit_share=0.2),
        "wfe": 0.9,
        "profitable_oos_share": 0.8,
        "param_cv": {"fast": 0.1},
        "n_free_params": 3,
        "logic_lines": 40,
        "trades_val": HEALTHY_TRADES,
        # A random-entry distribution the clean strategy clearly beats: p95 is
        # about 0.90 against its Sortino of 1.4.
        "random_entry_sortinos": [0.05 * i for i in range(20)],
        "removal": trade_removal_report(HEALTHY_TRADES),
        "evolution_evaluations": 120,
        "inner_oos_component": 0.8,
    }
    base.update(overrides)
    return ScoreInputs(**base)


def points(check_id: str, **overrides: Any) -> int:
    return score_strategy(clean(**overrides), gates(), SETTINGS)[check_id].points


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------
def test_a_clean_strategy_charges_nothing() -> None:
    """Guards every positive test below against charging for the wrong reason."""
    report = score_strategy(clean(), gates(), SETTINGS)
    assert report.overfit_score == 0
    assert report.charged == ()
    assert report.unmeasured == ()
    assert tuple(check.check_id for check in report.checks) == CHECK_IDS


def test_the_score_is_capped_at_one_hundred() -> None:
    """Section 14.4: ``overfit_score = min(100, sum(points))``. Uncapped, the
    verdict bands would still work but the number would stop meaning anything."""
    worst = score_strategy(
        ScoreInputs(
            deflated_sharpe=0.0,
            pbo=0.9,
            permutation_p=0.9,
            sensitivity=SensitivityReport(median_drop=0.9, n_neighbours=10),
            metrics_val=MetricSet(sortino=0.0, top5_profit_share=0.99),
            wfe=0.1,
            profitable_oos_share=0.1,
            param_cv={"fast": 2.0},
            n_free_params=30,
            logic_lines=500,
            trades_val=[
                trade(1, 100.0, exit_ts=quarter(2024, 1)),
                trade(2, 1.0, exit_ts=quarter(2024, 2)),
            ],
            random_entry_sortinos=[5.0] * 10,
            removal=trade_removal_report([trade(1, 100.0), trade(2, 1.0), trade(3, -0.5)]),
            evolution_evaluations=5_000,
            inner_oos_component=0.1,
        ),
        gates(),
        SETTINGS,
    )
    assert worst.overfit_score == MAX_SCORE
    assert len(worst.charged) >= 9


def test_points_are_charges_so_a_higher_score_is_worse() -> None:
    poor = score_strategy(clean(deflated_sharpe=0.0), gates(), SETTINGS)
    good = score_strategy(clean(), gates(), SETTINGS)
    assert poor.overfit_score > good.overfit_score
    assert poor.verdict != "CANDIDATE" or good.verdict == "CANDIDATE"


# ---------------------------------------------------------------------------
# each check, charged and not
# ---------------------------------------------------------------------------
def test_check_1_charges_a_result_the_search_cannot_support() -> None:
    assert points("C01_DEFLATED_SHARPE", deflated_sharpe=0.5) == 25
    assert points("C01_DEFLATED_SHARPE", deflated_sharpe=0.99) == 0


def test_check_2_is_banded_not_cumulative() -> None:
    """Both thresholds read the same quantity, and ``PBO > 0.5`` implies
    ``PBO > 0.3``; charging both would make the higher band worth 35 rather than
    the 25 section 14.4 assigns it."""
    assert points("C02_PBO", pbo=0.7) == 25
    assert points("C02_PBO", pbo=0.4) == 10
    assert points("C02_PBO", pbo=0.2) == 0


def test_check_3_charges_at_half_the_gate_s_threshold() -> None:
    """``G_PERM`` rejects above ``alpha``; this charges for getting close."""
    half = SETTINGS.permutation.alpha / 2
    assert points("C03_PERMUTATION", permutation_p=half + 0.001) == 10
    assert points("C03_PERMUTATION", permutation_p=half - 0.001) == 0


def test_check_4_charges_a_neighbourhood_that_collapses() -> None:
    steep = SensitivityReport(median_drop=SETTINGS.sensitivity_max_drop + 0.1, n_neighbours=24)
    assert points("C04_SENSITIVITY", sensitivity=steep) == 20
    assert points("C04_SENSITIVITY") == 0


def test_check_5_charges_profit_carried_by_a_few_trades() -> None:
    concentrated = MetricSet(sortino=1.4, top5_profit_share=0.9)
    assert points("C05_CONCENTRATION", metrics_val=concentrated) == 10
    assert points("C05_CONCENTRATION") == 0


def test_check_6_is_cumulative_across_three_symptoms() -> None:
    """Three different quantities, each its own symptom — unlike check 2."""
    assert points("C06_WALKFORWARD", wfe=0.2) == 20
    assert points("C06_WALKFORWARD", profitable_oos_share=0.3) == 10
    assert points("C06_WALKFORWARD", param_cv={"fast": 1.5}) == 5
    assert (
        points("C06_WALKFORWARD", wfe=0.2, profitable_oos_share=0.3, param_cv={"fast": 1.5}) == 35
    )


def test_check_7_charges_five_points_per_excess_parameter() -> None:
    assert points("C07_COMPLEXITY", n_free_params=SETTINGS.max_free_params + 2) == 10
    assert points("C07_COMPLEXITY", logic_lines=SETTINGS.max_logic_lines + 1) == 5
    assert (
        points("C07_COMPLEXITY", n_free_params=SETTINGS.max_free_params + 1, logic_lines=500) == 10
    )
    assert points("C07_COMPLEXITY") == 0


def test_check_8_charges_profit_confined_to_one_quarter() -> None:
    """A strategy that worked in one regime, not one that works."""
    one_quarter = [
        trade(1, 100.0, exit_ts=quarter(2024, 1)),
        trade(2, 1.0, exit_ts=quarter(2024, 3)),
    ]
    assert points("C08_REGIME", trades_val=one_quarter) == 10
    assert points("C08_REGIME") == 0


def test_check_8_does_not_charge_a_single_quarter_of_data() -> None:
    """Concentration in the only quarter there was is not a regime finding."""
    only = [trade(1, 50.0, exit_ts=quarter(2024, 2)), trade(2, 50.0, exit_ts=quarter(2024, 2, 5))]
    assert points("C08_REGIME", trades_val=only) == 0


def test_check_8_has_nothing_to_measure_without_profit() -> None:
    losing = [trade(1, -5.0, exit_ts=quarter(2024, 1)), trade(2, -3.0, exit_ts=quarter(2024, 2))]
    assert points("C08_REGIME", trades_val=losing) == 0


def test_check_9_charges_a_strategy_inside_the_random_distribution() -> None:
    """It has shown nothing that entering at random would not have shown."""
    beaten = MetricSet(sortino=0.2, top5_profit_share=0.2)
    assert points("C09_RANDOM_BASELINE", metrics_val=beaten) == 20
    assert points("C09_RANDOM_BASELINE") == 0


def test_check_9_charges_at_the_percentile_not_the_maximum() -> None:
    """The 95th percentile, so one lucky seed out of a hundred does not decide
    whether a real strategy is charged."""
    lucky_outlier = [0.1] * 99 + [9.0]
    assert points("C09_RANDOM_BASELINE", random_entry_sortinos=lucky_outlier) == 0


def test_check_10_is_cumulative_across_three_depths() -> None:
    """Retention is non-increasing in ``k``, but ``retention_1 < 0.40`` does not
    imply ``retention_3 < 0.20``; the three are separate symptoms."""
    fragile = [trade(1, 100.0), *[trade(i, 1.0) for i in range(2, 6)], trade(6, -20.0)]
    charged = points("C10_TRADE_REMOVAL", removal=trade_removal_report(fragile))
    assert charged >= 25
    assert points("C10_TRADE_REMOVAL") == 0


def test_check_11_needs_both_a_wide_search_and_a_failure_to_generalise() -> None:
    """A wide search that *did* generalise is what the platform is for."""
    assert (
        points("C11_EVOLUTION_PROVENANCE", evolution_evaluations=5_000, inner_oos_component=0.1)
        == 10
    )
    assert (
        points("C11_EVOLUTION_PROVENANCE", evolution_evaluations=5_000, inner_oos_component=0.9)
        == 0
    )
    assert (
        points("C11_EVOLUTION_PROVENANCE", evolution_evaluations=10, inner_oos_component=0.1) == 0
    )


# ---------------------------------------------------------------------------
# missing evidence
# ---------------------------------------------------------------------------
def test_an_unmeasured_check_charges_nothing_but_says_so() -> None:
    """A gate asks for evidence of safety and refuses without it; a soft check
    charges for evidence of fragility and has none. Charging for a step the
    pipeline skipped would penalise a strategy for the pipeline's own gaps."""
    report = score_strategy(ScoreInputs(), gates(), SETTINGS)
    assert report.overfit_score == 0
    assert set(report.unmeasured) == set(CHECK_IDS)


def test_a_candidate_built_on_no_evidence_is_visibly_so() -> None:
    """The number alone would read as a clean strategy; the report must not let
    it stand for evidence nobody gathered."""
    report = score_strategy(ScoreInputs(), gates(), SETTINGS)
    assert report.verdict == "CANDIDATE"
    assert len(report.unmeasured) == len(CHECK_IDS)


def test_a_partly_measured_score_records_which_parts() -> None:
    report = score_strategy(ScoreInputs(deflated_sharpe=0.5, pbo=0.7), gates(), SETTINGS)
    assert report.overfit_score == 50
    assert "C01_DEFLATED_SHARPE" not in report.unmeasured
    assert "C09_RANDOM_BASELINE" in report.unmeasured


# ---------------------------------------------------------------------------
# the verdict
# ---------------------------------------------------------------------------
def test_a_clean_strategy_that_passes_the_gates_is_a_candidate() -> None:
    assert score_strategy(clean(), gates(), SETTINGS).verdict == "CANDIDATE"


def test_a_middling_score_is_weak() -> None:
    settings = ValidationSettings(score=ScoreSettings(candidate_max=10, weak_max=60))
    report = score_strategy(clean(deflated_sharpe=0.5), gates(), settings)
    assert report.overfit_score == 25
    assert report.verdict == "WEAK"


def test_a_high_score_is_rejected_even_with_every_gate_passing() -> None:
    settings = ValidationSettings(score=ScoreSettings(candidate_max=10, weak_max=20))
    report = score_strategy(clean(deflated_sharpe=0.5, pbo=0.7), gates(), settings)
    assert report.overfit_score == 50
    assert report.verdict == "REJECT"


def test_a_failed_gate_is_a_rejection_whatever_the_score() -> None:
    """The gates are absolute: no soft check compensates for one."""
    report = score_strategy(clean(), gates(passed=False), SETTINGS)
    assert report.overfit_score == 0
    assert report.verdict == "REJECT"


def test_the_bands_are_inclusive_at_their_upper_edge() -> None:
    settings = ValidationSettings(score=ScoreSettings(candidate_max=25, weak_max=50))
    assert score_strategy(clean(deflated_sharpe=0.5), gates(), settings).verdict == "CANDIDATE"
    assert score_strategy(clean(deflated_sharpe=0.5, pbo=0.7), gates(), settings).verdict == "WEAK"


def test_the_pipeline_never_reaches_a_lockbox_verdict() -> None:
    """``LOCKBOX_PASS`` and ``LOCKBOX_FAIL`` exist in section 6's schema but are
    the lockbox's to write (section 14.6), never this pipeline's."""
    assert VERDICTS == ("REJECT", "WEAK", "CANDIDATE")
    for inputs in (ScoreInputs(), clean(), clean(deflated_sharpe=0.0)):
        for gate in (gates(), gates(passed=False)):
            assert score_strategy(inputs, gate, SETTINGS).verdict in VERDICTS


# ---------------------------------------------------------------------------
# the snapshot
# ---------------------------------------------------------------------------
def test_every_threshold_travels_with_the_verdict() -> None:
    """Section 14.4's ``thresholds_json``. A verdict read a year later must be
    interpretable against the rules that produced it."""
    snapshot = score_thresholds(SETTINGS)
    assert snapshot["dsr_threshold"] == SETTINGS.dsr_threshold
    assert snapshot["candidate_max"] == SETTINGS.score.candidate_max
    assert snapshot["retention_floors"] == {"1": 0.40, "3": 0.20, "5": 0.10}
    # The gates' thresholds are in there too: a verdict is both halves.
    assert snapshot["min_trades_val"] == SETTINGS.min_trades_val
    assert snapshot["permutation_alpha"] == SETTINGS.permutation.alpha


def test_the_snapshot_follows_the_configuration() -> None:
    tightened = ValidationSettings(dsr_threshold=0.99, permutation=PermutationSettings(alpha=0.01))
    snapshot = score_thresholds(tightened)
    assert snapshot["dsr_threshold"] == 0.99
    assert snapshot["permutation_alpha_half"] == pytest.approx(0.005)


def test_the_report_serialises_every_check() -> None:
    document = score_strategy(clean(), gates(), SETTINGS).soft_checks_json()
    assert set(document) == set(CHECK_IDS)
    for check_id, entry in document.items():
        assert set(entry) == {"points", "measured", "observed", "reason"}
        assert entry["reason"], check_id
