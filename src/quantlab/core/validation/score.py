"""Soft checks, the overfit score and the verdict (spec section 14.4).

The hard gates of section 14.3 ask whether a result is admissible at all. These
eleven checks ask a different question: *how much of it is likely to be
overfitting?* Each charges points for a specific symptom, the total is the
``overfit_score``, and the verdict falls out of the total together with the
gates:

* every gate passes and ``score <= candidate_max`` — ``CANDIDATE``
* every gate passes and ``score <= weak_max`` — ``WEAK``
* anything else — ``REJECT``

Points are **charges against** a strategy, so a higher score is worse. That
direction is worth stating plainly because every other number in the platform
runs the other way, and a reader who assumes otherwise reads every verdict
backwards.

Two rules govern missing evidence, and they differ from the gates' on purpose:

* A check whose input was never measured charges **nothing**. A gate asks for
  evidence of safety and refuses without it; a soft check charges for evidence of
  *fragility* and has none. Charging for a step the pipeline skipped would
  penalise a strategy for the pipeline's own gaps.
* The absence is **recorded** rather than silently treated as a pass, so a score
  assembled from half the evidence cannot be mistaken for a clean one.

Nothing here is ever read by the search. Section 14 judges; section 13 searches;
``tests/unit/test_architecture.py`` holds the two apart as an import rule.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

import numpy as np

from quantlab.core.config import ValidationSettings
from quantlab.core.metrics import MetricSet
from quantlab.core.types import Trade
from quantlab.core.validation.concentration import TradeRemovalReport
from quantlab.core.validation.gates import GateReport, gate_thresholds
from quantlab.core.validation.sensitivity import SensitivityReport, sensitivity_points

__all__ = [
    "CHECK_IDS",
    "MAX_SCORE",
    "VERDICTS",
    "CheckResult",
    "ScoreInputs",
    "ScoreReport",
    "Verdict",
    "score_strategy",
    "score_thresholds",
]

#: The eleven checks, in the order section 14.4 tabulates them.
CHECK_IDS: Final[tuple[str, ...]] = (
    "C01_DEFLATED_SHARPE",
    "C02_PBO",
    "C03_PERMUTATION",
    "C04_SENSITIVITY",
    "C05_CONCENTRATION",
    "C06_WALKFORWARD",
    "C07_COMPLEXITY",
    "C08_REGIME",
    "C09_RANDOM_BASELINE",
    "C10_TRADE_REMOVAL",
    "C11_EVOLUTION_PROVENANCE",
)

#: ``overfit_score = min(100, sum(points))`` (spec section 14.4).
MAX_SCORE: Final[int] = 100

Verdict = Literal["REJECT", "WEAK", "CANDIDATE"]

#: Every verdict this module can reach. ``LOCKBOX_PASS`` and ``LOCKBOX_FAIL``
#: exist in the schema of section 6 but are the lockbox's to write (section 14.6),
#: never this pipeline's.
VERDICTS: Final[tuple[str, ...]] = ("REJECT", "WEAK", "CANDIDATE")

#: Evolution runs wider than this are suspicious when they did not generalise
#: in-sample (section 14.4, check 11).
_WIDE_SEARCH: Final[int] = 500
_INNER_OOS_FLOOR: Final[float] = 0.5
_EPS: Final[float] = 1e-12


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One soft check: what it charged, and why."""

    check_id: str
    points: int
    #: ``False`` when the check's input was never measured. It then charges
    #: nothing, and this is how a reader tells that from a clean result.
    measured: bool = True
    observed: Any = None
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "points": self.points,
            "measured": self.measured,
            "observed": self.observed,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ScoreInputs:
    """Everything the eleven checks read (spec section 14.1, steps 1-5b, and 15).

    Assembled by the pipeline. Every field defaults to "not measured", so a
    partial pipeline produces a score that visibly rests on partial evidence
    rather than one that quietly looks clean.
    """

    # -- check 1
    deflated_sharpe: float | None = None
    # -- check 2
    pbo: float | None = None
    # -- check 3
    permutation_p: float | None = None
    # -- check 4
    sensitivity: SensitivityReport | None = None
    # -- check 5
    metrics_val: MetricSet = field(default_factory=MetricSet)
    # -- check 6
    wfe: float | None = None
    profitable_oos_share: float | None = None
    param_cv: Mapping[str, float] = field(default_factory=dict)
    # -- check 7
    n_free_params: int | None = None
    logic_lines: int | None = None
    # -- check 8
    trades_val: Sequence[Trade] = ()
    # -- check 9
    random_entry_sortinos: Sequence[float] = ()
    # -- check 10
    removal: TradeRemovalReport | None = None
    # -- check 11
    evolution_evaluations: int | None = None
    inner_oos_component: float | None = None


@dataclass(frozen=True, slots=True)
class ScoreReport:
    """The verdict and everything behind it (``validation_verdict``, section 6)."""

    verdict: Verdict
    overfit_score: int
    checks: tuple[CheckResult, ...] = ()
    gates: GateReport = field(default_factory=GateReport)
    thresholds: Mapping[str, Any] = field(default_factory=dict)

    @property
    def unmeasured(self) -> tuple[str, ...]:
        """Checks whose input was never supplied.

        A ``CANDIDATE`` with a long list here is not a clean strategy; it is a
        strategy nobody finished checking, and the report says so rather than
        letting the number speak for evidence that was not gathered.
        """
        return tuple(check.check_id for check in self.checks if not check.measured)

    @property
    def charged(self) -> tuple[str, ...]:
        return tuple(check.check_id for check in self.checks if check.points > 0)

    def __getitem__(self, check_id: str) -> CheckResult:
        for check in self.checks:
            if check.check_id == check_id:
                return check
        raise KeyError(check_id)

    def soft_checks_json(self) -> dict[str, dict[str, Any]]:
        """Section 6's ``soft_checks_json``."""
        return {check.check_id: check.as_dict() for check in self.checks}


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------
def _absent(check_id: str, what: str) -> CheckResult:
    return CheckResult(
        check_id=check_id, points=0, measured=False, reason=f"{what} was not measured"
    )


def _deflated_sharpe(inputs: ScoreInputs, settings: ValidationSettings) -> CheckResult:
    """Check 1. A result the size of the search cannot support."""
    value = inputs.deflated_sharpe
    if value is None:
        return _absent("C01_DEFLATED_SHARPE", "the deflated Sharpe ratio")
    charged = value < settings.dsr_threshold
    return CheckResult(
        check_id="C01_DEFLATED_SHARPE",
        points=25 if charged else 0,
        observed=value,
        reason=(
            f"deflated Sharpe {value:.3f} is below {settings.dsr_threshold}"
            if charged
            else "survives the size of the search"
        ),
    )


def _pbo(inputs: ScoreInputs) -> CheckResult:
    """Check 2, banded rather than cumulative.

    Both thresholds read the same quantity, and ``PBO > 0.5`` implies
    ``PBO > 0.3``; charging both would make the higher band worth 35 rather than
    the 25 section 14.4 assigns it.
    """
    value = inputs.pbo
    if value is None:
        return _absent("C02_PBO", "the probability of backtest overfitting")
    points = 25 if value > 0.5 else 10 if value > 0.3 else 0
    return CheckResult(
        check_id="C02_PBO",
        points=points,
        observed=value,
        reason=(
            f"the in-sample winner falls below median out of sample {value:.0%} of the time"
            if points
            else "the in-sample winner keeps winning"
        ),
    )


def _permutation(inputs: ScoreInputs, settings: ValidationSettings) -> CheckResult:
    """Check 3. Half the gate's threshold: ``G_PERM`` rejects above ``alpha``,
    and this charges for merely getting close to it."""
    value = inputs.permutation_p
    if value is None:
        return _absent("C03_PERMUTATION", "the market-permutation p-value")
    ceiling = settings.permutation.alpha / 2.0
    charged = value > ceiling
    return CheckResult(
        check_id="C03_PERMUTATION",
        points=10 if charged else 0,
        observed=value,
        reason=(
            f"p = {value:.4f} is above alpha/2 = {ceiling:.4f}"
            if charged
            else "comfortably distinguishable from its own permutations"
        ),
    )


def _sensitivity(inputs: ScoreInputs, settings: ValidationSettings) -> CheckResult:
    """Check 4. The neighbourhood is computed once, in ``optimize/plateau.py``."""
    report = inputs.sensitivity
    if report is None or not report.measured:
        return _absent("C04_SENSITIVITY", "the parameter neighbourhood")
    points = sensitivity_points(report, max_drop=settings.sensitivity_max_drop, points=20)
    return CheckResult(
        check_id="C04_SENSITIVITY",
        points=points,
        observed=report.median_drop,
        reason=(
            f"the median neighbour falls {float(report.median_drop or 0.0):.0%}"
            if points
            else "the objective holds up around the chosen parameters"
        ),
    )


def _concentration(inputs: ScoreInputs, settings: ValidationSettings) -> CheckResult:
    """Check 5. A few trades carrying the profit is a few things happening."""
    share = inputs.metrics_val.top5_profit_share
    if share is None:
        return _absent("C05_CONCENTRATION", "the top-trade profit share")
    charged = share > settings.concentration_max_share
    return CheckResult(
        check_id="C05_CONCENTRATION",
        points=10 if charged else 0,
        observed=share,
        reason=(
            f"the top {settings.concentration_top_n} trades carry {share:.0%} of gross profit"
            if charged
            else "profit is spread across many trades"
        ),
    )


def _walkforward(inputs: ScoreInputs) -> CheckResult:
    """Check 6, cumulative: three different quantities, each its own symptom."""
    if inputs.wfe is None and inputs.profitable_oos_share is None and not inputs.param_cv:
        return _absent("C06_WALKFORWARD", "the walk-forward")

    points = 0
    reasons: list[str] = []
    if inputs.wfe is not None and inputs.wfe < 0.5:
        points += 20
        reasons.append(f"walk-forward efficiency {inputs.wfe:.2f} is below 0.5")
    if inputs.profitable_oos_share is not None and inputs.profitable_oos_share < 0.5:
        points += 10
        reasons.append(f"only {inputs.profitable_oos_share:.0%} of windows were profitable")
    unstable = sorted(name for name, cv in inputs.param_cv.items() if cv > 0.5)
    if unstable:
        points += 5
        reasons.append(f"parameters re-chosen wildly across windows: {', '.join(unstable)}")

    return CheckResult(
        check_id="C06_WALKFORWARD",
        points=points,
        observed={
            "wfe": inputs.wfe,
            "profitable_oos_share": inputs.profitable_oos_share,
            "unstable_params": unstable,
        },
        reason="; ".join(reasons) if reasons else "the search still works when re-run through time",
    )


def _complexity(inputs: ScoreInputs, settings: ValidationSettings) -> CheckResult:
    """Check 7. Five points per free parameter above the limit, five for a long
    body — a soft charge, because the hard limits live in section 9.2 where a
    strategy is refused outright."""
    if inputs.n_free_params is None and inputs.logic_lines is None:
        return _absent("C07_COMPLEXITY", "the strategy's shape")

    excess = max(0, (inputs.n_free_params or 0) - settings.max_free_params)
    long_body = (inputs.logic_lines or 0) > settings.max_logic_lines
    points = 5 * excess + (5 if long_body else 0)
    reasons: list[str] = []
    if excess:
        reasons.append(f"{excess} parameter(s) above {settings.max_free_params}")
    if long_body:
        reasons.append(f"{inputs.logic_lines} logic lines above {settings.max_logic_lines}")

    return CheckResult(
        check_id="C07_COMPLEXITY",
        points=points,
        observed={"n_free_params": inputs.n_free_params, "logic_lines": inputs.logic_lines},
        reason="; ".join(reasons) if reasons else "within the complexity budget",
    )


def _regime(inputs: ScoreInputs) -> CheckResult:
    """Check 8. Profit concentrated in one calendar quarter is a strategy that
    worked in one regime, not one that works.

    Quarters are taken in UTC from each trade's *exit*, because that is when the
    profit was realised and the account felt it.
    """
    trades = list(inputs.trades_val)
    if not trades:
        return _absent("C08_REGIME", "the validation trade ledger")

    total = sum(trade.pnl for trade in trades)
    if total <= _EPS:
        return CheckResult(
            check_id="C08_REGIME",
            points=0,
            observed=None,
            reason="no net profit to concentrate in a quarter",
        )

    by_quarter: dict[tuple[int, int], float] = {}
    for trade in trades:
        moment = dt.datetime.fromtimestamp(trade.exit_ts / 1000.0, tz=dt.UTC)
        key = (moment.year, (moment.month - 1) // 3 + 1)
        by_quarter[key] = by_quarter.get(key, 0.0) + trade.pnl

    best = max(by_quarter.values())
    share = best / total
    charged = share > 0.9 and len(by_quarter) > 1
    return CheckResult(
        check_id="C08_REGIME",
        points=10 if charged else 0,
        observed={"largest_quarter_share": share, "n_quarters": len(by_quarter)},
        reason=(
            f"{share:.0%} of validation profit fell in one calendar quarter"
            if charged
            else "profit is spread across the validation period"
        ),
    )


def _random_baseline(inputs: ScoreInputs) -> CheckResult:
    """Check 9. A strategy inside the random-entry distribution has shown
    nothing that entering at random would not have shown."""
    sortinos = [float(value) for value in inputs.random_entry_sortinos]
    observed = inputs.metrics_val.sortino
    if not sortinos or observed is None:
        return _absent("C09_RANDOM_BASELINE", "the random-entry distribution")

    percentile = float(np.percentile(np.asarray(sortinos, dtype="float64"), 95))
    charged = observed <= percentile
    return CheckResult(
        check_id="C09_RANDOM_BASELINE",
        points=20 if charged else 0,
        observed={
            "sortino_val": observed,
            "random_entry_p95": percentile,
            "n_seeds": len(sortinos),
        },
        reason=(
            f"Sortino {observed:.2f} is inside the random-entry distribution (p95 {percentile:.2f})"
            if charged
            else "beats the random-entry baseline"
        ),
    )


def _trade_removal(inputs: ScoreInputs) -> CheckResult:
    """Check 10, cumulative: three depths, three independent questions.

    Retention is non-increasing in ``k``, but ``retention_1 < 0.40`` does not
    imply ``retention_3 < 0.20``, so the three thresholds are separate symptoms
    rather than bands on one quantity.
    """
    report = inputs.removal
    if report is None or not report.is_defined:
        return _absent("C10_TRADE_REMOVAL", "the trade-removal test")

    charges = ((1, 0.40, 25), (3, 0.20, 15), (5, 0.10, 10))
    points = 0
    reasons: list[str] = []
    for depth, floor, value in charges:
        retention = report.retention_at(depth)
        if retention is not None and retention < floor:
            points += value
            reasons.append(f"retention_{depth} = {retention:.2f} below {floor:.2f}")

    return CheckResult(
        check_id="C10_TRADE_REMOVAL",
        points=points,
        observed={depth: report.retention_at(depth) for depth, _floor, _p in charges},
        reason="; ".join(reasons) if reasons else "profit survives losing its best trades",
    )


def _provenance(inputs: ScoreInputs) -> CheckResult:
    """Check 11. A wide search that never generalised in-sample.

    Both conditions, not either: a wide search that *did* generalise is exactly
    what the platform is for, and a narrow search that did not is charged by
    other checks.
    """
    evaluations, inner = inputs.evolution_evaluations, inputs.inner_oos_component
    if evaluations is None or inner is None:
        return _absent("C11_EVOLUTION_PROVENANCE", "the evolution provenance")
    charged = evaluations > _WIDE_SEARCH and inner < _INNER_OOS_FLOOR
    return CheckResult(
        check_id="C11_EVOLUTION_PROVENANCE",
        points=10 if charged else 0,
        observed={"n_evaluations": evaluations, "inner_oos": inner},
        reason=(
            f"{evaluations} evaluations produced an inner-OOS component of {inner:.2f}"
            if charged
            else "the search was narrow, or it generalised in-sample"
        ),
    )


# ---------------------------------------------------------------------------
# the score and the verdict
# ---------------------------------------------------------------------------
def score_strategy(
    inputs: ScoreInputs, gates: GateReport, settings: ValidationSettings
) -> ScoreReport:
    """Run the eleven checks and reach a verdict (spec section 14.4).

    Args:
        inputs: Everything the checks read, assembled by the pipeline.
        gates: Section 14.3's verdict. Any failure makes this a ``REJECT``
            regardless of the score — the gates are absolute.
        settings: ``validation``.

    Returns:
        A :class:`ScoreReport` carrying the verdict, the score, every check and
        the thresholds it was all reached under.
    """
    checks = (
        _deflated_sharpe(inputs, settings),
        _pbo(inputs),
        _permutation(inputs, settings),
        _sensitivity(inputs, settings),
        _concentration(inputs, settings),
        _walkforward(inputs),
        _complexity(inputs, settings),
        _regime(inputs),
        _random_baseline(inputs),
        _trade_removal(inputs),
        _provenance(inputs),
    )
    score = min(MAX_SCORE, sum(check.points for check in checks))

    if not gates.passed:
        verdict: Verdict = "REJECT"
    elif score <= settings.score.candidate_max:
        verdict = "CANDIDATE"
    elif score <= settings.score.weak_max:
        verdict = "WEAK"
    else:
        verdict = "REJECT"

    return ScoreReport(
        verdict=verdict,
        overfit_score=score,
        checks=checks,
        gates=gates,
        thresholds=score_thresholds(settings),
    )


def score_thresholds(settings: ValidationSettings) -> dict[str, Any]:
    """Every threshold a verdict was reached under (``thresholds_json``, §14.4).

    Snapshotted with the verdict, gates included, because thresholds change: a
    verdict read a year later must be interpretable against the rules that
    produced it rather than against today's configuration.
    """
    return {
        **gate_thresholds(settings),
        "dsr_threshold": settings.dsr_threshold,
        "pbo_bands": [0.3, 0.5],
        "permutation_alpha_half": settings.permutation.alpha / 2.0,
        "sensitivity_max_drop": settings.sensitivity_max_drop,
        "concentration_top_n": settings.concentration_top_n,
        "concentration_max_share": settings.concentration_max_share,
        "max_free_params": settings.max_free_params,
        "max_logic_lines": settings.max_logic_lines,
        "regime_max_quarter_share": 0.9,
        "random_entry_percentile": 95,
        "retention_floors": {"1": 0.40, "3": 0.20, "5": 0.10},
        "wide_search_evaluations": _WIDE_SEARCH,
        "inner_oos_floor": _INNER_OOS_FLOOR,
        "candidate_max": settings.score.candidate_max,
        "weak_max": settings.score.weak_max,
        "max_score": MAX_SCORE,
    }
