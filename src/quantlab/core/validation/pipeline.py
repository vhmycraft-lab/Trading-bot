"""The validation pipeline (master spec section 14.1). 🔒

Ten steps, in the order section 14.1 gives them, and the order is the design.
Everything expensive happens before anything is judged, so a verdict is reached
on the whole picture rather than on whichever step ran first; and the family's
budget is charged in step 8, after the work, so a crashed pipeline does not spend
a validation touch on a strategy it never actually measured.

The pipeline is an **orchestrator**: it holds no statistics of its own. Gates
come from :mod:`~quantlab.core.validation.gates`, checks from
:mod:`~quantlab.core.validation.score`, and every run comes from an injected
evaluator. That keeps the one module that touches the validation segment free of
the arithmetic that judges it, and lets the whole sequence be tested without an
engine.

**Direction of information.** This is the end of the line. Nothing here is read
by the evolutionary search: INV-9 confines that to the train segment, and
``tests/unit/test_architecture.py`` forbids the import as well. A verdict informs
a person, and nothing else.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from quantlab.core.config import ValidationSettings
from quantlab.core.errors import StrategyError
from quantlab.core.hashing import canonical_json
from quantlab.core.logging import get_logger
from quantlab.core.metrics import MetricSet
from quantlab.core.types import Trade
from quantlab.core.validation.concentration import TradeRemovalReport, trade_removal_report
from quantlab.core.validation.gates import GateInputs, GateReport, evaluate_gates
from quantlab.core.validation.score import ScoreInputs, ScoreReport, score_strategy
from quantlab.core.validation.sensitivity import SensitivityReport

__all__ = [
    "OPEN_FAMILY_STATUS",
    "Evidence",
    "ValidationOutcome",
    "judge",
    "require_open_family",
    "validate",
]

log = get_logger(__name__)

#: The only family status section 14.1 step 0 will start from.
OPEN_FAMILY_STATUS: Final[str] = "open"

#: Set when a family has spent its validation budget (section 14.1, step 8).
FROZEN_FAMILY_STATUS: Final[str] = "frozen"


def require_open_family(status: str) -> str:
    """Section 14.1 step 0: refuse a family that is not open.

    A frozen family has spent its twenty validation touches; a closed one failed
    the lockbox. Both refusals exist for the same reason — every look at held-out
    data costs some of its power to say anything, and a budget that could be
    ignored is not a budget.

    Raises:
        StrategyError: the family is frozen or closed.
    """
    if status != OPEN_FAMILY_STATUS:
        raise StrategyError(
            "this family is not open for validation; every look at held-out data "
            "costs some of its power to say anything, and the budget is spent",
            status=status,
            required=OPEN_FAMILY_STATUS,
        )
    return status


@dataclass(frozen=True, slots=True)
class Evidence:
    """Everything steps 1-5b gathered, before anything judges it.

    Assembled by the caller, which owns the runs. Keeping it separate from the
    judging is what lets the verdict logic be tested against a written-down
    picture rather than against a live engine.
    """

    probe_passed: bool | None = None
    metrics_train: MetricSet = field(default_factory=MetricSet)
    metrics_val: MetricSet = field(default_factory=MetricSet)
    metrics_val_stressed: MetricSet | None = None
    metrics_buyhold_val: MetricSet | None = None
    trades_train: Sequence[Trade] = ()
    trades_val: Sequence[Trade] = ()
    position_frac_val: Sequence[float] | np.ndarray = ()
    max_position_fraction: float = 1.0

    # -- statistical evidence (sections 14.4 and 15)
    deflated_sharpe: float | None = None
    pbo: float | None = None
    permutation_p: float | None = None
    sensitivity: SensitivityReport | None = None
    wfe: float | None = None
    profitable_oos_share: float | None = None
    param_cv: Mapping[str, float] = field(default_factory=dict)
    random_entry_sortinos: Sequence[float] = ()

    # -- provenance
    n_free_params: int | None = None
    logic_lines: int | None = None
    evolution_evaluations: int | None = None
    inner_oos_component: int | float | None = None
    #: ``M`` for section 14.4: every evaluation that led here.
    n_trials_accounted: int = 0

    def removal(self) -> TradeRemovalReport:
        """Step 5b, on the validation ledger."""
        return trade_removal_report(self.trades_val)


@dataclass(frozen=True, slots=True)
class ValidationOutcome:
    """What one validation produced (spec section 14.1, steps 6-9)."""

    strategy_id: str
    family_id: str
    report: ScoreReport
    #: The family's touch count *after* this validation (step 8).
    validation_touches: int = 0
    family_frozen: bool = False
    verdict_id: str = ""

    @property
    def verdict(self) -> str:
        return self.report.verdict

    @property
    def gates(self) -> GateReport:
        return self.report.gates

    @property
    def overfit_score(self) -> int:
        return self.report.overfit_score


def judge(evidence: Evidence, settings: ValidationSettings) -> ScoreReport:
    """Steps 6 and 7: the gates, then the score (spec sections 14.3, 14.4).

    Pure, and separate from :func:`validate` on purpose. Everything expensive has
    already happened by the time this runs; keeping the judgement free of the
    store means the whole verdict logic can be exercised against a written-down
    picture, which is how ``test_validate_pipeline.py`` states its cases.
    """
    removal = evidence.removal()
    gates = evaluate_gates(
        GateInputs(
            probe_passed=evidence.probe_passed,
            metrics_train=evidence.metrics_train,
            metrics_val=evidence.metrics_val,
            metrics_val_stressed=evidence.metrics_val_stressed,
            trades_train=evidence.trades_train,
            trades_val=evidence.trades_val,
            position_frac_val=evidence.position_frac_val,
            permutation_p=evidence.permutation_p,
            metrics_buyhold_val=evidence.metrics_buyhold_val,
            max_position_fraction=evidence.max_position_fraction,
        ),
        settings,
    )
    return score_strategy(
        ScoreInputs(
            deflated_sharpe=evidence.deflated_sharpe,
            pbo=evidence.pbo,
            permutation_p=evidence.permutation_p,
            sensitivity=evidence.sensitivity,
            metrics_val=evidence.metrics_val,
            wfe=evidence.wfe,
            profitable_oos_share=evidence.profitable_oos_share,
            param_cv=evidence.param_cv,
            n_free_params=evidence.n_free_params,
            logic_lines=evidence.logic_lines,
            trades_val=evidence.trades_val,
            random_entry_sortinos=evidence.random_entry_sortinos,
            removal=removal,
            evolution_evaluations=evidence.evolution_evaluations,
            inner_oos_component=(
                None
                if evidence.inner_oos_component is None
                else float(evidence.inner_oos_component)
            ),
        ),
        gates,
        settings,
    )


def validate(
    store: Any,
    *,
    strategy_id: str,
    family_id: str,
    split_id: str,
    params_json: str,
    evidence: Evidence,
    settings: ValidationSettings,
    verdict_id: Callable[[], str],
) -> ValidationOutcome:
    """Judge, charge the family, and record (spec section 14.1, steps 6-9).

    Steps 1-5b are the caller's: they own the runs, and the runs are the
    expensive part. This is everything that happens once the evidence is in.

    The order inside is deliberate:

    1. **Refuse a family that is not open** (step 0, re-checked here because this
       is the last point before the budget is charged).
    2. **Judge** — gates, then score.
    3. **Charge the touch** (step 8). After the judging, so a pipeline that
       crashed mid-judgement does not spend a validation look it never used.
    4. **Freeze the family** at ``family_max_validation_touches``, so the
       twenty-first validation cannot happen rather than merely being noticed.
    5. **Record the verdict** (step 9), with the thresholds it was reached under.

    Raises:
        StrategyError: the family is frozen or closed.
    """
    family = store.find_family(family_id)
    if family is None:
        raise StrategyError("no such strategy family", family_id=family_id)
    require_open_family(str(family.status))

    report = judge(evidence, settings)

    touches = int(store.increment_validation_touches(family_id))
    frozen = touches >= settings.family_max_validation_touches
    if frozen:
        store.set_family_status(family_id, FROZEN_FAMILY_STATUS)
        log.warning(
            "family_frozen",
            family_id=family_id,
            validation_touches=touches,
            limit=settings.family_max_validation_touches,
        )

    identifier = verdict_id()
    store.record_verdict(
        verdict_id=identifier,
        strategy_id=strategy_id,
        split_id=split_id,
        params_json=params_json,
        verdict=report.verdict,
        overfit_score=float(report.overfit_score),
        hard_gates_json=_canonical(report.gates.as_dict()),
        soft_checks_json=_canonical(report.soft_checks_json()),
        thresholds_json=_canonical(dict(report.thresholds)),
        n_trials_accounted=evidence.n_trials_accounted,
    )
    log.info(
        "validation_finished",
        strategy_id=strategy_id,
        family_id=family_id,
        verdict=report.verdict,
        overfit_score=report.overfit_score,
        failed_gates=list(report.gates.failed),
        unmeasured_checks=list(report.unmeasured),
        validation_touches=touches,
    )
    return ValidationOutcome(
        strategy_id=strategy_id,
        family_id=family_id,
        report=report,
        validation_touches=touches,
        family_frozen=frozen,
        verdict_id=identifier,
    )


def _canonical(document: Mapping[str, Any]) -> str:
    return canonical_json(dict(document))
