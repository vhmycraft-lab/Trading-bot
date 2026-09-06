"""Lockbox eligibility and verdict (master spec section 14.6, INV-5). 🔒

The test partition is the last honest measurement the platform has, and it is
honest exactly once. Everything in this module exists to make that "once" a
property of the code rather than of anyone's discipline.

Four rules, and each closes a different way of spending the measurement twice:

* **A ``CANDIDATE`` verdict is required.** The lockbox is not a place to look for
  an edge; it is where an edge already argued for is confirmed. Anything else
  would make the test partition part of the search.
* **The budget is per family and per month.** One access per family, because a
  family is one *idea* and the second look would be at an idea already informed
  by the first. The monthly cap bounds the platform as a whole: twenty families
  looked at in a week is a search over the test partition however it is spelled.
* **``test_end_ts`` freezes on first use.** An open-ended test window that grew
  after being looked at would let a failure be waited out.
* **A failure closes the family for ever.** Section 14.1 step 0 then refuses to
  validate it, so a lockbox failure cannot be walked back by re-validating,
  re-optimising and returning.

Together these are what stops later optimisation from contaminating the evidence:
after an access the family can produce no new lockbox result at all, so nothing
learned from the test partition can feed a further search and come back.

The gates applied here are a **subset** of section 14.3's, and deliberately not
the whole set: the leakage probe, sanity and permutation checks are properties of
the strategy that validation has already established, and re-running them would
spend test data re-answering questions the validation segment answered.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

from quantlab.core.config import LockboxSettings, ValidationSettings
from quantlab.core.errors import LockboxViolation
from quantlab.core.validation.gates import GateInputs, GateReport, evaluate_gates

__all__ = [
    "CLOSED_FAMILY_STATUS",
    "LOCKBOX_GATES",
    "MIN_REASON_CHARS",
    "REQUIRED_VERDICT",
    "LockboxDecision",
    "LockboxVerdict",
    "decide",
    "lockbox_gates",
    "require_budget",
    "require_candidate",
    "require_reason",
]

#: The gates section 14.6 applies to a lockbox run.
#:
#: A subset of section 14.3's seven. ``G_LEAK``, ``G_SANITY`` and ``G_PERM`` are
#: properties of the strategy that validation established on the validation
#: segment; re-running them here would spend the test partition re-answering
#: questions already answered.
LOCKBOX_GATES: Final[tuple[str, ...]] = ("G_MIN_TRADES", "G_COST", "G_DEGRADE", "G_BENCH")

#: Section 14.6 requires a reason of at least this length.
MIN_REASON_CHARS: Final[int] = 20

#: What a family becomes on a lockbox failure — permanently.
CLOSED_FAMILY_STATUS: Final[str] = "closed"

#: The verdict a validation must have reached before the lockbox will open.
REQUIRED_VERDICT: Final[str] = "CANDIDATE"

LockboxVerdict = Literal["LOCKBOX_PASS", "LOCKBOX_FAIL"]


def require_candidate(verdicts: Sequence[Any]) -> Any:
    """The ``CANDIDATE`` verdict authorising this access (spec section 14.6).

    The **most recent** verdict must be ``CANDIDATE``. An earlier one that has
    since been superseded by a ``REJECT`` is not authorisation: the platform's
    current opinion is what matters, and letting a stale verdict open the lockbox
    would reward re-validating until one came out favourably.

    Raises:
        LockboxViolation: there is no verdict, or the latest is not ``CANDIDATE``.
    """
    if not verdicts:
        raise LockboxViolation(
            "the lockbox confirms an edge that has already been argued for; validate "
            "the strategy first",
            required=REQUIRED_VERDICT,
        )
    latest = verdicts[-1]
    if str(latest.verdict) != REQUIRED_VERDICT:
        raise LockboxViolation(
            "the most recent verdict is not CANDIDATE; a superseded verdict is not "
            "authorisation, or re-validating until one came out favourably would be",
            verdict=str(latest.verdict),
            required=REQUIRED_VERDICT,
        )
    return latest


def require_reason(reason: str) -> str:
    """A written reason of at least :data:`MIN_REASON_CHARS` (section 14.6).

    Not paperwork. The access is permanent and irreversible, and a person who
    cannot say in twenty characters why they are spending a family's one look at
    the test partition has not decided to spend it.

    Raises:
        LockboxViolation: the reason is too short.
    """
    text = reason.strip()
    if len(text) < MIN_REASON_CHARS:
        raise LockboxViolation(
            "a lockbox access needs a written reason; this look is permanent and "
            "cannot be taken back",
            length=len(text),
            required=MIN_REASON_CHARS,
        )
    return text


def require_budget(
    accesses: Sequence[Any],
    settings: LockboxSettings,
    *,
    family_id: str,
    now_ms: int,
) -> None:
    """Refuse an access the budget cannot afford (spec section 14.6).

    Two independent caps:

    * ``max_per_family`` — a family is one *idea*, and a second look would be at
      an idea already informed by the first.
    * ``max_per_month`` — bounds the platform as a whole. Twenty families looked
      at in a week is a search over the test partition however it is spelled.

    The month is a rolling thirty days rather than a calendar one, so the cap
    cannot be doubled by looking on the 31st and again on the 1st.

    Raises:
        LockboxViolation: either cap is already reached.
    """
    for_family = [row for row in accesses if str(row.family_id) == family_id]
    if len(for_family) >= settings.max_per_family:
        raise LockboxViolation(
            "this family has already used its lockbox access; a second look would be "
            "at an idea the first one already informed",
            family_id=family_id,
            used=len(for_family),
            allowed=settings.max_per_family,
        )

    horizon = now_ms - 30 * 24 * 60 * 60 * 1000
    recent = [row for row in accesses if int(row.created_at) >= horizon]
    if len(recent) >= settings.max_per_month:
        raise LockboxViolation(
            "the platform has used its lockbox budget for the last thirty days; "
            "looking at the test partition often enough is a search over it",
            used=len(recent),
            allowed=settings.max_per_month,
            since=dt.datetime.fromtimestamp(horizon / 1000.0, tz=dt.UTC).isoformat(),
        )


def lockbox_gates(inputs: GateInputs, settings: ValidationSettings) -> GateReport:
    """The four gates of section 14.6, applied to the test-segment run.

    ``G_MIN_TRADES`` uses the *validation* thresholds, as section 14.6 says: the
    test segment is the same length as validation, and holding it to the training
    threshold would fail strategies for the segment's size rather than their own.
    """
    full = evaluate_gates(inputs, settings)
    return GateReport(
        results=tuple(result for result in full.results if result.gate_id in LOCKBOX_GATES)
    )


@dataclass(frozen=True, slots=True)
class LockboxDecision:
    """The outcome of one lockbox access."""

    verdict: LockboxVerdict
    gates: GateReport
    family_closed: bool

    @property
    def passed(self) -> bool:
        return self.verdict == "LOCKBOX_PASS"


def decide(gates: GateReport) -> LockboxDecision:
    """``LOCKBOX_PASS`` or ``LOCKBOX_FAIL``, and what it does to the family.

    A failure closes the family permanently. That is the harshest rule in the
    platform and it is the right one: the family has now been measured on the
    only data nobody could optimise against, and it did not hold. Leaving it open
    would invite exactly the loop the lockbox exists to prevent — optimise more,
    return, look again.
    """
    passed = gates.passed
    return LockboxDecision(
        verdict="LOCKBOX_PASS" if passed else "LOCKBOX_FAIL",
        gates=gates,
        family_closed=not passed,
    )
