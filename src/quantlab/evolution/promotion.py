"""Promotion: the only route from evolution to validation (spec section 13.7). 🔒

Evolution reads the train segment and nothing else. A candidate reaches the
validation segment through exactly one door, and this module is it.

Two rules make that mechanical rather than aspirational:

* **The promotion row is written first.** :func:`promote` records a
  ``candidate_promotion`` before the run it authorises executes. A row written
  afterwards would document a decision already taken, which is a log rather than
  a gate; :func:`require_promotion` is what a validation run consults, and it can
  only find a row that already exists.
* **Every promotion is charged.** ``strategy_family.validation_touches`` is
  incremented per promotion, which is what section 14.1's freeze at twenty counts
  and what section 14.4 charges into the deflated Sharpe ratio's ``M``. A
  promotion that did not increment it would let a family be measured against the
  validation set indefinitely while its verdicts kept claiming otherwise.

**Who is promoted.** Section 13.7: the top ``n_promote`` candidates with
``fitness >= min_fitness``, subject to pairwise similarity below
``max_similarity_between_promoted``. The similarity rule is not a nicety — three
promotions of the same strategy would spend three of a family's twenty touches
learning one thing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from quantlab.core.config import PromotionSettings
from quantlab.core.errors import StrategyError
from quantlab.core.fitness import FITNESS_REJECTED
from quantlab.core.logging import get_logger
from quantlab.evolution.diversity import similarity
from quantlab.evolution.population import ScoredCandidate, rank_candidates
from quantlab.ports.store import ExperimentStore

__all__ = [
    "PROMOTED_SEGMENTS",
    "Promotion",
    "eligible",
    "promote",
    "require_promotion",
]

log = get_logger(__name__)

#: Segments a promotion may authorise. Never ``test``: the lockbox is a separate
#: door with its own record (section 14.6), and no amount of evolutionary success
#: opens it.
PROMOTED_SEGMENTS: Final[tuple[str, ...]] = ("val",)


@dataclass(frozen=True, slots=True)
class Promotion:
    """One recorded promotion, and the candidate it authorises a run for."""

    promotion_id: str
    candidate_id: str
    segment: str
    reason: str
    touches: int


def eligible(
    scored: Sequence[ScoredCandidate], settings: PromotionSettings
) -> list[ScoredCandidate]:
    """Which candidates section 13.7 admits, in the order it admits them.

    Three filters, applied in this order because each is cheaper than the next:
    a candidate must have cleared the fitness gates at all, must reach
    ``min_fitness``, and must not be too similar to a candidate already chosen.

    The similarity filter runs against the *already chosen*, walking the ranking
    downwards — the same shape as the niching of section 13.5, and for the same
    reason: the fitter of two near-identical candidates is the one worth
    measuring.
    """
    chosen: list[ScoredCandidate] = []
    for candidate in rank_candidates(scored):
        if len(chosen) >= settings.n_promote:
            break
        if candidate.fitness <= FITNESS_REJECTED or candidate.fitness < settings.min_fitness:
            continue
        if any(
            similarity(candidate.view, accepted.view) > settings.max_similarity_between_promoted
            for accepted in chosen
        ):
            continue
        chosen.append(candidate)
    return chosen


def promote(
    store: ExperimentStore,
    candidates: Sequence[ScoredCandidate],
    *,
    evolution_id: str,
    gen_index: int,
    settings: PromotionSettings,
    segment: str = "val",
    reason: str = "top candidate after evolution",
) -> list[Promotion]:
    """Record a promotion for each eligible candidate (spec section 13.7).

    Written **before** any validation run executes. That ordering is the
    invariant, not a convention: :func:`require_promotion` is what a validation
    run consults, so a row written afterwards would authorise nothing.

    Each promotion increments the candidate's family's ``validation_touches``,
    when ``settings.counts_as_validation_touch`` says it should. That count is
    what section 14.1 freezes a family at twenty of, and what section 14.4 charges
    into ``M``.

    Raises:
        StrategyError: ``segment`` is not one a promotion may authorise. The test
            partition is a separate door with its own record (section 14.6), and
            no amount of evolutionary success opens it.
    """
    if segment not in PROMOTED_SEGMENTS:
        raise StrategyError(
            "a promotion authorises a validation run and nothing else; the test "
            "partition is reachable only through the lockbox (INV-5)",
            segment=segment,
            allowed=list(PROMOTED_SEGMENTS),
        )

    promotions: list[Promotion] = []
    for candidate in eligible(candidates, settings):
        row = store.record_promotion(
            candidate_id=candidate.candidate_id,
            evolution_id=evolution_id,
            gen_index=gen_index,
            segment=segment,
            reason=reason,
        )
        touches = 0
        if settings.counts_as_validation_touch:
            touches = store.increment_validation_touches(_family_of(store, candidate.candidate_id))
        log.info(
            "candidate_promoted",
            evolution_id=evolution_id,
            candidate_id=candidate.candidate_id,
            gen_index=gen_index,
            segment=segment,
            fitness=candidate.fitness,
            validation_touches=touches,
        )
        promotions.append(
            Promotion(
                promotion_id=row.promotion_id,
                candidate_id=candidate.candidate_id,
                segment=segment,
                reason=reason,
                touches=touches,
            )
        )
    return promotions


def require_promotion(store: ExperimentStore, candidate_id: str, segment: str) -> Any:
    """The promotion authorising a run, or a refusal (INV-9). 🔒

    Called before a validation run is created. A candidate with no promotion row
    for this segment has not been through the recorded, budgeted act section 13.7
    requires, and evaluating it anyway would spend a validation touch nobody
    counted.

    Raises:
        StrategyError: no promotion authorises this segment for this candidate.
    """
    for row in store.promotions_for(candidate_id):
        if row.segment == segment:
            return row
    raise StrategyError(
        "a validation run may only be created for a promoted candidate, and the "
        "promotion must be recorded before the run (INV-9)",
        candidate_id=candidate_id,
        segment=segment,
    )


def _family_of(store: ExperimentStore, candidate_id: str) -> str:
    """The family a candidate's strategy version belongs to.

    Touches are counted per *family*, not per version: section 14.1's budget is
    about how often one idea has been measured against the validation set, and a
    re-parameterised version of it is the same idea.
    """
    chain = store.ancestry(candidate_id)
    if not chain:
        raise StrategyError("no such candidate", candidate_id=candidate_id)
    version = store.get_strategy_version(chain[-1].strategy_id)
    if version is None:
        raise StrategyError(
            "the candidate's strategy version is not registered",
            candidate_id=candidate_id,
            strategy_id=chain[-1].strategy_id,
        )
    return str(version.family_id)
