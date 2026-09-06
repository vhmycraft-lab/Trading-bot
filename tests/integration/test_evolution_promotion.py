"""INV-9: promotion is the only route to validation (spec section 13.7, task T53).

Section 22's acceptance criteria, one test each: a promotion row exists before its
run; a validation run created without one raises; promoting three candidates
increments ``validation_touches`` by three.

The invariant these defend is the platform's central claim. Evolution reads train
and nothing else; a candidate reaches the validation segment through one recorded,
budgeted act, and the record is written *first* so it is a gate rather than a log.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.unit.conftest_evolution import Harness, build_harness, small_settings

from quantlab.core.config import PromotionSettings
from quantlab.core.errors import StrategyError
from quantlab.core.fitness import FITNESS_REJECTED
from quantlab.evolution.diversity import CandidateView, Signature
from quantlab.evolution.loop import evolve
from quantlab.evolution.population import ScoredCandidate
from quantlab.evolution.promotion import PROMOTED_SEGMENTS, eligible, promote, require_promotion

pytestmark = pytest.mark.slow

EVOLUTION_ID = "ev_promote"


@pytest.fixture
def evolved(tmp_path: Path) -> tuple[Harness, list[ScoredCandidate]]:
    """A finished run, and its last generation's candidates."""
    harness = build_harness(tmp_path, small_settings(max_generations=3))
    harness.open_run(EVOLUTION_ID)
    result = evolve(**harness.kwargs(EVOLUTION_ID))
    return harness, list(result.generations[-1].scored)


def loose() -> PromotionSettings:
    """Promotion settings that admit this fixture's candidates.

    The shipped ``min_fitness`` of 0.35 is calibrated for a real campaign; the
    random genomes drawn here score below it, and a test that promoted nothing
    would assert nothing.
    """
    return PromotionSettings(n_promote=3, min_fitness=0.0, max_similarity_between_promoted=0.95)


# ---------------------------------------------------------------------------
# the three stated criteria
# ---------------------------------------------------------------------------
def test_a_promotion_row_exists_before_its_run(
    evolved: tuple[Harness, list[ScoredCandidate]],
) -> None:
    """Section 22's first criterion.

    The row is written with no ``run_id`` — the run does not exist yet — and the
    run is attached afterwards. A promotion written after its run would document
    a decision already taken, which is a log rather than a gate.
    """
    harness, candidates = evolved
    promotions = promote(
        harness.store, candidates, evolution_id=EVOLUTION_ID, gen_index=2, settings=loose()
    )
    assert promotions

    for record in promotions:
        rows = harness.store.promotions_for(record.candidate_id)
        assert rows
        assert rows[0].run_id is None
        assert rows[0].segment == "val"
        assert rows[0].created_at > 0


def test_a_validation_run_without_a_promotion_is_refused(
    evolved: tuple[Harness, list[ScoredCandidate]],
) -> None:
    """Section 22's second criterion, and INV-9's enforcement point."""
    harness, candidates = evolved
    unpromoted = candidates[0].candidate_id

    with pytest.raises(StrategyError, match="only be created for a promoted candidate"):
        require_promotion(harness.store, unpromoted, "val")

    promote(
        harness.store,
        [candidates[0]],
        evolution_id=EVOLUTION_ID,
        gen_index=2,
        settings=loose(),
    )
    assert require_promotion(harness.store, unpromoted, "val") is not None


def test_promoting_three_candidates_increments_touches_by_three(
    evolved: tuple[Harness, list[ScoredCandidate]],
) -> None:
    """Section 22's third criterion.

    Touches are counted per *family*: section 14.1's budget is about how often one
    idea has been measured against the validation set, and every candidate here
    descends from a genome registered under its own family, so three promotions
    of three families is three touches.
    """
    harness, candidates = evolved
    before = _total_touches(harness)

    promotions = promote(
        harness.store, candidates, evolution_id=EVOLUTION_ID, gen_index=2, settings=loose()
    )
    assert len(promotions) == 3
    assert _total_touches(harness) == before + 3
    assert all(record.touches >= 1 for record in promotions)


def _total_touches(harness: Harness) -> int:
    from sqlalchemy import select

    from quantlab.adapters.store.models import StrategyFamily
    from quantlab.adapters.store.sqlite import session_scope

    with session_scope(harness.store.factory) as session:
        rows = session.execute(select(StrategyFamily)).scalars().all()
        return sum(int(row.validation_touches) for row in rows)


# ---------------------------------------------------------------------------
# who is promoted
# ---------------------------------------------------------------------------
def test_only_candidates_above_the_fitness_floor_are_promoted(
    evolved: tuple[Harness, list[ScoredCandidate]],
) -> None:
    """Section 13.7. A validation touch spent on a candidate nobody believes in
    is a touch the family cannot spend on one it does."""
    _harness, candidates = evolved
    settings = PromotionSettings(n_promote=3, min_fitness=0.99)
    assert eligible(candidates, settings) == []


def test_a_rejected_candidate_is_never_promoted(
    evolved: tuple[Harness, list[ScoredCandidate]],
) -> None:
    _harness, candidates = evolved
    rejected = [c for c in candidates if c.fitness == FITNESS_REJECTED]
    if not rejected:
        pytest.skip("this fixture produced no rejected candidate")
    chosen = eligible(candidates, PromotionSettings(n_promote=len(candidates), min_fitness=0.0))
    assert all(c.fitness > FITNESS_REJECTED for c in chosen)


def test_the_fittest_candidate_is_promoted_first(
    evolved: tuple[Harness, list[ScoredCandidate]],
) -> None:
    _harness, candidates = evolved
    chosen = eligible(candidates, loose())
    assert chosen[0].fitness == max(c.fitness for c in candidates)
    assert [c.fitness for c in chosen] == sorted((c.fitness for c in chosen), reverse=True)


def test_three_promotions_are_never_spent_on_one_strategy() -> None:
    """Section 13.7's similarity rule. Three promotions of the same strategy would
    spend three of a family's twenty touches learning one thing."""
    signature = Signature(triples=(("indicator", "sma", ">"),))
    twins = [
        ScoredCandidate(
            view=CandidateView(
                candidate_id=f"twin{index}",
                fitness=0.9 - index * 0.01,
                signature=signature,
                positions=None,
            )
        )
        for index in range(4)
    ]
    settings = PromotionSettings(n_promote=3, min_fitness=0.0, max_similarity_between_promoted=0.8)
    chosen = eligible(twins, settings)
    assert [c.candidate_id for c in chosen] == ["twin0"]


def test_the_promotion_count_is_capped(evolved: tuple[Harness, list[ScoredCandidate]]) -> None:
    _harness, candidates = evolved
    settings = PromotionSettings(n_promote=1, min_fitness=0.0, max_similarity_between_promoted=0.95)
    assert len(eligible(candidates, settings)) == 1


# ---------------------------------------------------------------------------
# what a promotion may authorise
# ---------------------------------------------------------------------------
def test_a_promotion_never_opens_the_test_partition(
    evolved: tuple[Harness, list[ScoredCandidate]],
) -> None:
    """The lockbox is a separate door with its own record (section 14.6), and no
    amount of evolutionary success opens it."""
    harness, candidates = evolved
    for segment in ("test", "lockbox", "train"):
        with pytest.raises(StrategyError, match="authorises a validation run"):
            promote(
                harness.store,
                candidates,
                evolution_id=EVOLUTION_ID,
                gen_index=2,
                settings=loose(),
                segment=segment,
            )
    assert PROMOTED_SEGMENTS == ("val",)


def test_a_promotion_for_one_segment_does_not_authorise_another(
    evolved: tuple[Harness, list[ScoredCandidate]],
) -> None:
    harness, candidates = evolved
    promote(
        harness.store, [candidates[0]], evolution_id=EVOLUTION_ID, gen_index=2, settings=loose()
    )
    assert require_promotion(harness.store, candidates[0].candidate_id, "val")
    with pytest.raises(StrategyError, match="only be created for a promoted candidate"):
        require_promotion(harness.store, candidates[0].candidate_id, "test")


def test_touches_can_be_switched_off_for_a_dry_run(
    evolved: tuple[Harness, list[ScoredCandidate]],
) -> None:
    """``counts_as_validation_touch`` exists so a rehearsal does not spend budget;
    the promotion is still recorded, because the row is the gate."""
    harness, candidates = evolved
    before = _total_touches(harness)
    settings = PromotionSettings(
        n_promote=1,
        min_fitness=0.0,
        max_similarity_between_promoted=0.95,
        counts_as_validation_touch=False,
    )
    promotions = promote(
        harness.store, candidates, evolution_id=EVOLUTION_ID, gen_index=2, settings=settings
    )
    assert len(promotions) == 1
    assert _total_touches(harness) == before
    assert require_promotion(harness.store, promotions[0].candidate_id, "val")


def test_promoting_nothing_is_not_an_error(
    evolved: tuple[Harness, list[ScoredCandidate]],
) -> None:
    """A generation where nothing cleared the floor is a result, not a failure."""
    harness, candidates = evolved
    assert (
        promote(
            harness.store,
            candidates,
            evolution_id=EVOLUTION_ID,
            gen_index=2,
            settings=PromotionSettings(n_promote=3, min_fitness=0.99),
        )
        == []
    )
