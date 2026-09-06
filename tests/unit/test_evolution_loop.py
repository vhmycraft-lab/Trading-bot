"""The generation loop (master spec section 13.2, task T52).

Section 20 names four things for this file — generation bookkeeping, cache hits
for survivors, the stopping rules, and a failed candidate scored
``FITNESS_REJECTED`` rather than dropped — and each one is a way the loop could
go quietly wrong.

The last is the sharpest. A generation that shrank when a strategy raised would
break the population identity of step 7, and every deflated Sharpe ratio computed
afterwards would be charging the wrong number of trials.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from tests.unit.conftest_evolution import Harness, build_harness, small_settings

from quantlab.core.errors import EngineError, StrategyError
from quantlab.core.fitness import FITNESS_REJECTED
from quantlab.evolution.lineage import verify_run
from quantlab.evolution.loop import STOP_REASONS, evolve, resume_point


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    return build_harness(tmp_path)


def run(harness: Harness, evolution_id: str = "ev0", **overrides: Any):  # type: ignore[no-untyped-def]
    harness.open_run(evolution_id)
    return evolve(**harness.kwargs(evolution_id, **overrides))


# ---------------------------------------------------------------------------
# bookkeeping
# ---------------------------------------------------------------------------
def test_the_population_holds_at_exactly_its_size_every_generation(harness: Harness) -> None:
    """Section 13.2 step 7. Section 22 asks for this over five generations."""
    result = run(harness)
    assert len(result.generations) == harness.settings.max_generations
    assert {len(generation.scored) for generation in result.generations} == {
        harness.settings.population_size
    }


def test_every_candidate_is_recorded(harness: Harness) -> None:
    result = run(harness)
    stored = harness.store.candidates_for("ev0")
    assert len(stored) == harness.settings.population_size * len(result.generations)
    assert {row.candidate_id for row in stored} == set(result.candidate_ids)


def test_every_generation_is_recorded_with_its_statistics(harness: Harness) -> None:
    result = run(harness)
    rows = harness.store.generations_for("ev0")
    assert [row.gen_index for row in rows] == list(range(len(result.generations)))
    for row, generation in zip(rows, result.generations, strict=True):
        assert row.n_evaluated == generation.n_evaluated
        assert row.n_cache_hits == generation.n_cache_hits
        assert row.diversity == pytest.approx(generation.diversity)
        assert row.n_immigrants_used == generation.n_immigrants_used


def test_the_evaluation_count_matches_the_evaluations_made(harness: Harness) -> None:
    """Section 22: ``n_evaluations`` matches what the search actually did.

    It counts cache hits too, because a candidate evaluated once and reused ten
    times was still one hypothesis tested, and section 14.4 charges hypotheses.
    """
    result = run(harness)
    assert result.n_evaluations == sum(g.n_evaluated for g in result.generations)
    assert result.n_evaluations == harness.settings.population_size * len(result.generations)


def test_the_distinct_run_count_is_lower_than_the_evaluation_count(harness: Harness) -> None:
    """Which is the point of the cache: a survivor carried forward costs nothing."""
    result = run(harness)
    assert len(harness.store.query_runs()) < result.n_evaluations


def test_a_survivor_carried_forward_is_served_from_the_cache(harness: Harness) -> None:
    """Section 20's stated criterion, and section 13.2 step 1's reason for it."""
    result = run(harness)
    assert result.generations[0].n_cache_hits == 0
    assert all(generation.n_cache_hits > 0 for generation in result.generations[1:])


def test_every_candidate_carries_its_lineage(harness: Harness) -> None:
    result = run(harness)
    rows = {row.candidate_id: row for row in harness.store.candidates_for("ev0")}
    for row in rows.values():
        if row.parent_candidate_id is not None:
            assert row.parent_candidate_id in rows
        assert row.origin in ("seed", "survivor", "mutant", "immigrant")
    assert verify_run(harness.store, "ev0").ok
    del result


def test_the_rank_and_survivor_flag_are_written(harness: Harness) -> None:
    """A ranking that cannot be read back cannot be audited (section 13.2 step 2)."""
    run(harness)
    first = [row for row in harness.store.candidates_for("ev0") if row.gen_index == 0]
    assert sorted(row.rank for row in first) == list(range(1, len(first) + 1))
    assert sum(1 for row in first if row.survived) <= harness.settings.n_survivors


def test_the_components_and_penalties_are_stored_for_auditing(harness: Harness) -> None:
    """Section 13.2 step 2: a ranking can be taken apart after the fact."""
    import json

    run(harness)
    scored = [
        row
        for row in harness.store.candidates_for("ev0")
        if row.fitness is not None and row.fitness > FITNESS_REJECTED
    ]
    assert scored
    for row in scored:
        assert set(json.loads(row.components_json))
        assert set(json.loads(row.penalties_json))


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------
def test_the_same_seed_produces_the_same_candidate_sequence(tmp_path: Path) -> None:
    """INV-7, and what makes ``resume`` correct: a run that resumed into a
    different sequence would silently be a different search."""
    first = build_harness(tmp_path / "a")
    second = build_harness(tmp_path / "b")
    assert run(first).candidate_ids == run(second).candidate_ids


def test_a_different_seed_searches_differently(tmp_path: Path) -> None:
    first = build_harness(tmp_path / "a", small_settings(seed=1))
    second = build_harness(tmp_path / "b", small_settings(seed=2))
    assert run(first).candidate_ids != run(second).candidate_ids


def test_a_candidate_id_depends_on_the_candidate_not_on_the_clock(harness: Harness) -> None:
    result = run(harness)
    assert len(set(result.candidate_ids)) == len(result.candidate_ids)
    assert all(len(candidate_id) == 16 for candidate_id in result.candidate_ids)


# ---------------------------------------------------------------------------
# a candidate that fails
# ---------------------------------------------------------------------------
def test_a_failed_candidate_is_rejected_rather_than_dropped(tmp_path: Path) -> None:
    """Section 20's stated criterion. The generation must still hold
    ``population_size`` candidates, and the failure must be in the record: a
    generation that shrank when a strategy raised would break the population
    identity of step 7, and every deflated Sharpe ratio afterwards would charge
    the wrong number of trials."""

    class Exploding:
        engine_name = "simple_bar"
        engine_version = "1"

        def __init__(self, source: str) -> None:
            self.source = source

        def evaluate(self, *_args: Any, **_kwargs: Any) -> Any:
            raise StrategyError("this candidate always fails")

    harness = build_harness(tmp_path, small_settings(max_generations=1))
    result = run(harness, evaluator_for=Exploding)

    generation = result.generations[0]
    assert len(generation.scored) == harness.settings.population_size
    assert generation.n_rejected_by_gate == harness.settings.population_size
    assert all(candidate.fitness == FITNESS_REJECTED for candidate in generation.scored)
    assert result.best is None

    stored = harness.store.candidates_for("ev0")
    assert len(stored) == harness.settings.population_size
    assert all(row.gate_failure == "StrategyError" for row in stored)


def test_an_engine_error_is_not_hidden_behind_a_rejected_candidate(tmp_path: Path) -> None:
    """It means an accounting invariant broke inside the engine, which is a
    defect in the platform. Recording it as a rejected candidate would bury a bug
    behind a plausible-looking result."""

    class Broken:
        engine_name = "simple_bar"
        engine_version = "1"

        def __init__(self, source: str) -> None:
            self.source = source

        def evaluate(self, *_args: Any, **_kwargs: Any) -> Any:
            raise EngineError("equity does not reconcile")

    harness = build_harness(tmp_path, small_settings(max_generations=1))
    with pytest.raises(EngineError, match="does not reconcile"):
        run(harness, evaluator_for=Broken)


# ---------------------------------------------------------------------------
# stopping
# ---------------------------------------------------------------------------
def test_a_run_stops_at_max_generations(harness: Harness) -> None:
    result = run(harness)
    assert result.stop_reason == "max_generations"
    assert len(result.generations) == harness.settings.max_generations


def test_a_run_stops_when_the_evaluation_budget_is_spent(tmp_path: Path) -> None:
    """Section 13.2: stopping early does not change ``n_evaluations``, which is
    what section 14.4 charges."""
    harness = build_harness(tmp_path, small_settings(max_evaluations=16))
    result = run(harness)
    assert result.stop_reason == "max_evaluations"
    assert len(result.generations) == 2
    assert result.n_evaluations == 16


def test_a_run_stops_when_the_wall_clock_is_spent(tmp_path: Path) -> None:
    """The clock is injected so the rule can be tested without waiting for it."""
    harness = build_harness(tmp_path, small_settings(max_wall_clock_s=1))
    ticks = iter([0.0, 5.0, 10.0, 15.0, 20.0, 25.0])
    result = run(harness, clock=lambda: next(ticks))
    assert result.stop_reason == "max_wall_clock_s"
    assert len(result.generations) == 1


def test_a_run_stops_when_it_stops_improving(tmp_path: Path) -> None:
    harness = build_harness(
        tmp_path,
        small_settings(max_generations=10, stop_on_no_improvement_generations=2),
    )
    result = run(harness)
    assert result.stop_reason == "no_improvement"
    assert len(result.generations) < 10


def test_every_stop_reason_is_one_the_schema_allows(tmp_path: Path) -> None:
    """``evolution_run.stop_reason`` is what a later reader consults to know
    whether a search finished or ran out of budget."""
    harness = build_harness(tmp_path, small_settings(max_generations=1))
    assert run(harness).stop_reason in STOP_REASONS


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------
def test_a_finished_run_has_nothing_left_to_resume(harness: Harness) -> None:
    result = run(harness)
    assert resume_point(harness.store, "ev0") == len(result.generations)


def test_an_unstarted_run_resumes_at_zero(harness: Harness) -> None:
    assert resume_point(harness.store, "never-ran") == 0
