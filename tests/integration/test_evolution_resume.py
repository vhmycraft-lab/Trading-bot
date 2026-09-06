"""Resuming an interrupted evolution (master spec sections 13.9, 20; task T52).

Section 20's scenario: "kill mid-generation, ``evolve resume``, no duplicate runs
and no lost candidates".

The interruption is simulated by running fewer generations than the budget and
then re-entering, which is exactly the state a crash leaves: some generations
recorded, the rest not. What makes it safe is the run cache of section 11.2 —
every evaluation the interrupted run completed is already in the store, so
re-entering re-evaluates without re-executing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.unit.conftest_evolution import Harness, build_harness, small_settings

from quantlab.evolution.lineage import verify_run
from quantlab.evolution.loop import evolve, resume_point

pytestmark = pytest.mark.slow

EVOLUTION_ID = "ev_resume"


def interrupted(root: Path, *, after: int = 2) -> Harness:
    """A run stopped after ``after`` generations of a five-generation budget."""
    harness = build_harness(root, small_settings(max_generations=5))
    harness.open_run(EVOLUTION_ID)
    evolve(**harness.kwargs(EVOLUTION_ID, settings=small_settings(max_generations=after)))
    return harness


def test_the_resume_point_is_the_first_unrecorded_generation(tmp_path: Path) -> None:
    """Section 13.9's rule, and why the generation row is written last."""
    harness = interrupted(tmp_path, after=2)
    assert resume_point(harness.store, EVOLUTION_ID) == 2
    assert len(harness.store.generations_for(EVOLUTION_ID)) == 2


def test_resuming_finishes_the_run_without_losing_a_candidate(tmp_path: Path) -> None:
    harness = interrupted(tmp_path, after=2)
    before = len(harness.store.candidates_for(EVOLUTION_ID))
    assert before == 2 * harness.settings.population_size

    result = evolve(**harness.kwargs(EVOLUTION_ID, start_generation=2))

    stored = harness.store.candidates_for(EVOLUTION_ID)
    assert len(stored) == 5 * harness.settings.population_size
    assert len({row.candidate_id for row in stored}) == len(stored)
    assert len(result.generations) == 3
    assert [row.gen_index for row in harness.store.generations_for(EVOLUTION_ID)] == [0, 1, 2, 3, 4]


def test_resuming_creates_no_duplicate_runs(tmp_path: Path) -> None:
    """Section 13.9: "re-using cached runs so no work is repeated"."""
    harness = interrupted(tmp_path, after=2)
    runs_before = {row.run_id for row in harness.store.query_runs()}

    evolve(**harness.kwargs(EVOLUTION_ID, start_generation=2))

    runs_after = {row.run_id for row in harness.store.query_runs()}
    assert runs_before <= runs_after, "an existing run must never be replaced"
    assert len(runs_after) > len(runs_before), "later generations do produce new work"


def test_a_resumed_run_matches_one_that_was_never_interrupted(tmp_path: Path) -> None:
    """The strongest statement of correctness available: interruption changes
    nothing about the search, because determinism comes from the seed rather than
    from the process (INV-7)."""
    interrupted_harness = interrupted(tmp_path / "split", after=2)
    evolve(**interrupted_harness.kwargs(EVOLUTION_ID, start_generation=2))

    whole = build_harness(tmp_path / "whole", small_settings(max_generations=5))
    whole.open_run(EVOLUTION_ID)
    evolve(**whole.kwargs(EVOLUTION_ID))

    def snapshot(harness: Harness) -> list[tuple[object, ...]]:
        return [
            (row.gen_index, row.candidate_id, row.origin, row.fitness)
            for row in harness.store.candidates_for(EVOLUTION_ID)
        ]

    assert snapshot(interrupted_harness) == snapshot(whole)


def test_a_resumed_run_still_replays(tmp_path: Path) -> None:
    """INV-10 does not care that the run was interrupted."""
    harness = interrupted(tmp_path, after=2)
    evolve(**harness.kwargs(EVOLUTION_ID, start_generation=2))
    report = verify_run(harness.store, EVOLUTION_ID)
    assert report.ok, report.mismatched
    assert report.n_checked > 0


def test_a_finished_run_has_nothing_to_resume(tmp_path: Path) -> None:
    harness = build_harness(tmp_path, small_settings(max_generations=3))
    harness.open_run(EVOLUTION_ID)
    evolve(**harness.kwargs(EVOLUTION_ID))
    assert resume_point(harness.store, EVOLUTION_ID) == 3
