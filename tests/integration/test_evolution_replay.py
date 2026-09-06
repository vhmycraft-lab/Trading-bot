"""INV-7 over a whole evolution (master spec section 20, task T52).

Section 20's scenario: "an evolution replayed from its stored seed and config
reproduces the identical ``candidate_id`` sequence".

That is a stronger claim than it sounds. A candidate id here is derived from the
run, the slot and the candidate's own compiled content, so reproducing the
sequence means every draw, every mutation, every niching decision and every
parent selection came out the same — from the seed alone, in a fresh database,
with none of the first run's state available.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.unit.conftest_evolution import Harness, build_harness, small_settings

from quantlab.evolution.loop import evolve

pytestmark = pytest.mark.slow


def replay(root: Path, *, seed: int = 42, generations: int = 3) -> tuple[Harness, object]:
    """One evolution, in its own database, from nothing but its configuration."""
    harness = build_harness(root, small_settings(seed=seed, max_generations=generations))
    harness.open_run("ev_replay")
    return harness, evolve(**harness.kwargs("ev_replay"))


def test_the_candidate_sequence_reproduces_exactly(tmp_path: Path) -> None:
    """Section 20's stated criterion."""
    _first_harness, first = replay(tmp_path / "one")
    _second_harness, second = replay(tmp_path / "two")
    assert first.candidate_ids == second.candidate_ids  # type: ignore[attr-defined]
    assert first.n_evaluations == second.n_evaluations  # type: ignore[attr-defined]
    assert first.stop_reason == second.stop_reason  # type: ignore[attr-defined]


def test_the_stored_record_reproduces_too(tmp_path: Path) -> None:
    """Not just the in-memory result: the rows a later reader would consult."""
    first_harness, _ = replay(tmp_path / "one")
    second_harness, _ = replay(tmp_path / "two")

    def snapshot(harness: Harness) -> list[tuple[object, ...]]:
        return [
            (row.gen_index, row.candidate_id, row.origin, row.genome_json, row.params_json)
            for row in harness.store.candidates_for("ev_replay")
        ]

    assert snapshot(first_harness) == snapshot(second_harness)


def test_the_fitness_of_every_candidate_reproduces(tmp_path: Path) -> None:
    """A replayed run that scored differently would mean the engine, the metrics
    or the fitness function had a source of variation nobody declared."""
    first_harness, _ = replay(tmp_path / "one")
    second_harness, _ = replay(tmp_path / "two")

    def scores(harness: Harness) -> dict[str, float | None]:
        return {row.candidate_id: row.fitness for row in harness.store.candidates_for("ev_replay")}

    assert scores(first_harness) == scores(second_harness)


def test_the_generation_statistics_reproduce(tmp_path: Path) -> None:
    first_harness, _ = replay(tmp_path / "one")
    second_harness, _ = replay(tmp_path / "two")

    def summaries(harness: Harness) -> list[tuple[object, ...]]:
        return [
            (row.gen_index, row.diversity, row.best_fitness, row.n_rejected_by_gate)
            for row in harness.store.generations_for("ev_replay")
        ]

    assert summaries(first_harness) == summaries(second_harness)


def test_a_different_seed_produces_a_different_search(tmp_path: Path) -> None:
    """Guards the tests above against passing because the search is degenerate."""
    _one, first = replay(tmp_path / "one", seed=1)
    _two, second = replay(tmp_path / "two", seed=2)
    assert first.candidate_ids != second.candidate_ids  # type: ignore[attr-defined]


def test_the_recorded_mutations_reproduce(tmp_path: Path) -> None:
    """INV-10 and INV-7 together: the same search records the same edits, so a
    replayed lineage is the lineage."""
    first_harness, _ = replay(tmp_path / "one")
    second_harness, _ = replay(tmp_path / "two")

    def edits(harness: Harness) -> list[tuple[object, ...]]:
        return [
            (row.candidate_id, mutation.seq, mutation.operator, mutation.after_json)
            for row in harness.store.candidates_for("ev_replay")
            for mutation in harness.store.mutations_for(row.candidate_id)
        ]

    recorded = edits(first_harness)
    assert recorded, "a run with no mutations would prove nothing here"
    assert recorded == edits(second_harness)
