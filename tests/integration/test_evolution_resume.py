"""Resuming an interrupted evolution (master spec sections 13.9, 20; task T52).

Section 20's scenario: "kill mid-generation, ``evolve resume``, no duplicate runs
and no lost candidates".

The interruption is simulated by running fewer generations than the budget and
then re-entering, which is exactly the state a crash leaves: some generations
recorded, the rest not. What makes it safe is the run cache of section 11.2 —
every evaluation the interrupted run completed is already in the store, so
re-entering re-evaluates without re-executing.

**Both environment modes.** Every test here runs twice: once with the loop's
``environment_for`` unset, and once with Rome's real per-generation provider
(spec 1.2, sections 4-19). Resume is the invariant per-generation environments
most endanger — a resumed generation must be handed the environment it was
recorded with, or the resumed run is a different search wearing the same run id —
and a guarantee demonstrated in only one file is one refactor away from holding
in none. The environment-specific mechanics (recording order, audit rows, the
sampling report) stay in ``test_environment_evolution.py``; what is proved here
is that resume itself does not care which mode it is in.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.unit.conftest_evolution import (
    Harness,
    build_harness,
    environment_provider,
    small_settings,
)

from quantlab.evolution.lineage import verify_run
from quantlab.evolution.loop import EnvironmentFor, evolve, resume_point

pytestmark = pytest.mark.slow

EVOLUTION_ID = "ev_resume"


@pytest.fixture(
    params=[
        pytest.param(False, id="no_environment"),
        pytest.param(True, id="per_generation_environment"),
    ]
)
def with_environment(request: pytest.FixtureRequest) -> bool:
    """Run every test in this module under both environment modes."""
    return bool(request.param)


def interrupted(
    root: Path, *, after: int = 2, with_environment: bool = False
) -> tuple[Harness, EnvironmentFor | None]:
    """A run stopped after ``after`` generations of a five-generation budget.

    Returns the harness and the provider it ran under. The provider has to travel
    with the harness: it is the object holding the recorded environments, and
    handing the resume a *fresh* one would re-draw rather than replay.
    """
    harness = build_harness(root, small_settings(max_generations=5))
    harness.open_run(EVOLUTION_ID)
    provider = environment_provider(harness, EVOLUTION_ID) if with_environment else None
    evolve(
        **harness.kwargs(
            EVOLUTION_ID,
            settings=small_settings(max_generations=after),
            environment_for=provider,
        )
    )
    return harness, provider


def test_the_resume_point_is_the_first_unrecorded_generation(
    tmp_path: Path, with_environment: bool
) -> None:
    """Section 13.9's rule, and why the generation row is written last."""
    harness, _ = interrupted(tmp_path, after=2, with_environment=with_environment)
    assert resume_point(harness.store, EVOLUTION_ID) == 2
    assert len(harness.store.generations_for(EVOLUTION_ID)) == 2


def test_resuming_finishes_the_run_without_losing_a_candidate(
    tmp_path: Path, with_environment: bool
) -> None:
    harness, provider = interrupted(tmp_path, after=2, with_environment=with_environment)
    before = len(harness.store.candidates_for(EVOLUTION_ID))
    assert before == 2 * harness.settings.population_size

    result = evolve(**harness.kwargs(EVOLUTION_ID, start_generation=2, environment_for=provider))

    stored = harness.store.candidates_for(EVOLUTION_ID)
    assert len(stored) == 5 * harness.settings.population_size
    assert len({row.candidate_id for row in stored}) == len(stored)
    assert len(result.generations) == 3
    assert [row.gen_index for row in harness.store.generations_for(EVOLUTION_ID)] == [0, 1, 2, 3, 4]


def test_resuming_creates_no_duplicate_runs(tmp_path: Path, with_environment: bool) -> None:
    """Section 13.9: "re-using cached runs so no work is repeated"."""
    harness, provider = interrupted(tmp_path, after=2, with_environment=with_environment)
    runs_before = {row.run_id for row in harness.store.query_runs()}

    evolve(**harness.kwargs(EVOLUTION_ID, start_generation=2, environment_for=provider))

    runs_after = {row.run_id for row in harness.store.query_runs()}
    assert runs_before <= runs_after, "an existing run must never be replaced"
    assert len(runs_after) > len(runs_before), "later generations do produce new work"


def test_a_resumed_run_matches_one_that_was_never_interrupted(
    tmp_path: Path, with_environment: bool
) -> None:
    """The strongest statement of correctness available: interruption changes
    nothing about the search, because determinism comes from the seed rather than
    from the process (INV-7)."""
    split, provider = interrupted(tmp_path / "split", after=2, with_environment=with_environment)
    evolve(**split.kwargs(EVOLUTION_ID, start_generation=2, environment_for=provider))

    whole = build_harness(tmp_path / "whole", small_settings(max_generations=5))
    whole.open_run(EVOLUTION_ID)
    # The uninterrupted run is handed the *same* provider. Environments are drawn
    # from the OS CSPRNG (Rome section 5), so a second provider would draw a
    # different five and the comparison would be measuring the randomness rather
    # than the interruption. `provider` is bound to `split`'s store, where all
    # five are now recorded, so re-using it replays them against `whole`'s
    # identical bars and config. With no environment in play it is None and this
    # is the plain uninterrupted run.
    evolve(**whole.kwargs(EVOLUTION_ID, environment_for=provider))

    def snapshot(harness: Harness) -> list[tuple[object, ...]]:
        return [
            (row.gen_index, row.candidate_id, row.origin, row.fitness)
            for row in harness.store.candidates_for(EVOLUTION_ID)
        ]

    replayed = snapshot(split)
    # Compared lists that were both empty would satisfy the assertion below while
    # proving nothing, and "the store returned nothing" is a plausible failure —
    # a resume that recorded into the wrong run id looks exactly like it. State
    # the size the comparison is supposed to be over before making it.
    assert len(replayed) == 5 * whole.settings.population_size
    assert replayed == snapshot(whole)


def test_a_resumed_run_still_replays(tmp_path: Path, with_environment: bool) -> None:
    """INV-10 does not care that the run was interrupted."""
    harness, provider = interrupted(tmp_path, after=2, with_environment=with_environment)
    evolve(**harness.kwargs(EVOLUTION_ID, start_generation=2, environment_for=provider))
    report = verify_run(harness.store, EVOLUTION_ID)
    assert report.ok, report.mismatched
    assert report.n_checked > 0


def test_a_finished_run_has_nothing_to_resume(tmp_path: Path, with_environment: bool) -> None:
    harness = build_harness(tmp_path, small_settings(max_generations=3))
    harness.open_run(EVOLUTION_ID)
    provider = environment_provider(harness, EVOLUTION_ID) if with_environment else None
    evolve(**harness.kwargs(EVOLUTION_ID, environment_for=provider))
    assert resume_point(harness.store, EVOLUTION_ID) == 3
