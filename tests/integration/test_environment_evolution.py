"""Per-generation hidden environments, inside a real evolution run (Project Rome). 🔒

The selector is tested in ``tests/unit/test_environment.py``; this file is about
what happens when it is wired into the loop, and in particular about the two
orderings that have to hold and cannot be observed anywhere else:

* **an environment is recorded before its generation runs**, so a crashed
  generation still leaves behind what it was run against;
* **a resumed generation replays the environment it was recorded with**, so a
  resumed run stays candidate-for-candidate identical to the run it resumes.

The second is the invariant this feature most endangers. Randomising the training
environment per generation and replaying a run from the store are in direct
tension: if resume re-drew, every resumed run would silently be a different
search wearing the same run id. The tests below pin the resolution — the store is
the authority for a generation that already has one, and only genuinely new
generations draw.
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

from quantlab.core.environment import sampling_report
from quantlab.evolution.loop import evolve

pytestmark = pytest.mark.slow

EVOLUTION_ID = "ev_env"


def run(root: Path, *, generations: int, evolution_id: str = EVOLUTION_ID) -> Harness:
    harness = build_harness(root, small_settings(max_generations=generations))
    harness.open_run(evolution_id)
    provider = environment_provider(harness, evolution_id)
    evolve(
        **harness.kwargs(
            evolution_id,
            settings=small_settings(max_generations=generations),
            environment_for=provider,
        )
    )
    return harness


# ---------------------------------------------------------------------------
# one environment per generation, recorded
# ---------------------------------------------------------------------------
def test_every_generation_gets_its_own_recorded_environment(tmp_path: Path) -> None:
    harness = run(tmp_path, generations=4)
    rows = harness.store.training_environments_for(EVOLUTION_ID)
    assert [row.gen_index for row in rows] == [0, 1, 2, 3]
    assert len({row.environment_id for row in rows}) == 4
    assert len({row.seed_hex for row in rows}) == 4


def test_the_generations_do_not_all_share_one_window(tmp_path: Path) -> None:
    """Rome section 17: "Do not allow every strategy to repeatedly train on
    exactly the same historical window"."""
    harness = run(tmp_path, generations=5)
    rows = harness.store.training_environments_for(EVOLUTION_ID)
    windows = {(row.window_start_ts, row.window_end_ts) for row in rows}
    assert len(windows) > 1


def test_the_recorded_window_never_leaves_the_training_segment(tmp_path: Path) -> None:
    harness = run(tmp_path, generations=4)
    for row in harness.store.training_environments_for(EVOLUTION_ID):
        assert row.window_start_ts >= harness.bars.ts_open[0]
        assert row.window_end_ts <= harness.bars.ts_open[400]


def test_the_environment_is_recorded_before_its_generation_runs(tmp_path: Path) -> None:
    """A record written after the work would be missing exactly the generations
    that crashed, which are the ones an auditor most wants to reconstruct.

    The generation is made to fail; the environment row must be there anyway, and
    the *generation* row must not, because that one is written on completion.
    """
    harness = build_harness(tmp_path, small_settings(max_generations=2))
    harness.open_run(EVOLUTION_ID)
    provider = environment_provider(harness, EVOLUTION_ID)

    def explode(_source: str) -> object:
        raise RuntimeError("the generation failed after its environment was recorded")

    with pytest.raises(RuntimeError, match="after its environment was recorded"):
        evolve(
            **harness.kwargs(
                EVOLUTION_ID,
                settings=small_settings(max_generations=2),
                environment_for=provider,
                evaluator_for=explode,
            )
        )

    assert len(harness.store.training_environments_for(EVOLUTION_ID)) == 1
    assert harness.store.generations_for(EVOLUTION_ID) == []


# ---------------------------------------------------------------------------
# the resume invariant
# ---------------------------------------------------------------------------
def test_a_resumed_generation_replays_its_recorded_environment(tmp_path: Path) -> None:
    """The store is the authority for a generation that already has one.

    Re-drawing here would make the replayed generations evaluate on different
    history while claiming to be the same run.
    """
    harness = run(tmp_path, generations=2)
    before = [
        (row.environment_id, row.seed_hex, row.window_start_ts)
        for row in harness.store.training_environments_for(EVOLUTION_ID)
    ]

    provider = environment_provider(harness, EVOLUTION_ID)
    evolve(
        **harness.kwargs(
            EVOLUTION_ID,
            settings=small_settings(max_generations=4),
            environment_for=provider,
            start_generation=2,
        )
    )
    rows = harness.store.training_environments_for(EVOLUTION_ID)
    after = [(row.environment_id, row.seed_hex, row.window_start_ts) for row in rows[:2]]
    assert after == before
    assert len(rows) == 4


def test_a_resumed_run_reproduces_the_candidates_of_the_run_it_resumes(
    tmp_path: Path,
) -> None:
    """The invariant this feature most endangers, stated directly.

    The replayed generations must be candidate-for-candidate identical. They can
    only be if they were evaluated in the same environments, so this is the
    randomisation-aware form of the resume guarantee: identical where the run is
    replaying, free to differ where it is genuinely new work.
    """
    harness = run(tmp_path, generations=2)
    recorded = [candidate.candidate_id for candidate in harness.store.candidates_for(EVOLUTION_ID)]

    provider = environment_provider(harness, EVOLUTION_ID)
    result = evolve(
        **harness.kwargs(
            EVOLUTION_ID,
            settings=small_settings(max_generations=4),
            environment_for=provider,
            start_generation=2,
        )
    )
    everything = [
        candidate.candidate_id for candidate in harness.store.candidates_for(EVOLUTION_ID)
    ]
    assert everything[: len(recorded)] == recorded
    assert len(result.candidate_ids) > 0
    assert len(everything) > len(recorded)


def test_resuming_twice_adds_no_second_environment_for_a_generation(
    tmp_path: Path,
) -> None:
    """``UNIQUE (evolution_id, gen_index)`` makes a second draw a database error
    rather than a silently different experiment; the provider must never reach
    that constraint, because it reads before it writes."""
    harness = run(tmp_path, generations=3)
    for _ in range(2):
        provider = environment_provider(harness, EVOLUTION_ID)
        evolve(
            **harness.kwargs(
                EVOLUTION_ID,
                settings=small_settings(max_generations=3),
                environment_for=provider,
                start_generation=3,
            )
        )
    rows = harness.store.training_environments_for(EVOLUTION_ID)
    assert [row.gen_index for row in rows] == [0, 1, 2]


# ---------------------------------------------------------------------------
# the audit path
# ---------------------------------------------------------------------------
def test_a_privileged_audit_reproduces_every_recorded_environment(
    tmp_path: Path,
) -> None:
    """Rome sections 11 and 24. The seed is enough; everything else in the row is
    what the reproduction is checked against."""
    harness = run(tmp_path, generations=4)
    provider = environment_provider(harness, EVOLUTION_ID)
    for row in harness.store.training_environments_for(EVOLUTION_ID):
        rebuilt = provider.reproduce(row)
        assert rebuilt.environment_id == row.environment_id
        assert rebuilt.window.start_ts == row.window_start_ts
        assert rebuilt.window.end_ts == row.window_end_ts
        assert rebuilt.starting_capital == pytest.approx(row.starting_capital)
        assert rebuilt.slippage_bps == pytest.approx(row.slippage_bps)


def test_a_runs_history_exposure_can_be_summarised_from_its_records(
    tmp_path: Path,
) -> None:
    """Rome section 43: the exposure statistics are computable from the audit
    trail alone, which is what makes them auditable rather than merely printed."""
    from quantlab.core.environment import TrainingWindow

    harness = run(tmp_path, generations=5)
    provider = environment_provider(harness, EVOLUTION_ID)
    windows = [
        TrainingWindow(start_ts=row.window_start_ts, end_ts=row.window_end_ts)
        for row in harness.store.training_environments_for(EVOLUTION_ID)
    ]
    from tests.unit.conftest_evolution import environment_settings

    report = sampling_report(windows, provider.pool, environment_settings())
    assert report.n_environments == 5
    assert report.concentration > 0.0


# ---------------------------------------------------------------------------
# what the environment actually changes
# ---------------------------------------------------------------------------
def test_the_environment_varies_capital_and_slippage_but_never_commission(
    tmp_path: Path,
) -> None:
    """Rome sections 10-12. The two that may move do; the one that may not, does
    not — checked on the settings the loop was actually handed."""
    harness = build_harness(tmp_path, small_settings(max_generations=1))
    harness.open_run(EVOLUTION_ID)
    provider = environment_provider(harness, EVOLUTION_ID)
    resolved = [provider(index) for index in range(6)]

    assert len({env.config.initial_equity for env in resolved}) > 1
    assert len({env.config.slippage.fixed_bps for env in resolved}) > 1
    assert {env.config.fee_bps for env in resolved} == {harness.config.fee_bps}
    assert {env.config.fill_rule for env in resolved} == {harness.config.fill_rule}
    assert {env.config.cost_multiplier for env in resolved} == {harness.config.cost_multiplier}


def test_the_environment_qualifies_the_segment_so_the_run_cache_stays_honest(
    tmp_path: Path,
) -> None:
    """A run's identity (section 11.2) does not include the bar range, so two
    generations that drew different windows but the same capital and slippage
    would share a run id and the second would be served the first one's results.
    The segment carries the opaque environment id to prevent exactly that."""
    harness = build_harness(tmp_path, small_settings(max_generations=1))
    harness.open_run(EVOLUTION_ID)
    provider = environment_provider(harness, EVOLUTION_ID)
    first, second = provider(0), provider(1)
    assert first.segment_for("train") != second.segment_for("train")
    assert first.segment_for("train").startswith("train:")


def test_the_runs_recorded_by_an_evolution_name_their_environment(
    tmp_path: Path,
) -> None:
    """And the name is opaque: the segment carries the id, never the dates."""
    harness = run(tmp_path, generations=2)
    segments = {
        row.segment for row in harness.store.query_runs(experiment_id=harness.experiment_id)
    }
    assert all(segment.startswith("train:") for segment in segments), segments
    recorded = {row.environment_id for row in harness.store.training_environments_for(EVOLUTION_ID)}
    for segment in segments:
        suffix = segment.split(":", 1)[1]
        assert len(suffix) == 16
        assert suffix in recorded
        # Opaque: hexadecimal, and nothing a reader could mistake for a date.
        assert all(character in "0123456789abcdef" for character in suffix)


def test_a_survivor_is_re_evaluated_on_the_next_generations_history(
    tmp_path: Path,
) -> None:
    """The cost consequence, measured rather than assumed.

    With one fixed environment a survivor carried forward is a cache hit. Under
    per-generation environments it is not, and that is the point: a survivor that
    has only ever been measured on one window has not been shown to survive
    anything. This test exists because the opposite — a survivor silently served
    a cached result from a *different* window — is the failure mode that would
    make the whole feature decorative.
    """
    fixed = build_harness(tmp_path / "fixed", small_settings(max_generations=3))
    fixed.open_run("ev_fixed")
    fixed_result = evolve(**fixed.kwargs("ev_fixed", settings=small_settings(max_generations=3)))

    varied = build_harness(tmp_path / "varied", small_settings(max_generations=3))
    varied.open_run("ev_varied")
    varied_result = evolve(
        **varied.kwargs(
            "ev_varied",
            settings=small_settings(max_generations=3),
            environment_for=environment_provider(varied, "ev_varied"),
        )
    )

    def hits(result: object) -> int:
        return sum(outcome.n_cache_hits for outcome in result.generations)  # type: ignore[attr-defined]

    assert hits(fixed_result) > 0, "the fixed-environment run should reuse survivors"
    assert hits(varied_result) < hits(fixed_result)
