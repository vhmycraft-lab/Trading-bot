"""A whole evolution run (master spec section 20, task T52).

Section 20's scenario, verbatim: "5 generations by population 8 on the
synthetic-trend fixture: population size held exactly, lineage complete,
diversity above the floor, ``n_evaluations`` recorded".

Everything here is offline and real: the genome compiler, the AST check, the
store, the run cache, the fitness function and the engine. Only the sandbox is
replaced by an in-process evaluator, for the reason the harness module states — a
child process per candidate per generation would make this test cost minutes, and
every candidate in it was compiled by this repository from a genome the test drew.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.unit.conftest_evolution import Harness, build_harness, small_settings

from quantlab.core.fitness import FITNESS_REJECTED
from quantlab.evolution.diversity import population_diversity
from quantlab.evolution.lineage import lineage_tree, verify_run
from quantlab.evolution.loop import evolve

pytestmark = pytest.mark.slow

EVOLUTION_ID = "ev_e2e"


@pytest.fixture(scope="module")
def finished(tmp_path_factory: pytest.TempPathFactory) -> tuple[Harness, object]:
    """One five-generation run, shared by every assertion below."""
    harness = build_harness(tmp_path_factory.mktemp("evolution"), small_settings())
    harness.open_run(EVOLUTION_ID)
    return harness, evolve(**harness.kwargs(EVOLUTION_ID))


def test_the_population_is_held_at_exactly_its_size(finished: tuple[Harness, object]) -> None:
    """Section 13.2 step 7. A generation that quietly resized would make every
    downstream deflated Sharpe ratio wrong."""
    harness, result = finished
    assert len(result.generations) == 5  # type: ignore[attr-defined]
    for generation in result.generations:  # type: ignore[attr-defined]
        assert len(generation.scored) == harness.settings.population_size


def test_the_lineage_is_complete(finished: tuple[Harness, object]) -> None:
    """Every candidate but a seed names a parent that exists, and every chain
    reaches a seed without a cycle (INV-10's structural half)."""
    harness, _ = finished
    tree = lineage_tree(harness.store, EVOLUTION_ID)
    assert tree.nodes
    assert tree.roots

    for node in tree.nodes.values():
        if node.parent_candidate_id is not None:
            assert node.parent_candidate_id in tree.nodes
        path = tree.path_to_root(node.candidate_id)
        assert tree.nodes[path[-1]].parent_candidate_id is None


def test_every_recorded_mutation_replays(finished: tuple[Harness, object]) -> None:
    """INV-10 itself, over the whole run."""
    harness, _ = finished
    report = verify_run(harness.store, EVOLUTION_ID)
    assert report.ok, report.mismatched
    assert report.n_checked > 0, "a run with nothing to replay proves nothing"


def test_diversity_stays_above_collapse(finished: tuple[Harness, object]) -> None:
    """Section 13.5's whole purpose: without the mechanism the fittest
    candidate's descendants fill every slot within a few generations."""
    harness, result = finished
    for generation in result.generations:  # type: ignore[attr-defined]
        measured = population_diversity(
            [candidate.view for candidate in generation.scored], harness.settings.diversity
        )
        assert measured == pytest.approx(generation.diversity)
        assert measured > 0.0

    stored = harness.store.generations_for(EVOLUTION_ID)
    assert all(row.diversity > 0.0 for row in stored)


def test_at_least_one_immigrant_per_generation(finished: tuple[Harness, object]) -> None:
    """Section 13.5's reserved novelty: every generation holds a candidate that
    owes nothing to the current leader."""
    _harness, result = finished
    for generation in result.generations:  # type: ignore[attr-defined]
        assert generation.n_immigrants_used >= 1


def test_the_evaluation_count_is_recorded_on_the_run(finished: tuple[Harness, object]) -> None:
    """Section 14.4 charges ``n_evaluations`` into the deflated Sharpe ratio's
    ``M``, so a count that lived only in memory would make every later verdict
    too generous."""
    harness, result = finished
    row = harness.store.find_evolution_run(EVOLUTION_ID)
    assert row is not None
    assert row.n_evaluations == result.n_evaluations  # type: ignore[attr-defined]
    assert row.n_evaluations == 5 * harness.settings.population_size


def test_survivors_cost_nothing_to_carry_forward(finished: tuple[Harness, object]) -> None:
    """Section 13.2 step 1: only genuinely new candidates consume budget."""
    harness, result = finished
    distinct = len(harness.store.query_runs())
    assert distinct < result.n_evaluations  # type: ignore[attr-defined]
    assert sum(g.n_cache_hits for g in result.generations) > 0  # type: ignore[attr-defined]


def test_every_run_the_evolution_created_is_on_the_train_segment(
    finished: tuple[Harness, object],
) -> None:
    """INV-9, read back from the store rather than asserted in the loop."""
    harness, _ = finished
    segments = {row.segment for row in harness.store.query_runs()}
    assert segments == {"train"}


def test_every_candidate_carries_its_genome_and_signature(
    finished: tuple[Harness, object],
) -> None:
    """A candidate without them could not be replayed or compared to another."""
    harness, _ = finished
    for row in harness.store.candidates_for(EVOLUTION_ID):
        assert row.kind == "genome"
        assert json.loads(row.genome_json)["entry"]["conditions"]
        assert json.loads(row.signature_json)["triples"]


def test_a_scored_candidate_can_be_taken_apart(finished: tuple[Harness, object]) -> None:
    """Section 13.2 step 2: every component and penalty is stored, so a ranking
    can be audited after the fact."""
    harness, _ = finished
    scored = [
        row
        for row in harness.store.candidates_for(EVOLUTION_ID)
        if row.fitness is not None and row.fitness > FITNESS_REJECTED
    ]
    assert scored
    for row in scored:
        components = json.loads(row.components_json)
        penalties = json.loads(row.penalties_json)
        assert len(components) == 10
        assert len(penalties) == 7
        assert row.base_score is not None
        assert row.fitness == pytest.approx(row.base_score * row.penalty_product)


def test_the_inner_walk_forward_earns_the_generalisation_component(tmp_path: Path) -> None:
    """Section 13.6, wired through the loop.

    Without an inner walk-forward, ``inner_oos`` scores 0 for every candidate —
    honest and uniform, but it means the component that rewards generalisation
    contributes nothing. With one, candidates that held up out of sample score
    above those that did not, and none of it costs a look at the validation
    segment.
    """
    from tests.unit.conftest_evolution import InProcessEvaluator

    from quantlab.core.config import InnerWalkForwardSettings
    from quantlab.evolution.inner import inner_report, inner_windows

    harness = build_harness(tmp_path, small_settings(max_generations=2))
    harness.open_run("ev_inner")
    windows = inner_windows(
        harness.bars.n_bars, InnerWalkForwardSettings(n_folds=3, embargo_bars=12)
    )

    def inner_for(genome, params):  # type: ignore[no-untyped-def]
        from quantlab.evolution.compiler import compile_genome

        evaluator = InProcessEvaluator(compile_genome(genome))
        return inner_report(evaluator.evaluate, harness.bars, params, harness.config, windows)

    result = evolve(**harness.kwargs("ev_inner", inner_for=inner_for))

    components = [
        candidate.inner_oos
        for generation in result.generations
        for candidate in generation.scored
        if candidate.fitness > FITNESS_REJECTED
    ]
    assert components, "no candidate cleared the gates, so nothing was measured"
    assert any(value > 0.0 for value in components), (
        "an inner walk-forward that scored every candidate at zero measured nothing"
    )

    # Still train-only: the folds are inside train by construction (INV-9).
    assert {row.segment for row in harness.store.query_runs()} == {"train"}


def test_the_run_reports_why_it_stopped(finished: tuple[Harness, object]) -> None:
    _harness, result = finished
    assert result.stop_reason == "max_generations"  # type: ignore[attr-defined]


def test_the_best_candidate_is_the_fittest_recorded(finished: tuple[Harness, object]) -> None:
    harness, result = finished
    best = result.best  # type: ignore[attr-defined]
    if best is None:
        pytest.skip("this fixture produced no candidate that cleared the gates")
    stored = max(
        (row.fitness for row in harness.store.candidates_for(EVOLUTION_ID) if row.fitness),
        default=None,
    )
    assert stored == pytest.approx(best.fitness)


def test_the_generated_sources_are_on_disk_under_their_own_hashes(
    finished: tuple[Harness, object],
) -> None:
    """Generated code goes through the same loader a human strategy does, so the
    bytes a run executed survive the process that produced them (INV-7)."""
    harness, _ = finished
    for row in harness.store.candidates_for(EVOLUTION_ID)[:5]:
        assert harness.loader.verify(row.strategy_id)
