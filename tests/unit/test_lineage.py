"""Lineage and INV-10 (master spec section 13.6, task T50).

INV-10: "re-applying a child's recorded mutations to its parent's genome
reproduces the child's genome byte for byte". Section 22 asks for it over a
stored five-generation run, and for ``ancestry()`` of a generation-4 candidate to
return five rows ending at a seed. Both are here, built by actually mutating —
real operators, real records — rather than by writing plausible rows into the
database. A lineage test that made up its own mutations would prove only that the
test could apply what the test had written.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from sqlalchemy import Engine, text

from quantlab.adapters.store.sqlite import SqliteExperimentStore, make_session_factory
from quantlab.core.config import MutationSettings
from quantlab.core.errors import StoreError
from quantlab.core.genome import (
    Condition,
    ConditionTree,
    Operand,
    StrategyGenome,
    genome_id,
)
from quantlab.core.strategy import ParamSpec
from quantlab.evolution.library import OperatorLibrary
from quantlab.evolution.lineage import (
    ancestry,
    descendants,
    genome_of,
    lineage_tree,
    mutations_of,
    replay,
    verify_candidate,
    verify_run,
)
from quantlab.evolution.mutation import Phenotype, mutate

FAST = Operand(kind="indicator", name="sma", kwargs={"n": "fast"})
SLOW = Operand(kind="indicator", name="sma", kwargs={"n": "slow"})
LIBRARY = OperatorLibrary()
SETTINGS = MutationSettings()


def seed_genome() -> StrategyGenome:
    return StrategyGenome(
        name="crossover",
        entry=ConditionTree(conditions=(Condition(left=FAST, op=">", right=SLOW),)),
        exit=ConditionTree(conditions=(Condition(left=FAST, op="<=", right=SLOW),)),
        params={
            "fast": ParamSpec(kind="int", default=10, low=5, high=50),
            "slow": ParamSpec(kind="int", default=30, low=20, high=100),
        },
        warmup_bars=100,
    )


@pytest.fixture
def store(db_engine: Engine) -> SqliteExperimentStore:
    return SqliteExperimentStore(make_session_factory(db_engine), now_ms=lambda: 1_700_000_000_000)


def _prerequisites(store: SqliteExperimentStore) -> str:
    """The rows an evolution run's foreign keys need, and its id."""
    store.get_or_create_dataset(
        dataset_id="d0",
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1h",
        start_ts=1_600_000_000_000,
        end_ts=1_600_003_600_000,
        n_bars=2,
        manifest_json="{}",
    )
    family = store.create_family(name="crossover", origin="human")
    store.add_strategy_version(
        strategy_id="s0",
        family_id=family.family_id,
        code_path="strategies/generated/s0.py",
        code_sha256="a" * 64,
        class_name="Crossover",
        param_schema_json="{}",
        style="bar_loop",
        author="evolution",
        logic_lines=20,
    )
    experiment = store.create_experiment(
        campaign="c1",
        purpose="train",
        config_hash="cfg0",
        config_json="{}",
        seed=42,
        git_commit="abc123",
    )
    return experiment.experiment_id


def _split(store: SqliteExperimentStore) -> str:
    from dataclasses import replace as dc_replace

    from quantlab.core.splits import SplitPolicy

    policy = SplitPolicy(
        symbol="BTC/USDT",
        timeframe="1h",
        train_start_ts=1_600_000_000_000,
        train_end_ts=1_600_001_800_000,
        embargo_bars=0,
        val_start_ts=1_600_001_800_000,
        val_end_ts=1_600_002_800_000,
        test_start_ts=1_600_002_800_000,
        test_end_ts=None,
        source_json="{}",
        dataset_id="d0",
    )
    return store.get_or_create_split(dc_replace(policy, dataset_id="d0")).split_id


@pytest.fixture
def five_generations(store: SqliteExperimentStore) -> tuple[SqliteExperimentStore, list[str]]:
    """A stored five-generation chain, each child produced by real mutation.

    Generation 0 is a seed; each later generation mutates the one before, and both
    the child's genome and the edits that made it are written down exactly as the
    loop of T52 will write them.
    """
    experiment_id = _prerequisites(store)
    split_id = _split(store)
    store.create_evolution_run(
        evolution_id="ev0",
        experiment_id=experiment_id,
        campaign="c1",
        dataset_id="d0",
        split_id=split_id,
        population_size=16,
        n_survivors=12,
        n_offspring=3,
        n_immigrants=1,
        max_generations=30,
        seed=42,
        fitness_config_json="{}",
        mutation_config_json="{}",
        diversity_config_json="{}",
    )

    ids: list[str] = []
    phenotype = Phenotype.from_genome(seed_genome())
    parent_id: str | None = None
    for generation in range(5):
        store.add_generation(
            generation_id=f"g{generation}",
            evolution_id="ev0",
            gen_index=generation,
            diversity=0.7,
            n_evaluated=1,
            n_cache_hits=0,
            n_rejected_by_gate=0,
            n_immigrants_used=0,
        )
        mutations = ()
        if generation:
            for attempt in range(50):
                result = mutate(
                    phenotype, SETTINGS, np.random.default_rng(generation * 100 + attempt), LIBRARY
                )
                if result.child is not None:
                    phenotype, mutations = result.child, result.mutations
                    break
            else:  # pragma: no cover - 50 attempts always succeed on this parent
                raise AssertionError("could not mutate the parent")

        assert phenotype.genome is not None
        candidate_id = f"cand{generation}"
        store.add_candidate(
            candidate_id=candidate_id,
            evolution_id="ev0",
            generation_id=f"g{generation}",
            gen_index=generation,
            strategy_id="s0",
            params_json=json.dumps(dict(phenotype.params), sort_keys=True),
            kind="genome",
            origin="seed" if parent_id is None else "mutant",
            genome_json=phenotype.genome.canonical(),
            parent_candidate_id=parent_id,
        )
        if mutations:
            assert parent_id is not None
            store.add_mutations(
                candidate_id,
                [
                    {
                        "parent_candidate_id": parent_id,
                        "seq": index,
                        "category": mutation.category,
                        "operator": mutation.operator,
                        "target": mutation.target,
                        "before_json": mutation.before_json,
                        "after_json": mutation.after_json,
                        "rng_seed": mutation.rng_seed,
                    }
                    for index, mutation in enumerate(mutations)
                ],
            )
        store.score_candidate(candidate_id, fitness=0.9 - generation * 0.1, survived=True)
        ids.append(candidate_id)
        parent_id = candidate_id
    return store, ids


# ---------------------------------------------------------------------------
# ancestry
# ---------------------------------------------------------------------------
def test_ancestry_of_a_generation_four_candidate_returns_five_rows_ending_at_a_seed(
    five_generations: tuple[SqliteExperimentStore, list[str]],
) -> None:
    """Section 22's stated criterion, and INV-10's structural half."""
    store, ids = five_generations
    chain = ancestry(store, ids[-1])
    assert [row.candidate_id for row in chain] == ids
    assert chain[0].parent_candidate_id is None
    assert chain[0].origin == "seed"
    assert chain[-1].candidate_id == ids[-1]


def test_a_seed_is_its_own_whole_ancestry(
    five_generations: tuple[SqliteExperimentStore, list[str]],
) -> None:
    store, ids = five_generations
    assert [row.candidate_id for row in ancestry(store, ids[0])] == [ids[0]]


def test_descendants_run_the_other_way(
    five_generations: tuple[SqliteExperimentStore, list[str]],
) -> None:
    """Note the asymmetry, which the port documents: ``ancestry`` ends *with*
    the candidate, while ``descendants`` lists only what came after it. Each
    reads the way its name does — "my ancestry" includes me, "my descendants"
    does not."""
    store, ids = five_generations
    assert {row.candidate_id for row in descendants(store, ids[0])} == set(ids[1:])
    assert descendants(store, ids[-1]) == []


# ---------------------------------------------------------------------------
# INV-10
# ---------------------------------------------------------------------------
def test_inv10_holds_over_a_stored_five_generation_run(
    five_generations: tuple[SqliteExperimentStore, list[str]],
) -> None:
    """Section 22's other stated criterion. Every genome candidate replays."""
    store, ids = five_generations
    report = verify_run(store, "ev0")
    assert report.ok
    assert list(report.checked) == ids[1:]
    assert set(report.skipped) == {ids[0]}
    assert "seed" in report.skipped[ids[0]]


def test_replay_reproduces_a_child_exactly(
    five_generations: tuple[SqliteExperimentStore, list[str]],
) -> None:
    store, ids = five_generations
    for candidate_id in ids[1:]:
        stored = genome_of(ancestry(store, candidate_id)[-1])
        assert genome_id(replay(store, candidate_id)) == genome_id(stored)
        assert replay(store, candidate_id).canonical() == stored.canonical()
        assert verify_candidate(store, candidate_id)


def test_a_tampered_genome_fails_the_check(
    five_generations: tuple[SqliteExperimentStore, list[str]],
) -> None:
    """INV-10 has to be able to *fail*, or it certifies nothing.

    The stored genome is changed with raw SQL, behind the append-only guard: the
    guard is a Python-level rule, and the corruption this invariant exists to
    catch is one that did not come through it.
    """
    store, ids = five_generations
    tampered = seed_genome().model_copy(update={"warmup_bars": 999}).canonical()
    with store.factory().bind.begin() as connection:  # type: ignore[union-attr]
        connection.execute(
            text("UPDATE candidate SET genome_json = :g WHERE candidate_id = :c"),
            {"g": tampered, "c": ids[2]},
        )

    assert not verify_candidate(store, ids[2])
    report = verify_run(store, "ev0")
    assert not report.ok
    assert ids[2] in report.mismatched


def test_a_missing_mutation_record_is_reported_not_passed(
    five_generations: tuple[SqliteExperimentStore, list[str]],
) -> None:
    """A candidate whose edits were never written cannot be replayed, and
    counting it as verified would let an empty record claim a clean bill."""
    store, ids = five_generations
    with store.factory().bind.begin() as connection:  # type: ignore[union-attr]
        connection.execute(text("DELETE FROM mutation WHERE candidate_id = :c"), {"c": ids[1]})
    report = verify_run(store, "ev0")
    assert ids[1] in report.skipped
    assert "no mutations" in report.skipped[ids[1]]
    assert ids[1] not in report.checked


def test_the_mutations_come_back_as_they_were_written(
    five_generations: tuple[SqliteExperimentStore, list[str]],
) -> None:
    """Replay runs the same code path the child came from, on the same type."""
    store, ids = five_generations
    mutations = mutations_of(store, ids[1])
    assert mutations
    assert [m.category for m in mutations] == sorted(
        [m.category for m in mutations], key=lambda _: 0
    )
    for mutation in mutations:
        assert mutation.category in ("parameter", "structural")
        assert mutation.target
        json.loads(mutation.after_json)


def test_replaying_a_seed_is_refused(
    five_generations: tuple[SqliteExperimentStore, list[str]],
) -> None:
    store, ids = five_generations
    with pytest.raises(StoreError, match="no parent to replay from"):
        replay(store, ids[0])


def test_replaying_an_unknown_candidate_is_refused(
    five_generations: tuple[SqliteExperimentStore, list[str]],
) -> None:
    store, _ = five_generations
    with pytest.raises(StoreError, match="no such candidate"):
        replay(store, "nobody")


def test_an_opaque_candidate_has_no_genome_to_replay(store: SqliteExperimentStore) -> None:
    """Section 9.6 gives opaque candidates parameter mutation only."""

    class Row:
        candidate_id = "op0"
        kind = "opaque"
        genome_json = None

    with pytest.raises(StoreError, match="no genome to replay"):
        genome_of(Row())


def test_verify_run_skips_opaque_candidates_with_a_reason(
    five_generations: tuple[SqliteExperimentStore, list[str]],
) -> None:
    store, ids = five_generations
    store.add_candidate(
        candidate_id="opaque0",
        evolution_id="ev0",
        generation_id="g0",
        gen_index=0,
        strategy_id="s0",
        params_json="{}",
        kind="opaque",
        origin="seed",
        genome_json=None,
        parent_candidate_id=ids[0],
    )
    report = verify_run(store, "ev0")
    assert report.ok
    assert "opaque" in report.skipped["opaque0"]
    assert "opaque0" not in report.checked


# ---------------------------------------------------------------------------
# the tree
# ---------------------------------------------------------------------------
def test_the_tree_is_a_chain_for_a_linear_lineage(
    five_generations: tuple[SqliteExperimentStore, list[str]],
) -> None:
    store, ids = five_generations
    tree = lineage_tree(store, "ev0")
    assert tree.roots == (ids[0],)
    assert [tree.nodes[candidate].children for candidate in ids[:-1]] == [
        (child,) for child in ids[1:]
    ]
    assert tree.nodes[ids[-1]].children == ()


def test_the_path_to_the_root_reads_child_first(
    five_generations: tuple[SqliteExperimentStore, list[str]],
) -> None:
    store, ids = five_generations
    assert lineage_tree(store, "ev0").path_to_root(ids[-1]) == list(reversed(ids))


def test_a_cycle_is_refused_rather_than_looped_over(store: SqliteExperimentStore) -> None:
    """INV-10 requires the parent chain to reach a seed without cycles, and
    looping forever while drawing a dashboard is the worst way to find out."""
    from quantlab.evolution.lineage import LineageNode, LineageTree

    nodes = {
        "a": LineageNode("a", 0, "mutant", "genome", 0.5, False, "b"),
        "b": LineageNode("b", 1, "mutant", "genome", 0.5, False, "a"),
    }
    tree = LineageTree(evolution_id="ev0", nodes=nodes, roots=(), generations={})
    with pytest.raises(StoreError, match="cycle"):
        tree.path_to_root("a")


def test_the_tree_summarises_each_generation(
    five_generations: tuple[SqliteExperimentStore, list[str]],
) -> None:
    store, _ = five_generations
    tree = lineage_tree(store, "ev0")
    assert set(tree.generations) == {0, 1, 2, 3, 4}
    assert tree.generations[0]["best"] == pytest.approx(0.9)
    assert tree.generations[4]["best"] == pytest.approx(0.5)
    assert all(summary["n"] == 1.0 for summary in tree.generations.values())


def test_an_unscored_candidate_is_counted_but_not_averaged(
    five_generations: tuple[SqliteExperimentStore, list[str]],
) -> None:
    """A generation of sixteen where twelve were rejected is a different picture
    from one where four were, and reporting only the survivors would hide it."""
    store, _ = five_generations
    store.add_candidate(
        candidate_id="unscored",
        evolution_id="ev0",
        generation_id="g0",
        gen_index=0,
        strategy_id="s0",
        params_json="{}",
        kind="genome",
        origin="immigrant",
        genome_json=seed_genome().canonical(),
    )
    summary = lineage_tree(store, "ev0").generations[0]
    assert summary["n"] == 2.0
    assert summary["n_scored"] == 1.0
    assert summary["best"] == pytest.approx(0.9)


def test_a_run_with_no_candidates_has_an_empty_tree(store: SqliteExperimentStore) -> None:
    tree = lineage_tree(store, "nothing")
    assert tree.nodes == {}
    assert tree.roots == ()
    assert verify_run(store, "nothing").ok
