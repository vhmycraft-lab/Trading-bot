"""Registering a genome as a strategy version (master spec sections 9.6, 13).

This is the ``kind='genome'`` path of section 9.6: a candidate carrying a
structure, compiled once and admitted through the same loader as a human-written
strategy. The tests here are about the *seam* — that generated code takes no
shortcut past the AST check or the source store, that the two identities stay
distinct, and that re-registering a survivor is free.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import Engine

from quantlab.adapters.store.artifacts import FileSourceStore
from quantlab.adapters.store.sqlite import SqliteExperimentStore, make_session_factory
from quantlab.core.errors import StrategySafetyError
from quantlab.core.genome import Condition, ConditionTree, Operand, StrategyGenome, genome_id
from quantlab.core.hashing import strategy_id
from quantlab.core.strategy import ParamSpec
from quantlab.evolution.candidates import register_genome
from quantlab.evolution.compiler import compile_genome
from quantlab.strategies_io.loader import StrategyLoader


@pytest.fixture
def sources(tmp_path: Path) -> FileSourceStore:
    return FileSourceStore(tmp_path / "strategies" / "generated")


@pytest.fixture
def loader(db_engine: Engine, sources: FileSourceStore) -> StrategyLoader:
    store = SqliteExperimentStore(make_session_factory(db_engine), now_ms=lambda: 1_700_000_000_000)
    return StrategyLoader(store, sources)


def crossover(**overrides: object) -> StrategyGenome:
    fast = Operand(kind="indicator", name="sma", kwargs={"n": "fast"})
    slow = Operand(kind="indicator", name="sma", kwargs={"n": "slow"})
    fields: dict[str, object] = {
        "name": "crossover",
        "entry": ConditionTree(conditions=(Condition(left=fast, op=">", right=slow),)),
        "exit": ConditionTree(conditions=(Condition(left=fast, op="<=", right=slow),)),
        "params": {
            "fast": ParamSpec(kind="int", default=10, low=5, high=50),
            "slow": ParamSpec(kind="int", default=30, low=20, high=100),
        },
        "warmup_bars": 100,
    }
    fields.update(overrides)
    return StrategyGenome(**fields)  # type: ignore[arg-type]


def test_a_genome_becomes_a_registered_strategy_version(loader: StrategyLoader) -> None:
    candidate = register_genome(loader, crossover())

    assert candidate.genome_id == genome_id(crossover())
    assert candidate.strategy_id == strategy_id(compile_genome(crossover()))
    assert candidate.strategy.style == "bar_loop"
    assert candidate.strategy.class_name == "Crossover"
    assert candidate.strategy.warmup_bars == 100
    assert sorted(candidate.strategy.param_schema) == ["fast", "slow"]


def test_the_two_identities_are_kept_apart(loader: StrategyLoader) -> None:
    """The structure's id and the bytes' id answer different questions (§9.6)."""
    candidate = register_genome(loader, crossover())
    assert candidate.genome_id != candidate.strategy_id
    assert json.loads(candidate.genome_json)["name"] == "crossover"


def test_the_compiled_source_is_stored_and_verifiable(loader: StrategyLoader) -> None:
    """A run cites the version; the bytes it ran must be recoverable from it."""
    candidate = register_genome(loader, crossover())
    assert loader.verify(candidate.strategy_id) == compile_genome(crossover())


def test_registering_a_survivor_again_is_free(loader: StrategyLoader) -> None:
    """Compilation is deterministic, so carrying a candidate into the next
    generation reuses the same row rather than creating a competing one."""
    first = register_genome(loader, crossover())
    second = register_genome(loader, crossover())
    assert first.strategy_id == second.strategy_id
    assert first.strategy.code_path == second.strategy.code_path


def test_a_different_structure_is_a_different_version(loader: StrategyLoader) -> None:
    mutated = crossover(
        params={
            "fast": ParamSpec(kind="int", default=12, low=5, high=50),
            "slow": ParamSpec(kind="int", default=30, low=20, high=100),
        }
    )
    assert register_genome(loader, crossover()).strategy_id != (
        register_genome(loader, mutated).strategy_id
    )


def test_generated_code_still_faces_the_section_9_2_check(
    loader: StrategyLoader, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defence in depth (INV-4): the compiler's claim to produce valid strategies
    is not taken on trust. If it ever emitted something unsafe, the loader refuses
    it and nothing is registered."""
    monkeypatch.setattr(
        "quantlab.evolution.candidates.compile_genome",
        lambda genome: "import os\n\n\nclass Bad:\n    pass\n\n\nSTRATEGY = Bad\n",
    )
    with pytest.raises(StrategySafetyError):
        register_genome(loader, crossover())
    assert loader.store.query_runs() == []


def test_the_family_defaults_to_the_genome_name(loader: StrategyLoader) -> None:
    candidate = register_genome(loader, crossover())
    named = register_genome(loader, crossover(name="other"), family="campaign_a")
    assert candidate.strategy.family_id != named.strategy.family_id


def test_an_llm_proposed_genome_is_recorded_as_such(loader: StrategyLoader) -> None:
    """Section 12: what a language model wrote must be visible in the record."""
    candidate = register_genome(loader, crossover(), origin="llm", author="glm-4")
    row = loader.store.get_strategy_version(candidate.strategy_id)
    assert row is not None
    assert row.author == "glm-4"


def test_an_origin_section_6_does_not_allow_is_refused_before_any_write(
    loader: StrategyLoader,
) -> None:
    with pytest.raises(ValueError, match="family origin must be one of"):
        register_genome(loader, crossover(), origin="generated")
    assert loader.store.get_strategy_version(strategy_id(compile_genome(crossover()))) is None
