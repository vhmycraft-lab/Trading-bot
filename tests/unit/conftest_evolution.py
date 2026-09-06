"""A real, offline evolution run for the tests of section 13.2 (task T52).

Shared by the loop's unit tests and by the integration tests of section 20. The
engine, store, loader and runner are the real ones; only the *sandbox* is
replaced, by an in-process evaluator. That is a deliberate and bounded exception:
every candidate here is compiled by this repository's own genome compiler from a
genome the test wrote, so there is no untrusted code to isolate, and a child
process per candidate per generation would make a five-generation test cost
minutes rather than seconds. The sandbox itself is tested in
``tests/unit/test_sandbox.py`` and exercised end to end by the CLI integration
tests.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tests.helpers import make_bars, zero_cost_config

from quantlab.adapters.engine.simple_bar import SimpleBarEngine
from quantlab.adapters.store.artifacts import FileArtifactStore, FileSourceStore
from quantlab.adapters.store.sqlite import (
    SqliteExperimentStore,
    create_db_engine,
    make_session_factory,
    upgrade_to_head,
)
from quantlab.core.config import DiversitySettings, EvolutionSettings
from quantlab.core.genome import StrategyGenome
from quantlab.core.splits import SplitPolicy
from quantlab.core.types import BacktestConfig, BarFrame
from quantlab.evolution.compiler import compile_genome
from quantlab.evolution.library import OperatorLibrary
from quantlab.experiments.runner import ExperimentRunner
from quantlab.strategies_io.loader import StrategyLoader

EPOCH = 1_600_000_000_000
HOUR = 3_600_000


class InProcessEvaluator:
    """Runs a compiled genome in this process. See the module docstring."""

    engine_name = SimpleBarEngine.name
    engine_version = SimpleBarEngine.version

    def __init__(self, source: str) -> None:
        namespace: dict[str, Any] = {}
        exec(compile(source, "generated.py", "exec"), namespace)
        self.strategy_class = namespace["STRATEGY"]

    def evaluate(self, bars: BarFrame, params: Any, config: BacktestConfig) -> Any:
        return SimpleBarEngine().run(self.strategy_class(), bars, params, config)


@dataclass(frozen=True, slots=True)
class Harness:
    """Everything ``evolve`` needs, wired against a temporary database."""

    store: SqliteExperimentStore
    runner: ExperimentRunner
    loader: StrategyLoader
    bars: BarFrame
    experiment_id: str
    dataset_id: str
    split_id: str
    settings: EvolutionSettings
    config: BacktestConfig
    library: OperatorLibrary

    @property
    def register(self) -> Callable[[StrategyGenome], tuple[str, str]]:
        def _register(genome: StrategyGenome) -> tuple[str, str]:
            source = compile_genome(genome)
            loaded = self.loader.load_source(source, family=genome.name, author="evolution")
            return loaded.strategy_id, source

        return _register

    def open_run(self, evolution_id: str) -> None:
        """Open the ``evolution_run`` row the loop's foreign keys need."""
        self.store.create_evolution_run(
            evolution_id=evolution_id,
            experiment_id=self.experiment_id,
            campaign="tests",
            dataset_id=self.dataset_id,
            split_id=self.split_id,
            population_size=self.settings.population_size,
            n_survivors=self.settings.n_survivors,
            n_offspring=self.settings.n_offspring,
            n_immigrants=self.settings.n_immigrants,
            max_generations=self.settings.max_generations,
            seed=self.settings.seed,
            fitness_config_json="{}",
            mutation_config_json="{}",
            diversity_config_json="{}",
        )

    def kwargs(self, evolution_id: str, **overrides: Any) -> dict[str, Any]:
        """The argument set ``evolve`` takes, with fields replaced per test."""
        base: dict[str, Any] = {
            "evolution_id": evolution_id,
            "store": self.store,
            "runner": self.runner,
            "register": self.register,
            "evaluator_for": InProcessEvaluator,
            "bars_train": self.bars,
            "experiment_id": self.experiment_id,
            "dataset_id": self.dataset_id,
            "split_id": self.split_id,
            "settings": self.settings,
            "config": self.config,
            "library": self.library,
            "segment": "train",
        }
        base.update(overrides)
        return base


def small_settings(**overrides: Any) -> EvolutionSettings:
    """A population of eight: large enough to niche, small enough to be quick."""
    base: dict[str, Any] = {
        "population_size": 8,
        "n_survivors": 5,
        "n_offspring": 2,
        "n_immigrants": 1,
        "max_generations": 5,
        "seed": 42,
        "diversity": DiversitySettings(max_immigrants=3),
    }
    base.update(overrides)
    return EvolutionSettings(**base)


def build_harness(tmp_path: Path, settings: EvolutionSettings | None = None) -> Harness:
    """A migrated database, ingested bars and every prerequisite row."""
    engine = create_db_engine(tmp_path / "quantlab.db")
    upgrade_to_head(engine)
    store = SqliteExperimentStore(make_session_factory(engine))

    store.get_or_create_dataset(
        dataset_id="d0",
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1h",
        start_ts=EPOCH,
        end_ts=EPOCH + 599 * HOUR,
        n_bars=600,
        manifest_json="{}",
    )
    split = store.get_or_create_split(
        SplitPolicy(
            symbol="BTC/USDT",
            timeframe="1h",
            train_start_ts=EPOCH,
            train_end_ts=EPOCH + 400 * HOUR,
            embargo_bars=0,
            val_start_ts=EPOCH + 401 * HOUR,
            val_end_ts=EPOCH + 500 * HOUR,
            test_start_ts=EPOCH + 501 * HOUR,
            test_end_ts=None,
            source_json="{}",
            dataset_id="d0",
        )
    )
    experiment = store.create_experiment(
        campaign="tests",
        purpose="train",
        config_hash="cfg0",
        config_json="{}",
        seed=42,
        git_commit="abc123",
    )
    resolved = settings or small_settings()
    return Harness(
        store=store,
        runner=ExperimentRunner(store, FileArtifactStore(tmp_path / "artifacts")),
        loader=StrategyLoader(store, FileSourceStore(tmp_path / "generated")),
        bars=BarFrame(
            make_bars(600, start_ts=EPOCH, seed=5, drift=0.05),
            symbol="BTC/USDT",
            timeframe="1h",
        ),
        experiment_id=experiment.experiment_id,
        dataset_id="d0",
        split_id=split.split_id,
        settings=resolved,
        config=zero_cost_config(bars_per_year=8760),
        library=OperatorLibrary(limits=resolved.genome),
    )
