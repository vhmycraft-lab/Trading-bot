"""``quantlab evolve`` — the evolutionary search (master spec sections 13.2, 13.9).

    quantlab evolve run     --campaign my_campaign [--generations N] [--seed N]
    quantlab evolve resume  <evolution_id>
    quantlab evolve status  <evolution_id>
    quantlab evolve lineage <candidate_id> [--tree]

The command that actually spends the budget. Everything it evaluates is on the
**train** segment, and the loop asserts that for every run it creates (INV-9);
the validation segment is reachable only through ``evolve promote``, and the test
partition only through ``quantlab lockbox``.

``run`` is resumable: a crash mid-generation leaves the completed runs in the
store, and ``resume`` re-enters at the first generation with no ``generation``
row, re-using those runs from the cache of section 11.2 so no work is repeated.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from quantlab.adapters.engine.simple_bar import SimpleBarEngine
from quantlab.adapters.store.artifacts import FileArtifactStore, FileSourceStore
from quantlab.adapters.store.sqlite import SqliteExperimentStore, missing_tables
from quantlab.core.errors import ConfigError, StoreError
from quantlab.core.genome import StrategyGenome
from quantlab.core.hashing import short_id
from quantlab.core.types import BacktestConfig, SlippageConfig
from quantlab.evolution.compiler import compile_genome
from quantlab.evolution.library import OperatorLibrary
from quantlab.evolution.lineage import ancestry, lineage_tree, verify_run
from quantlab.evolution.loop import evolve as run_evolution
from quantlab.evolution.loop import resume_point
from quantlab.experiments.env import git_state
from quantlab.experiments.runner import ExperimentRunner
from quantlab.sandbox.runner import SandboxLimits, SandboxRunner
from quantlab.strategies_io.evaluators import SandboxEvaluator
from quantlab.strategies_io.loader import StrategyLoader

__all__ = ["app"]

app = typer.Typer(help="Evolutionary search over strategy candidates.", no_args_is_help=True)
console = Console()

ENGINE_MODULE = "quantlab.adapters.engine.simple_bar"
ENGINE_CLASS = "SimpleBarEngine"


def _require_migrated(container: Any) -> None:
    missing = missing_tables(container.db_engine)
    if missing:
        raise ConfigError(
            "the experiment database is not migrated; run `quantlab db upgrade`",
            missing_tables=list(missing),
        )


def _backtest_config(state: Any) -> BacktestConfig:
    settings = state.config.backtest
    return BacktestConfig(
        initial_equity=settings.initial_equity,
        fee_bps=settings.fee_bps,
        slippage=SlippageConfig.model_validate(settings.slippage.model_dump()),
        allow_short=settings.allow_short,
        short_borrow_bps_per_bar=settings.short_borrow_bps_per_bar,
        max_position_fraction=settings.max_position_fraction,
        fill_rule=settings.fill_rule,
        seed=state.config.evolution.seed,
    )


def _train_bars(container: Any) -> Any:
    """The train segment's bars — the only bars an evolution ever sees (INV-9)."""
    policy = container.split_policy
    if policy is None or container.market_data is None:
        raise ConfigError("a split policy and a market data source are both required")
    window = policy.segment("train")
    return policy, container.market_data.load(
        symbol=policy.symbol,
        timeframe=policy.timeframe,
        start_ts=window.start_ts,
        end_ts=window.end_ts,
    )


def _wiring(
    ctx: typer.Context,
) -> tuple[Any, SqliteExperimentStore, ExperimentRunner, StrategyLoader]:
    state = ctx.obj
    container, config = state.container, state.config
    _require_migrated(container)
    store = SqliteExperimentStore(container.session_factory)
    limits = SandboxLimits(
        cpu_seconds=config.sandbox.cpu_seconds,
        memory_mb=config.sandbox.memory_mb,
        wall_clock_s=config.sandbox.wall_clock_s,
        allowed_imports=config.sandbox.allowed_imports,
    )
    loader = StrategyLoader(
        store,
        FileSourceStore(Path("strategies") / FileSourceStore.DEFAULT_SUBDIR),
        allowed_imports=config.sandbox.allowed_imports,
        sandbox=SandboxRunner(limits=limits),
        engine_module=ENGINE_MODULE,
        engine_class=ENGINE_CLASS,
    )
    runner = ExperimentRunner(store, FileArtifactStore(config.project.artifacts_dir))
    return state, store, runner, loader


def _register(loader: StrategyLoader):  # type: ignore[no-untyped-def]
    """Compile a genome and admit it through the loader (INV-4).

    Generated code takes no shortcut past the AST check or the source store: the
    compiler makes valid strategies by construction, and defence in depth means
    not trusting that claim.
    """

    def register(genome: StrategyGenome) -> tuple[str, str]:
        source = compile_genome(genome)
        loaded = loader.load_source(source, family=genome.name, origin="human", author="evolution")
        return loaded.strategy_id, source

    return register


def _evaluator_for(source: str) -> SandboxEvaluator:
    """Every candidate runs in the sandbox — the platform's own included (§21.3)."""
    return SandboxEvaluator(
        source=source,
        engine_module=ENGINE_MODULE,
        engine_class=ENGINE_CLASS,
        engine_name=SimpleBarEngine.name,
        engine_version=SimpleBarEngine.version,
        runner=SandboxRunner(),
    )


@app.command("run")
def run(
    ctx: typer.Context,
    campaign: Annotated[str, typer.Option("--campaign", help="Campaign name.")] = "evolution",
    generations: Annotated[
        int, typer.Option("--generations", help="Override evolution.max_generations.")
    ] = 0,
    population: Annotated[
        int, typer.Option("--population", help="Override evolution.population_size.")
    ] = 0,
    seed: Annotated[int, typer.Option("--seed", help="Override evolution.seed.")] = -1,
) -> None:
    """Run an evolutionary search on the train segment."""
    state, store, runner, loader = _wiring(ctx)
    settings = state.config.evolution
    overrides: dict[str, Any] = {}
    if generations:
        overrides["max_generations"] = generations
    if population:
        overrides["population_size"] = population
    if seed >= 0:
        overrides["seed"] = seed
    if overrides:
        settings = settings.model_copy(update=overrides)

    policy, bars = _train_bars(state.container)
    config = _backtest_config(state)
    commit, _dirty = git_state()
    experiment = store.create_experiment(
        campaign=campaign,
        purpose="train",
        config_hash=config.config_hash(),
        config_json=json.dumps(config.model_dump(mode="json"), sort_keys=True),
        seed=settings.seed,
        git_commit=commit,
    )
    dataset_id = state.container.market_data.dataset_id(policy.symbol, policy.timeframe)
    stored_policy = store.get_or_create_split(replace(policy, dataset_id=dataset_id))
    evolution_id = short_id(f"{campaign}|{experiment.experiment_id}|{settings.seed}")

    store.create_evolution_run(
        evolution_id=evolution_id,
        experiment_id=experiment.experiment_id,
        campaign=campaign,
        dataset_id=dataset_id,
        split_id=stored_policy.split_id,
        population_size=settings.population_size,
        n_survivors=settings.n_survivors,
        n_offspring=settings.n_offspring,
        n_immigrants=settings.n_immigrants,
        max_generations=settings.max_generations,
        seed=settings.seed,
        fitness_config_json=settings.fitness.model_dump_json(),
        mutation_config_json=settings.mutation.model_dump_json(),
        diversity_config_json=settings.diversity.model_dump_json(),
    )
    result = run_evolution(
        evolution_id=evolution_id,
        store=store,
        runner=runner,
        register=_register(loader),
        evaluator_for=_evaluator_for,
        bars_train=bars,
        experiment_id=experiment.experiment_id,
        dataset_id=dataset_id,
        split_id=stored_policy.split_id,
        settings=settings,
        config=config,
        library=OperatorLibrary(limits=settings.genome),
        segment="train",
    )
    store.finish_evolution_run(evolution_id, status="finished", stop_reason=result.stop_reason)
    _report(result, store)


@app.command("resume")
def resume(
    ctx: typer.Context,
    evolution_id: Annotated[str, typer.Argument(help="The run to continue.")],
) -> None:
    """Continue a run at the first generation it has no record of.

    Runs already in the store are served from the cache of section 11.2, so an
    interrupted generation is re-evaluated without re-executing anything.
    """
    state, store, runner, loader = _wiring(ctx)
    row = _require_run(store, evolution_id)
    start = resume_point(store, evolution_id)
    if start >= row.max_generations:
        console.print(f"run {evolution_id} is already complete ({start} generations)")
        return

    _policy, bars = _train_bars(state.container)
    settings = state.config.evolution.model_copy(
        update={"population_size": row.population_size, "seed": row.seed}
    )
    result = run_evolution(
        evolution_id=evolution_id,
        store=store,
        runner=runner,
        register=_register(loader),
        evaluator_for=_evaluator_for,
        bars_train=bars,
        experiment_id=row.experiment_id,
        dataset_id=row.dataset_id,
        split_id=row.split_id,
        settings=settings,
        config=_backtest_config(state),
        library=OperatorLibrary(limits=settings.genome),
        segment="train",
        start_generation=start,
    )
    store.finish_evolution_run(evolution_id, status="finished", stop_reason=result.stop_reason)
    console.print(f"resumed at generation {start}")
    _report(result, store)


@app.command("status")
def status(
    ctx: typer.Context,
    evolution_id: Annotated[str, typer.Argument(help="The run to describe.")],
) -> None:
    """Show a run's generations, diversity trajectory and evaluation count."""
    _state, store, _runner, _loader = _wiring(ctx)
    row = _require_run(store, evolution_id)

    table = Table(title=f"evolution {evolution_id}")
    table.add_column("gen")
    table.add_column("best")
    table.add_column("median")
    table.add_column("diversity")
    table.add_column("rejected")
    table.add_column("cached")
    for generation in store.generations_for(evolution_id):
        table.add_row(
            str(generation.gen_index),
            "—" if generation.best_fitness is None else f"{generation.best_fitness:.4f}",
            "—" if generation.median_fitness is None else f"{generation.median_fitness:.4f}",
            f"{generation.diversity:.3f}",
            str(generation.n_rejected_by_gate),
            str(generation.n_cache_hits),
        )
    console.print(table)
    console.print(f"status         : {row.status}")
    console.print(f"stop_reason    : {row.stop_reason or '—'}")
    console.print(f"n_evaluations  : {row.n_evaluations}  [dim](charged into DSR's M, §14.4)[/dim]")

    report = verify_run(store, evolution_id)
    verdict = "holds" if report.ok else f"FAILS for {', '.join(report.mismatched)}"
    console.print(f"INV-10         : {verdict} over {report.n_checked} replayed candidate(s)")


@app.command("lineage")
def lineage(
    ctx: typer.Context,
    candidate_id: Annotated[str, typer.Argument(help="The candidate to trace.")],
    tree: Annotated[bool, typer.Option("--tree", help="Show the whole run's tree.")] = False,
) -> None:
    """Trace a candidate back to the seed it descends from."""
    _state, store, _runner, _loader = _wiring(ctx)
    chain = ancestry(store, candidate_id)
    if not chain:
        raise StoreError("no such candidate", candidate_id=candidate_id)

    table = Table(title=f"lineage of {candidate_id}")
    table.add_column("gen")
    table.add_column("candidate")
    table.add_column("origin")
    table.add_column("fitness")
    for row in chain:
        table.add_row(
            str(row.gen_index),
            row.candidate_id,
            row.origin,
            "—" if row.fitness is None else f"{row.fitness:.4f}",
        )
    console.print(table)

    if tree:
        whole = lineage_tree(store, chain[-1].evolution_id)
        for index, summary in sorted(whole.generations.items()):
            best = summary.get("best")
            console.print(
                f"gen {index:>2}: n={int(summary['n'])} scored={int(summary['n_scored'])} "
                f"best={'—' if best is None else f'{best:.4f}'}"
            )


def _require_run(store: SqliteExperimentStore, evolution_id: str) -> Any:
    row = store.find_evolution_run(evolution_id)
    if row is None:
        raise StoreError("no such evolution run", evolution_id=evolution_id)
    return row


def _report(result: Any, store: SqliteExperimentStore) -> None:
    table = Table(title=f"evolution {result.evolution_id}", show_header=False)
    table.add_row("generations", str(len(result.generations)))
    table.add_row("n_evaluations", str(result.n_evaluations))
    table.add_row("distinct runs", str(len(store.query_runs())))
    table.add_row("stop_reason", result.stop_reason)
    best = result.best
    table.add_row("best candidate", "—" if best is None else best.candidate_id)
    table.add_row("best fitness", "—" if best is None else f"{best.fitness:.4f}")
    console.print(table)
