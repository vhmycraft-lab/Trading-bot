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
from quantlab.core.environment import EnvironmentSelector, build_window_pool, sampling_report
from quantlab.core.errors import ConfigError, StoreError
from quantlab.core.genome import StrategyGenome
from quantlab.core.hashing import short_id
from quantlab.core.types import BacktestConfig, SlippageConfig
from quantlab.evolution.compiler import compile_genome
from quantlab.evolution.diversity import CandidateView, Signature
from quantlab.evolution.library import OperatorLibrary
from quantlab.evolution.lineage import ancestry, lineage_tree, verify_run
from quantlab.evolution.loop import evolve as run_evolution
from quantlab.evolution.loop import resume_point
from quantlab.evolution.population import ScoredCandidate
from quantlab.evolution.promotion import promote
from quantlab.experiments.env import git_state
from quantlab.experiments.environment import GenerationEnvironmentProvider
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
    store = container.store
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


def _environment_provider(
    state: Any,
    store: Any,
    *,
    evolution_id: str,
    policy: Any,
    bars: Any,
    config: Any,
    dataset_id: str,
) -> GenerationEnvironmentProvider | None:
    """Rome's per-generation environment service, or ``None`` if it is switched off.

    Built here because this is the only layer holding both a store adapter and
    market data (INV-8). The selector is constructed from the *split policy*, so
    the pool it enumerates is bounded by the training segment and cannot be
    widened by anything downstream (Project Rome section 8).

    The asset universe defaults to the split's own symbol. Rome section 15
    requires that an environment's assets actually existed in the period it
    covers, and inventing a universe for a single-symbol split would break that;
    ``environment.asset_universe`` is where a real multi-asset universe goes.
    """
    settings = state.config.environment
    if not settings.enabled:
        return None
    selector = EnvironmentSelector(
        build_window_pool(policy, settings),
        settings,
        universe=settings.asset_universe or (policy.symbol,),
        dataset_version=dataset_id,
        execution_model_version=f"{SimpleBarEngine.name}/{SimpleBarEngine.version}",
    )
    return GenerationEnvironmentProvider(
        selector=selector,
        store=store,
        evolution_id=evolution_id,
        bars_train=bars,
        base_config=config,
    )


def _report_sampling(state: Any, provider: GenerationEnvironmentProvider | None) -> None:
    """Say how evenly this run covered training history (Rome sections 18, 43).

    Printed at the end of every run rather than filed away, because a campaign
    that concentrated on one part of history should say so where the person who
    ran it will read it.
    """
    if provider is None:
        console.print("[dim]training-environment randomisation is disabled[/dim]")
        return
    report = sampling_report(provider.windows_used, provider.pool, state.config.environment)
    console.print(
        f"history exposure: {report.n_environments} environment(s), "
        f"{len(report.window_counts)} distinct window(s), "
        f"concentration {report.concentration:.2f}"
    )
    if report.flagged:
        console.print(
            "[yellow]sampling is uneven[/yellow]: "
            f"over-used buckets {list(report.overused_buckets)}, "
            f"never sampled {list(report.unsampled_buckets)}"
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
    provider = _environment_provider(
        state,
        store,
        evolution_id=evolution_id,
        policy=stored_policy,
        bars=bars,
        config=config,
        dataset_id=dataset_id,
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
        environment_for=provider,
        segment="train",
    )
    store.finish_evolution_run(evolution_id, status="finished", stop_reason=result.stop_reason)
    _report(result, store)
    _report_sampling(state, provider)


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

    policy, bars = _train_bars(state.container)
    settings = state.config.evolution.model_copy(
        update={"population_size": row.population_size, "seed": row.seed}
    )
    config = _backtest_config(state)
    # The provider replays generations that already have a recorded environment
    # and draws fresh ones only beyond the resume point, which is what keeps the
    # replayed portion candidate-for-candidate identical to the original run.
    provider = _environment_provider(
        state,
        store,
        evolution_id=evolution_id,
        policy=replace(policy, dataset_id=row.dataset_id),
        bars=bars,
        config=config,
        dataset_id=row.dataset_id,
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
        config=config,
        library=OperatorLibrary(limits=settings.genome),
        environment_for=provider,
        segment="train",
        start_generation=start,
    )
    store.finish_evolution_run(evolution_id, status="finished", stop_reason=result.stop_reason)
    console.print(f"resumed at generation {start}")
    _report(result, store)
    _report_sampling(state, provider)


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


@app.command("promote")
def promote_command(
    ctx: typer.Context,
    evolution_id: Annotated[str, typer.Argument(help="The run to promote from.")],
    n: Annotated[int, typer.Option("--n", help="Override promotion.n_promote.")] = 0,
    generation: Annotated[
        int, typer.Option("--generation", help="Promote from this generation; -1 for the last.")
    ] = -1,
) -> None:
    """Promote the best candidates to the validation segment (INV-9). 🔒

    The only route from evolution to validation. A ``candidate_promotion`` row is
    written for each promoted candidate **before** any validation run executes,
    and each promotion increments its family's ``validation_touches`` — the count
    section 14.1 freezes a family at twenty of, and section 14.4 charges into the
    deflated Sharpe ratio's ``M``.

    Promoting does not run the validation pipeline; it authorises it. Section
    14.1's pipeline is `quantlab validate`.
    """
    _state, store, _runner, _loader = _wiring(ctx)
    row = _require_run(store, evolution_id)
    settings = ctx.obj.config.evolution.promotion
    if n:
        settings = settings.model_copy(update={"n_promote": n})

    generations = store.generations_for(evolution_id)
    if not generations:
        raise StoreError("this run has no completed generation", evolution_id=evolution_id)
    gen_index = generations[-1].gen_index if generation < 0 else generation

    candidates = _scored_candidates(store, evolution_id, gen_index)
    promotions = promote(
        store,
        candidates,
        evolution_id=evolution_id,
        gen_index=gen_index,
        settings=settings,
        reason=f"top candidate of generation {gen_index}",
    )

    table = Table(title=f"promoted from {evolution_id} generation {gen_index}")
    table.add_column("candidate")
    table.add_column("promotion")
    table.add_column("family touches")
    for record in promotions:
        table.add_row(record.candidate_id, record.promotion_id, str(record.touches))
    console.print(table)
    if not promotions:
        console.print(
            "[yellow]nothing was promoted[/yellow]: no candidate reached "
            f"promotion.min_fitness ({settings.min_fitness})"
        )
    console.print("[dim]a validation run may now be created for these candidates only[/dim]")
    del row


def _scored_candidates(
    store: SqliteExperimentStore, evolution_id: str, gen_index: int
) -> list[ScoredCandidate]:
    """Rebuild one generation's candidates from the store, for ranking.

    Position series are not stored (only their digest, section 13.5), so the
    behavioural half of similarity is unavailable here and the promotion filter
    falls back to structure. That is the conservative direction: two candidates
    that look structurally alike are treated as alike, so the validation budget is
    spent on fewer, more clearly distinct strategies rather than more.
    """
    import json

    rebuilt: list[ScoredCandidate] = []
    for row in store.candidates_for(evolution_id, gen_index):
        if row.fitness is None:
            continue
        signature = Signature(
            triples=tuple(
                tuple(triple) for triple in json.loads(row.signature_json).get("triples", [])
            ),
            risk_controls=frozenset(json.loads(row.signature_json).get("risk_controls", [])),
        )
        rebuilt.append(
            ScoredCandidate(
                view=CandidateView(
                    candidate_id=row.candidate_id,
                    fitness=float(row.fitness),
                    signature=signature,
                    behaviour=row.behaviour_hash,
                ),
                inner_oos=float(json.loads(row.components_json).get("inner_oos", 0.0)),
                n_free_params=len(json.loads(row.params_json)),
            )
        )
    return rebuilt


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
