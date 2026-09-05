"""``quantlab backtest`` — evaluate one strategy on one segment (spec section 11).

    quantlab backtest --strategy strategies/baselines/sma_cross.py --segment train
    quantlab backtest --strategy ... --segment train --force

Everything the run depends on is resolved here and then handed to the runner: the
dataset, the split policy, the segment's bars, the loaded strategy and the engine.
The runner decides whether it has seen this exact combination before (§11.2); this
command's job is to make sure the combination is the one the user asked for, and
that nothing reaches the strategy that the profile forbids it to see (INV-5).
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
from quantlab.core.errors import ConfigError
from quantlab.core.types import BacktestConfig, SlippageConfig, format_ts
from quantlab.experiments.env import git_state
from quantlab.experiments.runner import ExperimentRunner, RunOutcome
from quantlab.sandbox.runner import SandboxLimits, SandboxRunner
from quantlab.strategies_io.evaluators import SandboxEvaluator
from quantlab.strategies_io.loader import StrategyLoader

__all__ = ["app"]

app = typer.Typer(help="Run a backtest and record it.", no_args_is_help=True)
console = Console()

ENGINE_MODULE = "quantlab.adapters.engine.simple_bar"
ENGINE_CLASS = "SimpleBarEngine"

StrategyOption = Annotated[Path, typer.Option("--strategy", help="Path to the strategy source.")]
SegmentOption = Annotated[
    str, typer.Option("--segment", help="'train' | 'val' | 'test' | 'wf_is:<k>' | 'wf_oos:<k>'.")
]


def _require_migrated(container: Any) -> None:
    """Refuse to run against a database that is not at a known revision.

    Section 6 gives Alembic sole ownership of the schema, so a research command
    must not create tables on the way past. Without this the first write fails
    somewhere inside SQLAlchemy with a missing-table error that says nothing about
    what to do next.
    """
    missing = missing_tables(container.db_engine)
    if missing:
        raise ConfigError(
            "the experiment database is not migrated; run `quantlab db upgrade`",
            missing_tables=list(missing),
        )


def _services(
    ctx: typer.Context,
) -> tuple[SqliteExperimentStore, FileArtifactStore, StrategyLoader]:
    """Build the store, artifact tree and loader from the container's config."""
    state = ctx.obj
    container = state.container
    config = state.config
    _require_migrated(container)
    store = SqliteExperimentStore(container.session_factory)
    artifacts = FileArtifactStore(config.project.artifacts_dir)
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
    return store, artifacts, loader


def _register_dataset(
    store: SqliteExperimentStore, container: Any, exchange: str, policy: Any
) -> Any:
    """Record the dataset the run's bars came from.

    The *dataset*, not the segment: `dataset_id` is a hash of the whole manifest
    (section 7.2), so registering a segment's extent under it would describe the
    same id differently for every segment — which the store correctly refuses as a
    conflict. The extent comes from the source's ``available_range``, which is
    what the id was computed over.

    ``available_range`` reaches past the guard's boundary by design: it reports
    what is *stored*, never bars, so no partition is read (INV-5).
    """
    source = container.market_data
    extent = source.available_range(policy.symbol, policy.timeframe)
    if extent is None:
        raise ConfigError(
            "no bars are stored for this market; run `quantlab data pull` first",
            symbol=policy.symbol,
            timeframe=policy.timeframe,
        )
    first, last = extent
    bar_ms = policy.bar_ms
    return store.get_or_create_dataset(
        dataset_id=source.dataset_id(policy.symbol, policy.timeframe),
        exchange=exchange,
        symbol=policy.symbol,
        timeframe=policy.timeframe,
        start_ts=int(first),
        end_ts=int(last),
        n_bars=int((last - first) // bar_ms) + 1,
        manifest_json="{}",
    )


def _resolve_segment(container: Any, segment: str) -> tuple[Any, Any]:
    """The split policy and the bars for ``segment``.

    The bars come through the container's guarded market-data source, so the
    lockbox applies here exactly as it applies everywhere else: asking for the
    test segment outside the ``lockbox`` profile raises rather than returns
    (INV-5). This command never special-cases that.
    """
    policy = container.split_policy
    if policy is None:
        raise ConfigError(
            "no split policy is configured; `quantlab backtest` needs one to know "
            "which bars a segment refers to"
        )
    if container.market_data is None:  # pragma: no cover - every profile wires one
        raise ConfigError("no market data source is configured")
    window = policy.segment(segment)
    bars = container.market_data.load(
        symbol=policy.symbol,
        timeframe=policy.timeframe,
        start_ts=window.start_ts,
        end_ts=window.end_ts,
    )
    return policy, bars


@app.command("run")
def run(
    ctx: typer.Context,
    strategy: StrategyOption,
    segment: SegmentOption = "train",
    family: Annotated[str, typer.Option("--family", help="Strategy family name.")] = "",
    campaign: Annotated[str, typer.Option("--campaign", help="Campaign name.")] = "manual",
    params: Annotated[
        str, typer.Option("--params", help="JSON object of parameter overrides.")
    ] = "{}",
    force: Annotated[
        bool, typer.Option("--force", help="Re-execute a run that is already recorded.")
    ] = False,
) -> None:
    """Load a strategy, evaluate it on one segment, and record the run."""
    state = ctx.obj
    container = state.container
    store, artifacts, loader = _services(ctx)
    policy, bars = _resolve_segment(container, segment)

    loaded = loader.load_path(strategy, family=family or strategy.stem)
    backtest_config = state.config.backtest
    settings = BacktestConfig(
        initial_equity=backtest_config.initial_equity,
        fee_bps=backtest_config.fee_bps,
        slippage=SlippageConfig.model_validate(backtest_config.slippage.model_dump()),
        allow_short=backtest_config.allow_short,
        short_borrow_bps_per_bar=backtest_config.short_borrow_bps_per_bar,
        max_position_fraction=backtest_config.max_position_fraction,
        fill_rule=backtest_config.fill_rule,
        seed=state.config.evolution.seed,
    )
    dataset = _register_dataset(store, container, state.config.market.exchange, policy)
    stored_policy = store.get_or_create_split(replace(policy, dataset_id=dataset.dataset_id))
    commit, _dirty = git_state()
    experiment = store.create_experiment(
        campaign=campaign,
        purpose="train" if segment == "train" else "validate",
        config_hash=settings.config_hash(),
        config_json=json.dumps(settings.model_dump(mode="json"), sort_keys=True),
        seed=settings.seed,
        git_commit=commit,
    )
    evaluator = SandboxEvaluator(
        source=loaded.source,
        engine_module=ENGINE_MODULE,
        engine_class=ENGINE_CLASS,
        engine_name=SimpleBarEngine.name,
        engine_version=SimpleBarEngine.version,
        runner=SandboxRunner(),
    )
    outcome = ExperimentRunner(store, artifacts).run(
        evaluator,
        experiment_id=experiment.experiment_id,
        strategy_id=loaded.strategy_id,
        dataset_id=dataset.dataset_id,
        split_id=stored_policy.split_id,
        segment=segment,
        bars=bars,
        params=json.loads(params),
        config=settings,
        force=force,
    )
    _report(outcome, segment=segment, strategy_id=loaded.strategy_id, artifacts=artifacts)


def _report(
    outcome: RunOutcome, *, segment: str, strategy_id: str, artifacts: FileArtifactStore
) -> None:
    table = Table(title=f"run {outcome.run_id}", show_header=False)
    table.add_row("strategy", strategy_id)
    table.add_row("segment", segment)
    table.add_row("status", outcome.record.status)
    table.add_row("cache_hit", "true" if outcome.cache_hit else "false")
    table.add_row("artifacts", str(artifacts.dir_for(outcome.run_id)))
    if outcome.metrics is not None:
        table.add_row("net_return", f"{outcome.metrics.net_return:.4f}")
        table.add_row("n_trades", str(outcome.metrics.n_trades))
        table.add_row("max_drawdown", f"{outcome.metrics.max_drawdown:.4f}")
    if outcome.result is not None and outcome.result.equity.index.size:
        first = int(outcome.result.equity.index[0])
        last = int(outcome.result.equity.index[-1])
        table.add_row("range", f"{format_ts(first)} .. {format_ts(last)}")
    console.print(table)
