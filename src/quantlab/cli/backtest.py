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
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from quantlab.adapters.engine.simple_bar import SimpleBarEngine
from quantlab.adapters.store.artifacts import FileArtifactStore, FileSourceStore
from quantlab.adapters.store.sqlite import SqliteExperimentStore
from quantlab.core.errors import ConfigError
from quantlab.core.types import BacktestConfig, format_ts
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


def _services(
    ctx: typer.Context,
) -> tuple[SqliteExperimentStore, FileArtifactStore, StrategyLoader]:
    """Build the store, artifact tree and loader from the container's config."""
    container = ctx.obj
    config = container.config
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
    container = ctx.obj
    store, artifacts, loader = _services(ctx)
    policy, bars = _resolve_segment(container, segment)

    loaded = loader.load_path(strategy, family=family or strategy.stem)
    settings = BacktestConfig(
        fee_bps=container.config.costs.fee_bps,
        slippage=container.config.costs.slippage,
        seed=container.config.run.seed,
    )
    dataset = store.get_or_create_dataset(
        dataset_id=bars.dataset_id or policy.dataset_id,
        exchange=container.config.market.exchange,
        symbol=bars.symbol,
        timeframe=bars.timeframe,
        start_ts=int(bars.start_ts or 0),
        end_ts=int(bars.end_ts or 0),
        n_bars=bars.n_bars,
        manifest_json="{}",
    )
    stored_policy = store.get_or_create_split(policy)
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
