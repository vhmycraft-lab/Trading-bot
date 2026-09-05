"""``quantlab optimize`` — refine a promoted candidate's parameters (spec 13.8).

    quantlab optimize run --strategy strategies/baselines/sma_cross.py
    quantlab optimize run --strategy ... --segment train --trials 50

Refinement, not search. Evolution (section 13) is what produces candidates; this
command explores the parameters *inside* a region one of them found, on the train
segment only, and reports both the point optimum and the plateau choice that
section 13.8 requires validation to use instead.

Every trial is an ordinary cached run (section 11.2) and counts towards
``n_trials_accounted``, which section 14.4 charges into the deflated Sharpe
ratio's ``M``. That is why the study is recorded rather than merely printed: a
search whose trials went unrecorded would make every verdict that followed it too
generous.
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
from quantlab.adapters.store.sqlite import SqliteExperimentStore, missing_tables
from quantlab.core.errors import ConfigError
from quantlab.core.metrics import compute_metrics
from quantlab.core.strategy import ParamSpec
from quantlab.core.types import BacktestConfig, SlippageConfig
from quantlab.experiments.env import git_state
from quantlab.optimize.objectives import objective_value
from quantlab.optimize.study import require_searchable_segment, run_study, study_record
from quantlab.sandbox.runner import SandboxLimits, SandboxRunner
from quantlab.strategies_io.evaluators import SandboxEvaluator
from quantlab.strategies_io.loader import StrategyLoader

__all__ = ["app"]

app = typer.Typer(help="Refine a promoted candidate's parameters.", no_args_is_help=True)
console = Console()

ENGINE_MODULE = "quantlab.adapters.engine.simple_bar"
ENGINE_CLASS = "SimpleBarEngine"


def _services(
    ctx: typer.Context,
) -> tuple[SqliteExperimentStore, FileArtifactStore, StrategyLoader]:
    state = ctx.obj
    container, config = state.container, state.config
    missing = missing_tables(container.db_engine)
    if missing:
        raise ConfigError(
            "the experiment database is not migrated; run `quantlab db upgrade`",
            missing_tables=list(missing),
        )
    limits = SandboxLimits(
        cpu_seconds=config.sandbox.cpu_seconds,
        memory_mb=config.sandbox.memory_mb,
        wall_clock_s=config.sandbox.wall_clock_s,
        allowed_imports=config.sandbox.allowed_imports,
    )
    loader = StrategyLoader(
        SqliteExperimentStore(container.session_factory),
        FileSourceStore(Path("strategies") / FileSourceStore.DEFAULT_SUBDIR),
        allowed_imports=config.sandbox.allowed_imports,
        sandbox=SandboxRunner(limits=limits),
        engine_module=ENGINE_MODULE,
        engine_class=ENGINE_CLASS,
    )
    return (
        SqliteExperimentStore(container.session_factory),
        FileArtifactStore(config.project.artifacts_dir),
        loader,
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


@app.command("run")
def run(
    ctx: typer.Context,
    strategy: Annotated[Path, typer.Option("--strategy", help="Path to the strategy source.")],
    segment: Annotated[
        str, typer.Option("--segment", help="'train' or 'wf_is:<k>'. Never val or test.")
    ] = "train",
    trials: Annotated[int, typer.Option("--trials", help="Override optimize.n_trials.")] = 0,
) -> None:
    """Search a strategy's parameters on the train segment and pick the plateau."""
    state = ctx.obj
    container = state.container
    require_searchable_segment(segment)

    store, _artifacts, loader = _services(ctx)
    policy = container.split_policy
    if policy is None or container.market_data is None:
        raise ConfigError("a split policy and a market data source are both required")
    window = policy.segment(segment)
    bars = container.market_data.load(
        symbol=policy.symbol,
        timeframe=policy.timeframe,
        start_ts=window.start_ts,
        end_ts=window.end_ts,
    )

    loaded = loader.load_path(strategy, family=strategy.stem)
    schema = {name: ParamSpec.model_validate(spec) for name, spec in loaded.param_schema.items()}
    settings = _backtest_config(state)
    evaluator = SandboxEvaluator(
        source=loaded.source,
        engine_module=ENGINE_MODULE,
        engine_class=ENGINE_CLASS,
        engine_name=SimpleBarEngine.name,
        engine_version=SimpleBarEngine.version,
        runner=SandboxRunner(),
    )

    def evaluate(params: Any) -> float:
        result = evaluator.evaluate(bars, dict(params), settings)
        return objective_value(compute_metrics(result), state.config.optimize)

    outcome = run_study(
        evaluate,
        schema,
        state.config.optimize,
        segment=segment,
        strategy_id=loaded.strategy_id,
        n_trials=trials or None,
    )

    commit, _dirty = git_state()
    experiment = store.create_experiment(
        campaign="refinement",
        purpose="train",
        config_hash=settings.config_hash(),
        config_json=json.dumps(settings.model_dump(mode="json"), sort_keys=True),
        seed=settings.seed,
        git_commit=commit,
    )
    store.record_optuna_study(
        **study_record(
            outcome, experiment_id=experiment.experiment_id, strategy_id=loaded.strategy_id
        )
    )
    _report(outcome, strategy_id=loaded.strategy_id)


def _report(outcome: Any, *, strategy_id: str) -> None:
    """Show both answers, as section 13.8 requires."""
    table = Table(title=f"study {outcome.study_id}", show_header=False)
    table.add_row("strategy", strategy_id)
    table.add_row("segment", outcome.segment)
    table.add_row("objective", outcome.objective)
    table.add_row("trials", str(outcome.n_trials))
    if outcome.best is not None:
        table.add_row("point optimum", json.dumps(dict(outcome.best.params), sort_keys=True))
        table.add_row("optimum value", f"{outcome.best.value:.6f}")
    if outcome.plateau is not None:
        table.add_row("plateau params", json.dumps(dict(outcome.params), sort_keys=True))
        drop = outcome.plateau.plateau.median_drop
        table.add_row("median neighbour drop", "—" if drop is None else f"{drop:.4f}")
        table.add_row("selection moved", "yes" if outcome.plateau.moved else "no")
    console.print(table)
    console.print(
        "[dim]validation uses the plateau parameters, never the point optimum (spec 13.8)[/dim]"
    )
