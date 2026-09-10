"""``quantlab report`` and ``quantlab reproduce`` (master spec sections 11.3, 11.4).

    quantlab report <run_id> [--out DIR]
    quantlab reproduce <run_id>

``report`` renders what was recorded. ``reproduce`` is the enforcement of INV-7:
it reloads a run's inputs from the store, executes it again in the sandbox, and
compares every metric the first run defined. A mismatch exits non-zero — the
platform's central claim is that a stored run can be rebuilt, and a command that
reported "close enough" would quietly retire that claim.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from quantlab.adapters.store.artifacts import FileArtifactStore, FileSourceStore
from quantlab.adapters.store.sqlite import SqliteExperimentStore, missing_tables
from quantlab.core.errors import ConfigError, StoreError
from quantlab.core.metrics import compute_metrics
from quantlab.core.types import BacktestConfig
from quantlab.reporting.markdown import RunReport, render_run_report
from quantlab.reporting.tearsheet import TEARSHEET_NAME, write_tearsheet
from quantlab.sandbox.runner import SandboxRunner
from quantlab.strategies_io.evaluators import SandboxEvaluator
from quantlab.strategies_io.loader import StrategyLoader

__all__ = ["REPORT_NAME", "app", "compare_metrics"]

app = typer.Typer(help="Render and verify recorded runs.", no_args_is_help=True)
console = Console()

REPORT_NAME = "report.md"
ENGINE_MODULE = "quantlab.adapters.engine.simple_bar"
ENGINE_CLASS = "SimpleBarEngine"

#: Section 11.4's tolerance. Tight on purpose: two runs of a deterministic engine
#: over the same bars differ only if something about them is not the same.
REL_TOL = 1e-9
ABS_TOL = 1e-12


def compare_metrics(
    reference: dict[str, Any], observed: dict[str, Any]
) -> list[tuple[str, Any, Any]]:
    """Metrics that disagree, per spec section 11.4.

    Only metrics the reference *defined* are compared: a metric that was ``None``
    was undefined for that run, and demanding it still be undefined would fail a
    reproduction for agreeing about nothing.
    """
    differences: list[tuple[str, Any, Any]] = []
    for name, expected in sorted(reference.items()):
        if expected is None:
            continue
        actual = observed.get(name)
        if actual is None:
            differences.append((name, expected, None))
        elif isinstance(expected, bool) or isinstance(actual, bool):
            if bool(expected) != bool(actual):
                differences.append((name, expected, actual))
        elif not math.isclose(float(expected), float(actual), rel_tol=REL_TOL, abs_tol=ABS_TOL):
            differences.append((name, expected, actual))
    return differences


def _services(ctx: typer.Context) -> tuple[SqliteExperimentStore, FileArtifactStore]:
    state = ctx.obj
    missing = missing_tables(state.container.db_engine)
    if missing:
        raise ConfigError(
            "the experiment database is not migrated; run `quantlab db upgrade`",
            missing_tables=list(missing),
        )
    return (
        state.container.store,
        FileArtifactStore(state.config.project.artifacts_dir),
    )


def _require_run(store: SqliteExperimentStore, run_id: str) -> Any:
    row = store.find_run(run_id)
    if row is None:
        raise StoreError("no such run", run_id=run_id)
    return row


@app.command("show")
def show(
    ctx: typer.Context,
    run_id: Annotated[str, typer.Argument(help="The run to report on.")],
    out: Annotated[Path | None, typer.Option("--out", help="Directory to write into.")] = None,
) -> None:
    """Render a run's Markdown report and tearsheet from what was recorded."""
    store, artifacts = _services(ctx)
    row = _require_run(store, run_id)
    metrics = store.metrics_for(run_id)
    environment = (
        artifacts.read_json(run_id, "env.json") if artifacts.exists(run_id, "env.json") else {}
    )
    params = (
        artifacts.read_json(run_id, "params.json")
        if artifacts.exists(run_id, "params.json")
        else {}
    )

    directory = Path(out) if out is not None else artifacts.dir_for(run_id)
    directory.mkdir(parents=True, exist_ok=True)

    tearsheet = ""
    if artifacts.exists(run_id, "equity.parquet"):
        frame = artifacts.read_parquet(run_id, "equity.parquet")
        write_tearsheet(frame["equity"], directory / TEARSHEET_NAME, title=f"run {run_id}")
        tearsheet = TEARSHEET_NAME

    report = RunReport(
        run_id=run_id,
        strategy_id=row.strategy_id,
        segment=row.segment,
        status=row.status,
        engine=f"{row.engine_name}:{row.engine_version}",
        dataset_id=row.dataset_id,
        split_id=row.split_id,
        params=dict(params),
        metrics=dict(metrics),
        environment=dict(environment),
        n_trades=len(store.trades_for(run_id)),
        artifact_dir=row.artifact_dir,
        tearsheet=tearsheet,
    )
    path = directory / REPORT_NAME
    path.write_text(render_run_report(report), encoding="utf-8")
    console.print(f"wrote {path}")
    if tearsheet:
        console.print(f"wrote {directory / TEARSHEET_NAME}")


@app.command("reproduce")
def reproduce(
    ctx: typer.Context,
    run_id: Annotated[str, typer.Argument(help="The run to reproduce.")],
) -> None:
    """Re-run a stored run and compare every metric it defined (INV-7).

    Exits non-zero on any disagreement, and prints the trades that differ so the
    divergence can be located rather than merely reported.
    """
    container = ctx.obj.container
    store, artifacts = _services(ctx)
    row = _require_run(store, run_id)
    if row.status != "ok":
        raise StoreError(
            "only a successful run can be reproduced", run_id=run_id, status=row.status
        )

    loader = StrategyLoader(
        store, FileSourceStore(Path("strategies") / FileSourceStore.DEFAULT_SUBDIR)
    )
    source = loader.verify(row.strategy_id)
    settings = BacktestConfig.model_validate(artifacts.read_json(run_id, "backtest_config.json"))
    params = artifacts.read_json(run_id, "params.json")

    if container.market_data is None:  # pragma: no cover - every profile wires one
        raise StoreError("no market data source is configured")
    policy = container.split_policy
    window = policy.segment(row.segment)
    bars = container.market_data.load(
        symbol=policy.symbol,
        timeframe=policy.timeframe,
        start_ts=window.start_ts,
        end_ts=window.end_ts,
    )

    evaluator = SandboxEvaluator(
        source=source,
        engine_module=ENGINE_MODULE,
        engine_class=ENGINE_CLASS,
        engine_name=row.engine_name,
        engine_version=row.engine_version,
        runner=SandboxRunner(),
    )
    fresh = compute_metrics(evaluator.evaluate(bars, params, settings))
    differences = compare_metrics(dict(store.metrics_for(run_id)), fresh.as_dict())

    if not differences:
        console.print(f"[green]run {run_id} reproduces exactly[/green]")
        return

    table = Table(title=f"run {run_id} did NOT reproduce")
    table.add_column("metric")
    table.add_column("stored")
    table.add_column("reproduced")
    for name, expected, actual in differences:
        table.add_row(name, json.dumps(expected), json.dumps(actual))
    console.print(table)
    _print_trade_diff(store, run_id, evaluator, bars, params, settings)
    raise typer.Exit(1)


def _print_trade_diff(
    store: SqliteExperimentStore,
    run_id: str,
    evaluator: SandboxEvaluator,
    bars: Any,
    params: Any,
    settings: BacktestConfig,
) -> None:
    """Show where the two trade ledgers first diverge (spec section 11.4)."""
    stored = store.trades_for(run_id)
    fresh = evaluator.evaluate(bars, params, settings).trades
    table = Table(title="first trade divergence")
    table.add_column("field")
    table.add_column("stored")
    table.add_column("reproduced")
    for index in range(max(len(stored), len(fresh))):
        left = stored[index] if index < len(stored) else None
        right = fresh[index] if index < len(fresh) else None
        if (
            left is None
            or right is None
            or (left.entry_ts != right.entry_ts or left.exit_ts != right.exit_ts)
        ):
            table.add_row("trade_no", str(index + 1), str(index + 1))
            table.add_row("stored", "—" if left is None else f"{left.entry_ts}->{left.exit_ts}", "")
            table.add_row(
                "reproduced", "", "—" if right is None else f"{right.entry_ts}->{right.exit_ts}"
            )
            console.print(table)
            return
    console.print(f"trade ledgers agree ({len(stored)} trades); the difference is in the metrics")
