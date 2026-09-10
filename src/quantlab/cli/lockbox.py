"""``quantlab-lockbox`` — the one door onto the test partition (spec section 14.6, INV-5). 🔒

    quantlab-lockbox evaluate <strategy_id> --params <run_id> --reason "<text>"

This module is a **separate console script**, not a subcommand of ``quantlab``.
Section 14.6 requires that the research CLI never import it, and the surest way
to keep an import from happening is for there to be nothing to import it *for*:
the research binary has no lockbox command to route to, so no research code path
can reach the unguarded data source however it is invoked.

The order of operations is the whole security property, and it is deliberately
front-loaded:

1. every eligibility rule is checked (:mod:`quantlab.core.validation.lockbox`)
2. ``test_end_ts`` is frozen, so the window cannot grow after being looked at
3. **the access is recorded** — before a single test bar is read
4. only then is the unguarded container built and the run executed

Step 3 before step 4 is what makes the budget real. An access recorded after the
run would be free whenever the run crashed, and "crash, look at the traceback,
try again" is a search over the test partition.

Nothing here is reachable from the evolutionary loop, from ``quantlab
validate``, or from any prompt: the numbers this command computes go into a
``LOCKBOX_PASS``/``LOCKBOX_FAIL`` verdict and stop there (INV-6).
"""

from __future__ import annotations

import getpass
import json
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from quantlab.adapters.engine.simple_bar import SimpleBarEngine
from quantlab.adapters.store.artifacts import FileArtifactStore, FileSourceStore
from quantlab.adapters.store.sqlite import SqliteExperimentStore, missing_tables
from quantlab.cli._exit import QuantLabGroup
from quantlab.container import build_container
from quantlab.core.config import load_config
from quantlab.core.errors import ConfigError, LockboxViolation, StoreError
from quantlab.core.hashing import short_id
from quantlab.core.metrics import compute_metrics
from quantlab.core.splits import SplitPolicy, load_split_policy
from quantlab.core.types import BacktestConfig, BarFrame, SlippageConfig
from quantlab.core.validation.deflated_sharpe import M_FORMULA_VERSION
from quantlab.core.validation.gates import GateInputs, GateReport
from quantlab.core.validation.lockbox import (
    CLOSED_FAMILY_STATUS,
    LockboxDecision,
    decide,
    lockbox_gates,
    require_budget,
    require_candidate,
    require_reason,
)
from quantlab.sandbox.runner import SandboxLimits, SandboxRunner
from quantlab.strategies_io.evaluators import SandboxEvaluator
from quantlab.strategies_io.loader import StrategyLoader

__all__ = ["app", "evaluate_lockbox", "main"]

console = Console()

ENGINE_MODULE = "quantlab.adapters.engine.simple_bar"
ENGINE_CLASS = "SimpleBarEngine"

#: Section 14.6: "base + 2x costs on the test segment".
COST_MULTIPLIERS: tuple[float, ...] = (1.0, 2.0)

app = typer.Typer(
    cls=QuantLabGroup,
    name="quantlab-lockbox",
    help="Evaluate one strategy against the held-out test partition. Irreversible.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)


def _config_at(config: Any, multiplier: float) -> BacktestConfig:
    """The backtest settings at ``multiplier`` times the base costs."""
    settings = config.backtest
    return BacktestConfig(
        initial_equity=settings.initial_equity,
        fee_bps=settings.fee_bps,
        slippage=SlippageConfig.model_validate(settings.slippage.model_dump()),
        allow_short=settings.allow_short,
        short_borrow_bps_per_bar=settings.short_borrow_bps_per_bar,
        max_position_fraction=settings.max_position_fraction,
        fill_rule=settings.fill_rule,
        seed=config.evolution.seed,
        cost_multiplier=multiplier,
    )


def _frozen_policy(store: SqliteExperimentStore, policy: SplitPolicy, source: Any) -> SplitPolicy:
    """Register the split and pin its test end (section 14.6, step "freezes on first use").

    An unfrozen test segment is open-ended, and an open-ended window is one that
    could be waited out: a family that failed in March could return in June to a
    strictly longer test period. The end is pinned to the last bar the dataset
    actually holds, once, and every later access to this split sees the same one.
    """
    stored = store.get_or_create_split(policy)
    if stored.test_end_ts is not None:
        return stored
    available = source.available_range(policy.symbol, policy.timeframe)
    if available is None:
        raise ConfigError(
            "the dataset holds no bars, so the test window cannot be frozen",
            symbol=policy.symbol,
            timeframe=policy.timeframe,
        )
    return store.freeze_test_end(stored.split_id, int(available[1]))


def _bars(source: Any, policy: SplitPolicy, segment: str) -> BarFrame:
    window = policy.segment(segment)
    bars: BarFrame = source.load(
        symbol=policy.symbol,
        timeframe=policy.timeframe,
        start_ts=window.start_ts,
        end_ts=window.end_ts,
    )
    return bars


def _baseline(loader: StrategyLoader, bars: BarFrame, config: BacktestConfig) -> Any:
    """``buy_and_hold`` on the test segment, for ``G_BENCH``.

    Through the loader and the sandbox like everything else (INV-4).
    """
    source = (Path("strategies") / "baselines" / "buy_and_hold.py").read_text(encoding="utf-8")
    loaded = loader.load_source(source, family="buy_and_hold", origin="human", author="human")
    evaluator = SandboxEvaluator(
        source=loaded.source,
        engine_module=ENGINE_MODULE,
        engine_class=ENGINE_CLASS,
        engine_name=SimpleBarEngine.name,
        engine_version=SimpleBarEngine.version,
        runner=SandboxRunner(),
    )
    return compute_metrics(evaluator.evaluate(bars, {}, config))


def evaluate_lockbox(
    config: Any,
    *,
    strategy_id: str,
    run_id: str,
    reason: str,
    os_user: str,
    baselines: bool = True,
) -> LockboxDecision:
    """One lockbox access, end to end (spec section 14.6).

    Separated from the Typer command so the ordering that matters — eligibility,
    freeze, **record**, then run — is testable without a terminal.

    Args:
        config: the resolved :class:`~quantlab.core.config.AppConfig`.
        strategy_id: the registered version to evaluate.
        run_id: a recorded run whose ``params.json`` supplies the parameters.
            Section 14.6 spells ``--params`` as a run id rather than a JSON
            literal on purpose: the parameters must be ones the platform already
            measured on train and validation, not a fresh set typed at the door.
        reason: why this family is spending its one look.
        os_user: who is spending it.
        baselines: run ``buy_and_hold`` on the test segment for ``G_BENCH``.

    Raises:
        LockboxViolation: any eligibility rule refused the access. Nothing has
            been read and nothing recorded.
        StoreError: the strategy, its family, or the cited run is unknown.
    """
    reason = require_reason(reason)

    # -- eligibility, entirely from the store: no bars are touched yet --------
    research = build_container(config, profile="research")
    missing = missing_tables(research.db_engine)
    if missing:
        raise ConfigError(
            "the experiment database is not migrated; run `quantlab db upgrade`",
            missing_tables=list(missing),
        )
    store = SqliteExperimentStore(research.session_factory)

    version = store.get_strategy_version(strategy_id)
    if version is None:
        raise StoreError("no such strategy version", strategy_id=strategy_id)
    family_id = str(version.family_id)

    family = store.find_family(family_id)
    if family is None:
        raise StoreError("no such strategy family", family_id=family_id)
    if str(family.status) == CLOSED_FAMILY_STATUS:
        raise LockboxViolation(
            "this family is closed; a closed family has already been measured on the "
            "test partition and did not hold",
            family_id=family_id,
        )

    require_candidate(store.verdicts_for(strategy_id))
    require_budget(
        store.lockbox_accesses(),
        config.lockbox,
        family_id=family_id,
        now_ms=store.now_ms(),
    )

    artifacts = FileArtifactStore(config.project.artifacts_dir)
    if not artifacts.exists(run_id, "params.json"):
        raise StoreError(
            "the cited run has no recorded parameters; --params names a run whose "
            "parameters were already measured on train and validation",
            run_id=run_id,
        )
    params = dict(artifacts.read_json(run_id, "params.json"))

    # -- the unguarded source, and the frozen window -------------------------
    vault = build_container(config, profile="lockbox")
    if vault.market_data is None:  # pragma: no cover - the lockbox profile always wires one
        raise ConfigError("no market data source is configured")
    policy = load_split_policy(config.splits.policy_file)
    dataset_id = vault.market_data.dataset_id(policy.symbol, policy.timeframe)
    frozen = _frozen_policy(store, replace(policy, dataset_id=dataset_id), vault.market_data)

    # -- the access is recorded BEFORE a single test bar is read -------------
    store.record_lockbox_access(
        strategy_id=strategy_id,
        family_id=family_id,
        os_user=os_user,
        reason=reason,
    )

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
    loaded = loader.load_registered(strategy_id)
    evaluator = SandboxEvaluator(
        source=loaded.source,
        engine_module=ENGINE_MODULE,
        engine_class=ENGINE_CLASS,
        engine_name=SimpleBarEngine.name,
        engine_version=SimpleBarEngine.version,
        runner=SandboxRunner(),
    )

    train_bars = _bars(vault.market_data, frozen, "train")
    test_bars = _bars(vault.market_data, frozen, "test")
    train_result = evaluator.evaluate(train_bars, params, _config_at(config, 1.0))
    at_cost = {
        multiplier: evaluator.evaluate(test_bars, params, _config_at(config, multiplier))
        for multiplier in COST_MULTIPLIERS
    }
    base = at_cost[1.0]
    stressed = at_cost[2.0]

    inputs = GateInputs(
        probe_passed=loaded.probe.passed if loaded.probe is not None else True,
        metrics_train=compute_metrics(train_result),
        metrics_val=compute_metrics(base),
        metrics_val_stressed=compute_metrics(stressed),
        metrics_buyhold_val=(
            _baseline(loader, test_bars, _config_at(config, 1.0)) if baselines else None
        ),
        trades_train=train_result.trades,
        trades_val=base.trades,
        position_frac_val=np.asarray(base.position_frac, dtype="float64"),
        max_position_fraction=config.backtest.max_position_fraction,
    )
    decision = decide(lockbox_gates(inputs, config.validation))

    store.record_verdict(
        verdict_id=short_id(f"lockbox|{strategy_id}|{frozen.split_id}|{json.dumps(params)}"),
        strategy_id=strategy_id,
        split_id=frozen.split_id,
        params_json=json.dumps(params, sort_keys=True),
        verdict=decision.verdict,
        overfit_score=0.0,
        hard_gates_json=json.dumps(_gates_json(decision.gates), sort_keys=True),
        soft_checks_json="{}",
        thresholds_json=json.dumps(config.validation.model_dump(mode="json"), sort_keys=True),
        # The same M the validation pipeline charges (section 14.4), assembled from
        # the store rather than from the family's touches alone: a lockbox verdict
        # that recorded a smaller M than the validation it followed would understate
        # the search on the one record nobody can ever redo.
        n_trials_accounted=int(store.search_trials_for_family(family_id)["total"]),
        m_formula_version=M_FORMULA_VERSION,
    )
    if decision.family_closed:
        store.set_family_status(family_id, CLOSED_FAMILY_STATUS)
    return decision


def _gates_json(gates: GateReport) -> dict[str, Any]:
    return {
        result.gate_id: {"passed": result.passed, "reason": result.reason}
        for result in gates.results
    }


@app.callback()
def _root() -> None:
    """Keep ``evaluate`` a subcommand.

    Typer collapses a single-command app into that command, which would make the
    invocation ``quantlab-lockbox <strategy_id>`` and silently change the spelling
    section 14.6 documents. An empty callback keeps the group.
    """


@app.command("evaluate")
def evaluate(
    strategy_id: Annotated[str, typer.Argument(help="A registered strategy version.")],
    params: Annotated[
        str, typer.Option("--params", help="Run id whose recorded parameters to use.")
    ],
    reason: Annotated[
        str, typer.Option("--reason", help="Why this family is spending its one look.")
    ],
    config: Annotated[
        list[Path] | None,
        typer.Option("--config", "-c", help="YAML file merged over configs/default.yaml."),
    ] = None,
    set_: Annotated[
        list[str] | None,
        typer.Option("--set", "-s", metavar="SECTION.KEY=VALUE", help="Override one value."),
    ] = None,
    baselines: Annotated[
        bool, typer.Option("--baselines/--no-baselines", help="Run buy_and_hold for G_BENCH.")
    ] = True,
) -> None:
    """Evaluate one strategy on the test partition. This cannot be undone."""
    resolved = load_config(list(config or []), list(set_ or []))
    decision = evaluate_lockbox(
        resolved,
        strategy_id=strategy_id,
        run_id=params,
        reason=reason,
        os_user=getpass.getuser(),
        baselines=baselines,
    )

    table = Table(title=f"lockbox evaluation of {strategy_id}", show_header=False)
    table.add_row("verdict", f"[bold]{decision.verdict}[/bold]")
    table.add_row("failed gates", ", ".join(decision.gates.failed) or "none")
    table.add_row("family", "closed" if decision.family_closed else "open")
    console.print(table)
    for result in decision.gates.results:
        if not result.passed:
            console.print(f"  [red]{result.gate_id}[/red]: {result.reason}")
    if not decision.passed:
        raise typer.Exit(1)


def main() -> None:  # pragma: no cover - console-script shim
    app()
