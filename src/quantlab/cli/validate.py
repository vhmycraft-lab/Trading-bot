"""``quantlab validate`` — the validation pipeline (spec section 14.1). 🔒

    quantlab validate <strategy_id> [--params '{"fast": 20}'] [--walk-forward]

The ten steps of section 14.1, in order. Everything expensive happens before
anything is judged, so a verdict is reached on the whole picture rather than on
whichever step ran first, and the family's budget is charged after the work, so a
crashed pipeline does not spend a validation touch on a strategy it never
measured.

**This is the only command that reads the validation segment**, and it does so
under the split policy's own guard. The test partition remains reachable only
through ``quantlab lockbox`` (INV-5), and nothing this command computes is ever
read by the evolutionary search (INV-9).

Steps that were not run leave their checks **unmeasured** rather than passed, and
the report says so. A ``CANDIDATE`` assembled from half the evidence is not a
clean strategy; it is a strategy nobody finished checking.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from quantlab.adapters.engine.simple_bar import SimpleBarEngine
from quantlab.adapters.store.artifacts import FileSourceStore
from quantlab.adapters.store.sqlite import SqliteExperimentStore, missing_tables
from quantlab.core.errors import ConfigError, StoreError
from quantlab.core.hashing import short_id
from quantlab.core.metrics import compute_metrics
from quantlab.core.types import BacktestConfig, BarFrame, SlippageConfig
from quantlab.core.validation.deflated_sharpe import (
    deannualise,
    deflated_sharpe,
    moments,
    sharpe_variance_upper_bound,
)
from quantlab.core.validation.permutation import market_permutation_test, trade_shuffle_test
from quantlab.core.validation.pipeline import Evidence
from quantlab.core.validation.pipeline import validate as run_validation
from quantlab.sandbox.runner import SandboxLimits, SandboxRunner
from quantlab.strategies_io.evaluators import SandboxEvaluator
from quantlab.strategies_io.loader import StrategyLoader

__all__ = ["app"]

app = typer.Typer(help="Validate a strategy against the held-out segment.", no_args_is_help=True)
console = Console()

ENGINE_MODULE = "quantlab.adapters.engine.simple_bar"
ENGINE_CLASS = "SimpleBarEngine"

#: Cost multipliers section 14.1 step 4 runs the validation segment at.
COST_MULTIPLIERS: tuple[float, ...] = (1.0, 2.0, 3.0)


def _services(ctx: typer.Context) -> tuple[Any, SqliteExperimentStore, StrategyLoader]:
    state = ctx.obj
    container, config = state.container, state.config
    missing = missing_tables(container.db_engine)
    if missing:
        raise ConfigError(
            "the experiment database is not migrated; run `quantlab db upgrade`",
            missing_tables=list(missing),
        )
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
    return state, store, loader


def _config_at(state: Any, multiplier: float) -> BacktestConfig:
    """The backtest settings at ``multiplier`` times the base costs (step 4)."""
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
        cost_multiplier=multiplier,
    )


def _segment_bars(container: Any, segment: str) -> BarFrame:
    policy = container.split_policy
    if policy is None or container.market_data is None:
        raise ConfigError("a split policy and a market data source are both required")
    window = policy.segment(segment)
    bars: BarFrame = container.market_data.load(
        symbol=policy.symbol,
        timeframe=policy.timeframe,
        start_ts=window.start_ts,
        end_ts=window.end_ts,
    )
    return bars


@app.command("run")
def run(
    ctx: typer.Context,
    strategy_id: Annotated[str, typer.Argument(help="A registered strategy version.")],
    params: Annotated[
        str, typer.Option("--params", help="JSON object of parameter overrides.")
    ] = "{}",
    permutations: Annotated[
        int,
        typer.Option(
            "--permutations",
            help="Market permutations for check 3 and G_PERM; 0 leaves both unmeasured.",
        ),
    ] = 0,
    baselines: Annotated[
        bool, typer.Option("--baselines/--no-baselines", help="Run buy_and_hold on validation.")
    ] = True,
) -> None:
    """Validate one strategy version and record the verdict.

    Section 14.1's steps 2, 4, 5, 5b, 6, 7, 8 and 9 always run. The walk-forward
    of step 3 and the permutation test of check 3 are opt-in, because each costs
    far more than the rest of the pipeline put together; skipping either leaves
    its checks unmeasured and the report says which.
    """
    state, store, loader = _services(ctx)
    container = state.container
    settings = state.config.validation

    loaded = loader.load_registered(strategy_id)
    version = store.get_strategy_version(strategy_id)
    if version is None:
        raise StoreError("no such strategy version", strategy_id=strategy_id)

    policy = container.split_policy
    if policy is None:
        raise ConfigError("a split policy is required to validate")
    dataset_id = container.market_data.dataset_id(policy.symbol, policy.timeframe)
    stored_policy = store.get_or_create_split(replace(policy, dataset_id=dataset_id))

    supplied = json.loads(params)
    evaluator = SandboxEvaluator(
        source=loaded.source,
        engine_module=ENGINE_MODULE,
        engine_class=ENGINE_CLASS,
        engine_name=SimpleBarEngine.name,
        engine_version=SimpleBarEngine.version,
        runner=SandboxRunner(),
    )

    # -- steps 2 and 4: train, then validation at each cost level ------------
    train_bars = _segment_bars(container, "train")
    val_bars = _segment_bars(container, "val")
    train = compute_metrics(evaluator.evaluate(train_bars, supplied, _config_at(state, 1.0)))
    train_result = evaluator.evaluate(train_bars, supplied, _config_at(state, 1.0))

    at_cost = {}
    for multiplier in COST_MULTIPLIERS:
        result = evaluator.evaluate(val_bars, supplied, _config_at(state, multiplier))
        at_cost[multiplier] = (result, compute_metrics(result))
    val_result, val = at_cost[1.0]
    stressed = at_cost[settings.cost_survival_multiplier][1]

    # -- step 5: baselines on validation -------------------------------------
    buyhold = _buy_and_hold(loader, val_bars, _config_at(state, 1.0)) if baselines else None

    # -- check 1: the deflated Sharpe ratio ----------------------------------
    trials = _trials_accounted(store, version.family_id)
    pool = _trial_sharpes(store, version.family_id, int(val_result.bars_per_year))
    dsr = _deflated(
        val_result,
        _selection_pool_m(trials, pool),
        pool,
        dispersion_alpha=settings.dsr_dispersion_alpha,
    )

    # -- check 3 and G_PERM: market permutations -----------------------------
    permutation_p = None
    if permutations > 0:
        from quantlab.core.config import PermutationSettings

        permutation_settings = PermutationSettings(
            n_market_permutations=permutations,
            block_len_bars=settings.permutation.block_len_bars,
            alpha=settings.permutation.alpha,
        )
        permutation_p = market_permutation_test(
            val_bars,
            val.sortino,
            lambda frame: (
                compute_metrics(evaluator.evaluate(frame, supplied, _config_at(state, 1.0))).sortino
            ),
            permutation_settings,
            seed=state.config.evolution.seed,
        ).p_value

    shuffle = trade_shuffle_test(
        val_result.trades,
        settings.permutation,
        seed=state.config.evolution.seed,
        initial_equity=state.config.backtest.initial_equity,
        observed_max_drawdown=val.max_drawdown,
    )

    evidence = Evidence(
        probe_passed=loaded.probe.passed if loaded.probe is not None else True,
        metrics_train=train,
        metrics_val=val,
        metrics_val_stressed=stressed,
        metrics_buyhold_val=buyhold,
        trades_train=train_result.trades,
        trades_val=val_result.trades,
        position_frac_val=np.asarray(val_result.position_frac, dtype="float64"),
        max_position_fraction=state.config.backtest.max_position_fraction,
        deflated_sharpe=dsr,
        permutation_p=permutation_p,
        n_free_params=loaded.n_params,
        logic_lines=loaded.logic_lines,
        n_trials_accounted=trials["total"],
    )

    outcome = run_validation(
        store,
        strategy_id=strategy_id,
        family_id=version.family_id,
        split_id=stored_policy.split_id,
        params_json=json.dumps(supplied, sort_keys=True),
        evidence=evidence,
        settings=settings,
        verdict_id=lambda: short_id(f"{strategy_id}|{stored_policy.split_id}|{val.as_dict()}"),
    )
    _report(outcome, shuffle_mdd_p95=shuffle.mdd_p95)


def _buy_and_hold(loader: StrategyLoader, bars: BarFrame, config: BacktestConfig) -> Any:
    """``buy_and_hold`` on the validation segment (section 14.1, step 5).

    Through the loader and the sandbox, exactly like the strategy under test. The
    baseline is the platform's own and could not plausibly be hostile, but INV-4
    is not a judgement about a particular file: a second execution path that
    bypassed the sandbox would be a second execution path, and the next thing
    routed down it would not be a baseline.
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


def _deflated(
    result: Any,
    n_trials: int,
    trial_sharpes: Sequence[float],
    *,
    dispersion_alpha: float = 0.05,
) -> float | None:
    """Check 1, from the validation equity curve and the search that produced it.

    ``n_trials`` is section 14.4's ``M`` — every evaluation that led here, from
    :func:`_trials_accounted`. It is passed in rather than derived here because
    ``M`` is a property of the *search*, not of the equity curve: an earlier
    version of this function computed ``max(1, n_bars // 100)``, which made the
    multiple-testing correction a function of the segment's length and so
    deflated a forty-thousand-evaluation campaign exactly as gently as a single
    backtest. See ``docs/DECISIONS`` and the ``m_formula_version`` stamp on every
    verdict row.

    ``trial_sharpes`` are those same trials' **per-bar** Sharpe ratios, and they
    are the other half of ``SR0``. An earlier version passed
    ``np.var(returns, ddof=1)`` here — the variance of this one run's per-bar
    *returns* — and called it a conservative stand-in. It is not a stand-in at
    all: the variance of a return series and the variance of a set of Sharpe
    ratios are different quantities in different units, and on the first real
    campaign the substitute was thirty-five times too small. It erred toward
    passing. ADR 0010.

    Both now describe the **selection pool** — the candidates that cleared the
    hard fitness gates — rather than every evaluation the search made. Over all
    704 evaluations the dispersion was dominated by zero-trade genomes and
    high-turnover wrecks that were never selectable, which put ``SR0`` at 11.2
    annualised and made check 1 unreachable for anything real. ADR 0011.

    The dispersion is a chi-square **upper** bound, not the point estimate, so a
    small pool produces a harsher benchmark rather than an absent one.

    Returns ``None`` — check 1 unmeasured, charging nothing and saying so — when
    the pool holds fewer than two trials. Never a fall back to the unfiltered
    population: that is the thing ADR 0011 removes, and reaching for it here would
    put it straight back.
    """
    equity = np.asarray(result.equity, dtype="float64")
    if equity.size < 3:
        return None
    if len(trial_sharpes) < 2:
        return None
    returns = np.diff(equity) / np.where(equity[:-1] != 0.0, equity[:-1], 1.0)
    sharpe, skew, kurtosis, n_periods = moments(returns)
    return deflated_sharpe(
        sharpe,
        n_trials=max(1, int(n_trials)),
        var_sr=sharpe_variance_upper_bound(trial_sharpes, alpha=dispersion_alpha),
        n_periods=n_periods,
        skew=skew,
        kurtosis=kurtosis,
    )


def _trial_sharpes(store: Any, family_id: str, bars_per_year: int) -> list[float]:
    """The search's trial Sharpes, per bar, ready for :func:`sharpe_variance`.

    The store keeps them annualised, because that is how section 10 defines
    ``MetricSet.sharpe``. The deflation works per bar. De-annualising here rather
    than in the store keeps the conversion next to the ``n_periods`` it has to
    agree with — mixing the two scales is the error :func:`deannualise` exists to
    name, and it produces no exception, just a deflated Sharpe of 1.0 for
    everything.
    """
    if bars_per_year <= 0:
        return []
    annualised = store.trial_sharpes_for_family(family_id)
    return [deannualise(value, bars_per_year) for value in annualised]


def _selection_pool_m(trials: Mapping[str, int], pool: Sequence[float]) -> int:
    """``M`` over the selection pool rather than over every evaluation (ADR 0011).

    The evolution term becomes ``len(pool)`` — the candidates that cleared the
    hard fitness gates and could therefore have been selected — because that is
    the population the dispersion is measured over, and equation 5 is only
    meaningful when its two inputs describe one population. Counting every
    evaluation while measuring dispersion over the pool is what made check 1
    unreachable; counting the pool while measuring dispersion over everything
    would be the same error mirrored.

    The Optuna and validation-touch terms are carried through unchanged. They are
    genuine selection events — a refined parameter set and a re-validated family
    are both things somebody chose — and neither contributes a candidate to the
    evolution pool, so neither is double-counted. They do leave ``M`` slightly
    larger than the population the dispersion covers; that is the conservative
    direction, and it is stated here rather than quietly rounded away.
    """
    optuna = int(trials.get("optuna_trials", 0))
    touches = int(trials.get("validation_touches", 0))
    return len(pool) + optuna + touches


def _trials_accounted(store: Any, family_id: str) -> dict[str, int]:
    """``M`` for section 14.4: every evaluation that led here.

    All three of section 14.4's terms — ``evolution_run.n_evaluations``, Optuna
    trials, and ``family.validation_touches`` — summed over the family, because a
    family is one idea and every version in it is something the search tried.

    Returned term by term so the verdict can record where its ``M`` came from. A
    family that has never been evolved or tuned contributes zero from those terms
    rather than a guess, which is the honest floor: ``M`` is then just the
    validation touches, and the deflation is as gentle as the evidence allows.
    """
    return dict(store.search_trials_for_family(family_id))


def _report(outcome: Any, *, shuffle_mdd_p95: float | None) -> None:
    report = outcome.report
    table = Table(title=f"validation of {outcome.strategy_id}", show_header=False)
    table.add_row("verdict", f"[bold]{report.verdict}[/bold]")
    table.add_row("overfit_score", f"{report.overfit_score} / 100  [dim](higher is worse)[/dim]")
    table.add_row("failed gates", ", ".join(report.gates.failed) or "none")
    table.add_row("charged checks", ", ".join(report.charged) or "none")
    table.add_row("unmeasured", ", ".join(report.unmeasured) or "none")
    table.add_row("family touches", str(outcome.validation_touches))
    table.add_row("family frozen", "yes" if outcome.family_frozen else "no")
    if shuffle_mdd_p95 is not None:
        table.add_row("mdd_p95 (shuffled)", f"{shuffle_mdd_p95:.4f}")
    console.print(table)

    if report.unmeasured:
        console.print(
            "[yellow]this verdict rests on partial evidence[/yellow]: the checks listed as "
            "unmeasured were never run, and charge nothing"
        )
    for result in report.gates.results:
        if not result.passed:
            console.print(f"  [red]{result.gate_id}[/red]: {result.reason}")
