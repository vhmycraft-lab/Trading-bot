"""Run identity, caching, and the record of what a run produced (spec section 11).

A run is identified by everything it depends on (section 11.2): the strategy, its
parameters, the data, the partition, the segment, the costs, the seed and the
engine build. Two requests that agree on all of those *are* the same run, and the
second one must not execute — not to save time, though it does, but because an
evolutionary generation carries survivors forward unchanged and would otherwise
re-derive numbers it already has, inflating the evaluation count that section 14.4
uses to deflate for multiple testing.

The runner stores; it does not decide. Metrics come from ``core.metrics``, the
identity rule from ``core.hashing``, the simulation from an injected evaluator.
This module's whole job is to make sure that what happened is written down once,
completely, and in a form the run can be rebuilt from.
"""

from __future__ import annotations

import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd

from quantlab.core.errors import EngineError, RunConflict, StrategyError
from quantlab.core.hashing import canonical_json
from quantlab.core.hashing import run_id as compute_run_id
from quantlab.core.logging import get_logger
from quantlab.core.metrics import MetricSet, compute_metrics
from quantlab.core.types import BacktestConfig, BacktestResult, BarFrame
from quantlab.experiments.env import environment_report
from quantlab.ports.engine import StrategyEvaluator
from quantlab.ports.store import ArtifactStore, ExperimentStore, RunRecord

__all__ = ["ARTIFACT_NAMES", "ExperimentRunner", "RunOutcome"]

log = get_logger(__name__)

#: The files every run leaves behind (spec section 11.3). Named here so a run that
#: wrote fewer of them is a detectable defect rather than a quiet gap.
ARTIFACT_NAMES: Final[tuple[str, ...]] = (
    "params.json",
    "backtest_config.json",
    "equity.parquet",
    "position.parquet",
    "signals.parquet",
    "trades.parquet",
    "fills.parquet",
    "metrics.json",
    "engine_log.txt",
    "env.json",
)

#: Lines of traceback kept in ``error_json`` (spec section 18.2). Enough to see
#: where a strategy broke, bounded so a runaway traceback cannot fill the database.
_TRACEBACK_TAIL_LINES: Final[int] = 20


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """What one request produced, whether or not anything was executed."""

    run_id: str
    record: RunRecord
    cache_hit: bool
    result: BacktestResult | None = None
    metrics: MetricSet | None = None

    @property
    def ok(self) -> bool:
        return self.record.status == "ok"


class ExperimentRunner:
    """Executes a backtest at most once per identity (spec section 11.2)."""

    __slots__ = ("artifacts", "store")

    def __init__(self, store: ExperimentStore, artifacts: ArtifactStore) -> None:
        self.store = store
        self.artifacts = artifacts

    def __repr__(self) -> str:
        return f"ExperimentRunner(store={self.store!r})"

    # -- identity -----------------------------------------------------------
    @staticmethod
    def run_id_for(
        *,
        strategy_id: str,
        params: Mapping[str, Any],
        dataset_id: str,
        split_id: str,
        segment: str,
        config: BacktestConfig,
        seed: int,
        engine_name: str,
        engine_version: str,
    ) -> str:
        """The run id of spec section 11.2.

        Delegated to ``core.hashing`` rather than spelled out here: the loader
        derives ``strategy_id`` from the same module, and two implementations of
        one identity rule would split the cache without anyone noticing.
        """
        return compute_run_id(
            strategy=strategy_id,
            params=params,
            dataset=dataset_id,
            split=split_id,
            segment=segment,
            backtest_config_hash=config.config_hash(),
            seed=seed,
            engine_name=engine_name,
            engine_version=engine_version,
        )

    # -- running ------------------------------------------------------------
    def run(
        self,
        evaluator: StrategyEvaluator,
        *,
        experiment_id: str,
        strategy_id: str,
        dataset_id: str,
        split_id: str,
        segment: str,
        bars: BarFrame,
        params: Mapping[str, Any] | None = None,
        config: BacktestConfig | None = None,
        force: bool = False,
    ) -> RunOutcome:
        """Execute the backtest, or return the one already stored.

        Spec section 11.2: a run already recorded as ``ok`` is returned without
        executing unless ``force``. A **failed** run is re-executed — it produced
        no results anyone can be citing, and the usual reason it failed is that
        something has since been fixed.

        Raises:
            EngineError: an accounting invariant broke. Never caught here: it means
                the engine is wrong, not the strategy, and a run recorded as
                ``failed`` would hide a bug behind a plausible-looking result.
        """
        settings = config or BacktestConfig()
        supplied = dict(params or {})
        run_id = self.run_id_for(
            strategy_id=strategy_id,
            params=supplied,
            dataset_id=dataset_id,
            split_id=split_id,
            segment=segment,
            config=settings,
            seed=settings.seed,
            engine_name=evaluator.engine_name,
            engine_version=evaluator.engine_version,
        )

        existing = self.store.find_run(run_id)
        if existing is not None and existing.status == "ok" and not force:
            log.info(
                "run_cache_hit",
                run_id=run_id,
                strategy_id=strategy_id,
                segment=segment,
                cache_hit=True,
            )
            return RunOutcome(run_id=run_id, record=existing, cache_hit=True)

        if existing is not None and existing.status == "ok":
            # `force` on a recorded run: re-execute and hand back what came out,
            # without touching the row. Section 6 permits no deletion of a
            # successful run — its numbers may already have been cited — so the
            # honest shape of "re-run" is to produce a fresh result for the caller
            # to compare against, which is exactly what `quantlab reproduce` does.
            return self._rerun(evaluator, run_id, existing, bars, supplied, settings)

        self._open(
            run_id=run_id,
            existing=existing,
            experiment_id=experiment_id,
            strategy_id=strategy_id,
            dataset_id=dataset_id,
            split_id=split_id,
            segment=segment,
            params_json=canonical_json(supplied),
            settings=settings,
            evaluator=evaluator,
        )
        log.info(
            "run_started",
            run_id=run_id,
            strategy_id=strategy_id,
            segment=segment,
            cache_hit=False,
            n_bars=bars.n_bars,
        )

        try:
            result = evaluator.evaluate(bars, supplied, settings)
        except EngineError:
            # Section 18.2: an accounting invariant is a bug in the platform. It
            # must fail the command, not be filed as a badly behaved strategy.
            raise
        except (StrategyError, ValueError, ArithmeticError, TypeError, KeyError) as exc:
            return self._record_failure(run_id, exc)

        metrics = compute_metrics(result)
        self._write_artifacts(run_id, result, metrics, supplied, settings)
        finished = self.store.finish_run(
            run_id, "ok", metrics=metrics.as_dict(), trades=result.trades
        )
        log.info(
            "run_finished",
            run_id=run_id,
            status="ok",
            n_trades=len(result.trades),
            final_equity=result.final_equity,
        )
        return RunOutcome(
            run_id=run_id, record=finished, cache_hit=False, result=result, metrics=metrics
        )

    # -- internals ----------------------------------------------------------
    def _open(
        self,
        *,
        run_id: str,
        existing: RunRecord | None,
        experiment_id: str,
        strategy_id: str,
        dataset_id: str,
        split_id: str,
        segment: str,
        params_json: str,
        settings: BacktestConfig,
        evaluator: StrategyEvaluator,
    ) -> RunRecord:
        """Get the run row into a state that can be executed.

        A previous attempt that failed is cleared rather than reused: its metrics
        and trades are keyed by run id, and finishing the same id twice would
        either collide or silently discard the second set.
        """
        if existing is not None:
            if existing.status == "failed":
                # Cleared rather than reused: metrics and trades are keyed by run
                # id, so finishing the same id twice would either collide or
                # silently discard the second set.
                self.store.delete_run(run_id, confirm=True)
            elif existing.status in ("pending", "running"):
                return existing
            else:
                raise RunConflict(
                    "a run in this state cannot be executed",
                    run_id=run_id,
                    status=existing.status,
                )
        return self.store.create_run(
            run_id=run_id,
            experiment_id=experiment_id,
            strategy_id=strategy_id,
            dataset_id=dataset_id,
            split_id=split_id,
            segment=segment,
            params_json=params_json,
            engine_name=evaluator.engine_name,
            engine_version=evaluator.engine_version,
            artifact_dir=str(self.artifacts.dir_for(run_id)),
            cost_multiplier=settings.cost_multiplier,
        )

    def _rerun(
        self,
        evaluator: StrategyEvaluator,
        run_id: str,
        record: RunRecord,
        bars: BarFrame,
        params: Mapping[str, Any],
        settings: BacktestConfig,
    ) -> RunOutcome:
        """Execute a recorded run again and return the fresh result, storing nothing."""
        log.info("run_forced", run_id=run_id, cache_hit=False, forced=True)
        result = evaluator.evaluate(bars, params, settings)
        return RunOutcome(
            run_id=run_id,
            record=record,
            cache_hit=False,
            result=result,
            metrics=compute_metrics(result),
        )

    def _record_failure(self, run_id: str, exc: BaseException) -> RunOutcome:
        """File a strategy failure as a failed run (spec section 18.2)."""
        tail = "".join(traceback.format_exception(exc)).splitlines()[-_TRACEBACK_TAIL_LINES:]
        error = {
            "type": type(exc).__name__,
            "message": str(exc)[:2000],
            "traceback_tail": "\n".join(tail),
        }
        finished = self.store.finish_run(run_id, "failed", error=error)
        log.warning("run_failed", run_id=run_id, error_type=error["type"])
        return RunOutcome(run_id=run_id, record=finished, cache_hit=False)

    def _write_artifacts(
        self,
        run_id: str,
        result: BacktestResult,
        metrics: MetricSet,
        params: Mapping[str, Any],
        settings: BacktestConfig,
    ) -> None:
        """Write the ten files of spec section 11.3, in canonical form."""
        self.artifacts.write_json(run_id, "params.json", dict(params))
        self.artifacts.write_json(run_id, "backtest_config.json", settings.model_dump(mode="json"))
        index = np.asarray(result.equity.index, dtype="int64")
        for name, series in (
            ("equity", result.equity),
            ("position", result.position_frac),
            ("signals", result.signals),
        ):
            self.artifacts.write_parquet(
                run_id,
                f"{name}.parquet",
                pd.DataFrame({"ts_open": index, name: list(series)}),
            )
        self.artifacts.write_parquet(run_id, "trades.parquet", _trades_frame(result))
        self.artifacts.write_parquet(run_id, "fills.parquet", _fills_frame(result))
        self.artifacts.write_json(run_id, "metrics.json", metrics.as_dict())
        _write_text(self.artifacts, run_id, "engine_log.txt", "\n".join(result.log) + "\n")
        self.artifacts.write_json(
            run_id,
            "env.json",
            environment_report(engine_version=result.engine_version),
        )


def _write_text(artifacts: ArtifactStore, run_id: str, name: str, text: str) -> None:
    """Write plain text through whichever artifact store the caller supplied.

    ``write_text`` is not on the port — the port is the four methods section 11.1
    names — so a store that does not offer it gets the log as JSON rather than
    losing it.
    """
    writer = getattr(artifacts, "write_text", None)
    if callable(writer):
        writer(run_id, name, text)
    else:  # pragma: no cover - every shipped store has write_text
        artifacts.write_json(run_id, f"{name}.json", {"text": text})


def _trades_frame(result: BacktestResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_no": pd.Series([t.trade_no for t in result.trades], dtype="int64"),
            "side": [str(t.side) for t in result.trades],
            "entry_ts": pd.Series([t.entry_ts for t in result.trades], dtype="int64"),
            "entry_px": pd.Series([t.entry_px for t in result.trades], dtype="float64"),
            "exit_ts": pd.Series([t.exit_ts for t in result.trades], dtype="int64"),
            "exit_px": pd.Series([t.exit_px for t in result.trades], dtype="float64"),
            "qty": pd.Series([t.qty for t in result.trades], dtype="float64"),
            "fees": pd.Series([t.fees for t in result.trades], dtype="float64"),
            "slippage_cost": pd.Series([t.slippage_cost for t in result.trades], dtype="float64"),
            "pnl": pd.Series([t.pnl for t in result.trades], dtype="float64"),
            "pnl_pct": pd.Series([t.pnl_pct for t in result.trades], dtype="float64"),
            "bars_held": pd.Series([t.bars_held for t in result.trades], dtype="int64"),
            "exit_reason": [str(t.exit_reason) for t in result.trades],
        }
    )


def _fills_frame(result: BacktestResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bar_index": pd.Series([f.bar_index for f in result.fills], dtype="int64"),
            "ts": pd.Series([f.ts for f in result.fills], dtype="int64"),
            "side": [str(f.side) for f in result.fills],
            "qty": pd.Series([f.qty for f in result.fills], dtype="float64"),
            "ref_price": pd.Series([f.ref_price for f in result.fills], dtype="float64"),
            "fill_price": pd.Series([f.fill_price for f in result.fills], dtype="float64"),
            "fee": pd.Series([f.fee for f in result.fills], dtype="float64"),
            "slippage_cost": pd.Series([f.slippage_cost for f in result.fills], dtype="float64"),
        }
    )
