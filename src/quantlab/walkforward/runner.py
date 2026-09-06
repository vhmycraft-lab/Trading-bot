"""Walk-forward testing (master spec section 15).

Evolution's inner walk-forward (section 13.6) asks "does this candidate
generalise at all?", cheaply enough to run every generation, entirely inside
train. This one asks a different and much more expensive question: **does the
search procedure still work when it is re-run through time?** It spans train and
validation, re-searches each in-sample window, and measures the out-of-sample
result of a search that never saw the window it is being judged on.

Neither replaces the other, and section 15 says so. A candidate can generalise
inside train while the *procedure* that found it turns out to be fitting the
particular stretch of history it was handed.

Three consequences shape this module:

* **Every evaluation counts.** Trials inside every window add to
  ``n_trials_accounted``, which section 14.4 charges into the deflated Sharpe
  ratio's ``M``. Re-searching each window raises that bar substantially, which is
  correct — it is a much larger search.
* **Fixed parameters are weaker evidence.** With ``reoptimize_each_window``
  false the walk-forward reuses one parameter set and never tests whether the
  *search* generalises. The result says so in as many words, so a report cannot
  quietly present it as the stronger thing.
* **The optimiser is injected.** Section 13.8's Optuna study and section 13.2's
  evolution are both valid searches for a window, and INV-8 forbids this layer
  from importing the latter. The caller supplies whichever the configuration
  names.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import pandas as pd

from quantlab.core.config import WalkForwardSettings
from quantlab.core.errors import ConfigError
from quantlab.core.logging import get_logger
from quantlab.core.metrics import MetricSet, compute_metrics
from quantlab.core.splits import SplitPolicy, WalkForwardWindow, walk_forward_windows
from quantlab.core.types import BacktestConfig, BarFrame

__all__ = [
    "WalkForwardResult",
    "WindowResult",
    "WindowSearch",
    "stitch_equity",
    "walk_forward",
]

log = get_logger(__name__)

_EPS: Final[float] = 1e-12


@dataclass(frozen=True, slots=True)
class WindowSearch:
    """What a search over one in-sample window produced.

    ``n_evaluations`` is what the window cost the multiple-testing budget. It is
    reported by the search rather than counted here, because only the search
    knows whether a repeated parameter set was served from the run cache.
    """

    params: Mapping[str, Any]
    n_evaluations: int = 0
    #: Whether the parameters were searched for or carried in unchanged.
    searched: bool = True


#: Searches one in-sample window. Injected so this layer needs neither Optuna nor
#: the evolution loop (INV-8).
Optimizer = Callable[[BarFrame, int], WindowSearch]

#: Evaluates one parameter set over one slice of bars, through the run cache.
Evaluate = Callable[[BarFrame, Mapping[str, Any], str], Any]


@dataclass(frozen=True, slots=True)
class WindowResult:
    """One window: what was searched in-sample, and what happened out of sample."""

    k: int
    window: WalkForwardWindow
    params: Mapping[str, Any]
    is_metrics: MetricSet
    oos_metrics: MetricSet
    oos_equity: pd.Series
    #: The window's own trades, which the stitched account also took.
    oos_trades: tuple[Any, ...] = ()
    n_evaluations: int = 0
    searched: bool = True

    @property
    def profitable(self) -> bool:
        return (self.oos_metrics.net_return or 0.0) > 0.0


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """The whole walk-forward (spec section 15, steps 3-5)."""

    windows: tuple[WindowResult, ...] = ()
    stitched_equity: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    stitched: MetricSet = field(default_factory=MetricSet)
    #: ``cagr_oos_stitched / mean_k(cagr_is_k)``; ``None`` when the denominator
    #: is not positive, where the ratio would report a number about nothing.
    wfe: float | None = None
    profitable_oos_share: float = 0.0
    #: Coefficient of variation of each parameter across windows (section 15.4).
    param_cv: Mapping[str, float] = field(default_factory=dict)
    n_trials_accounted: int = 0
    reoptimized: bool = True

    @property
    def evidence(self) -> str:
        """How strong the result is, said plainly (section 15.2).

        A walk-forward that reused fixed parameters never tested whether the
        search generalises, and a report must not present it as though it had.
        """
        return (
            "re-searched each window"
            if self.reoptimized
            else "reused fixed parameters; this does not test whether the search generalises"
        )

    def as_document(self) -> dict[str, Any]:
        """The ``walkforward.json`` artifact of section 15.5."""
        return {
            "n_windows": len(self.windows),
            "wfe": self.wfe,
            "profitable_oos_share": self.profitable_oos_share,
            "param_cv": dict(self.param_cv),
            "n_trials_accounted": self.n_trials_accounted,
            "reoptimized": self.reoptimized,
            "evidence": self.evidence,
            "metrics_wf_oos": self.stitched.as_dict(),
            "windows": [
                {
                    "k": result.k,
                    "params": dict(result.params),
                    "is_start_ts": result.window.is_start_ts,
                    "is_end_ts": result.window.is_end_ts,
                    "oos_start_ts": result.window.oos_start_ts,
                    "oos_end_ts": result.window.oos_end_ts,
                    "is_cagr": result.is_metrics.cagr,
                    "oos_cagr": result.oos_metrics.cagr,
                    "oos_net_return": result.oos_metrics.net_return,
                    "n_evaluations": result.n_evaluations,
                }
                for result in self.windows
            ],
        }


def stitch_equity(curves: Sequence[pd.Series]) -> pd.Series:
    """Chain out-of-sample equity curves end to end (spec section 15.3).

    ``E_{k+1,0} = E_{k,end}``: each window's *returns* are applied to where the
    previous window finished, so the stitched curve is what an account that
    followed the procedure through time would have done.

    Chaining returns rather than concatenating levels is the whole point. Each
    window's backtest starts from the same initial equity, so concatenating the
    levels would reset the account at every boundary and hide every compounding
    effect — including the one that matters, a drawdown that carries across a
    window edge.
    """
    chained: list[pd.Series] = []
    level = None
    for curve in curves:
        values = curve.astype("float64")
        if values.empty:
            continue
        if level is None:
            chained.append(values)
            level = float(values.iloc[-1])
            continue
        start = float(values.iloc[0])
        if abs(start) <= _EPS:  # pragma: no cover - equity of zero is ruin (section 8.7)
            continue
        scaled = values * (level / start)
        # Drop the first point: it is the previous window's last, by construction.
        chained.append(scaled.iloc[1:])
        level = float(scaled.iloc[-1])

    if not chained:
        return pd.Series(dtype="float64")
    stitched: pd.Series = pd.concat(chained)
    return stitched


def _cv(values: Sequence[float]) -> float | None:
    """Coefficient of variation, or ``None`` where it would be unbounded."""
    numbers = [float(value) for value in values]
    if len(numbers) < 2:
        return None
    mean = statistics.fmean(numbers)
    if abs(mean) <= _EPS:
        return None
    return statistics.stdev(numbers) / abs(mean)


def parameter_stability(windows: Sequence[WindowResult]) -> dict[str, float]:
    """Per-parameter coefficient of variation across windows (section 15.4).

    A parameter that had to be re-chosen wildly from window to window is one the
    data does not determine; section 14.4's soft check 6 charges for it. Only
    numeric parameters are measured — a categorical has no coefficient of
    variation — and a parameter with an undefined one is omitted rather than
    reported as zero.
    """
    by_name: dict[str, list[float]] = {}
    for result in windows:
        for name, value in result.params.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            by_name.setdefault(name, []).append(float(value))

    stability: dict[str, float] = {}
    for name, values in sorted(by_name.items()):
        measured = _cv(values)
        if measured is not None:
            stability[name] = measured
    return stability


def walk_forward(
    bars: BarFrame,
    policy: SplitPolicy,
    settings: WalkForwardSettings,
    *,
    optimizer: Optimizer,
    evaluate: Evaluate,
    config: BacktestConfig,
) -> WalkForwardResult:
    """Run the walk-forward of section 15.

    Args:
        bars: Train and validation bars. The test partition is not here and never
            is: a walk-forward spans train and validation only (section 15.1),
            and the container's guard would refuse the rest anyway (INV-5).
        policy: The split the windows are generated over.
        settings: ``walkforward``.
        optimizer: Searches one in-sample window and returns the parameters to
            carry into the out-of-sample one. Injected because both section
            13.8's study and section 13.2's evolution are valid searches, and
            INV-8 forbids this layer from importing the latter.
        evaluate: Runs one parameter set over one slice, under a named segment.
            Every call goes through the run cache of section 11.2.
        config: The backtest settings every window is evaluated under.

    Returns:
        A :class:`WalkForwardResult` carrying the stitched out-of-sample curve,
        the walk-forward efficiency, and what the whole thing cost the
        multiple-testing budget.

    Raises:
        ConfigError: the timeline is too short for one window. Reported here
            rather than as an empty result, which a caller would read as "the
            walk-forward found nothing" instead of "it never ran".
    """
    windows = walk_forward_windows(
        policy,
        is_bars=settings.is_bars,
        oos_bars=settings.oos_bars,
        step_bars=settings.step_bars,
        scheme=settings.scheme,
    )
    if not windows:  # pragma: no cover - the generator raises first
        raise ConfigError("no walk-forward window fits this split")

    results: list[WindowResult] = []
    evaluations = 0
    for window in windows:
        in_sample = bars.slice(window.is_start_ts, window.is_end_ts)
        out_of_sample = bars.slice(window.oos_start_ts, window.oos_end_ts)

        search = optimizer(in_sample, window.k)
        evaluations += max(0, int(search.n_evaluations))

        is_result = evaluate(in_sample, search.params, f"wf_is:{window.k}")
        oos_result = evaluate(out_of_sample, search.params, f"wf_oos:{window.k}")
        results.append(
            WindowResult(
                k=window.k,
                window=window,
                params=dict(search.params),
                is_metrics=compute_metrics(is_result),
                oos_metrics=compute_metrics(oos_result),
                oos_equity=oos_result.equity,
                oos_trades=tuple(oos_result.trades),
                n_evaluations=search.n_evaluations,
                searched=search.searched,
            )
        )
        log.info(
            "walkforward_window",
            k=window.k,
            is_bars=in_sample.n_bars,
            oos_bars=out_of_sample.n_bars,
            n_evaluations=search.n_evaluations,
        )

    return _summarise(results, config, reoptimized=settings.reoptimize_each_window)


def _summarise(
    results: Sequence[WindowResult], config: BacktestConfig, *, reoptimized: bool
) -> WalkForwardResult:
    """Stitch, score, and compute the walk-forward efficiency (section 15.3-15.4)."""
    stitched_equity = stitch_equity([result.oos_equity for result in results])
    stitched = _stitched_metrics(stitched_equity, results, config)

    is_cagrs = [result.is_metrics.cagr for result in results if result.is_metrics.cagr is not None]
    mean_is = statistics.fmean(is_cagrs) if is_cagrs else 0.0
    wfe = None
    if mean_is > _EPS and stitched.cagr is not None:
        wfe = stitched.cagr / mean_is

    profitable = sum(1 for result in results if result.profitable)
    return WalkForwardResult(
        windows=tuple(results),
        stitched_equity=stitched_equity,
        stitched=stitched,
        wfe=wfe,
        profitable_oos_share=profitable / len(results) if results else 0.0,
        param_cv=parameter_stability(results),
        n_trials_accounted=sum(result.n_evaluations for result in results),
        reoptimized=reoptimized,
    )


def _stitched_metrics(
    equity: pd.Series, results: Sequence[WindowResult], config: BacktestConfig
) -> MetricSet:
    """Metrics on the stitched curve (``metrics_wf_oos``, section 15.3).

    Built from a synthetic :class:`~quantlab.core.types.BacktestResult` so that
    the one metric implementation of section 10 is used, rather than a second one
    that computes "walk-forward Sharpe" slightly differently. Trades are the
    concatenation of every window's, renumbered, because a trade belongs to
    exactly one window and the stitched account took all of them.
    """
    from quantlab.core.types import BacktestResult, Trade

    if equity.empty:
        return MetricSet()

    from dataclasses import replace as dc_replace

    trades: list[Trade] = []
    for result in results:
        for trade in result.oos_trades:
            trades.append(dc_replace(trade, trade_no=len(trades) + 1))

    index = equity.index
    zeros = pd.Series(np.zeros(len(equity), dtype="float64"), index=index)
    synthetic = BacktestResult(
        equity=equity,
        position_frac=zeros,
        signals=pd.Series(["flat"] * len(equity), index=index),
        trades=tuple(trades),
        n_bars=len(equity),
        bars_per_year=config.bars_per_year,
        engine_name="walkforward",
        engine_version="1",
    )
    return compute_metrics(synthetic)
