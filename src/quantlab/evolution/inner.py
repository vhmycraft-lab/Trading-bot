"""The inner walk-forward (master spec section 13.6).

Fitness has to reward generalisation, and the obvious way — measure it on the
validation segment — is exactly what section 13.7 forbids: sixteen candidates a
generation for thirty generations would spend the validation set four hundred and
eighty times, and after that it measures nothing.

So the train segment is split into its own in-sample/out-of-sample folds, with
their own embargo, entirely inside train. A candidate's ``inner_oos`` fitness
component, its ``p_instability`` penalty and its ``p_divergence`` penalty all come
from here, and none of them costs a look at data the platform is protecting.

The folds are a property of the *bars*, not of the candidate, so they are computed
once per generation and reused. That matters: with four folds this doubles the
evaluations a generation costs, and computing the split per candidate would make
it worse for no gain.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from quantlab.core.config import InnerWalkForwardSettings
from quantlab.core.errors import ConfigError
from quantlab.core.fitness import InnerFold, InnerFoldReport
from quantlab.core.metrics import compute_metrics
from quantlab.core.types import BacktestConfig, BacktestResult, BarFrame

__all__ = ["InnerWindow", "inner_report", "inner_windows"]

#: Evaluates one candidate over one slice of bars. Injected, so this module needs
#: no engine and no sandbox of its own.
Evaluate = Callable[[BarFrame, Mapping[str, Any], BacktestConfig], BacktestResult]


@dataclass(frozen=True, slots=True)
class InnerWindow:
    """One in-sample/out-of-sample pair, by bar index into the train segment."""

    index: int
    is_start: int
    is_end: int
    oos_start: int
    oos_end: int

    @property
    def is_bars(self) -> int:
        return self.is_end - self.is_start

    @property
    def oos_bars(self) -> int:
        return self.oos_end - self.oos_start


def inner_windows(
    n_bars: int, settings: InnerWalkForwardSettings, *, warmup_bars: int = 0
) -> list[InnerWindow]:
    """Split ``n_bars`` of train data into folds (spec section 13.6).

    ``rolling`` gives each fold its own in-sample stretch; ``anchored`` grows the
    in-sample from the start of train. Both leave ``embargo_bars`` between the
    in-sample end and the out-of-sample start, for the same reason the outer split
    does (section 7.3): a position open at the boundary would otherwise be closed
    on data the in-sample fit already saw.

    Every out-of-sample window is required to hold at least ``warmup_bars`` plus
    one bar. A window shorter than the candidate's own warm-up would produce no
    signals at all, and a fold reporting "no trades" for that reason would be
    read as a strategy that declined to trade.

    Raises:
        ConfigError: the segment cannot be divided into folds of usable length.
            Better here than as an inner-OOS component that is silently always
            zero for every candidate in the run.
    """
    folds = settings.n_folds
    embargo = settings.embargo_bars
    minimum_oos = max(1, warmup_bars + 1)

    # One block per fold plus the first in-sample stretch: the earliest fold needs
    # in-sample data before it, so `folds + 1` blocks divide the segment.
    block = (n_bars - folds * embargo) // (folds + 1)
    if block < minimum_oos:
        raise ConfigError(
            "the train segment is too short to split into inner folds; the fitness "
            "component that measures generalisation would be zero for every candidate",
            n_bars=n_bars,
            n_folds=folds,
            embargo_bars=embargo,
            needed_per_fold=minimum_oos,
        )

    windows: list[InnerWindow] = []
    for index in range(folds):
        is_end = block * (index + 1)
        oos_start = is_end + embargo
        oos_end = min(oos_start + block, n_bars)
        if oos_end - oos_start < minimum_oos:
            break
        windows.append(
            InnerWindow(
                index=index,
                is_start=0 if settings.scheme == "anchored" else block * index,
                is_end=is_end,
                oos_start=oos_start,
                oos_end=oos_end,
            )
        )
    if not windows:  # pragma: no cover - the block check above already caught this
        raise ConfigError("no usable inner fold", n_bars=n_bars, n_folds=folds)
    return windows


def inner_report(
    evaluate: Evaluate,
    bars: BarFrame,
    params: Mapping[str, Any],
    config: BacktestConfig,
    windows: Sequence[InnerWindow],
) -> InnerFoldReport:
    """Evaluate a candidate on each fold and summarise (spec section 13.6).

    Both sides of every fold are evaluated: the in-sample Sortino is what the
    out-of-sample one is compared against, and a report carrying only the
    out-of-sample half could not tell a candidate that generalised from one that
    was mediocre everywhere.

    A fold whose evaluation fails contributes ``None`` rather than a zero. It is a
    fold that was not measured, and section 13.3's components already treat an
    unmeasured input as undemonstrated without also penalising it.
    """
    folds: list[InnerFold] = []
    for window in windows:
        folds.append(
            InnerFold(
                is_sortino=_sortino(evaluate, bars, params, config, window.is_start, window.is_end),
                oos_sortino=_sortino(
                    evaluate, bars, params, config, window.oos_start, window.oos_end
                ),
                oos_return=_net_return(
                    evaluate, bars, params, config, window.oos_start, window.oos_end
                ),
            )
        )
    return InnerFoldReport(folds=tuple(folds))


def _window_bars(bars: BarFrame, start: int, end: int) -> BarFrame:
    """The bars at positions ``[start, end)``.

    ``BarFrame.slice`` takes timestamps rather than positions, which is the right
    interface for a segment boundary but not for a fold index. Converting here
    keeps the fold arithmetic in one place instead of spreading timestamps
    through it.
    """
    timestamps = bars.ts_open
    if start >= end or start >= len(timestamps):
        return bars.slice(int(timestamps[0]), int(timestamps[0]) - 1)
    last = min(end, len(timestamps)) - 1
    return bars.slice(int(timestamps[start]), int(timestamps[last]))


def _evaluate_slice(
    evaluate: Evaluate,
    bars: BarFrame,
    params: Mapping[str, Any],
    config: BacktestConfig,
    start: int,
    end: int,
) -> Any:
    """Metrics for one fold, or ``None`` if it could not be measured.

    A broad catch, deliberately: a fold that raised is one this candidate could
    not be scored on, and section 13.3 already treats an unmeasured input as
    undemonstrated. Letting it propagate would end the generation over one
    candidate's bad slice.
    """
    try:
        return compute_metrics(evaluate(_window_bars(bars, start, end), params, config))
    except Exception:
        return None


def _sortino(
    evaluate: Evaluate,
    bars: BarFrame,
    params: Mapping[str, Any],
    config: BacktestConfig,
    start: int,
    end: int,
) -> float | None:
    metrics = _evaluate_slice(evaluate, bars, params, config, start, end)
    return None if metrics is None else metrics.sortino


def _net_return(
    evaluate: Evaluate,
    bars: BarFrame,
    params: Mapping[str, Any],
    config: BacktestConfig,
    start: int,
    end: int,
) -> float | None:
    metrics = _evaluate_slice(evaluate, bars, params, config, start, end)
    return None if metrics is None else metrics.net_return
