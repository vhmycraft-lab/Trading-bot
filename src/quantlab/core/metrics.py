"""Performance metrics (master spec section 10).

Every metric is computed from the **equity series and the trade list only**.
Nothing here re-reads prices (buy-and-hold is the one documented exception), so
any engine adapter that produces a correct equity curve gets identical metric
semantics, and a metric can never disagree with the accounting that produced it.

``None`` means *undefined*, not zero. A profit factor with no losing trades, a
Sharpe ratio with zero variance and a CAGR over three days are all genuinely
undefined, and reporting a number for them invites a comparison that means
nothing.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict

from quantlab.core.types import BacktestResult, Trade

__all__ = [
    "DEFAULT_RETENTION_K",
    "LOW_TRADE_THRESHOLD",
    "MetricSet",
    "compute_metrics",
    "drawdown_series",
    "retention_after_removing_top_winners",
    "trade_expectancy_pct",
    "trade_profit_factor",
    "trade_win_rate",
]

#: Fewer trades than this and every statistic below is decoration, not evidence.
LOW_TRADE_THRESHOLD: Final[int] = 30

#: Trade-removal points reported on every run (spec section 14.4).
DEFAULT_RETENTION_K: Final[tuple[int, ...]] = (1, 3, 5)

#: Bars per consistency window: 30 days of hourly bars.
DEFAULT_CONSISTENCY_BARS: Final[int] = 720

_EPS: Final[float] = 1e-12
_Z95: Final[float] = 1.959963984540054

#: Dispersion below this fraction of the return RMS is float noise, not risk.
#:
#: The guard it replaces was an absolute ``1e-12`` applied to a scale-dependent
#: quantity: per-bar returns are of order 1e-3, so almost any downside at all
#: cleared it. A single bar dipping by 1e-6 in an otherwise rising curve produced
#: a Sortino of 2.1e10 — which then clipped to a *perfect* risk-adjusted score,
#: while the same curve with no dip at all scored zero. Adding an economically
#: meaningless loss raised fitness by 31 %, so the search was paid to inject one.
#:
#: Comparing against the RMS of the same series makes the test scale-invariant:
#: it asks whether the dispersion is measurable *at the scale of the returns*,
#: which is the question, rather than whether it exceeds a fixed constant.
#:
#: Sharpe carried the identical defect and it mattered more, because Sharpe is
#: what the deflated Sharpe ratio deflates. An equity curve drifting 1e-6 per bar
#: — annualised volatility 5.4e-08, flat to within float noise — produced a
#: Sharpe of 162,313, and ``deflated_sharpe`` then reported **1.0 even at 1000
#: trials**. The one defence against multiple testing returned maximum
#: confidence for a curve that does nothing.
_MIN_DISPERSION_FRACTION: Final[float] = 1e-3


class MetricSet(BaseModel):
    """The metric vector for one run.  Every field is ``float | None``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    net_return: float | None = None
    cagr: float | None = None
    max_drawdown: float | None = None
    max_drawdown_bars: float | None = None
    profit_factor: float | None = None
    sharpe: float | None = None
    sortino: float | None = None
    win_rate: float | None = None
    n_trades: float = 0.0
    avg_trade_pct: float | None = None
    avg_trade_usdt: float | None = None
    expectancy_pct: float | None = None
    exposure: float | None = None
    calmar: float | None = None
    ann_volatility: float | None = None
    avg_bars_held: float | None = None
    longest_losing_streak: float | None = None
    top5_profit_share: float | None = None
    turnover: float | None = None
    buy_hold_return: float | None = None
    excess_vs_buy_hold: float | None = None
    sharpe_ci_low: float | None = None
    sharpe_ci_high: float | None = None
    #: Share of consistency windows that finished in profit (spec section 13.3).
    consistency: float | None = None
    #: Gross-profit retention after removing the top 1 / 3 / 5 winning trades.
    retention_1: float | None = None
    retention_3: float | None = None
    retention_5: float | None = None
    low_trade_warning: bool = False

    def as_dict(self) -> dict[str, float | bool | None]:
        """Flat mapping, for storage in the ``metric`` table."""
        return dict(self.model_dump())


def drawdown_series(equity: np.ndarray) -> np.ndarray:
    """Fractional drawdown from the running peak, per bar, in ``[0, 1]``.

    A blown-up short can leave equity below zero, where the raw formula exceeds
    1. Spec section 10 defines drawdown as a fraction in ``[0, 1]``, so the value
    is clamped: losing everything is 100 %, and the fact that the account also
    owes money is reported by ``BacktestResult.ruined`` and by the equity series
    itself rather than by a drawdown of 200 %.
    """
    values = np.asarray(equity, dtype="float64")
    if values.size == 0:
        return values
    peak = np.maximum.accumulate(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.where(np.abs(peak) > _EPS, 1.0 - values / peak, 0.0)
    return np.clip(raw, 0.0, 1.0)


def _max_drawdown_bars(equity: np.ndarray) -> int:
    """Longest peak-to-recovery span, in bars (spec section 10).

    Measured from the peak to the bar that regains it, so a dip lasting one bar
    and recovering on the next spans two.  A curve that only rises has spans of
    zero, not of one: there was no drawdown to wait out.  An unrecovered
    drawdown counts to the end of the series, because "never recovered" is the
    longest wait there is, not an absent one.
    """
    values = np.asarray(equity, dtype="float64")
    if values.size < 2:
        return 0

    longest = 0
    peak_value = float(values[0])
    peak_index = 0
    under_water = False
    for i in range(1, values.size):
        value = float(values[i])
        if value < peak_value:
            under_water = True
            continue
        if under_water:
            longest = max(longest, i - peak_index)
        peak_value = value
        peak_index = i
        under_water = False
    if under_water:
        longest = max(longest, values.size - 1 - peak_index)
    return longest


def trade_win_rate(trades: Sequence[Trade]) -> float | None:
    """Share of round trips that closed in profit; ``None`` with no trades."""
    if not trades:
        return None
    wins = sum(1 for trade in trades if trade.pnl > 0)
    return wins / len(trades)


def trade_profit_factor(trades: Sequence[Trade]) -> float | None:
    """Gross profit over gross loss (spec section 10).

    ``None`` when nothing was lost: the ratio is unbounded there, and reporting a
    huge number would read as a strong result rather than as an absent
    denominator.  Callers that score it treat ``None`` as perfect *and* raise the
    low-trade warning, which is what section 13.3 requires.
    """
    if not trades:
        return None
    pnls = np.array([trade.pnl for trade in trades], dtype="float64")
    gross_loss = float(np.abs(pnls[pnls < 0].sum()))
    if gross_loss <= _EPS:
        return None
    return float(pnls[pnls > 0].sum()) / gross_loss


def trade_expectancy_pct(trades: Sequence[Trade]) -> float | None:
    """Expected return per trade as a fraction (spec section 10).

    ``win_rate * mean win% - (1 - win_rate) * mean |loss%|``.  Extracted so that
    the trade-removal test of section 14.4 recomputes it with the same rule
    rather than with a second one that could drift.
    """
    if not trades:
        return None
    rate = trade_win_rate(trades)
    if rate is None:  # pragma: no cover - guarded by the emptiness check above
        return None
    pnls = np.array([trade.pnl for trade in trades], dtype="float64")
    pnl_pcts = np.array([trade.pnl_pct for trade in trades], dtype="float64")
    win_pcts = pnl_pcts[pnls > 0]
    loss_pcts = pnl_pcts[pnls < 0]
    mean_win = float(win_pcts.mean()) if win_pcts.size else 0.0
    mean_loss = float(np.abs(loss_pcts.mean())) if loss_pcts.size else 0.0
    return rate * mean_win - (1.0 - rate) * mean_loss


def retention_after_removing_top_winners(
    trades: Sequence[Trade], k_values: Sequence[int] = DEFAULT_RETENTION_K
) -> dict[int, float | None]:
    """Gross-profit retention after removing the ``k`` largest winning trades.

    ``retention_k = sum(pnl of remaining) / sum(pnl of all)``, computed on the
    trade list exactly and without re-simulation.  A strategy whose retention
    collapses has not shown an edge; it has shown that a few things happened.

    ``None`` when total pnl is not positive, where the ratio would be
    uninterpretable (dividing by a loss flips its sign).
    """
    pnls = np.array([trade.pnl for trade in trades], dtype="float64")
    total = float(pnls.sum())
    if pnls.size == 0 or total <= _EPS:
        return dict.fromkeys(k_values)

    # Only *winners* are removed: dropping the k largest of a list that has no
    # k winners must not start removing the least-bad losses.
    winners = np.sort(pnls[pnls > 0])[::-1]
    out: dict[int, float | None] = {}
    for k in k_values:
        removed = float(winners[: max(0, int(k))].sum())
        out[int(k)] = (total - removed) / total
    return out


def _consistency(equity: np.ndarray, window: int) -> float | None:
    """Share of non-overlapping windows of ``window`` bars that ended in profit."""
    values = np.asarray(equity, dtype="float64")
    if values.size < 2 or window < 1:
        return None
    n_windows = (values.size - 1) // window
    if n_windows < 1:
        return None
    profitable = 0
    for w in range(n_windows):
        start = values[w * window]
        end = values[min((w + 1) * window, values.size - 1)]
        if abs(start) > _EPS and end > start:
            profitable += 1
    return profitable / n_windows


def _sharpe_ci(sharpe: float | None, n_periods: int) -> tuple[float | None, float | None]:
    """Lo (2002) asymptotic 95 % interval, annualised consistently with the estimate."""
    if sharpe is None or n_periods < 2:
        return None, None
    half_width = _Z95 * math.sqrt((1.0 + 0.5 * sharpe * sharpe) / n_periods)
    return sharpe - half_width, sharpe + half_width


def _longest_losing_streak(trades: Sequence[Trade]) -> int:
    longest = current = 0
    for trade in trades:
        current = current + 1 if trade.pnl < 0 else 0
        longest = max(longest, current)
    return longest


def compute_metrics(
    result: BacktestResult,
    *,
    rf_annual: float = 0.0,
    consistency_bars: int = DEFAULT_CONSISTENCY_BARS,
    buy_hold_return: float | None = None,
) -> MetricSet:
    """Compute the metric vector for a finished run (spec section 10).

    Returns are measured on bars **after** the warm-up; ``T`` is the number of
    such bars and ``Y = T / bars_per_year``.

    Args:
        result: A finished backtest.
        rf_annual: Annual risk-free rate; the spec fixes this at 0.
        consistency_bars: Window length for the consistency metric.
        buy_hold_return: Supplied by the caller, which is the only metric that
            needs prices.  ``None`` leaves it and the excess undefined.
    """
    equity_full = result.equity.to_numpy(dtype="float64")
    warmup = min(int(result.warmup_bars), max(0, equity_full.size - 1))
    equity = equity_full[warmup:]
    trades = list(result.trades)
    n_trades = len(trades)

    if equity.size == 0:
        return MetricSet(n_trades=float(n_trades), low_trade_warning=True)

    start, end = float(equity[0]), float(equity[-1])
    net_return = (end / start - 1.0) if abs(start) > _EPS else None

    t_bars = max(0, equity.size - 1)
    years = t_bars / float(result.bars_per_year) if result.bars_per_year else 0.0
    cagr: float | None = None
    if net_return is not None and years >= 1.0 / 365.0 and start > _EPS and end > _EPS:
        cagr = (end / start) ** (1.0 / years) - 1.0

    drawdowns = drawdown_series(equity)
    max_drawdown = float(np.max(drawdowns)) if drawdowns.size else None

    # -- per-bar returns ---------------------------------------------------
    returns = np.diff(equity) / np.where(np.abs(equity[:-1]) > _EPS, equity[:-1], np.nan)
    returns = returns[np.isfinite(returns)]
    rf_bar = rf_annual / float(result.bars_per_year) if result.bars_per_year else 0.0
    excess = returns - rf_bar

    # The scale every dispersion test below is measured against.
    scale = float(math.sqrt(float(np.mean(excess**2)))) if excess.size else 0.0

    sharpe: float | None = None
    ann_volatility: float | None = None
    if excess.size >= 2:
        deviation = float(np.std(excess, ddof=1))
        ann_volatility = deviation * math.sqrt(result.bars_per_year)
        # Relative, not absolute: see _MIN_DISPERSION_FRACTION. A curve whose
        # returns barely vary has no measurable Sharpe, and reporting the ratio
        # anyway hands the deflated Sharpe an unbounded input.
        if deviation > max(_EPS, _MIN_DISPERSION_FRACTION * scale):
            sharpe = float(np.mean(excess)) / deviation * math.sqrt(result.bars_per_year)

    sortino: float | None = None
    if excess.size >= 1:
        downside = np.minimum(excess, 0.0)
        downside_deviation = float(math.sqrt(float(np.mean(downside**2))))
        # Relative, not absolute: see _MIN_DISPERSION_FRACTION. A downside that
        # is negligible against the returns' own scale is not a small risk, it is
        # an unmeasured one, and it is reported as undefined rather than as a
        # ratio whose size is an artefact of the one bar in the denominator.
        if downside_deviation > max(_EPS, _MIN_DISPERSION_FRACTION * scale):
            sortino = float(np.mean(excess)) / downside_deviation * math.sqrt(result.bars_per_year)

    sharpe_low, sharpe_high = _sharpe_ci(sharpe, int(excess.size))

    calmar: float | None = None
    if cagr is not None and max_drawdown is not None and max_drawdown > _EPS:
        calmar = cagr / max_drawdown

    # -- trade statistics --------------------------------------------------
    pnls = np.array([trade.pnl for trade in trades], dtype="float64")
    pnl_pcts = np.array([trade.pnl_pct for trade in trades], dtype="float64")
    wins = pnls[pnls > 0]

    win_rate = trade_win_rate(trades)
    profit_factor = trade_profit_factor(trades)
    expectancy_pct = trade_expectancy_pct(trades)

    top5_share: float | None = None
    if n_trades and float(wins.sum()) > _EPS:
        # Summing the same floats in two orders can differ in the last bit, so a
        # share of "all the winners" can land a hair above 1.0. Clamp: a share
        # outside [0, 1] is a rounding artefact, and downstream code that treats
        # it as a probability should never have to know that.
        top5_share = min(1.0, max(0.0, float(np.sort(wins)[::-1][:5].sum()) / float(wins.sum())))

    retention = retention_after_removing_top_winners(trades)

    exposure: float | None = None
    fractions = result.position_frac.to_numpy(dtype="float64")[warmup:]
    if fractions.size:
        exposure = float(np.count_nonzero(np.abs(fractions) > _EPS) / fractions.size)

    excess_vs_buy_hold = (
        net_return - buy_hold_return
        if net_return is not None and buy_hold_return is not None
        else None
    )

    return MetricSet(
        net_return=net_return,
        cagr=cagr,
        max_drawdown=max_drawdown,
        max_drawdown_bars=float(_max_drawdown_bars(equity)),
        profit_factor=profit_factor,
        sharpe=sharpe,
        sortino=sortino,
        win_rate=win_rate,
        n_trades=float(n_trades),
        avg_trade_pct=float(pnl_pcts.mean()) if n_trades else None,
        avg_trade_usdt=float(pnls.mean()) if n_trades else None,
        expectancy_pct=expectancy_pct,
        exposure=exposure,
        calmar=calmar,
        ann_volatility=ann_volatility,
        avg_bars_held=(float(np.mean([trade.bars_held for trade in trades])) if n_trades else None),
        longest_losing_streak=float(_longest_losing_streak(trades)),
        top5_profit_share=top5_share,
        turnover=float(result.cost_summary.get("turnover", 0.0)),
        buy_hold_return=buy_hold_return,
        excess_vs_buy_hold=excess_vs_buy_hold,
        sharpe_ci_low=sharpe_low,
        sharpe_ci_high=sharpe_high,
        consistency=_consistency(equity, consistency_bars),
        retention_1=retention.get(1),
        retention_3=retention.get(3),
        retention_5=retention.get(5),
        low_trade_warning=n_trades < LOW_TRADE_THRESHOLD or profit_factor is None,
    )
