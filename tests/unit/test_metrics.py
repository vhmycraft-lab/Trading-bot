"""Performance metrics (master spec section 10).

The core of this file is a hand-built equity curve and trade list whose every
metric can be worked out on paper. Anything computed from a real backtest would
only be testing the engine twice.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quantlab.core.metrics import (
    LOW_TRADE_THRESHOLD,
    MetricSet,
    compute_metrics,
    drawdown_series,
    retention_after_removing_top_winners,
)
from quantlab.core.types import BacktestResult, Side, Trade

HOUR = 3_600_000


def make_trade(no: int, pnl: float, *, pnl_pct: float | None = None, bars: int = 5) -> Trade:
    return Trade(
        trade_no=no,
        side=Side.LONG,
        entry_ts=no * HOUR,
        entry_px=100.0,
        exit_ts=(no + bars) * HOUR,
        exit_px=100.0 + pnl / 10.0,
        qty=10.0,
        fees=0.0,
        slippage_cost=0.0,
        pnl=pnl,
        pnl_pct=pnl / 10_000.0 if pnl_pct is None else pnl_pct,
        bars_held=bars,
        exit_reason="signal",
    )


def make_result(
    equity: list[float],
    trades: list[Trade] | None = None,
    *,
    warmup: int = 0,
    bars_per_year: int = 8_760,
    position_frac: list[float] | None = None,
) -> BacktestResult:
    index = pd.Index([i * HOUR for i in range(len(equity))], name="ts_open", dtype="int64")
    fractions = position_frac if position_frac is not None else [1.0] * len(equity)
    return BacktestResult(
        equity=pd.Series(equity, index=index, dtype="float64"),
        position_frac=pd.Series(fractions, index=index, dtype="float64"),
        signals=pd.Series(["long"] * len(equity), index=index, dtype="object"),
        trades=tuple(trades or ()),
        warmup_bars=warmup,
        n_bars=len(equity),
        bars_per_year=bars_per_year,
        cost_summary={"turnover": 1234.0},
    )


# ---------------------------------------------------------------------------
# the hand-computed fixture
# ---------------------------------------------------------------------------
HAND_EQUITY = [10_000.0, 11_000.0, 9_900.0, 12_000.0, 12_000.0]
HAND_TRADES = [
    make_trade(1, 1_000.0),
    make_trade(2, -1_100.0),
    make_trade(3, 2_100.0),
]


@pytest.fixture
def hand() -> MetricSet:
    return compute_metrics(make_result(HAND_EQUITY, HAND_TRADES))


def test_net_return(hand: MetricSet) -> None:
    assert hand.net_return == pytest.approx(0.2)  # 12000 / 10000 - 1


def test_max_drawdown(hand: MetricSet) -> None:
    assert hand.max_drawdown == pytest.approx(1.0 - 9_900.0 / 11_000.0)  # 0.10


def test_max_drawdown_bars(hand: MetricSet) -> None:
    """Peak at bar 1, regained at bar 3: a two-bar peak-to-recovery span."""
    assert hand.max_drawdown_bars == 2.0


def test_a_monotonically_rising_curve_has_no_drawdown_span() -> None:
    metrics = compute_metrics(make_result([100.0, 110.0, 120.0, 130.0]))
    assert metrics.max_drawdown_bars == 0.0
    assert metrics.max_drawdown == pytest.approx(0.0)


def test_trade_counts_and_win_rate(hand: MetricSet) -> None:
    assert hand.n_trades == 3.0
    assert hand.win_rate == pytest.approx(2.0 / 3.0)
    assert hand.longest_losing_streak == 1.0


def test_profit_factor(hand: MetricSet) -> None:
    assert hand.profit_factor == pytest.approx(3_100.0 / 1_100.0)


def test_average_trade(hand: MetricSet) -> None:
    assert hand.avg_trade_usdt == pytest.approx(2_000.0 / 3.0)
    assert hand.avg_trade_pct == pytest.approx((0.1 - 0.11 + 0.21) / 3.0)


def test_expectancy(hand: MetricSet) -> None:
    win_rate = 2.0 / 3.0
    mean_win = (0.1 + 0.21) / 2.0
    mean_loss = 0.11
    assert hand.expectancy_pct == pytest.approx(win_rate * mean_win - (1 - win_rate) * mean_loss)


def test_per_bar_return_statistics(hand: MetricSet) -> None:
    returns = np.diff(HAND_EQUITY) / np.array(HAND_EQUITY[:-1])
    assert hand.ann_volatility == pytest.approx(np.std(returns, ddof=1) * math.sqrt(8_760))
    expected_sharpe = np.mean(returns) / np.std(returns, ddof=1) * math.sqrt(8_760)
    assert hand.sharpe == pytest.approx(expected_sharpe)


def test_sortino_uses_downside_deviation(hand: MetricSet) -> None:
    returns = np.diff(HAND_EQUITY) / np.array(HAND_EQUITY[:-1])
    downside = math.sqrt(float(np.mean(np.minimum(returns, 0.0) ** 2)))
    assert hand.sortino == pytest.approx(np.mean(returns) / downside * math.sqrt(8_760))


def test_sharpe_confidence_interval_is_the_lo_2002_form(hand: MetricSet) -> None:
    n = len(HAND_EQUITY) - 1
    half = 1.959963984540054 * math.sqrt((1 + 0.5 * hand.sharpe**2) / n)
    assert hand.sharpe_ci_low == pytest.approx(hand.sharpe - half)
    assert hand.sharpe_ci_high == pytest.approx(hand.sharpe + half)


def test_exposure(hand: MetricSet) -> None:
    assert hand.exposure == pytest.approx(1.0)


def test_top5_profit_share(hand: MetricSet) -> None:
    """Only two winners, so they are the whole of the gross profit."""
    assert hand.top5_profit_share == pytest.approx(1.0)


def test_turnover_comes_from_the_cost_summary(hand: MetricSet) -> None:
    assert hand.turnover == pytest.approx(1234.0)


def test_low_trade_warning_fires_below_the_threshold(hand: MetricSet) -> None:
    assert LOW_TRADE_THRESHOLD == 30
    assert hand.low_trade_warning is True


# ---------------------------------------------------------------------------
# undefined is None, not zero
# ---------------------------------------------------------------------------
def test_profit_factor_is_none_without_losers() -> None:
    metrics = compute_metrics(make_result([10_000.0, 11_000.0], [make_trade(1, 1_000.0)]))
    assert metrics.profit_factor is None
    assert metrics.low_trade_warning is True


def test_sharpe_is_none_when_equity_never_moves() -> None:
    metrics = compute_metrics(make_result([10_000.0] * 20))
    assert metrics.sharpe is None
    assert metrics.sortino is None
    assert metrics.sharpe_ci_low is None


def test_sortino_is_none_without_any_downside() -> None:
    metrics = compute_metrics(make_result([10_000.0, 10_100.0, 10_200.0, 10_300.0]))
    assert metrics.sortino is None
    assert metrics.sharpe is not None


def test_cagr_is_none_over_too_short_a_window() -> None:
    metrics = compute_metrics(make_result([10_000.0, 10_500.0, 11_000.0]))
    assert metrics.cagr is None


def test_cagr_over_a_full_year() -> None:
    equity = [10_000.0] * 8_760 + [20_000.0]
    metrics = compute_metrics(make_result(equity))
    assert metrics.cagr == pytest.approx(1.0, rel=1e-3)


def test_calmar_is_cagr_over_drawdown() -> None:
    equity = [10_000.0, 9_000.0] + [10_000.0] * 8_759 + [12_000.0]
    metrics = compute_metrics(make_result(equity))
    assert metrics.calmar == pytest.approx(metrics.cagr / metrics.max_drawdown)


def test_no_trades_leaves_trade_metrics_undefined() -> None:
    metrics = compute_metrics(make_result([10_000.0] * 10, []))
    assert metrics.n_trades == 0.0
    assert metrics.win_rate is None
    assert metrics.avg_trade_pct is None
    assert metrics.expectancy_pct is None
    assert metrics.top5_profit_share is None


def test_an_empty_result_is_handled() -> None:
    metrics = compute_metrics(make_result([]))
    assert metrics.n_trades == 0.0
    assert metrics.net_return is None


# ---------------------------------------------------------------------------
# warm-up
# ---------------------------------------------------------------------------
def test_metrics_start_after_the_warm_up() -> None:
    """Equity is flat through the warm-up, so including it would dilute every statistic."""
    equity = [10_000.0] * 5 + [10_000.0, 12_000.0]
    metrics = compute_metrics(make_result(equity, warmup=5))
    assert metrics.net_return == pytest.approx(0.2)
    full = compute_metrics(make_result(equity, warmup=0))
    assert full.net_return == pytest.approx(0.2)
    assert metrics.exposure == 1.0


def test_a_warm_up_covering_everything_is_clamped() -> None:
    metrics = compute_metrics(make_result([10_000.0, 11_000.0], warmup=99))
    assert metrics.n_trades == 0.0


# ---------------------------------------------------------------------------
# drawdown
# ---------------------------------------------------------------------------
def test_drawdown_series_matches_the_definition() -> None:
    equity = np.array([100.0, 120.0, 90.0, 150.0], dtype="float64")
    assert list(drawdown_series(equity)) == pytest.approx([0.0, 0.0, 0.25, 0.0])


def test_drawdown_of_an_empty_series() -> None:
    assert drawdown_series(np.array([], dtype="float64")).size == 0


def test_an_unrecovered_drawdown_counts_to_the_end() -> None:
    metrics = compute_metrics(make_result([10_000.0, 12_000.0, 9_000.0, 8_000.0, 8_500.0]))
    assert metrics.max_drawdown_bars == 3.0
    assert metrics.max_drawdown == pytest.approx(1.0 - 8_000.0 / 12_000.0)


# ---------------------------------------------------------------------------
# the trade-removal test (spec section 14.4)
# ---------------------------------------------------------------------------
def test_retention_with_one_dominant_winner() -> None:
    """900 of the 1 000 profit comes from one trade: retention_1 is 0.1."""
    trades = [make_trade(1, 900.0), make_trade(2, 60.0), make_trade(3, 40.0)]
    retention = retention_after_removing_top_winners(trades)
    assert retention[1] == pytest.approx(0.1)
    assert retention[3] == pytest.approx(0.0)
    assert retention[5] == pytest.approx(0.0)


def test_retention_with_evenly_spread_profit() -> None:
    trades = [make_trade(i, 100.0) for i in range(1, 11)]
    retention = retention_after_removing_top_winners(trades)
    assert retention[1] == pytest.approx(0.9)
    assert retention[3] == pytest.approx(0.7)
    assert retention[5] == pytest.approx(0.5)


def test_retention_can_go_negative() -> None:
    """Removing the best trade can turn a profitable strategy into a losing one."""
    trades = [make_trade(1, 1_000.0), make_trade(2, -400.0), make_trade(3, -500.0)]
    retention = retention_after_removing_top_winners(trades)
    assert retention[1] == pytest.approx(-9.0)
    assert retention[1] < 0


def test_retention_is_non_increasing_in_k() -> None:
    trades = [make_trade(i, float(100 - i)) for i in range(1, 20)]
    retention = retention_after_removing_top_winners(trades)
    assert retention[1] >= retention[3] >= retention[5]


def test_retention_only_removes_winners() -> None:
    """With one winner, k=3 and k=5 must not start removing the least-bad losses."""
    trades = [make_trade(1, 500.0), make_trade(2, -100.0), make_trade(3, -50.0)]
    retention = retention_after_removing_top_winners(trades)
    assert retention[3] == retention[5] == pytest.approx((350.0 - 500.0) / 350.0)


def test_retention_is_none_when_total_pnl_is_not_positive() -> None:
    trades = [make_trade(1, -100.0), make_trade(2, -50.0)]
    assert retention_after_removing_top_winners(trades) == {1: None, 3: None, 5: None}


def test_retention_is_none_without_trades() -> None:
    assert retention_after_removing_top_winners([]) == {1: None, 3: None, 5: None}


def test_retention_appears_on_the_metric_set() -> None:
    trades = [make_trade(1, 900.0), make_trade(2, 60.0), make_trade(3, 40.0)]
    metrics = compute_metrics(make_result([10_000.0, 11_000.0], trades))
    assert metrics.retention_1 == pytest.approx(0.1)
    assert metrics.retention_3 == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# consistency
# ---------------------------------------------------------------------------
def test_consistency_counts_profitable_windows() -> None:
    equity = [100.0, 110.0, 105.0, 115.0, 110.0, 120.0, 115.0]
    metrics = compute_metrics(make_result(equity), consistency_bars=2)
    assert metrics.consistency == pytest.approx(1.0)


def test_consistency_is_zero_when_every_window_loses() -> None:
    # Windows start at bars 0, 2 and 4; each ends lower than it began.
    equity = [100.0, 110.0, 90.0, 80.0, 70.0, 60.0, 50.0]
    metrics = compute_metrics(make_result(equity), consistency_bars=2)
    assert metrics.consistency == pytest.approx(0.0)


def test_consistency_is_the_share_of_winning_windows() -> None:
    # 100 -> 110 wins; 110 -> 90 and 90 -> 70 lose.
    equity = [100.0, 90.0, 110.0, 100.0, 90.0, 80.0, 70.0]
    metrics = compute_metrics(make_result(equity), consistency_bars=2)
    assert metrics.consistency == pytest.approx(1.0 / 3.0)


def test_consistency_is_none_when_no_window_fits() -> None:
    metrics = compute_metrics(make_result([100.0, 110.0]), consistency_bars=720)
    assert metrics.consistency is None


# ---------------------------------------------------------------------------
# exposure and buy-and-hold
# ---------------------------------------------------------------------------
def test_exposure_counts_bars_in_the_market() -> None:
    metrics = compute_metrics(make_result([10_000.0] * 5, position_frac=[1.0, 0.0, 1.0, 0.0, 0.0]))
    assert metrics.exposure == pytest.approx(0.4)


def test_buy_and_hold_is_supplied_by_the_caller() -> None:
    metrics = compute_metrics(make_result(HAND_EQUITY, HAND_TRADES), buy_hold_return=0.05)
    assert metrics.buy_hold_return == pytest.approx(0.05)
    assert metrics.excess_vs_buy_hold == pytest.approx(0.15)


def test_buy_and_hold_is_undefined_when_not_supplied(hand: MetricSet) -> None:
    assert hand.buy_hold_return is None
    assert hand.excess_vs_buy_hold is None


# ---------------------------------------------------------------------------
# the model itself
# ---------------------------------------------------------------------------
def test_the_metric_set_is_frozen(hand: MetricSet) -> None:
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
        hand.net_return = 1.0


def test_as_dict_round_trips(hand: MetricSet) -> None:
    payload = hand.as_dict()
    assert payload["net_return"] == pytest.approx(0.2)
    assert set(payload) == set(MetricSet.model_fields)
