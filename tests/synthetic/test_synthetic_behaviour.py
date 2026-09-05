"""Does the engine behave sensibly on series whose answer we know (spec T16)?

A backtester can be internally consistent and still wrong. These tests feed it
market regimes with a known correct outcome and check it agrees:

* on a **random walk**, no strategy should make money after costs
* on a **trend**, a trend follower should
* on a **mean-reverting** series, a reversion strategy should

Tolerances are stated with their reasoning. They are loose on purpose — the
point is to catch a sign error or a cost that is not being charged, not to
certify a strategy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.helpers import ScriptedStrategy, zero_cost_config

from quantlab.adapters.engine import SimpleBarEngine
from quantlab.core.metrics import compute_metrics
from quantlab.core.types import BarFrame, SlippageConfig, coerce_bar_frame_df

HOUR = 3_600_000
FEE_BPS = 10.0
SLIP_BPS = 5.0


def frame_from_closes(closes: np.ndarray, *, spread: float = 0.001) -> BarFrame:
    """Build bars around a close series, with opens equal to the previous close."""
    closes = np.asarray(closes, dtype="float64")
    opens = np.concatenate([[closes[0]], closes[:-1]])
    band = np.maximum(np.abs(closes) * spread, 1e-6)
    frame = pd.DataFrame(
        {
            "ts_open": np.arange(len(closes), dtype="int64") * HOUR,
            "open": opens,
            "high": np.maximum(opens, closes) + band,
            "low": np.minimum(opens, closes) - band,
            "close": closes,
            "volume": np.full(len(closes), 100.0),
            "quote_volume": np.full(len(closes), 1e6),
            "trades": np.full(len(closes), 50, dtype="int64"),
            "is_gap_filled": np.zeros(len(closes), dtype=bool),
        }
    )
    return BarFrame(coerce_bar_frame_df(frame), symbol="SYN/USDT", timeframe="1h")


def random_walk(seed: int, n: int = 1_500, sigma: float = 0.004) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(0.0, sigma, n)))


def trending(seed: int, n: int = 1_500, drift: float = 0.0012) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.004, n)))


def mean_reverting(seed: int, n: int = 1_500, kappa: float = 0.05) -> np.ndarray:
    """Ornstein-Uhlenbeck around a constant level."""
    rng = np.random.default_rng(seed)
    level = np.log(100.0)
    x = level
    out = np.empty(n)
    for i in range(n):
        x += kappa * (level - x) + rng.normal(0.0, 0.02)
        out[i] = x
    return np.exp(out)


def run_strategy(strategy, closes: np.ndarray, params=None, *, costs: bool = True):
    config = zero_cost_config(
        fee_bps=FEE_BPS if costs else 0.0,
        slippage=SlippageConfig(fixed_bps=SLIP_BPS if costs else 0.0),
        lot_step=1e-10,
        min_notional=0.0,
    )
    return SimpleBarEngine().run(strategy, frame_from_closes(closes), params or {}, config)


def sma_cross(fast: int = 20, slow: int = 80):
    from strategies.baselines.sma_cross import STRATEGY as SmaCross

    class Tuned(SmaCross):  # type: ignore[misc, valid-type]
        warmup_bars = slow + 2

    return Tuned(), {"fast": fast, "slow": slow}


# ---------------------------------------------------------------------------
# a random walk is not a business
# ---------------------------------------------------------------------------
def test_a_trend_follower_does_not_profit_on_a_random_walk() -> None:
    """Averaged over seeds, the outcome must sit between minus costs and zero.

    A backtester that finds an edge in pure noise has a bug in its fills, its
    cost accounting, or its bar alignment. This is the single most informative
    sanity test in the suite.
    """
    returns = []
    for seed in range(20):
        strategy, params = sma_cross()
        result = run_strategy(strategy, random_walk(seed), params)
        returns.append(compute_metrics(result).net_return)

    mean_return = float(np.mean(returns))
    assert mean_return < 0.02, f"found an edge in noise: mean net return {mean_return:.4f}"
    assert mean_return > -0.60, "losses far beyond plausible costs suggest an accounting bug"


def test_buy_and_hold_on_a_random_walk_tracks_the_walk() -> None:
    """The one strategy whose answer is arithmetic: it should match, minus a round trip."""
    from strategies.baselines.buy_and_hold import STRATEGY as BuyAndHold

    closes = random_walk(3)
    result = run_strategy(BuyAndHold(), closes)
    metrics = compute_metrics(result)

    market = closes[-1] / closes[0] - 1.0
    assert metrics.net_return == pytest.approx(market, abs=0.02)
    assert metrics.net_return < market, "costs must make holding slightly worse"
    assert metrics.n_trades == 1.0


def test_costs_always_make_a_strategy_worse() -> None:
    """Whatever the regime, the same strategy must earn less once it pays to trade."""
    for seed in range(5):
        strategy, params = sma_cross()
        free = compute_metrics(run_strategy(strategy, random_walk(seed), params, costs=False))
        strategy, params = sma_cross()
        paid = compute_metrics(run_strategy(strategy, random_walk(seed), params))
        assert paid.net_return <= free.net_return + 1e-12


# ---------------------------------------------------------------------------
# a trend is
# ---------------------------------------------------------------------------
def test_a_trend_follower_profits_on_a_trending_series() -> None:
    returns = []
    for seed in range(10):
        strategy, params = sma_cross()
        result = run_strategy(strategy, trending(seed), params)
        returns.append(compute_metrics(result).net_return)
    assert float(np.mean(returns)) > 0.10


def test_a_trend_follower_beats_a_random_walk_on_a_trend() -> None:
    strategy, params = sma_cross()
    on_trend = compute_metrics(run_strategy(strategy, trending(1), params)).net_return
    strategy, params = sma_cross()
    on_noise = compute_metrics(run_strategy(strategy, random_walk(1), params)).net_return
    assert on_trend > on_noise


# ---------------------------------------------------------------------------
# mean reversion
# ---------------------------------------------------------------------------
def test_rsi_reversion_profits_on_a_mean_reverting_series() -> None:
    from strategies.baselines.rsi_reversion import STRATEGY as RsiReversion

    returns = []
    for seed in range(10):
        result = run_strategy(
            RsiReversion(), mean_reverting(seed), {"n": 14, "oversold": 35.0, "exit_level": 55.0}
        )
        returns.append(compute_metrics(result).net_return)
    assert float(np.mean(returns)) > 0.0


def test_a_trend_follower_underperforms_on_a_mean_reverting_series() -> None:
    """Whipsaw: the regime the strategy is wrong for should cost it money."""
    strategy, params = sma_cross()
    reverting = compute_metrics(run_strategy(strategy, mean_reverting(2), params)).net_return
    strategy, params = sma_cross()
    trend = compute_metrics(run_strategy(strategy, trending(2), params)).net_return
    assert reverting < trend


# ---------------------------------------------------------------------------
# random entry is the control
# ---------------------------------------------------------------------------
def test_random_entry_loses_roughly_its_costs_on_a_random_walk() -> None:
    from strategies.baselines.random_entry import STRATEGY as RandomEntry

    results = []
    for seed in range(10):
        config = zero_cost_config(
            seed=seed,
            fee_bps=FEE_BPS,
            slippage=SlippageConfig(fixed_bps=SLIP_BPS),
            lot_step=1e-10,
            min_notional=0.0,
        )
        result = SimpleBarEngine().run(
            RandomEntry(),
            frame_from_closes(random_walk(seed)),
            {"p_enter": 0.05, "hold_bars": 12},
            config,
        )
        results.append(compute_metrics(result))

    mean_return = float(np.mean([m.net_return for m in results]))
    assert mean_return < 0.05, "random entry must not look like an edge"
    assert all(m.n_trades > 0 for m in results)


def test_more_trading_costs_more() -> None:
    """Doubling turnover should roughly double the fees paid."""
    from strategies.baselines.random_entry import STRATEGY as RandomEntry

    closes = random_walk(11)
    config = zero_cost_config(
        seed=1,
        fee_bps=FEE_BPS,
        slippage=SlippageConfig(fixed_bps=SLIP_BPS),
        lot_step=1e-10,
        min_notional=0.0,
    )
    rare = SimpleBarEngine().run(
        RandomEntry(), frame_from_closes(closes), {"p_enter": 0.02, "hold_bars": 48}, config
    )
    often = SimpleBarEngine().run(
        RandomEntry(), frame_from_closes(closes), {"p_enter": 0.30, "hold_bars": 2}, config
    )
    assert often.cost_summary["total_fees"] > rare.cost_summary["total_fees"] * 2


# ---------------------------------------------------------------------------
# shorts mirror longs
# ---------------------------------------------------------------------------
def test_shorting_a_falling_market_profits() -> None:
    falling = 100.0 * np.exp(np.cumsum(np.full(400, -0.001)))
    config = zero_cost_config(allow_short=True, lot_step=1e-10, min_notional=0.0)
    result = SimpleBarEngine().run(
        ScriptedStrategy(["short"] * 400), frame_from_closes(falling), {}, config
    )
    assert compute_metrics(result).net_return > 0.20


def test_long_and_short_are_symmetric_without_costs() -> None:
    """The same path, traded both ways, should give mirrored returns."""
    rising = 100.0 * np.exp(np.cumsum(np.full(300, 0.001)))
    config = zero_cost_config(allow_short=True, lot_step=1e-10, min_notional=0.0)
    bars = frame_from_closes(rising)

    long_run = SimpleBarEngine().run(ScriptedStrategy(["long"] * 300), bars, {}, config)
    short_run = SimpleBarEngine().run(ScriptedStrategy(["short"] * 300), bars, {}, config)

    long_pnl = long_run.equity.iloc[-1] - 10_000.0
    short_pnl = short_run.equity.iloc[-1] - 10_000.0
    assert long_pnl > 0 > short_pnl
    # Not exactly equal and opposite: a long position grows as it wins while a
    # short shrinks, so the compounding differs. The magnitudes should be close.
    assert abs(long_pnl) == pytest.approx(abs(short_pnl), rel=0.35)
