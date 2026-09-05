"""Execution semantics of the bar-close engine (master spec sections 8.4, 8.6).

Every expected number in this file is derived on paper from the toy bars, not
copied from a run.  A test that merely records what the engine did would pass
just as happily after the engine started doing the wrong thing.

Fees and slippage are switched off unless a test is specifically about them, so
one mechanism is under test at a time.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
from tests.helpers import ScriptedStrategy, flat_frame, toy_frame, zero_cost_config

from quantlab.adapters.engine import ENGINE_VERSION, SimpleBarEngine
from quantlab.core.errors import StrategyRuntimeError
from quantlab.core.types import (
    BarFrame,
    RiskSpec,
    Side,
    Signal,
    SignalKind,
    SizingSpec,
    SlippageConfig,
)
from quantlab.ports.engine import BacktestEngine

HOUR = 3_600_000


@pytest.fixture
def engine() -> SimpleBarEngine:
    return SimpleBarEngine()


def run(engine, strategy, bars, config=None, params=None):
    return engine.run(strategy, bars, params or {}, config or zero_cost_config())


# ---------------------------------------------------------------------------
# the port and the shape of a result
# ---------------------------------------------------------------------------
def test_the_engine_satisfies_the_port(engine: SimpleBarEngine) -> None:
    assert isinstance(engine, BacktestEngine)
    assert engine.name == "simple_bar"
    assert engine.version == ENGINE_VERSION


def test_result_has_one_row_per_bar(engine: SimpleBarEngine) -> None:
    bars = flat_frame([100.0] * 6)
    result = run(engine, ScriptedStrategy(["flat"] * 6), bars)
    assert result.n_bars == 6
    assert len(result.equity) == len(result.position_frac) == len(result.signals) == 6
    assert list(result.equity.index) == list(bars.ts_open)


def test_a_flat_strategy_never_trades(engine: SimpleBarEngine) -> None:
    result = run(engine, ScriptedStrategy(["flat"] * 6), flat_frame([100.0] * 6))
    assert result.fills == ()
    assert result.trades == ()
    assert (result.equity == 10_000.0).all()
    assert (result.position_frac == 0.0).all()


def test_an_empty_frame_produces_an_empty_result(engine: SimpleBarEngine) -> None:
    result = run(engine, ScriptedStrategy([]), BarFrame.empty(symbol="BTC/USDT", timeframe="1h"))
    assert result.n_bars == 0
    assert result.trades == ()


# ---------------------------------------------------------------------------
# next-open fills (spec section 8.4 step 1)
# ---------------------------------------------------------------------------
def test_a_signal_at_bar_t_fills_at_the_open_of_bar_t_plus_one(
    engine: SimpleBarEngine,
) -> None:
    """The single most important rule in the engine."""
    bars = toy_frame(
        [
            (100.0, 100.0, 100.0, 100.0),
            (110.0, 110.0, 110.0, 110.0),
            (120.0, 120.0, 120.0, 120.0),
        ]
    )
    result = run(engine, ScriptedStrategy(["long", "long", "long"]), bars)

    assert len(result.fills) == 2  # entry at bar 1, end-of-data exit at bar 2
    entry = result.fills[0]
    assert entry.bar_index == 1
    assert entry.fill_price == 110.0, "filled at bar 1's OPEN, not bar 0's close"
    assert result.equity.iloc[0] == 10_000.0, "no position yet at the deciding bar"


def test_no_fill_happens_on_the_deciding_bar(engine: SimpleBarEngine) -> None:
    bars = toy_frame([(100.0, 100.0, 100.0, 100.0), (200.0, 200.0, 200.0, 200.0)])
    result = run(engine, ScriptedStrategy(["long", "flat"]), bars)
    assert all(fill.bar_index >= 1 for fill in result.fills)


def test_the_last_bar_creates_no_order(engine: SimpleBarEngine) -> None:
    """There is no bar to fill it at, so asking for one would be a phantom trade."""
    bars = flat_frame([100.0, 100.0, 100.0])
    result = run(engine, ScriptedStrategy(["flat", "flat", "long"]), bars)
    assert result.fills == ()


# ---------------------------------------------------------------------------
# long profit and loss, hand-calculated
# ---------------------------------------------------------------------------
def test_long_pnl_without_costs(engine: SimpleBarEngine) -> None:
    """10 000 at 100 buys 100 units; selling at 120 is exactly +2 000."""
    bars = toy_frame(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
            (120.0, 120.0, 120.0, 120.0),
        ]
    )
    result = run(engine, ScriptedStrategy(["long", "flat", "flat"]), bars)

    trade = result.trades[0]
    assert trade.side is Side.LONG
    assert trade.qty == pytest.approx(100.0)
    assert trade.entry_px == pytest.approx(100.0)
    assert trade.exit_px == pytest.approx(120.0)
    assert trade.pnl == pytest.approx(2_000.0)
    assert trade.pnl_pct == pytest.approx(0.2)
    assert result.equity.iloc[-1] == pytest.approx(12_000.0)


def test_long_loss_without_costs(engine: SimpleBarEngine) -> None:
    bars = toy_frame(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
            (90.0, 90.0, 90.0, 90.0),
        ]
    )
    result = run(engine, ScriptedStrategy(["long", "flat", "flat"]), bars)
    assert result.trades[0].pnl == pytest.approx(-1_000.0)
    assert result.equity.iloc[-1] == pytest.approx(9_000.0)


# ---------------------------------------------------------------------------
# short profit and loss
# ---------------------------------------------------------------------------
def test_shorts_are_refused_unless_enabled(engine: SimpleBarEngine) -> None:
    bars = flat_frame([100.0] * 4)
    result = run(engine, ScriptedStrategy(["short"] * 4), bars)
    assert result.fills == ()
    assert any("short_ignored" in line for line in result.log)


def test_short_profits_when_the_price_falls(engine: SimpleBarEngine) -> None:
    """Sell 100 units at 100, buy back at 80: +2 000."""
    bars = toy_frame(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
            (80.0, 80.0, 80.0, 80.0),
        ]
    )
    result = run(
        engine,
        ScriptedStrategy(["short", "flat", "flat"]),
        bars,
        zero_cost_config(allow_short=True),
    )
    trade = result.trades[0]
    assert trade.side is Side.SHORT
    assert trade.qty == pytest.approx(100.0)
    assert trade.pnl == pytest.approx(2_000.0)
    assert result.equity.iloc[-1] == pytest.approx(12_000.0)


def test_short_loses_when_the_price_rises(engine: SimpleBarEngine) -> None:
    bars = toy_frame(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
            (110.0, 110.0, 110.0, 110.0),
        ]
    )
    result = run(
        engine,
        ScriptedStrategy(["short", "flat", "flat"]),
        bars,
        zero_cost_config(allow_short=True),
    )
    assert result.trades[0].pnl == pytest.approx(-1_000.0)


def test_short_position_fraction_is_negative(engine: SimpleBarEngine) -> None:
    bars = flat_frame([100.0] * 4)
    result = run(engine, ScriptedStrategy(["short"] * 4), bars, zero_cost_config(allow_short=True))
    assert result.position_frac.iloc[1] == pytest.approx(-1.0)


def test_short_borrow_is_charged_per_bar(engine: SimpleBarEngine) -> None:
    """10 bps per bar on a 10 000 short is 10 per bar, for the four bars it is held.

    The borrow is charged on the *position*, not on equity, so it stays at 10
    even as equity decays; the rate is small enough that the decay never crosses
    the 1 % rebalance band and trims the short.
    """
    bars = flat_frame([100.0] * 5)
    config = zero_cost_config(allow_short=True, short_borrow_bps_per_bar=10.0)
    result = run(
        engine, ScriptedStrategy(["short", "short", "short", "short", "flat"]), bars, config
    )
    assert len(result.fills) == 2, "opened once, closed once: no rebalancing churn"
    assert result.cost_summary["total_borrow"] == pytest.approx(40.0)
    assert result.equity.iloc[-1] == pytest.approx(9_960.0)


def test_borrow_decay_eventually_trims_the_short(engine: SimpleBarEngine) -> None:
    """A borrow large enough to move exposure past the band does rebalance.

    Not a bug: the position has drifted above its target fraction and correcting
    it is what the band is for.
    """
    bars = flat_frame([100.0] * 6)
    config = zero_cost_config(allow_short=True, short_borrow_bps_per_bar=100.0)
    result = run(engine, ScriptedStrategy(["short"] * 6), bars, config)
    assert len(result.fills) > 2


# ---------------------------------------------------------------------------
# reversal: two fills, two fees (spec section 8.4 step 5)
# ---------------------------------------------------------------------------
def test_a_reversal_is_two_fills_at_the_same_open(engine: SimpleBarEngine) -> None:
    bars = flat_frame([100.0] * 5)
    result = run(
        engine,
        ScriptedStrategy(["long", "short", "short", "short", "flat"]),
        bars,
        zero_cost_config(allow_short=True),
    )
    at_bar_2 = [fill for fill in result.fills if fill.bar_index == 2]
    assert len(at_bar_2) == 2, "sell to flat, then sell short, at the same open"
    assert at_bar_2[0].side is Side.SHORT
    assert at_bar_2[1].side is Side.SHORT
    assert len(result.trades) == 2


def test_a_reversal_charges_two_fees(engine: SimpleBarEngine) -> None:
    bars = flat_frame([100.0] * 5)
    config = zero_cost_config(allow_short=True, fee_bps=10.0)
    result = run(
        engine, ScriptedStrategy(["long", "short", "short", "short", "flat"]), bars, config
    )
    at_bar_2 = [fill for fill in result.fills if fill.bar_index == 2]
    assert len(at_bar_2) == 2
    assert all(fill.fee > 0 for fill in at_bar_2)


# ---------------------------------------------------------------------------
# duplicate entries and the rebalance band
# ---------------------------------------------------------------------------
def test_holding_a_signal_does_not_re_enter(engine: SimpleBarEngine) -> None:
    """The classic duplicate-entry bug: one trade, not one per bar."""
    bars = flat_frame([100.0] * 10)
    result = run(engine, ScriptedStrategy(["long"] * 10), bars)
    entries = [fill for fill in result.fills if fill.side is Side.LONG]
    assert len(entries) == 1
    assert len(result.trades) == 1


def test_a_tiny_exposure_change_is_not_worth_a_round_trip(engine: SimpleBarEngine) -> None:
    """Below the 1 % band, rebalancing costs more than the drift it corrects."""

    class Drifting(ScriptedStrategy):
        def on_bar(self, ctx):
            return Signal(SignalKind.LONG, target_fraction=0.5 + 0.001 * ctx.i)

    bars = flat_frame([100.0] * 8)
    result = run(engine, Drifting(["long"] * 8), bars)
    assert len([f for f in result.fills if f.bar_index < 7]) == 1


def test_a_large_exposure_change_does_rebalance(engine: SimpleBarEngine) -> None:
    class Stepping(ScriptedStrategy):
        def on_bar(self, ctx):
            return Signal(SignalKind.LONG, target_fraction=0.3 if ctx.i < 3 else 0.9)

    bars = flat_frame([100.0] * 8)
    result = run(engine, Stepping(["long"] * 8), bars)
    assert len([f for f in result.fills if f.bar_index < 7]) == 2


# ---------------------------------------------------------------------------
# fees and slippage
# ---------------------------------------------------------------------------
def test_fee_is_charged_on_both_sides(engine: SimpleBarEngine) -> None:
    """10 bps on ~10 000 in and ~10 000 out is very close to 20."""
    bars = flat_frame([100.0] * 4)
    result = run(
        engine,
        ScriptedStrategy(["long", "long", "flat", "flat"]),
        bars,
        zero_cost_config(fee_bps=10.0),
    )
    assert len(result.fills) == 2
    assert result.cost_summary["total_fees"] == pytest.approx(
        sum(fill.fee for fill in result.fills)
    )
    assert result.cost_summary["total_fees"] == pytest.approx(19.98, abs=0.05)


def test_slippage_moves_the_price_against_the_trader(engine: SimpleBarEngine) -> None:
    bars = flat_frame([100.0] * 4)
    config = zero_cost_config(slippage=SlippageConfig(model="fixed_bps", fixed_bps=50.0))
    result = run(engine, ScriptedStrategy(["long", "long", "flat", "flat"]), bars, config)

    buy, sell = result.fills[0], result.fills[1]
    assert buy.fill_price == pytest.approx(100.0 * 1.005), "buys fill higher"
    assert sell.fill_price == pytest.approx(100.0 * 0.995), "sells fill lower"
    assert buy.ref_price == 100.0
    assert result.equity.iloc[-1] < 10_000.0, "a round trip in a flat market must lose"


def test_slippage_cost_is_measured_against_the_reference_price(
    engine: SimpleBarEngine,
) -> None:
    bars = flat_frame([100.0] * 4)
    config = zero_cost_config(slippage=SlippageConfig(model="fixed_bps", fixed_bps=50.0))
    result = run(engine, ScriptedStrategy(["long", "long", "flat", "flat"]), bars, config)
    for fill in result.fills:
        assert fill.slippage_cost == pytest.approx(abs(fill.fill_price - fill.ref_price) * fill.qty)


def test_the_cost_multiplier_scales_both_costs(engine: SimpleBarEngine) -> None:
    bars = flat_frame([100.0] * 4)
    script = ["long", "long", "flat", "flat"]
    base = run(
        engine,
        ScriptedStrategy(script),
        bars,
        zero_cost_config(fee_bps=10.0, slippage=SlippageConfig(fixed_bps=5.0)),
    )
    stressed = run(
        engine,
        ScriptedStrategy(script),
        bars,
        zero_cost_config(fee_bps=10.0, slippage=SlippageConfig(fixed_bps=5.0), cost_multiplier=3.0),
    )
    assert stressed.cost_summary["total_slippage"] > 2.9 * base.cost_summary["total_slippage"]
    assert stressed.equity.iloc[-1] < base.equity.iloc[-1]


# ---------------------------------------------------------------------------
# position sizing
# ---------------------------------------------------------------------------
def test_full_allocation_uses_exactly_all_the_equity(engine: SimpleBarEngine) -> None:
    """With fees, the size must shrink so that cash lands at zero, not below it.

    Sizing from the gross notional would finance the fee, leaving cash negative
    and position_frac above max_position_fraction.
    """
    bars = flat_frame([100.0] * 4)
    result = run(
        engine,
        ScriptedStrategy(["long"] * 4),
        bars,
        zero_cost_config(fee_bps=10.0, slippage=SlippageConfig(fixed_bps=5.0)),
    )
    entry = result.fills[0]
    assert entry.qty * entry.fill_price + entry.fee == pytest.approx(10_000.0)
    assert result.position_frac.iloc[1] == pytest.approx(1.0)


def test_position_fraction_never_exceeds_the_cap(engine: SimpleBarEngine) -> None:
    class Greedy(ScriptedStrategy):
        def on_bar(self, ctx):
            return Signal(SignalKind.LONG, target_fraction=5.0)

    bars = flat_frame([100.0] * 6)
    result = run(
        engine,
        Greedy(["long"] * 6),
        bars,
        zero_cost_config(fee_bps=10.0, max_position_fraction=0.5),
    )
    assert result.position_frac.abs().max() <= 0.5 + 1e-9


def test_target_fraction_scales_the_position(engine: SimpleBarEngine) -> None:
    class Half(ScriptedStrategy):
        def on_bar(self, ctx):
            return Signal(SignalKind.LONG, target_fraction=0.5)

    bars = flat_frame([100.0] * 5)
    result = run(engine, Half(["long"] * 5), bars)
    assert result.position_frac.iloc[1] == pytest.approx(0.5)
    assert result.fills[0].qty == pytest.approx(50.0)


def test_sizing_fraction_scales_the_position(engine: SimpleBarEngine) -> None:
    bars = flat_frame([100.0] * 5)
    config = zero_cost_config(sizing=SizingSpec(mode="fixed_fraction", fraction=0.25))
    result = run(engine, ScriptedStrategy(["long"] * 5), bars, config)
    assert result.position_frac.iloc[1] == pytest.approx(0.25)


def test_atr_risk_sizing_shrinks_when_volatility_rises(engine: SimpleBarEngine) -> None:
    calm = toy_frame([(100.0, 100.5, 99.5, 100.0)] * 40)
    wild = toy_frame([(100.0, 110.0, 90.0, 100.0)] * 40)
    config = zero_cost_config(sizing=SizingSpec(mode="atr_risk", atr_period=14, atr_risk_pct=0.01))
    calm_run = run(engine, ScriptedStrategy(["long"] * 40), calm, config)
    wild_run = run(engine, ScriptedStrategy(["long"] * 40), wild, config)
    assert abs(calm_run.position_frac.iloc[20]) > abs(wild_run.position_frac.iloc[20])


def test_volatility_target_sizing_shrinks_when_volatility_rises(
    engine: SimpleBarEngine,
) -> None:
    calm = flat_frame([100.0 + 0.01 * (i % 2) for i in range(60)])
    wild = flat_frame([100.0 + 5.0 * (i % 2) for i in range(60)])
    config = zero_cost_config(
        sizing=SizingSpec(mode="volatility_target", target_vol_annual=0.20, vol_lookback=20)
    )
    calm_run = run(engine, ScriptedStrategy(["long"] * 60), calm, config)
    wild_run = run(engine, ScriptedStrategy(["long"] * 60), wild, config)
    assert abs(calm_run.position_frac.iloc[40]) > abs(wild_run.position_frac.iloc[40])


# ---------------------------------------------------------------------------
# minimum notional and lot step
# ---------------------------------------------------------------------------
def test_an_order_below_the_minimum_notional_is_dropped(engine: SimpleBarEngine) -> None:
    class Tiny(ScriptedStrategy):
        def on_bar(self, ctx):
            return Signal(SignalKind.LONG, target_fraction=0.02)

    bars = flat_frame([100.0] * 5)
    result = run(engine, Tiny(["long"] * 5), bars, zero_cost_config(min_notional=1_000.0))
    assert result.fills == ()
    assert any("below_min_notional" in line for line in result.log)


def test_the_minimum_notional_never_traps_a_position(engine: SimpleBarEngine) -> None:
    """An exit is always allowed: no exchange forbids closing a small position."""
    bars = flat_frame([100.0] * 5)
    config = zero_cost_config(min_notional=100.0)
    result = run(engine, ScriptedStrategy(["long", "long", "flat", "flat", "flat"]), bars, config)
    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "signal"


def test_quantities_are_rounded_down_to_the_lot_step(engine: SimpleBarEngine) -> None:
    bars = flat_frame([100.0] * 4)
    result = run(engine, ScriptedStrategy(["long"] * 4), bars, zero_cost_config(lot_step=0.5))
    qty = result.fills[0].qty
    assert qty == pytest.approx(round(qty / 0.5) * 0.5)
    assert qty <= 100.0


# ---------------------------------------------------------------------------
# gap-filled bars (spec section 7.3)
# ---------------------------------------------------------------------------
def test_no_order_fills_on_a_gap_filled_bar(engine: SimpleBarEngine) -> None:
    bars = toy_frame(
        [(100.0, 100.0, 100.0, 100.0)] * 4,
        gap_filled=[False, True, False, False],
    )
    result = run(engine, ScriptedStrategy(["long", "long", "long", "long"]), bars)
    assert all(fill.bar_index != 1 for fill in result.fills)
    assert any("order_deferred" in line for line in result.log)


def test_a_deferred_order_fills_on_the_next_real_bar(engine: SimpleBarEngine) -> None:
    bars = toy_frame(
        [(100.0, 100.0, 100.0, 100.0)] * 5,
        gap_filled=[False, True, False, False, False],
    )
    result = run(engine, ScriptedStrategy(["long"] * 5), bars)
    assert result.fills[0].bar_index == 2


# ---------------------------------------------------------------------------
# end of data (spec section 8.4 step 7)
# ---------------------------------------------------------------------------
def test_an_open_position_is_closed_at_the_final_close(engine: SimpleBarEngine) -> None:
    bars = toy_frame(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 130.0, 100.0, 130.0),
        ]
    )
    result = run(engine, ScriptedStrategy(["long", "long", "long"]), bars)
    trade = result.trades[0]
    assert trade.exit_reason == "end_of_data"
    assert trade.exit_px == pytest.approx(130.0)
    assert result.position_frac.iloc[-1] == 0.0


def test_end_of_data_pays_full_costs(engine: SimpleBarEngine) -> None:
    bars = flat_frame([100.0] * 3)
    result = run(
        engine,
        ScriptedStrategy(["long"] * 3),
        bars,
        zero_cost_config(fee_bps=10.0, slippage=SlippageConfig(fixed_bps=5.0)),
    )
    closing = result.fills[-1]
    assert closing.fee > 0
    assert closing.slippage_cost > 0


# ---------------------------------------------------------------------------
# warm-up
# ---------------------------------------------------------------------------
def test_the_strategy_is_not_asked_during_warm_up(engine: SimpleBarEngine) -> None:
    strategy = ScriptedStrategy(["long"] * 10, warmup=4)
    result = run(engine, strategy, flat_frame([100.0] * 10))
    assert strategy.seen == list(range(4, 10))
    assert list(result.signals)[:4] == ["flat"] * 4
    assert all(fill.bar_index > 4 for fill in result.fills)


# ---------------------------------------------------------------------------
# strategy failures
# ---------------------------------------------------------------------------
def test_a_strategy_exception_fails_the_run(engine: SimpleBarEngine) -> None:
    class Exploding(ScriptedStrategy):
        def on_bar(self, ctx):
            raise ValueError("boom")

    with pytest.raises(StrategyRuntimeError, match="strategy raised"):
        run(engine, Exploding(["long"]), flat_frame([100.0] * 3))


def test_a_strategy_returning_a_non_signal_fails_the_run(engine: SimpleBarEngine) -> None:
    class Confused(ScriptedStrategy):
        def on_bar(self, ctx):
            return "long"

    with pytest.raises(StrategyRuntimeError, match="not a Signal"):
        run(engine, Confused(["long"]), flat_frame([100.0] * 3))


def test_unknown_parameters_are_rejected(engine: SimpleBarEngine) -> None:
    from quantlab.core.errors import ConfigError

    with pytest.raises(ConfigError, match="does not declare"):
        run(engine, ScriptedStrategy(["flat"] * 3), flat_frame([100.0] * 3), params={"x": 1})


# ---------------------------------------------------------------------------
# equity accounting
# ---------------------------------------------------------------------------
def test_equity_equals_cash_plus_position_value_every_bar(engine: SimpleBarEngine) -> None:
    """The engine raises EngineError if this ever drifts; assert it stayed silent."""
    bars = flat_frame([100.0 + i for i in range(30)])
    result = run(
        engine,
        ScriptedStrategy(["long", "flat"] * 15),
        bars,
        zero_cost_config(fee_bps=10.0, slippage=SlippageConfig(fixed_bps=5.0)),
    )
    assert result.equity.notna().all()
    assert (result.equity > 0).all()


def test_realised_pnl_reconciles_with_the_equity_change(engine: SimpleBarEngine) -> None:
    bars = flat_frame([100.0, 100.0, 110.0, 110.0, 95.0, 95.0])
    result = run(engine, ScriptedStrategy(["long", "flat", "long", "flat", "flat", "flat"]), bars)
    total_pnl = sum(trade.pnl for trade in result.trades)
    assert result.equity.iloc[-1] - result.equity.iloc[0] == pytest.approx(total_pnl)


def test_costs_are_reported_consistently(engine: SimpleBarEngine) -> None:
    bars = flat_frame([100.0] * 12)
    result = run(
        engine,
        ScriptedStrategy(["long", "flat"] * 6),
        bars,
        zero_cost_config(fee_bps=10.0, slippage=SlippageConfig(fixed_bps=5.0)),
    )
    assert result.cost_summary["total_fees"] == pytest.approx(
        sum(fill.fee for fill in result.fills)
    )
    assert result.cost_summary["total_slippage"] == pytest.approx(
        sum(fill.slippage_cost for fill in result.fills)
    )
    assert result.cost_summary["turnover"] == pytest.approx(
        sum(fill.qty * fill.fill_price for fill in result.fills)
    )


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------
def test_two_identical_runs_agree_bit_for_bit(engine: SimpleBarEngine) -> None:
    bars = flat_frame([100.0 + (i % 7) for i in range(60)])
    config = zero_cost_config(fee_bps=10.0, slippage=SlippageConfig(fixed_bps=5.0))
    script = ["long", "long", "flat"] * 20

    first = engine.run(ScriptedStrategy(script), bars, {}, config)
    second = SimpleBarEngine().run(ScriptedStrategy(script), bars, {}, config)

    assert list(first.equity) == list(second.equity)
    assert first.fills == second.fills
    assert first.trades == second.trades
    assert first.log == second.log


def test_a_seeded_strategy_is_reproducible(engine: SimpleBarEngine) -> None:
    from strategies.baselines.random_entry import STRATEGY as RandomEntry

    bars = flat_frame([100.0 + (i % 5) for i in range(120)])
    config = zero_cost_config(seed=1234)
    first = engine.run(RandomEntry(), bars, {}, config)
    second = engine.run(RandomEntry(), bars, {}, config)
    assert first.trades == second.trades


def test_a_different_seed_gives_a_different_path(engine: SimpleBarEngine) -> None:
    from strategies.baselines.random_entry import STRATEGY as RandomEntry

    bars = flat_frame([100.0 + (i % 5) for i in range(200)])
    a = engine.run(RandomEntry(), bars, {}, zero_cost_config(seed=1))
    b = engine.run(RandomEntry(), bars, {}, zero_cost_config(seed=2))
    assert list(a.signals) != list(b.signals)


# ---------------------------------------------------------------------------
# timestamps
# ---------------------------------------------------------------------------
def test_trade_timestamps_come_from_the_bars(engine: SimpleBarEngine) -> None:
    bars = flat_frame([100.0] * 5)
    result = run(engine, ScriptedStrategy(["long", "long", "flat", "flat", "flat"]), bars)
    trade = result.trades[0]
    assert trade.entry_ts == int(bars.ts_open[1])
    assert trade.exit_ts == int(bars.ts_open[3])
    assert trade.bars_held == 2


def test_the_result_index_is_int64_millisecond_timestamps(engine: SimpleBarEngine) -> None:
    bars = flat_frame([100.0] * 4)
    result = run(engine, ScriptedStrategy(["flat"] * 4), bars)
    assert result.equity.index.dtype == "int64"
    assert result.equity.index[1] - result.equity.index[0] == HOUR


def test_a_daily_timeframe_works_the_same(engine: SimpleBarEngine) -> None:
    bars = flat_frame([100.0] * 5, timeframe="1d")
    result = run(engine, ScriptedStrategy(["long"] * 5), bars, zero_cost_config(bars_per_year=365))
    assert result.bars_per_year == 365
    assert result.equity.index[1] - result.equity.index[0] == 86_400_000


# ---------------------------------------------------------------------------
# vectorized strategies
# ---------------------------------------------------------------------------
def test_a_vectorized_strategy_is_shifted_by_one_bar(engine: SimpleBarEngine) -> None:
    """Spec section 9.1: vectorised output is shifted, as insurance against non-causal code."""
    import pandas as pd

    class Vector:
        name, version, style = "vector", "1", "vectorized"
        params: ClassVar[dict] = {}
        warmup_bars = 0
        risk = RiskSpec()
        sizing = SizingSpec()

        def prepare(self, params): ...

        def signals(self, bars, params):
            return pd.Series(["long"] * len(bars))

    bars = flat_frame([100.0] * 5)
    result = run(engine, Vector(), bars)
    assert next(iter(result.signals)) == "flat", "the first emitted signal is shifted away"
    assert list(result.signals)[1] == "long"


def test_a_vectorized_strategy_of_the_wrong_length_fails(engine: SimpleBarEngine) -> None:
    import pandas as pd

    class Short:
        name, version, style = "short_series", "1", "vectorized"
        params: ClassVar[dict] = {}
        warmup_bars = 0
        risk = RiskSpec()
        sizing = SizingSpec()

        def prepare(self, params): ...

        def signals(self, bars, params):
            return pd.Series(["long"] * 2)

    with pytest.raises(StrategyRuntimeError, match="wrong number of signals"):
        run(engine, Short(), flat_frame([100.0] * 5))
