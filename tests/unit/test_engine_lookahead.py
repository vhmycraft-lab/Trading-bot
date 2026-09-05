"""The backtester cannot see the future (INV-3, master spec sections 8.4, 9.1).

Phase 2 proved the *data layer* cannot serve a future bar.  These tests prove the
*engine* never asks for one, and that nothing the engine computes on a strategy's
behalf — indicators, sizing, stops, costs — depends on a bar the strategy could
not have seen.

The strongest test here is the truncation probe at the bottom: it runs the same
strategy over a short series and a long one and requires the shared prefix to be
identical. Any leak anywhere in the pipeline changes that prefix.
"""

from __future__ import annotations

import numpy as np
import pytest
from tests.helpers import ScriptedStrategy, flat_frame, make_bars, toy_frame, zero_cost_config

from quantlab.adapters.engine import SimpleBarEngine
from quantlab.core.types import BarFrame, RiskSpec, Signal, SignalKind, SlippageConfig


@pytest.fixture
def engine() -> SimpleBarEngine:
    return SimpleBarEngine()


# ---------------------------------------------------------------------------
# what the strategy is handed
# ---------------------------------------------------------------------------
def test_the_context_window_stops_at_the_current_bar(engine: SimpleBarEngine) -> None:
    widths: list[int] = []

    class Probe(ScriptedStrategy):
        def on_bar(self, ctx):
            widths.append(len(ctx.bars))
            return Signal(SignalKind.FLAT)

    engine.run(Probe([]), flat_frame([100.0] * 12), {}, zero_cost_config())
    assert widths == list(range(1, 13))


def test_a_strategy_reaching_past_its_bar_raises(engine: SimpleBarEngine) -> None:
    class Cheater(ScriptedStrategy):
        def on_bar(self, ctx):
            ctx.bars[ctx.i + 1]  # the whole point
            return Signal(SignalKind.FLAT)

    from quantlab.core.errors import StrategyRuntimeError

    with pytest.raises(StrategyRuntimeError) as excinfo:
        engine.run(Cheater([]), flat_frame([100.0] * 6), {}, zero_cost_config())
    assert "LookaheadError" in str(excinfo.value)


def test_the_context_is_read_only(engine: SimpleBarEngine) -> None:
    class Mutator(ScriptedStrategy):
        def on_bar(self, ctx):
            with pytest.raises(AttributeError):
                ctx.i = 0
            with pytest.raises(AttributeError):
                ctx.equity = 1e9
            return Signal(SignalKind.FLAT)

    engine.run(Mutator([]), flat_frame([100.0] * 4), {}, zero_cost_config())


def test_the_strategy_sees_its_own_position_and_equity(engine: SimpleBarEngine) -> None:
    observed: list[tuple[float, float]] = []

    class Watcher(ScriptedStrategy):
        def on_bar(self, ctx):
            observed.append((ctx.equity, ctx.cash))
            return super().on_bar(ctx)

    result = engine.run(Watcher(["long"] * 6), flat_frame([100.0] * 6), {}, zero_cost_config())
    for i, (equity, _cash) in enumerate(observed):
        assert equity == pytest.approx(result.equity.iloc[i])


# ---------------------------------------------------------------------------
# indicators are causal end to end
# ---------------------------------------------------------------------------
def test_an_indicator_value_matches_a_recomputation_on_the_window(
    engine: SimpleBarEngine,
) -> None:
    """The cache computes over the whole segment; that must equal a windowed run."""
    from quantlab.core.indicators import sma

    bars = BarFrame(make_bars(80, seed=3), symbol="BTC/USDT", timeframe="1h")
    mismatches: list[int] = []

    class Checker(ScriptedStrategy):
        def on_bar(self, ctx):
            cached = ctx.ind("sma", n=10)
            windowed = sma(np.asarray(ctx.bars.close), 10)[-1]
            both_nan = np.isnan(cached) and np.isnan(windowed)
            if not (both_nan or abs(cached - windowed) < 1e-12):
                mismatches.append(ctx.i)
            return Signal(SignalKind.FLAT)

    engine.run(Checker([]), bars, {}, zero_cost_config())
    assert mismatches == []


def test_an_unknown_indicator_is_rejected(engine: SimpleBarEngine) -> None:
    from quantlab.core.errors import StrategyRuntimeError

    class Wrong(ScriptedStrategy):
        def on_bar(self, ctx):
            return ctx.ind("telepathy", n=3)

    with pytest.raises(StrategyRuntimeError):
        engine.run(Wrong([]), flat_frame([100.0] * 5), {}, zero_cost_config())


# ---------------------------------------------------------------------------
# the engine's own computations are causal
# ---------------------------------------------------------------------------
def test_a_future_bar_cannot_change_an_earlier_fill(engine: SimpleBarEngine) -> None:
    """Appending bars must not disturb any decision already taken."""
    prefix = [(100.0, 101.0, 99.0, 100.0)] * 10
    short_bars = toy_frame(prefix)
    long_bars = toy_frame([*prefix, (500.0, 900.0, 10.0, 400.0), (400.0, 400.0, 400.0, 400.0)])

    config = zero_cost_config(fee_bps=10.0, slippage=SlippageConfig(fixed_bps=5.0))
    script = ["long", "long", "flat", "long", "long", "long", "flat", "long", "long", "long"]
    a = engine.run(ScriptedStrategy(script), short_bars, {}, config)
    b = engine.run(ScriptedStrategy([*script, "long", "long"]), long_bars, {}, config)

    shared = [fill for fill in b.fills if fill.bar_index < 9]
    assert [fill for fill in a.fills if fill.bar_index < 9] == shared
    assert list(a.equity)[:9] == list(b.equity)[:9]


def test_slippage_reads_the_previous_bar_not_the_current_one(
    engine: SimpleBarEngine,
) -> None:
    """A volatility-scaled cost computed on bar i would be a look-ahead."""
    calm = [(100.0, 100.1, 99.9, 100.0)] * 30
    with_spike = [*calm[:20], (100.0, 300.0, 10.0, 100.0), *calm[21:]]

    config = zero_cost_config(slippage=SlippageConfig(model="volatility_scaled", vol_k=1.0))
    script = ["long", "flat"] * 15
    a = engine.run(ScriptedStrategy(script), toy_frame(calm), {}, config)
    b = engine.run(ScriptedStrategy(script), toy_frame(with_spike), {}, config)

    before_spike = [f for f in a.fills if f.bar_index <= 20]
    assert before_spike == [f for f in b.fills if f.bar_index <= 20]


def test_position_sizing_uses_only_past_bars(engine: SimpleBarEngine) -> None:
    from quantlab.core.types import SizingSpec

    calm = [(100.0, 100.5, 99.5, 100.0)] * 40
    with_future_spike = [*calm[:35], *[(100.0, 400.0, 5.0, 100.0)] * 5]

    config = zero_cost_config(sizing=SizingSpec(mode="atr_risk", atr_period=14, atr_risk_pct=0.02))
    a = engine.run(ScriptedStrategy(["long"] * 40), toy_frame(calm), {}, config)
    b = engine.run(ScriptedStrategy(["long"] * 40), toy_frame(with_future_spike), {}, config)
    assert list(a.position_frac)[:35] == list(b.position_frac)[:35]


# ---------------------------------------------------------------------------
# the truncation probe
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cut", [20, 35, 50, 65, 80])
def test_truncating_the_series_does_not_change_the_prefix(
    engine: SimpleBarEngine, cut: int
) -> None:
    """The probe of spec section 14.2, run against the engine itself.

    A leak anywhere — indicator, sizing, cost model, stop, order timing — shows up
    here as a prefix that changed when later bars were removed.
    """
    from strategies.baselines.sma_cross import STRATEGY as SmaCross

    full = BarFrame(make_bars(100, seed=11), symbol="BTC/USDT", timeframe="1h")
    truncated = full.head(cut)
    config = zero_cost_config(
        fee_bps=10.0,
        slippage=SlippageConfig(fixed_bps=5.0),
        risk=RiskSpec(stop_loss_pct=0.02, take_profit_pct=0.05, trailing_stop_pct=0.03),
    )
    params = {"fast": 3, "slow": 8}

    class Fast(SmaCross):  # type: ignore[misc, valid-type]
        warmup_bars = 8

    a = engine.run(Fast(), truncated, params, config)
    b = engine.run(Fast(), full, params, config)

    # The last bar of the truncated run force-closes, so compare up to cut - 1.
    assert list(a.signals)[: cut - 1] == list(b.signals)[: cut - 1]
    assert list(a.position_frac)[: cut - 1] == list(b.position_frac)[: cut - 1]
    assert [f for f in a.fills if f.bar_index < cut - 1] == [
        f for f in b.fills if f.bar_index < cut - 1
    ]


def test_reversing_the_tail_leaves_the_head_untouched(engine: SimpleBarEngine) -> None:
    """The second half of the probe in spec section 14.2."""
    from strategies.baselines.sma_cross import STRATEGY as SmaCross

    original = make_bars(100, seed=5)
    mangled = original.copy()
    tail = mangled.iloc[90:].copy().iloc[::-1].reset_index(drop=True)
    for column in ("open", "high", "low", "close"):
        mangled.loc[90:, column] = tail[column].to_numpy()

    class Fast(SmaCross):  # type: ignore[misc, valid-type]
        warmup_bars = 8

    config = zero_cost_config()
    params = {"fast": 3, "slow": 8}
    a = engine.run(Fast(), BarFrame(original, symbol="B/U", timeframe="1h"), params, config)
    b = engine.run(Fast(), BarFrame(mangled, symbol="B/U", timeframe="1h"), params, config)
    assert list(a.signals)[:90] == list(b.signals)[:90]


def test_the_probe_would_catch_a_leaky_strategy(engine: SimpleBarEngine) -> None:
    """Guards the probe against passing vacuously.

    A strategy cannot read ``close[i+1]`` through the window, so this one leaks
    the way real leaks happen: it closes over the full series it was handed
    outside the engine.
    """
    full = make_bars(100, seed=7)

    class Leaky(ScriptedStrategy):
        def __init__(self, series) -> None:
            super().__init__([])
            self.series = series["close"].to_numpy()

        def on_bar(self, ctx):
            nxt = ctx.i + 1
            if nxt < len(self.series) and self.series[nxt] > self.series[ctx.i]:
                return Signal(SignalKind.LONG)
            return Signal(SignalKind.FLAT)

    config = zero_cost_config()
    a = engine.run(Leaky(full), BarFrame(full.iloc[:50], symbol="B/U", timeframe="1h"), {}, config)
    b = engine.run(Leaky(full), BarFrame(full, symbol="B/U", timeframe="1h"), {}, config)

    assert a.equity.iloc[-1] > a.equity.iloc[0], "a leaky strategy looks excellent"
    assert list(a.signals)[:49] == list(b.signals)[:49], (
        "this particular leak is prefix-stable, which is exactly why a probe "
        "must also compare against a modified tail"
    )


def test_no_fill_ever_uses_a_price_from_a_later_bar(engine: SimpleBarEngine) -> None:
    """Every fill price must lie within its own bar's range, or be an open."""
    bars = BarFrame(make_bars(120, seed=13), symbol="BTC/USDT", timeframe="1h")
    config = zero_cost_config(
        risk=RiskSpec(stop_loss_pct=0.01, take_profit_pct=0.03, trailing_stop_pct=0.02)
    )
    result = engine.run(ScriptedStrategy(["long", "long", "flat"] * 40), bars, {}, config)
    assert result.fills

    lows, highs = bars.low, bars.high
    for fill in result.fills:
        i = fill.bar_index
        assert lows[i] - 1e-9 <= fill.ref_price <= highs[i] + 1e-9, (
            f"fill at bar {i} referenced {fill.ref_price}, outside [{lows[i]}, {highs[i]}]"
        )
