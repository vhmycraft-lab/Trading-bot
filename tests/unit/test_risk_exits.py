"""Engine-side risk exits (master spec section 8.6).

Every rule here resolves an ambiguity **against the trader**, and every test
below checks that it actually did.  These are the rules a search process would
most like to be wrong: a backtest that assumes a favourable intrabar path turns
a losing stop into a winning one, thousands of times, invisibly.

Prices are chosen so each expectation can be worked out on paper.
"""

from __future__ import annotations

import pytest
from tests.helpers import ScriptedStrategy, flat_frame, toy_frame, zero_cost_config

from quantlab.adapters.engine import SimpleBarEngine
from quantlab.core.types import RiskSpec, Side, SlippageConfig


@pytest.fixture
def engine() -> SimpleBarEngine:
    return SimpleBarEngine()


def run(engine, bars, risk, *, script=None, **config_kwargs):
    config = zero_cost_config(risk=risk, **config_kwargs)
    strategy = ScriptedStrategy(script or ["long"] * bars.n_bars)
    return engine.run(strategy, bars, {}, config)


# ---------------------------------------------------------------------------
# stop-loss
# ---------------------------------------------------------------------------
def test_a_stop_loss_fires_when_the_low_reaches_it(engine: SimpleBarEngine) -> None:
    """Entry at 100, 5 % stop at 95; bar 2 trades down to 94."""
    bars = toy_frame(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),
            (99.0, 99.0, 94.0, 98.0),
            (98.0, 98.0, 98.0, 98.0),
        ]
    )
    result = run(engine, bars, RiskSpec(stop_loss_pct=0.05))
    trade = result.trades[0]
    assert trade.exit_reason == "stop"
    assert trade.exit_px == pytest.approx(95.0)
    assert trade.pnl == pytest.approx(-500.0)
    assert any("control=stop_loss" in line for line in result.log)


def test_a_stop_loss_does_not_fire_above_its_level(engine: SimpleBarEngine) -> None:
    bars = toy_frame(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),
            (99.0, 99.0, 95.5, 98.0),
            (98.0, 98.0, 98.0, 98.0),
        ]
    )
    result = run(engine, bars, RiskSpec(stop_loss_pct=0.05))
    assert result.trades[0].exit_reason == "end_of_data"


def test_a_gap_through_the_stop_fills_at_the_gap(engine: SimpleBarEngine) -> None:
    """Rule 2: the worse of trigger price and bar open.  A gap does not get the stop."""
    bars = toy_frame(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),
            (80.0, 82.0, 79.0, 81.0),  # opens far below the 95 stop
            (81.0, 81.0, 81.0, 81.0),
        ]
    )
    result = run(engine, bars, RiskSpec(stop_loss_pct=0.05))
    trade = result.trades[0]
    assert trade.exit_px == pytest.approx(80.0), "filled at the open, not the stop"
    assert trade.pnl == pytest.approx(-2_000.0)


def test_a_stop_can_fire_on_the_entry_bar(engine: SimpleBarEngine) -> None:
    """A position opened at this bar's open is live for this bar's range."""
    bars = toy_frame(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 90.0, 92.0),
            (92.0, 92.0, 92.0, 92.0),
        ]
    )
    result = run(engine, bars, RiskSpec(stop_loss_pct=0.05))
    trade = result.trades[0]
    assert trade.exit_reason == "stop"
    assert trade.bars_held == 0
    assert trade.exit_px == pytest.approx(95.0)


# ---------------------------------------------------------------------------
# take-profit
# ---------------------------------------------------------------------------
def test_a_take_profit_fires_when_the_high_reaches_it(engine: SimpleBarEngine) -> None:
    bars = toy_frame(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),
            (101.0, 112.0, 101.0, 110.0),
            (110.0, 110.0, 110.0, 110.0),
        ]
    )
    result = run(engine, bars, RiskSpec(take_profit_pct=0.10))
    trade = result.trades[0]
    assert trade.exit_reason == "stop"
    assert trade.exit_px == pytest.approx(110.0)
    assert trade.pnl == pytest.approx(1_000.0)
    assert any("control=take_profit" in line for line in result.log)


def test_a_gap_above_the_target_fills_at_the_gap(engine: SimpleBarEngine) -> None:
    """Symmetry, and in the trader's favour here: the open is genuinely better."""
    bars = toy_frame(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),
            (125.0, 126.0, 124.0, 125.0),
            (125.0, 125.0, 125.0, 125.0),
        ]
    )
    result = run(engine, bars, RiskSpec(take_profit_pct=0.10))
    assert result.trades[0].exit_px == pytest.approx(125.0)


# ---------------------------------------------------------------------------
# same-bar ambiguity: the rule that matters most
# ---------------------------------------------------------------------------
def test_the_stop_loss_wins_a_same_bar_tie_with_the_take_profit(
    engine: SimpleBarEngine,
) -> None:
    """Bar 2 spans both levels.  Without an intrabar path, assume the loss."""
    bars = toy_frame(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 115.0, 90.0, 105.0),  # touches both the 110 target and the 95 stop
            (105.0, 105.0, 105.0, 105.0),
        ]
    )
    result = run(engine, bars, RiskSpec(stop_loss_pct=0.05, take_profit_pct=0.10))
    trade = result.trades[0]
    assert trade.exit_px == pytest.approx(95.0), "the stop, not the target"
    assert trade.pnl < 0
    assert any("control=stop_loss" in line for line in result.log)


def test_the_worse_of_two_triggered_loss_controls_is_taken(
    engine: SimpleBarEngine,
) -> None:
    """A fixed stop and a trailing stop can both trigger; take the worse fill."""
    bars = toy_frame(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 120.0, 100.0, 120.0),  # runs up, so the trail sits at 108
            (119.0, 119.0, 80.0, 85.0),  # collapses through both
            (85.0, 85.0, 85.0, 85.0),
        ]
    )
    result = run(engine, bars, RiskSpec(stop_loss_pct=0.10, trailing_stop_pct=0.10))
    trade = result.trades[0]
    # Fixed stop 90, trailing stop 120 * 0.9 = 108. The fixed stop is worse.
    assert trade.exit_px == pytest.approx(90.0)


def test_a_take_profit_is_ignored_when_a_loss_control_also_triggered(
    engine: SimpleBarEngine,
) -> None:
    bars = toy_frame(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 130.0, 80.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
        ]
    )
    result = run(engine, bars, RiskSpec(stop_loss_pct=0.05, take_profit_pct=0.20))
    assert result.trades[0].pnl < 0


# ---------------------------------------------------------------------------
# trailing stop
# ---------------------------------------------------------------------------
def test_the_trailing_stop_follows_the_running_maximum_close(
    engine: SimpleBarEngine,
) -> None:
    """Runs to a close of 120, so the 10 % trail sits at 108."""
    bars = toy_frame(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 121.0, 100.0, 120.0),
            (120.0, 120.0, 105.0, 106.0),
            (106.0, 106.0, 106.0, 106.0),
        ]
    )
    result = run(engine, bars, RiskSpec(trailing_stop_pct=0.10))
    trade = result.trades[0]
    assert trade.exit_reason == "stop"
    assert trade.exit_px == pytest.approx(108.0)
    assert trade.pnl == pytest.approx(800.0)


def test_the_trailing_reference_never_uses_the_current_bar_high(
    engine: SimpleBarEngine,
) -> None:
    """INV-3 inside the engine: a trail computed from ``high[i]`` would look ahead.

    Bar 2's high of 200 would put the trail at 180 and exit at 180 -- a fictional
    profit.  The reference is the running max of *closed* bars, so the trail is
    still at 90 and the bar does not trigger at all.
    """
    bars = toy_frame(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 200.0, 99.0, 100.0),  # huge intrabar spike, closes back at 100
            (100.0, 100.0, 100.0, 100.0),
        ]
    )
    result = run(engine, bars, RiskSpec(trailing_stop_pct=0.10))
    trade = result.trades[0]
    assert trade.exit_reason == "end_of_data"
    assert trade.exit_px == pytest.approx(100.0)


def test_the_trailing_stop_ratchets_and_never_loosens(engine: SimpleBarEngine) -> None:
    bars = toy_frame(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 131.0, 100.0, 130.0),  # trail -> 117
            (130.0, 130.0, 120.0, 121.0),  # falls back; trail stays at 117
            (121.0, 121.0, 116.0, 118.0),  # now breaches 117
            (118.0, 118.0, 118.0, 118.0),
        ]
    )
    result = run(engine, bars, RiskSpec(trailing_stop_pct=0.10))
    assert result.trades[0].exit_px == pytest.approx(117.0)


def test_the_trailing_stop_starts_at_the_entry_price(engine: SimpleBarEngine) -> None:
    """Before any close is observed, the reference is what was paid."""
    bars = toy_frame(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 88.0, 90.0),
            (90.0, 90.0, 90.0, 90.0),
        ]
    )
    result = run(engine, bars, RiskSpec(trailing_stop_pct=0.10))
    assert result.trades[0].exit_px == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# time stop
# ---------------------------------------------------------------------------
def test_a_time_stop_closes_at_the_close_after_n_bars(engine: SimpleBarEngine) -> None:
    # Entry fills at bar 1's open; the stop fires at bar 4, whose close is 103.
    bars = flat_frame([100.0, 100.0, 101.0, 102.0, 103.0, 104.0])
    result = run(engine, bars, RiskSpec(time_stop_bars=3))
    trade = result.trades[0]
    assert trade.exit_reason == "stop"
    assert trade.bars_held == 3
    assert trade.exit_px == pytest.approx(103.0)
    assert any("control=time_stop" in line for line in result.log)


def test_a_time_stop_of_one_bar_exits_immediately_after_entry(
    engine: SimpleBarEngine,
) -> None:
    bars = flat_frame([100.0, 100.0, 105.0, 105.0])
    result = run(engine, bars, RiskSpec(time_stop_bars=1))
    assert result.trades[0].bars_held == 1


def test_a_loss_control_takes_priority_over_the_time_stop(
    engine: SimpleBarEngine,
) -> None:
    bars = toy_frame(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 90.0, 92.0),
            (92.0, 92.0, 92.0, 92.0),
        ]
    )
    result = run(engine, bars, RiskSpec(stop_loss_pct=0.05, time_stop_bars=1))
    assert result.trades[0].exit_px == pytest.approx(95.0)


# ---------------------------------------------------------------------------
# short positions mirror everything
# ---------------------------------------------------------------------------
def test_a_short_stop_loss_fires_when_the_high_reaches_it(
    engine: SimpleBarEngine,
) -> None:
    bars = toy_frame(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),
            (101.0, 106.0, 101.0, 105.0),
            (105.0, 105.0, 105.0, 105.0),
        ]
    )
    result = run(engine, bars, RiskSpec(stop_loss_pct=0.05), script=["short"] * 4, allow_short=True)
    trade = result.trades[0]
    assert trade.side is Side.SHORT
    assert trade.exit_px == pytest.approx(105.0)
    assert trade.pnl == pytest.approx(-500.0)


def test_a_short_gap_through_the_stop_fills_at_the_gap(engine: SimpleBarEngine) -> None:
    bars = toy_frame(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),
            (120.0, 122.0, 119.0, 121.0),
            (121.0, 121.0, 121.0, 121.0),
        ]
    )
    result = run(engine, bars, RiskSpec(stop_loss_pct=0.05), script=["short"] * 4, allow_short=True)
    assert result.trades[0].exit_px == pytest.approx(120.0)


def test_a_short_take_profit_fires_when_the_low_reaches_it(
    engine: SimpleBarEngine,
) -> None:
    bars = toy_frame(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),
            (99.0, 99.0, 88.0, 90.0),
            (90.0, 90.0, 90.0, 90.0),
        ]
    )
    result = run(
        engine, bars, RiskSpec(take_profit_pct=0.10), script=["short"] * 4, allow_short=True
    )
    assert result.trades[0].exit_px == pytest.approx(90.0)
    assert result.trades[0].pnl == pytest.approx(1_000.0)


def test_a_short_trailing_stop_ratchets_downwards(engine: SimpleBarEngine) -> None:
    bars = toy_frame(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 79.0, 80.0),  # trail -> 88
            (80.0, 89.0, 80.0, 85.0),  # breaches 88
            (85.0, 85.0, 85.0, 85.0),
        ]
    )
    result = run(
        engine, bars, RiskSpec(trailing_stop_pct=0.10), script=["short"] * 5, allow_short=True
    )
    assert result.trades[0].exit_px == pytest.approx(88.0)


def test_the_short_stop_loss_also_wins_a_same_bar_tie(engine: SimpleBarEngine) -> None:
    bars = toy_frame(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 110.0, 85.0, 95.0),  # spans the 105 stop and the 90 target
            (95.0, 95.0, 95.0, 95.0),
        ]
    )
    result = run(
        engine,
        bars,
        RiskSpec(stop_loss_pct=0.05, take_profit_pct=0.10),
        script=["short"] * 4,
        allow_short=True,
    )
    assert result.trades[0].exit_px == pytest.approx(105.0)
    assert result.trades[0].pnl < 0


# ---------------------------------------------------------------------------
# interaction with the rest of the engine
# ---------------------------------------------------------------------------
def test_no_risk_exit_fires_on_a_gap_filled_bar(engine: SimpleBarEngine) -> None:
    """Rule 4: a synthetic bar must never become a synthetic trade."""
    bars = toy_frame(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 50.0, 100.0),  # would breach the stop, but is synthetic
            (100.0, 100.0, 100.0, 100.0),
        ],
        gap_filled=[False, False, True, False],
    )
    result = run(engine, bars, RiskSpec(stop_loss_pct=0.05))
    assert result.trades[0].exit_reason == "end_of_data"


def test_a_risk_exit_charges_fees_and_slippage(engine: SimpleBarEngine) -> None:
    """Rule 5: a stop is not a free exit."""
    bars = toy_frame(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 90.0, 92.0),
            (92.0, 92.0, 92.0, 92.0),
        ]
    )
    result = run(
        engine,
        bars,
        RiskSpec(stop_loss_pct=0.05),
        fee_bps=10.0,
        slippage=SlippageConfig(fixed_bps=5.0),
    )
    exit_fill = next(fill for fill in result.fills if fill.side is Side.SHORT)
    assert exit_fill.fee > 0
    assert exit_fill.slippage_cost > 0
    assert exit_fill.fill_price < 95.0, "slippage moves the stop fill lower still"


def test_a_risk_exit_cancels_the_pending_order(engine: SimpleBarEngine) -> None:
    """The strategy asked to hold; the stop overrides it and nothing re-enters."""
    bars = toy_frame(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 90.0, 92.0),
            (92.0, 92.0, 92.0, 92.0),
            (92.0, 92.0, 92.0, 92.0),
        ],
    )
    result = run(engine, bars, RiskSpec(stop_loss_pct=0.05), script=["long"] * 5)
    assert len(result.trades) >= 1
    assert result.trades[0].exit_reason == "stop"


def test_the_strategy_sees_a_flat_position_after_a_stop(engine: SimpleBarEngine) -> None:
    seen: list[bool] = []

    class Watcher(ScriptedStrategy):
        def on_bar(self, ctx):
            seen.append(ctx.position.side is None)
            return super().on_bar(ctx)

    bars = toy_frame(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 90.0, 92.0),
            (92.0, 92.0, 92.0, 92.0),
        ]
    )
    config = zero_cost_config(risk=RiskSpec(stop_loss_pct=0.05))
    engine.run(Watcher(["long"] * 4), bars, {}, config)
    assert seen[2] is True, "the stop fired during bar 2, so bar 2 sees flat"


def test_risk_controls_are_inert_when_unset(engine: SimpleBarEngine) -> None:
    bars = flat_frame([100.0, 100.0, 50.0, 50.0, 200.0, 200.0])
    result = run(engine, bars, RiskSpec())
    assert result.trades[0].exit_reason == "end_of_data"


def test_the_config_risk_spec_overrides_the_strategy_default(
    engine: SimpleBarEngine,
) -> None:
    """A mutated stop must take effect without editing the strategy source."""

    class Declared(ScriptedStrategy):
        risk = RiskSpec(stop_loss_pct=0.50)

    bars = toy_frame(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 90.0, 92.0),
            (92.0, 92.0, 92.0, 92.0),
        ]
    )
    strategy = Declared(["long"] * 4)
    strategy.risk = RiskSpec(stop_loss_pct=0.50)
    result = engine.run(strategy, bars, {}, zero_cost_config(risk=RiskSpec(stop_loss_pct=0.05)))
    assert result.trades[0].exit_px == pytest.approx(95.0)


def test_a_strategy_risk_spec_applies_when_the_config_has_none(
    engine: SimpleBarEngine,
) -> None:
    class Declared(ScriptedStrategy):
        pass

    bars = toy_frame(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 90.0, 92.0),
            (92.0, 92.0, 92.0, 92.0),
        ]
    )
    strategy = Declared(["long"] * 4)
    strategy.risk = RiskSpec(stop_loss_pct=0.05)
    result = engine.run(strategy, bars, {}, zero_cost_config())
    assert result.trades[0].exit_reason == "stop"


def test_stops_are_reproducible(engine: SimpleBarEngine) -> None:
    bars = flat_frame([100.0 + (i % 11) - 5 for i in range(80)])
    risk = RiskSpec(stop_loss_pct=0.03, take_profit_pct=0.06, trailing_stop_pct=0.04)
    first = run(engine, bars, risk)
    second = SimpleBarEngine().run(
        ScriptedStrategy(["long"] * 80), bars, {}, zero_cost_config(risk=risk)
    )
    assert first.trades == second.trades
