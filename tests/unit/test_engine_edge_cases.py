"""Accounting and floating-point edge cases (master spec sections 8.4, 8.7).

The cases here are the ones a strategy search would eventually reach on its own:
degenerate frames, impossible sizes, and the arithmetic of an account that has
run out of money. Each one is somewhere the engine could quietly produce a
number that is not a price, a quantity, or an amount of equity.
"""

from __future__ import annotations

import numpy as np
import pytest
from tests.helpers import ScriptedStrategy, flat_frame, toy_frame, zero_cost_config

from quantlab.adapters.engine import SimpleBarEngine
from quantlab.core.metrics import compute_metrics
from quantlab.core.types import BarFrame, RiskSpec, SlippageConfig


@pytest.fixture
def engine() -> SimpleBarEngine:
    return SimpleBarEngine()


# ---------------------------------------------------------------------------
# ruin (spec section 8.7)
# ---------------------------------------------------------------------------
def test_a_short_that_blows_up_liquidates_the_account(engine: SimpleBarEngine) -> None:
    """Equity reaching zero ends the run; it does not go negative and carry on.

    Without this the engine keeps trading a negative balance, and any search
    would eventually discover that negative-equity arithmetic can be made to
    look profitable.
    """
    bars = flat_frame([100.0, 100.0, 200.0, 300.0, 400.0])
    result = engine.run(
        ScriptedStrategy(["short"] * 5), bars, {}, zero_cost_config(allow_short=True)
    )
    assert result.ruined is True
    assert float(result.equity.min()) >= 0.0
    assert list(result.position_frac)[2:] == [0.0, 0.0, 0.0]
    assert len(result.trades) == 1
    assert any("account_ruined" in line for line in result.log)


def test_a_ruined_account_stops_trading(engine: SimpleBarEngine) -> None:
    bars = flat_frame([100.0, 100.0, 250.0, 100.0, 100.0, 100.0])
    result = engine.run(
        ScriptedStrategy(["short"] * 6), bars, {}, zero_cost_config(allow_short=True)
    )
    assert result.ruined is True
    assert all(fill.bar_index <= 2 for fill in result.fills), "no trading after ruin"


def test_a_solvent_run_is_not_marked_ruined(engine: SimpleBarEngine) -> None:
    result = engine.run(
        ScriptedStrategy(["long"] * 6), flat_frame([100.0] * 6), {}, zero_cost_config()
    )
    assert result.ruined is False


def test_metrics_of_a_ruined_run_are_finite(engine: SimpleBarEngine) -> None:
    bars = flat_frame([100.0, 100.0, 300.0, 100.0, 100.0])
    result = engine.run(
        ScriptedStrategy(["short"] * 5), bars, {}, zero_cost_config(allow_short=True)
    )
    metrics = compute_metrics(result)
    # The gap past zero left the account owing money, which the equity series
    # records honestly; drawdown is still a fraction, capped at a total loss.
    assert float(result.equity.iloc[-1]) < 0.0
    assert metrics.max_drawdown == pytest.approx(1.0)
    assert metrics.net_return == pytest.approx(-2.0)
    assert metrics.n_trades == 1.0
    assert np.isfinite([metrics.max_drawdown, metrics.net_return]).all()


# ---------------------------------------------------------------------------
# degenerate frames
# ---------------------------------------------------------------------------
def test_a_single_bar_frame_produces_no_trades(engine: SimpleBarEngine) -> None:
    """There is no next bar to fill at, so nothing can be executed."""
    result = engine.run(ScriptedStrategy(["long"]), flat_frame([100.0]), {}, zero_cost_config())
    assert result.n_bars == 1
    assert result.fills == ()
    assert result.equity.iloc[0] == 10_000.0


def test_a_two_bar_frame_enters_and_immediately_closes(engine: SimpleBarEngine) -> None:
    result = engine.run(
        ScriptedStrategy(["long", "long"]), flat_frame([100.0, 100.0]), {}, zero_cost_config()
    )
    assert len(result.fills) == 2
    assert result.trades[0].exit_reason == "end_of_data"
    assert result.trades[0].bars_held == 0


def test_a_flat_market_returns_exactly_the_initial_equity(engine: SimpleBarEngine) -> None:
    """No costs, no movement: the equity curve must not drift by a rounding error."""
    result = engine.run(
        ScriptedStrategy(["long"] * 50), flat_frame([100.0] * 50), {}, zero_cost_config()
    )
    assert float(result.equity.iloc[-1]) == pytest.approx(10_000.0, abs=1e-9)


def test_a_warm_up_longer_than_the_series_never_trades(engine: SimpleBarEngine) -> None:
    result = engine.run(
        ScriptedStrategy(["long"] * 5, warmup=99), flat_frame([100.0] * 5), {}, zero_cost_config()
    )
    assert result.fills == ()
    assert set(result.signals) == {"flat"}


# ---------------------------------------------------------------------------
# extreme sizes
# ---------------------------------------------------------------------------
def test_a_lot_step_larger_than_the_position_drops_the_order(
    engine: SimpleBarEngine,
) -> None:
    """Rounding down to zero must be a dropped order, not a zero-quantity fill."""
    result = engine.run(
        ScriptedStrategy(["long"] * 5),
        flat_frame([100.0] * 5),
        {},
        zero_cost_config(lot_step=1_000.0),
    )
    assert result.fills == ()


def test_a_tiny_initial_equity_is_handled(engine: SimpleBarEngine) -> None:
    result = engine.run(
        ScriptedStrategy(["long"] * 5),
        flat_frame([100.0] * 5),
        {},
        zero_cost_config(initial_equity=0.01, min_notional=0.0, lot_step=1e-12),
    )
    assert float(result.equity.iloc[-1]) == pytest.approx(0.01, abs=1e-9)


def test_a_very_large_equity_keeps_its_precision(engine: SimpleBarEngine) -> None:
    result = engine.run(
        ScriptedStrategy(["long", "long", "flat", "flat"]),
        flat_frame([100.0] * 4),
        {},
        zero_cost_config(initial_equity=1e12, lot_step=1e-12),
    )
    assert float(result.equity.iloc[-1]) == pytest.approx(1e12, rel=1e-12)


def test_extreme_prices_do_not_break_the_accounting(engine: SimpleBarEngine) -> None:
    bars = toy_frame([(1e-4, 1e-4, 1e-4, 1e-4)] * 3 + [(1e6, 1e6, 1e6, 1e6)] * 3)
    result = engine.run(ScriptedStrategy(["long"] * 6), bars, {}, zero_cost_config(lot_step=1e-12))
    assert np.isfinite(result.equity.to_numpy()).all()
    assert float(result.equity.iloc[-1]) > 0


# ---------------------------------------------------------------------------
# costs at their limits
# ---------------------------------------------------------------------------
def test_a_fee_of_one_hundred_percent_still_reconciles(engine: SimpleBarEngine) -> None:
    result = engine.run(
        ScriptedStrategy(["long", "long", "flat", "flat"]),
        flat_frame([100.0] * 4),
        {},
        zero_cost_config(fee_bps=10_000.0, lot_step=1e-12, min_notional=0.0),
    )
    change = float(result.equity.iloc[-1] - result.equity.iloc[0])
    assert change == pytest.approx(sum(t.pnl for t in result.trades), abs=1e-6)
    assert result.equity.iloc[-1] < 10_000.0


def test_zero_costs_leave_a_round_trip_exactly_flat(engine: SimpleBarEngine) -> None:
    result = engine.run(
        ScriptedStrategy(["long", "long", "flat", "flat"]),
        flat_frame([100.0] * 4),
        {},
        zero_cost_config(),
    )
    assert float(result.equity.iloc[-1]) == pytest.approx(10_000.0, abs=1e-9)
    assert result.trades[0].pnl == pytest.approx(0.0, abs=1e-9)


def test_a_huge_slippage_never_inverts_the_price(engine: SimpleBarEngine) -> None:
    """A cost so large it would drive a sell price negative must not do so."""
    result = engine.run(
        ScriptedStrategy(["long", "long", "flat", "flat"]),
        flat_frame([100.0] * 4),
        {},
        zero_cost_config(
            slippage=SlippageConfig(fixed_bps=20_000.0), lot_step=1e-12, min_notional=0.0
        ),
    )
    for fill in result.fills:
        assert fill.fill_price > 0.0


# ---------------------------------------------------------------------------
# risk controls at their limits
# ---------------------------------------------------------------------------
def test_a_stop_at_the_entry_price_is_rejected_by_the_spec() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RiskSpec(stop_loss_pct=0.0)


def test_a_very_tight_stop_exits_on_the_entry_bar(engine: SimpleBarEngine) -> None:
    bars = toy_frame(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.1, 99.9, 100.0),
            (100.0, 100.0, 100.0, 100.0),
        ]
    )
    result = engine.run(
        ScriptedStrategy(["long"] * 3),
        bars,
        {},
        zero_cost_config(risk=RiskSpec(stop_loss_pct=0.0005)),
    )
    assert result.trades[0].exit_reason == "stop"
    assert result.trades[0].bars_held == 0


def test_every_risk_control_at_once_still_reconciles(engine: SimpleBarEngine) -> None:
    from tests.helpers import make_bars

    bars = BarFrame(make_bars(200, seed=21), symbol="BTC/USDT", timeframe="1h")
    risk = RiskSpec(
        stop_loss_pct=0.01, take_profit_pct=0.02, trailing_stop_pct=0.015, time_stop_bars=10
    )
    result = engine.run(
        ScriptedStrategy(["long"] * 200),
        bars,
        {},
        zero_cost_config(risk=risk, fee_bps=10.0, slippage=SlippageConfig(fixed_bps=5.0)),
    )
    change = float(result.equity.iloc[-1] - result.equity.iloc[0])
    assert change == pytest.approx(sum(t.pnl for t in result.trades), abs=1e-6)
    assert len(result.trades) > 1


# ---------------------------------------------------------------------------
# the engine log
# ---------------------------------------------------------------------------
def test_the_log_is_bounded(engine: SimpleBarEngine) -> None:
    from tests.helpers import make_bars

    from quantlab.core.types import MAX_ENGINE_LOG_LINES

    bars = BarFrame(make_bars(3_000, seed=1), symbol="BTC/USDT", timeframe="1h")
    result = engine.run(ScriptedStrategy(["long", "flat"] * 1_500), bars, {}, zero_cost_config())
    assert len(result.log) <= MAX_ENGINE_LOG_LINES


def test_the_log_records_why_an_order_was_dropped(engine: SimpleBarEngine) -> None:
    result = engine.run(
        ScriptedStrategy(["long"] * 4),
        flat_frame([100.0] * 4),
        {},
        zero_cost_config(min_notional=1e9),
    )
    assert any("below_min_notional" in line for line in result.log)
