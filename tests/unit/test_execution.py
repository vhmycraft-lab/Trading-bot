"""The shared fill arithmetic (spec sections 8.4 and 16.2).

These functions exist so that the backtest engine and the paper broker cannot
disagree. They are pure and take floats, so every assertion here is against
arithmetic worked out on paper rather than against a fixture — which is the
point: a fixture would only tell you the two agree with *each other*.
"""

from __future__ import annotations

import pytest

from quantlab.core.execution import (
    EPS,
    REBALANCE_EPS,
    fee_of,
    needs_order,
    round_to_lot,
    size_from_notional,
    slipped_price,
)
from quantlab.core.types import BacktestConfig, SlippageConfig

FREE = BacktestConfig(fee_bps=0.0, slippage=SlippageConfig(model="fixed_bps", fixed_bps=0.0))
TEN_BPS = BacktestConfig(fee_bps=10.0)


# ---------------------------------------------------------------------------
# slippage is always adverse
# ---------------------------------------------------------------------------
def test_a_buyer_pays_more_and_a_seller_receives_less() -> None:
    assert slipped_price(100.0, 10.0, buying=True, cost_multiplier=1.0) == pytest.approx(100.1)
    assert slipped_price(100.0, 10.0, buying=False, cost_multiplier=1.0) == pytest.approx(99.9)


def test_negative_slippage_is_clamped_rather_than_paid_out() -> None:
    """A slippage model returning a negative number would be a source of free
    money, and a cost stress test (``G_COST``) would report an improvement."""
    assert slipped_price(100.0, -50.0, buying=True, cost_multiplier=1.0) == 100.0
    assert slipped_price(100.0, -50.0, buying=False, cost_multiplier=1.0) == 100.0


def test_the_cost_multiplier_scales_slippage() -> None:
    """Section 14.3's ``G_COST`` runs the same strategy at 2x costs; if the
    multiplier did not reach slippage, the stress would only test fees."""
    doubled = slipped_price(100.0, 10.0, buying=True, cost_multiplier=2.0)
    assert doubled == pytest.approx(100.2)


# ---------------------------------------------------------------------------
# fees
# ---------------------------------------------------------------------------
def test_a_fee_is_charged_on_notional() -> None:
    assert fee_of(2.0, 100.0, TEN_BPS) == pytest.approx(0.2)


def test_a_sale_is_charged_the_same_as_a_purchase() -> None:
    """Fees are per side and do not care about direction; charging only buys
    would halve the round-trip cost of every trade in the platform."""
    assert fee_of(-2.0, 100.0, TEN_BPS) == fee_of(2.0, 100.0, TEN_BPS)


def test_the_multiplication_order_is_the_engine_s_own() -> None:
    """Floating-point multiplication does not associate.

    ``(qty * price * bps * mult) / 1e4`` and ``notional * (bps * mult / 1e4)``
    differ in the last bit, and the golden baselines of section 20 record trade
    prices exactly. This pins the grouping rather than leaving it to whoever
    next tidies the expression.
    """
    qty, price = 2.7182818, 31_415.926
    expected = qty * price * TEN_BPS.fee_bps * TEN_BPS.cost_multiplier / 1e4
    assert fee_of(qty, price, TEN_BPS) == expected


# ---------------------------------------------------------------------------
# sizing
# ---------------------------------------------------------------------------
def test_a_full_target_stays_within_the_account_once_the_fee_is_paid() -> None:
    """Dividing by ``1 + fee_rate`` is what keeps a 100 % target at exactly
    100 % of equity. Without it the fee is financed: cash goes negative, the
    realised fraction exceeds ``max_position_fraction``, and ``G_SANITY`` fails
    a strategy that did nothing wrong."""
    equity, price = 10_000.0, 100.0
    qty = size_from_notional(equity, price, TEN_BPS)
    assert qty * price + fee_of(qty, price, TEN_BPS) == pytest.approx(equity)


def test_with_no_fee_the_size_is_just_the_division() -> None:
    assert size_from_notional(10_000.0, 100.0, FREE) == pytest.approx(100.0)


def test_a_non_positive_price_sizes_to_nothing_rather_than_dividing_by_zero() -> None:
    assert size_from_notional(10_000.0, 0.0, TEN_BPS) == 0.0


# ---------------------------------------------------------------------------
# lot rounding
# ---------------------------------------------------------------------------
def test_a_quantity_is_truncated_towards_zero() -> None:
    """Rounding up would fill a quantity the account cannot pay for, and doing
    it on the way out would sell more than is held."""
    assert round_to_lot(1.29, 0.1) == pytest.approx(1.2)
    assert round_to_lot(-1.29, 0.1) == pytest.approx(-1.2)


def test_a_binary_representation_hair_under_a_lot_multiple_still_counts() -> None:
    """``0.3 / 0.1`` is 2.9999999999999996 in binary floating point. Without the
    tolerance, a strategy asking for exactly three lots gets two."""
    assert round_to_lot(0.3, 0.1) == pytest.approx(0.3)


def test_a_lot_step_of_zero_leaves_the_quantity_alone() -> None:
    assert round_to_lot(1.23456789, 0.0) == 1.23456789


# ---------------------------------------------------------------------------
# the deadband
# ---------------------------------------------------------------------------
def test_a_tiny_drift_does_not_justify_a_round_trip() -> None:
    """The reason the deadband exists. A held position's fraction moves with the
    mark price on every bar; without this, a strategy that entered once and held
    for a month posts hundreds of trades and loses money doing nothing."""
    assert not needs_order(0.500, 0.505)


def test_a_real_change_of_exposure_does_trade() -> None:
    assert needs_order(1.0, 0.5)


def test_closing_out_is_always_worth_trading() -> None:
    """A change of state, not of degree. A deadband that swallowed "go flat"
    would leave a position open that the strategy asked to close."""
    assert needs_order(0.0, 0.001)


def test_crossing_zero_is_always_worth_trading() -> None:
    assert needs_order(0.005, -0.005)


def test_the_deadband_boundary_is_open_on_the_side_that_refuses() -> None:
    """A change of exactly the deadband does not trade; anything larger does.

    An off-by-one here is a fee charged on every bar, or a signal ignored.
    Stated from zero rather than as ``0.5 + REBALANCE_EPS`` versus ``0.5``:
    that difference is ``0.010000000000000009`` in binary floating point, so it
    would test the representation rather than the rule.
    """
    assert not needs_order(REBALANCE_EPS, 0.0)
    assert needs_order(REBALANCE_EPS * 2, 0.0)


def test_holding_flat_is_not_a_trade() -> None:
    """Guards the close-out rule against firing when there is nothing to close;
    otherwise a flat strategy submits an order on every bar for ever."""
    assert not needs_order(0.0, 0.0)
    assert not needs_order(0.0, EPS / 2)
