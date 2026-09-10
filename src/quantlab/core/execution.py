"""The fill arithmetic, in one place (master spec sections 8.4 and 16.2).

Section 16.2 requires a paper session to fill "with the same fee/slippage model
as backtests", and section 16's acceptance criterion is stronger than that: the
trades a replayed stream produces must equal the engine's on the same bars,
**bit for bit on quantity and price**.

A second implementation cannot keep that promise. Not because anyone would write
it carelessly, but because the two would then drift independently — a rounding
change made in one, a fee applied before sizing rather than after in the other —
and the divergence would show up as a paper session that quietly disagrees with
the backtest that authorised it. Which is the one comparison paper trading
exists to make.

So the arithmetic lives here, as pure functions over floats, and both
``adapters/engine/simple_bar.py`` and ``paper/broker.py`` call *these*. Neither
owns them. What stays in each is the bookkeeping — trades, equity curves,
persisted state — which legitimately differs between simulating a completed
history and following a stream.

Everything here is deliberately free of :class:`BarFrame`, position state and
configuration objects beyond the two the numbers actually depend on, so each
function can be checked against arithmetic rather than against a fixture.
"""

from __future__ import annotations

import math
from typing import Final

from quantlab.core.types import BacktestConfig

__all__ = [
    "EPS",
    "REBALANCE_EPS",
    "fee_of",
    "needs_order",
    "round_to_lot",
    "size_from_notional",
    "slipped_price",
]

#: The tolerance below which a quantity or price is treated as zero. Shared so
#: that "did this fill happen?" has one answer on both sides.
EPS: Final[float] = 1e-12

#: Exposure changes smaller than this do not justify a round trip of costs.
REBALANCE_EPS: Final[float] = 0.01


def needs_order(desired_fraction: float, current_fraction: float) -> bool:
    """Whether a change from ``current_fraction`` to ``desired_fraction`` is worth trading.

    A deadband, and it is not an optimisation — it is part of what the strategy
    *is*. Without it, a held position is re-sized on every single bar, because a
    fraction computed from a moving mark price is never exactly the fraction it
    was, and each of those micro-adjustments pays a full round trip of fees and
    slippage. A strategy that entered once and held for a month would post
    hundreds of trades and lose money doing nothing.

    Two changes are always worth trading regardless of size: crossing zero, and
    closing out. Both are changes of *state* rather than of degree, and a
    deadband that swallowed "go flat" would leave a position open that the
    strategy asked to close.

    The paper runtime calls this so that a live session opens and closes on the
    same bars its backtest did (section 16.2). Restating the rule there would
    make the two diverge on exactly the bars where they are hardest to compare.
    """
    crosses_zero = (desired_fraction > 0 > current_fraction) or (
        desired_fraction < 0 < current_fraction
    )
    closes_out = abs(desired_fraction) <= EPS and abs(current_fraction) > EPS
    return crosses_zero or closes_out or abs(desired_fraction - current_fraction) > REBALANCE_EPS


def slipped_price(
    reference: float, slip_bps: float, *, buying: bool, cost_multiplier: float
) -> float:
    """``reference`` moved against the trader by ``slip_bps`` basis points.

    Always adverse, in both directions: a buyer pays more and a seller receives
    less. Slippage that helped would make a cost stress test (section 14.3's
    ``G_COST``) report an improvement, which is not a failure mode worth having.

    ``slip_bps`` is clamped at zero for the same reason — a slippage model that
    returned a negative number would otherwise be a source of free money.
    """
    slip = max(0.0, slip_bps) * cost_multiplier / 1e4
    return reference * (1.0 + slip) if buying else reference * (1.0 - slip)


def fee_rate_of(config: BacktestConfig) -> float:
    """The per-side fee as a fraction, not basis points."""
    return config.fee_bps * config.cost_multiplier / 1e4


def fee_of(qty: float, price: float, config: BacktestConfig) -> float:
    """What ``qty`` at ``price`` costs to transact.

    Written out rather than as ``notional * fee_rate_of(config)``, and the
    difference is not stylistic: floating-point multiplication does not
    associate, so the two group the same factors differently and disagree in the
    last bit. The golden baselines of section 20 record trade prices exactly, and
    they caught the regrouped form immediately. This order is the engine's
    original one, and it is what "bit for bit" in section 16.2 is measured
    against.
    """
    return abs(qty) * price * config.fee_bps * config.cost_multiplier / 1e4


def size_from_notional(target_notional: float, price: float, config: BacktestConfig) -> float:
    """The quantity whose cost, fee included, is ``target_notional``.

    Dividing by ``1 + fee_rate`` rather than sizing on the price alone is what
    keeps a 100 % target at exactly 100 % of equity. Without it the fee is
    financed: cash goes negative, the realised position fraction exceeds
    ``max_position_fraction``, and section 14.3's ``G_SANITY`` gate fails a
    strategy that did nothing wrong.
    """
    if price <= EPS:
        return 0.0
    return target_notional / (price * (1.0 + fee_rate_of(config)))


def round_to_lot(qty: float, lot_step: float) -> float:
    """``qty`` truncated towards zero to a whole number of lots.

    Truncated, never rounded to nearest: rounding up would fill a quantity the
    account cannot pay for, and doing it on the way out would sell more than is
    held. The ``1e-9`` allows a quantity that is a lot multiple in exact
    arithmetic but a hair under it in binary floating point — without it,
    ``0.3 / 0.1`` yields two lots rather than three.
    """
    if lot_step <= EPS:
        return float(qty)
    steps = math.floor(abs(qty) / lot_step + 1e-9)
    return math.copysign(steps * lot_step, qty)
