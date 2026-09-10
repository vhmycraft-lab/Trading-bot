"""The paper broker (master spec sections 16.1, 16.2; INV-1).

The only implementation of :class:`~quantlab.ports.broker.Broker` there is, and
the only one there may be.

Its correctness condition is stated in section 16's acceptance criterion and is
unusually sharp: *the trades a replayed stream produces must equal the engine's
trades on the same bars, bit for bit on quantity and price.* That is not a
tolerance — it is equality — and it is only reachable because this class does
not do its own arithmetic. Prices, sizes, fees and lot rounding all come from
:mod:`quantlab.core.execution`, which ``adapters/engine/simple_bar.py`` also
calls. A paper broker that re-derived any of them would agree with the backtest
right up until one of the two was edited.

What is genuinely this class's own is the *timing*, and it matches section 8.4
exactly: an order submitted at a bar's close fills at the **next** bar's open.
Never on submission, never at the close it was decided on. That single rule is
most of what separates an honest paper session from one that quietly trades on
information the backtest never had.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from quantlab.core.execution import (
    EPS,
    fee_of,
    round_to_lot,
    size_from_notional,
    slipped_price,
)
from quantlab.core.types import BacktestConfig, Bar, Fill, Order, Position, Side

__all__ = ["PaperBroker"]


@dataclass
class PaperBroker:
    """Fills at the next bar's open, with the engine's cost model."""

    config: BacktestConfig
    cash: float
    #: While warm, bars advance indicator state and **nothing fills** (section
    #: 16.2's backfill replay). The flag is on the broker rather than on the
    #: runtime so that "did this bar trade?" has one answer, in the object that
    #: would have done the trading.
    warm: bool = False

    position_qty: float = 0.0
    entry_px: float = 0.0
    entry_bar: int = -1
    bar_index: int = -1
    total_fees: float = 0.0
    total_slippage: float = 0.0
    fills: list[Fill] = field(default_factory=list)
    _pending: Order | None = None
    _pending_notional: float = 0.0

    # -- the port -----------------------------------------------------------
    def submit(self, order: Order, *, target_notional: float | None = None) -> None:
        """Accept ``order`` for the next bar's open, replacing any pending one.

        Replacing rather than queueing: a strategy is asked once per bar what it
        wants to hold, so two pending orders would mean two answers to one
        question and the older one is simply stale. Queueing them would fill
        both at the same open and double the position.
        """
        self._pending = order
        self._pending_notional = (
            float(target_notional) if target_notional is not None else order.target_qty
        )

    def on_bar_open(self, bar: Bar) -> list[Fill]:
        """Fill whatever is pending at ``bar.open``."""
        self.bar_index += 1
        order, self._pending = self._pending, None
        if order is None or self.warm:
            return []

        reference = float(bar.open)
        if reference <= EPS:
            # A non-positive open is bad data, not a trading opportunity. The
            # engine drops the order rather than filling at zero, and so does
            # this — the two must agree even about what they refuse to do.
            return []

        produced: list[Fill] = []
        target_notional = self._pending_notional
        current_qty = self.position_qty

        # A reversal is two fills at the same open — close to flat, then open
        # the other way — because two fills is two fee charges, which is what
        # really happens. Section 8.4, and the engine does the same.
        reversing = (current_qty * target_notional < 0) or (
            abs(target_notional) <= EPS and abs(current_qty) > EPS
        )
        if reversing:
            price = self._price(reference, buying=current_qty < 0)
            fill = self._fill(bar, -current_qty, price, reference)
            if fill is not None:
                produced.append(fill)
            current_qty = 0.0

        provisional = self._price(reference, buying=target_notional > current_qty * reference)
        if provisional <= EPS:
            return produced
        target_qty = size_from_notional(target_notional, provisional, self.config)

        delta = round_to_lot(target_qty - current_qty, self.config.lot_step)
        if abs(delta) <= EPS:
            return produced

        price = self._price(reference, buying=delta > 0)
        fill = self._fill(bar, delta, price, reference)
        if fill is not None:
            produced.append(fill)
        return produced

    def position(self) -> Position:
        if abs(self.position_qty) <= EPS:
            return Position.flat()
        return Position(
            side=Side.LONG if self.position_qty > 0 else Side.SHORT,
            qty=abs(self.position_qty),
            entry_px=self.entry_px,
            entry_bar=self.entry_bar,
        )

    def equity(self, mark_price: float) -> float:
        return self.cash + self.position_qty * float(mark_price)

    # -- internals ----------------------------------------------------------
    def _price(self, reference: float, *, buying: bool) -> float:
        """The fill price for this bar.

        A live session has no next bar to measure volatility or volume against,
        so the configured *fixed* rate is used and a volatility- or volume-scaled
        model degrades to its base rate. That is a real limitation and it is
        stated rather than hidden: a paper session under a scaled model will not
        match its backtest fill-for-fill, and the replay test therefore pins the
        fixed model, where the two are identical by construction.
        """
        return slipped_price(
            reference,
            self.config.slippage.fixed_bps,
            buying=buying,
            cost_multiplier=self.config.cost_multiplier,
        )

    def _fill(self, bar: Bar, delta_qty: float, price: float, reference: float) -> Fill | None:
        if abs(delta_qty) <= EPS or price <= EPS:
            return None

        buying = delta_qty > 0
        qty = abs(delta_qty)

        # The minimum-notional rule applies to opening and increasing fills
        # only. Enforcing it on the way out would trap a position that had
        # shrunk below the exchange minimum, which no exchange actually does.
        increasing = abs(self.position_qty + delta_qty) > abs(self.position_qty) + EPS
        if increasing and qty * price < self.config.min_notional:
            return None

        fee = fee_of(qty, price, self.config)
        slippage_cost = abs(price - reference) * qty

        previous_qty = self.position_qty
        self.cash += (-qty * price - fee) if buying else (qty * price - fee)
        self.position_qty = round_to_lot(previous_qty + delta_qty, self.config.lot_step)
        self.total_fees += fee
        self.total_slippage += slippage_cost

        if abs(previous_qty) <= EPS and abs(self.position_qty) > EPS:
            self.entry_px = price
            self.entry_bar = self.bar_index

        fill = Fill(
            bar_index=self.bar_index,
            ts=int(bar.ts_open),
            side=Side.LONG if buying else Side.SHORT,
            qty=qty,
            ref_price=reference,
            fill_price=price,
            fee=fee,
            slippage_cost=slippage_cost,
        )
        self.fills.append(fill)
        return fill
