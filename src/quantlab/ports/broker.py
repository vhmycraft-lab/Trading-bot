"""Broker and live-feed ports (master spec section 16.1).

**INV-1 lives here.** The invariant says this codebase "has no real-money order
path", and the load-bearing part of that claim is not a missing class — it is
that :class:`Broker` has no method that could send anything anywhere. Look at
what it exposes: submit an order *into this object*, ask it what filled when a
bar opened, ask it what it holds. No credentials, no endpoint, no network, no
async. There is nothing to point at an exchange.

That is deliberate and it is checkable. ``tests/unit/test_architecture.py``
enumerates every implementation of this protocol and asserts there is exactly
one, :class:`~quantlab.paper.broker.PaperBroker`; a second one would have to be
written, named, and would fail that test on the way in.

:class:`MarketDataFeed` is the one thing here that does reach a network, and it
reaches it read-only: it yields **closed** bars. A feed that emitted the forming
bar would hand a live strategy the one thing INV-3 forbids a backtested one —
information from the future of the bar it is deciding on — and the paper session
would outperform its own backtest for a reason nobody would enjoy discovering.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol, runtime_checkable

from quantlab.core.types import Bar, Fill, Order, Position

__all__ = ["Broker", "MarketDataFeed"]


@runtime_checkable
class MarketDataFeed(Protocol):
    """A live source of **closed** bars."""

    def bars(self, symbol: str, timeframe: str) -> AsyncIterator[Bar]:
        """Yield each bar as it closes, indefinitely.

        Implementations MUST NOT yield the bar currently forming. A partially
        formed bar's ``close`` is the last trade, not the close, and a strategy
        that acted on it would be reading inside its own decision bar.
        """
        ...

    async def backfill(
        self, symbol: str, timeframe: str, since_ts: int, limit: int
    ) -> Sequence[Bar]:
        """Closed bars from ``since_ts``, oldest first, at most ``limit``.

        Used to rebuild indicator state at start-up and to fill the gap after a
        restart (section 16.2). The returned bars MUST be the same bars the
        historical source would give for that range — a session resumed from a
        different set of bars is a different session.
        """
        ...


@runtime_checkable
class Broker(Protocol):
    """Holds one position and fills orders at the next bar's open.

    The only implementation is ``PaperBroker`` (INV-1).
    """

    def submit(self, order: Order) -> None:
        """Accept ``order`` for execution at the **next** bar's open.

        Never fills immediately. Section 8.4's timing is the whole reason the
        backtest is honest, and a paper broker that filled on submission would
        be measuring a different strategy than the one that was validated.
        """
        ...

    def on_bar_open(self, bar: Bar) -> Sequence[Fill]:
        """Fill whatever is pending at ``bar.open`` and return the fills."""
        ...

    def position(self) -> Position:
        """What is currently held."""
        ...

    def equity(self, mark_price: float) -> float:
        """Cash plus the position marked at ``mark_price``."""
        ...
