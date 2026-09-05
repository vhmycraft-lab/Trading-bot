"""Market data ports (master spec sections 7.4 and 16.1).

A :class:`MarketDataSource` serves **closed historical bars**; there is no way
to ask it for anything else.  The research profile wraps the real source in a
partition guard that refuses any range reaching into the locked test partition
(INV-5), so a strategy under research cannot see held-out data even if a caller
asks for it by mistake.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol, runtime_checkable

from quantlab.core.types import Bar, BarFrame

__all__ = ["MarketDataFeed", "MarketDataSource"]


@runtime_checkable
class MarketDataSource(Protocol):
    """A read-only source of historical bars."""

    def load(self, symbol: str, timeframe: str, start_ts: int, end_ts: int) -> BarFrame:
        """Return the validated bars whose open time lies in ``[start_ts, end_ts]``.

        Both bounds are inclusive and expressed in milliseconds since the Unix
        epoch, UTC.  Implementations return an empty :class:`BarFrame` rather
        than raising when the range simply holds no bars.
        """
        ...

    def dataset_id(self, symbol: str, timeframe: str) -> str:
        """Return the content hash identifying the stored dataset (spec section 7.2)."""
        ...

    def available_range(self, symbol: str, timeframe: str) -> tuple[int, int] | None:
        """Return ``(first_ts, last_ts)`` held locally, or ``None`` if nothing is."""
        ...


@runtime_checkable
class MarketDataFeed(Protocol):
    """A live feed of closed bars.  Implemented in the paper-trading phase."""

    async def bars(self, symbol: str, timeframe: str) -> AsyncIterator[Bar]:
        """Yield each bar as it closes.  Unclosed bars are never emitted."""
        ...

    async def backfill(
        self, symbol: str, timeframe: str, since_ts: int, limit: int
    ) -> Sequence[Bar]:
        """Return up to ``limit`` closed bars at or after ``since_ts``."""
        ...
