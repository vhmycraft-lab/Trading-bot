"""The partition guard that keeps held-out data out of research (INV-5).

Wrapping a :class:`~quantlab.ports.data.MarketDataSource` in a
:class:`PartitionGuard` makes it *structurally* impossible for a research run to
read the test partition: the guard checks the requested range before the source
is touched and raises :class:`~quantlab.core.errors.LockboxViolation`, which is
never caught except at the CLI top level.

The ``lockbox`` profile is the only one that omits the guard, and every use of
it is recorded before the run starts.
"""

from __future__ import annotations

from quantlab.core.splits import SplitPolicy
from quantlab.core.types import BarFrame
from quantlab.ports.data import MarketDataSource

__all__ = ["PartitionGuard"]


class PartitionGuard:
    """A :class:`MarketDataSource` that refuses to serve test-partition bars.

    The guard is deliberately dumb: it does not clip the request to the allowed
    range, because silently returning less data than asked for would turn a
    programming error into a subtly wrong backtest.  It raises instead.
    """

    def __init__(self, source: MarketDataSource, policy: SplitPolicy) -> None:
        self._source = source
        self._policy = policy

    @property
    def policy(self) -> SplitPolicy:
        return self._policy

    def load(self, symbol: str, timeframe: str, start_ts: int, end_ts: int) -> BarFrame:
        """Load bars, refusing any range that reaches the test partition.

        Raises:
            LockboxViolation: if ``[start_ts, end_ts]`` intersects the test partition.
        """
        self._policy.assert_no_test_data(start_ts, end_ts)
        return self._source.load(symbol, timeframe, start_ts, end_ts)

    def dataset_id(self, symbol: str, timeframe: str) -> str:
        return self._source.dataset_id(symbol, timeframe)

    def available_range(self, symbol: str, timeframe: str) -> tuple[int, int] | None:
        """Return the range that research may see, clipped below the test start.

        Reporting the true end here would leak where the held-out data begins
        and how much of it exists.
        """
        available = self._source.available_range(symbol, timeframe)
        if available is None:
            return None
        first, last = available
        limit = self._policy.test_start_ts - self._policy.bar_ms
        if first > limit:
            return None
        return first, min(last, limit)
