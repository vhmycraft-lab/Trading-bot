"""REST tail updates: pagination, de-duplication, unclosed bars (spec section 7.1).

The exchange client is a fake, so these run offline and deterministically.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from tests.helpers import make_bars, ohlcv_rows

from quantlab.adapters.data.ccxt_rest import (
    DEFAULT_PAGE_LIMIT,
    CcxtRestSource,
    ohlcv_to_frame,
)
from quantlab.core.errors import DataError
from quantlab.core.types import format_ts, to_ms

BAR_MS = 3_600_000


class FakeExchange:
    """Serves a fixed series the way Binance does: ascending, page-limited, overlapping."""

    def __init__(self, rows: list[list[float]], *, page_limit: int = DEFAULT_PAGE_LIMIT) -> None:
        self.rows = rows
        self.page_limit = page_limit
        self.calls: list[tuple[int | None, int | None]] = []

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, since: int | None = None, limit: int | None = None
    ) -> Sequence[Sequence[Any]]:
        self.calls.append((since, limit))
        start = 0 if since is None else int(since)
        selected = [row for row in self.rows if row[0] >= start]
        return selected[: limit or self.page_limit]


def test_ohlcv_to_frame_maps_the_columns() -> None:
    bars = make_bars(5)
    frame = ohlcv_to_frame(ohlcv_rows(bars))
    assert frame["ts_open"].tolist() == bars["ts_open"].tolist()
    assert frame["close"].tolist() == pytest.approx(bars["close"].tolist())
    # ccxt does not report these; zero is honest, a guess would not be.
    assert (frame["quote_volume"] == 0.0).all()
    assert (frame["trades"] == 0).all()


def test_ohlcv_to_frame_on_no_rows() -> None:
    assert ohlcv_to_frame([]).empty


def test_ohlcv_to_frame_rejects_a_short_row() -> None:
    with pytest.raises(DataError, match="too short"):
        ohlcv_to_frame([[1, 2, 3]])


def test_page_limit_must_be_positive() -> None:
    with pytest.raises(DataError, match="page_limit"):
        CcxtRestSource(FakeExchange([]), page_limit=0)


def test_fetch_paginates_until_exhausted() -> None:
    bars = make_bars(250)
    exchange = FakeExchange(ohlcv_rows(bars))
    source = CcxtRestSource(exchange, page_limit=100)

    frame = source.fetch_range("BTC/USDT", "1h", since_ts=int(bars["ts_open"].iloc[0]))
    assert len(frame) == 250
    assert len(exchange.calls) == 3
    assert frame["ts_open"].tolist() == bars["ts_open"].tolist()


def test_pagination_overlap_is_deduplicated() -> None:
    """A client that re-returns the boundary bar must not duplicate it."""

    class OverlappingExchange(FakeExchange):
        def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):  # type: ignore[no-untyped-def]
            rows = list(super().fetch_ohlcv(symbol, timeframe, since, limit))
            if since and rows:
                previous = [r for r in self.rows if r[0] == int(since) - BAR_MS]
                rows = previous + rows
            return rows

    bars = make_bars(250)
    source = CcxtRestSource(OverlappingExchange(ohlcv_rows(bars)), page_limit=100)
    frame = source.fetch_range("BTC/USDT", "1h", since_ts=int(bars["ts_open"].iloc[0]))
    assert len(frame) == 250
    assert frame["ts_open"].is_unique


def test_forming_bars_are_dropped() -> None:
    """The bar being built right now must never enter the dataset."""
    bars = make_bars(10)
    source = CcxtRestSource(FakeExchange(ohlcv_rows(bars)))

    # "Now" is midway through the last bar, so that bar has not closed.
    now = int(bars["ts_open"].iloc[-1]) + BAR_MS // 2
    frame = source.fetch_range("BTC/USDT", "1h", since_ts=int(bars["ts_open"].iloc[0]), now_ms=now)
    assert len(frame) == 9
    assert format_ts(int(frame["ts_open"].iloc[-1])) == "2023-01-01T08:00:00Z"


def test_a_bar_is_closed_the_instant_the_next_one_begins() -> None:
    bars = make_bars(10)
    source = CcxtRestSource(FakeExchange(ohlcv_rows(bars)))
    now = int(bars["ts_open"].iloc[-1]) + BAR_MS
    frame = source.fetch_range("BTC/USDT", "1h", since_ts=int(bars["ts_open"].iloc[0]), now_ms=now)
    assert len(frame) == 10


def test_until_stops_the_walk() -> None:
    bars = make_bars(250)
    exchange = FakeExchange(ohlcv_rows(bars), page_limit=100)
    source = CcxtRestSource(exchange, page_limit=100)
    frame = source.fetch_range(
        "BTC/USDT",
        "1h",
        since_ts=int(bars["ts_open"].iloc[0]),
        until_ts=int(bars["ts_open"].iloc[49]),
    )
    assert len(frame) == 50
    assert len(exchange.calls) == 1


def test_an_empty_exchange_returns_an_empty_frame() -> None:
    source = CcxtRestSource(FakeExchange([]))
    assert source.fetch_range("BTC/USDT", "1h", since_ts=0).empty


def test_a_stuck_exchange_does_not_spin_forever() -> None:
    """A client that keeps returning the same page must terminate the walk."""

    class StuckExchange(FakeExchange):
        def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):  # type: ignore[no-untyped-def]
            self.calls.append((since, limit))
            return self.rows[: limit or self.page_limit]

    bars = make_bars(10)
    source = CcxtRestSource(StuckExchange(ohlcv_rows(bars)), page_limit=10)
    frame = source.fetch_range("BTC/USDT", "1h", since_ts=int(bars["ts_open"].iloc[0]))
    assert len(frame) == 10


def test_fetch_tail_starts_after_the_last_stored_bar() -> None:
    bars = make_bars(50)
    exchange = FakeExchange(ohlcv_rows(bars))
    source = CcxtRestSource(exchange)

    stored_end = int(bars["ts_open"].iloc[19])
    frame = source.fetch_tail("BTC/USDT", "1h", after_ts=stored_end)
    assert exchange.calls[0][0] == stored_end + BAR_MS
    assert len(frame) == 30
    assert int(frame["ts_open"].iloc[0]) == stored_end + BAR_MS


def test_fetch_tail_from_the_beginning() -> None:
    bars = make_bars(20)
    source = CcxtRestSource(FakeExchange(ohlcv_rows(bars)))
    assert len(source.fetch_tail("BTC/USDT", "1h", after_ts=None)) == 20


def test_tail_output_is_sorted_and_contiguous() -> None:
    bars = make_bars(60)
    shuffled = ohlcv_rows(bars.sample(frac=1.0, random_state=5).reset_index(drop=True))

    class ShuffledExchange(FakeExchange):
        def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):  # type: ignore[no-untyped-def]
            self.calls.append((since, limit))
            return [] if since and int(since) > int(bars["ts_open"].iloc[0]) else shuffled

    source = CcxtRestSource(ShuffledExchange(shuffled), page_limit=1000)
    frame = source.fetch_range("BTC/USDT", "1h", since_ts=int(bars["ts_open"].iloc[0]))
    ts = frame["ts_open"].to_numpy()
    assert (ts[1:] - ts[:-1] == BAR_MS).all()


def test_as_bar_frame_carries_no_dataset_id() -> None:
    """REST output is not reproducible history and must not claim to be."""
    bars = make_bars(5)
    source = CcxtRestSource(FakeExchange(ohlcv_rows(bars)))
    frame = source.as_bar_frame("BTC/USDT", "1h", bars)
    assert frame.dataset_id == ""
    assert frame.n_bars == 5


def test_tail_merges_into_the_dataset_and_gains_an_identity(parquet_store) -> None:
    """The tail becomes reproducible history only once it is stored."""
    from quantlab.adapters.data.binance_archive import BinanceArchiveIngestor

    archive_bars = make_bars(24)
    tail_bars = make_bars(12, start_ts=to_ms("2023-01-02T00:00:00Z"), seed=9)

    source = CcxtRestSource(FakeExchange(ohlcv_rows(tail_bars)))
    tail = source.fetch_tail("BTC/USDT", "1h", after_ts=int(archive_bars["ts_open"].iloc[-1]))

    ingestor = BinanceArchiveIngestor(parquet_store)
    result = ingestor.rebuild("BTC/USDT", "1h", extra_frames=[archive_bars, tail])
    assert result.n_bars == 36
    assert len(result.dataset_id) == 16
