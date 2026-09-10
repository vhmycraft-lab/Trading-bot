"""The live kline feed (spec section 16.1).

One property carries this file: **only closed bars leave the adapter**.

Binance emits a kline message on every trade, each carrying the candle as it
stands at that instant plus a flag saying whether it has closed. Forwarding
those would hand a live strategy the running close of the bar it is deciding on
— the look-ahead INV-3 forbids in a backtest, arriving through a door the
backtest does not have. The paper session would then beat the backtest that
authorised it, and the gap would read as alpha.

Everything is driven through a scripted socket. That is not only for speed: the
reconnect duplicate and the mid-stream gap are rare against a healthy exchange
and routine against a closed laptop lid, and they are unreachable in a test that
needs the real endpoint to produce them.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from quantlab.adapters.data.binance_ws import (
    DEFAULT_WS_URL,
    BinanceKlineFeed,
    parse_kline,
    stream_name,
)
from quantlab.core.errors import DataError

HOUR_MS = 3_600_000
T0 = 1_700_000_000_000 - (1_700_000_000_000 % HOUR_MS)


def kline(ts: int, *, closed: bool = True, close: float = 100.0) -> str:
    return json.dumps(
        {
            "e": "kline",
            "s": "BTCUSDT",
            "k": {
                "t": ts,
                "T": ts + HOUR_MS - 1,
                "o": "99.0",
                "h": "101.0",
                "l": "98.0",
                "c": str(close),
                "v": "12.5",
                "q": "1250.0",
                "n": 42,
                "x": closed,
            },
        }
    )


class FakeSocket:
    """Replays a scripted list of messages, then ends the stream."""

    def __init__(self, messages: list[str]) -> None:
        self.messages = messages
        self.sent: list[str] = []

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def __aiter__(self) -> AsyncIterator[str]:
        for message in self.messages:
            yield message


class FakeConnect:
    def __init__(self, socket: FakeSocket) -> None:
        self.socket = socket
        self.urls: list[str] = []

    def __call__(self, url: str) -> FakeConnect:
        self.urls.append(url)
        return self

    async def __aenter__(self) -> FakeSocket:
        return self.socket

    async def __aexit__(self, *exc: Any) -> bool:
        return False


def _drain(feed: BinanceKlineFeed) -> list[Any]:
    """Run an async generator to exhaustion, synchronously.

    ``asyncio.run`` rather than ``pytest-asyncio``: four tests do not justify a
    new test dependency in the lockfile, and the adapter's async-ness is not
    what any of them are about.
    """

    async def _go() -> list[Any]:
        return [bar async for bar in feed.bars("BTC/USDT", "1h")]

    return asyncio.run(_go())


def _collect(messages: list[str], **kwargs: Any) -> list[Any]:
    return _drain(BinanceKlineFeed(FakeConnect(FakeSocket(messages)), **kwargs))


# ---------------------------------------------------------------------------
# the causality filter
# ---------------------------------------------------------------------------
def test_a_forming_bar_never_leaves_the_adapter() -> None:
    """The single most important line in the adapter."""
    assert parse_kline(kline(T0, closed=False)) is None


def test_a_closed_bar_is_yielded() -> None:
    bar = parse_kline(kline(T0))
    assert bar is not None
    assert (bar.ts_open, bar.close, bar.volume) == (T0, 100.0, 12.5)


def test_the_stream_drops_every_forming_update_of_the_same_bar() -> None:
    """How the exchange actually behaves: dozens of updates per bar, one of
    which is closed. A feed that forwarded them would call ``on_bar`` dozens of
    times for one bar, each with a different close."""
    messages = [kline(T0, closed=False, close=c) for c in (99.0, 100.0, 101.0)]
    messages.append(kline(T0, close=102.0))
    bars = _collect(messages)
    assert [b.close for b in bars] == [102.0]


def test_a_subscription_acknowledgement_is_not_a_bar() -> None:
    bars = _collect([json.dumps({"result": None, "id": 1}), kline(T0)])
    assert len(bars) == 1


def test_a_control_frame_without_a_kline_is_not_a_bar() -> None:
    assert parse_kline(json.dumps({"e": "ping"})) is None


# ---------------------------------------------------------------------------
# malformed data is not silently skipped
# ---------------------------------------------------------------------------
def test_a_closed_kline_missing_a_price_raises() -> None:
    """Silently skipping unreadable market data is how a feed loses bars without
    anybody noticing — and the loss is invisible precisely because the session
    keeps running."""
    payload = json.loads(kline(T0))
    del payload["k"]["c"]
    with pytest.raises(DataError, match="malformed"):
        parse_kline(json.dumps(payload))


def test_a_message_that_is_not_an_object_raises() -> None:
    with pytest.raises(DataError, match="not a JSON object"):
        parse_kline("[1, 2, 3]")


def test_a_k_field_that_is_not_an_object_raises() -> None:
    with pytest.raises(DataError, match="not an object"):
        parse_kline(json.dumps({"k": "surprise"}))


# ---------------------------------------------------------------------------
# ordering and gaps
# ---------------------------------------------------------------------------
def test_a_duplicate_bar_after_a_reconnect_is_dropped() -> None:
    """Binance resends the last bar when a subscription resumes. Yielding it
    twice would have the strategy decide twice on one bar, and the second
    decision would see the first one's fill."""
    bars = _collect([kline(T0), kline(T0), kline(T0 + HOUR_MS)])
    assert [b.ts_open for b in bars] == [T0, T0 + HOUR_MS]


def test_an_out_of_order_bar_is_dropped() -> None:
    bars = _collect([kline(T0 + HOUR_MS), kline(T0)])
    assert [b.ts_open for b in bars] == [T0 + HOUR_MS]


def test_a_gap_is_reported_rather_than_papered_over() -> None:
    """The runtime backfills a gap. A feed that quietly skipped it would leave
    the strategy's indicators computed over a history with holes in it — and
    every number downstream would be subtly wrong with nothing to point at."""
    with pytest.raises(DataError, match="skipped from"):
        _collect([kline(T0), kline(T0 + 5 * HOUR_MS)])


def test_consecutive_bars_are_not_treated_as_a_gap() -> None:
    """Guards the test above: a gap check that fired on the normal case would
    make the feed unusable and would be removed rather than fixed."""
    bars = _collect([kline(T0 + i * HOUR_MS) for i in range(4)])
    assert len(bars) == 4


# ---------------------------------------------------------------------------
# subscription
# ---------------------------------------------------------------------------
def test_the_stream_name_is_built_from_the_symbol_and_timeframe() -> None:
    assert stream_name("BTC/USDT", "1h") == "btcusdt@kline_1h"
    assert stream_name("eth-usdt", "15m") == "ethusdt@kline_15m"


def test_a_timeframe_with_no_stream_is_refused_with_the_list() -> None:
    """A refusal a reader cannot act on gets worked around rather than fixed."""
    with pytest.raises(DataError, match="supported"):
        stream_name("BTC/USDT", "7h")


def test_the_adapter_subscribes_before_reading() -> None:
    socket = FakeSocket([kline(T0)])
    _drain(BinanceKlineFeed(FakeConnect(socket)))
    assert socket.sent, "nothing was subscribed to"
    assert json.loads(socket.sent[0])["params"] == ["btcusdt@kline_1h"]


def test_the_default_endpoint_is_the_public_one() -> None:
    """Public data only. This adapter has no key, no signing, and no way to
    acquire either (INV-1)."""
    connect = FakeConnect(FakeSocket([]))
    _drain(BinanceKlineFeed(connect))
    assert connect.urls == [DEFAULT_WS_URL]
    assert "stream.binance.com" in DEFAULT_WS_URL


# ---------------------------------------------------------------------------
# history is somebody else's job
# ---------------------------------------------------------------------------
def test_backfill_refuses_rather_than_offering_a_second_history() -> None:
    """A second source of historical bars would be a second answer to "what were
    the bars?", and a session resumed from the wrong one is a different
    session."""
    feed = BinanceKlineFeed(FakeConnect(FakeSocket([])))
    with pytest.raises(NotImplementedError, match="CcxtRestSource"):
        asyncio.run(feed.backfill("BTC/USDT", "1h", 0, 10))
