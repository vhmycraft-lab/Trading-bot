"""Live closed bars over Binance's kline websocket (master spec section 16.1).

The only part of this platform that watches a market in real time, and the
narrowest thing that could do the job: it subscribes to a kline stream, drops
everything that is not a **closed** bar, and yields the rest.

That filter is the whole adapter, really. Binance emits a kline message on every
trade, each carrying the candle *as it stands right now* and a flag saying
whether it has closed. A feed that forwarded those would hand a live strategy
the running close of the bar it is currently deciding on — the exact
look-ahead that INV-3 forbids in a backtest, arriving through a door the
backtest does not have. The paper session would then beat its own backtest, and
the discrepancy would look like alpha.

The connection is injected. Every test here runs offline against a scripted
socket, which is also how the reconnect and out-of-order cases are reachable at
all: they are rare against a healthy exchange and routine against a laptop lid.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any, Final, Protocol

from quantlab.core.errors import DataError
from quantlab.core.types import Bar, timeframe_ms

__all__ = ["DEFAULT_WS_URL", "BinanceKlineFeed", "WebSocketLike", "parse_kline"]

#: Binance's public combined-stream endpoint. Public data only — this adapter
#: has no key, no signing, and no way to acquire either (INV-1).
DEFAULT_WS_URL: Final[str] = "wss://stream.binance.com:9443/ws"

#: Binance's own timeframe spellings differ from ccxt's for a couple of values.
_STREAM_INTERVAL: Final[dict[str, str]] = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1d",
    "1w": "1w",
}


class WebSocketLike(Protocol):
    """The two methods this adapter uses. Injected, so tests run offline."""

    async def send(self, message: str) -> None: ...

    def __aiter__(self) -> AsyncIterator[str]: ...


def stream_name(symbol: str, timeframe: str) -> str:
    """``btcusdt@kline_1h`` for ``BTC/USDT`` and ``1h``."""
    interval = _STREAM_INTERVAL.get(timeframe)
    if interval is None:
        raise DataError(
            f"timeframe {timeframe!r} has no Binance kline stream; "
            f"supported: {', '.join(sorted(_STREAM_INTERVAL))}",
            timeframe=timeframe,
        )
    return f"{symbol.replace('/', '').replace('-', '').lower()}@kline_{interval}"


def parse_kline(message: str | Any) -> Bar | None:
    """A closed bar, or ``None`` for anything else.

    ``None`` covers three genuinely different situations — a bar still forming,
    a subscription acknowledgement, a heartbeat — and they are collapsed on
    purpose: the caller's only question is "is there a bar to act on?", and a
    caller that had to tell them apart would end up with a branch per message
    type of a protocol it does not own.

    A message that is *malformed* is not collapsed into ``None``: it raises,
    because silently skipping unreadable market data is how a feed loses bars
    without anybody noticing.
    """
    payload = json.loads(message) if isinstance(message, str | bytes) else message
    if not isinstance(payload, dict):
        raise DataError("kline message is not a JSON object")

    kline = payload.get("k")
    if kline is None:
        # A subscription result or a control frame. Not a bar, not an error.
        return None
    if not isinstance(kline, dict):
        raise DataError("kline message has a 'k' field that is not an object")

    if not kline.get("x", False):
        # Still forming. This is the line that keeps the feed causal.
        return None

    try:
        return Bar(
            ts_open=int(kline["t"]),
            open=float(kline["o"]),
            high=float(kline["h"]),
            low=float(kline["l"]),
            close=float(kline["c"]),
            volume=float(kline["v"]),
            quote_volume=float(kline.get("q", 0.0)),
            trades=int(kline.get("n", 0)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DataError(f"closed kline is missing or malformed: {exc}") from exc


class BinanceKlineFeed:
    """Yields closed bars from an injected websocket connection."""

    def __init__(
        self,
        connect: Any,
        *,
        url: str = DEFAULT_WS_URL,
    ) -> None:
        #: An async callable returning an async context manager over a
        #: :class:`WebSocketLike`. ``websockets.connect`` satisfies it, and so
        #: does a scripted fake.
        self._connect = connect
        self._url = url

    async def bars(self, symbol: str, timeframe: str) -> AsyncIterator[Bar]:
        """Every closed bar for ``symbol``, in order, without duplicates.

        Two guards, both learned from what a real connection does rather than
        from what the protocol promises:

        * a bar whose open time is not **after** the last one is dropped, which
          covers the duplicate Binance sends on reconnect;
        * a gap larger than one bar is reported rather than papered over — the
          runtime backfills it, and a feed that silently skipped the missing
          bars would leave a strategy's indicators computed over a history with
          holes in it.
        """
        name = stream_name(symbol, timeframe)
        bar_ms = timeframe_ms(timeframe)
        last_ts: int | None = None

        async with self._connect(self._url) as socket:
            await socket.send(json.dumps({"method": "SUBSCRIBE", "params": [name], "id": 1}))
            async for message in socket:
                bar = parse_kline(message)
                if bar is None:
                    continue
                if last_ts is not None:
                    if bar.ts_open <= last_ts:
                        continue
                    if bar.ts_open - last_ts > bar_ms:
                        raise DataError(
                            f"the {symbol} {timeframe} stream skipped from {last_ts} to "
                            f"{bar.ts_open}; the session must backfill the gap rather than "
                            "continue with a history that has holes in it",
                            symbol=symbol,
                            timeframe=timeframe,
                            last_ts=last_ts,
                            next_ts=bar.ts_open,
                        )
                last_ts = bar.ts_open
                yield bar

    async def backfill(
        self, symbol: str, timeframe: str, since_ts: int, limit: int
    ) -> Sequence[Bar]:
        """Not this adapter's job.

        The websocket carries the live tail and nothing else. History comes from
        :class:`~quantlab.adapters.data.ccxt_rest.CcxtRestSource`, which is
        already the platform's one answer to "what were the bars?" — a second
        source of historical bars would be a second answer, and a session
        resumed from the wrong one is a different session.
        """
        raise NotImplementedError(
            "BinanceKlineFeed carries live bars only; backfill through CcxtRestSource "
            "so that resumed sessions read the same history the backtest did"
        )
