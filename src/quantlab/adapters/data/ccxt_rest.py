"""REST tail updates via ccxt (master spec section 7.1).

The bulk archive lags the market by up to a month, so the last stretch of
history comes from ``fetch_ohlcv`` instead.  Two properties matter:

* **only closed bars.** The exchange happily returns the bar that is forming
  right now.  Including it would put a partially-formed candle into the
  dataset, which is a look-ahead bug wearing a disguise, so it is dropped.
* **overlaps are de-duplicated.** Pagination re-returns the boundary bar, and
  the tail overlaps the archive; both are resolved by normalisation, which
  raises if the same timestamp ever arrives with different data.

The ccxt client is injected, so every test here runs offline.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final, Protocol

import pandas as pd

from quantlab.core.data_validation import normalise_bars
from quantlab.core.errors import DataError
from quantlab.core.types import BarFrame, coerce_bar_frame_df, normalise_epoch, timeframe_ms

__all__ = ["DEFAULT_PAGE_LIMIT", "CcxtRestSource", "OhlcvClient", "ohlcv_to_frame"]

#: Binance's maximum page size for ``fetch_ohlcv``.
DEFAULT_PAGE_LIMIT: Final[int] = 1000

#: Safety valve: refuse to loop forever if an exchange keeps returning pages.
_MAX_PAGES: Final[int] = 10_000


class OhlcvClient(Protocol):
    """The one ccxt method this adapter uses."""

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, since: int | None = None, limit: int | None = None
    ) -> Sequence[Sequence[Any]]:
        """Return ``[[ts, open, high, low, close, volume], ...]`` ascending by ts."""
        ...


def ohlcv_to_frame(rows: Sequence[Sequence[Any]]) -> pd.DataFrame:
    """Convert ccxt OHLCV rows into the canonical bar schema.

    ccxt does not report quote volume or trade count, so both are recorded as
    zero rather than guessed; the canonical schema keeps the columns so archive
    and REST bars stay interchangeable.
    """
    if not rows:
        return coerce_bar_frame_df(
            pd.DataFrame(
                {
                    "ts_open": [],
                    "open": [],
                    "high": [],
                    "low": [],
                    "close": [],
                    "volume": [],
                    "quote_volume": [],
                    "trades": [],
                }
            )
        )

    for row in rows:
        if len(row) < 6:
            raise DataError("ccxt OHLCV row is too short", row=list(row))

    frame = pd.DataFrame(
        {
            "ts_open": [normalise_epoch(int(row[0])) for row in rows],
            "open": [float(row[1]) for row in rows],
            "high": [float(row[2]) for row in rows],
            "low": [float(row[3]) for row in rows],
            "close": [float(row[4]) for row in rows],
            "volume": [float(row[5]) for row in rows],
            "quote_volume": [0.0] * len(rows),
            "trades": [0] * len(rows),
        }
    )
    frame["is_gap_filled"] = False
    return coerce_bar_frame_df(frame)


class CcxtRestSource:
    """Paginated, closed-bars-only OHLCV fetching."""

    def __init__(
        self,
        client: OhlcvClient,
        *,
        page_limit: int = DEFAULT_PAGE_LIMIT,
    ) -> None:
        if page_limit <= 0:
            raise DataError("page_limit must be positive", page_limit=page_limit)
        self.client = client
        self.page_limit = page_limit

    def fetch_range(
        self,
        symbol: str,
        timeframe: str,
        *,
        since_ts: int,
        until_ts: int | None = None,
        now_ms: int | None = None,
    ) -> pd.DataFrame:
        """Fetch closed bars from ``since_ts`` (inclusive) onward.

        Args:
            since_ts: First bar open time to request, in ms UTC.
            until_ts: Stop once this open time is reached (inclusive).
            now_ms: The current time.  Bars whose close lies at or after it are
                still forming and are dropped.  Passing this explicitly keeps
                the adapter deterministic under test.

        Returns:
            A canonical, sorted, de-duplicated frame; empty if nothing closed.
        """
        bar_ms = timeframe_ms(timeframe)
        collected: list[Sequence[Any]] = []
        cursor = int(since_ts)
        seen_last: int | None = None

        for _page in range(_MAX_PAGES):
            rows = list(self.client.fetch_ohlcv(symbol, timeframe, cursor, self.page_limit))
            if not rows:
                break
            collected.extend(rows)

            last_ts = normalise_epoch(int(rows[-1][0]))
            if seen_last is not None and last_ts <= seen_last:
                break  # the exchange stopped advancing; do not spin
            seen_last = last_ts
            if until_ts is not None and last_ts >= int(until_ts):
                break
            if len(rows) < self.page_limit:
                break
            cursor = last_ts + bar_ms
        else:  # pragma: no cover - only reachable with a pathological client
            raise DataError("ccxt pagination did not terminate", symbol=symbol, pages=_MAX_PAGES)

        frame = ohlcv_to_frame(collected)
        if frame.empty:
            return frame

        ts = frame["ts_open"].to_numpy()
        keep = ts >= int(since_ts)
        if until_ts is not None:
            keep &= ts <= int(until_ts)
        if now_ms is not None:
            # A bar is closed once its *next* bar has begun.
            keep &= (ts + bar_ms) <= int(now_ms)
        frame = frame.loc[keep].reset_index(drop=True)

        normalised, _report = normalise_bars(frame, timeframe, symbol=symbol, allow_gaps=True)
        return normalised

    def fetch_tail(
        self,
        symbol: str,
        timeframe: str,
        *,
        after_ts: int | None,
        now_ms: int | None = None,
    ) -> pd.DataFrame:
        """Fetch the bars that follow ``after_ts``, the last bar already stored.

        Passing ``None`` fetches from the beginning of the exchange's history.
        """
        bar_ms = timeframe_ms(timeframe)
        since = 0 if after_ts is None else int(after_ts) + bar_ms
        return self.fetch_range(symbol, timeframe, since_ts=since, now_ms=now_ms)

    def as_bar_frame(self, symbol: str, timeframe: str, frame: pd.DataFrame) -> BarFrame:
        """Wrap a fetched frame.

        The result carries an empty ``dataset_id``: REST output is not a stored,
        content-hashed dataset, and pretending otherwise would let an
        unreproducible run masquerade as a reproducible one.
        """
        return BarFrame(frame, symbol=symbol, timeframe=timeframe, dataset_id="")
