"""Core value types: time, bars, and the trading primitives (spec sections 7.2 and 8.1).

Two fixed decisions from spec section 1.3 shape everything here:

* every timestamp is an ``int64`` count of **milliseconds since the Unix epoch,
  UTC**, and always names a bar's **open** time (bars are left-labelled)
* every price and quantity is ``float64``

The look-ahead guarantee (INV-3) lives in :class:`BarWindow`: a strategy
deciding at bar ``i`` is handed a window over ``[0..i]`` that raises
:class:`~quantlab.core.errors.LookaheadError` on any attempt to read further.
It is not a convention a strategy can forget to follow — reading bar ``i+1``
raises rather than returning a number.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final, Literal, Self

import numpy as np
import pandas as pd

from quantlab.core.errors import DataValidationError, LookaheadError

__all__ = [
    "BARS_PER_YEAR",
    "BAR_COLUMNS",
    "BAR_DTYPES",
    "MAX_WINDOW_DF_ROWS",
    "TIMEFRAME_MS",
    "Bar",
    "BarFrame",
    "BarWindow",
    "Fill",
    "Order",
    "Position",
    "Side",
    "Signal",
    "SignalKind",
    "Trade",
    "bars_per_year",
    "empty_bar_frame_df",
    "format_ts",
    "from_ms",
    "timeframe_ms",
    "to_ms",
]

# ---------------------------------------------------------------------------
# time
# ---------------------------------------------------------------------------
_MS_PER_SECOND: Final[int] = 1_000
_MS_PER_MINUTE: Final[int] = 60 * _MS_PER_SECOND
_MS_PER_HOUR: Final[int] = 60 * _MS_PER_MINUTE
_MS_PER_DAY: Final[int] = 24 * _MS_PER_HOUR

#: Bar duration in milliseconds for every timeframe the platform understands.
TIMEFRAME_MS: Final[Mapping[str, int]] = {
    "1m": _MS_PER_MINUTE,
    "3m": 3 * _MS_PER_MINUTE,
    "5m": 5 * _MS_PER_MINUTE,
    "15m": 15 * _MS_PER_MINUTE,
    "30m": 30 * _MS_PER_MINUTE,
    "1h": _MS_PER_HOUR,
    "2h": 2 * _MS_PER_HOUR,
    "4h": 4 * _MS_PER_HOUR,
    "6h": 6 * _MS_PER_HOUR,
    "8h": 8 * _MS_PER_HOUR,
    "12h": 12 * _MS_PER_HOUR,
    "1d": _MS_PER_DAY,
}

#: Bars per year for annualisation (spec section 1.3): crypto trades 24/7.
BARS_PER_YEAR: Final[Mapping[str, int]] = {
    "1m": 525_600,
    "3m": 175_200,
    "5m": 105_120,
    "15m": 35_040,
    "30m": 17_520,
    "1h": 8_760,
    "2h": 4_380,
    "4h": 2_190,
    "6h": 1_460,
    "8h": 1_095,
    "12h": 730,
    "1d": 365,
}

#: Timestamps at or above this value are microseconds, not milliseconds.
#: Binance switched its archive columns to microseconds during 2025; 1e15 ms is
#: the year 33658, so the split is unambiguous for any realistic market data.
_MICROSECOND_THRESHOLD: Final[int] = 1_000_000_000_000_000


def timeframe_ms(timeframe: str) -> int:
    """Return the bar duration of ``timeframe`` in milliseconds.

    Raises:
        DataValidationError: if the timeframe is not supported.
    """
    try:
        return TIMEFRAME_MS[timeframe]
    except KeyError:
        raise DataValidationError(
            "unsupported timeframe", timeframe=timeframe, supported=sorted(TIMEFRAME_MS)
        ) from None


def bars_per_year(timeframe: str) -> int:
    """Return the annualisation factor for ``timeframe`` (spec section 1.3)."""
    try:
        return BARS_PER_YEAR[timeframe]
    except KeyError:
        raise DataValidationError(
            "unsupported timeframe", timeframe=timeframe, supported=sorted(BARS_PER_YEAR)
        ) from None


def to_ms(value: int | float | str | dt.date | dt.datetime | np.integer) -> int:
    """Normalise ``value`` to milliseconds since the Unix epoch, UTC.

    Accepted forms:

    * ``int`` / ``numpy`` integer — already milliseconds, returned unchanged
    * ``datetime`` — converted to UTC; a **naive** datetime is read as UTC
    * ``date`` — midnight UTC on that day
    * ``str`` — ISO 8601 (``Z`` and numeric offsets both work), or the short
      forms ``YYYY`` and ``YYYY-MM`` that the archive CLI uses

    An offset-aware input is converted, never truncated: ``2023-01-01T00:00+02:00``
    and ``2022-12-31T22:00:00Z`` produce the same number.

    Raises:
        DataValidationError: if the value cannot be interpreted as a timestamp.
    """
    if isinstance(value, bool):  # bool is an int subclass; almost certainly a bug
        raise DataValidationError("timestamp must not be a bool", value=value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        if not float(value).is_integer():
            raise DataValidationError("float timestamp must be a whole number of ms", value=value)
        return int(value)
    if isinstance(value, dt.datetime):
        aware = value if value.tzinfo is not None else value.replace(tzinfo=dt.UTC)
        return int(aware.astimezone(dt.UTC).timestamp() * _MS_PER_SECOND)
    if isinstance(value, dt.date):
        return to_ms(dt.datetime(value.year, value.month, value.day, tzinfo=dt.UTC))
    if isinstance(value, str):
        return _parse_timestamp_string(value)
    raise DataValidationError("cannot interpret value as a timestamp", value=repr(value))


def _parse_timestamp_string(text: str) -> int:
    raw = text.strip()
    if not raw:
        raise DataValidationError("empty timestamp string")

    candidate = raw
    if len(raw) == 4 and raw.isdigit():  # YYYY
        candidate = f"{raw}-01-01"
    elif len(raw) == 7 and raw[4] == "-":  # YYYY-MM, the form the archive CLI takes
        candidate = f"{raw}-01"
    elif raw.isdigit():  # a bare epoch in milliseconds
        return int(raw)

    try:
        parsed = dt.datetime.fromisoformat(candidate)
    except ValueError:
        raise DataValidationError("timestamp string is not ISO 8601", value=text) from None
    return to_ms(parsed)


def from_ms(ms: int) -> dt.datetime:
    """Return the timezone-aware UTC datetime for ``ms``."""
    return dt.datetime.fromtimestamp(int(ms) / _MS_PER_SECOND, tz=dt.UTC)


def format_ts(ms: int | None) -> str:
    """Render ``ms`` as an ISO 8601 UTC string, or ``"-"`` for ``None``.

    Never raises.  This helper is used inside error messages — including the
    lockbox guard's — so a value too large to be a date must not turn a precise
    :class:`~quantlab.core.errors.LockboxViolation` into an unrelated
    :class:`ValueError` from the formatter.
    """
    if ms is None:
        return "-"
    try:
        return from_ms(ms).isoformat().replace("+00:00", "Z")
    except (ValueError, OverflowError, OSError):
        return f"{int(ms)}ms"


def normalise_epoch(value: int) -> int:
    """Return ``value`` in milliseconds, converting from microseconds if needed.

    Binance's archive files carry microsecond timestamps from 2025 onwards while
    older files use milliseconds; both appear in one dataset.
    """
    number = int(value)
    return number // 1_000 if abs(number) >= _MICROSECOND_THRESHOLD else number


# ---------------------------------------------------------------------------
# canonical bar schema (spec section 7.2)
# ---------------------------------------------------------------------------
#: Column order of the canonical bar frame.  Parquet files use exactly this.
BAR_COLUMNS: Final[tuple[str, ...]] = (
    "ts_open",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
    "is_gap_filled",
)

#: numpy dtype per canonical column.
BAR_DTYPES: Final[Mapping[str, str]] = {
    "ts_open": "int64",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
    "quote_volume": "float64",
    "trades": "int64",
    "is_gap_filled": "bool",
}

_PRICE_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close")

#: Upper bound on the rows a strategy may materialise as a DataFrame at once.
MAX_WINDOW_DF_ROWS: Final[int] = 5_000


def empty_bar_frame_df() -> pd.DataFrame:
    """Return an empty DataFrame with the canonical schema and dtypes."""
    return pd.DataFrame({name: pd.Series(dtype=BAR_DTYPES[name]) for name in BAR_COLUMNS})


def coerce_bar_frame_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` as a new frame with exactly the canonical columns and dtypes.

    ``is_gap_filled`` defaults to ``False`` when absent, so a raw exchange frame
    can be coerced before validation.

    Raises:
        DataValidationError: if a required column is missing or cannot be cast.
    """
    missing = [name for name in BAR_COLUMNS if name not in df.columns and name != "is_gap_filled"]
    if missing:
        raise DataValidationError("bar frame is missing columns", missing=missing)

    out = pd.DataFrame(index=pd.RangeIndex(len(df)))
    for name in BAR_COLUMNS:
        if name == "is_gap_filled" and name not in df.columns:
            out[name] = np.zeros(len(df), dtype=bool)
            continue
        column = df[name].to_numpy()
        try:
            if name == "ts_open":
                out[name] = np.array([normalise_epoch(v) for v in column], dtype="int64")
            else:
                out[name] = column.astype(BAR_DTYPES[name])
        except (TypeError, ValueError) as exc:
            raise DataValidationError(
                "column cannot be cast to its canonical dtype",
                column=name,
                dtype=BAR_DTYPES[name],
            ) from exc
    return out


def _read_only(array: np.ndarray) -> np.ndarray:
    """Return a non-writeable view of ``array``.

    The view is what callers receive, so a strategy cannot reach back through it
    and mutate the loaded dataset.
    """
    view = array.view()
    view.flags.writeable = False
    return view


class BarFrame:
    """An immutable, validated OHLCV series for one symbol and timeframe.

    The wrapped DataFrame is copied on construction and never handed out;
    :meth:`to_pandas` returns a fresh copy and the array accessors return
    non-writeable views.  A ``BarFrame`` is therefore safe to share between the
    engine, the validators and a strategy without any of them being able to
    corrupt it for the others.

    Slicing is by **timestamp**, not by position, so a segment always means the
    same bars regardless of what else has been loaded.
    """

    __slots__ = ("_columns", "_frame", "dataset_id", "symbol", "timeframe")

    _frame: pd.DataFrame
    _columns: dict[str, np.ndarray]
    symbol: str
    timeframe: str
    dataset_id: str

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        symbol: str,
        timeframe: str,
        dataset_id: str = "",
    ) -> None:
        canonical = coerce_bar_frame_df(frame)
        object.__setattr__(self, "_frame", canonical)
        object.__setattr__(
            self, "_columns", {name: canonical[name].to_numpy() for name in BAR_COLUMNS}
        )
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "timeframe", timeframe)
        object.__setattr__(self, "dataset_id", dataset_id)
        # Reject anything that is not a usable series before it reaches a strategy.
        timeframe_ms(timeframe)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    # -- shape -------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._frame)

    @property
    def n_bars(self) -> int:
        """Number of bars in the frame."""
        return len(self)

    @property
    def bar_ms(self) -> int:
        """Bar duration in milliseconds."""
        return timeframe_ms(self.timeframe)

    @property
    def start_ts(self) -> int | None:
        """Open time of the first bar, or ``None`` when the frame is empty."""
        return None if not len(self) else int(self._columns["ts_open"][0])

    @property
    def end_ts(self) -> int | None:
        """Open time of the last bar, or ``None`` when the frame is empty."""
        return None if not len(self) else int(self._columns["ts_open"][-1])

    # -- columns -----------------------------------------------------------
    def column(self, name: str) -> np.ndarray:
        """Return a read-only view of one canonical column."""
        try:
            return _read_only(self._columns[name])
        except KeyError:
            raise DataValidationError(
                "unknown bar column", column=name, known=list(BAR_COLUMNS)
            ) from None

    @property
    def ts_open(self) -> np.ndarray:
        return self.column("ts_open")

    @property
    def open(self) -> np.ndarray:
        return self.column("open")

    @property
    def high(self) -> np.ndarray:
        return self.column("high")

    @property
    def low(self) -> np.ndarray:
        return self.column("low")

    @property
    def close(self) -> np.ndarray:
        return self.column("close")

    @property
    def volume(self) -> np.ndarray:
        return self.column("volume")

    @property
    def quote_volume(self) -> np.ndarray:
        return self.column("quote_volume")

    @property
    def trades(self) -> np.ndarray:
        return self.column("trades")

    @property
    def is_gap_filled(self) -> np.ndarray:
        return self.column("is_gap_filled")

    # -- views -------------------------------------------------------------
    def to_pandas(self) -> pd.DataFrame:
        """Return an independent copy of the underlying DataFrame."""
        return self._frame.copy()

    def slice(self, start_ts: int | None = None, end_ts: int | None = None) -> BarFrame:
        """Return the bars whose open time lies in ``[start_ts, end_ts]``.

        Both bounds are inclusive and either may be ``None`` for "unbounded".
        """
        ts = self._columns["ts_open"]
        lo = 0 if start_ts is None else int(np.searchsorted(ts, int(start_ts), side="left"))
        hi = len(ts) if end_ts is None else int(np.searchsorted(ts, int(end_ts), side="right"))
        if hi < lo:
            hi = lo
        return self._rebuild(self._frame.iloc[lo:hi])

    def head(self, n: int) -> BarFrame:
        """Return the first ``n`` bars."""
        return self._rebuild(self._frame.iloc[: max(0, int(n))])

    def tail(self, n: int) -> BarFrame:
        """Return the last ``n`` bars."""
        count = max(0, int(n))
        return self._rebuild(
            self._frame.iloc[len(self._frame) - count :] if count else self._frame.iloc[0:0]
        )

    def _rebuild(self, frame: pd.DataFrame) -> BarFrame:
        return BarFrame(
            frame.reset_index(drop=True),
            symbol=self.symbol,
            timeframe=self.timeframe,
            dataset_id=self.dataset_id,
        )

    def index_of(self, ts: int) -> int:
        """Return the position of the bar opening exactly at ``ts``.

        Raises:
            DataValidationError: if no bar opens at ``ts``.
        """
        position = int(np.searchsorted(self._columns["ts_open"], int(ts), side="left"))
        if position >= len(self) or int(self._columns["ts_open"][position]) != int(ts):
            raise DataValidationError("no bar opens at that timestamp", ts=int(ts))
        return position

    def window(self, i: int) -> BarWindow:
        """Return the causal view of bars ``[0..i]`` used when deciding at bar ``i``."""
        return BarWindow(self, i)

    def bar(self, i: int) -> Bar:
        """Return bar ``i`` as a standalone value."""
        if not 0 <= i < len(self):
            raise IndexError(f"bar index {i} out of range for {len(self)} bars")
        columns = self._columns
        return Bar(
            ts_open=int(columns["ts_open"][i]),
            open=float(columns["open"][i]),
            high=float(columns["high"][i]),
            low=float(columns["low"][i]),
            close=float(columns["close"][i]),
            volume=float(columns["volume"][i]),
            quote_volume=float(columns["quote_volume"][i]),
            trades=int(columns["trades"][i]),
            is_gap_filled=bool(columns["is_gap_filled"][i]),
        )

    def __iter__(self) -> Iterator[Bar]:
        for i in range(len(self)):
            yield self.bar(i)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BarFrame):
            return NotImplemented
        return (
            self.symbol == other.symbol
            and self.timeframe == other.timeframe
            and self._frame.equals(other._frame)
        )

    def __hash__(self) -> int:  # pragma: no cover - frames are compared, not keyed
        raise TypeError("BarFrame is not hashable")

    def __repr__(self) -> str:
        return (
            f"BarFrame(symbol={self.symbol!r}, timeframe={self.timeframe!r}, "
            f"n_bars={len(self)}, start={format_ts(self.start_ts)}, "
            f"end={format_ts(self.end_ts)}, dataset_id={self.dataset_id!r})"
        )

    @classmethod
    def empty(cls, *, symbol: str, timeframe: str, dataset_id: str = "") -> Self:
        """Return an empty frame with the canonical schema."""
        return cls(empty_bar_frame_df(), symbol=symbol, timeframe=timeframe, dataset_id=dataset_id)


class BarWindow:
    """A read-only view of ``bars[0..i]`` — the enforcement point for INV-3.

    Every accessor stops at ``i``.  Reading a later bar raises
    :class:`~quantlab.core.errors.LookaheadError` rather than returning a value,
    so a strategy cannot see the future by accident, and a strategy that tries
    to on purpose fails loudly in the first backtest instead of producing an
    impressive equity curve.

    Positional indexing follows NumPy: ``window[0]`` is the first bar of the
    series and ``window[-1]`` is the current bar.
    """

    __slots__ = ("_frame", "_i")

    _frame: BarFrame
    _i: int

    def __init__(self, frame: BarFrame, i: int) -> None:
        if not isinstance(i, (int, np.integer)) or isinstance(i, bool):
            raise LookaheadError("bar index must be an integer", index=repr(i))
        index = int(i)
        if index < 0:
            raise LookaheadError("bar index must not be negative", index=index)
        if index >= len(frame):
            raise LookaheadError(
                "bar index is past the end of the series", index=index, n_bars=len(frame)
            )
        object.__setattr__(self, "_frame", frame)
        object.__setattr__(self, "_i", index)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("BarWindow is immutable")

    @property
    def i(self) -> int:
        """Index of the current (closed) bar."""
        return self._i

    @property
    def symbol(self) -> str:
        return self._frame.symbol

    @property
    def timeframe(self) -> str:
        return self._frame.timeframe

    @property
    def bar_ms(self) -> int:
        return self._frame.bar_ms

    def __len__(self) -> int:
        return self._i + 1

    # -- causal column access ---------------------------------------------
    def column(self, name: str) -> np.ndarray:
        """Return a read-only view of ``name`` over ``[0..i]``."""
        return _read_only(self._frame.column(name)[: self._i + 1])

    @property
    def ts_open(self) -> np.ndarray:
        return self.column("ts_open")

    @property
    def open(self) -> np.ndarray:
        return self.column("open")

    @property
    def high(self) -> np.ndarray:
        return self.column("high")

    @property
    def low(self) -> np.ndarray:
        return self.column("low")

    @property
    def close(self) -> np.ndarray:
        return self.column("close")

    @property
    def volume(self) -> np.ndarray:
        return self.column("volume")

    @property
    def quote_volume(self) -> np.ndarray:
        return self.column("quote_volume")

    @property
    def trades(self) -> np.ndarray:
        return self.column("trades")

    @property
    def is_gap_filled(self) -> np.ndarray:
        return self.column("is_gap_filled")

    # -- indexing ----------------------------------------------------------
    def _check(self, index: int) -> int:
        """Resolve ``index`` against the window, raising past the current bar."""
        resolved = index + self._i + 1 if index < 0 else index
        if resolved < 0:
            raise LookaheadError(
                "bar index is before the start of the series", index=index, i=self._i
            )
        if resolved > self._i:
            raise LookaheadError(
                "a strategy deciding at bar i may not read a later bar",
                index=index,
                resolved=resolved,
                i=self._i,
            )
        return resolved

    def __getitem__(self, key: int | slice) -> Bar | list[Bar]:
        if isinstance(key, slice):
            if key.stop is not None:
                # A slice that reaches past the current bar is a look-ahead too,
                # and silently truncating it would hide the strategy's intent.
                self._check(int(key.stop) - 1)
            start, stop, step = key.indices(self._i + 1)
            return [self.bar(position) for position in range(start, stop, step)]
        return self.bar(int(key))

    def bar(self, index: int) -> Bar:
        """Return one bar, refusing any index past the current one."""
        return self._frame.bar(self._check(int(index)))

    def value(self, column: str, index: int = -1) -> float:
        """Return one column of one bar as a float (default: the current bar)."""
        return float(self._frame.column(column)[self._check(int(index))])

    def last(self, column: str, n: int) -> np.ndarray:
        """Return the last ``n`` values of ``column`` ending at the current bar."""
        count = int(n)
        if count <= 0:
            raise LookaheadError("window length must be positive", n=count)
        start = max(0, self._i + 1 - count)
        return _read_only(self._frame.column(column)[start : self._i + 1])

    def df(self, n: int | None = None) -> pd.DataFrame:
        """Return the last ``n`` bars (default: the whole window) as a copy.

        Bounded to :data:`MAX_WINDOW_DF_ROWS` rows so a strategy cannot make a
        multi-gigabyte copy of the dataset per bar.
        """
        count = self._i + 1 if n is None else int(n)
        if count <= 0:
            raise LookaheadError("window length must be positive", n=count)
        if count > MAX_WINDOW_DF_ROWS:
            raise LookaheadError(
                "window DataFrame is limited to MAX_WINDOW_DF_ROWS rows",
                n=count,
                limit=MAX_WINDOW_DF_ROWS,
            )
        start = max(0, self._i + 1 - count)
        window: pd.DataFrame = self._frame.to_pandas().iloc[start : self._i + 1]
        return window.reset_index(drop=True)

    def __repr__(self) -> str:
        return f"BarWindow(i={self._i}, symbol={self.symbol!r}, timeframe={self.timeframe!r})"


# ---------------------------------------------------------------------------
# trading primitives (spec section 8.1)
# ---------------------------------------------------------------------------
class Side(StrEnum):
    LONG = "long"
    SHORT = "short"


class SignalKind(StrEnum):
    LONG = "long"
    FLAT = "flat"
    SHORT = "short"


@dataclass(frozen=True, slots=True)
class Bar:
    """One closed OHLCV bar, left-labelled by its open time."""

    ts_open: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float = 0.0
    trades: int = 0
    is_gap_filled: bool = False


@dataclass(frozen=True, slots=True)
class Signal:
    """What a strategy wants at the close of the current bar."""

    kind: SignalKind
    #: Fraction of equity, in ``[0, max_position_fraction]``; ignored for FLAT.
    target_fraction: float = 1.0
    #: Free text for analysis, at most 32 characters.
    tag: str = ""

    def __post_init__(self) -> None:
        if len(self.tag) > 32:
            raise ValueError("Signal.tag must be at most 32 characters")


@dataclass(frozen=True, slots=True)
class Order:
    """A market order to be filled at the next real bar's open.

    There is deliberately no ``send`` method and no broker behind it: nothing in
    this codebase can transmit an order to an exchange (INV-1).
    """

    bar_index: int
    side: Side
    target_qty: float
    created_ts: int


@dataclass(frozen=True, slots=True)
class Fill:
    bar_index: int
    ts: int
    side: Side
    qty: float
    ref_price: float
    fill_price: float
    fee: float
    slippage_cost: float


@dataclass(frozen=True, slots=True)
class Trade:
    """A completed round trip."""

    trade_no: int
    side: Side
    entry_ts: int
    entry_px: float
    exit_ts: int
    exit_px: float
    qty: float
    fees: float
    slippage_cost: float
    pnl: float
    pnl_pct: float
    bars_held: int
    exit_reason: Literal["signal", "end_of_data", "stop"]


@dataclass(frozen=True, slots=True)
class Position:
    side: Side | None
    qty: float
    entry_px: float
    entry_bar: int

    @property
    def is_flat(self) -> bool:
        return self.side is None or self.qty == 0.0

    @classmethod
    def flat(cls) -> Position:
        return cls(side=None, qty=0.0, entry_px=0.0, entry_bar=-1)
