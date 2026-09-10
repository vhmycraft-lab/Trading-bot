"""Bar validation and normalisation (master spec section 7.3).

This module fails **closed**.  Bad market data is the cheapest way to produce a
convincing but wrong backtest, so every check here raises rather than repairing
silently, with two deliberate exceptions that are recorded in the report:

* runs of at most :data:`MAX_AUTO_FILL_BARS` missing bars are filled forward and
  flagged ``is_gap_filled=True`` — the engine refuses to fill orders on those
  bars, so a synthetic bar can never become a synthetic trade
* duplicate rows that agree on every field are collapsed, because exchange
  archives overlap at month boundaries by construction
* bars whose ``ts_open`` is off the timeframe grid are dropped — but only when
  the caller passes ``off_grid="drop"``, and the window they occupied is then a
  gap like any other, so the engine will not trade on it

Anything else — a longer gap, a duplicate timestamp with different data, an
out-of-order row, an impossible OHLC relationship, a negative volume, a NaN —
is an error the caller must decide about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal

import numpy as np
import pandas as pd

from quantlab.core.errors import DataGapError, DataValidationError
from quantlab.core.types import (
    BAR_COLUMNS,
    coerce_bar_frame_df,
    format_ts,
    timeframe_ms,
)

__all__ = [
    "MAX_AUTO_FILL_BARS",
    "Gap",
    "OffGridPolicy",
    "OffGridWindow",
    "ValidationReport",
    "find_duplicates",
    "find_gaps",
    "normalise_bars",
    "validate_bars",
]

#: Runs of missing bars up to this length are filled automatically.
MAX_AUTO_FILL_BARS: Final[int] = 3

#: Rows this many bars apart or fewer are considered adjacent (no gap).
_ADJACENT: Final[int] = 1

#: What to do with bars whose ``ts_open`` is not a multiple of the timeframe.
#:
#: ``"error"`` is the default and the only safe choice for stored data.  An
#: off-grid bar is a real observation the exchange published, but it cannot be
#: placed on our grid without either fabricating its boundaries or colliding
#: with the bar that legitimately owns that slot, so we refuse.
#:
#: ``"drop"`` is for ingesting a vendor archive that is known to contain such a
#: run — see ``docs/DECISIONS/0005-off-grid-bars.md``.  It does not repair anything: it removes
#: the rows and leaves a hole, which the ordinary gap machinery then has to
#: account for.  A long hole still needs ``allow_gaps``, so accepting off-grid
#: data takes two separate, deliberate acknowledgements.
OffGridPolicy = Literal["error", "drop"]


@dataclass(frozen=True, slots=True)
class Gap:
    """A run of missing bars, described by the bars that bracket it."""

    #: Open time of the last bar present before the gap.
    prev_ts: int
    #: Open time of the first bar present after the gap.
    next_ts: int
    #: Number of bars that should have been between them.
    n_missing: int

    @property
    def first_missing_ts(self) -> int:
        return self.prev_ts + (self.next_ts - self.prev_ts) // (self.n_missing + 1)

    def describe(self) -> str:
        return (
            f"{self.n_missing} bar(s) missing between "
            f"{format_ts(self.prev_ts)} and {format_ts(self.next_ts)}"
        )


@dataclass(frozen=True, slots=True)
class OffGridWindow:
    """A contiguous run of dropped bars whose ``ts_open`` was off the grid."""

    #: Open time of the first dropped bar, as the vendor published it.
    first_ts: int
    #: Open time of the last dropped bar, as the vendor published it.
    last_ts: int
    #: How many bars were dropped.
    n_bars: int
    #: Their common offset from the grid, in milliseconds, when they share one.
    offset_ms: int | None = None

    def describe(self) -> str:
        offset = "mixed offsets" if self.offset_ms is None else f"offset +{self.offset_ms}ms"
        return (
            f"{self.n_bars} off-grid bar(s) dropped between "
            f"{format_ts(self.first_ts)} and {format_ts(self.last_ts)} ({offset})"
        )


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """What validation found.  Attached to every ingestion and printed by the CLI."""

    symbol: str
    timeframe: str
    n_bars: int
    start_ts: int | None
    end_ts: int | None
    n_duplicates_dropped: int = 0
    n_rows_reordered: int = 0
    n_gap_filled_bars: int = 0
    #: Synthetic bars replaced by real ones that arrived later.
    n_gap_fills_superseded: int = 0
    #: Bars discarded because their ``ts_open`` was off the timeframe grid.
    n_off_grid_dropped: int = 0
    gaps_filled: tuple[Gap, ...] = field(default_factory=tuple)
    gaps_reported: tuple[Gap, ...] = field(default_factory=tuple)
    off_grid_windows: tuple[OffGridWindow, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        """True when the series needed no repair beyond short gap fills."""
        return not self.gaps_reported and not self.off_grid_windows

    @property
    def n_short_gap_bars(self) -> int:
        """Bars filled for gaps within the auto-fill limit.

        ``n_gap_filled_bars`` counts every synthetic bar, including those filling
        a long gap accepted via ``allow_gaps``; attributing that total to the
        short gaps alone would overstate how routine the repair was.
        """
        return sum(gap.n_missing for gap in self.gaps_filled)

    @property
    def n_gaps(self) -> int:
        return len(self.gaps_filled) + len(self.gaps_reported)

    def summary(self) -> str:
        """A one-paragraph human summary, used by ``quantlab data validate``."""
        lines = [
            f"{self.symbol} {self.timeframe}: {self.n_bars} bars",
            f"  range           : {format_ts(self.start_ts)} .. {format_ts(self.end_ts)}",
            f"  duplicates      : {self.n_duplicates_dropped} dropped",
            f"  reordered rows  : {self.n_rows_reordered}",
            f"  gap-filled bars : {self.n_gap_filled_bars} total,"
            f" {self.n_short_gap_bars} in {len(self.gaps_filled)} short gap(s)",
        ]
        if self.n_gap_fills_superseded:
            lines.append(
                f"  superseded fills: {self.n_gap_fills_superseded}"
                " (real bars arrived for previously filled gaps)"
            )
        if self.off_grid_windows:
            lines.append(
                f"  off-grid bars   : {self.n_off_grid_dropped} dropped"
                f" in {len(self.off_grid_windows)} window(s) (accepted via off_grid='drop')"
            )
            for window in self.off_grid_windows[:10]:
                lines.append(f"      {window.describe()}")
            if len(self.off_grid_windows) > 10:
                lines.append(f"      ... and {len(self.off_grid_windows) - 10} more")
        if self.gaps_reported:
            lines.append(f"  long gaps       : {len(self.gaps_reported)} (accepted via allow_gaps)")
            for gap in self.gaps_reported[:10]:
                lines.append(f"      {gap.describe()}")
            if len(self.gaps_reported) > 10:
                lines.append(f"      ... and {len(self.gaps_reported) - 10} more")
        return "\n".join(lines)


def find_duplicates(ts: np.ndarray) -> np.ndarray:
    """Return the timestamps that appear more than once in ``ts``."""
    if ts.size == 0:
        return np.empty(0, dtype="int64")
    values, counts = np.unique(ts, return_counts=True)
    return values[counts > 1]


def find_gaps(ts: np.ndarray, bar_ms: int) -> tuple[Gap, ...]:
    """Return every run of missing bars in a sorted, de-duplicated ``ts``."""
    if ts.size < 2:
        return ()
    steps = np.diff(ts)
    gap_positions = np.flatnonzero(steps > bar_ms * _ADJACENT)
    return tuple(
        Gap(
            prev_ts=int(ts[position]),
            next_ts=int(ts[position + 1]),
            n_missing=int(steps[position] // bar_ms) - 1,
        )
        for position in gap_positions
    )


def _check_ohlc(df: pd.DataFrame) -> None:
    prices = df[["open", "high", "low", "close"]].to_numpy()
    if not np.all(np.isfinite(prices)):
        raise DataValidationError("OHLC contains NaN or infinite values")
    if not np.all(prices > 0):
        bad = int(np.flatnonzero(~np.all(prices > 0, axis=1))[0])
        raise DataValidationError(
            "OHLC prices must all be > 0", row=bad, ts=int(df["ts_open"].to_numpy()[bad])
        )

    low = df["low"].to_numpy()
    high = df["high"].to_numpy()
    lower = np.minimum(df["open"].to_numpy(), df["close"].to_numpy())
    upper = np.maximum(df["open"].to_numpy(), df["close"].to_numpy())

    bad_low = np.flatnonzero(low > lower)
    if bad_low.size:
        row = int(bad_low[0])
        raise DataValidationError(
            "low must be <= min(open, close)",
            row=row,
            ts=int(df["ts_open"].to_numpy()[row]),
            low=float(low[row]),
            min_open_close=float(lower[row]),
        )
    bad_high = np.flatnonzero(upper > high)
    if bad_high.size:
        row = int(bad_high[0])
        raise DataValidationError(
            "high must be >= max(open, close)",
            row=row,
            ts=int(df["ts_open"].to_numpy()[row]),
            high=float(high[row]),
            max_open_close=float(upper[row]),
        )


def _check_volumes(df: pd.DataFrame) -> None:
    for name in ("volume", "quote_volume"):
        values = df[name].to_numpy()
        if not np.all(np.isfinite(values)):
            raise DataValidationError(f"{name} contains NaN or infinite values")
        negative = np.flatnonzero(values < 0)
        if negative.size:
            row = int(negative[0])
            raise DataValidationError(
                f"{name} must be >= 0", row=row, ts=int(df["ts_open"].to_numpy()[row])
            )
    trades = df["trades"].to_numpy()
    negative = np.flatnonzero(trades < 0)
    if negative.size:
        row = int(negative[0])
        raise DataValidationError(
            "trades must be >= 0", row=row, ts=int(df["ts_open"].to_numpy()[row])
        )


def validate_bars(df: pd.DataFrame, timeframe: str) -> ValidationReport:
    """Validate a canonical bar frame without modifying it (spec section 7.3).

    Args:
        df: A frame carrying the canonical columns of spec section 7.2.
        timeframe: The timeframe the spacing must match, e.g. ``"1h"``.

    Returns:
        A :class:`ValidationReport`.  Gaps of any length are *reported* here,
        never repaired; use :func:`normalise_bars` to repair the short ones.

    Raises:
        DataValidationError: on a missing column, NaN, non-monotonic or
            duplicate ``ts_open``, misaligned spacing, an impossible OHLC
            relationship, or a negative volume or trade count.
    """
    bar_ms = timeframe_ms(timeframe)
    frame = coerce_bar_frame_df(df)
    ts = frame["ts_open"].to_numpy()

    if ts.size:
        duplicates = find_duplicates(ts)
        if duplicates.size:
            raise DataValidationError(
                "duplicate ts_open values",
                n_duplicates=int(duplicates.size),
                first=format_ts(int(duplicates[0])),
            )
        if np.any(np.diff(ts) <= 0):
            position = int(np.flatnonzero(np.diff(ts) <= 0)[0])
            raise DataValidationError(
                "ts_open must be strictly increasing",
                row=position + 1,
                previous=format_ts(int(ts[position])),
                current=format_ts(int(ts[position + 1])),
            )
        misaligned = np.flatnonzero(ts % bar_ms != 0)
        if misaligned.size:
            row = int(misaligned[0])
            raise DataValidationError(
                "ts_open is not aligned to the timeframe grid",
                row=row,
                ts=format_ts(int(ts[row])),
                timeframe=timeframe,
            )
        _check_ohlc(frame)
        _check_volumes(frame)

    gaps = find_gaps(ts, bar_ms)
    return ValidationReport(
        symbol="",
        timeframe=timeframe,
        n_bars=int(ts.size),
        start_ts=int(ts[0]) if ts.size else None,
        end_ts=int(ts[-1]) if ts.size else None,
        n_gap_filled_bars=int(frame["is_gap_filled"].to_numpy().sum()),
        gaps_reported=gaps,
    )


def _supersede_gap_fills(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop synthetic bars for timestamps a real bar now covers.

    A gap fill is a placeholder, not an observation.  When the exchange later
    publishes the real bar — a backfilled archive month, a REST tail — the real
    one wins.  Without this rule the two would look like a contradiction and
    ingestion would refuse to proceed.
    """
    filled = frame["is_gap_filled"].to_numpy()
    if not filled.any():
        return frame, 0
    real_ts = set(frame.loc[~frame["is_gap_filled"], "ts_open"].tolist())
    superseded = frame["is_gap_filled"] & frame["ts_open"].isin(real_ts)
    count = int(superseded.sum())
    if not count:
        return frame, 0
    return frame.loc[~superseded].reset_index(drop=True), count


def _dedupe(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows whose timestamp repeats, keeping the first of identical rows.

    A repeated timestamp carrying *different* data is a real inconsistency and
    is raised rather than resolved by a coin flip.
    """
    ts = frame["ts_open"].to_numpy()
    duplicates = find_duplicates(ts)
    if not duplicates.size:
        return frame, 0

    payload = [name for name in BAR_COLUMNS if name != "is_gap_filled"]
    conflicting = [
        int(value)
        for value in duplicates
        if len(frame.loc[frame["ts_open"] == value, payload].drop_duplicates()) > 1
    ]
    if conflicting:
        raise DataValidationError(
            "the same timestamp appears twice with different data",
            n_conflicts=len(conflicting),
            first=format_ts(conflicting[0]),
        )

    deduped = frame.drop_duplicates(subset="ts_open", keep="first").reset_index(drop=True)
    return deduped, int(len(frame) - len(deduped))


def _fill_gap(previous: pd.Series, timestamps: list[int]) -> list[dict[str, object]]:
    """Build synthetic flat bars carrying forward ``previous``'s close."""
    close = float(previous["close"])
    return [
        {
            "ts_open": int(ts),
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 0.0,
            "quote_volume": 0.0,
            "trades": 0,
            "is_gap_filled": True,
        }
        for ts in timestamps
    ]


def _split_off_grid(
    frame: pd.DataFrame, bar_ms: int
) -> tuple[pd.DataFrame, tuple[OffGridWindow, ...]]:
    """Remove off-grid rows, describing each contiguous run that was removed.

    Runs are grouped by adjacency in the *frame*, not by timestamp arithmetic:
    consecutive dropped rows are one window, which is how an outage actually
    looks in a vendor archive.
    """
    ts = frame["ts_open"].to_numpy()
    off = np.flatnonzero(ts % bar_ms != 0)
    if not off.size:
        return frame, ()

    windows: list[OffGridWindow] = []
    for run in np.split(off, np.flatnonzero(np.diff(off) != 1) + 1):
        offsets = {int(value) % bar_ms for value in ts[run]}
        windows.append(
            OffGridWindow(
                first_ts=int(ts[run[0]]),
                last_ts=int(ts[run[-1]]),
                n_bars=int(run.size),
                offset_ms=offsets.pop() if len(offsets) == 1 else None,
            )
        )

    kept = frame.drop(index=frame.index[off]).reset_index(drop=True)
    return kept, tuple(windows)


def normalise_bars(
    df: pd.DataFrame,
    timeframe: str,
    *,
    symbol: str = "",
    allow_gaps: bool = False,
    off_grid: OffGridPolicy = "error",
) -> tuple[pd.DataFrame, ValidationReport]:
    """Sort, de-duplicate, gap-fill and validate a raw bar frame.

    This is the single entry point every ingestion path uses, so raw data from
    the bulk archive and from the REST tail are normalised identically.

    Args:
        df: Raw bars carrying at least the canonical columns.
        timeframe: The target timeframe.
        symbol: Recorded in the report; does not affect validation.
        allow_gaps: When true, gaps longer than :data:`MAX_AUTO_FILL_BARS` are
            filled the same way and listed in the report instead of raising.
        off_grid: ``"error"`` (the default) refuses bars whose ``ts_open`` is
            not on the timeframe grid.  ``"drop"`` discards them and records
            each run in the report; the hole is then an ordinary gap, so a long
            one still needs ``allow_gaps``.

    Returns:
        The normalised frame and its report.

    Raises:
        DataGapError: on a long gap when ``allow_gaps`` is false.
        DataValidationError: on any inconsistency that cannot be repaired safely.
    """
    bar_ms = timeframe_ms(timeframe)
    frame = coerce_bar_frame_df(df)

    ts = frame["ts_open"].to_numpy()
    n_reordered = int(np.count_nonzero(np.diff(ts) < 0)) if ts.size > 1 else 0
    if n_reordered:
        frame = frame.sort_values("ts_open", kind="stable").reset_index(drop=True)

    frame, n_superseded = _supersede_gap_fills(frame)
    frame, n_duplicates = _dedupe(frame)

    off_grid_windows: tuple[OffGridWindow, ...] = ()
    if len(frame):
        if off_grid == "drop":
            frame, off_grid_windows = _split_off_grid(frame, bar_ms)
        else:
            ts = frame["ts_open"].to_numpy()
            misaligned = np.flatnonzero(ts % bar_ms != 0)
            if misaligned.size:
                row = int(misaligned[0])
                raise DataValidationError(
                    "ts_open is not aligned to the timeframe grid",
                    row=row,
                    ts=format_ts(int(ts[row])),
                    timeframe=timeframe,
                    hint="pass off_grid='drop' to discard the run and leave a gap instead",
                )

    ts = frame["ts_open"].to_numpy()
    gaps = find_gaps(ts, bar_ms)
    long_gaps = tuple(gap for gap in gaps if gap.n_missing > MAX_AUTO_FILL_BARS)
    if long_gaps and not allow_gaps:
        raise DataGapError(
            "gap longer than the auto-fill limit; re-run with allow_gaps to accept it",
            n_long_gaps=len(long_gaps),
            first=long_gaps[0].describe(),
            limit=MAX_AUTO_FILL_BARS,
        )

    filled_rows: list[dict[str, object]] = []
    for gap in gaps:
        previous = frame.loc[frame["ts_open"] == gap.prev_ts].iloc[0]
        timestamps = [gap.prev_ts + bar_ms * (k + 1) for k in range(gap.n_missing)]
        filled_rows.extend(_fill_gap(previous, timestamps))

    if filled_rows:
        frame = pd.concat([frame, pd.DataFrame(filled_rows)], ignore_index=True)
        frame = frame.sort_values("ts_open", kind="stable").reset_index(drop=True)
        frame = coerce_bar_frame_df(frame)

    report = validate_bars(frame, timeframe)
    short_gaps = tuple(gap for gap in gaps if gap.n_missing <= MAX_AUTO_FILL_BARS)
    return frame, ValidationReport(
        symbol=symbol,
        timeframe=timeframe,
        n_bars=report.n_bars,
        start_ts=report.start_ts,
        end_ts=report.end_ts,
        n_duplicates_dropped=n_duplicates,
        n_rows_reordered=n_reordered,
        n_gap_fills_superseded=n_superseded,
        n_gap_filled_bars=int(frame["is_gap_filled"].to_numpy().sum()),
        n_off_grid_dropped=sum(window.n_bars for window in off_grid_windows),
        gaps_filled=short_gaps,
        gaps_reported=long_gaps,
        off_grid_windows=off_grid_windows,
    )
