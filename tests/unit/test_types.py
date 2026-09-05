"""Timestamps, timezone handling and the canonical bar frame (spec sections 1.3, 7.2)."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from tests.helpers import make_bars

from quantlab.core.errors import DataValidationError
from quantlab.core.types import (
    BAR_COLUMNS,
    BARS_PER_YEAR,
    TIMEFRAME_MS,
    Bar,
    BarFrame,
    Position,
    Side,
    Signal,
    SignalKind,
    bars_per_year,
    coerce_bar_frame_df,
    empty_bar_frame_df,
    format_ts,
    from_ms,
    normalise_epoch,
    timeframe_ms,
    to_ms,
)


# ---------------------------------------------------------------------------
# timeframes
# ---------------------------------------------------------------------------
def test_timeframe_ms_known_values() -> None:
    assert timeframe_ms("1m") == 60_000
    assert timeframe_ms("1h") == 3_600_000
    assert timeframe_ms("1d") == 86_400_000


def test_timeframe_ms_rejects_unknown() -> None:
    with pytest.raises(DataValidationError, match="unsupported timeframe"):
        timeframe_ms("7h")


def test_bars_per_year_matches_a_24_7_market() -> None:
    """Crypto trades continuously, so the year is 8760 hours, not 252 sessions."""
    assert bars_per_year("1h") == 8_760
    assert bars_per_year("1m") == 525_600
    assert bars_per_year("1d") == 365
    with pytest.raises(DataValidationError):
        bars_per_year("7h")


@pytest.mark.parametrize("timeframe", sorted(TIMEFRAME_MS))
def test_annualisation_is_consistent_with_the_bar_length(timeframe: str) -> None:
    year_ms = 365 * 24 * 3_600_000
    assert BARS_PER_YEAR[timeframe] == year_ms // TIMEFRAME_MS[timeframe]


# ---------------------------------------------------------------------------
# timestamp normalisation and timezones
# ---------------------------------------------------------------------------
def test_to_ms_passes_integers_through() -> None:
    assert to_ms(1_672_531_200_000) == 1_672_531_200_000
    assert to_ms(np.int64(42)) == 42


def test_to_ms_parses_iso_utc() -> None:
    assert to_ms("2023-01-01T00:00:00Z") == 1_672_531_200_000


def test_to_ms_converts_a_positive_offset_to_utc() -> None:
    """An offset is honoured, not stripped: +02:00 is two hours earlier in UTC."""
    assert to_ms("2023-01-01T00:00:00+02:00") == to_ms("2022-12-31T22:00:00Z")


def test_to_ms_converts_a_negative_offset_to_utc() -> None:
    assert to_ms("2023-01-01T00:00:00-05:00") == to_ms("2023-01-01T05:00:00Z")


def test_to_ms_treats_a_naive_datetime_as_utc() -> None:
    naive = dt.datetime(2023, 1, 1, 0, 0, 0)  # noqa: DTZ001 - deliberately naive
    aware = dt.datetime(2023, 1, 1, 0, 0, 0, tzinfo=dt.UTC)
    assert to_ms(naive) == to_ms(aware)


def test_to_ms_converts_an_aware_datetime_in_another_zone() -> None:
    tokyo = dt.timezone(dt.timedelta(hours=9))
    assert to_ms(dt.datetime(2023, 1, 1, 9, 0, tzinfo=tokyo)) == to_ms("2023-01-01T00:00:00Z")


def test_to_ms_accepts_a_date_as_midnight_utc() -> None:
    assert to_ms(dt.date(2023, 1, 1)) == to_ms("2023-01-01T00:00:00Z")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2023", "2023-01-01T00:00:00Z"),
        ("2017-08", "2017-08-01T00:00:00Z"),
        ("2023-01-15", "2023-01-15T00:00:00Z"),
    ],
)
def test_to_ms_accepts_the_short_forms_the_cli_uses(text: str, expected: str) -> None:
    assert to_ms(text) == to_ms(expected)


def test_to_ms_accepts_a_bare_epoch_string() -> None:
    assert to_ms("1672531200000") == 1_672_531_200_000


@pytest.mark.parametrize("value", ["", "not-a-date", "2023-13-01T00:00:00Z", None, [1]])
def test_to_ms_rejects_nonsense(value: object) -> None:
    with pytest.raises(DataValidationError):
        to_ms(value)  # type: ignore[arg-type]


def test_to_ms_rejects_a_bool() -> None:
    """``True`` is an int in Python; accepting it would silently mean 1970."""
    with pytest.raises(DataValidationError, match="must not be a bool"):
        to_ms(True)


def test_to_ms_rejects_a_fractional_millisecond() -> None:
    with pytest.raises(DataValidationError, match="whole number of ms"):
        to_ms(1.5)


def test_from_ms_returns_an_aware_utc_datetime() -> None:
    moment = from_ms(1_672_531_200_000)
    assert moment.tzinfo is dt.UTC
    assert moment.year == 2023
    assert moment.hour == 0


def test_round_trip_through_ms_is_lossless() -> None:
    for text in ("2017-08-17T00:00:00Z", "2024-02-29T23:00:00Z", "1970-01-01T00:00:00Z"):
        assert format_ts(to_ms(text)) == text


def test_format_ts_handles_none() -> None:
    assert format_ts(None) == "-"


def test_normalise_epoch_detects_microseconds() -> None:
    """Binance switched the archive to microseconds during 2025; both must work."""
    ms = 1_672_531_200_000
    assert normalise_epoch(ms) == ms
    assert normalise_epoch(ms * 1000) == ms


# ---------------------------------------------------------------------------
# canonical schema
# ---------------------------------------------------------------------------
def test_empty_frame_has_the_canonical_columns_and_dtypes() -> None:
    frame = empty_bar_frame_df()
    assert tuple(frame.columns) == BAR_COLUMNS
    assert frame["ts_open"].dtype == np.int64
    assert frame["close"].dtype == np.float64
    assert frame["is_gap_filled"].dtype == np.bool_


def test_coerce_defaults_is_gap_filled_to_false() -> None:
    raw = make_bars(3).drop(columns=["is_gap_filled"])
    coerced = coerce_bar_frame_df(raw)
    assert not coerced["is_gap_filled"].any()


def test_coerce_reports_missing_columns() -> None:
    with pytest.raises(DataValidationError, match="missing columns"):
        coerce_bar_frame_df(make_bars(3).drop(columns=["close"]))


def test_coerce_normalises_microsecond_timestamps() -> None:
    raw = make_bars(3)
    raw["ts_open"] = raw["ts_open"] * 1000
    assert coerce_bar_frame_df(raw)["ts_open"].tolist() == make_bars(3)["ts_open"].tolist()


def test_coerce_rejects_an_uncastable_column() -> None:
    raw = make_bars(3)
    raw["close"] = ["a", "b", "c"]
    with pytest.raises(DataValidationError, match="canonical dtype"):
        coerce_bar_frame_df(raw)


# ---------------------------------------------------------------------------
# BarFrame
# ---------------------------------------------------------------------------
def test_bar_frame_reports_its_shape(bar_frame: BarFrame) -> None:
    assert bar_frame.n_bars == len(bar_frame) == 24
    assert bar_frame.bar_ms == 3_600_000
    assert bar_frame.symbol == "BTC/USDT"
    assert bar_frame.timeframe == "1h"
    assert bar_frame.dataset_id == "test0000"
    assert format_ts(bar_frame.start_ts) == "2023-01-01T00:00:00Z"
    assert format_ts(bar_frame.end_ts) == "2023-01-01T23:00:00Z"


def test_empty_bar_frame() -> None:
    frame = BarFrame.empty(symbol="BTC/USDT", timeframe="1h")
    assert frame.n_bars == 0
    assert frame.start_ts is None
    assert frame.end_ts is None


def test_bar_frame_is_immutable(bar_frame: BarFrame) -> None:
    with pytest.raises(AttributeError):
        bar_frame.symbol = "ETH/USDT"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        del bar_frame.symbol  # type: ignore[attr-defined]


def test_bar_frame_columns_are_read_only(bar_frame: BarFrame) -> None:
    """A strategy holding a column cannot rewrite history through it."""
    closes = bar_frame.close
    assert not closes.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        closes[0] = 1.0


def test_bar_frame_copies_its_input(bars_df: pd.DataFrame) -> None:
    frame = BarFrame(bars_df, symbol="BTC/USDT", timeframe="1h")
    original = float(frame.close[0])
    bars_df.loc[0, "close"] = 99_999.0
    assert float(frame.close[0]) == original


def test_to_pandas_returns_an_independent_copy(bar_frame: BarFrame) -> None:
    copy = bar_frame.to_pandas()
    copy.loc[0, "close"] = 99_999.0
    assert float(bar_frame.close[0]) != 99_999.0


def test_bar_frame_rejects_an_unknown_timeframe(bars_df: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="unsupported timeframe"):
        BarFrame(bars_df, symbol="BTC/USDT", timeframe="7h")


def test_column_rejects_an_unknown_name(bar_frame: BarFrame) -> None:
    with pytest.raises(DataValidationError, match="unknown bar column"):
        bar_frame.column("vwap")


def test_slice_is_by_timestamp_and_inclusive(bar_frame: BarFrame) -> None:
    start = to_ms("2023-01-01T05:00:00Z")
    end = to_ms("2023-01-01T09:00:00Z")
    sliced = bar_frame.slice(start, end)
    assert sliced.n_bars == 5
    assert sliced.start_ts == start
    assert sliced.end_ts == end
    assert sliced.symbol == bar_frame.symbol
    assert sliced.dataset_id == bar_frame.dataset_id


def test_slice_with_open_bounds(bar_frame: BarFrame) -> None:
    assert bar_frame.slice(None, None).n_bars == 24
    assert bar_frame.slice(to_ms("2023-01-01T20:00:00Z"), None).n_bars == 4
    assert bar_frame.slice(None, to_ms("2023-01-01T03:00:00Z")).n_bars == 4


def test_slice_outside_the_range_is_empty(bar_frame: BarFrame) -> None:
    assert bar_frame.slice(to_ms("2024-01-01T00:00:00Z"), to_ms("2024-02-01T00:00:00Z")).n_bars == 0


def test_head_and_tail(bar_frame: BarFrame) -> None:
    assert bar_frame.head(3).n_bars == 3
    assert bar_frame.tail(3).start_ts == to_ms("2023-01-01T21:00:00Z")
    assert bar_frame.head(0).n_bars == 0
    assert bar_frame.tail(0).n_bars == 0


def test_index_of(bar_frame: BarFrame) -> None:
    assert bar_frame.index_of(to_ms("2023-01-01T05:00:00Z")) == 5
    with pytest.raises(DataValidationError, match="no bar opens"):
        bar_frame.index_of(to_ms("2023-01-01T05:30:00Z"))


def test_bar_access_and_iteration(bar_frame: BarFrame) -> None:
    bar = bar_frame.bar(0)
    assert isinstance(bar, Bar)
    assert bar.ts_open == bar_frame.start_ts
    assert bar.is_gap_filled is False
    assert len(list(bar_frame)) == 24
    with pytest.raises(IndexError):
        bar_frame.bar(24)


def test_bar_frame_equality_and_repr(bars_df: pd.DataFrame, bar_frame: BarFrame) -> None:
    twin = BarFrame(bars_df, symbol="BTC/USDT", timeframe="1h", dataset_id="other")
    assert twin == bar_frame  # identity is the data, not the dataset label
    assert bar_frame != BarFrame(bars_df, symbol="ETH/USDT", timeframe="1h")
    assert bar_frame != "not a frame"
    assert "BTC/USDT" in repr(bar_frame)


def test_bar_frame_is_not_hashable(bar_frame: BarFrame) -> None:
    with pytest.raises(TypeError):
        hash(bar_frame)


# ---------------------------------------------------------------------------
# trading primitives
# ---------------------------------------------------------------------------
def test_signal_defaults_and_tag_limit() -> None:
    signal = Signal(SignalKind.LONG)
    assert signal.target_fraction == 1.0
    assert signal.tag == ""
    with pytest.raises(ValueError, match="32 characters"):
        Signal(SignalKind.LONG, tag="x" * 33)


def test_value_types_are_frozen() -> None:
    import dataclasses

    signal = Signal(SignalKind.FLAT)
    with pytest.raises(dataclasses.FrozenInstanceError):
        signal.kind = SignalKind.LONG  # type: ignore[misc]


def test_position_flat_helper() -> None:
    flat = Position.flat()
    assert flat.is_flat
    assert not Position(side=Side.LONG, qty=1.0, entry_px=10.0, entry_bar=0).is_flat


def test_enums_are_strings() -> None:
    assert Side.LONG == "long"
    assert SignalKind.FLAT == "flat"
