"""INV-3: a strategy deciding at bar ``t`` cannot observe bar ``t+1``.

These tests exist because look-ahead is the single most common way a backtest
becomes a lie, and it is almost always silent: the equity curve simply looks
excellent.  :class:`~quantlab.core.types.BarWindow` converts that silent failure
into a loud one — every route to a future bar raises
:class:`~quantlab.core.errors.LookaheadError` instead of returning a number.

"Every route" is the point.  A test per accessor is deliberate: a window that
guards ``close`` but not ``high``, or ``__getitem__`` but not slices, would be
worse than no guard at all, because it would be trusted.
"""

from __future__ import annotations

import numpy as np
import pytest
from tests.helpers import make_bars

from quantlab.core.errors import LookaheadError
from quantlab.core.types import MAX_WINDOW_DF_ROWS, BarFrame

COLUMNS = (
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


@pytest.fixture
def frame() -> BarFrame:
    return BarFrame(make_bars(20), symbol="BTC/USDT", timeframe="1h", dataset_id="d0")


# ---------------------------------------------------------------------------
# the window covers exactly [0..i]
# ---------------------------------------------------------------------------
def test_window_length_is_i_plus_one(frame: BarFrame) -> None:
    for i in (0, 1, 9, 19):
        assert len(frame.window(i)) == i + 1


@pytest.mark.parametrize("column", COLUMNS)
def test_every_column_stops_at_the_current_bar(frame: BarFrame, column: str) -> None:
    window = frame.window(7)
    values = window.column(column)
    assert len(values) == 8
    assert np.array_equal(values, frame.column(column)[:8])


@pytest.mark.parametrize("column", COLUMNS)
def test_column_accessors_agree_with_column(frame: BarFrame, column: str) -> None:
    window = frame.window(5)
    assert np.array_equal(getattr(window, column), window.column(column))


def test_window_columns_are_read_only(frame: BarFrame) -> None:
    closes = frame.window(5).close
    assert not closes.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        closes[0] = 0.0


# ---------------------------------------------------------------------------
# reading past the current bar raises
# ---------------------------------------------------------------------------
def test_reading_the_next_bar_raises(frame: BarFrame) -> None:
    window = frame.window(5)
    with pytest.raises(LookaheadError, match="may not read a later bar"):
        window[6]


def test_reading_far_into_the_future_raises(frame: BarFrame) -> None:
    with pytest.raises(LookaheadError):
        frame.window(5)[19]


def test_reading_the_current_bar_is_allowed(frame: BarFrame) -> None:
    window = frame.window(5)
    assert window[5].ts_open == frame.bar(5).ts_open
    assert window[-1].ts_open == frame.bar(5).ts_open


def test_negative_indices_are_relative_to_the_current_bar(frame: BarFrame) -> None:
    window = frame.window(5)
    assert window[-1].ts_open == frame.bar(5).ts_open
    assert window[-6].ts_open == frame.bar(0).ts_open


def test_negative_index_before_the_start_raises(frame: BarFrame) -> None:
    with pytest.raises(LookaheadError, match="before the start"):
        frame.window(5)[-7]


def test_value_accessor_refuses_a_future_bar(frame: BarFrame) -> None:
    window = frame.window(5)
    assert window.value("close") == pytest.approx(float(frame.close[5]))
    assert window.value("close", 0) == pytest.approx(float(frame.close[0]))
    with pytest.raises(LookaheadError):
        window.value("close", 6)


def test_bar_accessor_refuses_a_future_bar(frame: BarFrame) -> None:
    with pytest.raises(LookaheadError):
        frame.window(3).bar(4)


def test_a_slice_reaching_past_the_current_bar_raises(frame: BarFrame) -> None:
    """Truncating the slice instead would hide the strategy's intent."""
    window = frame.window(5)
    with pytest.raises(LookaheadError):
        window[0:7]


def test_a_slice_within_the_window_is_allowed(frame: BarFrame) -> None:
    bars = frame.window(5)[0:6]
    assert isinstance(bars, list)
    assert len(bars) == 6
    assert bars[-1].ts_open == frame.bar(5).ts_open


def test_open_ended_slice_stops_at_the_current_bar(frame: BarFrame) -> None:
    bars = frame.window(5)[:]
    assert len(bars) == 6


def test_last_n_never_reaches_forward(frame: BarFrame) -> None:
    window = frame.window(10)
    assert np.array_equal(window.last("close", 3), frame.close[8:11])
    assert len(window.last("close", 999)) == 11  # clipped at the start, never the end
    with pytest.raises(LookaheadError, match="must be positive"):
        window.last("close", 0)


def test_df_stops_at_the_current_bar(frame: BarFrame) -> None:
    window = frame.window(9)
    assert len(window.df()) == 10
    assert len(window.df(4)) == 4
    assert int(window.df()["ts_open"].iloc[-1]) == int(frame.ts_open[9])


def test_df_is_a_copy(frame: BarFrame) -> None:
    window = frame.window(9)
    copy = window.df()
    copy.loc[0, "close"] = 12_345.0
    assert float(frame.close[0]) != 12_345.0


def test_df_is_bounded(frame: BarFrame) -> None:
    with pytest.raises(LookaheadError, match="MAX_WINDOW_DF_ROWS"):
        frame.window(9).df(MAX_WINDOW_DF_ROWS + 1)
    with pytest.raises(LookaheadError, match="must be positive"):
        frame.window(9).df(0)


# ---------------------------------------------------------------------------
# constructing a window is guarded too
# ---------------------------------------------------------------------------
def test_a_window_past_the_end_of_the_series_raises(frame: BarFrame) -> None:
    with pytest.raises(LookaheadError, match="past the end"):
        frame.window(20)


def test_a_negative_window_index_raises(frame: BarFrame) -> None:
    with pytest.raises(LookaheadError, match="must not be negative"):
        frame.window(-1)


def test_a_non_integer_window_index_raises(frame: BarFrame) -> None:
    with pytest.raises(LookaheadError, match="must be an integer"):
        BarFrame.window(frame, 1.5)  # type: ignore[arg-type]
    with pytest.raises(LookaheadError, match="must be an integer"):
        frame.window(True)  # type: ignore[arg-type]


def test_window_is_immutable(frame: BarFrame) -> None:
    window = frame.window(5)
    with pytest.raises(AttributeError):
        window._i = 19  # type: ignore[misc]


def test_window_exposes_its_market(frame: BarFrame) -> None:
    window = frame.window(5)
    assert window.i == 5
    assert window.symbol == "BTC/USDT"
    assert window.timeframe == "1h"
    assert window.bar_ms == 3_600_000
    assert "i=5" in repr(window)


# ---------------------------------------------------------------------------
# the property that actually matters
# ---------------------------------------------------------------------------
def test_a_window_is_unchanged_by_bars_that_arrive_later() -> None:
    """The load-bearing guarantee: appending future bars cannot alter the past.

    A truncation probe over the whole series.  If any accessor were reading
    beyond ``i``, the value it returned would differ between the short series
    and the long one.
    """
    short = BarFrame(make_bars(30), symbol="BTC/USDT", timeframe="1h")
    long = BarFrame(make_bars(60), symbol="BTC/USDT", timeframe="1h")

    for i in range(30):
        a, b = short.window(i), long.window(i)
        for column in COLUMNS:
            assert np.array_equal(a.column(column), b.column(column)), (i, column)
        assert a.df().equals(b.df())


def test_the_series_after_the_cut_really_does_differ() -> None:
    """Guards the test above against passing vacuously."""
    short = make_bars(30)
    long = make_bars(60)
    assert len(long) > len(short)
    assert not np.array_equal(long["close"].to_numpy()[:30], long["close"].to_numpy()[30:60])


@pytest.mark.parametrize("i", [0, 1, 2, 5, 13, 19])
def test_no_accessor_leaks_a_future_value(frame: BarFrame, i: int) -> None:
    """Sweep every public accessor at every bar and assert nothing exceeds i."""
    window = frame.window(i)
    horizon = int(frame.ts_open[i])
    assert int(window.ts_open.max()) == horizon
    assert int(window.df()["ts_open"].max()) == horizon
    assert window.bar(-1).ts_open == horizon
    assert int(window.last("ts_open", 5).max()) == horizon
    assert int(window.value("ts_open")) == horizon
