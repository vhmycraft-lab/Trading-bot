"""Validation, duplicates, ordering and gaps (master spec section 7.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.helpers import drop_bars, make_bars

from quantlab.core.data_validation import (
    MAX_AUTO_FILL_BARS,
    Gap,
    ValidationReport,
    find_duplicates,
    find_gaps,
    normalise_bars,
    validate_bars,
)
from quantlab.core.errors import DataGapError, DataValidationError
from quantlab.core.types import format_ts, to_ms

BAR_MS = 3_600_000


# ---------------------------------------------------------------------------
# a clean series validates
# ---------------------------------------------------------------------------
def test_clean_bars_validate() -> None:
    report = validate_bars(make_bars(48), "1h")
    assert report.ok
    assert report.n_bars == 48
    assert report.n_gaps == 0
    assert format_ts(report.start_ts) == "2023-01-01T00:00:00Z"


def test_empty_frame_validates() -> None:
    report = validate_bars(make_bars(0), "1h")
    assert report.n_bars == 0
    assert report.start_ts is None
    assert report.ok


def test_validate_rejects_an_unknown_timeframe() -> None:
    with pytest.raises(DataValidationError, match="unsupported timeframe"):
        validate_bars(make_bars(3), "7h")


# ---------------------------------------------------------------------------
# timestamps: ordering, duplication, alignment
# ---------------------------------------------------------------------------
def test_out_of_order_timestamps_are_rejected() -> None:
    frame = make_bars(5)
    frame = frame.iloc[[0, 2, 1, 3, 4]].reset_index(drop=True)
    with pytest.raises(DataValidationError, match="strictly increasing"):
        validate_bars(frame, "1h")


def test_reversed_series_is_rejected() -> None:
    frame = make_bars(5).iloc[::-1].reset_index(drop=True)
    with pytest.raises(DataValidationError, match="strictly increasing"):
        validate_bars(frame, "1h")


def test_duplicate_timestamps_are_rejected() -> None:
    frame = pd.concat([make_bars(5), make_bars(5).iloc[[2]]], ignore_index=True)
    frame = frame.sort_values("ts_open").reset_index(drop=True)
    with pytest.raises(DataValidationError, match="duplicate ts_open"):
        validate_bars(frame, "1h")


def test_misaligned_timestamps_are_rejected() -> None:
    """A 1h series whose bars open at :30 is not a 1h series."""
    frame = make_bars(5)
    frame["ts_open"] = frame["ts_open"] + 1_800_000
    with pytest.raises(DataValidationError, match="aligned to the timeframe grid"):
        validate_bars(frame, "1h")


def test_find_duplicates() -> None:
    assert find_duplicates(np.array([], dtype="int64")).size == 0
    assert find_duplicates(np.array([1, 2, 3], dtype="int64")).size == 0
    assert find_duplicates(np.array([1, 2, 2, 3, 3], dtype="int64")).tolist() == [2, 3]


# ---------------------------------------------------------------------------
# OHLC consistency
# ---------------------------------------------------------------------------
def test_low_above_open_is_rejected() -> None:
    frame = make_bars(5)
    frame.loc[2, "low"] = float(frame.loc[2, "high"]) + 1.0
    with pytest.raises(DataValidationError, match=r"low must be <="):
        validate_bars(frame, "1h")


def test_high_below_close_is_rejected() -> None:
    frame = make_bars(5)
    frame.loc[3, "high"] = float(frame.loc[3, "low"]) - 1.0
    with pytest.raises(DataValidationError, match=r"high must be >="):
        validate_bars(frame, "1h")


@pytest.mark.parametrize("column", ["open", "high", "low", "close"])
def test_non_positive_prices_are_rejected(column: str) -> None:
    frame = make_bars(5)
    frame.loc[1, column] = 0.0
    with pytest.raises(DataValidationError, match="must all be > 0"):
        validate_bars(frame, "1h")


def test_nan_prices_are_rejected() -> None:
    frame = make_bars(5)
    frame.loc[2, "close"] = np.nan
    with pytest.raises(DataValidationError, match="NaN"):
        validate_bars(frame, "1h")


def test_infinite_prices_are_rejected() -> None:
    frame = make_bars(5)
    frame.loc[2, "high"] = np.inf
    with pytest.raises(DataValidationError, match="NaN or infinite"):
        validate_bars(frame, "1h")


# ---------------------------------------------------------------------------
# volumes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("column", ["volume", "quote_volume"])
def test_negative_volume_is_rejected(column: str) -> None:
    frame = make_bars(5)
    frame.loc[2, column] = -1.0
    with pytest.raises(DataValidationError, match=">= 0"):
        validate_bars(frame, "1h")


def test_nan_volume_is_rejected() -> None:
    frame = make_bars(5)
    frame.loc[2, "volume"] = np.nan
    with pytest.raises(DataValidationError, match="NaN"):
        validate_bars(frame, "1h")


def test_negative_trade_count_is_rejected() -> None:
    frame = make_bars(5)
    frame.loc[2, "trades"] = -3
    with pytest.raises(DataValidationError, match="trades must be >= 0"):
        validate_bars(frame, "1h")


# ---------------------------------------------------------------------------
# gaps
# ---------------------------------------------------------------------------
def test_find_gaps_on_a_contiguous_series() -> None:
    assert find_gaps(make_bars(10)["ts_open"].to_numpy(), BAR_MS) == ()


def test_find_gaps_reports_the_missing_count() -> None:
    ts = drop_bars(make_bars(10), [4, 5])["ts_open"].to_numpy()
    gaps = find_gaps(ts, BAR_MS)
    assert len(gaps) == 1
    assert gaps[0].n_missing == 2
    assert format_ts(gaps[0].prev_ts) == "2023-01-01T03:00:00Z"
    assert format_ts(gaps[0].next_ts) == "2023-01-01T06:00:00Z"


def test_find_gaps_needs_two_bars() -> None:
    assert find_gaps(np.array([1], dtype="int64"), BAR_MS) == ()


def test_gap_describe() -> None:
    gap = Gap(
        prev_ts=to_ms("2023-01-01T03:00:00Z"), next_ts=to_ms("2023-01-01T06:00:00Z"), n_missing=2
    )
    assert "2 bar(s) missing" in gap.describe()
    assert gap.first_missing_ts == to_ms("2023-01-01T04:00:00Z")


# ---------------------------------------------------------------------------
# normalisation: sorting, de-duplication, gap filling
# ---------------------------------------------------------------------------
def test_normalise_sorts_an_unordered_series() -> None:
    shuffled = make_bars(10).sample(frac=1.0, random_state=3).reset_index(drop=True)
    frame, report = normalise_bars(shuffled, "1h")
    ts = frame["ts_open"].to_numpy()
    assert np.all(np.diff(ts) == BAR_MS)
    assert report.n_rows_reordered > 0
    assert report.n_bars == 10


def test_normalise_is_a_no_op_on_a_clean_series() -> None:
    frame, report = normalise_bars(make_bars(10), "1h")
    assert report.n_rows_reordered == 0
    assert report.n_duplicates_dropped == 0
    assert report.n_gap_filled_bars == 0
    assert report.ok
    pd.testing.assert_frame_equal(frame, make_bars(10))


def test_normalise_collapses_identical_duplicates() -> None:
    """Month archives overlap at the boundary; identical rows are safe to merge."""
    bars = make_bars(10)
    doubled = pd.concat([bars, bars.iloc[3:7]], ignore_index=True)
    _frame, report = normalise_bars(doubled, "1h")
    assert report.n_duplicates_dropped == 4
    assert report.n_bars == 10


def test_normalise_refuses_conflicting_duplicates() -> None:
    """The same bar with different prices is a real inconsistency, not an overlap."""
    bars = make_bars(10)
    conflicting = bars.iloc[[4]].copy()
    conflicting.loc[:, "close"] = float(conflicting["close"].iloc[0]) + 5.0
    doubled = pd.concat([bars, conflicting], ignore_index=True)
    with pytest.raises(DataValidationError, match="different data"):
        normalise_bars(doubled, "1h")


@pytest.mark.parametrize("n_missing", [1, 2, 3])
def test_short_gaps_are_filled_and_flagged(n_missing: int) -> None:
    positions = list(range(4, 4 + n_missing))
    frame, report = normalise_bars(drop_bars(make_bars(12), positions), "1h")

    assert report.n_gap_filled_bars == n_missing
    assert len(report.gaps_filled) == 1
    assert report.gaps_reported == ()
    assert report.ok
    assert report.n_bars == 12

    filled = frame.loc[frame["is_gap_filled"]]
    assert len(filled) == n_missing
    previous_close = float(make_bars(12).loc[3, "close"])
    for column in ("open", "high", "low", "close"):
        assert np.allclose(filled[column].to_numpy(), previous_close)
    assert np.all(filled["volume"].to_numpy() == 0.0)
    assert np.all(filled["trades"].to_numpy() == 0)


def test_gap_filled_bars_keep_the_grid_contiguous() -> None:
    frame, _report = normalise_bars(drop_bars(make_bars(12), [4, 5]), "1h")
    ts = frame["ts_open"].to_numpy()
    assert np.all(np.diff(ts) == BAR_MS)


def test_a_long_gap_raises_by_default() -> None:
    missing = list(range(4, 4 + MAX_AUTO_FILL_BARS + 1))
    with pytest.raises(DataGapError, match="longer than the auto-fill limit"):
        normalise_bars(drop_bars(make_bars(20), missing), "1h")


def test_a_ten_bar_gap_raises() -> None:
    with pytest.raises(DataGapError):
        normalise_bars(drop_bars(make_bars(30), list(range(5, 15))), "1h")


def test_allow_gaps_fills_and_reports_a_long_gap() -> None:
    frame, report = normalise_bars(
        drop_bars(make_bars(30), list(range(5, 15))), "1h", allow_gaps=True
    )
    assert report.n_gap_filled_bars == 10
    assert len(report.gaps_reported) == 1
    assert report.gaps_reported[0].n_missing == 10
    assert not report.ok
    assert report.n_bars == 30
    assert np.all(np.diff(frame["ts_open"].to_numpy()) == BAR_MS)


def test_mixed_short_and_long_gaps_are_classified() -> None:
    _frame, report = normalise_bars(
        drop_bars(make_bars(40), [5, 20, 21, 22, 23, 24]), "1h", allow_gaps=True
    )
    assert len(report.gaps_filled) == 1
    assert len(report.gaps_reported) == 1
    assert report.gaps_filled[0].n_missing == 1
    assert report.gaps_reported[0].n_missing == 5
    assert report.n_bars == 40


def test_normalise_rejects_misaligned_timestamps() -> None:
    frame = make_bars(5)
    frame.loc[2, "ts_open"] = int(frame.loc[2, "ts_open"]) + 60_000
    with pytest.raises(DataValidationError, match="aligned to the timeframe grid"):
        normalise_bars(frame, "1h")


def test_normalise_handles_every_problem_at_once() -> None:
    """Unsorted, duplicated and gapped: the shape real archives actually arrive in."""
    bars = make_bars(20)
    damaged = pd.concat([bars.iloc[10:], bars.iloc[:8], bars.iloc[2:5]], ignore_index=True)
    frame, report = normalise_bars(damaged, "1h", symbol="BTC/USDT")

    assert report.n_rows_reordered > 0
    assert report.n_duplicates_dropped == 3
    assert report.n_gap_filled_bars == 2  # bars 8 and 9 were dropped
    assert report.n_bars == 20
    assert np.all(np.diff(frame["ts_open"].to_numpy()) == BAR_MS)
    validate_bars(frame, "1h")  # must not raise


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def test_report_summary_mentions_the_essentials() -> None:
    _frame, report = normalise_bars(
        drop_bars(make_bars(40), [5, 20, 21, 22, 23, 24]),
        "1h",
        symbol="BTC/USDT",
        allow_gaps=True,
    )
    text = report.summary()
    assert "BTC/USDT 1h" in text
    assert "40 bars" in text
    assert "long gaps" in text


def test_report_summary_truncates_a_long_gap_list() -> None:
    report = ValidationReport(
        symbol="BTC/USDT",
        timeframe="1h",
        n_bars=0,
        start_ts=None,
        end_ts=None,
        gaps_reported=tuple(Gap(prev_ts=i, next_ts=i + 1, n_missing=9) for i in range(15)),
    )
    assert "and 5 more" in report.summary()


# ---------------------------------------------------------------------------
# off-grid bars: an exchange outage that resumes on a shifted phase
#
# Binance's own BTCUSDT 1h archive for 2018-02 contains such a run: the exchange
# went down mid-bar, was out for 33 hours, resumed with its kline windows
# phase-shifted by +28m14.789s for 43 bars, then re-synced to the hour.  Those
# 43 bars are real observations, but no hourly grid can hold them, so the only
# honest options are to refuse them or to drop them and admit the hole.  Both
# directions are pinned here: relaxing the check by default would let a shifted
# bar masquerade as an aligned one, which is the failure this guards against.
# ---------------------------------------------------------------------------
def _with_off_grid_run(n: int, start: int, length: int, offset: int) -> pd.DataFrame:
    """A clean series whose bars ``start..start+length`` are shifted by ``offset``."""
    frame = make_bars(n)
    rows = frame.index[start : start + length]
    frame.loc[rows, "ts_open"] = frame.loc[rows, "ts_open"] + offset
    return frame


def test_off_grid_bars_are_rejected_by_default() -> None:
    """The default must stay fail-closed: no flag, no off-grid data."""
    frame = _with_off_grid_run(20, start=5, length=4, offset=1_694_789)
    with pytest.raises(DataValidationError, match="aligned to the timeframe grid"):
        normalise_bars(frame, "1h")


def test_the_rejection_names_the_way_out() -> None:
    frame = _with_off_grid_run(20, start=5, length=4, offset=1_694_789)
    with pytest.raises(DataValidationError) as excinfo:
        normalise_bars(frame, "1h")
    assert "off_grid='drop'" in str(excinfo.value)


def test_dropping_off_grid_bars_records_the_window() -> None:
    frame = _with_off_grid_run(20, start=5, length=4, offset=1_694_789)
    normalised, report = normalise_bars(frame, "1h", off_grid="drop", allow_gaps=True)

    assert report.n_off_grid_dropped == 4
    assert len(report.off_grid_windows) == 1
    window = report.off_grid_windows[0]
    assert window.n_bars == 4
    assert window.offset_ms == 1_694_789
    assert np.all(normalised["ts_open"].to_numpy() % BAR_MS == 0)


def test_a_dropped_window_is_not_silently_forgiven() -> None:
    """``ok`` stays false, so a caller checking it cannot miss the drop."""
    frame = _with_off_grid_run(20, start=5, length=4, offset=1_694_789)
    _normalised, report = normalise_bars(frame, "1h", off_grid="drop", allow_gaps=True)
    assert not report.ok
    assert "off-grid" in report.summary()
    assert "4 off-grid bar(s) dropped" in report.summary()


def test_dropping_alone_does_not_accept_the_hole_it_leaves() -> None:
    """Two separate acknowledgements: --drop-off-grid is not --allow-gaps."""
    frame = _with_off_grid_run(20, start=5, length=8, offset=1_694_789)
    with pytest.raises(DataGapError):
        normalise_bars(frame, "1h", off_grid="drop")


def test_the_dropped_window_becomes_untradeable() -> None:
    """The engine refuses to fill on gap-filled bars, so the hole cannot trade.

    This is the whole point of routing off-grid bars through the gap machinery
    rather than snapping them: the window is preserved as time that existed and
    on which no decision may be made.
    """
    frame = _with_off_grid_run(20, start=5, length=4, offset=1_694_789)
    normalised, _report = normalise_bars(frame, "1h", off_grid="drop", allow_gaps=True)

    original = make_bars(20)["ts_open"].to_numpy()
    dropped = original[5:9]
    filled = normalised.loc[normalised["is_gap_filled"], "ts_open"].to_numpy()
    assert set(dropped.tolist()) == set(filled.tolist())
    # and the series is still a complete, evenly spaced grid
    assert np.all(np.diff(normalised["ts_open"].to_numpy()) == BAR_MS)


def test_two_separate_runs_are_two_windows() -> None:
    frame = make_bars(30)
    frame.loc[frame.index[4:7], "ts_open"] += 1_694_789
    frame.loc[frame.index[20:22], "ts_open"] += 1_694_789
    _normalised, report = normalise_bars(frame, "1h", off_grid="drop", allow_gaps=True)
    assert [window.n_bars for window in report.off_grid_windows] == [3, 2]
    assert report.n_off_grid_dropped == 5


def test_a_window_with_mixed_offsets_reports_no_single_offset() -> None:
    frame = make_bars(20)
    frame.loc[frame.index[5], "ts_open"] += 1_694_789
    frame.loc[frame.index[6], "ts_open"] += 900_000
    _normalised, report = normalise_bars(frame, "1h", off_grid="drop", allow_gaps=True)
    assert len(report.off_grid_windows) == 1
    assert report.off_grid_windows[0].offset_ms is None


def test_stored_data_is_never_forgiven_its_off_grid_bars() -> None:
    """``validate_bars`` has no policy knob: stored off-grid data is a defect."""
    frame = _with_off_grid_run(20, start=5, length=4, offset=1_694_789)
    with pytest.raises(DataValidationError, match="aligned to the timeframe grid"):
        validate_bars(frame, "1h")


def test_a_clean_series_is_unchanged_by_the_drop_policy() -> None:
    """The policy must be inert on data that does not need it."""
    frame = make_bars(48)
    with_policy, report = normalise_bars(frame, "1h", off_grid="drop")
    without, baseline = normalise_bars(frame, "1h")
    pd.testing.assert_frame_equal(with_policy, without)
    assert report.off_grid_windows == ()
    assert report.n_off_grid_dropped == 0
    assert report.ok and baseline.ok


def test_the_summary_does_not_credit_long_gap_fills_to_the_short_gaps() -> None:
    """169 filled bars "in 17 short gaps" overstated how routine the repair was."""
    frame = drop_bars(make_bars(60), range(10, 12))  # short: 2 bars
    frame = drop_bars(frame, range(28, 40))  # long: 12 bars
    _normalised, report = normalise_bars(frame, "1h", allow_gaps=True)

    assert report.n_gap_filled_bars == 14
    assert report.n_short_gap_bars == 2
    assert "14 total, 2 in 1 short gap(s)" in report.summary()
