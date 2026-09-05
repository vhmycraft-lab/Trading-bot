"""Property tests for the data subsystem (master spec section 19).

Where the unit tests check specific cases, these check the invariants across
generated inputs: normalisation always yields a strictly increasing, correctly
spaced series; timestamps round-trip through every timezone; and no window ever
reaches past its own bar.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from tests.helpers import make_bars

from quantlab.core.data_validation import normalise_bars, validate_bars
from quantlab.core.splits import parse_split_policy, walk_forward_windows
from quantlab.core.types import BarFrame, format_ts, from_ms, timeframe_ms, to_ms

SETTINGS = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

TIMEFRAMES = st.sampled_from(["1m", "5m", "1h", "4h", "1d"])


# ---------------------------------------------------------------------------
# timestamps
# ---------------------------------------------------------------------------
@settings(max_examples=100, deadline=None)
@given(
    st.integers(min_value=0, max_value=4_102_444_800_000),  # 1970 .. 2100
    st.integers(min_value=-14 * 3600, max_value=14 * 3600),
)
def test_timestamps_round_trip_through_any_offset(ms: int, offset_seconds: int) -> None:
    """A timestamp survives being written in an arbitrary timezone and read back."""
    ms -= ms % 1000  # ISO 8601 text carries whole seconds
    tz = dt.timezone(dt.timedelta(seconds=offset_seconds - offset_seconds % 60))
    local = from_ms(ms).astimezone(tz)
    assert to_ms(local.isoformat()) == ms
    assert to_ms(local) == ms


@settings(max_examples=100, deadline=None)
@given(st.integers(min_value=0, max_value=4_102_444_800_000))
def test_format_and_parse_are_inverse(ms: int) -> None:
    ms -= ms % 1000
    assert to_ms(format_ts(ms)) == ms


@settings(max_examples=100, deadline=None)
@given(st.integers(min_value=-(2**63) + 1, max_value=2**63 - 1))
def test_format_ts_never_raises(ms: int) -> None:
    """It is used inside error messages, including the lockbox guard's."""
    assert isinstance(format_ts(ms), str)


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------
@SETTINGS
@given(
    n=st.integers(min_value=2, max_value=60),
    timeframe=TIMEFRAMES,
    seed=st.integers(min_value=0, max_value=50),
)
def test_normalise_yields_a_strictly_increasing_grid(n: int, timeframe: str, seed: int) -> None:
    bars = make_bars(n, timeframe=timeframe, seed=seed)
    frame, _report = normalise_bars(bars, timeframe)
    ts = frame["ts_open"].to_numpy()
    assert np.all(np.diff(ts) == timeframe_ms(timeframe))
    validate_bars(frame, timeframe)


@SETTINGS
@given(
    n=st.integers(min_value=3, max_value=40),
    seed=st.integers(min_value=0, max_value=50),
    shuffle_seed=st.integers(min_value=0, max_value=1000),
)
def test_normalisation_is_order_independent(n: int, seed: int, shuffle_seed: int) -> None:
    """However the rows arrive, the normalised series is the same."""
    bars = make_bars(n, seed=seed)
    shuffled = bars.sample(frac=1.0, random_state=shuffle_seed).reset_index(drop=True)
    a, _ = normalise_bars(bars, "1h")
    b, _ = normalise_bars(shuffled, "1h")
    pd.testing.assert_frame_equal(a, b)


@SETTINGS
@given(
    n=st.integers(min_value=2, max_value=40),
    copies=st.integers(min_value=2, max_value=4),
    seed=st.integers(min_value=0, max_value=50),
)
def test_normalisation_is_idempotent_under_duplication(n: int, copies: int, seed: int) -> None:
    """Re-ingesting overlapping archives cannot change the dataset."""
    bars = make_bars(n, seed=seed)
    duplicated = pd.concat([bars] * copies, ignore_index=True)
    once, _ = normalise_bars(bars, "1h")
    many, report = normalise_bars(duplicated, "1h")
    pd.testing.assert_frame_equal(once, many)
    assert report.n_duplicates_dropped == n * (copies - 1)


@SETTINGS
@given(
    n=st.integers(min_value=8, max_value=40),
    gap_at=st.integers(min_value=2, max_value=5),
    gap_len=st.integers(min_value=1, max_value=3),
    seed=st.integers(min_value=0, max_value=50),
)
def test_short_gaps_are_always_filled_flat_and_flagged(
    n: int, gap_at: int, gap_len: int, seed: int
) -> None:
    # The gap must be interior: removing the tail is a truncation, not a gap.
    assume(gap_at + gap_len < n)

    bars = make_bars(n, seed=seed)
    positions = set(range(gap_at, gap_at + gap_len))
    gapped = bars.iloc[[i for i in range(n) if i not in positions]].reset_index(drop=True)

    frame, report = normalise_bars(gapped, "1h")
    assert report.n_bars == n
    assert report.n_gap_filled_bars == gap_len

    filled = frame.loc[frame["is_gap_filled"]]
    # A synthetic bar has no range and no volume: it can never look like a signal.
    assert np.allclose(filled["high"].to_numpy(), filled["low"].to_numpy())
    assert np.all(filled["volume"].to_numpy() == 0.0)


# ---------------------------------------------------------------------------
# frames and windows
# ---------------------------------------------------------------------------
@SETTINGS
@given(
    n=st.integers(min_value=1, max_value=40),
    seed=st.integers(min_value=0, max_value=50),
)
def test_slicing_a_frame_never_invents_or_loses_bars(n: int, seed: int) -> None:
    frame = BarFrame(make_bars(n, seed=seed), symbol="BTC/USDT", timeframe="1h")
    assert frame.slice(frame.start_ts, frame.end_ts).n_bars == n
    for i in range(n):
        ts = int(frame.ts_open[i])
        assert frame.slice(ts, ts).n_bars == 1
        assert frame.index_of(ts) == i


@SETTINGS
@given(
    n=st.integers(min_value=1, max_value=30),
    seed=st.integers(min_value=0, max_value=50),
)
def test_a_window_never_reaches_past_its_own_bar(n: int, seed: int) -> None:
    """INV-3, over generated series rather than one fixture."""
    frame = BarFrame(make_bars(n, seed=seed), symbol="BTC/USDT", timeframe="1h")
    for i in range(n):
        window = frame.window(i)
        assert len(window) == i + 1
        assert int(window.ts_open.max()) == int(frame.ts_open[i])


@SETTINGS
@given(
    n=st.integers(min_value=2, max_value=30),
    extra=st.integers(min_value=1, max_value=30),
    seed=st.integers(min_value=0, max_value=50),
)
def test_appending_future_bars_cannot_change_the_past(n: int, extra: int, seed: int) -> None:
    short = BarFrame(make_bars(n, seed=seed), symbol="BTC/USDT", timeframe="1h")
    long = BarFrame(make_bars(n + extra, seed=seed), symbol="BTC/USDT", timeframe="1h")
    for i in range(n):
        assert np.array_equal(short.window(i).close, long.window(i).close)


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------
@SETTINGS
@given(
    train_days=st.integers(min_value=2, max_value=30),
    embargo=st.integers(min_value=0, max_value=48),
    val_days=st.integers(min_value=2, max_value=20),
)
def test_segments_never_overlap_and_honour_the_embargo(
    train_days: int, embargo: int, val_days: int
) -> None:
    hour = 3_600_000
    train_start = to_ms("2020-01-01T00:00:00Z")
    train_end = train_start + train_days * 24 * hour - hour
    val_start = train_end + (embargo + 1) * hour
    val_end = val_start + val_days * 24 * hour - hour
    test_start = val_end + (embargo + 1) * hour

    policy = parse_split_policy(
        {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "train": {"start": train_start, "end": train_end},
            "embargo_bars": embargo,
            "validation": {"start": val_start, "end": val_end},
            "test": {"start": test_start, "end": None},
        }
    )
    assert policy.train_end_ts < policy.val_start_ts < policy.val_end_ts < policy.test_start_ts
    assert policy.val_start_ts - policy.train_end_ts >= (embargo + 1) * hour
    assert not policy.intersects_test(policy.train_start_ts, policy.val_end_ts)
    assert policy.intersects_test(policy.train_start_ts, policy.test_start_ts)


@SETTINGS
@given(
    is_bars=st.integers(min_value=10, max_value=200),
    oos_bars=st.integers(min_value=5, max_value=100),
    step_bars=st.integers(min_value=5, max_value=100),
)
def test_walk_forward_windows_are_well_formed(is_bars: int, oos_bars: int, step_bars: int) -> None:
    hour = 3_600_000
    train_start = to_ms("2020-01-01T00:00:00Z")
    train_end = train_start + 2000 * hour
    val_start = train_end + 25 * hour
    val_end = val_start + 800 * hour

    policy = parse_split_policy(
        {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "train": {"start": train_start, "end": train_end},
            "embargo_bars": 24,
            "validation": {"start": val_start, "end": val_end},
            "test": {"start": val_end + 25 * hour, "end": None},
        }
    )
    if is_bars + oos_bars > 2801:
        return

    windows = walk_forward_windows(policy, is_bars=is_bars, oos_bars=oos_bars, step_bars=step_bars)
    embargo_range = (policy.train_end_ts + hour, policy.val_start_ts - hour)
    for window in windows:
        assert window.is_start_ts <= window.is_end_ts < window.oos_start_ts <= window.oos_end_ts
        assert window.oos_end_ts < policy.test_start_ts
        for ts in (window.is_start_ts, window.is_end_ts, window.oos_start_ts, window.oos_end_ts):
            assert not embargo_range[0] <= ts <= embargo_range[1]
