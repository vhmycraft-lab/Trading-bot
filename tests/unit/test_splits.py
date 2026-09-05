"""Split policy, embargo and walk-forward windows (master spec sections 5, 15)."""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest
import yaml

from quantlab.core.errors import ConfigError, LockboxViolation
from quantlab.core.splits import (
    SEGMENT_NAMES,
    Segment,
    SplitPolicy,
    describe_windows,
    load_split_policy,
    parse_split_policy,
    segment_bar_counts,
    walk_forward_windows,
)
from quantlab.core.types import format_ts, to_ms

HOUR = 3_600_000

BASE_DOCUMENT = {
    "symbol": "BTC/USDT",
    "timeframe": "1h",
    "train": {"start": "2020-01-01T00:00:00Z", "end": "2020-12-31T23:00:00Z"},
    "embargo_bars": 24,
    "validation": {"start": "2021-01-02T00:00:00Z", "end": "2021-06-30T23:00:00Z"},
    "test": {"start": "2021-08-01T00:00:00Z", "end": None},
}


@pytest.fixture
def policy() -> SplitPolicy:
    return parse_split_policy(BASE_DOCUMENT, dataset_id="ds000001")


@pytest.fixture
def default_policy(repo_root: Path) -> SplitPolicy:
    return load_split_policy(repo_root / "configs" / "splits" / "btcusdt_1h.yaml")


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
def test_boundaries_are_normalised_to_utc_ms(policy: SplitPolicy) -> None:
    assert policy.train_start_ts == to_ms("2020-01-01T00:00:00Z")
    assert policy.train_end_ts == to_ms("2020-12-31T23:00:00Z")
    assert policy.test_end_ts is None
    assert policy.bar_ms == HOUR


def test_offset_boundaries_are_converted_not_truncated() -> None:
    document = {
        **BASE_DOCUMENT,
        "train": {"start": "2020-01-01T02:00:00+02:00", "end": "2020-12-31T23:00:00Z"},
    }
    assert parse_split_policy(document).train_start_ts == to_ms("2020-01-01T00:00:00Z")


def test_segments_are_accessible_by_name(policy: SplitPolicy) -> None:
    for name in SEGMENT_NAMES:
        assert isinstance(policy.segment(name), Segment)
    assert policy.segment("train").start_ts == policy.train_start_ts
    with pytest.raises(ConfigError, match="unknown segment"):
        policy.segment("holdout")


def test_segment_membership_and_counts(policy: SplitPolicy) -> None:
    train = policy.train
    assert train.contains(policy.train_start_ts)
    assert train.contains(policy.train_end_ts)
    assert not train.contains(policy.train_end_ts + HOUR)
    assert train.n_bars(HOUR) == 366 * 24  # 2020 was a leap year
    assert policy.test.n_bars(HOUR) is None
    assert policy.test.contains(policy.test_start_ts + 10**9)


def test_segment_bar_counts(default_policy: SplitPolicy) -> None:
    counts = segment_bar_counts(default_policy)
    assert counts["train"] == 47_112
    assert counts["val"] == 16_824
    assert counts["test"] is None


def test_missing_keys_are_reported() -> None:
    with pytest.raises(ConfigError, match="missing a key"):
        parse_split_policy({"symbol": "BTC/USDT"})


def test_a_range_must_be_a_mapping() -> None:
    with pytest.raises(ConfigError, match="mapping with start and end"):
        parse_split_policy({**BASE_DOCUMENT, "train": "2020"})


def test_a_null_start_is_rejected() -> None:
    with pytest.raises(ConfigError, match="start must not be null"):
        parse_split_policy(
            {**BASE_DOCUMENT, "train": {"start": None, "end": "2020-12-31T23:00:00Z"}}
        )


def test_only_test_may_be_open_ended() -> None:
    with pytest.raises(ConfigError, match="only the test segment"):
        parse_split_policy(
            {**BASE_DOCUMENT, "train": {"start": "2020-01-01T00:00:00Z", "end": None}}
        )


def test_an_inverted_range_is_rejected() -> None:
    with pytest.raises(ConfigError, match="precedes its start"):
        parse_split_policy(
            {
                **BASE_DOCUMENT,
                "train": {"start": "2020-12-31T00:00:00Z", "end": "2020-01-01T00:00:00Z"},
            }
        )


def test_an_inverted_test_range_is_rejected() -> None:
    with pytest.raises(ConfigError, match="precedes its start"):
        parse_split_policy(
            {
                **BASE_DOCUMENT,
                "test": {"start": "2021-08-01T00:00:00Z", "end": "2021-07-01T00:00:00Z"},
            }
        )


def test_boundaries_must_sit_on_the_timeframe_grid() -> None:
    with pytest.raises(ConfigError, match="aligned to the timeframe grid"):
        parse_split_policy(
            {
                **BASE_DOCUMENT,
                "train": {"start": "2020-01-01T00:30:00Z", "end": "2020-12-31T23:00:00Z"},
            }
        )


def test_a_negative_embargo_is_rejected() -> None:
    with pytest.raises(ConfigError, match="must not be negative"):
        parse_split_policy({**BASE_DOCUMENT, "embargo_bars": -1})


# ---------------------------------------------------------------------------
# the embargo
# ---------------------------------------------------------------------------
def test_segments_closer_than_the_embargo_are_rejected() -> None:
    """The check that stops a position at the end of train leaking into validation."""
    with pytest.raises(ConfigError, match="closer than the embargo allows"):
        parse_split_policy({**BASE_DOCUMENT, "embargo_bars": 24 * 30})


def test_exactly_the_embargo_is_accepted() -> None:
    """train_end + (embargo + 1) bars is the first legal validation start."""
    document = {
        **BASE_DOCUMENT,
        "embargo_bars": 24,
        "validation": {"start": "2021-01-02T00:00:00Z", "end": "2021-06-30T23:00:00Z"},
    }
    policy = parse_split_policy(document)
    assert policy.val_start_ts == policy.train_end_ts + HOUR * 25


def test_one_bar_short_of_the_embargo_is_rejected() -> None:
    with pytest.raises(ConfigError, match="closer than the embargo allows"):
        parse_split_policy(
            {
                **BASE_DOCUMENT,
                "validation": {"start": "2021-01-01T23:00:00Z", "end": "2021-06-30T23:00:00Z"},
            }
        )


def test_the_shipped_policy_honours_its_own_embargo(default_policy: SplitPolicy) -> None:
    bar = default_policy.bar_ms
    embargo = default_policy.embargo_bars
    assert default_policy.val_start_ts >= default_policy.train_end_ts + bar * (embargo + 1)
    assert default_policy.test_start_ts >= default_policy.val_end_ts + bar * (embargo + 1)


def test_segments_never_overlap(default_policy: SplitPolicy) -> None:
    assert default_policy.train_end_ts < default_policy.val_start_ts
    assert default_policy.val_end_ts < default_policy.test_start_ts


# ---------------------------------------------------------------------------
# split_id
# ---------------------------------------------------------------------------
def test_split_id_is_sixteen_hex_characters(policy: SplitPolicy) -> None:
    assert len(policy.split_id) == 16
    assert set(policy.split_id) <= set("0123456789abcdef")


def test_split_id_is_stable(policy: SplitPolicy) -> None:
    assert policy.split_id == parse_split_policy(BASE_DOCUMENT, dataset_id="ds000001").split_id


def test_split_id_changes_with_the_dataset(policy: SplitPolicy) -> None:
    other = parse_split_policy(BASE_DOCUMENT, dataset_id="ds000002")
    assert other.split_id != policy.split_id


def test_split_id_changes_when_a_boundary_moves(policy: SplitPolicy) -> None:
    moved = parse_split_policy(
        {
            **BASE_DOCUMENT,
            "validation": {"start": "2021-01-03T00:00:00Z", "end": "2021-06-30T23:00:00Z"},
        },
        dataset_id="ds000001",
    )
    assert moved.split_id != policy.split_id


def test_split_id_is_insensitive_to_yaml_key_order(tmp_path: Path, repo_root: Path) -> None:
    source = repo_root / "configs" / "splits" / "btcusdt_1h.yaml"
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    shuffled = tmp_path / "shuffled.yaml"
    shuffled.write_text(
        yaml.safe_dump(dict(reversed(list(document.items()))), sort_keys=False), encoding="utf-8"
    )
    assert load_split_policy(shuffled).split_id == load_split_policy(source).split_id


# ---------------------------------------------------------------------------
# the lockbox boundary
# ---------------------------------------------------------------------------
def test_a_range_below_the_test_start_is_allowed(policy: SplitPolicy) -> None:
    policy.assert_no_test_data(policy.train_start_ts, policy.val_end_ts)
    assert not policy.intersects_test(policy.train_start_ts, policy.val_end_ts)


def test_the_last_bar_before_the_test_start_is_allowed(policy: SplitPolicy) -> None:
    policy.assert_no_test_data(policy.train_start_ts, policy.test_start_ts - HOUR)


def test_the_first_test_bar_is_refused(policy: SplitPolicy) -> None:
    with pytest.raises(LockboxViolation, match="locked test partition"):
        policy.assert_no_test_data(policy.train_start_ts, policy.test_start_ts)


def test_an_open_ended_range_is_refused(policy: SplitPolicy) -> None:
    """ "Give me everything" must not quietly include the held-out period."""
    assert policy.intersects_test(policy.train_start_ts, None)
    with pytest.raises(LockboxViolation):
        policy.assert_no_test_data(policy.train_start_ts, None)


def test_an_empty_range_reaches_nothing(policy: SplitPolicy) -> None:
    assert not policy.intersects_test(policy.test_start_ts + HOUR, policy.test_start_ts)


# ---------------------------------------------------------------------------
# freezing the test end
# ---------------------------------------------------------------------------
def test_freezing_pins_the_end_and_changes_the_id(policy: SplitPolicy) -> None:
    end = to_ms("2025-06-30T23:00:00Z")
    frozen = policy.freeze_test_end(end)
    assert frozen.test_end_ts == end
    assert policy.test_end_ts is None, "freezing must not mutate the original"
    assert frozen.test.n_bars(HOUR) is not None


def test_refreezing_to_the_same_value_is_allowed(policy: SplitPolicy) -> None:
    end = to_ms("2025-06-30T23:00:00Z")
    assert policy.freeze_test_end(end).freeze_test_end(end).test_end_ts == end


def test_refreezing_to_a_different_value_is_refused(policy: SplitPolicy) -> None:
    frozen = policy.freeze_test_end(to_ms("2025-06-30T23:00:00Z"))
    with pytest.raises(ConfigError, match="already frozen"):
        frozen.freeze_test_end(to_ms("2025-07-31T23:00:00Z"))


def test_freezing_before_the_test_start_is_refused(policy: SplitPolicy) -> None:
    with pytest.raises(ConfigError, match="must not precede"):
        policy.freeze_test_end(policy.test_start_ts - HOUR)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def test_load_the_shipped_policy(default_policy: SplitPolicy) -> None:
    assert default_policy.symbol == "BTC/USDT"
    assert default_policy.timeframe == "1h"
    assert default_policy.embargo_bars == 720
    assert format_ts(default_policy.train_start_ts) == "2017-08-17T00:00:00Z"
    assert default_policy.test_end_ts is None


def test_a_missing_policy_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_split_policy(tmp_path / "absent.yaml")


def test_invalid_yaml_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("a: [1, 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_split_policy(path)


def test_a_non_mapping_policy_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_split_policy(path)


def test_describe_mentions_every_segment(default_policy: SplitPolicy) -> None:
    text = default_policy.describe()
    assert default_policy.split_id in text
    for name in ("train", "val", "test"):
        assert name in text
    assert "47112 bars" in text


# ---------------------------------------------------------------------------
# walk-forward windows
# ---------------------------------------------------------------------------
def test_default_policy_yields_the_expected_window_count(default_policy: SplitPolicy) -> None:
    """Spec T09 acceptance criterion: 20-24 rolling windows for the shipped config."""
    windows = walk_forward_windows(default_policy, is_bars=13_140, oos_bars=2_190, step_bars=2_190)
    assert 20 <= len(windows) <= 24
    assert len(windows) == 23


def test_windows_are_contiguous_and_ordered(default_policy: SplitPolicy) -> None:
    windows = walk_forward_windows(default_policy, is_bars=13_140, oos_bars=2_190, step_bars=2_190)
    for k, window in enumerate(windows):
        assert window.k == k
        assert window.is_start_ts <= window.is_end_ts < window.oos_start_ts <= window.oos_end_ts
        assert window.is_segment == f"wf_is:{k}"
        assert window.oos_segment == f"wf_oos:{k}"
    for earlier, later in itertools.pairwise(windows):
        assert later.oos_start_ts > earlier.oos_start_ts


def test_no_window_contains_an_embargoed_bar(default_policy: SplitPolicy) -> None:
    """The property spec section 15.1 asks for: windows never straddle the embargo."""
    embargo_start = default_policy.train_end_ts + default_policy.bar_ms
    embargo_end = default_policy.val_start_ts - default_policy.bar_ms
    windows = walk_forward_windows(default_policy, is_bars=13_140, oos_bars=2_190, step_bars=2_190)
    for window in windows:
        for ts in (window.is_start_ts, window.is_end_ts, window.oos_start_ts, window.oos_end_ts):
            assert not (embargo_start <= ts <= embargo_end), window.describe()


def test_no_window_reaches_the_test_partition(default_policy: SplitPolicy) -> None:
    windows = walk_forward_windows(default_policy, is_bars=13_140, oos_bars=2_190, step_bars=2_190)
    for window in windows:
        assert window.oos_end_ts < default_policy.test_start_ts


def test_windows_stay_inside_train_and_validation(default_policy: SplitPolicy) -> None:
    windows = walk_forward_windows(default_policy, is_bars=13_140, oos_bars=2_190, step_bars=2_190)
    assert windows[0].is_start_ts == default_policy.train_start_ts
    assert windows[-1].oos_end_ts <= default_policy.val_end_ts


def test_anchored_windows_share_a_start(default_policy: SplitPolicy) -> None:
    windows = walk_forward_windows(
        default_policy, is_bars=13_140, oos_bars=2_190, step_bars=2_190, scheme="anchored"
    )
    assert {w.is_start_ts for w in windows} == {default_policy.train_start_ts}
    assert len({w.oos_start_ts for w in windows}) == len(windows)


def test_rolling_windows_move(default_policy: SplitPolicy) -> None:
    windows = walk_forward_windows(
        default_policy, is_bars=13_140, oos_bars=2_190, step_bars=2_190, scheme="rolling"
    )
    assert len({w.is_start_ts for w in windows}) == len(windows)


def test_step_size_controls_the_count(default_policy: SplitPolicy) -> None:
    dense = walk_forward_windows(default_policy, is_bars=13_140, oos_bars=2_190, step_bars=1_095)
    sparse = walk_forward_windows(default_policy, is_bars=13_140, oos_bars=2_190, step_bars=4_380)
    assert len(dense) > 23 > len(sparse)


def test_non_positive_sizes_are_rejected(default_policy: SplitPolicy) -> None:
    with pytest.raises(ConfigError, match="must be positive"):
        walk_forward_windows(default_policy, is_bars=0, oos_bars=10, step_bars=10)


def test_a_timeline_too_short_for_one_window_is_reported(policy: SplitPolicy) -> None:
    with pytest.raises(ConfigError, match="too short"):
        walk_forward_windows(policy, is_bars=100_000, oos_bars=10, step_bars=10)


def test_describe_windows(default_policy: SplitPolicy) -> None:
    windows = walk_forward_windows(default_policy, is_bars=13_140, oos_bars=2_190, step_bars=2_190)
    text = describe_windows(windows[:2])
    assert text.count("\n") == 1
    assert "IS" in text and "OOS" in text
