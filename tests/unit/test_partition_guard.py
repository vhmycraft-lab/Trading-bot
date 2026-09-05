"""INV-5: research can never load a bar from the held-out test partition.

Look-ahead has two forms.  :class:`~quantlab.core.types.BarWindow` stops a
strategy from reading bar ``t+1`` *within* a run.  The partition guard stops the
whole system from reading the held-out period *at all* while a strategy is still
being developed on it — the slower, more damaging kind of peeking, where a
researcher tunes against the final test set one experiment at a time.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.helpers import make_bars

from quantlab.adapters.data.binance_archive import BinanceArchiveIngestor, ParquetBarStore
from quantlab.adapters.data.guard import PartitionGuard
from quantlab.core.errors import LockboxViolation
from quantlab.core.splits import SplitPolicy, parse_split_policy
from quantlab.core.types import format_ts, to_ms
from quantlab.ports.data import MarketDataSource

HOUR = 3_600_000

POLICY_DOCUMENT = {
    "symbol": "BTC/USDT",
    "timeframe": "1h",
    "train": {"start": "2023-01-01T00:00:00Z", "end": "2023-01-01T11:00:00Z"},
    "embargo_bars": 2,
    "validation": {"start": "2023-01-01T15:00:00Z", "end": "2023-01-01T19:00:00Z"},
    "test": {"start": "2023-01-01T23:00:00Z", "end": None},
}


@pytest.fixture
def policy() -> SplitPolicy:
    return parse_split_policy(POLICY_DOCUMENT, dataset_id="ds0")


@pytest.fixture
def store(tmp_path: Path) -> ParquetBarStore:
    store = ParquetBarStore(tmp_path / "data")
    BinanceArchiveIngestor(store).rebuild("BTC/USDT", "1h", extra_frames=[make_bars(48)])
    return store


@pytest.fixture
def guarded(store: ParquetBarStore, policy: SplitPolicy) -> PartitionGuard:
    return PartitionGuard(store, policy)


def test_the_guard_satisfies_the_port(guarded: PartitionGuard) -> None:
    assert isinstance(guarded, MarketDataSource)


def test_training_data_loads_normally(guarded: PartitionGuard, policy: SplitPolicy) -> None:
    frame = guarded.load("BTC/USDT", "1h", policy.train_start_ts, policy.train_end_ts)
    assert frame.n_bars == 12
    assert format_ts(frame.end_ts) == "2023-01-01T11:00:00Z"


def test_validation_data_loads_normally(guarded: PartitionGuard, policy: SplitPolicy) -> None:
    frame = guarded.load("BTC/USDT", "1h", policy.val_start_ts, policy.val_end_ts)
    assert frame.n_bars == 5


def test_the_last_bar_before_the_test_partition_loads(
    guarded: PartitionGuard, policy: SplitPolicy
) -> None:
    frame = guarded.load("BTC/USDT", "1h", policy.train_start_ts, policy.test_start_ts - HOUR)
    assert frame.end_ts == policy.test_start_ts - HOUR


def test_the_first_test_bar_is_refused(guarded: PartitionGuard, policy: SplitPolicy) -> None:
    with pytest.raises(LockboxViolation, match="locked test partition"):
        guarded.load("BTC/USDT", "1h", policy.train_start_ts, policy.test_start_ts)


def test_a_range_wholly_inside_the_test_partition_is_refused(
    guarded: PartitionGuard, policy: SplitPolicy
) -> None:
    with pytest.raises(LockboxViolation):
        guarded.load("BTC/USDT", "1h", policy.test_start_ts, policy.test_start_ts + 10 * HOUR)


def test_asking_for_everything_is_refused(guarded: PartitionGuard) -> None:
    with pytest.raises(LockboxViolation):
        guarded.load("BTC/USDT", "1h", 0, 2**62)


def test_the_guard_does_not_silently_clip(guarded: PartitionGuard, policy: SplitPolicy) -> None:
    """Returning fewer bars than asked for would turn a bug into a wrong backtest."""
    with pytest.raises(LockboxViolation):
        guarded.load("BTC/USDT", "1h", policy.train_start_ts, policy.test_start_ts + HOUR)


def test_the_error_names_the_split_but_not_the_data(
    guarded: PartitionGuard, policy: SplitPolicy
) -> None:
    with pytest.raises(LockboxViolation) as excinfo:
        guarded.load("BTC/USDT", "1h", 0, 2**62)
    message = str(excinfo.value)
    assert policy.split_id in message
    assert format_ts(policy.test_start_ts) in message


def test_available_range_is_clipped_below_the_test_start(
    guarded: PartitionGuard, store: ParquetBarStore, policy: SplitPolicy
) -> None:
    """Even the *extent* of the held-out data stays hidden from research."""
    true_first, true_last = store.available_range("BTC/USDT", "1h")
    first, last = guarded.available_range("BTC/USDT", "1h")
    assert first == true_first
    assert last == policy.test_start_ts - HOUR
    assert last < true_last


def test_available_range_is_none_when_everything_is_held_out(
    store: ParquetBarStore,
) -> None:
    """A split whose test partition starts before the stored data reveals nothing."""
    early = parse_split_policy(
        {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "train": {"start": "2022-01-01T00:00:00Z", "end": "2022-01-01T11:00:00Z"},
            "embargo_bars": 2,
            "validation": {"start": "2022-01-01T15:00:00Z", "end": "2022-01-01T19:00:00Z"},
            "test": {"start": "2022-01-01T23:00:00Z", "end": None},
        },
        dataset_id="ds0",
    )
    assert PartitionGuard(store, early).available_range("BTC/USDT", "1h") is None


def test_available_range_is_none_without_a_dataset(tmp_path: Path, policy: SplitPolicy) -> None:
    empty = ParquetBarStore(tmp_path / "empty")
    assert PartitionGuard(empty, policy).available_range("BTC/USDT", "1h") is None


def test_dataset_id_passes_through(guarded: PartitionGuard, store: ParquetBarStore) -> None:
    assert guarded.dataset_id("BTC/USDT", "1h") == store.dataset_id("BTC/USDT", "1h")


def test_the_guard_exposes_its_policy(guarded: PartitionGuard, policy: SplitPolicy) -> None:
    assert guarded.policy.split_id == policy.split_id


def test_the_unguarded_store_can_reach_the_test_partition(
    store: ParquetBarStore, policy: SplitPolicy
) -> None:
    """The lockbox profile is the only route to that data, and it is deliberate."""
    frame = store.load("BTC/USDT", "1h", policy.test_start_ts, policy.test_start_ts + 5 * HOUR)
    assert frame.n_bars > 0


# ---------------------------------------------------------------------------
# the composition root attaches the guard, so no caller has to remember to
# ---------------------------------------------------------------------------
def test_every_profile_but_lockbox_is_guarded(tmp_path: Path, default_config_path: Path) -> None:
    from quantlab.container import PROFILES, build_container
    from quantlab.core.config import load_config

    config = load_config(
        [],
        [f"project.db_path={tmp_path / 'q.db'}", f"project.data_dir={tmp_path / 'data'}"],
        default_path=default_config_path,
    )
    for profile in PROFILES:
        container = build_container(config, profile=profile)  # type: ignore[arg-type]
        try:
            guarded_here = isinstance(container.market_data, PartitionGuard)
            assert guarded_here == (profile != "lockbox"), profile
            assert container.may_read_test_partition == (profile == "lockbox")
        finally:
            container.db_engine.dispose()


def test_a_guarded_profile_without_a_split_policy_refuses_to_build(
    tmp_path: Path, default_config_path: Path
) -> None:
    """Failing to build beats building an unguarded research container."""
    from quantlab.container import build_container
    from quantlab.core.config import load_config
    from quantlab.core.errors import ConfigError

    config = load_config(
        [],
        [
            f"project.db_path={tmp_path / 'q.db'}",
            f"project.data_dir={tmp_path / 'data'}",
            f"splits.policy_file={tmp_path / 'missing.yaml'}",
        ],
        default_path=default_config_path,
    )
    with pytest.raises(ConfigError, match="split policy is required"):
        build_container(config, profile="research")

    lockbox = build_container(config, profile="lockbox")
    try:
        assert not isinstance(lockbox.market_data, PartitionGuard)
    finally:
        lockbox.db_engine.dispose()


def test_the_guarded_container_refuses_test_bars_end_to_end(
    tmp_path: Path, default_config_path: Path, repo_root: Path
) -> None:
    from quantlab.container import build_container
    from quantlab.core.config import load_config

    data_dir = tmp_path / "data"
    store = ParquetBarStore(data_dir)
    BinanceArchiveIngestor(store).rebuild(
        "BTC/USDT", "1h", extra_frames=[make_bars(24, start_ts=to_ms("2025-02-01T00:00:00Z"))]
    )

    config = load_config(
        [],
        [
            f"project.db_path={tmp_path / 'q.db'}",
            f"project.data_dir={data_dir}",
            f"splits.policy_file={repo_root / 'configs' / 'splits' / 'btcusdt_1h.yaml'}",
        ],
        default_path=default_config_path,
    )
    container = build_container(config, profile="research")
    try:
        assert container.market_data is not None
        with pytest.raises(LockboxViolation):
            container.market_data.load(
                "BTC/USDT", "1h", to_ms("2025-02-01T00:00:00Z"), to_ms("2025-02-01T23:00:00Z")
            )
    finally:
        container.db_engine.dispose()
