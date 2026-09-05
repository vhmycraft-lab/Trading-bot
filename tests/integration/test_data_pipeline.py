"""End to end: archive zip -> Parquet -> manifest -> validate -> guarded load.

Spec section 20 lists this as ``test_data_pipeline.py``.  It runs entirely
offline against a synthetic archive and asserts the property the whole phase
exists for: what the backtester receives is exactly what was ingested, minus
anything it is not allowed to see.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.helpers import drop_bars, make_bars, make_kline_zip, ohlcv_rows

from quantlab.adapters.data.binance_archive import (
    ArchiveFile,
    BinanceArchiveIngestor,
    ParquetBarStore,
)
from quantlab.adapters.data.ccxt_rest import CcxtRestSource
from quantlab.adapters.data.guard import PartitionGuard
from quantlab.core.data_validation import validate_bars
from quantlab.core.errors import LockboxViolation
from quantlab.core.splits import parse_split_policy
from quantlab.core.types import format_ts, to_ms

MONTHS = ("2023-01", "2023-02")


def build_archive(frames: dict[str, pd.DataFrame]) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for month, frame in frames.items():
        archive = ArchiveFile(symbol="BTC/USDT", timeframe="1h", period=month, kind="monthly")
        payload = make_kline_zip(frame, name=f"BTCUSDT-1h-{month}.csv")
        payloads[archive.url] = payload
        payloads[archive.checksum_url] = (
            f"{hashlib.sha256(payload).hexdigest()}  {archive.name}\n".encode()
        )
    return payloads


@pytest.fixture
def january() -> pd.DataFrame:
    return make_bars(31 * 24, start_ts=to_ms("2023-01-01T00:00:00Z"), seed=1)


@pytest.fixture
def february() -> pd.DataFrame:
    return make_bars(28 * 24, start_ts=to_ms("2023-02-01T00:00:00Z"), seed=2)


@pytest.fixture
def store(tmp_path: Path) -> ParquetBarStore:
    return ParquetBarStore(tmp_path / "data")


def test_full_pipeline(
    store: ParquetBarStore, january: pd.DataFrame, february: pd.DataFrame
) -> None:
    payloads = build_archive({"2023-01": january, "2023-02": february})
    ingestor = BinanceArchiveIngestor(store, fetch=lambda url: payloads[url])

    # 1. pull -----------------------------------------------------------------
    result = ingestor.pull("BTC/USDT", "1h", months=MONTHS, built_at="2026-01-01T00:00:00Z")
    assert result.n_bars == (31 + 28) * 24
    assert result.report.ok

    # 2. raw and processed are separate --------------------------------------
    raw_files = sorted(p.name for p in store.raw_dir("BTC/USDT", "1h").glob("*.zip"))
    assert raw_files == ["BTCUSDT-1h-2023-01.zip", "BTCUSDT-1h-2023-02.zip"]
    processed = sorted(
        p.relative_to(store.dataset_dir("BTC/USDT", "1h")).as_posix()
        for p in store.dataset_dir("BTC/USDT", "1h").rglob("*.parquet")
    )
    assert processed == ["year=2023/bars.parquet"]

    # 3. the manifest is canonical JSON and names the dataset ------------------
    manifest_text = store.manifest_path("BTC/USDT", "1h").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["dataset_id"] == result.dataset_id
    assert manifest["symbol"] == "BTC/USDT"
    assert manifest_text == json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )

    # 4. validate -------------------------------------------------------------
    report = store.validate("BTC/USDT", "1h")
    assert report.n_bars == (31 + 28) * 24
    assert report.ok

    # 5. load, and check it round-trips the ingested bars ----------------------
    loaded = store.load("BTC/USDT", "1h", 0, to_ms("2023-03-01T00:00:00Z"))
    assert loaded.n_bars == result.n_bars
    assert loaded.dataset_id == result.dataset_id
    expected = pd.concat([january, february], ignore_index=True)
    assert np.allclose(loaded.close, expected["close"].to_numpy())
    assert np.array_equal(loaded.ts_open, expected["ts_open"].to_numpy())
    validate_bars(loaded.to_pandas(), "1h")


def test_dataset_id_is_stable_across_a_rebuild(
    store: ParquetBarStore, january: pd.DataFrame
) -> None:
    payloads = build_archive({"2023-01": january})
    ingestor = BinanceArchiveIngestor(store, fetch=lambda url: payloads[url])

    first = ingestor.pull("BTC/USDT", "1h", months=["2023-01"], built_at="2026-01-01T00:00:00Z")
    parquet = store.dataset_dir("BTC/USDT", "1h") / "year=2023/bars.parquet"
    first_bytes = parquet.read_bytes()

    second = ingestor.pull("BTC/USDT", "1h", months=["2023-01"], built_at="2026-01-01T00:00:00Z")
    assert second.dataset_id == first.dataset_id
    assert parquet.read_bytes() == first_bytes


def test_dataset_id_changes_when_the_data_changes(
    store: ParquetBarStore, january: pd.DataFrame, february: pd.DataFrame
) -> None:
    ingestor = BinanceArchiveIngestor(store, fetch=lambda url: build_archive({})[url])
    one = ingestor.rebuild("BTC/USDT", "1h", extra_frames=[january])
    two = ingestor.rebuild("BTC/USDT", "1h", extra_frames=[january, february])
    assert one.dataset_id != two.dataset_id


def test_a_gapped_archive_is_refused_then_filled_and_flagged(
    store: ParquetBarStore, january: pd.DataFrame
) -> None:
    gapped = drop_bars(january, list(range(100, 110)))
    payloads = build_archive({"2023-01": gapped})
    ingestor = BinanceArchiveIngestor(store, fetch=lambda url: payloads[url])

    with pytest.raises(Exception, match="auto-fill limit"):
        ingestor.pull("BTC/USDT", "1h", months=["2023-01"])

    result = ingestor.pull("BTC/USDT", "1h", months=["2023-01"], allow_gaps=True)
    assert result.n_bars == 31 * 24
    assert result.report.n_gap_filled_bars == 10

    loaded = store.load("BTC/USDT", "1h", 0, to_ms("2023-03-01T00:00:00Z"))
    assert int(loaded.is_gap_filled.sum()) == 10
    # Synthetic bars are contiguous with the rest and carry no volume.
    assert np.all(np.diff(loaded.ts_open) == 3_600_000)
    assert np.all(loaded.volume[loaded.is_gap_filled] == 0.0)


def test_the_rest_tail_extends_the_archive(
    store: ParquetBarStore, january: pd.DataFrame, february: pd.DataFrame
) -> None:
    """The archive lags the market; the tail closes the gap and is validated identically."""
    payloads = build_archive({"2023-01": january})
    ingestor = BinanceArchiveIngestor(store, fetch=lambda url: payloads[url])
    ingestor.pull("BTC/USDT", "1h", months=["2023-01"])

    class FakeExchange:
        def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):  # type: ignore[no-untyped-def]
            rows = ohlcv_rows(february)
            start = 0 if since is None else int(since)
            return [row for row in rows if row[0] >= start][: limit or 1000]

    manifest = store.require_manifest("BTC/USDT", "1h")
    tail = CcxtRestSource(FakeExchange()).fetch_tail(
        "BTC/USDT", "1h", after_ts=manifest.end_ts, now_ms=to_ms("2023-03-01T00:00:00Z")
    )
    assert len(tail) == 28 * 24

    result = ingestor.rebuild("BTC/USDT", "1h", extra_frames=[tail])
    assert result.n_bars == (31 + 28) * 24
    assert format_ts(result.end_ts) == "2023-02-28T23:00:00Z"
    assert store.validate("BTC/USDT", "1h").ok


def test_the_guarded_source_serves_research_and_refuses_the_lockbox(
    store: ParquetBarStore, january: pd.DataFrame, february: pd.DataFrame
) -> None:
    """The property this phase exists for, stated once, end to end."""
    payloads = build_archive({"2023-01": january, "2023-02": february})
    BinanceArchiveIngestor(store, fetch=lambda url: payloads[url]).pull(
        "BTC/USDT", "1h", months=MONTHS
    )

    policy = parse_split_policy(
        {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "train": {"start": "2023-01-01T00:00:00Z", "end": "2023-01-20T23:00:00Z"},
            "embargo_bars": 24,
            "validation": {"start": "2023-01-22T00:00:00Z", "end": "2023-02-10T23:00:00Z"},
            "test": {"start": "2023-02-12T00:00:00Z", "end": None},
        },
        dataset_id=store.dataset_id("BTC/USDT", "1h"),
    )
    guarded = PartitionGuard(store, policy)

    train = guarded.load("BTC/USDT", "1h", policy.train_start_ts, policy.train_end_ts)
    assert train.n_bars == 20 * 24

    val = guarded.load("BTC/USDT", "1h", policy.val_start_ts, policy.val_end_ts)
    assert val.n_bars == (10 + 10) * 24  # Jan 22-31 and Feb 1-10

    with pytest.raises(LockboxViolation):
        guarded.load("BTC/USDT", "1h", policy.train_start_ts, policy.test_start_ts)

    # And the amount of held-out data is not disclosed either.
    _first, last = guarded.available_range("BTC/USDT", "1h")
    assert last == policy.test_start_ts - 3_600_000

    # Only the lockbox route reaches it.
    assert (
        store.load("BTC/USDT", "1h", policy.test_start_ts, to_ms("2023-03-01T00:00:00Z")).n_bars > 0
    )
