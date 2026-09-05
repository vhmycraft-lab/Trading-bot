"""Archive parsing, checksums, Parquet storage and loading (spec sections 7.1, 7.2, 21.6).

Everything here is offline: archives are built in memory by ``tests.helpers``
and the downloader is a dict-backed fake.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.helpers import drop_bars, make_bars, make_kline_zip

from quantlab.adapters.data.binance_archive import (
    ArchiveFile,
    BinanceArchiveIngestor,
    Manifest,
    ManifestEntry,
    ParquetBarStore,
    exchange_symbol,
    months_between,
    parse_kline_bytes,
    read_checksum_file,
    verify_checksum,
)
from quantlab.core.errors import DataError, DataValidationError, ManifestMismatchError
from quantlab.core.types import BAR_COLUMNS, format_ts, to_ms


def fake_fetcher(payloads: dict[str, bytes]):
    """A downloader backed by a dict, so ingestion tests never touch the network."""

    def fetch(url: str) -> bytes:
        try:
            return payloads[url]
        except KeyError:
            raise DataError("404 in the fake archive", url=url) from None

    return fetch


def archive_payloads(frame: pd.DataFrame, *, month: str = "2023-01") -> dict[str, bytes]:
    archive = ArchiveFile(symbol="BTC/USDT", timeframe="1h", period=month, kind="monthly")
    payload = make_kline_zip(frame, name=f"BTCUSDT-1h-{month}.csv")
    digest = hashlib.sha256(payload).hexdigest()
    return {
        archive.url: payload,
        archive.checksum_url: f"{digest}  BTCUSDT-1h-{month}.zip\n".encode(),
    }


# ---------------------------------------------------------------------------
# paths and naming
# ---------------------------------------------------------------------------
def test_exchange_symbol_strips_the_separator() -> None:
    assert exchange_symbol("BTC/USDT") == "BTCUSDT"
    assert exchange_symbol("btc-usdt") == "BTCUSDT"


def test_archive_urls_match_the_published_layout() -> None:
    archive = ArchiveFile(symbol="BTC/USDT", timeframe="1h", period="2023-01", kind="monthly")
    assert archive.name == "BTCUSDT-1h-2023-01.zip"
    assert archive.url.endswith("/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2023-01.zip")
    assert archive.checksum_url == archive.url + ".CHECKSUM"


def test_months_between() -> None:
    assert months_between("2023-01", "2023-03") == ("2023-01", "2023-02", "2023-03")
    assert months_between("2022-11", "2023-02") == ("2022-11", "2022-12", "2023-01", "2023-02")
    assert months_between("2023-05", "2023-05") == ("2023-05",)
    with pytest.raises(DataError, match="precedes"):
        months_between("2023-05", "2023-04")


def test_raw_and_processed_directories_are_separate(parquet_store: ParquetBarStore) -> None:
    """Downloaded archives never sit alongside the processed dataset."""
    raw = parquet_store.raw_dir("BTC/USDT", "1h")
    processed = parquet_store.dataset_dir("BTC/USDT", "1h")
    assert "raw" in raw.parts
    assert "raw" not in processed.parts
    assert not str(processed).startswith(str(raw))
    assert not str(raw).startswith(str(processed))


# ---------------------------------------------------------------------------
# checksums
# ---------------------------------------------------------------------------
def test_read_checksum_file() -> None:
    digest = "a" * 64
    assert read_checksum_file(f"{digest}  BTCUSDT-1h-2023-01.zip\n") == digest


@pytest.mark.parametrize("text", ["", "short  file.zip", "z" * 64])
def test_malformed_checksum_files_are_rejected(text: str) -> None:
    with pytest.raises(DataError):
        read_checksum_file(text)


def test_verify_checksum_accepts_a_match(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"quantlab")
    verify_checksum(path, hashlib.sha256(b"quantlab").hexdigest())


def test_verify_checksum_aborts_on_a_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"quantlab")
    with pytest.raises(ManifestMismatchError, match="checksum mismatch"):
        verify_checksum(path, "0" * 64)


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
def test_parse_headerless_archive() -> None:
    bars = make_bars(24)
    parsed = parse_kline_bytes(make_kline_zip(bars))
    assert tuple(parsed.columns) == BAR_COLUMNS
    assert parsed["ts_open"].tolist() == bars["ts_open"].tolist()
    assert np.allclose(parsed["close"].to_numpy(), bars["close"].to_numpy())
    assert not parsed["is_gap_filled"].any()


def test_parse_archive_with_a_header_row() -> None:
    """Binance added a header row to the archive during 2024."""
    bars = make_bars(24)
    parsed = parse_kline_bytes(make_kline_zip(bars, header=True))
    assert parsed["ts_open"].tolist() == bars["ts_open"].tolist()


def test_parse_archive_with_microsecond_timestamps() -> None:
    """And switched the timestamp unit during 2025; both generations must load."""
    bars = make_bars(24)
    parsed = parse_kline_bytes(make_kline_zip(bars, microseconds=True))
    assert parsed["ts_open"].tolist() == bars["ts_open"].tolist()


def test_parse_rejects_a_non_zip() -> None:
    with pytest.raises(DataValidationError, match="not a valid zip"):
        parse_kline_bytes(b"definitely not a zip")


def test_parse_rejects_an_empty_csv(tmp_path: Path) -> None:
    import io

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("empty.csv", "")
    with pytest.raises(DataValidationError, match="empty"):
        parse_kline_bytes(buffer.getvalue())


def test_parse_rejects_an_archive_with_two_csvs() -> None:
    import io

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("a.csv", "1,2,3\n")
        archive.writestr("b.csv", "1,2,3\n")
    with pytest.raises(DataValidationError, match="exactly one CSV"):
        parse_kline_bytes(buffer.getvalue())


def test_parse_rejects_a_csv_missing_columns() -> None:
    import io

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("a.csv", "open_time,open\n1,2\n")
    with pytest.raises(DataValidationError, match="missing columns"):
        parse_kline_bytes(buffer.getvalue())


# ---------------------------------------------------------------------------
# manifest and dataset_id
# ---------------------------------------------------------------------------
def test_manifest_dataset_id_is_content_addressed() -> None:
    entries = (
        ManifestEntry(path="year=2023/bars.parquet", sha256="aa", n_bars=2, first_ts=0, last_ts=1),
    )
    manifest = Manifest(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1h",
        files=entries,
        built_at="2026-01-01T00:00:00Z",
        source_versions={},
    )
    changed = Manifest(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1h",
        files=(entries[0].__class__(**{**entries[0].to_dict(), "sha256": "bb"}),),
        built_at="2026-01-01T00:00:00Z",
        source_versions={},
    )
    assert len(manifest.dataset_id) == 16
    assert manifest.dataset_id != changed.dataset_id


def test_manifest_round_trips_through_json() -> None:
    manifest = Manifest(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1h",
        files=(
            ManifestEntry(
                path="year=2023/bars.parquet", sha256="aa", n_bars=2, first_ts=0, last_ts=1
            ),
        ),
        built_at="2026-01-01T00:00:00Z",
        source_versions={"archives": ["x.zip"]},
    )
    restored = Manifest.from_dict(json.loads(json.dumps(manifest.to_dict())))
    assert restored.dataset_id == manifest.dataset_id
    assert restored.files == manifest.files


def test_missing_dataset_explains_how_to_create_one(parquet_store: ParquetBarStore) -> None:
    assert parquet_store.read_manifest("BTC/USDT", "1h") is None
    assert parquet_store.available_range("BTC/USDT", "1h") is None
    with pytest.raises(DataError, match="quantlab data pull"):
        parquet_store.require_manifest("BTC/USDT", "1h")


def test_corrupt_manifest_json_is_rejected(parquet_store: ParquetBarStore) -> None:
    path = parquet_store.manifest_path("BTC/USDT", "1h")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ManifestMismatchError, match="not valid JSON"):
        parquet_store.read_manifest("BTC/USDT", "1h")


# ---------------------------------------------------------------------------
# ingestion
# ---------------------------------------------------------------------------
def test_pull_writes_parquet_and_a_manifest(parquet_store: ParquetBarStore) -> None:
    bars = make_bars(72)
    ingestor = BinanceArchiveIngestor(parquet_store, fetch=fake_fetcher(archive_payloads(bars)))
    result = ingestor.pull("BTC/USDT", "1h", months=["2023-01"], built_at="2026-01-01T00:00:00Z")

    assert result.n_bars == 72
    assert result.files_written == ("year=2023/bars.parquet",)
    assert (parquet_store.dataset_dir("BTC/USDT", "1h") / "year=2023/bars.parquet").is_file()
    assert parquet_store.manifest_path("BTC/USDT", "1h").is_file()
    assert len(result.dataset_id) == 16
    assert "72 bars" in result.summary()


def test_pull_verifies_the_checksum(parquet_store: ParquetBarStore) -> None:
    payloads = archive_payloads(make_bars(24))
    archive = ArchiveFile(symbol="BTC/USDT", timeframe="1h", period="2023-01", kind="monthly")
    payloads[archive.checksum_url] = b"0" * 64 + b"  x.zip\n"

    ingestor = BinanceArchiveIngestor(parquet_store, fetch=fake_fetcher(payloads))
    with pytest.raises(ManifestMismatchError, match="checksum mismatch"):
        ingestor.pull("BTC/USDT", "1h", months=["2023-01"])


def test_a_failed_checksum_download_leaves_no_partial_archive(
    parquet_store: ParquetBarStore,
) -> None:
    payloads = archive_payloads(make_bars(24))
    archive = ArchiveFile(symbol="BTC/USDT", timeframe="1h", period="2023-01", kind="monthly")
    del payloads[archive.checksum_url]

    ingestor = BinanceArchiveIngestor(parquet_store, fetch=fake_fetcher(payloads))
    with pytest.raises(DataError):
        ingestor.pull("BTC/USDT", "1h", months=["2023-01"])
    assert not (parquet_store.raw_dir("BTC/USDT", "1h") / archive.name).exists()


def test_pull_is_idempotent_and_hashes_are_stable(parquet_store: ParquetBarStore) -> None:
    """Spec T06: a second run must rebuild identical hashes."""
    ingestor = BinanceArchiveIngestor(
        parquet_store, fetch=fake_fetcher(archive_payloads(make_bars(72)))
    )
    first = ingestor.pull("BTC/USDT", "1h", months=["2023-01"], built_at="2026-01-01T00:00:00Z")
    manifest_a = json.loads(parquet_store.manifest_path("BTC/USDT", "1h").read_text("utf-8"))

    second = ingestor.pull("BTC/USDT", "1h", months=["2023-01"], built_at="2026-01-01T00:00:00Z")
    manifest_b = json.loads(parquet_store.manifest_path("BTC/USDT", "1h").read_text("utf-8"))

    assert first.dataset_id == second.dataset_id
    assert manifest_a == manifest_b


def test_ingest_without_a_fetcher_or_cache_fails_clearly(
    parquet_store: ParquetBarStore,
) -> None:
    ingestor = BinanceArchiveIngestor(parquet_store)
    with pytest.raises(DataError, match="no cached archives"):
        ingestor.rebuild("BTC/USDT", "1h")
    with pytest.raises(DataError, match="no fetcher configured"):
        ingestor.ensure_archive(
            ArchiveFile(symbol="BTC/USDT", timeframe="1h", period="2023-01", kind="monthly")
        )


def test_two_months_merge_into_one_dataset(parquet_store: ParquetBarStore) -> None:
    january = make_bars(31 * 24)
    february = make_bars(28 * 24, start_ts=to_ms("2023-02-01T00:00:00Z"), seed=11)
    payloads = {
        **archive_payloads(january, month="2023-01"),
        **archive_payloads(february, month="2023-02"),
    }

    ingestor = BinanceArchiveIngestor(parquet_store, fetch=fake_fetcher(payloads))
    result = ingestor.pull("BTC/USDT", "1h", months=["2023-01", "2023-02"])
    assert result.n_bars == (31 + 28) * 24
    assert format_ts(result.end_ts) == "2023-02-28T23:00:00Z"


def test_overlapping_months_are_deduplicated(parquet_store: ParquetBarStore) -> None:
    bars = make_bars(48)
    payloads = {
        **archive_payloads(bars, month="2023-01"),
        **archive_payloads(bars.iloc[24:], month="2023-02"),
    }
    ingestor = BinanceArchiveIngestor(parquet_store, fetch=fake_fetcher(payloads))
    result = ingestor.pull("BTC/USDT", "1h", months=["2023-01", "2023-02"])
    assert result.n_bars == 48
    assert result.report.n_duplicates_dropped == 24


def test_years_are_written_to_separate_files(parquet_store: ParquetBarStore) -> None:
    bars = make_bars(48, start_ts=to_ms("2022-12-31T00:00:00Z"))
    ingestor = BinanceArchiveIngestor(parquet_store)
    result = ingestor.rebuild("BTC/USDT", "1h", extra_frames=[bars])
    assert result.files_written == ("year=2022/bars.parquet", "year=2023/bars.parquet")


def test_a_gap_in_the_archive_is_refused_then_accepted(parquet_store: ParquetBarStore) -> None:
    gapped = drop_bars(make_bars(48), list(range(10, 20)))
    ingestor = BinanceArchiveIngestor(parquet_store)
    with pytest.raises(Exception, match="auto-fill limit"):
        ingestor.rebuild("BTC/USDT", "1h", extra_frames=[gapped])

    result = ingestor.rebuild("BTC/USDT", "1h", extra_frames=[gapped], allow_gaps=True)
    assert result.report.n_gap_filled_bars == 10


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def test_load_returns_the_requested_range(ingested_store: ParquetBarStore) -> None:
    frame = ingested_store.load(
        "BTC/USDT", "1h", to_ms("2023-01-01T05:00:00Z"), to_ms("2023-01-01T09:00:00Z")
    )
    assert frame.n_bars == 5
    assert format_ts(frame.start_ts) == "2023-01-01T05:00:00Z"
    assert format_ts(frame.end_ts) == "2023-01-01T09:00:00Z"
    assert frame.symbol == "BTC/USDT"
    assert frame.dataset_id == ingested_store.dataset_id("BTC/USDT", "1h")


def test_load_round_trips_the_data(ingested_store: ParquetBarStore, bars_df: pd.DataFrame) -> None:
    loaded = ingested_store.load(
        "BTC/USDT", "1h", int(bars_df["ts_open"].iloc[0]), int(bars_df["ts_open"].iloc[-1])
    )
    pd.testing.assert_frame_equal(loaded.to_pandas(), bars_df)


def test_load_outside_the_stored_range_is_empty(ingested_store: ParquetBarStore) -> None:
    frame = ingested_store.load(
        "BTC/USDT", "1h", to_ms("2030-01-01T00:00:00Z"), to_ms("2030-02-01T00:00:00Z")
    )
    assert frame.n_bars == 0


def test_available_range(ingested_store: ParquetBarStore) -> None:
    first, last = ingested_store.available_range("BTC/USDT", "1h")
    assert format_ts(first) == "2023-01-01T00:00:00Z"
    assert format_ts(last) == "2023-01-01T23:00:00Z"


def test_load_reads_only_the_years_it_needs(
    parquet_store: ParquetBarStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Efficiency: a one-day backtest must not read eight years of Parquet."""
    bars = make_bars(24 * 40, start_ts=to_ms("2022-12-01T00:00:00Z"))
    BinanceArchiveIngestor(parquet_store).rebuild("BTC/USDT", "1h", extra_frames=[bars])

    read_paths: list[str] = []
    original = pd.read_parquet

    def spy(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        read_paths.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)
    parquet_store.load(
        "BTC/USDT", "1h", to_ms("2023-01-05T00:00:00Z"), to_ms("2023-01-06T00:00:00Z")
    )
    assert len(read_paths) == 1
    assert "year=2023" in read_paths[0]


def test_a_tampered_parquet_file_is_detected(ingested_store: ParquetBarStore) -> None:
    """INV: the manifest hash is verified before every load (spec section 21.6)."""
    path = ingested_store.dataset_dir("BTC/USDT", "1h") / "year=2023/bars.parquet"
    tampered = pd.read_parquet(path)
    tampered.loc[0, "close"] = 99_999.0
    tampered.to_parquet(path, engine="pyarrow", compression="zstd", index=False)

    with pytest.raises(ManifestMismatchError, match="does not match its manifest hash"):
        ingested_store.load("BTC/USDT", "1h", 0, 2**62)


def test_a_missing_parquet_file_is_detected(ingested_store: ParquetBarStore) -> None:
    (ingested_store.dataset_dir("BTC/USDT", "1h") / "year=2023/bars.parquet").unlink()
    with pytest.raises(ManifestMismatchError, match="missing"):
        ingested_store.load("BTC/USDT", "1h", 0, 2**62)


def test_validate_reports_on_the_stored_dataset(ingested_store: ParquetBarStore) -> None:
    report = ingested_store.validate("BTC/USDT", "1h")
    assert report.n_bars == 24
    assert report.ok


# ---------------------------------------------------------------------------
# rebuilding is additive
# ---------------------------------------------------------------------------
def test_rebuild_extends_the_stored_dataset_instead_of_replacing_it(
    parquet_store: ParquetBarStore,
) -> None:
    """Regression: an update must never truncate history it did not fetch.

    ``rebuild`` used to read only the cached raw archives, so a dataset whose
    zips had been pruned was silently reduced to whatever was passed in.
    """
    ingestor = BinanceArchiveIngestor(parquet_store)
    ingestor.rebuild("BTC/USDT", "1h", extra_frames=[make_bars(24)])

    tail = make_bars(12, start_ts=to_ms("2023-01-02T00:00:00Z"), seed=9)
    result = ingestor.rebuild("BTC/USDT", "1h", extra_frames=[tail])

    assert result.n_bars == 36
    assert format_ts(result.start_ts) == "2023-01-01T00:00:00Z"
    assert format_ts(result.end_ts) == "2023-01-02T11:00:00Z"


def test_rebuilding_with_nothing_new_is_a_no_op(parquet_store: ParquetBarStore) -> None:
    ingestor = BinanceArchiveIngestor(parquet_store)
    first = ingestor.rebuild(
        "BTC/USDT", "1h", extra_frames=[make_bars(24)], built_at="2026-01-01T00:00:00Z"
    )
    second = ingestor.rebuild("BTC/USDT", "1h", built_at="2026-01-01T00:00:00Z")
    assert second.dataset_id == first.dataset_id
    assert second.n_bars == 24


def test_include_stored_false_rebuilds_from_raw_sources_only(
    parquet_store: ParquetBarStore,
) -> None:
    ingestor = BinanceArchiveIngestor(parquet_store)
    ingestor.rebuild("BTC/USDT", "1h", extra_frames=[make_bars(24)])
    result = ingestor.rebuild("BTC/USDT", "1h", extra_frames=[make_bars(6)], include_stored=False)
    assert result.n_bars == 6


def test_a_real_bar_supersedes_a_gap_fill(parquet_store: ParquetBarStore) -> None:
    """A placeholder is not an observation: when the real bar arrives, it wins."""
    ingestor = BinanceArchiveIngestor(parquet_store)
    complete = make_bars(24)
    ingestor.rebuild("BTC/USDT", "1h", extra_frames=[drop_bars(complete, [10, 11])])
    assert int(parquet_store.load("BTC/USDT", "1h", 0, 2**42).is_gap_filled.sum()) == 2

    result = ingestor.rebuild("BTC/USDT", "1h", extra_frames=[complete.iloc[10:12]])
    assert result.report.n_gap_fills_superseded == 2
    assert result.n_bars == 24

    loaded = parquet_store.load("BTC/USDT", "1h", 0, 2**42)
    assert int(loaded.is_gap_filled.sum()) == 0
    assert float(loaded.close[10]) == pytest.approx(float(complete.loc[10, "close"]))
    assert "superseded fills: 2" in result.report.summary()


def test_conflicting_real_bars_still_raise(parquet_store: ParquetBarStore) -> None:
    """Superseding applies only to synthetic bars, never to two real observations."""
    ingestor = BinanceArchiveIngestor(parquet_store)
    bars = make_bars(24)
    ingestor.rebuild("BTC/USDT", "1h", extra_frames=[bars])

    altered = bars.iloc[[5]].copy()
    altered.loc[:, "close"] = float(altered["close"].iloc[0]) + 10.0
    with pytest.raises(DataValidationError, match="different data"):
        ingestor.rebuild("BTC/USDT", "1h", extra_frames=[altered])
