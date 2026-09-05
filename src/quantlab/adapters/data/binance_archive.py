"""Binance bulk-archive ingestion and the local Parquet dataset (spec section 7).

Raw and processed data are kept apart on disk and never mix:

``data/raw/binance/BTCUSDT/1h/``
    the downloaded ``.zip`` and ``.CHECKSUM`` files, exactly as published.
    Read-only inputs: nothing here is ever edited, only added to.

``data/binance/BTCUSDT/1h/year=YYYY/bars.parquet`` (+ ``manifest.json``)
    the processed dataset: canonical schema, validated, gap-flagged, sorted,
    de-duplicated, and content-hashed into a ``dataset_id``.

Every archive file's SHA-256 is checked against its published ``.CHECKSUM``
before it is parsed, and every Parquet file's SHA-256 is checked against the
manifest before it is loaded, so a corrupted or edited byte fails the run
instead of quietly changing a backtest (spec section 21.6).
"""

from __future__ import annotations

import datetime as dt
import io
import json
import zipfile
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from quantlab.core.data_validation import ValidationReport, normalise_bars
from quantlab.core.errors import DataError, DataValidationError, ManifestMismatchError
from quantlab.core.hashing import canonical_json, file_sha256
from quantlab.core.hashing import dataset_id as compute_dataset_id
from quantlab.core.types import (
    BAR_COLUMNS,
    BarFrame,
    coerce_bar_frame_df,
    empty_bar_frame_df,
    format_ts,
    from_ms,
    normalise_epoch,
    timeframe_ms,
)

__all__ = [
    "ARCHIVE_BASE_URL",
    "MANIFEST_NAME",
    "ArchiveFile",
    "BinanceArchiveIngestor",
    "Manifest",
    "ManifestEntry",
    "ParquetBarStore",
    "kline_columns",
    "manifest_path_for",
    "months_between",
    "parse_kline_bytes",
    "read_checksum_file",
    "verify_checksum",
]

ARCHIVE_BASE_URL: Final[str] = "https://data.binance.vision/data/spot"
MANIFEST_NAME: Final[str] = "manifest.json"

#: Column order of a Binance monthly/daily kline CSV.
_KLINE_COLUMNS: Final[tuple[str, ...]] = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)

#: Fixed Parquet options so a rebuild produces byte-identical files.
_PARQUET_KWARGS: Final[dict[str, Any]] = {
    "engine": "pyarrow",
    "compression": "zstd",
    "index": False,
}


def kline_columns() -> tuple[str, ...]:
    """Return the Binance kline CSV column names, in file order."""
    return _KLINE_COLUMNS


def exchange_symbol(symbol: str) -> str:
    """``BTC/USDT`` -> ``BTCUSDT`` (the form the archive paths use)."""
    return symbol.replace("/", "").replace("-", "").upper()


def months_between(start: str, end: str) -> tuple[str, ...]:
    """Return the ``YYYY-MM`` months from ``start`` to ``end``, inclusive."""
    first = dt.date.fromisoformat(f"{start}-01")
    last = dt.date.fromisoformat(f"{end}-01")
    if last < first:
        raise DataError("the end month precedes the start month", start=start, end=end)
    months: list[str] = []
    year, month = first.year, first.month
    while (year, month) <= (last.year, last.month):
        months.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return tuple(months)


@dataclass(frozen=True, slots=True)
class ArchiveFile:
    """One published archive file and where it lives locally."""

    symbol: str
    timeframe: str
    period: str  # YYYY-MM for monthly, YYYY-MM-DD for daily
    kind: str  # "monthly" | "daily"

    @property
    def name(self) -> str:
        return f"{exchange_symbol(self.symbol)}-{self.timeframe}-{self.period}.zip"

    @property
    def url(self) -> str:
        market = exchange_symbol(self.symbol)
        return f"{ARCHIVE_BASE_URL}/{self.kind}/klines/{market}/{self.timeframe}/{self.name}"

    @property
    def checksum_url(self) -> str:
        return f"{self.url}.CHECKSUM"


def read_checksum_file(text: str) -> str:
    """Extract the SHA-256 digest from a Binance ``.CHECKSUM`` file.

    Raises:
        DataError: if the file does not contain a 64-character hex digest.
    """
    parts = text.split()
    if not parts or len(parts[0]) != 64:
        raise DataError("checksum file is malformed", content=text[:80])
    digest = parts[0].lower()
    if any(character not in "0123456789abcdef" for character in digest):
        raise DataError("checksum is not hexadecimal", content=text[:80])
    return digest


def verify_checksum(path: Path, expected: str) -> None:
    """Abort unless ``path`` hashes to ``expected`` (spec section 7.1).

    Raises:
        ManifestMismatchError: on any mismatch.
    """
    actual = file_sha256(path)
    if actual != expected.lower():
        raise ManifestMismatchError(
            "archive checksum mismatch; refusing to use the file",
            path=str(path),
            expected=expected,
            actual=actual,
        )


def parse_kline_bytes(payload: bytes, *, source: str = "<memory>") -> pd.DataFrame:
    """Parse a Binance kline zip into the canonical bar schema.

    Handles both archive generations transparently: files with and without a
    header row, and open times in milliseconds (pre-2025) or microseconds.

    Raises:
        DataValidationError: if the archive is not a readable kline CSV.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(names) != 1:
                raise DataValidationError(
                    "archive must contain exactly one CSV", source=source, entries=names
                )
            raw = archive.read(names[0])
    except zipfile.BadZipFile as exc:
        raise DataValidationError("archive is not a valid zip file", source=source) from exc

    if not raw.strip():
        raise DataValidationError("archive CSV is empty", source=source)

    first_field = raw.split(b",", 1)[0].strip().strip(b'"')
    has_header = not first_field.isdigit()

    frame = pd.read_csv(
        io.BytesIO(raw),
        header=0 if has_header else None,
        names=None if has_header else list(_KLINE_COLUMNS),
    )
    if has_header:
        frame.columns = [str(name).strip().lower() for name in frame.columns]

    missing = [name for name in ("open_time", "open", "high", "low", "close") if name not in frame]
    if missing:
        raise DataValidationError("archive CSV is missing columns", source=source, missing=missing)

    out = pd.DataFrame(
        {
            "ts_open": [normalise_epoch(int(value)) for value in frame["open_time"]],
            "open": frame["open"].astype("float64"),
            "high": frame["high"].astype("float64"),
            "low": frame["low"].astype("float64"),
            "close": frame["close"].astype("float64"),
            "volume": frame["volume"].astype("float64"),
            "quote_volume": frame.get("quote_volume", pd.Series(np.zeros(len(frame)))).astype(
                "float64"
            ),
            "trades": frame.get("trades", pd.Series(np.zeros(len(frame)))).astype("int64"),
        }
    )
    out["is_gap_filled"] = False
    return coerce_bar_frame_df(out)


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One Parquet file of the processed dataset."""

    path: str
    sha256: str
    n_bars: int
    first_ts: int
    last_ts: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "n_bars": self.n_bars,
            "first_ts": self.first_ts,
            "last_ts": self.last_ts,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ManifestEntry:
        return cls(
            path=str(payload["path"]),
            sha256=str(payload["sha256"]),
            n_bars=int(payload["n_bars"]),
            first_ts=int(payload["first_ts"]),
            last_ts=int(payload["last_ts"]),
        )


@dataclass(frozen=True, slots=True)
class Manifest:
    """The processed dataset's index and identity (spec section 7.2)."""

    exchange: str
    symbol: str
    timeframe: str
    files: tuple[ManifestEntry, ...]
    built_at: str
    source_versions: dict[str, Any]

    @property
    def dataset_id(self) -> str:
        """Content hash of the whole dataset (spec section 7.2)."""
        return compute_dataset_id(
            ((entry.path, entry.sha256) for entry in self.files), self.symbol, self.timeframe
        )

    @property
    def n_bars(self) -> int:
        return sum(entry.n_bars for entry in self.files)

    @property
    def start_ts(self) -> int | None:
        return min((entry.first_ts for entry in self.files), default=None)

    @property
    def end_ts(self) -> int | None:
        return max((entry.last_ts for entry in self.files), default=None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "files": [entry.to_dict() for entry in sorted(self.files, key=lambda e: e.path)],
            "dataset_id": self.dataset_id,
            "built_at": self.built_at,
            "source_versions": self.source_versions,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Manifest:
        return cls(
            exchange=str(payload["exchange"]),
            symbol=str(payload["symbol"]),
            timeframe=str(payload["timeframe"]),
            files=tuple(ManifestEntry.from_dict(item) for item in payload["files"]),
            built_at=str(payload.get("built_at", "")),
            source_versions=dict(payload.get("source_versions", {})),
        )


def dataset_dir_for(data_dir: Path, exchange: str, symbol: str, timeframe: str) -> Path:
    """Processed dataset directory: ``<data>/<exchange>/<SYMBOL>/<tf>/``."""
    return Path(data_dir) / exchange / exchange_symbol(symbol) / timeframe


def raw_dir_for(data_dir: Path, exchange: str, symbol: str, timeframe: str) -> Path:
    """Raw download directory: ``<data>/raw/<exchange>/<SYMBOL>/<tf>/``."""
    return Path(data_dir) / "raw" / exchange / exchange_symbol(symbol) / timeframe


def manifest_path_for(data_dir: Path, exchange: str, symbol: str, timeframe: str) -> Path:
    """Path of the processed dataset's manifest."""
    return dataset_dir_for(data_dir, exchange, symbol, timeframe) / MANIFEST_NAME


# ---------------------------------------------------------------------------
# the processed store
# ---------------------------------------------------------------------------
class ParquetBarStore:
    """Reads and writes the processed year-partitioned Parquet dataset.

    Implements :class:`~quantlab.ports.data.MarketDataSource`.  Loading reads
    only the year files that overlap the requested range and only the canonical
    columns, so a two-month backtest does not pay for eight years of history.
    """

    def __init__(self, data_dir: str | Path, *, exchange: str = "binance") -> None:
        self.data_dir = Path(data_dir)
        self.exchange = exchange

    # -- layout ------------------------------------------------------------
    def dataset_dir(self, symbol: str, timeframe: str) -> Path:
        return dataset_dir_for(self.data_dir, self.exchange, symbol, timeframe)

    def raw_dir(self, symbol: str, timeframe: str) -> Path:
        return raw_dir_for(self.data_dir, self.exchange, symbol, timeframe)

    def manifest_path(self, symbol: str, timeframe: str) -> Path:
        return self.dataset_dir(symbol, timeframe) / MANIFEST_NAME

    # -- manifest ----------------------------------------------------------
    def read_manifest(self, symbol: str, timeframe: str) -> Manifest | None:
        """Return the stored manifest, or ``None`` if the dataset does not exist."""
        path = self.manifest_path(symbol, timeframe)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ManifestMismatchError("manifest is not valid JSON", path=str(path)) from exc
        return Manifest.from_dict(payload)

    def require_manifest(self, symbol: str, timeframe: str) -> Manifest:
        """Return the manifest or explain how to create one.

        Raises:
            DataError: if no dataset has been ingested yet.
        """
        manifest = self.read_manifest(symbol, timeframe)
        if manifest is None:
            raise DataError(
                "no dataset found; run `quantlab data pull` first",
                symbol=symbol,
                timeframe=timeframe,
                expected=str(self.manifest_path(symbol, timeframe)),
            )
        return manifest

    def write_manifest(self, manifest: Manifest) -> Path:
        """Write the manifest as canonical JSON so rebuilds compare byte for byte."""
        path = self.manifest_path(manifest.symbol, manifest.timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(canonical_json(manifest.to_dict()), encoding="utf-8")
        return path

    # -- writing -----------------------------------------------------------
    def write_years(
        self, symbol: str, timeframe: str, frame: pd.DataFrame
    ) -> tuple[ManifestEntry, ...]:
        """Split ``frame`` by calendar year and write one Parquet file per year."""
        canonical = coerce_bar_frame_df(frame)
        root = self.dataset_dir(symbol, timeframe)
        root.mkdir(parents=True, exist_ok=True)

        entries: list[ManifestEntry] = []
        years = np.array([from_ms(int(ts)).year for ts in canonical["ts_open"]], dtype="int64")
        for year in sorted({int(value) for value in years}):
            chunk = canonical.loc[years == year].reset_index(drop=True)
            relative = f"year={year}/bars.parquet"
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            chunk.to_parquet(target, **_PARQUET_KWARGS)
            entries.append(
                ManifestEntry(
                    path=relative,
                    sha256=file_sha256(target),
                    n_bars=len(chunk),
                    first_ts=int(chunk["ts_open"].iloc[0]),
                    last_ts=int(chunk["ts_open"].iloc[-1]),
                )
            )
        return tuple(entries)

    # -- reading -----------------------------------------------------------
    def _entries_overlapping(
        self, manifest: Manifest, start_ts: int | None, end_ts: int | None
    ) -> Iterator[ManifestEntry]:
        for entry in sorted(manifest.files, key=lambda e: e.first_ts):
            if start_ts is not None and entry.last_ts < start_ts:
                continue
            if end_ts is not None and entry.first_ts > end_ts:
                continue
            yield entry

    def read_frame(
        self,
        symbol: str,
        timeframe: str,
        start_ts: int | None = None,
        end_ts: int | None = None,
        *,
        verify: bool = True,
    ) -> pd.DataFrame:
        """Read the processed bars overlapping ``[start_ts, end_ts]``.

        Raises:
            ManifestMismatchError: if a Parquet file no longer matches its
                manifest hash (spec section 21.6).
        """
        manifest = self.require_manifest(symbol, timeframe)
        root = self.dataset_dir(symbol, timeframe)

        chunks: list[pd.DataFrame] = []
        for entry in self._entries_overlapping(manifest, start_ts, end_ts):
            path = root / entry.path
            if not path.is_file():
                raise ManifestMismatchError(
                    "a file listed in the manifest is missing", path=str(path)
                )
            if verify:
                actual = file_sha256(path)
                if actual != entry.sha256:
                    raise ManifestMismatchError(
                        "dataset file does not match its manifest hash",
                        path=str(path),
                        expected=entry.sha256,
                        actual=actual,
                    )
            chunks.append(pd.read_parquet(path, columns=list(BAR_COLUMNS)))

        if not chunks:
            return empty_bar_frame_df()
        frame = pd.concat(chunks, ignore_index=True)
        mask = np.ones(len(frame), dtype=bool)
        ts = frame["ts_open"].to_numpy()
        if start_ts is not None:
            mask &= ts >= int(start_ts)
        if end_ts is not None:
            mask &= ts <= int(end_ts)
        return coerce_bar_frame_df(frame.loc[mask].reset_index(drop=True))

    # -- MarketDataSource --------------------------------------------------
    def load(self, symbol: str, timeframe: str, start_ts: int, end_ts: int) -> BarFrame:
        """Return the validated bars in ``[start_ts, end_ts]`` as a :class:`BarFrame`."""
        frame = self.read_frame(symbol, timeframe, start_ts, end_ts)
        return BarFrame(
            frame,
            symbol=symbol,
            timeframe=timeframe,
            dataset_id=self.dataset_id(symbol, timeframe),
        )

    def dataset_id(self, symbol: str, timeframe: str) -> str:
        return self.require_manifest(symbol, timeframe).dataset_id

    def available_range(self, symbol: str, timeframe: str) -> tuple[int, int] | None:
        manifest = self.read_manifest(symbol, timeframe)
        if manifest is None or not manifest.files:
            return None
        start, end = manifest.start_ts, manifest.end_ts
        return None if start is None or end is None else (start, end)

    def validate(self, symbol: str, timeframe: str) -> ValidationReport:
        """Re-validate the whole stored dataset and return its report."""
        frame = self.read_frame(symbol, timeframe)
        _validated, report = normalise_bars(frame, timeframe, symbol=symbol, allow_gaps=True)
        return report


# ---------------------------------------------------------------------------
# ingestion
# ---------------------------------------------------------------------------
#: Fetches one URL and returns its bytes.  Injected so ingestion is testable offline.
Fetcher = Callable[[str], bytes]


def httpx_fetcher(*, timeout: float = 60.0) -> Fetcher:
    """Return a fetcher backed by ``httpx``.  Used by the CLI, never by tests."""

    def fetch(url: str) -> bytes:
        import httpx

        response = httpx.get(url, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
        return bytes(response.content)

    return fetch


@dataclass(frozen=True, slots=True)
class IngestResult:
    """What one ``pull`` or ``update`` did."""

    symbol: str
    timeframe: str
    dataset_id: str
    n_bars: int
    start_ts: int | None
    end_ts: int | None
    files_written: tuple[str, ...]
    archives_used: tuple[str, ...]
    report: ValidationReport

    def summary(self) -> str:
        return (
            f"{self.symbol} {self.timeframe}: {self.n_bars} bars "
            f"{format_ts(self.start_ts)} .. {format_ts(self.end_ts)}\n"
            f"  dataset_id : {self.dataset_id}\n"
            f"  parquet    : {len(self.files_written)} file(s)\n"
            f"  archives   : {len(self.archives_used)} used"
        )


class BinanceArchiveIngestor:
    """Downloads, verifies, normalises and stores Binance bulk archives.

    Ingestion is idempotent by construction: raw archives are cached and their
    checksums verified, the processed frame is rebuilt from the full set of raw
    files every time, and the resulting Parquet files are written with fixed
    options.  Running ``pull`` twice therefore produces identical hashes and an
    identical ``dataset_id``.
    """

    def __init__(
        self,
        store: ParquetBarStore,
        *,
        fetch: Fetcher | None = None,
        exchange: str = "binance",
    ) -> None:
        self.store = store
        self.exchange = exchange
        self._fetch = fetch

    # -- raw layer ---------------------------------------------------------
    def _download(self, url: str) -> bytes:
        if self._fetch is None:
            raise DataError(
                "no fetcher configured; this ingestor can only use cached archives", url=url
            )
        return self._fetch(url)

    def ensure_archive(self, archive: ArchiveFile, *, verify: bool = True) -> Path:
        """Return the local path of ``archive``, downloading it if needed.

        The published ``.CHECKSUM`` is fetched alongside and the payload is
        verified before it is stored, so a truncated download never becomes part
        of the dataset.

        Raises:
            ManifestMismatchError: on a checksum mismatch.
        """
        raw_dir = self.store.raw_dir(archive.symbol, archive.timeframe)
        raw_dir.mkdir(parents=True, exist_ok=True)
        target = raw_dir / archive.name
        checksum_target = raw_dir / f"{archive.name}.CHECKSUM"

        if not target.is_file():
            payload = self._download(archive.url)
            target.write_bytes(payload)
            if verify:
                try:
                    checksum_target.write_bytes(self._download(archive.checksum_url))
                except Exception:
                    target.unlink(missing_ok=True)
                    raise

        if verify and checksum_target.is_file():
            verify_checksum(target, read_checksum_file(checksum_target.read_text("utf-8")))
        return target

    def local_archives(self, symbol: str, timeframe: str) -> tuple[Path, ...]:
        """Every cached raw archive for this symbol and timeframe, sorted by name."""
        raw_dir = self.store.raw_dir(symbol, timeframe)
        if not raw_dir.is_dir():
            return ()
        return tuple(sorted(raw_dir.glob("*.zip")))

    # -- processed layer ---------------------------------------------------
    def rebuild(
        self,
        symbol: str,
        timeframe: str,
        *,
        extra_frames: Sequence[pd.DataFrame] = (),
        allow_gaps: bool = False,
        built_at: str | None = None,
        include_stored: bool = True,
    ) -> IngestResult:
        """Rebuild the processed dataset from every available source.

        The inputs are, in order: the bars already stored, every cached raw
        archive, and ``extra_frames`` (the REST tail).  They go through one
        normalisation pass, so archive bars and tail bars are validated
        identically and cannot disagree without being noticed.

        Including the stored dataset makes rebuilding **additive**: an update
        extends history rather than replacing it.  Without that, a dataset whose
        raw archives were pruned — or that was seeded from anywhere other than a
        cached zip — would be silently truncated to whatever was passed in.
        Pass ``include_stored=False`` only to rebuild from raw sources on purpose.

        Raises:
            DataError: if there is nothing to build from.
        """
        timeframe_ms(timeframe)
        frames: list[pd.DataFrame] = []

        if include_stored and self.store.read_manifest(symbol, timeframe) is not None:
            stored = self.store.read_frame(symbol, timeframe)
            if len(stored):
                frames.append(stored)

        archives = self.local_archives(symbol, timeframe)
        for path in archives:
            frames.append(parse_kline_bytes(path.read_bytes(), source=str(path)))
        frames.extend(coerce_bar_frame_df(frame) for frame in extra_frames)

        if not frames:
            raise DataError(
                "nothing to ingest: no stored dataset, no cached archives, no supplied bars",
                symbol=symbol,
                timeframe=timeframe,
            )

        merged = pd.concat(frames, ignore_index=True)
        normalised, report = normalise_bars(merged, timeframe, symbol=symbol, allow_gaps=allow_gaps)

        entries = self.store.write_years(symbol, timeframe, normalised)
        manifest = Manifest(
            exchange=self.exchange,
            symbol=symbol,
            timeframe=timeframe,
            files=entries,
            built_at=built_at or _utc_now_iso(),
            source_versions={"archives": [path.name for path in archives]},
        )
        self.store.write_manifest(manifest)

        return IngestResult(
            symbol=symbol,
            timeframe=timeframe,
            dataset_id=manifest.dataset_id,
            n_bars=manifest.n_bars,
            start_ts=manifest.start_ts,
            end_ts=manifest.end_ts,
            files_written=tuple(entry.path for entry in entries),
            archives_used=tuple(path.name for path in archives),
            report=report,
        )

    def pull(
        self,
        symbol: str,
        timeframe: str,
        *,
        months: Iterable[str],
        allow_gaps: bool = False,
        verify: bool = True,
        built_at: str | None = None,
    ) -> IngestResult:
        """Fetch the given ``YYYY-MM`` months, then rebuild the dataset."""
        for month in months:
            self.ensure_archive(
                ArchiveFile(symbol=symbol, timeframe=timeframe, period=month, kind="monthly"),
                verify=verify,
            )
        return self.rebuild(symbol, timeframe, allow_gaps=allow_gaps, built_at=built_at)


def _utc_now_iso() -> str:
    """Current UTC time as an ISO string.

    Lives in the adapter layer on purpose: spec section 0.3 forbids reading a
    clock inside ``core/`` and the engine, and a manifest's build time is
    metadata, never an input to a backtest.
    """
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
