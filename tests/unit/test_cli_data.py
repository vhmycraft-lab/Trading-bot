"""``quantlab data`` end to end (master spec section 7.5).

Runs against a temporary working directory with a fake archive; no network.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from tests.helpers import drop_bars, make_bars, make_kline_zip
from typer.testing import CliRunner

from quantlab.adapters.data.binance_archive import (
    ArchiveFile,
    BinanceArchiveIngestor,
    ParquetBarStore,
)
from quantlab.cli import app
from quantlab.core.errors import EXIT_CODES
from quantlab.core.types import to_ms


@pytest.fixture
def project(tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway project directory carrying the real configs."""
    configs = tmp_path / "configs"
    (configs / "splits").mkdir(parents=True)
    (configs / "default.yaml").write_text(
        (repo_root / "configs" / "default.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (configs / "splits" / "btcusdt_1h.yaml").write_text(
        (repo_root / "configs" / "splits" / "btcusdt_1h.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    # Rich wraps to 80 columns when stdout is not a terminal, which would
    # truncate the table cells these tests assert on.
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def ingest(project: Path, frame=None, **kwargs) -> ParquetBarStore:
    store = ParquetBarStore(project / "data")
    BinanceArchiveIngestor(store).rebuild(
        "BTC/USDT",
        "1h",
        extra_frames=[make_bars(48) if frame is None else frame],
        built_at="2026-01-01T00:00:00Z",
        **kwargs,
    )
    return store


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------
def test_info_prints_the_manifest_and_dataset_id(cli: CliRunner, project: Path) -> None:
    store = ingest(project)
    result = cli.invoke(app, ["data", "info"])
    assert result.exit_code == 0, result.output
    assert store.dataset_id("BTC/USDT", "1h") in result.stdout
    assert "year=2023/bars.parquet" in result.stdout
    assert "2023-01-01T00:00:00Z" in result.stdout


def test_info_without_a_dataset_exits_with_the_data_code(cli: CliRunner, project: Path) -> None:
    result = cli.invoke(app, ["data", "info"])
    assert result.exit_code == EXIT_CODES["DataError"] == 3


def test_dataset_id_is_stable_across_invocations(cli: CliRunner, project: Path) -> None:
    ingest(project)
    first = cli.invoke(app, ["data", "info"])
    second = cli.invoke(app, ["data", "info"])
    assert first.stdout == second.stdout


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------
def test_validate_reports_a_clean_dataset(cli: CliRunner, project: Path) -> None:
    ingest(project)
    result = cli.invoke(app, ["data", "validate"])
    assert result.exit_code == 0
    assert "48 bars" in result.stdout
    assert "consistent" in result.stdout


def test_validate_reports_gap_filled_bars(cli: CliRunner, project: Path) -> None:
    ingest(project, drop_bars(make_bars(48), [10, 11]))
    result = cli.invoke(app, ["data", "validate"])
    assert result.exit_code == 0
    assert "gap-filled bars : 2" in result.stdout


# ---------------------------------------------------------------------------
# pull
# ---------------------------------------------------------------------------
def test_pull_downloads_verifies_and_stores(
    cli: CliRunner, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bars = make_bars(72)
    archive = ArchiveFile(symbol="BTC/USDT", timeframe="1h", period="2023-01", kind="monthly")
    payload = make_kline_zip(bars, name="BTCUSDT-1h-2023-01.csv")
    digest = hashlib.sha256(payload).hexdigest()
    payloads = {archive.url: payload, archive.checksum_url: f"{digest}  x.zip\n".encode()}

    monkeypatch.setattr(
        "quantlab.cli.data.httpx_fetcher", lambda **_kwargs: lambda url: payloads[url]
    )

    result = cli.invoke(app, ["data", "pull", "--from", "2023-01", "--to", "2023-01"])
    assert result.exit_code == 0, result.output
    assert "72 bars" in result.stdout
    assert (project / "data" / "binance" / "BTCUSDT" / "1h" / "manifest.json").is_file()
    assert (project / "data" / "raw" / "binance" / "BTCUSDT" / "1h" / archive.name).is_file()


def test_pull_reports_a_checksum_mismatch(
    cli: CliRunner, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = ArchiveFile(symbol="BTC/USDT", timeframe="1h", period="2023-01", kind="monthly")
    payloads = {
        archive.url: make_kline_zip(make_bars(24)),
        archive.checksum_url: b"0" * 64 + b"  x.zip\n",
    }
    monkeypatch.setattr(
        "quantlab.cli.data.httpx_fetcher", lambda **_kwargs: lambda url: payloads[url]
    )
    result = cli.invoke(app, ["data", "pull", "--from", "2023-01", "--to", "2023-01"])
    assert result.exit_code == EXIT_CODES["DataError"]


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------
def test_splits_prints_segments_and_window_count(cli: CliRunner, project: Path) -> None:
    """Spec T09 acceptance criterion, through the CLI."""
    result = cli.invoke(app, ["data", "splits"])
    assert result.exit_code == 0, result.output
    assert "train" in result.stdout
    assert "47112 bars" in result.stdout
    assert "23 rolling window(s)" in result.stdout


def test_splits_can_list_the_windows(cli: CliRunner, project: Path) -> None:
    result = cli.invoke(app, ["data", "splits", "--windows"])
    assert result.exit_code == 0
    assert result.stdout.count("OOS") == 23


def test_splits_binds_the_dataset_id_when_one_exists(cli: CliRunner, project: Path) -> None:
    store = ingest(project)
    result = cli.invoke(app, ["data", "splits"])
    assert store.dataset_id("BTC/USDT", "1h") in result.stdout


# ---------------------------------------------------------------------------
# range: what the backtester actually receives
# ---------------------------------------------------------------------------
def test_range_loads_through_the_guard(cli: CliRunner, project: Path) -> None:
    ingest(project)
    result = cli.invoke(
        app,
        ["data", "range", "--from", "2023-01-01T00:00:00Z", "--to", "2023-01-01T05:00:00Z"],
    )
    assert result.exit_code == 0, result.output
    assert "n_bars=6" in result.stdout
    assert "always UTC" in result.stdout


def test_range_refuses_test_partition_bars(cli: CliRunner, project: Path) -> None:
    """The guard is on by default, and it exits with the lockbox code."""
    ingest(project, make_bars(24, start_ts=to_ms("2025-03-01T00:00:00Z")))
    result = cli.invoke(
        app,
        ["data", "range", "--from", "2025-03-01T00:00:00Z", "--to", "2025-03-01T05:00:00Z"],
    )
    assert result.exit_code == EXIT_CODES["LockboxViolation"] == 7


def test_range_unguarded_is_an_explicit_choice(cli: CliRunner, project: Path) -> None:
    ingest(project, make_bars(24, start_ts=to_ms("2025-03-01T00:00:00Z")))
    result = cli.invoke(
        app,
        [
            "data",
            "range",
            "--unguarded",
            "--from",
            "2025-03-01T00:00:00Z",
            "--to",
            "2025-03-01T05:00:00Z",
        ],
    )
    assert result.exit_code == 0
    assert "n_bars=6" in result.stdout


def test_range_without_a_dataset_exits_with_the_data_code(cli: CliRunner, project: Path) -> None:
    assert cli.invoke(app, ["data", "range"]).exit_code == EXIT_CODES["DataError"]


def test_range_rejects_a_policy_for_another_market(cli: CliRunner, project: Path) -> None:
    ingest(project)
    result = cli.invoke(app, ["--set", "market.symbol=ETH/USDT", "data", "range"])
    assert result.exit_code in {EXIT_CODES["ConfigError"], EXIT_CODES["DataError"]}


def test_data_help_lists_every_command(cli: CliRunner) -> None:
    result = cli.invoke(app, ["data", "--help"])
    assert result.exit_code == 0
    for command in ("pull", "update", "validate", "info", "splits", "range"):
        assert command in result.stdout


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------
def test_update_extends_the_dataset_from_the_rest_api(
    cli: CliRunner, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ingest(project, make_bars(24))
    tail = make_bars(12, start_ts=to_ms("2023-01-02T00:00:00Z"), seed=9)

    class FakeBinance:
        def __init__(self, *_args, **_kwargs) -> None: ...

        def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):  # type: ignore[no-untyped-def]
            from tests.helpers import ohlcv_rows

            start = 0 if since is None else int(since)
            return [row for row in ohlcv_rows(tail) if row[0] >= start][: limit or 1000]

    monkeypatch.setattr("ccxt.binance", FakeBinance)
    result = cli.invoke(app, ["data", "update"])
    assert result.exit_code == 0, result.output
    assert "12 new closed bar(s)" in result.stdout
    assert "36 bars" in result.stdout

    store = ParquetBarStore(project / "data")
    assert store.available_range("BTC/USDT", "1h")[1] == to_ms("2023-01-02T11:00:00Z")


def test_update_without_a_dataset_exits_with_the_data_code(cli: CliRunner, project: Path) -> None:
    assert cli.invoke(app, ["data", "update"]).exit_code == EXIT_CODES["DataError"]
