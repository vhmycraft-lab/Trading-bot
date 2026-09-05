"""Shared pytest fixtures.

Every fixture here is offline and hermetic: no network, no writes outside
``tmp_path``, and no dependence on the developer's environment variables.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import Engine
from typer.testing import CliRunner

from quantlab.adapters.data.binance_archive import ParquetBarStore
from quantlab.adapters.store.sqlite import create_db_engine, upgrade_to_head
from quantlab.core.config import AppConfig, load_config
from quantlab.core.types import BarFrame

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return REPO_ROOT


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove ambient QUANTLAB__* and secret variables so tests are reproducible."""
    for name in list(os.environ):
        if name.upper().startswith("QUANTLAB__"):
            monkeypatch.delenv(name, raising=False)
    for name in ("GLM_API_KEY", "BINANCE_API_KEY", "BINANCE_API_SECRET"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def default_config_path(repo_root: Path) -> Path:
    """Path to the authoritative ``configs/default.yaml``."""
    return repo_root / "configs" / "default.yaml"


@pytest.fixture
def config(default_config_path: Path) -> AppConfig:
    """The default configuration, loaded from the repository."""
    return load_config(default_path=default_config_path)


@pytest.fixture
def db_engine(tmp_path: Path) -> Iterator[Engine]:
    """A migrated SQLite database in a temporary directory."""
    engine = create_db_engine(tmp_path / "quantlab.db")
    upgrade_to_head(engine)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def empty_engine(tmp_path: Path) -> Iterator[Engine]:
    """An empty, unmigrated SQLite database."""
    engine = create_db_engine(tmp_path / "empty.db")
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def cli() -> CliRunner:
    """Typer CLI runner with stderr captured separately."""
    return CliRunner()


@pytest.fixture
def bars_df() -> pd.DataFrame:
    """24 canonical 1h bars starting 2023-01-01T00:00:00Z."""
    from tests.helpers import make_bars

    return make_bars(24)


@pytest.fixture
def bar_frame(bars_df: pd.DataFrame) -> BarFrame:
    """The same bars wrapped as an immutable :class:`BarFrame`."""
    from quantlab.core.types import BarFrame

    return BarFrame(bars_df, symbol="BTC/USDT", timeframe="1h", dataset_id="test0000")


@pytest.fixture
def parquet_store(tmp_path: Path) -> ParquetBarStore:
    """An empty processed-dataset store rooted in a temporary directory."""
    from quantlab.adapters.data.binance_archive import ParquetBarStore

    return ParquetBarStore(tmp_path / "data", exchange="binance")


@pytest.fixture
def ingested_store(parquet_store: ParquetBarStore, bars_df: pd.DataFrame) -> ParquetBarStore:
    """A store holding one ingested month of bars."""
    from quantlab.adapters.data.binance_archive import BinanceArchiveIngestor

    ingestor = BinanceArchiveIngestor(parquet_store)
    ingestor.rebuild("BTC/USDT", "1h", extra_frames=[bars_df], built_at="2026-01-01T00:00:00Z")
    return parquet_store
