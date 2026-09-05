"""A throwaway project directory that the CLI can actually run in.

The `quantlab` commands resolve their configuration, data, split policy and
database relative to the working directory, so an integration test has to build a
real one. Everything here is offline: bars are generated, not downloaded, and the
split policy is written to match them rather than the repository's real one, which
spans 2017-2025 and would leave every segment empty.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
from tests.helpers import make_bars
from typer.testing import CliRunner

from quantlab.adapters.data.binance_archive import BinanceArchiveIngestor, ParquetBarStore
from quantlab.cli import app
from quantlab.core.types import format_ts

#: Enough bars for a warm-up plus three segments with an embargo between them.
N_BARS: Final[int] = 900
HOUR_MS: Final[int] = 3_600_000
EPOCH_2023: Final[int] = 1_672_531_200_000
EMBARGO_BARS: Final[int] = 24

TRAIN_END: Final[int] = 399
VAL_START: Final[int] = TRAIN_END + EMBARGO_BARS + 1
VAL_END: Final[int] = 699
TEST_START: Final[int] = VAL_END + EMBARGO_BARS + 1


def _iso(index: int) -> str:
    return format_ts(EPOCH_2023 + index * HOUR_MS)


SPLIT_POLICY: Final[str] = f"""
symbol: BTC/USDT
timeframe: 1h
train:      {{ start: "{_iso(0)}", end: "{_iso(TRAIN_END)}" }}
embargo_bars: {EMBARGO_BARS}
validation: {{ start: "{_iso(VAL_START)}", end: "{_iso(VAL_END)}" }}
test:       {{ start: "{_iso(TEST_START)}", end: null }}
"""


@pytest.fixture
def project(tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A project directory with configs, a matching split policy and ingested bars."""
    configs = tmp_path / "configs"
    (configs / "splits").mkdir(parents=True)
    (configs / "default.yaml").write_text(
        (repo_root / "configs" / "default.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (configs / "splits" / "btcusdt_1h.yaml").write_text(SPLIT_POLICY, encoding="utf-8")

    store = ParquetBarStore(tmp_path / "data")
    BinanceArchiveIngestor(store).rebuild(
        "BTC/USDT",
        "1h",
        extra_frames=[make_bars(N_BARS, drift=0.03)],
        built_at="2026-01-01T00:00:00Z",
    )

    # Rich wraps to 80 columns off a terminal, which would truncate the run id
    # these tests read out of the table.
    monkeypatch.setenv("COLUMNS", "220")
    monkeypatch.chdir(tmp_path)

    # The schema is Alembic's (section 6); no command creates tables on the way
    # past, so the database is migrated here exactly as a person would.
    result = CliRunner().invoke(app, ["db", "upgrade"])
    assert result.exit_code == 0, result.output
    return tmp_path


@pytest.fixture
def strategy_file(project: Path, repo_root: Path) -> Path:
    """A fast-warming copy of ``sma_cross``.

    The shipped baseline declares 400 bars of warm-up, which would consume the
    whole training segment and leave the run with nothing to trade on.
    """
    source = (repo_root / "strategies" / "baselines" / "sma_cross.py").read_text(encoding="utf-8")
    source = (
        source.replace("default=50, low=5, high=200", "default=10, low=5, high=200")
        .replace("default=200, low=20, high=400", "default=30, low=20, high=400")
        .replace("warmup_bars = 400", "warmup_bars = 40")
    )
    path = project / "sma_fast.py"
    path.write_text(source, encoding="utf-8")
    return path
