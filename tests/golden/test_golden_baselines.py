"""Golden regression tests (master spec T15, section 19).

Each baseline is run over ``tests/fixtures/data/btcusdt_1h_2023-01_02.parquet`` --
1 416 **real** BTC/USDT hourly bars from January and February 2023, downloaded
from the Binance archive and verified against its published SHA-256 -- and its
trades and metrics are compared against recorded output.

These tests exist to fail. Any change to a fill rule, a cost, a risk exit or a
metric definition changes these numbers, and the failure is the point: it forces
the change to be deliberate, explained, and accompanied by an ``engine_version``
bump. A golden test that is routinely regenerated to make it pass is worse than
no golden test, because it looks like coverage.

Regenerate deliberately, and only with a version bump::

    uv run python scripts/make_fixtures.py golden
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from scripts.make_fixtures import (
    GOLDEN_CASES,
    GOLDEN_CONFIG,
    metrics_payload,
    trades_frame,
)

from quantlab.adapters.engine import ENGINE_VERSION, SimpleBarEngine
from quantlab.core.hashing import canonical_json, file_sha256
from quantlab.core.types import BarFrame

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "data" / "btcusdt_1h_2023-01_02.parquet"
GOLDEN_DIR = REPO_ROOT / "tests" / "fixtures" / "golden"

#: The exact bytes every golden was recorded against.
FIXTURE_SHA256 = "fa56a544a12756ae64d79e88b6a31aab6ee475a04fe8718648e84a9e7629383c"

pytestmark = pytest.mark.skipif(
    not DATA_FIXTURE.is_file(),
    reason="the committed data fixture is missing; run scripts/make_fixtures.py data",
)

CASE_IDS = [case[0] for case in GOLDEN_CASES]


@pytest.fixture(scope="module")
def bars() -> BarFrame:
    return BarFrame(
        pd.read_parquet(DATA_FIXTURE), symbol="BTC/USDT", timeframe="1h", dataset_id="fixture"
    )


def run_case(bars: BarFrame, module: str, params: dict, risk):
    import importlib

    strategy = importlib.import_module(module).STRATEGY()
    config = GOLDEN_CONFIG.model_copy(update={"risk": risk})
    return SimpleBarEngine().run(strategy, bars, params, config), config


# ---------------------------------------------------------------------------
# the fixture itself
# ---------------------------------------------------------------------------
def test_the_fixture_is_the_expected_real_data(bars: BarFrame) -> None:
    """Real market data, not a synthetic stand-in: the numbers are checkable."""
    from quantlab.core.types import format_ts

    assert bars.n_bars == 1_416  # 31 + 28 days of hourly bars
    assert format_ts(bars.start_ts) == "2023-01-01T00:00:00Z"
    assert format_ts(bars.end_ts) == "2023-02-28T23:00:00Z"
    assert float(bars.open[0]) == pytest.approx(16_541.77)
    assert float(bars.close[-1]) == pytest.approx(23_141.57)
    assert int(bars.is_gap_filled.sum()) == 0


def test_the_fixture_passes_validation(bars: BarFrame) -> None:
    from quantlab.core.data_validation import validate_bars

    report = validate_bars(bars.to_pandas(), "1h")
    assert report.n_bars == 1_416
    assert report.ok


def test_the_fixture_file_is_stable() -> None:
    """A changed byte here changes every golden, so it is pinned."""
    assert file_sha256(DATA_FIXTURE) == FIXTURE_SHA256


# ---------------------------------------------------------------------------
# the goldens
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("name", "module", "params", "risk"), GOLDEN_CASES, ids=CASE_IDS)
def test_trades_match_the_recorded_golden(
    bars: BarFrame, name: str, module: str, params: dict, risk
) -> None:
    expected = pd.read_parquet(GOLDEN_DIR / name / "trades.parquet")
    result, _config = run_case(bars, module, params, risk)
    pd.testing.assert_frame_equal(trades_frame(result.trades), expected, check_exact=True)


@pytest.mark.parametrize(("name", "module", "params", "risk"), GOLDEN_CASES, ids=CASE_IDS)
def test_metrics_match_the_recorded_golden(
    bars: BarFrame, name: str, module: str, params: dict, risk
) -> None:
    """Byte-for-byte against the recorded JSON, at full float precision."""
    recorded = (GOLDEN_DIR / name / "metrics.json").read_text(encoding="utf-8")
    result, config = run_case(bars, module, params, risk)
    payload = metrics_payload(result, config, params, risk)
    assert canonical_json(payload) + "\n" == recorded


@pytest.mark.parametrize(("name", "module", "params", "risk"), GOLDEN_CASES, ids=CASE_IDS)
def test_a_golden_is_reproducible_within_one_session(
    bars: BarFrame, name: str, module: str, params: dict, risk
) -> None:
    """Spec T15: the golden test must pass twice in a row."""
    first, _ = run_case(bars, module, params, risk)
    second, _ = run_case(bars, module, params, risk)
    assert first.trades == second.trades
    assert list(first.equity) == list(second.equity)
    assert first.cost_summary == second.cost_summary


# ---------------------------------------------------------------------------
# what the goldens are guarding
# ---------------------------------------------------------------------------
def test_every_golden_records_the_current_engine_version() -> None:
    """An ``engine_version`` bump must be accompanied by regenerated goldens.

    This is the mechanism that makes a fill-rule change impossible to ship
    silently: bump the version without regenerating, and this fails.
    """
    for name, *_ in GOLDEN_CASES:
        recorded = json.loads((GOLDEN_DIR / name / "metrics.json").read_text(encoding="utf-8"))
        assert recorded["engine_version"] == ENGINE_VERSION, (
            f"golden {name} was recorded under engine_version "
            f"{recorded['engine_version']!r}, but the engine is now {ENGINE_VERSION!r}. "
            "Re-record with `uv run python scripts/make_fixtures.py golden`."
        )


def test_buy_and_hold_tracks_the_market_it_held(bars: BarFrame) -> None:
    """The one golden whose answer is arithmetic, checked against the real move.

    BTC opened 2023 at 16 541.77 and closed February at 23 141.57: +39.9 %.
    Holding it through 10 bps of fees and 5 bps of slippage each way must land
    just below that, and nowhere else.
    """
    result, _ = run_case(bars, "strategies.baselines.buy_and_hold", {}, GOLDEN_CASES[0][3])
    market = float(bars.close[-1]) / float(bars.open[0]) - 1.0
    realised = float(result.equity.iloc[-1]) / float(result.equity.iloc[0]) - 1.0

    assert market == pytest.approx(0.399, abs=0.002)
    assert realised == pytest.approx(0.3958, abs=0.001)
    assert realised < market, "costs must make holding slightly worse than the market"
    assert market - realised < 0.005, "and only slightly: one round trip is ~30 bps"


def test_the_stopped_case_actually_exercises_the_risk_exits(bars: BarFrame) -> None:
    """A golden that never fires a stop would not guard the section 8.6 rules."""
    name, module, params, risk = GOLDEN_CASES[4]
    assert name == "sma_cross_stopped"
    result, _ = run_case(bars, module, params, risk)
    assert "stop" in {trade.exit_reason for trade in result.trades}
    assert len(result.trades) > 10


def test_the_goldens_cover_every_exit_reason() -> None:
    reasons: set[str] = set()
    for name, *_ in GOLDEN_CASES:
        trades = pd.read_parquet(GOLDEN_DIR / name / "trades.parquet")
        if len(trades):
            reasons |= set(trades["exit_reason"].tolist())
    assert {"signal", "end_of_data", "stop"} <= reasons
