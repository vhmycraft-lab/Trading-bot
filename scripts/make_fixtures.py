#!/usr/bin/env python3
"""Build the committed test fixtures (master spec T15).

Two stages, deliberately separable:

``data``
    Downloads the real Binance monthly archives for 2023-01 and 2023-02,
    verifies each against its published SHA-256, normalises them through the
    ordinary ingestion path, and writes
    ``tests/fixtures/data/btcusdt_1h_2023-01_02.parquet``.
    **Requires network access.** This is the only stage that does.

``golden``
    Runs each baseline strategy over that committed fixture with a fixed
    configuration and writes ``trades.parquet`` and ``metrics.json`` per
    strategy under ``tests/fixtures/golden/``.
    **Offline**: it reads the committed Parquet file, so anyone can regenerate
    the goldens without touching the network.

Usage::

    uv run python scripts/make_fixtures.py all       # both stages
    uv run python scripts/make_fixtures.py data      # network
    uv run python scripts/make_fixtures.py golden    # offline

Regenerating goldens is a deliberate act. They exist to fail when a fill or
accounting rule changes, so a diff to ``tests/fixtures/golden/`` must always be
accompanied by an ``engine_version`` bump and an explanation.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
# ``src`` for the package, the repo root for ``strategies``.
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from quantlab.adapters.data.binance_archive import (  # noqa: E402
    BinanceArchiveIngestor,
    ParquetBarStore,
    httpx_fetcher,
)
from quantlab.adapters.engine import SimpleBarEngine  # noqa: E402
from quantlab.core.hashing import canonical_json  # noqa: E402
from quantlab.core.metrics import compute_metrics  # noqa: E402
from quantlab.core.types import (  # noqa: E402
    BacktestConfig,
    BarFrame,
    RiskSpec,
    SlippageConfig,
)

SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
MONTHS = ("2023-01", "2023-02")

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"
DATA_FIXTURE = FIXTURE_DIR / "data" / "btcusdt_1h_2023-01_02.parquet"
GOLDEN_DIR = FIXTURE_DIR / "golden"

#: Fixed Parquet options, so a rebuild is byte-identical.
PARQUET_KWARGS = {"engine": "pyarrow", "compression": "zstd", "index": False}

#: The configuration the goldens are recorded under. Changing any value here
#: changes every golden, so it is written down rather than assembled ad hoc.
GOLDEN_CONFIG = BacktestConfig(
    initial_equity=10_000.0,
    fee_bps=10.0,
    slippage=SlippageConfig(model="fixed_bps", fixed_bps=5.0),
    lot_step=0.00001,
    min_notional=5.0,
    max_position_fraction=1.0,
    bars_per_year=8_760,
    seed=42,
)

#: (name, module, params, risk) per golden case. The stopped case is included so
#: a change to the section 8.6 exit rules cannot pass unnoticed.
GOLDEN_CASES: tuple[tuple[str, str, dict, RiskSpec], ...] = (
    ("buy_and_hold", "strategies.baselines.buy_and_hold", {}, RiskSpec()),
    ("sma_cross", "strategies.baselines.sma_cross", {"fast": 50, "slow": 200}, RiskSpec()),
    (
        "rsi_reversion",
        "strategies.baselines.rsi_reversion",
        {"n": 14, "oversold": 30.0, "exit_level": 55.0},
        RiskSpec(),
    ),
    (
        "random_entry",
        "strategies.baselines.random_entry",
        {"p_enter": 0.02, "hold_bars": 24},
        RiskSpec(),
    ),
    (
        "sma_cross_stopped",
        "strategies.baselines.sma_cross",
        {"fast": 50, "slow": 200},
        RiskSpec(stop_loss_pct=0.02, take_profit_pct=0.05, trailing_stop_pct=0.03),
    ),
)


def trades_frame(trades) -> pd.DataFrame:
    """Flatten a run's trades into the recorded golden schema."""
    return pd.DataFrame(
        [
            {
                "trade_no": t.trade_no,
                "side": str(t.side),
                "entry_ts": t.entry_ts,
                "entry_px": t.entry_px,
                "exit_ts": t.exit_ts,
                "exit_px": t.exit_px,
                "qty": t.qty,
                "fees": t.fees,
                "slippage_cost": t.slippage_cost,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "bars_held": t.bars_held,
                "exit_reason": t.exit_reason,
            }
            for t in trades
        ]
    )


def metrics_payload(result, config, params: dict, risk: RiskSpec) -> dict:
    """The recorded metrics document for one golden case."""
    return {
        "engine_name": result.engine_name,
        "engine_version": result.engine_version,
        "n_bars": result.n_bars,
        "warmup_bars": result.warmup_bars,
        "params": params,
        "risk": risk.model_dump(mode="json"),
        "config_hash": config.config_hash(),
        "cost_summary": result.cost_summary,
        "ruined": result.ruined,
        "final_equity": float(result.equity.iloc[-1]),
        "metrics": compute_metrics(result).as_dict(),
    }


def build_data_fixture() -> Path:
    """Download, verify and store the real two-month fixture.  Needs network."""
    with tempfile.TemporaryDirectory() as scratch:
        store = ParquetBarStore(Path(scratch) / "data")
        ingestor = BinanceArchiveIngestor(store, fetch=httpx_fetcher(timeout=120.0))
        print(f"downloading {', '.join(MONTHS)} from the Binance archive ...")
        result = ingestor.pull(SYMBOL, TIMEFRAME, months=MONTHS, built_at="2026-09-05T00:00:00Z")
        print(result.summary())
        print(result.report.summary())
        frame = store.read_frame(SYMBOL, TIMEFRAME)

    DATA_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(DATA_FIXTURE, **PARQUET_KWARGS)
    print(
        f"\nwrote {DATA_FIXTURE.relative_to(REPO_ROOT)} "
        f"({len(frame)} bars, {DATA_FIXTURE.stat().st_size} bytes)"
    )
    return DATA_FIXTURE


def load_fixture() -> BarFrame:
    """Load the committed fixture.  Offline."""
    if not DATA_FIXTURE.is_file():
        raise SystemExit(
            f"{DATA_FIXTURE.relative_to(REPO_ROOT)} is missing; run "
            "`python scripts/make_fixtures.py data` first (needs network)"
        )
    return BarFrame(
        pd.read_parquet(DATA_FIXTURE), symbol=SYMBOL, timeframe=TIMEFRAME, dataset_id="fixture"
    )


def build_goldens() -> None:
    """Record each baseline's trades and metrics over the fixture.  Offline."""
    import importlib

    bars = load_fixture()
    engine = SimpleBarEngine()

    for name, module, params, risk in GOLDEN_CASES:
        strategy = importlib.import_module(module).STRATEGY()
        config = GOLDEN_CONFIG.model_copy(update={"risk": risk})
        result = engine.run(strategy, bars, params, config)

        target = GOLDEN_DIR / name
        target.mkdir(parents=True, exist_ok=True)
        trades_frame(result.trades).to_parquet(target / "trades.parquet", **PARQUET_KWARGS)
        payload = metrics_payload(result, config, params, risk)
        (target / "metrics.json").write_text(canonical_json(payload) + "\n", encoding="utf-8")

        print(
            f"  {name:<20} trades={len(result.trades):>4}  "
            f"final_equity={payload['final_equity']:.6f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["data", "golden", "all"], nargs="?", default="all")
    args = parser.parse_args()

    if args.stage in ("data", "all"):
        build_data_fixture()
    if args.stage in ("golden", "all"):
        print("\nrecording goldens:")
        build_goldens()
        print("\nGoldens updated. A diff here MUST come with an engine_version bump.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
