"""Data layer to backtest, end to end.

Phase 2 guarantees the backtester cannot be *handed* future data; phase 3
guarantees it never *asks* for any. This test runs the two together: bars are
ingested, stored, hash-verified, loaded through the partition guard, and then
backtested — and the guard still refuses the held-out period even though the
request now comes from the engine's caller rather than from a CLI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from tests.helpers import ScriptedStrategy, make_bars

from quantlab.adapters.data.binance_archive import BinanceArchiveIngestor, ParquetBarStore
from quantlab.adapters.data.guard import PartitionGuard
from quantlab.adapters.engine import SimpleBarEngine
from quantlab.core.errors import LockboxViolation
from quantlab.core.metrics import compute_metrics
from quantlab.core.splits import parse_split_policy
from quantlab.core.types import BacktestConfig, RiskSpec, SlippageConfig, to_ms

POLICY = {
    "symbol": "BTC/USDT",
    "timeframe": "1h",
    "train": {"start": "2023-01-01T00:00:00Z", "end": "2023-01-10T23:00:00Z"},
    "embargo_bars": 24,
    "validation": {"start": "2023-01-12T00:00:00Z", "end": "2023-01-15T23:00:00Z"},
    "test": {"start": "2023-01-17T00:00:00Z", "end": None},
}


@pytest.fixture
def store(tmp_path: Path) -> ParquetBarStore:
    store = ParquetBarStore(tmp_path / "data")
    BinanceArchiveIngestor(store).rebuild(
        "BTC/USDT", "1h", extra_frames=[make_bars(24 * 25, seed=17, drift=0.02)]
    )
    return store


@pytest.fixture
def config() -> BacktestConfig:
    return BacktestConfig(
        initial_equity=10_000.0,
        fee_bps=10.0,
        slippage=SlippageConfig(model="fixed_bps", fixed_bps=5.0),
        lot_step=0.00001,
        min_notional=5.0,
        bars_per_year=8_760,
    )


def test_a_backtest_runs_on_stored_data(store: ParquetBarStore, config: BacktestConfig) -> None:
    policy = parse_split_policy(POLICY, dataset_id=store.dataset_id("BTC/USDT", "1h"))
    guarded = PartitionGuard(store, policy)

    bars = guarded.load("BTC/USDT", "1h", policy.train_start_ts, policy.train_end_ts)
    assert bars.n_bars == 10 * 24
    assert bars.dataset_id == store.dataset_id("BTC/USDT", "1h")

    result = SimpleBarEngine().run(
        ScriptedStrategy(["long", "long", "flat"] * 100), bars, {}, config
    )
    assert result.n_bars == bars.n_bars
    assert list(result.equity.index) == list(bars.ts_open)

    metrics = compute_metrics(result)
    assert metrics.n_trades > 0
    assert metrics.max_drawdown is not None
    assert np.isfinite(result.equity.to_numpy()).all()


def test_the_engine_cannot_reach_the_test_partition(
    store: ParquetBarStore, config: BacktestConfig
) -> None:
    """A caller that asks for everything is refused, not quietly clipped."""
    policy = parse_split_policy(POLICY, dataset_id=store.dataset_id("BTC/USDT", "1h"))
    guarded = PartitionGuard(store, policy)
    with pytest.raises(LockboxViolation):
        guarded.load("BTC/USDT", "1h", policy.train_start_ts, to_ms("2023-02-01T00:00:00Z"))


def test_train_and_validation_runs_are_independent(
    store: ParquetBarStore, config: BacktestConfig
) -> None:
    """A run on train must be identical whether or not validation was also loaded."""
    policy = parse_split_policy(POLICY, dataset_id=store.dataset_id("BTC/USDT", "1h"))
    guarded = PartitionGuard(store, policy)
    engine = SimpleBarEngine()
    script = ["long", "long", "flat"] * 100

    train = guarded.load("BTC/USDT", "1h", policy.train_start_ts, policy.train_end_ts)
    both = guarded.load("BTC/USDT", "1h", policy.train_start_ts, policy.val_end_ts)

    train_only = engine.run(ScriptedStrategy(script), train, {}, config)
    over_both = engine.run(ScriptedStrategy(script), both, {}, config)

    shared = train.n_bars - 1  # the last bar of the shorter run force-closes
    assert list(train_only.signals)[:shared] == list(over_both.signals)[:shared]
    assert list(train_only.equity)[:shared] == list(over_both.equity)[:shared]


def test_gap_filled_bars_survive_into_the_backtest(tmp_path: Path, config: BacktestConfig) -> None:
    """A synthetic bar must reach the engine flagged, so no order fills on it."""
    from tests.helpers import drop_bars

    store = ParquetBarStore(tmp_path / "data")
    bars = make_bars(240, seed=5)
    BinanceArchiveIngestor(store).rebuild(
        "BTC/USDT", "1h", extra_frames=[drop_bars(bars, [100, 101])]
    )
    loaded = store.load("BTC/USDT", "1h", 0, 2**42)
    assert int(loaded.is_gap_filled.sum()) == 2

    result = SimpleBarEngine().run(ScriptedStrategy(["long", "flat"] * 200), loaded, {}, config)
    filled_bars = set(np.flatnonzero(loaded.is_gap_filled).tolist())
    assert not {fill.bar_index for fill in result.fills} & filled_bars


def test_a_full_run_with_risk_controls_reconciles(
    store: ParquetBarStore, config: BacktestConfig
) -> None:
    policy = parse_split_policy(POLICY, dataset_id=store.dataset_id("BTC/USDT", "1h"))
    bars = PartitionGuard(store, policy).load(
        "BTC/USDT", "1h", policy.train_start_ts, policy.val_end_ts
    )
    with_risk = config.model_copy(
        update={"risk": RiskSpec(stop_loss_pct=0.02, take_profit_pct=0.05, trailing_stop_pct=0.03)}
    )
    result = SimpleBarEngine().run(ScriptedStrategy(["long"] * bars.n_bars), bars, {}, with_risk)

    assert any(trade.exit_reason == "stop" for trade in result.trades)
    change = float(result.equity.iloc[-1] - result.equity.iloc[0])
    assert change == pytest.approx(sum(t.pnl for t in result.trades), abs=1e-6)


def test_the_whole_pipeline_is_reproducible(store: ParquetBarStore, config: BacktestConfig) -> None:
    """INV-7 at the level a run is actually stored: same inputs, same numbers."""
    policy = parse_split_policy(POLICY, dataset_id=store.dataset_id("BTC/USDT", "1h"))
    guarded = PartitionGuard(store, policy)
    script = ["long", "long", "flat"] * 100

    outputs = []
    for _ in range(2):
        bars = guarded.load("BTC/USDT", "1h", policy.train_start_ts, policy.train_end_ts)
        result = SimpleBarEngine().run(ScriptedStrategy(script), bars, {}, config)
        outputs.append((list(result.equity), result.trades, compute_metrics(result).as_dict()))

    assert outputs[0] == outputs[1]


def test_the_backtest_config_hashes_stably(config: BacktestConfig) -> None:
    """The run identity of spec section 11.2 depends on this being deterministic."""
    assert config.config_hash() == config.config_hash()
    changed = config.model_copy(update={"fee_bps": 11.0})
    assert changed.config_hash() != config.config_hash()
