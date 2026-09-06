"""The ports are satisfied by the adapters that claim them (spec section 2.3).

Ports are structural :class:`typing.Protocol` definitions, so nothing forces an
adapter to keep matching one as either side changes.  These tests are that
force: they fail when a port and its adapters drift apart, which is the moment
the swap-an-adapter promise of spec section 2.3 quietly stops being true.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd

from quantlab.adapters.data.binance_archive import ParquetBarStore
from quantlab.adapters.data.ccxt_rest import CcxtRestSource
from quantlab.adapters.data.guard import PartitionGuard
from quantlab.adapters.secrets import ChainedSecrets, DotEnvSecrets, KeychainSecrets
from quantlab.core.splits import SplitPolicy, parse_split_policy
from quantlab.core.types import BarFrame
from quantlab.ports.clock import Clock
from quantlab.ports.data import MarketDataSource
from quantlab.ports.secrets import Secrets
from quantlab.ports.store import ArtifactStore, DatasetRecord, ExperimentStore

POLICY = {
    "symbol": "BTC/USDT",
    "timeframe": "1h",
    "train": {"start": "2023-01-01T00:00:00Z", "end": "2023-01-01T11:00:00Z"},
    "embargo_bars": 2,
    "validation": {"start": "2023-01-01T15:00:00Z", "end": "2023-01-01T19:00:00Z"},
    "test": {"start": "2023-01-01T23:00:00Z", "end": None},
}


# ---------------------------------------------------------------------------
# MarketDataSource
# ---------------------------------------------------------------------------
def test_the_parquet_store_is_a_market_data_source(tmp_path: Path) -> None:
    assert isinstance(ParquetBarStore(tmp_path), MarketDataSource)


def test_the_guard_is_a_market_data_source(tmp_path: Path) -> None:
    guard = PartitionGuard(ParquetBarStore(tmp_path), parse_split_policy(POLICY))
    assert isinstance(guard, MarketDataSource)


def test_the_rest_adapter_is_deliberately_not_a_market_data_source() -> None:
    """Live REST output has no dataset_id, so a run using it is not reproducible.

    Keeping it outside the port is the point: nothing can be handed to a
    backtest as if it were stored, validated, content-addressed history.
    """
    source = CcxtRestSource(_StubExchange())
    assert not isinstance(source, MarketDataSource)
    assert not hasattr(source, "dataset_id")


class _StubExchange:
    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):  # type: ignore[no-untyped-def]
        return []


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------
class ManualClock:
    """A clock a test drives by hand; the shape the paper runtime will inject."""

    def __init__(self, now_ms: int = 0) -> None:
        self._now = now_ms
        self.slept: list[float] = []

    def now_ms(self) -> int:
        return self._now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self._now += int(seconds * 1000)


def test_a_manual_clock_satisfies_the_port() -> None:
    assert isinstance(ManualClock(), Clock)


def test_a_clock_without_sleep_does_not_satisfy_the_port() -> None:
    class Broken:
        def now_ms(self) -> int:
            return 0

    assert not isinstance(Broken(), Clock)


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------
def test_the_secret_adapters_satisfy_the_port() -> None:
    for backend in (KeychainSecrets(), DotEnvSecrets(".env"), ChainedSecrets([])):
        assert isinstance(backend, Secrets)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------
class StubDataset:
    dataset_id = "d0"
    exchange = "binance"
    symbol = "BTC/USDT"
    timeframe = "1h"
    start_ts = 0
    end_ts = 1
    n_bars = 2


class StubExperimentStore:
    """A store that does nothing, to prove the port is satisfiable without SQL.

    It carries every method the protocol declares. When the port grows, this
    fails to satisfy it until it grows too — which is the point: the protocol is
    only a contract if something independent of the adapter can meet it.
    """

    def get_or_create_dataset(self, **_kwargs: Any) -> StubDataset:
        return StubDataset()

    def get_or_create_split(self, policy: SplitPolicy) -> SplitPolicy:
        return policy

    def freeze_test_end(self, split_id: str, end_ts: int) -> SplitPolicy:
        return parse_split_policy(POLICY).freeze_test_end(end_ts)

    def create_family(self, **_kwargs: Any) -> Any:
        return None

    def add_strategy_version(self, **_kwargs: Any) -> Any:
        return None

    def get_strategy_version(self, strategy_id: str) -> Any:
        return None

    def lineage(self, strategy_id: str) -> list[Any]:
        return []

    def find_family(self, family_id: str) -> Any:
        return None

    def record_verdict(self, **_fields: Any) -> Any:
        return None

    def verdicts_for(self, strategy_id: str) -> list[Any]:
        return []

    def increment_validation_touches(self, family_id: str) -> int:
        return 0

    def set_family_status(self, family_id: str, status: str) -> Any:
        return None

    def create_experiment(self, **_kwargs: Any) -> Any:
        return None

    def metrics_for(self, run_id: str) -> dict[str, float | None]:
        return {}

    def trades_for(self, run_id: str) -> list[Any]:
        return []

    def find_run(self, run_id: str) -> Any:
        return None

    def create_run(self, **_kwargs: Any) -> Any:
        return None

    def finish_run(self, run_id: str, status: str, *_args: Any, **_kwargs: Any) -> Any:
        return None

    def query_runs(self, **_filters: Any) -> list[Any]:
        return []

    def delete_run(self, run_id: str, *, confirm: Literal[True]) -> None:
        return None

    def save_verdict(self, **_kwargs: Any) -> Any:
        return None

    def record_llm_interaction(self, **_fields: Any) -> Any:
        return None

    def record_lockbox_access(self, **_kwargs: Any) -> Any:
        return None

    def record_training_environment(self, **_kwargs: Any) -> Any:
        return None

    def find_training_environment(self, evolution_id: str, gen_index: int) -> Any:
        return None

    def training_environments_for(self, evolution_id: str) -> Any:
        return []

    def create_evolution_run(self, **_fields: Any) -> Any:
        return None

    def finish_evolution_run(self, evolution_id: str, status: str, *_a: Any, **_k: Any) -> Any:
        return None

    def count_evaluation(self, evolution_id: str, n: int = 1) -> int:
        return 0

    def add_generation(self, **_fields: Any) -> Any:
        return None

    def score_candidate(self, candidate_id: str, **_fields: Any) -> Any:
        return None

    def add_candidate(self, **_fields: Any) -> Any:
        return None

    def add_mutations(self, candidate_id: str, mutations: Any) -> None:
        return None

    def record_optuna_study(self, **_fields: Any) -> Any:
        return None

    def record_promotion(self, **_fields: Any) -> Any:
        return None

    def promotions_for(self, candidate_id: str) -> list[Any]:
        return []

    def candidates_for(self, evolution_id: str, gen_index: int | None = None) -> list[Any]:
        return []

    def find_evolution_run(self, evolution_id: str) -> Any:
        return None

    def generations_for(self, evolution_id: str) -> list[Any]:
        return []

    def mutations_for(self, candidate_id: str) -> list[Any]:
        return []

    def ancestry(self, candidate_id: str) -> list[Any]:
        return []

    def descendants(self, candidate_id: str) -> list[Any]:
        return []


class StubArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def dir_for(self, run_id: str) -> Path:
        return self.root / run_id

    def write_parquet(self, run_id: str, name: str, df: pd.DataFrame) -> Path:
        return self.dir_for(run_id) / name

    def write_json(self, run_id: str, name: str, obj: Any) -> Path:
        return self.dir_for(run_id) / name

    def read_json(self, run_id: str, name: str) -> Any:
        return {}


def test_the_store_protocols_are_satisfiable(tmp_path: Path) -> None:
    assert isinstance(StubDataset(), DatasetRecord)
    assert isinstance(StubExperimentStore(), ExperimentStore)
    assert isinstance(StubArtifactStore(tmp_path), ArtifactStore)


def test_freeze_test_end_flows_through_the_store_protocol() -> None:
    store: ExperimentStore = StubExperimentStore()
    frozen = store.freeze_test_end("s0", 1_700_000_000_000)
    assert frozen.test_end_ts == 1_700_000_000_000


def test_a_partial_store_does_not_satisfy_the_protocol() -> None:
    class Partial:
        def get_or_create_split(self, policy: SplitPolicy) -> SplitPolicy:
            return policy

    assert not isinstance(Partial(), ExperimentStore)


# ---------------------------------------------------------------------------
# a source built only from the port still works end to end
# ---------------------------------------------------------------------------
def test_a_hand_written_source_can_replace_the_adapter() -> None:
    """Spec section 2.3: replacing a component means writing one adapter, nothing more."""
    from tests.helpers import make_bars

    class InMemorySource:
        def __init__(self) -> None:
            self.frame = BarFrame(
                make_bars(24), symbol="BTC/USDT", timeframe="1h", dataset_id="mem0"
            )

        def load(self, symbol: str, timeframe: str, start_ts: int, end_ts: int) -> BarFrame:
            return self.frame.slice(start_ts, end_ts)

        def dataset_id(self, symbol: str, timeframe: str) -> str:
            return "mem0"

        def available_range(self, symbol: str, timeframe: str) -> tuple[int, int] | None:
            return (self.frame.start_ts or 0, self.frame.end_ts or 0)

    source = InMemorySource()
    assert isinstance(source, MarketDataSource)

    guarded = PartitionGuard(source, parse_split_policy(POLICY))
    loaded = guarded.load("BTC/USDT", "1h", 0, parse_split_policy(POLICY).test_start_ts - 1)
    assert loaded.n_bars == 23
    assert loaded.dataset_id == "mem0"
