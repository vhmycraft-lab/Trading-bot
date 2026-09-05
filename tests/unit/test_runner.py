"""Run identity, caching and the record of a run (master spec section 11, task T22).

The acceptance criterion is short — cache hit, `--force`, failed runs re-run — but
what it is protecting is not. Section 11.2's cache is what lets an evolutionary
generation carry survivors forward without re-deriving their numbers, and section
14.4 deflates for how many things were tried; a cache that missed when it should
have hit would inflate that count, and one that hit when it should have missed
would serve numbers produced by different inputs.

The engine is a spy rather than the real one wherever the test is about the
runner: whether something executed is the fact under test, and a spy is the only
way to assert it directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine
from structlog.testing import capture_logs

from quantlab.adapters.engine.simple_bar import SimpleBarEngine
from quantlab.adapters.store.artifacts import FileArtifactStore
from quantlab.adapters.store.sqlite import SqliteExperimentStore, make_session_factory
from quantlab.core.errors import EngineError, RunConflict, StrategyRuntimeError
from quantlab.core.hashing import run_id as canonical_run_id
from quantlab.core.splits import parse_split_policy
from quantlab.core.types import BacktestConfig, BacktestResult, BarFrame
from quantlab.experiments.env import environment_report, git_state
from quantlab.experiments.runner import ARTIFACT_NAMES, ExperimentRunner, RunOutcome
from quantlab.ports.engine import StrategyEvaluator

HOUR = 3_600_000


def make_bars(n: int = 200, seed: int = 5) -> BarFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + rng.normal(0, 0.6, n).cumsum() + 6.0 * np.sin(np.arange(n) / 11.0)
    return BarFrame(
        pd.DataFrame(
            {
                "ts_open": (np.arange(n, dtype="int64") * HOUR),
                "open": close,
                "high": close + np.abs(rng.normal(0, 0.4, n)),
                "low": close - np.abs(rng.normal(0, 0.4, n)),
                "close": close,
                "volume": np.full(n, 10.0),
                "quote_volume": np.full(n, 1000.0),
                "trades": np.full(n, 5, dtype="int64"),
                "is_gap_filled": np.zeros(n, dtype=bool),
            }
        ),
        symbol="BTCUSDT",
        timeframe="1h",
        dataset_id="d0",
    )


TREND_STRATEGY = """
from quantlab.core.strategy import Context, ParamSpec, Signal, SignalKind


class Trend:
    name = "trend"
    version = "1"
    style = "bar_loop"
    params = {"n": ParamSpec(kind="int", default=5, low=2, high=50)}
    warmup_bars = 20

    def prepare(self, params):
        self.p = dict(params)

    def on_bar(self, ctx: Context) -> Signal:
        fast = ctx.ind("sma", n=self.p["n"])
        slow = ctx.ind("sma", n=self.p["n"] * 3)
        return Signal(SignalKind.LONG) if fast > slow else Signal(SignalKind.FLAT)
"""


class SpyEvaluator:
    """A real engine behind a counter.

    Real, not fake: a cache test that never produced a result would prove the
    cache returns *something*, not that it returns the same thing.
    """

    engine_name = SimpleBarEngine.name
    engine_version = SimpleBarEngine.version

    def __init__(self, source: str = TREND_STRATEGY) -> None:
        namespace: dict[str, Any] = {}
        exec(compile(source + "\n\nSTRATEGY = Trend\n", "spy.py", "exec"), namespace)
        self._strategy = namespace["STRATEGY"]()
        self.calls = 0
        self.segments: list[int] = []

    def evaluate(self, bars: BarFrame, params: Any, config: BacktestConfig) -> BacktestResult:
        self.calls += 1
        self.segments.append(bars.n_bars)
        return SimpleBarEngine().run(self._strategy, bars, params, config)


class RaisingEvaluator:
    engine_name = SimpleBarEngine.name
    engine_version = SimpleBarEngine.version

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc
        self.calls = 0

    def evaluate(self, bars: BarFrame, params: Any, config: BacktestConfig) -> BacktestResult:
        self.calls += 1
        raise self.exc


POLICY = {
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "train": {"start": 0, "end": 100 * HOUR},
    "validation": {"start": 102 * HOUR, "end": 150 * HOUR},
    "test": {"start": 152 * HOUR, "end": None},
    "embargo_bars": 1,
}


@pytest.fixture
def store(db_engine: Engine) -> SqliteExperimentStore:
    return SqliteExperimentStore(make_session_factory(db_engine))


@pytest.fixture
def artifacts(tmp_path: Path) -> FileArtifactStore:
    return FileArtifactStore(tmp_path / "artifacts")


@pytest.fixture
def runner(store: SqliteExperimentStore, artifacts: FileArtifactStore) -> ExperimentRunner:
    return ExperimentRunner(store, artifacts)


@pytest.fixture
def wired(store: SqliteExperimentStore) -> dict[str, Any]:
    """The rows a run must be able to cite before it can exist."""
    store.get_or_create_dataset(
        dataset_id="d0",
        exchange="binance",
        symbol="BTCUSDT",
        timeframe="1h",
        start_ts=0,
        end_ts=200 * HOUR,
        n_bars=200,
        manifest_json="{}",
    )
    policy = store.get_or_create_split(parse_split_policy(POLICY, dataset_id="d0"))
    family = store.create_family(name="trend", origin="human")
    store.add_strategy_version(
        strategy_id="s0",
        family_id=family.family_id,
        code_path="s0.py",
        code_sha256="a" * 64,
        class_name="Trend",
        param_schema_json="{}",
        style="bar_loop",
        author="human",
        logic_lines=12,
    )
    experiment = store.create_experiment(
        campaign="c1",
        purpose="train",
        config_hash="cfg",
        config_json="{}",
        seed=1,
        git_commit="abc",
    )
    return {"split_id": policy.split_id, "experiment_id": experiment.experiment_id}


def _kwargs(wired: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    base = {
        "experiment_id": wired["experiment_id"],
        "strategy_id": "s0",
        "dataset_id": "d0",
        "split_id": wired["split_id"],
        "segment": "train",
        "bars": make_bars(),
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# identity (section 11.2)
# ---------------------------------------------------------------------------
def test_the_run_id_is_the_canonical_rule_not_a_second_one() -> None:
    """Section 11.2's hash lives in ``core.hashing``. Two spellings of it would
    split the cache without anyone noticing."""
    config = BacktestConfig(fee_bps=7.0)
    computed = ExperimentRunner.run_id_for(
        strategy_id="s0",
        params={"n": 5},
        dataset_id="d0",
        split_id="sp0",
        segment="train",
        config=config,
        seed=config.seed,
        engine_name="simple_bar",
        engine_version="1",
    )
    assert computed == canonical_run_id(
        strategy="s0",
        params={"n": 5},
        dataset="d0",
        split="sp0",
        segment="train",
        backtest_config_hash=config.config_hash(),
        seed=config.seed,
        engine_name="simple_bar",
        engine_version="1",
    )
    assert len(computed) == 16


def test_the_runner_does_not_reimplement_the_identity_rule() -> None:
    source = (Path("src/quantlab/experiments/runner.py")).read_text(encoding="utf-8")
    assert "hashlib" not in source
    assert "from quantlab.core.hashing import" in source


@pytest.mark.parametrize(
    "change",
    [
        {"strategy_id": "other"},
        {"params": {"n": 6}},
        {"dataset_id": "other"},
        {"split_id": "other"},
        {"segment": "val"},
        {"seed": 99},
        {"engine_name": "other"},
        {"engine_version": "2"},
        {"config": BacktestConfig(fee_bps=99.0)},
    ],
)
def test_every_input_the_result_depends_on_changes_the_run_id(change: dict[str, Any]) -> None:
    """If a changed input did not change the id, the cache would serve numbers
    produced by different inputs — the exact failure section 11.2 exists to
    prevent."""
    base = {
        "strategy_id": "s0",
        "params": {"n": 5},
        "dataset_id": "d0",
        "split_id": "sp0",
        "segment": "train",
        "config": BacktestConfig(),
        "seed": 0,
        "engine_name": "simple_bar",
        "engine_version": "1",
    }
    changed = {**base, **change}
    assert ExperimentRunner.run_id_for(**base) != ExperimentRunner.run_id_for(**changed)


def test_parameter_order_does_not_change_the_run_id() -> None:
    """Identity is the parameters, not how the dict was built."""
    common = {
        "strategy_id": "s0",
        "dataset_id": "d0",
        "split_id": "sp0",
        "segment": "train",
        "config": BacktestConfig(),
        "seed": 0,
        "engine_name": "simple_bar",
        "engine_version": "1",
    }
    first = ExperimentRunner.run_id_for(params={"a": 1, "b": 2}, **common)
    second = ExperimentRunner.run_id_for(params={"b": 2, "a": 1}, **common)
    assert first == second


# ---------------------------------------------------------------------------
# the acceptance criterion: cache hit, force, failed re-runs
# ---------------------------------------------------------------------------
def test_the_same_backtest_twice_executes_once(
    runner: ExperimentRunner, wired: dict[str, Any]
) -> None:
    evaluator = SpyEvaluator()
    first = runner.run(evaluator, **_kwargs(wired))
    second = runner.run(evaluator, **_kwargs(wired))

    assert evaluator.calls == 1, "the engine ran again on a cache hit"
    assert first.run_id == second.run_id
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.record.status == "ok"


def test_a_cache_hit_is_logged_as_one(runner: ExperimentRunner, wired: dict[str, Any]) -> None:
    """Section 11.2 asks for ``cache_hit=True`` in the log, and the acceptance
    criterion is stated in terms of what the log says.

    Captured through structlog rather than ``caplog``: the events carry structured
    fields, and ``cache_hit`` being *the* field under test is worth asserting on
    directly rather than through the rendered string.
    """
    evaluator = SpyEvaluator()
    with capture_logs() as events:
        first = runner.run(evaluator, **_kwargs(wired))
        second = runner.run(evaluator, **_kwargs(wired))

    by_event = {entry["event"]: entry for entry in events}
    assert by_event["run_started"]["cache_hit"] is False
    assert by_event["run_cache_hit"]["cache_hit"] is True
    assert by_event["run_cache_hit"]["run_id"] == first.run_id == second.run_id


def test_a_forced_re_run_is_logged_as_forced(
    runner: ExperimentRunner, wired: dict[str, Any]
) -> None:
    evaluator = SpyEvaluator()
    runner.run(evaluator, **_kwargs(wired))
    with capture_logs() as events:
        runner.run(evaluator, force=True, **_kwargs(wired))
    forced = [entry for entry in events if entry["event"] == "run_forced"]
    assert forced and forced[0]["forced"] is True and forced[0]["cache_hit"] is False


def test_force_re_executes(runner: ExperimentRunner, wired: dict[str, Any]) -> None:
    evaluator = SpyEvaluator()
    first = runner.run(evaluator, **_kwargs(wired))
    forced = runner.run(evaluator, force=True, **_kwargs(wired))

    assert evaluator.calls == 2
    assert forced.cache_hit is False
    assert forced.result is not None
    assert forced.run_id == first.run_id


def test_a_forced_re_run_reproduces_the_stored_result(
    runner: ExperimentRunner, store: SqliteExperimentStore, wired: dict[str, Any]
) -> None:
    """INV-7 in miniature: the same inputs give the same numbers, so a forced
    re-run agrees with what was stored. This is what makes ``force`` a
    verification rather than an overwrite."""
    evaluator = SpyEvaluator()
    first = runner.run(evaluator, **_kwargs(wired))
    forced = runner.run(evaluator, force=True, **_kwargs(wired))

    assert first.metrics is not None and forced.metrics is not None
    assert forced.metrics.as_dict() == first.metrics.as_dict()
    assert forced.result is not None and first.result is not None
    assert forced.result.trades == first.result.trades


def test_a_forced_re_run_does_not_rewrite_the_stored_run(
    runner: ExperimentRunner, store: SqliteExperimentStore, wired: dict[str, Any]
) -> None:
    """Section 6 permits no deletion or rewrite of a successful run: its numbers
    may already have been cited."""
    evaluator = SpyEvaluator()
    first = runner.run(evaluator, **_kwargs(wired))
    before = store.metrics_for(first.run_id)
    runner.run(evaluator, force=True, **_kwargs(wired))
    assert store.metrics_for(first.run_id) == before
    assert len(store.query_runs()) == 1


def test_a_failed_run_is_re_executed(
    runner: ExperimentRunner, store: SqliteExperimentStore, wired: dict[str, Any]
) -> None:
    """Section 11.2: failed runs are re-executed. They produced no results anyone
    can be citing, and the usual reason one failed is that something has since
    been fixed."""
    broken = RaisingEvaluator(StrategyRuntimeError("strategy raised while deciding"))
    failed = runner.run(broken, **_kwargs(wired))
    assert failed.record.status == "failed"
    assert failed.result is None

    working = SpyEvaluator()
    recovered = runner.run(working, **_kwargs(wired))
    assert working.calls == 1
    assert recovered.cache_hit is False
    assert recovered.record.status == "ok"
    assert recovered.run_id == failed.run_id


def test_a_failed_run_records_what_broke(
    runner: ExperimentRunner, store: SqliteExperimentStore, wired: dict[str, Any]
) -> None:
    """Section 18.2: ``error_json = {type, message, traceback_tail}``."""
    runner.run(RaisingEvaluator(ValueError("deliberate")), **_kwargs(wired))
    row = store.query_runs(status="failed")[0]
    assert row.error_json is not None
    error = json.loads(row.error_json)
    assert error["type"] == "ValueError"
    assert "deliberate" in error["message"]
    assert "Traceback" in error["traceback_tail"] or error["traceback_tail"]


def test_a_failed_run_leaves_no_metrics_or_trades(
    runner: ExperimentRunner, store: SqliteExperimentStore, wired: dict[str, Any]
) -> None:
    outcome = runner.run(RaisingEvaluator(ValueError("boom")), **_kwargs(wired))
    assert store.metrics_for(outcome.run_id) == {}
    assert store.trades_for(outcome.run_id) == []


def test_an_engine_error_fails_the_command_rather_than_the_strategy(
    runner: ExperimentRunner, store: SqliteExperimentStore, wired: dict[str, Any]
) -> None:
    """Section 18.2: an accounting invariant breaking is a bug in the engine. Filed
    as a failed run it would look like a badly behaved strategy, and the bug would
    survive in a plausible-looking record."""
    with pytest.raises(EngineError):
        runner.run(RaisingEvaluator(EngineError("equity != cash + position")), **_kwargs(wired))
    assert store.query_runs(status="failed") == []


# ---------------------------------------------------------------------------
# what a run records
# ---------------------------------------------------------------------------
def test_a_run_writes_every_artifact_the_spec_names(
    runner: ExperimentRunner, artifacts: FileArtifactStore, wired: dict[str, Any]
) -> None:
    outcome = runner.run(SpyEvaluator(), **_kwargs(wired))
    assert set(ARTIFACT_NAMES) <= set(artifacts.listdir(outcome.run_id))
    assert len(ARTIFACT_NAMES) == 10


def test_the_artifacts_hold_what_the_run_produced(
    runner: ExperimentRunner, artifacts: FileArtifactStore, wired: dict[str, Any]
) -> None:
    outcome = runner.run(SpyEvaluator(), params={"n": 5}, **_kwargs(wired))
    assert outcome.result is not None

    assert artifacts.read_json(outcome.run_id, "params.json") == {"n": 5}
    equity = artifacts.read_parquet(outcome.run_id, "equity.parquet")
    assert len(equity) == outcome.result.n_bars
    assert float(equity["equity"].iloc[-1]) == outcome.result.final_equity
    trades = artifacts.read_parquet(outcome.run_id, "trades.parquet")
    assert len(trades) == len(outcome.result.trades)
    metrics = artifacts.read_json(outcome.run_id, "metrics.json")
    assert metrics["n_trades"] == len(outcome.result.trades)


def test_the_environment_is_recorded_so_the_run_can_be_reproduced(
    runner: ExperimentRunner, artifacts: FileArtifactStore, wired: dict[str, Any]
) -> None:
    """INV-7: a metric can be recomputed from stored inputs; the NumPy that
    produced it cannot."""
    outcome = runner.run(SpyEvaluator(), **_kwargs(wired))
    env = artifacts.read_json(outcome.run_id, "env.json")
    assert set(env) == {
        "python",
        "platform",
        "packages",
        "git_commit",
        "git_dirty",
        "engine_version",
        "created_at",
    }
    assert env["engine_version"] == SimpleBarEngine.version
    assert {"numpy", "pandas"} <= set(env["packages"])
    assert isinstance(env["git_dirty"], bool)
    assert env["created_at"] > 1_600_000_000_000


def test_a_dirty_tree_is_recorded_not_refused() -> None:
    """Research happens on dirty trees. A platform that refused one would simply
    be lied to; section 11.3 flags it instead."""
    commit, dirty = git_state()
    assert isinstance(dirty, bool)
    assert commit == "" or len(commit) == 40


def test_the_environment_report_is_self_contained() -> None:
    report = environment_report(engine_version="1")
    assert report["engine_version"] == "1"
    assert report["python"].count(".") >= 2


def test_metrics_and_trades_reach_the_store(
    runner: ExperimentRunner, store: SqliteExperimentStore, wired: dict[str, Any]
) -> None:
    outcome = runner.run(SpyEvaluator(), **_kwargs(wired))
    assert outcome.result is not None
    stored = store.metrics_for(outcome.run_id)
    assert stored["n_trades"] == float(len(outcome.result.trades))
    assert len(store.trades_for(outcome.run_id)) == len(outcome.result.trades)


def test_the_run_row_points_at_its_artifacts(
    runner: ExperimentRunner, artifacts: FileArtifactStore, wired: dict[str, Any]
) -> None:
    outcome = runner.run(SpyEvaluator(), **_kwargs(wired))
    assert outcome.record.artifact_dir == str(artifacts.dir_for(outcome.run_id))


# ---------------------------------------------------------------------------
# isolation and determinism
# ---------------------------------------------------------------------------
def test_the_runner_evaluates_only_the_bars_it_was_given(
    runner: ExperimentRunner, wired: dict[str, Any]
) -> None:
    """INV-5 at this layer: the runner never widens a segment. Whatever the caller
    resolved is exactly what the engine sees, so the lockbox decision is made once,
    upstream, and not re-litigated here."""
    evaluator = SpyEvaluator()
    bars = make_bars(120)
    runner.run(evaluator, **_kwargs(wired, bars=bars))
    assert evaluator.segments == [120]


def test_two_runs_of_the_same_request_agree(
    runner: ExperimentRunner, wired: dict[str, Any]
) -> None:
    """INV-7 across the cache: forcing gives the same numbers as the first run."""
    evaluator = SpyEvaluator()
    first = runner.run(evaluator, **_kwargs(wired))
    again = runner.run(evaluator, force=True, **_kwargs(wired))
    assert first.result is not None and again.result is not None
    pd.testing.assert_series_equal(first.result.equity, again.result.equity)


def test_different_segments_are_different_runs(
    runner: ExperimentRunner, store: SqliteExperimentStore, wired: dict[str, Any]
) -> None:
    evaluator = SpyEvaluator()
    train = runner.run(evaluator, **_kwargs(wired, segment="train"))
    val = runner.run(evaluator, **_kwargs(wired, segment="val"))
    assert train.run_id != val.run_id
    assert evaluator.calls == 2
    assert {r.segment for r in store.query_runs()} == {"train", "val"}


def test_the_outcome_reports_whether_it_succeeded(
    runner: ExperimentRunner, wired: dict[str, Any]
) -> None:
    assert runner.run(SpyEvaluator(), **_kwargs(wired)).ok
    assert not runner.run(RaisingEvaluator(ValueError("x")), **_kwargs(wired, segment="val")).ok


def test_a_spy_evaluator_satisfies_the_port() -> None:
    """The runner depends on the port, not on any adapter (INV-8)."""
    assert isinstance(SpyEvaluator(), StrategyEvaluator)
    assert isinstance(RaisingEvaluator(ValueError("x")), StrategyEvaluator)


def test_the_runner_reprs_usefully(runner: ExperimentRunner) -> None:
    assert "ExperimentRunner" in repr(runner)


def test_an_unknown_run_state_is_refused(
    runner: ExperimentRunner, store: SqliteExperimentStore, wired: dict[str, Any]
) -> None:
    """A run recorded as ``rejected`` is a decision, not a scratch pad."""
    evaluator = SpyEvaluator()
    run_id = ExperimentRunner.run_id_for(
        strategy_id="s0",
        params={},
        dataset_id="d0",
        split_id=wired["split_id"],
        segment="train",
        config=BacktestConfig(),
        seed=0,
        engine_name=evaluator.engine_name,
        engine_version=evaluator.engine_version,
    )
    store.create_run(
        run_id=run_id,
        experiment_id=wired["experiment_id"],
        strategy_id="s0",
        dataset_id="d0",
        split_id=wired["split_id"],
        segment="train",
        params_json="{}",
        engine_name=evaluator.engine_name,
        engine_version=evaluator.engine_version,
        artifact_dir="x",
    )
    store.finish_run(run_id, "rejected")
    with pytest.raises(RunConflict, match="cannot be executed"):
        runner.run(evaluator, **_kwargs(wired))


def test_run_outcome_is_immutable(runner: ExperimentRunner, wired: dict[str, Any]) -> None:
    outcome = runner.run(SpyEvaluator(), **_kwargs(wired))
    with pytest.raises((AttributeError, TypeError)):
        outcome.run_id = "tampered"  # type: ignore[misc]
    assert isinstance(outcome, RunOutcome)
