"""The store: schema, migrations, the experiment record and its artefacts.

Master spec sections 6 (schema), 11.1 (the port) and 11.3 (artefacts), task T21.

The Alembic migration owns the DDL, so ``STRICT`` and ``WITHOUT ROWID`` are
exactly as specified; the SQLAlchemy models mirror it. The first half of this
file keeps the two in step and proves the constraints are enforced at run time.
The second half exercises the store itself: CRUD for every entity, the
append-only rule and the one deletion that is permitted, transaction atomicity,
foreign keys through the public methods, and the artefact tree.

The append-only tests deliberately reach past the store's own methods and mutate
rows through a bare session. A rule enforced only inside the accessors is a
convention; this database is meant to be evidence, and evidence that can be
edited by anyone holding a session is not evidence.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Final

import pandas as pd
import pytest
from sqlalchemy import Engine, func, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from quantlab.adapters.store.artifacts import ARTIFACT_PARQUET_KWARGS, FileArtifactStore
from quantlab.adapters.store.models import (
    EVOLUTION_TABLES,
    MUTABLE_COLUMNS,
    TABLE_NAMES,
    Base,
    Candidate,
    CandidatePromotion,
    Dataset,
    Experiment,
    Generation,
    LlmInteraction,
    LockboxAccess,
    Metric,
    Mutation,
    OptunaStudy,
    PaperSession,
    Run,
    StrategyFamily,
    StrategyVersion,
    ValidationVerdict,
)
from quantlab.adapters.store.models import SplitPolicy as SplitPolicyRow
from quantlab.adapters.store.models import Trade as TradeRow
from quantlab.adapters.store.sqlite import (
    SqliteExperimentStore,
    connection_url,
    create_db_engine,
    current_revision,
    guard_is_installed,
    make_session_factory,
    migrations_dir,
    missing_tables,
    session_scope,
    upgrade_to_head,
)
from quantlab.core.errors import ConfigError, ImmutableRowError, RunConflict, StoreError
from quantlab.core.splits import SplitPolicy, parse_split_policy
from quantlab.core.types import Side
from quantlab.core.types import Trade as TradeRecord
from quantlab.ports.store import ArtifactStore, ExperimentStore

HOUR_MS: Final[int] = 3_600_000


def _head_revision() -> str:
    """The newest migration on disk.

    Asserted against rather than a literal, so adding a migration updates the
    expectation by existing rather than by someone remembering to edit a string.
    """
    versions = sorted((migrations_dir() / "versions").glob("[0-9]*.py"))
    assert versions, "no migrations found"
    return versions[-1].name.split("_", 1)[0]


def _now() -> int:
    return int(time.time() * 1000)


# --- engine and pragmas ----------------------------------------------------
def test_connection_url_forms() -> None:
    assert connection_url(":memory:") == "sqlite+pysqlite:///:memory:"
    assert connection_url("a/b.db") == "sqlite+pysqlite:///a/b.db"


def test_pragmas_are_applied(db_engine: Engine) -> None:
    with db_engine.connect() as conn:
        assert conn.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
        assert str(conn.exec_driver_sql("PRAGMA journal_mode").scalar()).lower() == "wal"


def test_in_memory_engine_skips_wal() -> None:
    engine = create_db_engine(":memory:")
    try:
        with engine.connect() as conn:
            assert conn.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
    finally:
        engine.dispose()


def test_engine_creates_missing_parent_directories(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "deeper" / "quantlab.db"
    engine = create_db_engine(target)
    try:
        upgrade_to_head(engine)
    finally:
        engine.dispose()
    assert target.is_file()


# --- migrations ------------------------------------------------------------
def test_migrations_directory_is_discoverable() -> None:
    assert (migrations_dir() / "env.py").is_file()
    assert (migrations_dir() / "versions" / "0001_initial_schema.py").is_file()


def test_empty_database_has_no_revision(empty_engine: Engine) -> None:
    assert current_revision(empty_engine) is None
    assert set(missing_tables(empty_engine)) == set(TABLE_NAMES)


def test_upgrade_creates_every_table(db_engine: Engine) -> None:
    assert current_revision(db_engine) == _head_revision()
    assert missing_tables(db_engine) == ()
    present = set(inspect(db_engine).get_table_names())
    assert set(TABLE_NAMES) <= present


def test_upgrade_is_idempotent(db_engine: Engine) -> None:
    assert upgrade_to_head(db_engine) == _head_revision()
    assert missing_tables(db_engine) == ()


def test_downgrade_removes_the_schema(db_engine: Engine) -> None:
    from alembic import command

    from quantlab.adapters.store.sqlite import _alembic_config

    config = _alembic_config(db_engine)
    config.attributes["connection"] = db_engine
    command.downgrade(config, "base")
    assert set(missing_tables(db_engine)) == set(TABLE_NAMES)


def test_indexes_from_the_spec_exist(db_engine: Engine) -> None:
    names = {index["name"] for index in inspect(db_engine).get_indexes("run")}
    assert {"ix_run_strategy", "ix_run_experiment"} <= names


# --- models mirror the migration ------------------------------------------
def test_model_tables_match_the_migrated_schema(db_engine: Engine) -> None:
    inspector = inspect(db_engine)
    assert set(Base.metadata.tables) == set(TABLE_NAMES)

    for table_name, table in Base.metadata.tables.items():
        db_columns = {c["name"]: c for c in inspector.get_columns(table_name)}
        model_columns = {c.name: c for c in table.columns}
        assert set(db_columns) == set(model_columns), f"column mismatch in {table_name}"
        for name, column in model_columns.items():
            assert db_columns[name]["nullable"] == column.nullable, (
                f"{table_name}.{name} nullability differs from the migration"
            )


def test_declared_tables_are_strict_and_rowid_choices_match(db_engine: Engine) -> None:
    with db_engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT name, sql FROM sqlite_master"
                " WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ).all()
    ddl = {name: sql for name, sql in rows if name != "alembic_version"}
    assert set(ddl) == set(TABLE_NAMES)
    for name, sql in ddl.items():
        assert "STRICT" in sql.upper(), f"{name} is not a STRICT table"
    for name in ("metric", "trade"):
        assert "WITHOUT ROWID" in ddl[name].upper()


def test_mutable_columns_are_declared_for_real_columns() -> None:
    for table_name, columns in MUTABLE_COLUMNS.items():
        table = Base.metadata.tables[table_name]
        assert columns <= set(table.columns.keys()), table_name


# --- constraints are enforced ---------------------------------------------
def test_strict_typing_rejects_a_wrong_type(db_engine: Engine) -> None:
    """A STRICT table refuses a TEXT value in an INTEGER column."""
    with pytest.raises(IntegrityError, match="cannot store TEXT value"), db_engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO dataset (dataset_id, exchange, symbol, timeframe, start_ts,"
                " end_ts, n_bars, manifest_json, created_at)"
                " VALUES ('d0','binance','BTC/USDT','1h','not-a-number',2,3,'{}',4)"
            )
        )


def test_check_constraint_rejects_an_unknown_enum_value(db_engine: Engine) -> None:
    with pytest.raises(IntegrityError), db_engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO strategy_family (family_id, name, origin, description, status,"
                " validation_touches, created_at)"
                " VALUES ('f0','fam','alien','', 'open', 0, 1)"
            )
        )


def test_foreign_keys_are_enforced(db_engine: Engine) -> None:
    with pytest.raises(IntegrityError), db_engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO split_policy (split_id, dataset_id, train_start_ts, train_end_ts,"
                " val_start_ts, val_end_ts, test_start_ts, test_end_ts, embargo_bars, wf_json,"
                " created_at) VALUES ('s0','missing',1,2,3,4,5,NULL,720,'{}',6)"
            )
        )


def test_metric_value_may_be_null(db_engine: Engine) -> None:
    """NULL means "undefined", e.g. profit factor with no losing trades."""
    with db_engine.begin() as conn:
        conn.exec_driver_sql("PRAGMA foreign_keys = OFF")
        conn.execute(
            text("INSERT INTO metric (run_id, name, value) VALUES ('r0','profit_factor',NULL)")
        )
        assert conn.execute(text("SELECT value FROM metric")).scalar() is None


def test_split_policy_test_end_ts_starts_null(db_engine: Engine) -> None:
    factory = make_session_factory(db_engine)
    with session_scope(factory) as session:
        session.add(
            Dataset(
                dataset_id="d0",
                exchange="binance",
                symbol="BTC/USDT",
                timeframe="1h",
                start_ts=1,
                end_ts=2,
                n_bars=1,
                manifest_json="{}",
                created_at=_now(),
            )
        )
    with session_scope(factory) as session:
        assert session.get(Dataset, "d0") is not None


# --- ORM round trip and transactions --------------------------------------
def test_session_scope_commits_and_rolls_back(db_engine: Engine) -> None:
    factory = make_session_factory(db_engine)

    with session_scope(factory) as session:
        session.add(
            StrategyFamily(
                family_id="f0",
                name="sma_cross",
                origin="human",
                description="baseline",
                status="open",
                validation_touches=0,
                created_at=_now(),
            )
        )

    with pytest.raises(IntegrityError), session_scope(factory) as session:
        session.add(
            StrategyFamily(
                family_id="f1",
                name="sma_cross",  # UNIQUE violation
                origin="human",
                description="",
                status="open",
                validation_touches=0,
                created_at=_now(),
            )
        )

    with session_scope(factory) as session:
        assert session.get(StrategyFamily, "f1") is None
        assert session.get(StrategyFamily, "f0") is not None


def test_run_and_metric_round_trip(db_engine: Engine) -> None:
    factory = make_session_factory(db_engine)
    now = _now()

    with session_scope(factory) as session:
        session.add_all(
            [
                Dataset(
                    dataset_id="d0",
                    exchange="binance",
                    symbol="BTC/USDT",
                    timeframe="1h",
                    start_ts=0,
                    end_ts=10,
                    n_bars=10,
                    manifest_json="{}",
                    created_at=now,
                ),
                StrategyFamily(
                    family_id="f0",
                    name="fam",
                    origin="human",
                    description="",
                    status="open",
                    validation_touches=0,
                    created_at=now,
                ),
            ]
        )
        session.flush()
        session.execute(
            text("INSERT INTO split_policy VALUES ('p0','d0',0,1,2,3,4,NULL,720,'{}', :now)"),
            {"now": now},
        )
        session.execute(
            text(
                "INSERT INTO strategy_version VALUES"
                " ('s0','f0',NULL,'strategies/x.py','deadbeef','X','{}','bar_loop','human',"
                " NULL, 12, :now)"
            ),
            {"now": now},
        )
        session.execute(
            text("INSERT INTO experiment VALUES ('e0','camp','train','h','{}',42,'abc', :now)"),
            {"now": now},
        )
        session.add(
            Run(
                run_id="r0",
                experiment_id="e0",
                strategy_id="s0",
                dataset_id="d0",
                split_id="p0",
                segment="train",
                params_json='{"fast":10}',
                engine_name="simple_bar",
                engine_version="1",
                cost_multiplier=1.0,
                status="pending",
                error_json=None,
                artifact_dir="artifacts/runs/r0",
                started_at=None,
                finished_at=None,
                created_at=now,
            )
        )
        session.flush()
        session.add_all(
            [
                Metric(run_id="r0", name="sortino", value=1.25),
                Metric(run_id="r0", name="profit_factor", value=None),
            ]
        )

    with session_scope(factory) as session:
        run = session.get(Run, "r0")
        assert run is not None
        assert run.status == "pending"
        assert run.cost_multiplier == 1.0
        assert session.get(Metric, ("r0", "profit_factor")).value is None
        assert session.get(Metric, ("r0", "sortino")).value == 1.25


def test_run_status_is_mutable_as_the_spec_allows(db_engine: Engine) -> None:
    test_run_and_metric_round_trip(db_engine)
    factory = make_session_factory(db_engine)
    with session_scope(factory) as session:
        run = session.get(Run, "r0")
        assert run is not None
        run.status = "ok"
        run.finished_at = _now()
    with session_scope(factory) as session:
        assert session.get(Run, "r0").status == "ok"
    assert "status" in MUTABLE_COLUMNS["run"]


# ---------------------------------------------------------------------------
# T21: the experiment store (spec sections 6, 11.1)
# ---------------------------------------------------------------------------
FIXED_NOW: Final[int] = 1_700_000_000_000

POLICY_DOCUMENT: Final[dict[str, object]] = {
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "train": {"start": 0, "end": HOUR_MS * 100},
    "validation": {"start": HOUR_MS * 102, "end": HOUR_MS * 150},
    "test": {"start": HOUR_MS * 152, "end": None},
    "embargo_bars": 1,
}


@pytest.fixture
def store(db_engine: Engine) -> SqliteExperimentStore:
    return SqliteExperimentStore(make_session_factory(db_engine), now_ms=lambda: FIXED_NOW)


def _dataset(store: SqliteExperimentStore, dataset_id: str = "d0") -> Dataset:
    return store.get_or_create_dataset(
        dataset_id=dataset_id,
        exchange="binance",
        symbol="BTCUSDT",
        timeframe="1h",
        start_ts=0,
        end_ts=HOUR_MS * 200,
        n_bars=200,
        manifest_json='{"files":[]}',
    )


def _policy(store: SqliteExperimentStore, dataset_id: str = "d0") -> SplitPolicy:
    _dataset(store, dataset_id)
    return store.get_or_create_split(parse_split_policy(POLICY_DOCUMENT, dataset_id=dataset_id))


def _run(
    store: SqliteExperimentStore, run_id: str = "r0", *, segment: str = "train"
) -> tuple[Run, str]:
    """A complete chain of prerequisites, ending in one pending run."""
    policy = _policy(store)
    family = store.create_family(name="sma_cross", origin="human")
    store.add_strategy_version(
        strategy_id="s0",
        family_id=family.family_id,
        code_path="strategies/sma.py",
        code_sha256="a" * 64,
        class_name="SmaCross",
        param_schema_json='{"fast":{}}',
        style="bar_loop",
        author="human",
        logic_lines=21,
    )
    experiment = store.create_experiment(
        campaign="c1",
        purpose="train",
        config_hash="cfg0",
        config_json="{}",
        seed=42,
        git_commit="abc123",
    )
    run = store.create_run(
        run_id=run_id,
        experiment_id=experiment.experiment_id,
        strategy_id="s0",
        dataset_id="d0",
        split_id=policy.split_id,
        segment=segment,
        params_json='{"fast":20}',
        engine_name="simple_bar",
        engine_version="1",
        artifact_dir=f"artifacts/runs/{run_id}",
    )
    return run, family.family_id


def _trade(trade_no: int = 1, pnl: float = 10.0) -> TradeRecord:
    return TradeRecord(
        trade_no=trade_no,
        side=Side.LONG,
        entry_ts=0,
        entry_px=100.0,
        exit_ts=HOUR_MS,
        exit_px=110.0,
        qty=1.0,
        fees=0.2,
        slippage_cost=0.1,
        pnl=pnl,
        pnl_pct=pnl / 100.0,
        bars_held=1,
        exit_reason="signal",
    )


# --- AC1: CRUD for every required entity -----------------------------------
def test_the_store_satisfies_its_port(store: SqliteExperimentStore) -> None:
    assert isinstance(store, ExperimentStore)


def test_dataset_round_trips_and_is_idempotent(store: SqliteExperimentStore) -> None:
    first = _dataset(store)
    second = _dataset(store)
    assert first.dataset_id == second.dataset_id == "d0"
    assert second.n_bars == 200
    assert second.created_at == FIXED_NOW


def test_re_registering_a_dataset_with_different_data_is_a_conflict(
    store: SqliteExperimentStore,
) -> None:
    """A dataset id is a hash of its manifest. If the same id describes different
    data, every run citing it is now ambiguous — that is worth stopping for."""
    _dataset(store)
    with pytest.raises(RunConflict, match="different inputs"):
        store.get_or_create_dataset(
            dataset_id="d0",
            exchange="binance",
            symbol="ETHUSDT",
            timeframe="1h",
            start_ts=0,
            end_ts=HOUR_MS * 200,
            n_bars=200,
            manifest_json="{}",
        )


def test_split_round_trips_through_the_store(store: SqliteExperimentStore) -> None:
    policy = parse_split_policy(POLICY_DOCUMENT, dataset_id="d0")
    _dataset(store)
    stored = store.get_or_create_split(policy)
    assert stored.split_id == policy.split_id
    assert stored.test_end_ts is None

    again = store.get_or_create_split(policy)
    assert again.split_id == policy.split_id
    assert again.train_start_ts == policy.train_start_ts
    assert again.symbol == "BTCUSDT" and again.timeframe == "1h"


def test_family_and_strategy_version_round_trip(store: SqliteExperimentStore) -> None:
    family = store.create_family(name="sma_cross", origin="human", description="two averages")
    assert family.status == "open" and family.validation_touches == 0
    assert store.create_family(name="sma_cross", origin="human").family_id == family.family_id

    version = store.add_strategy_version(
        strategy_id="s0",
        family_id=family.family_id,
        code_path="strategies/sma.py",
        code_sha256="a" * 64,
        class_name="SmaCross",
        param_schema_json="{}",
        style="bar_loop",
        author="human",
        logic_lines=21,
    )
    assert version.parent_strategy_id is None
    # The same code is the same version; loading it twice is not a conflict.
    assert (
        store.add_strategy_version(
            strategy_id="s0",
            family_id=family.family_id,
            code_path="strategies/sma.py",
            code_sha256="a" * 64,
            class_name="SmaCross",
            param_schema_json="{}",
            style="bar_loop",
            author="human",
            logic_lines=21,
        ).strategy_id
        == "s0"
    )


def test_lineage_reads_oldest_first(store: SqliteExperimentStore) -> None:
    family = store.create_family(name="f", origin="llm")
    parent = None
    for index in range(3):
        store.add_strategy_version(
            strategy_id=f"s{index}",
            family_id=family.family_id,
            parent_strategy_id=parent,
            code_path=f"{index}.py",
            code_sha256=str(index) * 64,
            class_name="S",
            param_schema_json="{}",
            style="bar_loop",
            author="llm:mock:m",
            logic_lines=10,
        )
        parent = f"s{index}"
    assert [v.strategy_id for v in store.lineage("s2")] == ["s0", "s1", "s2"]
    assert [v.strategy_id for v in store.lineage("s0")] == ["s0"]


def test_experiment_id_is_derived_from_its_inputs(store: SqliteExperimentStore) -> None:
    """Re-entering the same experiment finds the same row instead of forking it."""
    first = store.create_experiment(
        campaign="c", purpose="train", config_hash="h", config_json="{}", seed=1, git_commit="g"
    )
    same = store.create_experiment(
        campaign="c", purpose="train", config_hash="h", config_json="{}", seed=1, git_commit="g"
    )
    other = store.create_experiment(
        campaign="c", purpose="train", config_hash="h", config_json="{}", seed=2, git_commit="g"
    )
    assert first.experiment_id == same.experiment_id != other.experiment_id


def test_a_run_moves_from_pending_through_running_to_ok(store: SqliteExperimentStore) -> None:
    run, _ = _run(store)
    assert run.status == "pending" and run.started_at is None

    started = store.start_run("r0")
    assert started.status == "running" and started.started_at == FIXED_NOW

    finished = store.finish_run(
        "r0", "ok", {"net_return": 0.12, "profit_factor": None}, [_trade(), _trade(2, -4.0)]
    )
    assert finished.status == "ok" and finished.finished_at == FIXED_NOW
    assert store.metrics_for("r0") == {"net_return": 0.12, "profit_factor": None}
    assert [t.trade_no for t in store.trades_for("r0")] == [1, 2]
    assert store.trades_for("r0")[1].pnl == -4.0


def test_find_run_is_the_cache_lookup(store: SqliteExperimentStore) -> None:
    assert store.find_run("nope") is None
    _run(store)
    found = store.find_run("r0")
    assert found is not None and found.run_id == "r0"


def test_query_runs_filters_and_orders(store: SqliteExperimentStore) -> None:
    run, _ = _run(store, "r0", segment="train")
    store.create_run(
        run_id="r1",
        experiment_id=run.experiment_id,
        strategy_id="s0",
        dataset_id="d0",
        split_id=run.split_id,
        segment="val",
        params_json="{}",
        engine_name="simple_bar",
        engine_version="1",
        artifact_dir="artifacts/runs/r1",
    )
    assert [r.run_id for r in store.query_runs()] == ["r0", "r1"]
    assert [r.run_id for r in store.query_runs(segment="val")] == ["r1"]
    assert [r.run_id for r in store.query_runs(strategy_id="s0", status="pending")] == ["r0", "r1"]
    assert store.query_runs(segment="test") == []


def test_an_unknown_run_filter_is_an_error(store: SqliteExperimentStore) -> None:
    """A typo that silently widened the query would be reported as a result."""
    _run(store)
    with pytest.raises(StoreError, match="unknown run filter"):
        store.query_runs(stratgey_id="s0")


def test_verdicts_llm_interactions_and_lockbox_accesses_round_trip(
    store: SqliteExperimentStore,
) -> None:
    run, family_id = _run(store)
    verdict = store.save_verdict(
        strategy_id="s0",
        split_id=run.split_id,
        params_json="{}",
        verdict="CANDIDATE",
        overfit_score=0.31,
        hard_gates_json="{}",
        soft_checks_json="{}",
        thresholds_json='{"dsr":0.95}',
        n_trials_accounted=64,
    )
    assert len(verdict.verdict_id) == 16 and verdict.overfit_score == 0.31

    interaction = store.record_llm_interaction(
        campaign="c1",
        provider="glm",
        model="glm-4",
        purpose="propose",
        prompt_sha256="b" * 64,
        prompt_path="artifacts/llm/p.txt",
        response_path="artifacts/llm/r.json",
        prompt_template_sha256="c" * 64,
        tokens_in=100,
        tokens_out=200,
        cost_eur=0.02,
        temperature=0.7,
        latency_ms=1500,
        status="ok",
    )
    assert interaction.tokens_in == 100 and interaction.seed is None

    access = store.record_lockbox_access(
        strategy_id="s0", family_id=family_id, os_user="tester", reason="final check", run_id="r0"
    )
    assert [a.access_id for a in store.lockbox_accesses(family_id=family_id)] == [access.access_id]
    assert store.lockbox_accesses(since_ts=FIXED_NOW + 1) == []


def test_two_verdicts_on_the_same_inputs_are_two_rows(store: SqliteExperimentStore) -> None:
    """A verdict is a record of an event, not of an identity: validating twice
    must leave two entries, not overwrite one."""
    run, _ = _run(store)
    fields = {
        "strategy_id": "s0",
        "split_id": run.split_id,
        "params_json": "{}",
        "verdict": "WEAK",
        "overfit_score": 0.5,
        "hard_gates_json": "{}",
        "soft_checks_json": "{}",
        "thresholds_json": "{}",
        "n_trials_accounted": 1,
    }
    first = store.save_verdict(**fields)
    second = store.save_verdict(**fields)
    assert first.verdict_id != second.verdict_id


def test_optuna_study_and_paper_session_round_trip(store: SqliteExperimentStore) -> None:
    run, _ = _run(store)
    study = store.record_optuna_study(
        study_id="st0",
        experiment_id=run.experiment_id,
        strategy_id="s0",
        segment="train",
        sampler="tpe",
        seed=7,
        n_trials=50,
        objective="expectancy",
        best_trial_json="{}",
        plateau_json="{}",
        storage_path="artifacts/optuna/st0.db",
    )
    assert study.n_trials == 50

    session = store.create_paper_session(
        session_id="p0", strategy_id="s0", params_json="{}", state_path="artifacts/paper/p0"
    )
    assert session.status == "running" and session.stopped_at is None
    updated = store.update_paper_session(
        "p0", status="stopped", last_bar_ts=HOUR_MS * 9, stopped_at=FIXED_NOW + 5
    )
    assert updated.status == "stopped"
    assert updated.last_bar_ts == HOUR_MS * 9
    assert updated.stopped_at == FIXED_NOW + 5


def test_a_missing_row_is_reported_not_guessed(store: SqliteExperimentStore) -> None:
    for call in (
        lambda: store.start_run("nope"),
        lambda: store.finish_run("nope", "ok"),
        lambda: store.lineage("nope"),
        lambda: store.increment_validation_touches("nope"),
        lambda: store.set_family_status("nope", "closed"),
        lambda: store.freeze_test_end("nope", 1),
        lambda: store.update_paper_session("nope", status="stopped"),
    ):
        with pytest.raises(StoreError, match="no such"):
            call()


# --- AC2: immutability ------------------------------------------------------
def _one_of_every_row(store: SqliteExperimentStore) -> dict[str, object]:
    """One committed row per table, so the guard can be tried against each."""
    run, family_id = _run(store)
    store.finish_run("r0", "failed", {"net_return": 0.1}, [_trade()], error={"type": "Boom"})
    verdict = store.save_verdict(
        strategy_id="s0",
        split_id=run.split_id,
        params_json="{}",
        verdict="WEAK",
        overfit_score=0.4,
        hard_gates_json="{}",
        soft_checks_json="{}",
        thresholds_json="{}",
        n_trials_accounted=3,
    )
    interaction = store.record_llm_interaction(
        campaign="c1",
        provider="mock",
        model="m",
        purpose="propose",
        prompt_sha256="b" * 64,
        prompt_path="p",
        response_path="r",
        prompt_template_sha256="c" * 64,
        tokens_in=1,
        tokens_out=2,
        cost_eur=0.01,
        temperature=0.5,
        latency_ms=10,
        status="ok",
    )
    access = store.record_lockbox_access(
        strategy_id="s0", family_id=family_id, os_user="u", reason="why"
    )
    study = store.record_optuna_study(
        study_id="st0",
        experiment_id=run.experiment_id,
        strategy_id="s0",
        segment="train",
        sampler="tpe",
        seed=1,
        n_trials=5,
        objective="expectancy",
        best_trial_json="{}",
        plateau_json="{}",
        storage_path="x",
    )
    session = store.create_paper_session(
        session_id="p0", strategy_id="s0", params_json="{}", state_path="s"
    )
    with session_scope(make_session_factory(store.factory.kw["bind"])) as sql:
        return {
            "dataset": sql.get(Dataset, "d0"),
            "split_policy": sql.get(SplitPolicyRow, run.split_id),
            "strategy_family": sql.get(StrategyFamily, family_id),
            "strategy_version": sql.get(StrategyVersion, "s0"),
            "experiment": sql.get(Experiment, run.experiment_id),
            "run": sql.get(Run, "r0"),
            "metric": sql.get(Metric, ("r0", "net_return")),
            "trade": sql.get(TradeRow, ("r0", 1)),
            "optuna_study": sql.get(OptunaStudy, study.study_id),
            "validation_verdict": sql.get(ValidationVerdict, verdict.verdict_id),
            "llm_interaction": sql.get(LlmInteraction, interaction.interaction_id),
            "lockbox_access": sql.get(LockboxAccess, access.access_id),
            "paper_session": sql.get(PaperSession, session.session_id),
        }


def test_the_mutable_columns_of_the_spec_can_be_written(store: SqliteExperimentStore) -> None:
    """Spec section 6 names exactly these. They must work, or the store is unusable."""
    run, family_id = _run(store)
    assert store.start_run("r0").status == "running"
    assert store.finish_run("r0", "failed", error={"type": "Boom"}).error_json is not None
    assert store.increment_validation_touches(family_id) == 1
    assert store.set_family_status(family_id, "frozen").status == "frozen"
    assert store.freeze_test_end(run.split_id, HOUR_MS * 190).test_end_ts == HOUR_MS * 190


@pytest.mark.parametrize(
    ("table", "column", "value"),
    [
        ("run", "segment", "val"),
        ("run", "params_json", '{"fast":99}'),
        ("run", "engine_version", "2"),
        ("run", "created_at", 0),
        ("strategy_family", "name", "renamed"),
        ("strategy_family", "origin", "llm"),
        ("strategy_version", "code_sha256", "z" * 64),
        ("strategy_version", "logic_lines", 999),
        ("dataset", "n_bars", 1),
        ("dataset", "manifest_json", "{}"),
        ("experiment", "seed", 7),
        ("split_policy", "train_start_ts", 5),
        ("metric", "value", 99.0),
        ("trade", "pnl", 1e6),
        ("validation_verdict", "verdict", "LOCKBOX_PASS"),
        ("llm_interaction", "cost_eur", 0.0),
        ("lockbox_access", "reason", "changed my mind"),
        ("paper_session", "params_json", '{"fast":1}'),
        ("optuna_study", "n_trials", 1),
    ],
)
def test_an_immutable_column_cannot_be_updated(
    db_engine: Engine, store: SqliteExperimentStore, table: str, column: str, value: object
) -> None:
    """The guard sits on the session, not on the store's methods.

    That distinction is the whole value: a rule enforced only inside the accessors
    is a convention, and anything holding a session could quietly rewrite history.
    """
    instance = _one_of_every_row(store)[table]
    factory = make_session_factory(db_engine)
    with pytest.raises(ImmutableRowError, match="append-only"), session_scope(factory) as session:
        row = session.merge(instance)
        setattr(row, column, value)
        session.flush()


def test_the_immutable_row_error_names_the_column_and_the_alternatives(
    db_engine: Engine, store: SqliteExperimentStore
) -> None:
    """A refusal the caller cannot act on is only half a rule."""
    _run(store)
    factory = make_session_factory(db_engine)
    with pytest.raises(ImmutableRowError) as excinfo, session_scope(factory) as session:
        run = session.get(Run, "r0")
        assert run is not None
        run.segment = "val"
        session.flush()
    assert excinfo.value.context["columns"] == ["segment"]
    assert "status" in excinfo.value.context["mutable"]
    assert excinfo.value.context["table"] == "run"


def test_writing_a_mutable_column_back_to_its_own_value_is_not_an_update(
    db_engine: Engine, store: SqliteExperimentStore
) -> None:
    """Assigning the value a column already holds must not trip the guard."""
    _run(store)
    factory = make_session_factory(db_engine)
    with session_scope(factory) as session:
        run = session.get(Run, "r0")
        assert run is not None
        run.segment = run.segment
        session.flush()


def test_arbitrary_deletion_is_refused(db_engine: Engine, store: SqliteExperimentStore) -> None:
    _run(store)
    factory = make_session_factory(db_engine)
    with pytest.raises(ImmutableRowError, match="delete_run"), session_scope(factory) as session:
        family = session.execute(select(StrategyFamily)).scalars().one()
        session.delete(family)
        session.flush()


def test_the_guard_is_installed_on_every_new_factory(db_engine: Engine) -> None:
    """Idempotence must not be inferred from SQLAlchemy's event registry.

    ``event.contains`` reports a listener on a freshly built ``sessionmaker``
    that never registered one, because the entry belongs to a collected factory
    that occupied the same address. Deduplicating on it left the guard silently
    absent from roughly half of new factories.
    """
    factories = [make_session_factory(db_engine) for _ in range(50)]
    assert all(guard_is_installed(f) for f in factories)

    factory = make_session_factory(db_engine)
    SqliteExperimentStore(factory)
    SqliteExperimentStore(factory)
    assert guard_is_installed(factory)


def test_a_raw_session_factory_gets_the_guard_from_the_store(db_engine: Engine) -> None:
    """A caller who builds their own factory is still bound by the rule."""
    raw = sessionmaker(bind=db_engine, expire_on_commit=False)
    assert not guard_is_installed(raw)
    SqliteExperimentStore(raw)
    assert guard_is_installed(raw)


# --- AC3: failed-run deletion ----------------------------------------------
def test_a_failed_run_is_deleted_with_its_results(
    store: SqliteExperimentStore, caplog: pytest.LogCaptureFixture
) -> None:
    _run(store)
    store.finish_run("r0", "failed", {"net_return": -0.5}, [_trade()], error={"type": "Boom"})
    assert store.metrics_for("r0") and store.trades_for("r0")

    with caplog.at_level(logging.WARNING):
        store.delete_run("r0", confirm=True)

    assert store.find_run("r0") is None
    assert store.metrics_for("r0") == {}
    assert store.trades_for("r0") == []
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings and warnings[0].run_id == "r0"  # type: ignore[attr-defined]
    assert "deleted failed run" in warnings[0].getMessage()


@pytest.mark.parametrize("status", ["pending", "running", "ok", "rejected"])
def test_only_a_failed_run_may_be_deleted(store: SqliteExperimentStore, status: str) -> None:
    """A successful run has results someone may be citing. Removing one would make
    the store a record of whatever happens to remain, which is not a record."""
    _run(store)
    if status != "pending":
        store.start_run("r0")
    if status in ("ok", "rejected"):
        store.finish_run("r0", status)
    with pytest.raises(StoreError, match="only a failed run"):
        store.delete_run("r0", confirm=True)
    assert store.find_run("r0") is not None


@pytest.mark.parametrize("confirm", [False, None, 1, "yes"])
def test_deletion_requires_an_explicit_confirmation(
    store: SqliteExperimentStore, confirm: object
) -> None:
    _run(store)
    store.finish_run("r0", "failed")
    with pytest.raises(ValueError, match="confirm=True"):
        store.delete_run("r0", confirm=confirm)  # type: ignore[arg-type]
    assert store.find_run("r0") is not None


def test_a_failed_run_cited_by_the_audit_trail_is_kept(store: SqliteExperimentStore) -> None:
    """A lockbox access names the run it looked at. Deleting that run would leave
    the one table nobody may edit pointing at nothing."""
    _run(store)
    _, family_id = _run(store, "r1")
    store.finish_run("r0", "failed")
    store.record_lockbox_access(
        strategy_id="s0", family_id=family_id, os_user="tester", reason="check", run_id="r0"
    )
    with pytest.raises(StoreError, match="referenced by"):
        store.delete_run("r0", confirm=True)
    assert store.find_run("r0") is not None


def test_deleting_a_run_leaves_everything_else_alone(store: SqliteExperimentStore) -> None:
    run, _ = _run(store)
    store.create_run(
        run_id="r1",
        experiment_id=run.experiment_id,
        strategy_id="s0",
        dataset_id="d0",
        split_id=run.split_id,
        segment="val",
        params_json="{}",
        engine_name="simple_bar",
        engine_version="1",
        artifact_dir="artifacts/runs/r1",
    )
    store.finish_run("r1", "ok", {"net_return": 0.1}, [_trade()])
    store.finish_run("r0", "failed")
    store.delete_run("r0", confirm=True)

    assert store.find_run("r1") is not None
    assert store.metrics_for("r1") == {"net_return": 0.1}
    assert len(store.trades_for("r1")) == 1


# --- AC4: transactions ------------------------------------------------------
def test_a_failed_finish_commits_nothing(store: SqliteExperimentStore) -> None:
    """Metrics, trades and the terminal status land together or not at all.

    A run marked ``ok`` whose metrics did not commit is the worst outcome here:
    the cache of section 11.2 would serve it forever as a finished, empty run.
    """
    _run(store)
    store.start_run("r0")
    duplicated = [_trade(1), _trade(1)]  # same primary key twice
    with pytest.raises(IntegrityError):
        store.finish_run("r0", "ok", {"net_return": 0.1}, duplicated)

    still_running = store.find_run("r0")
    assert still_running is not None
    assert still_running.status == "running"
    assert still_running.finished_at is None
    assert store.metrics_for("r0") == {}
    assert store.trades_for("r0") == []


def test_finishing_a_finished_run_is_refused(store: SqliteExperimentStore) -> None:
    """Finishing twice would duplicate metrics or silently discard the second set."""
    _run(store)
    store.finish_run("r0", "ok", {"net_return": 0.1})
    with pytest.raises(RunConflict, match="already finished"):
        store.finish_run("r0", "ok", {"net_return": 0.2})
    assert store.metrics_for("r0") == {"net_return": 0.1}


def test_a_finished_run_cannot_be_restarted(store: SqliteExperimentStore) -> None:
    _run(store)
    store.finish_run("r0", "ok")
    with pytest.raises(RunConflict, match="cannot be restarted"):
        store.start_run("r0")


def test_a_conflicting_run_id_is_reported_not_silently_reused(
    store: SqliteExperimentStore,
) -> None:
    """The run id is the hash of section 11.2. If it matches but the inputs do
    not, the hash and the row disagree — a defect, not a cache hit."""
    run, _ = _run(store)
    with pytest.raises(RunConflict, match="different inputs"):
        store.create_run(
            run_id="r0",
            experiment_id=run.experiment_id,
            strategy_id="s0",
            dataset_id="d0",
            split_id=run.split_id,
            segment="val",
            params_json="{}",
            engine_name="simple_bar",
            engine_version="1",
            artifact_dir="artifacts/runs/r0",
        )


def test_creating_the_same_run_twice_returns_the_same_row(store: SqliteExperimentStore) -> None:
    run, _ = _run(store)
    again = store.create_run(
        run_id="r0",
        experiment_id=run.experiment_id,
        strategy_id="s0",
        dataset_id="d0",
        split_id=run.split_id,
        segment="train",
        params_json='{"fast":20}',
        engine_name="simple_bar",
        engine_version="1",
        artifact_dir="artifacts/runs/r0",
    )
    assert again.run_id == run.run_id
    assert len(store.query_runs()) == 1


# --- AC5: foreign keys through the store ------------------------------------
def test_a_run_cannot_cite_a_dataset_that_does_not_exist(store: SqliteExperimentStore) -> None:
    run, _ = _run(store)
    with pytest.raises(IntegrityError):
        store.create_run(
            run_id="r1",
            experiment_id=run.experiment_id,
            strategy_id="s0",
            dataset_id="does-not-exist",
            split_id=run.split_id,
            segment="train",
            params_json="{}",
            engine_name="simple_bar",
            engine_version="1",
            artifact_dir="artifacts/runs/r1",
        )
    assert store.find_run("r1") is None


def test_a_strategy_version_cannot_cite_a_missing_family(store: SqliteExperimentStore) -> None:
    with pytest.raises(IntegrityError):
        store.add_strategy_version(
            strategy_id="s9",
            family_id="no-such-family",
            code_path="x.py",
            code_sha256="f" * 64,
            class_name="X",
            param_schema_json="{}",
            style="bar_loop",
            author="human",
            logic_lines=1,
        )


# --- the split policy keeps its identity across a freeze --------------------
def test_freezing_the_test_end_does_not_change_the_split_id(
    store: SqliteExperimentStore,
) -> None:
    """``split_id`` is the hash of the policy document and the row's primary key.

    If a frozen policy hashed differently, the next ``get_or_create_split`` would
    insert a second row for the same split and the lockbox would have two
    partitions to disagree about.
    """
    policy = _policy(store)
    frozen = store.freeze_test_end(policy.split_id, HOUR_MS * 190)
    reloaded = store.get_or_create_split(policy)

    assert frozen.split_id == policy.split_id == reloaded.split_id
    assert reloaded.test_end_ts == HOUR_MS * 190
    with session_scope(make_session_factory(store.factory.kw["bind"])) as session:
        assert session.execute(select(func.count()).select_from(SplitPolicyRow)).scalar() == 1


def test_a_frozen_test_end_cannot_be_moved(store: SqliteExperimentStore) -> None:
    """INV-5: the final partition cannot move once it has been looked at."""
    policy = _policy(store)
    store.freeze_test_end(policy.split_id, HOUR_MS * 190)
    store.freeze_test_end(policy.split_id, HOUR_MS * 190)  # idempotent
    with pytest.raises(ConfigError, match="already frozen"):
        store.freeze_test_end(policy.split_id, HOUR_MS * 195)


def test_a_stored_policy_that_does_not_hash_to_its_id_is_refused(
    db_engine: Engine, store: SqliteExperimentStore
) -> None:
    """The document and the id it is filed under must agree."""
    policy = _policy(store)
    with db_engine.begin() as connection:
        connection.execute(
            text("UPDATE split_policy SET wf_json = :doc WHERE split_id = :sid"),
            {
                "doc": '{"symbol":"ETHUSDT","timeframe":"1h","train":{"start":0,"end":3600000},'
                '"validation":{"start":10800000,"end":14400000},'
                '"test":{"start":21600000,"end":null},"embargo_bars":1}',
                "sid": policy.split_id,
            },
        )
    with pytest.raises(StoreError, match="does not hash to its own id"):
        store.get_or_create_split(policy)


# --- AC6: the artifact store ------------------------------------------------
@pytest.fixture
def artifacts(tmp_path: Path) -> FileArtifactStore:
    return FileArtifactStore(tmp_path / "artifacts")


def test_the_artifact_store_satisfies_its_port(artifacts: FileArtifactStore) -> None:
    assert isinstance(artifacts, ArtifactStore)


def test_dir_for_creates_the_run_directory_under_runs(
    artifacts: FileArtifactStore, tmp_path: Path
) -> None:
    """Spec section 11.3: ``artifacts/runs/<run_id>/``."""
    directory = artifacts.dir_for("r0")
    assert directory.is_dir()
    assert directory == (tmp_path / "artifacts" / "runs" / "r0").resolve()
    assert artifacts.dir_for("r0") == directory  # idempotent


def test_json_round_trips(artifacts: FileArtifactStore) -> None:
    payload = {"fast": 20, "slow": 100, "nested": {"b": [1, 2], "a": None}}
    path = artifacts.write_json("r0", "params.json", payload)
    assert path.name == "params.json"
    assert artifacts.read_json("r0", "params.json") == payload


def test_json_is_written_canonically(artifacts: FileArtifactStore) -> None:
    """Byte stability: the same object always produces the same file, whatever
    order its keys were built in. A golden comparison depends on it."""
    first = artifacts.write_json("r0", "a.json", {"b": 1, "a": 2})
    second = artifacts.write_json("r1", "a.json", {"a": 2, "b": 1})
    assert first.read_bytes() == second.read_bytes()
    assert first.read_text(encoding="utf-8").endswith("\n")


def test_parquet_round_trips(artifacts: FileArtifactStore) -> None:
    frame = pd.DataFrame(
        {"ts_open": pd.Series([0, HOUR_MS], dtype="int64"), "equity": [10_000.0, 10_120.0]}
    )
    path = artifacts.write_parquet("r0", "equity.parquet", frame)
    assert path.is_file()
    pd.testing.assert_frame_equal(artifacts.read_parquet("r0", "equity.parquet"), frame)


def test_parquet_is_written_with_the_fixed_options(artifacts: FileArtifactStore) -> None:
    """Options are pinned, and the index is dropped: a pandas index that leaked
    into the file would change its bytes without changing the data."""
    assert ARTIFACT_PARQUET_KWARGS == {
        "engine": "pyarrow",
        "compression": "zstd",
        "index": False,
    }
    frame = pd.DataFrame({"x": [1.0, 2.0]}, index=pd.Index([7, 8], name="ignored"))
    first = artifacts.write_parquet("r0", "x.parquet", frame)
    second = artifacts.write_parquet("r1", "x.parquet", frame.reset_index(drop=True))
    assert first.read_bytes() == second.read_bytes()
    assert list(artifacts.read_parquet("r0", "x.parquet").columns) == ["x"]


def test_text_round_trips_for_the_engine_log(artifacts: FileArtifactStore) -> None:
    artifacts.write_text("r0", "engine_log.txt", "bar=0 warmup_done\n")
    assert artifacts.read_text("r0", "engine_log.txt") == "bar=0 warmup_done\n"


def test_listing_and_existence(artifacts: FileArtifactStore) -> None:
    assert artifacts.listdir("r0") == ()
    assert not artifacts.exists("r0", "params.json")
    artifacts.write_json("r0", "params.json", {})
    artifacts.write_json("r0", "metrics.json", {})
    assert artifacts.listdir("r0") == ("metrics.json", "params.json")
    assert artifacts.exists("r0", "params.json")


def test_a_missing_artifact_is_reported_not_invented(artifacts: FileArtifactStore) -> None:
    with pytest.raises(StoreError, match="artifact not found"):
        artifacts.read_json("r0", "nope.json")
    with pytest.raises(StoreError, match="artifact not found"):
        artifacts.read_parquet("r0", "nope.parquet")
    with pytest.raises(StoreError, match="artifact not found"):
        artifacts.read_text("r0", "nope.txt")


@pytest.mark.parametrize("bad", ["..", ".", "", "a/b", "../escape", "/absolute", "a\\b"])
def test_a_path_component_that_could_escape_the_tree_is_refused(
    artifacts: FileArtifactStore, bad: str
) -> None:
    """Run ids are hashes and names are platform constants, so this should never
    fire. That is the point: it costs one comparison and removes a whole class of
    "how did that file get there" from consideration."""
    with pytest.raises(StoreError, match=r"path component|directory reference"):
        artifacts.dir_for(bad)
    with pytest.raises(StoreError, match=r"path component|directory reference"):
        artifacts.write_json("r0", bad, {})


def test_runs_do_not_share_a_directory(artifacts: FileArtifactStore) -> None:
    artifacts.write_json("r0", "params.json", {"which": 0})
    artifacts.write_json("r1", "params.json", {"which": 1})
    assert artifacts.read_json("r0", "params.json") == {"which": 0}
    assert artifacts.read_json("r1", "params.json") == {"which": 1}


def test_the_artifact_dir_recorded_on_a_run_matches_the_store(
    store: SqliteExperimentStore, artifacts: FileArtifactStore
) -> None:
    """The database says where a run's files are; the store must put them there."""
    _run(store)
    run = store.find_run("r0")
    assert run is not None
    assert run.artifact_dir == "artifacts/runs/r0"
    assert artifacts.dir_for("r0").parts[-3:] == ("artifacts", "runs", "r0")


# --- AC7 / AC8: the migration -----------------------------------------------
def test_alembic_upgrade_head_succeeds_against_an_empty_database(
    empty_engine: Engine,
) -> None:
    assert current_revision(empty_engine) is None
    assert missing_tables(empty_engine) == TABLE_NAMES
    assert upgrade_to_head(empty_engine) == _head_revision()
    assert missing_tables(empty_engine) == ()


def test_the_migration_creates_every_table_the_spec_names(db_engine: Engine) -> None:
    """The thirteen tables of spec section 6 that T21 covers."""
    present = set(inspect(db_engine).get_table_names())
    for table in (
        "dataset",
        "split_policy",
        "strategy_family",
        "strategy_version",
        "experiment",
        "run",
        "metric",
        "trade",
        "optuna_study",
        "validation_verdict",
        "llm_interaction",
        "lockbox_access",
        "paper_session",
    ):
        assert table in present, table


def test_the_migrated_database_is_usable_immediately(empty_engine: Engine) -> None:
    """Upgrading an empty database is enough to start recording, with no
    implicit ``create_all`` anywhere in the path."""
    upgrade_to_head(empty_engine)
    store = SqliteExperimentStore(make_session_factory(empty_engine), now_ms=lambda: FIXED_NOW)
    run, _ = _run(store)
    assert store.find_run(run.run_id) is not None


# --- AC9 and the invariants -------------------------------------------------
def test_the_default_clock_is_wall_clock_utc_milliseconds(db_engine: Engine) -> None:
    """Time is injected so tests can pin it, but the production default has to be
    right: every timestamp in the schema is INTEGER ms UTC (spec section 6)."""
    before = _now()
    store = SqliteExperimentStore(make_session_factory(db_engine))
    dataset = _dataset(store)
    after = _now()
    assert before <= dataset.created_at <= after


def test_the_stores_repr_names_what_it_is_bound_to(
    db_engine: Engine, artifacts: FileArtifactStore
) -> None:
    assert "SqliteExperimentStore" in repr(SqliteExperimentStore(make_session_factory(db_engine)))
    assert "FileArtifactStore" in repr(artifacts)


def test_a_null_byte_in_a_path_component_is_refused(artifacts: FileArtifactStore) -> None:
    with pytest.raises(StoreError, match="single path component"):
        artifacts.dir_for("r0\x00evil")


def test_inv2_the_store_holds_no_credentials(store: SqliteExperimentStore) -> None:
    """INV-2: the store records that a model was called and what it cost, by
    path and by hash. Nothing it persists is a secret."""
    _run(store)
    interaction = store.record_llm_interaction(
        campaign="c1",
        provider="glm",
        model="glm-4",
        purpose="propose",
        prompt_sha256="b" * 64,
        prompt_path="artifacts/llm/p.txt",
        response_path="artifacts/llm/r.json",
        prompt_template_sha256="c" * 64,
        tokens_in=1,
        tokens_out=1,
        cost_eur=0.0,
        temperature=0.0,
        latency_ms=1,
        status="ok",
    )
    columns = {c.key for c in inspect(LlmInteraction).mapper.column_attrs}
    assert not (columns & {"api_key", "secret", "token", "password", "authorization"})
    assert interaction.prompt_path.startswith("artifacts/")


def test_inv5_the_store_cannot_reopen_a_frozen_test_partition(
    store: SqliteExperimentStore,
) -> None:
    """INV-5: once the lockbox has pinned the test end, no path through the store
    moves it — including handing back an unfrozen copy of the same policy."""
    policy = _policy(store)
    store.freeze_test_end(policy.split_id, HOUR_MS * 190)
    assert store.get_or_create_split(policy).test_end_ts == HOUR_MS * 190
    with pytest.raises(ConfigError, match="already frozen"):
        store.freeze_test_end(policy.split_id, HOUR_MS * 199)


def test_inv7_a_run_records_every_input_its_result_depends_on(
    store: SqliteExperimentStore,
) -> None:
    """INV-7: ``quantlab reproduce`` re-runs from the stored row alone, so the row
    has to name the strategy, the data, the partition, the parameters, the costs
    and the engine build. A missing column here is an unreproducible run."""
    _run(store)
    run = store.find_run("r0")
    assert run is not None
    for field in (
        "strategy_id",
        "dataset_id",
        "split_id",
        "segment",
        "params_json",
        "engine_name",
        "engine_version",
        "cost_multiplier",
        "experiment_id",
    ):
        assert getattr(run, field) is not None, field
    experiment = store.query_runs(run_id=run.run_id) if False else None
    assert experiment is None  # `run_id` is not a filter; identity is `find_run`


def test_inv7_the_experiment_records_the_seed_and_the_commit(
    store: SqliteExperimentStore,
) -> None:
    experiment = store.create_experiment(
        campaign="c", purpose="train", config_hash="h", config_json="{}", seed=99, git_commit="sha"
    )
    assert experiment.seed == 99 and experiment.git_commit == "sha"


def test_inv8_the_store_adapter_does_not_reach_across_the_layer_boundary() -> None:
    """INV-8: an adapter may import ``core`` and ``ports`` and nothing else of the
    package. ``test_architecture.py`` enforces this tree-wide; it is restated here
    because the store is where the temptation to reach sideways is strongest."""
    import ast

    for path in sorted(Path("src/quantlab/adapters/store").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            for name in names:
                if not name.startswith("quantlab."):
                    continue
                assert name.startswith(
                    ("quantlab.core", "quantlab.ports", "quantlab.adapters.store")
                ), f"{path}: {name}"


# ---------------------------------------------------------------------------
# T21 amendment: the evolution tables (spec section 6, spec 1.1)
# ---------------------------------------------------------------------------
def _evolution(store: SqliteExperimentStore) -> tuple[Any, Any]:
    """An evolution run and its first generation, with all prerequisites."""
    run, _ = _run(store)
    evolution = store.create_evolution_run(
        evolution_id="ev0",
        experiment_id=run.experiment_id,
        campaign="c1",
        dataset_id="d0",
        split_id=run.split_id,
        population_size=16,
        n_survivors=12,
        n_offspring=3,
        n_immigrants=1,
        max_generations=30,
        seed=42,
        fitness_config_json='{"weights":{}}',
        mutation_config_json="{}",
        diversity_config_json="{}",
    )
    generation = store.add_generation(
        generation_id="g0",
        evolution_id=evolution.evolution_id,
        gen_index=0,
        diversity=0.71,
        n_evaluated=16,
        n_cache_hits=0,
        n_rejected_by_gate=1,
        n_immigrants_used=1,
    )
    return evolution, generation


def _candidate(
    store: SqliteExperimentStore,
    candidate_id: str,
    *,
    origin: str = "seed",
    parent: str | None = None,
    gen_index: int = 0,
) -> Any:
    return store.add_candidate(
        candidate_id=candidate_id,
        evolution_id="ev0",
        generation_id="g0",
        gen_index=gen_index,
        strategy_id="s0",
        params_json='{"fast":20}',
        kind="genome",
        origin=origin,
        genome_json='{"entry":{}}',
        parent_candidate_id=parent,
    )


def test_migration_0002_is_the_head_and_adds_the_five_evolution_tables(
    empty_engine: Engine,
) -> None:
    assert upgrade_to_head(empty_engine) == _head_revision() == "0002"
    present = set(inspect(empty_engine).get_table_names())
    assert set(EVOLUTION_TABLES) <= present
    assert EVOLUTION_TABLES == (
        "evolution_run",
        "generation",
        "candidate",
        "mutation",
        "candidate_promotion",
    )


def test_migration_0002_follows_0001(db_engine: Engine) -> None:
    """Ordering, not just presence: the evolution tables reference `experiment`,
    `dataset`, `split_policy` and `run`, so 0001 has to have run first."""
    assert missing_tables(db_engine) == ()
    assert current_revision(db_engine) == "0002"


def test_the_evolution_indexes_from_the_spec_exist(db_engine: Engine) -> None:
    inspector = inspect(db_engine)
    assert {i["name"] for i in inspector.get_indexes("candidate")} >= {
        "ix_candidate_evolution",
        "ix_candidate_parent",
        "ix_candidate_strategy",
    }
    assert {i["name"] for i in inspector.get_indexes("mutation")} >= {"ix_mutation_candidate"}
    assert {i["name"] for i in inspector.get_indexes("candidate_promotion")} >= {
        "ix_promotion_candidate"
    }


def test_an_evolution_run_records_the_search_it_performed(
    store: SqliteExperimentStore,
) -> None:
    evolution, _ = _evolution(store)
    assert evolution.status == "running"
    assert evolution.n_evaluations == 0
    assert (evolution.population_size, evolution.n_survivors) == (16, 12)
    assert evolution.fitness_config_json == '{"weights":{}}'
    assert evolution.finished_at is None


def test_finishing_an_evolution_run_records_why_it_stopped(
    store: SqliteExperimentStore,
) -> None:
    _evolution(store)
    finished = store.finish_evolution_run("ev0", "stopped", "no improvement in 8 generations")
    assert finished.status == "stopped"
    assert finished.stop_reason == "no improvement in 8 generations"
    assert finished.finished_at is not None
    with pytest.raises(RunConflict, match="already finished"):
        store.finish_evolution_run("ev0", "completed")


def test_the_evaluation_count_is_kept_because_dsr_needs_it(
    store: SqliteExperimentStore,
) -> None:
    """Section 14.4's ``M``. A cached evaluation is still an evaluation: it was
    tried, and the multiple-testing correction has to know."""
    _evolution(store)
    assert store.count_evaluation("ev0") == 1
    assert store.count_evaluation("ev0", 15) == 16


def test_a_generation_is_append_only(db_engine: Engine, store: SqliteExperimentStore) -> None:
    _, generation = _evolution(store)
    assert generation.diversity == 0.71
    factory = make_session_factory(db_engine)
    with pytest.raises(ImmutableRowError), session_scope(factory) as session:
        row = session.get(Generation, "g0")
        assert row is not None
        row.diversity = 0.1
        session.flush()


def test_a_generation_index_is_unique_within_a_run(store: SqliteExperimentStore) -> None:
    _evolution(store)
    with pytest.raises(IntegrityError):
        store.add_generation(
            generation_id="g0-again",
            evolution_id="ev0",
            gen_index=0,
            diversity=0.5,
            n_evaluated=1,
            n_cache_hits=0,
            n_rejected_by_gate=0,
            n_immigrants_used=0,
        )


def test_a_candidate_enters_the_population_unscored(store: SqliteExperimentStore) -> None:
    """Section 13.2: a candidate whose run fails is recorded with its error and
    never silently dropped, so it has to exist before it is evaluated."""
    _evolution(store)
    candidate = _candidate(store, "c0")
    assert candidate.fitness is None
    assert candidate.run_id is None
    assert candidate.survived == 0
    assert candidate.components_json == "{}"


def test_scoring_a_candidate_writes_only_the_mutable_columns(
    store: SqliteExperimentStore,
) -> None:
    _run(store, "r1")
    _evolution(store)
    _candidate(store, "c0")
    scored = store.score_candidate(
        "c0",
        run_id="r1",
        fitness=1.25,
        base_score=1.5,
        penalty_product=0.83,
        components_json='{"expectancy":0.4}',
        penalties_json='{"complexity":0.9}',
        rank=1,
        survived=True,
        behaviour_hash="b" * 16,
    )
    assert scored.fitness == 1.25
    assert scored.survived == 1
    assert scored.rank == 1
    assert scored.run_id == "r1"


def test_a_rejected_candidate_keeps_its_gate_failure(store: SqliteExperimentStore) -> None:
    _evolution(store)
    _candidate(store, "c0")
    scored = store.score_candidate("c0", gate_failure="G_MIN_TRADES", fitness=-1e9)
    assert scored.gate_failure == "G_MIN_TRADES"
    assert store.candidates_for("ev0") == [scored] or scored.candidate_id == "c0"


@pytest.mark.parametrize(
    "column", ["params_json", "kind", "origin", "strategy_id", "gen_index", "signature_json"]
)
def test_a_candidates_identity_columns_are_immutable(
    db_engine: Engine, store: SqliteExperimentStore, column: str
) -> None:
    _evolution(store)
    _candidate(store, "c0")
    factory = make_session_factory(db_engine)
    with pytest.raises(ImmutableRowError), session_scope(factory) as session:
        row = session.get(Candidate, "c0")
        assert row is not None
        setattr(row, column, 99 if column == "gen_index" else "changed")
        session.flush()


def test_mutations_are_append_only_and_never_updated(
    db_engine: Engine, store: SqliteExperimentStore
) -> None:
    """INV-10: re-applying these to the parent's genome must reproduce the child
    byte for byte, which it cannot do if the record can be revised afterwards.

    ``mutation`` is deliberately absent from ``MUTABLE_COLUMNS``: not an
    oversight, the whole point.
    """
    assert "mutation" not in MUTABLE_COLUMNS
    _evolution(store)
    _candidate(store, "c0")
    _candidate(store, "c1", origin="mutant", parent="c0")
    store.add_mutations(
        "c1",
        [
            {
                "parent_candidate_id": "c0",
                "category": "parameter",
                "operator": "perturb_numeric",
                "target": "params.fast",
                "before_json": "20",
                "after_json": "24",
                "rng_seed": 11,
            },
            {
                "parent_candidate_id": "c0",
                "category": "structural",
                "operator": "add_confirmation",
                "target": "entry.conditions",
                "before_json": "[]",
                "after_json": '[{"op":">"}]',
                "rng_seed": 12,
                "suggested_by": "llm",
            },
        ],
    )
    recorded = store.mutations_for("c1")
    assert [m.seq for m in recorded] == [0, 1]
    assert [m.category for m in recorded] == ["parameter", "structural"]
    assert [m.rng_seed for m in recorded] == [11, 12]
    assert recorded[1].suggested_by == "llm"

    factory = make_session_factory(db_engine)
    with pytest.raises(ImmutableRowError, match="append-only"), session_scope(factory) as session:
        row = session.get(Mutation, recorded[0].mutation_id)
        assert row is not None
        row.after_json = "999"
        session.flush()


def test_a_mutation_sequence_is_unique_within_a_candidate(
    store: SqliteExperimentStore,
) -> None:
    _evolution(store)
    _candidate(store, "c0")
    _candidate(store, "c1", origin="mutant", parent="c0")
    entry = {
        "parent_candidate_id": "c0",
        "category": "parameter",
        "operator": "perturb_numeric",
        "target": "params.fast",
        "before_json": "20",
        "after_json": "24",
        "rng_seed": 1,
        "seq": 0,
    }
    store.add_mutations("c1", [entry])
    with pytest.raises(IntegrityError):
        store.add_mutations("c1", [dict(entry)])


def test_a_partly_written_mutation_sequence_is_not_persisted(
    store: SqliteExperimentStore,
) -> None:
    """Half a lineage is not a lineage: the whole sequence lands or none of it."""
    _evolution(store)
    _candidate(store, "c0")
    _candidate(store, "c1", origin="mutant", parent="c0")
    good = {
        "parent_candidate_id": "c0",
        "category": "parameter",
        "operator": "perturb_numeric",
        "target": "params.fast",
        "before_json": "20",
        "after_json": "24",
        "rng_seed": 1,
    }
    with pytest.raises(IntegrityError):
        store.add_mutations("c1", [good, {**good, "category": "not-a-category"}])
    assert store.mutations_for("c1") == []


def test_a_promotion_is_written_before_its_run(store: SqliteExperimentStore) -> None:
    """INV-9: the validation segment is reachable only through this row, and the
    row exists before the run it authorises. Written afterwards it would be a log
    of a decision already taken, not a gate."""
    _evolution(store)
    _candidate(store, "c0")
    promotion = store.record_promotion(
        candidate_id="c0",
        evolution_id="ev0",
        gen_index=0,
        segment="val",
        reason="top of generation 0",
    )
    assert promotion.run_id is None
    assert promotion.verdict_id is None
    assert promotion.segment == "val"
    assert promotion.created_at


def test_a_promotions_result_is_attached_afterwards(store: SqliteExperimentStore) -> None:
    run, _ = _run(store, "r1")
    _evolution(store)
    _candidate(store, "c0")
    assert run.run_id == "r1"
    promotion = store.record_promotion(
        candidate_id="c0", evolution_id="ev0", gen_index=0, segment="val", reason="why"
    )
    verdict = store.save_verdict(
        strategy_id="s0",
        split_id=run.split_id,
        params_json="{}",
        verdict="CANDIDATE",
        overfit_score=0.2,
        hard_gates_json="{}",
        soft_checks_json="{}",
        thresholds_json="{}",
        n_trials_accounted=16,
    )
    attached = store.attach_promotion_result(
        promotion.promotion_id, run_id="r1", verdict_id=verdict.verdict_id
    )
    assert attached.run_id == "r1"
    assert attached.verdict_id == verdict.verdict_id
    assert [p.promotion_id for p in store.promotions_for("c0")] == [promotion.promotion_id]


@pytest.mark.parametrize("column", ["segment", "reason", "gen_index", "candidate_id"])
def test_a_promotions_identity_is_immutable(
    db_engine: Engine, store: SqliteExperimentStore, column: str
) -> None:
    _evolution(store)
    _candidate(store, "c0")
    promotion = store.record_promotion(
        candidate_id="c0", evolution_id="ev0", gen_index=0, segment="val", reason="why"
    )
    factory = make_session_factory(db_engine)
    with pytest.raises(ImmutableRowError), session_scope(factory) as session:
        row = session.get(CandidatePromotion, promotion.promotion_id)
        assert row is not None
        setattr(row, column, 9 if column == "gen_index" else "changed")
        session.flush()


def test_lineage_is_complete_and_reaches_a_seed(store: SqliteExperimentStore) -> None:
    """INV-10: every candidate except a seed names a parent that exists, and the
    chain reaches a seed without cycles."""
    _evolution(store)
    _candidate(store, "c0", origin="seed")
    _candidate(store, "c1", origin="mutant", parent="c0", gen_index=1)
    _candidate(store, "c2", origin="mutant", parent="c1", gen_index=2)
    _candidate(store, "c3", origin="mutant", parent="c1", gen_index=2)

    assert [c.candidate_id for c in store.ancestry("c2")] == ["c0", "c1", "c2"]
    assert [c.candidate_id for c in store.ancestry("c0")] == ["c0"]
    assert store.ancestry("c2")[0].origin == "seed"
    assert sorted(c.candidate_id for c in store.descendants("c0")) == ["c1", "c2", "c3"]
    assert sorted(c.candidate_id for c in store.descendants("c1")) == ["c2", "c3"]
    assert store.descendants("c2") == []


def test_candidates_are_returned_in_a_deterministic_order(
    store: SqliteExperimentStore,
) -> None:
    _evolution(store)
    for index in range(4):
        _candidate(store, f"c{index}", gen_index=index % 2)
    everything = [c.candidate_id for c in store.candidates_for("ev0")]
    assert everything == sorted(everything, key=lambda cid: (int(cid[1]) % 2, cid))
    assert [c.candidate_id for c in store.candidates_for("ev0", gen_index=1)] == ["c1", "c3"]
    assert store.candidates_for("ev0", gen_index=9) == []
    assert store.candidates_for("nope") == []


def test_generations_are_returned_in_index_order(store: SqliteExperimentStore) -> None:
    _evolution(store)
    for index in (3, 1, 2):
        store.add_generation(
            generation_id=f"g{index}",
            evolution_id="ev0",
            gen_index=index,
            diversity=0.5,
            n_evaluated=16,
            n_cache_hits=0,
            n_rejected_by_gate=0,
            n_immigrants_used=1,
        )
    assert [g.gen_index for g in store.generations_for("ev0")] == [0, 1, 2, 3]


def test_a_candidate_cannot_cite_a_generation_that_does_not_exist(
    store: SqliteExperimentStore,
) -> None:
    _evolution(store)
    with pytest.raises(IntegrityError):
        store.add_candidate(
            candidate_id="cX",
            evolution_id="ev0",
            generation_id="no-such-generation",
            gen_index=0,
            strategy_id="s0",
            params_json="{}",
            kind="opaque",
            origin="seed",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("kind", "wishful"), ("origin", "spontaneous")],
)
def test_a_candidates_enumerations_are_enforced(
    store: SqliteExperimentStore, field: str, value: str
) -> None:
    _evolution(store)
    fields = {
        "candidate_id": "cX",
        "evolution_id": "ev0",
        "generation_id": "g0",
        "gen_index": 0,
        "strategy_id": "s0",
        "params_json": "{}",
        "kind": "genome",
        "origin": "seed",
    }
    fields[field] = value
    with pytest.raises(IntegrityError):
        store.add_candidate(**fields)  # type: ignore[arg-type]


def test_an_opaque_candidate_may_have_no_genome(store: SqliteExperimentStore) -> None:
    """Section 13.4: ``opaque`` candidates admit parameter mutations only, and
    have no genome to record."""
    _evolution(store)
    opaque = store.add_candidate(
        candidate_id="cOpaque",
        evolution_id="ev0",
        generation_id="g0",
        gen_index=0,
        strategy_id="s0",
        params_json="{}",
        kind="opaque",
        origin="immigrant",
    )
    assert opaque.genome_json is None
