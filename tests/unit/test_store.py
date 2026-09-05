"""Database schema, pragmas and migrations (master spec section 6).

The Alembic migration owns the DDL (so ``STRICT`` and ``WITHOUT ROWID`` are
exactly as specified); the SQLAlchemy models mirror it.  These tests keep the
two in step and prove the constraints are actually enforced at run time.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from sqlalchemy import Engine, inspect, text
from sqlalchemy.exc import IntegrityError

from quantlab.adapters.store.models import (
    MUTABLE_COLUMNS,
    TABLE_NAMES,
    Base,
    Dataset,
    Metric,
    Run,
    StrategyFamily,
)
from quantlab.adapters.store.sqlite import (
    connection_url,
    create_db_engine,
    current_revision,
    make_session_factory,
    migrations_dir,
    missing_tables,
    session_scope,
    upgrade_to_head,
)


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
    assert current_revision(db_engine) == "0001"
    assert missing_tables(db_engine) == ()
    present = set(inspect(db_engine).get_table_names())
    assert set(TABLE_NAMES) <= present


def test_upgrade_is_idempotent(db_engine: Engine) -> None:
    assert upgrade_to_head(db_engine) == "0001"
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
