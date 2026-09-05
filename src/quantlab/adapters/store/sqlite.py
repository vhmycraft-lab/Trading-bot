"""SQLite engine, pragmas and migration plumbing (master spec sections 6 and 21).

Every connection this module hands out has:

* ``PRAGMA foreign_keys = ON``     — referential integrity is enforced, not assumed
* ``PRAGMA journal_mode = WAL``    — concurrent readers while a run writes
* ``PRAGMA synchronous = NORMAL``  — durable enough with WAL, much faster
* ``PRAGMA busy_timeout = 5000``   — a parallel run waits instead of failing

The schema itself is owned by Alembic (``migrations/``); this module never
creates tables implicitly, so a database is always at a known revision.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, TypeVar

from sqlalchemy import Engine, create_engine, delete, event, inspect, select, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.orm.state import InstanceState

from quantlab.adapters.store.models import (
    MUTABLE_COLUMNS,
    TABLE_NAMES,
    Dataset,
    Experiment,
    LlmInteraction,
    LockboxAccess,
    Metric,
    OptunaStudy,
    PaperSession,
    Run,
    StrategyFamily,
    StrategyVersion,
    Trade,
    ValidationVerdict,
)
from quantlab.adapters.store.models import SplitPolicy as SplitPolicyRow
from quantlab.core.errors import ConfigError, ImmutableRowError, RunConflict, StoreError
from quantlab.core.hashing import canonical_json, short_id
from quantlab.core.splits import SplitPolicy, parse_split_policy
from quantlab.core.types import Trade as TradeRecord

__all__ = [
    "ALEMBIC_VERSION_TABLE",
    "IN_MEMORY_URL",
    "RUN_FILTERS",
    "SqliteExperimentStore",
    "connection_url",
    "create_db_engine",
    "current_revision",
    "enforce_append_only",
    "guard_is_installed",
    "install_append_only_guard",
    "make_session_factory",
    "migrations_dir",
    "missing_tables",
    "session_scope",
    "upgrade_to_head",
]

log = logging.getLogger(__name__)

ALEMBIC_VERSION_TABLE: Final[str] = "alembic_version"
IN_MEMORY_URL: Final[str] = "sqlite+pysqlite:///:memory:"

#: Escape hatch for installed (non-editable) deployments.
MIGRATIONS_DIR_ENV: Final[str] = "QUANTLAB_MIGRATIONS_DIR"


def migrations_dir() -> Path:
    """Locate the Alembic ``migrations/`` directory.

    Order: ``QUANTLAB_MIGRATIONS_DIR``, then the repository root inferred from
    this file, then the current working directory.
    """
    candidates: list[Path] = []
    override = os.environ.get(MIGRATIONS_DIR_ENV)
    if override:
        candidates.append(Path(override))
    # src/quantlab/adapters/store/sqlite.py -> repo root is four parents up from src/
    candidates.append(Path(__file__).resolve().parents[4] / "migrations")
    candidates.append(Path.cwd() / "migrations")

    for candidate in candidates:
        if (candidate / "env.py").is_file():
            return candidate
    raise ConfigError(
        "could not locate the migrations directory; "
        f"set {MIGRATIONS_DIR_ENV} to the absolute path of migrations/",
        tried=[str(c) for c in candidates],
    )


def connection_url(db_path: str | Path) -> str:
    """Build a SQLAlchemy URL for ``db_path``.

    ``":memory:"`` is passed through so tests can run against a private database.
    """
    if str(db_path) == ":memory:":
        return IN_MEMORY_URL
    return f"sqlite+pysqlite:///{Path(db_path).as_posix()}"


def _install_pragmas(engine: Engine, *, wal: bool) -> None:
    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_connection: Any, _record: Any) -> None:
        if not isinstance(dbapi_connection, sqlite3.Connection):  # pragma: no cover
            return
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute("PRAGMA busy_timeout = 5000")
            if wal:
                cursor.execute("PRAGMA journal_mode = WAL")
                cursor.execute("PRAGMA synchronous = NORMAL")
        finally:
            cursor.close()


def create_db_engine(
    db_path: str | Path,
    *,
    echo: bool = False,
    create_parents: bool = True,
) -> Engine:
    """Create an :class:`~sqlalchemy.Engine` for ``db_path`` with the pragmas applied.

    WAL is skipped for in-memory databases, where it is meaningless.
    """
    is_memory = str(db_path) == ":memory:"
    if not is_memory and create_parents:
        parent = Path(db_path).resolve().parent
        parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(connection_url(db_path), echo=echo, future=True)
    _install_pragmas(engine, wal=not is_memory)
    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return a session factory bound to ``engine``, with the append-only guard."""
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    install_append_only_guard(factory)
    return factory


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Run a unit of work in one transaction: commit on success, roll back on error.

    Spec section 18.2: partial results are never persisted as ``ok``.
    """
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _quiet_alembic_plugin_chatter() -> None:
    """Silence the per-plugin INFO lines Alembic emits while importing.

    They arrive before any migration runs and say nothing a caller needs; the
    revision itself is reported by ``alembic.runtime.migration`` and by us.
    """
    logging.getLogger("alembic.runtime.plugins").setLevel(logging.WARNING)


def _alembic_config(engine: Engine) -> Any:
    _quiet_alembic_plugin_chatter()
    from alembic.config import Config  # local import: alembic is only needed here

    directory = migrations_dir()
    config = Config()
    config.set_main_option("script_location", str(directory))
    config.set_main_option("sqlalchemy.url", str(engine.url))
    return config


def upgrade_to_head(engine: Engine) -> str:
    """Apply every outstanding migration and return the resulting revision."""
    _quiet_alembic_plugin_chatter()
    from alembic import command

    config = _alembic_config(engine)
    # Hand the live engine to migrations/env.py so an in-memory database works.
    config.attributes["connection"] = engine
    command.upgrade(config, "head")
    revision = current_revision(engine)
    if revision is None:  # pragma: no cover - only if migrations are empty
        raise StoreError("alembic upgrade produced no revision")
    return revision


def current_revision(engine: Engine) -> str | None:
    """Return the Alembic revision stored in the database, or ``None`` if unmanaged."""
    with engine.connect() as connection:
        if not inspect(connection).has_table(ALEMBIC_VERSION_TABLE):
            return None
        row = connection.execute(text("SELECT version_num FROM alembic_version")).first()
    return None if row is None else str(row[0])


def missing_tables(engine: Engine) -> tuple[str, ...]:
    """Return the schema tables that do not exist yet, in declaration order."""
    with engine.connect() as connection:
        present = set(inspect(connection).get_table_names())
    return tuple(name for name in TABLE_NAMES if name not in present)


# ---------------------------------------------------------------------------
# append-only enforcement (spec section 6)
# ---------------------------------------------------------------------------
#: Key set on ``Session.info`` to authorise the one deletion the spec permits.
#: Not a secret — the guard it lifts is against accident, not against an attacker
#: who is already executing in this process.
_DELETION_FLAG: Final[str] = "quantlab_permitted_deletion"


def _changed_columns(state: InstanceState[Any]) -> tuple[str, ...]:
    """Names of the mapped columns whose value differs from what was loaded."""
    changed: list[str] = []
    for attribute in state.mapper.column_attrs:
        history = state.attrs[attribute.key].history
        if history.has_changes():
            changed.append(attribute.key)
    return tuple(sorted(changed))


def enforce_append_only(session: Session, *_args: Any) -> None:
    """Reject writes the spec's append-only rule does not allow.

    Registered on the session's ``before_flush``, so it applies to *any* code
    holding a session, not only to the store's own methods. That distinction is
    the whole value: a rule enforced solely inside the accessor methods is a
    convention, and the audit trail this database exists to be is worth more than
    a convention.

    Raises:
        ImmutableRowError: on an update to a column outside
            :data:`~quantlab.adapters.store.models.MUTABLE_COLUMNS`, or on any
            deletion not made through :meth:`SqliteExperimentStore.delete_run`.
    """
    for instance in session.dirty:
        if not session.is_modified(instance, include_collections=False):
            continue
        table = instance.__tablename__
        mutable = MUTABLE_COLUMNS.get(table, frozenset())
        frozen = tuple(name for name in _changed_columns(inspect(instance)) if name not in mutable)
        if frozen:
            raise ImmutableRowError(
                f"{table} rows are append-only; these columns may not be updated",
                table=table,
                columns=list(frozen),
                mutable=sorted(mutable),
            )

    if session.deleted and not session.info.get(_DELETION_FLAG):
        tables = sorted({instance.__tablename__ for instance in session.deleted})
        raise ImmutableRowError(
            "store rows may not be deleted; a failed run is removed with "
            "delete_run(run_id, confirm=True)",
            tables=tables,
        )


#: Marks a factory whose sessions already carry the guard.
_GUARD_INSTALLED: Final[str] = "_quantlab_append_only_guard"


def install_append_only_guard(factory: sessionmaker[Session]) -> None:
    """Attach :func:`enforce_append_only` to every session ``factory`` makes.

    Idempotence is tracked with an attribute on the factory rather than with
    ``event.contains``: SQLAlchemy's registry can report a listener on a freshly
    built ``sessionmaker`` that never registered one, because the entry belongs to
    a collected factory that occupied the same address. Trusting it meant the
    guard was silently *not* installed on roughly half of new factories — the
    failure mode of a safety check that is skipped rather than tripped, which is
    the one that does not announce itself.
    """
    if getattr(factory, _GUARD_INSTALLED, False):
        return
    event.listen(factory, "before_flush", enforce_append_only)
    setattr(factory, _GUARD_INSTALLED, True)


def guard_is_installed(factory: sessionmaker[Session]) -> bool:
    """Whether ``factory``'s sessions carry the append-only guard."""
    return bool(getattr(factory, _GUARD_INSTALLED, False))


@contextmanager
def _permit_deletion(session: Session) -> Iterator[None]:
    """Authorise deletions for the duration of one block."""
    session.info[_DELETION_FLAG] = True
    try:
        yield
    finally:
        session.info.pop(_DELETION_FLAG, None)


# ---------------------------------------------------------------------------
# the experiment store (spec section 11.1)
# ---------------------------------------------------------------------------
#: Columns ``query_runs`` accepts as filters. An unknown keyword is an error
#: rather than a silently ignored filter that returns too many rows.
RUN_FILTERS: Final[frozenset[str]] = frozenset(
    {"experiment_id", "strategy_id", "dataset_id", "split_id", "segment", "status"}
)

_TERMINAL_RUN_STATUS: Final[frozenset[str]] = frozenset({"ok", "failed", "rejected"})


def _wall_clock_ms() -> int:
    """Default clock. Injected rather than read ambiently, so tests can pin it."""
    return int(datetime.now(UTC).timestamp() * 1000)


class SqliteExperimentStore:
    """The append-only record of everything an experiment depended on.

    Implements :class:`quantlab.ports.store.ExperimentStore`. It stores; it does
    not decide. Run identity (§11.2), metric values (§10) and verdicts (§14) are
    computed by the modules that own those rules and arrive here already formed —
    a store that recomputed them would become a second, divergent implementation
    of the thing it is meant to be evidence for.

    Every method is one transaction. A partial write is never visible: §18.2
    requires that nothing is recorded as ``ok`` unless all of it succeeded.
    """

    __slots__ = ("_now", "factory")

    def __init__(
        self,
        factory: sessionmaker[Session],
        *,
        now_ms: Callable[[], int] = _wall_clock_ms,
    ) -> None:
        self.factory = factory
        self._now = now_ms
        install_append_only_guard(factory)

    def __repr__(self) -> str:
        return f"SqliteExperimentStore(bind={self.factory.kw.get('bind')!r})"

    # -- datasets and splits ------------------------------------------------
    def get_or_create_dataset(
        self,
        *,
        dataset_id: str,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_ts: int,
        end_ts: int,
        n_bars: int,
        manifest_json: str,
    ) -> Dataset:
        """Register a dataset, returning the existing row if the id is known.

        Raises:
            RunConflict: the id is known but describes different data. A dataset
                id is a hash of its manifest, so this means two different datasets
                hashed the same, or a row was edited outside the store. Either
                way, silently returning the old row would corrupt every run that
                cites it.
        """
        with session_scope(self.factory) as session:
            existing = session.get(Dataset, dataset_id)
            if existing is not None:
                _require_same(
                    "dataset",
                    dataset_id,
                    existing,
                    {
                        "exchange": exchange,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "start_ts": int(start_ts),
                        "end_ts": int(end_ts),
                        "n_bars": int(n_bars),
                    },
                )
                return existing
            row = Dataset(
                dataset_id=dataset_id,
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                start_ts=int(start_ts),
                end_ts=int(end_ts),
                n_bars=int(n_bars),
                manifest_json=manifest_json,
                created_at=self._now(),
            )
            session.add(row)
            return row

    def get_or_create_split(self, policy: SplitPolicy) -> SplitPolicy:
        """Register a split policy, returning the stored one if it already exists.

        The stored row wins on ``test_end_ts``: once the lockbox has pinned the
        test end, a caller arriving with an unfrozen copy of the same policy must
        be handed the frozen one, never allowed to reopen it (INV-5).
        """
        with session_scope(self.factory) as session:
            existing = session.get(SplitPolicyRow, policy.split_id)
            if existing is not None:
                return _policy_from_row(existing)
            session.add(
                SplitPolicyRow(
                    split_id=policy.split_id,
                    dataset_id=policy.dataset_id,
                    train_start_ts=policy.train_start_ts,
                    train_end_ts=policy.train_end_ts,
                    val_start_ts=policy.val_start_ts,
                    val_end_ts=policy.val_end_ts,
                    test_start_ts=policy.test_start_ts,
                    test_end_ts=policy.test_end_ts,
                    embargo_bars=policy.embargo_bars,
                    wf_json=_policy_document(policy),
                    created_at=self._now(),
                )
            )
            return policy

    def freeze_test_end(self, split_id: str, end_ts: int) -> SplitPolicy:
        """Pin a split's open test end the first time the lockbox is used.

        Raises:
            ConfigError: the end is already frozen to a different value, or lies
                before the test start. The whole point of the lockbox is that the
                final partition cannot move once it has been looked at (INV-5).
        """
        with session_scope(self.factory) as session:
            row = self._require_row(session, SplitPolicyRow, split_id, "split_policy")
            frozen = _policy_from_row(row).freeze_test_end(int(end_ts))
            row.test_end_ts = frozen.test_end_ts
            return frozen

    # -- strategies ---------------------------------------------------------
    def create_family(
        self, *, name: str, origin: str, description: str = "", family_id: str | None = None
    ) -> StrategyFamily:
        """Register a strategy family. Returns the existing row for a known name."""
        identifier = family_id or short_id(f"family|{name}")
        with session_scope(self.factory) as session:
            existing = session.get(StrategyFamily, identifier)
            if existing is not None:
                _require_same("strategy_family", identifier, existing, {"name": name})
                return existing
            row = StrategyFamily(
                family_id=identifier,
                name=name,
                origin=origin,
                description=description,
                status="open",
                validation_touches=0,
                created_at=self._now(),
            )
            session.add(row)
            return row

    def add_strategy_version(
        self,
        *,
        strategy_id: str,
        family_id: str,
        code_path: str,
        code_sha256: str,
        class_name: str,
        param_schema_json: str,
        style: str,
        author: str,
        logic_lines: int,
        parent_strategy_id: str | None = None,
        llm_interaction_id: str | None = None,
    ) -> StrategyVersion:
        """Record one immutable version of a strategy's source.

        Returns the existing row when the id is already known: ``strategy_id`` is
        ``sha256(code)[:16]``, so the same code is the same version and loading it
        twice is not a conflict.
        """
        with session_scope(self.factory) as session:
            existing = session.get(StrategyVersion, strategy_id)
            if existing is not None:
                _require_same(
                    "strategy_version",
                    strategy_id,
                    existing,
                    {"family_id": family_id, "code_sha256": code_sha256},
                )
                return existing
            row = StrategyVersion(
                strategy_id=strategy_id,
                family_id=family_id,
                parent_strategy_id=parent_strategy_id,
                code_path=code_path,
                code_sha256=code_sha256,
                class_name=class_name,
                param_schema_json=param_schema_json,
                style=style,
                author=author,
                llm_interaction_id=llm_interaction_id,
                logic_lines=int(logic_lines),
                created_at=self._now(),
            )
            session.add(row)
            return row

    def lineage(self, strategy_id: str) -> list[StrategyVersion]:
        """Every ancestor of ``strategy_id``, oldest first, ending with itself."""
        chain: list[StrategyVersion] = []
        seen: set[str] = set()
        with session_scope(self.factory) as session:
            current: StrategyVersion | None = self._require_row(
                session, StrategyVersion, strategy_id, "strategy_version"
            )
            # `seen` bounds the walk: a cycle cannot be created through the store,
            # but a lineage that never terminates would hang rather than report.
            while current is not None and current.strategy_id not in seen:
                chain.append(current)
                seen.add(current.strategy_id)
                parent = current.parent_strategy_id
                current = None if parent is None else session.get(StrategyVersion, parent)
        return list(reversed(chain))

    def increment_validation_touches(self, family_id: str) -> int:
        """Count one more look at the validation partition; return the new total.

        The counter is why §14.4 can put a number on how much searching a family
        has had, which is what the deflated Sharpe ratio needs.
        """
        with session_scope(self.factory) as session:
            row = self._require_row(session, StrategyFamily, family_id, "strategy_family")
            row.validation_touches += 1
            return int(row.validation_touches)

    def set_family_status(self, family_id: str, status: str) -> StrategyFamily:
        """Open, freeze or close a family."""
        with session_scope(self.factory) as session:
            row = self._require_row(session, StrategyFamily, family_id, "strategy_family")
            row.status = status
            return row

    # -- experiments and runs ----------------------------------------------
    def create_experiment(
        self,
        *,
        campaign: str,
        purpose: str,
        config_hash: str,
        config_json: str,
        seed: int,
        git_commit: str,
        experiment_id: str | None = None,
    ) -> Experiment:
        """Register an experiment: one campaign, one purpose, one configuration.

        The id defaults to a hash of exactly those inputs, so re-entering the same
        experiment finds the same row instead of forking the record.
        """
        identifier = experiment_id or short_id(
            f"experiment|{campaign}|{purpose}|{config_hash}|{seed}|{git_commit}"
        )
        with session_scope(self.factory) as session:
            existing = session.get(Experiment, identifier)
            if existing is not None:
                _require_same(
                    "experiment",
                    identifier,
                    existing,
                    {"campaign": campaign, "purpose": purpose, "config_hash": config_hash},
                )
                return existing
            row = Experiment(
                experiment_id=identifier,
                campaign=campaign,
                purpose=purpose,
                config_hash=config_hash,
                config_json=config_json,
                seed=int(seed),
                git_commit=git_commit,
                created_at=self._now(),
            )
            session.add(row)
            return row

    def find_run(self, run_id: str) -> Run | None:
        """The run with this id, or ``None``. The cache lookup of §11.2."""
        with session_scope(self.factory) as session:
            return session.get(Run, run_id)

    def create_run(
        self,
        *,
        run_id: str,
        experiment_id: str,
        strategy_id: str,
        dataset_id: str,
        split_id: str,
        segment: str,
        params_json: str,
        engine_name: str,
        engine_version: str,
        artifact_dir: str,
        cost_multiplier: float = 1.0,
    ) -> Run:
        """Open a run in ``pending``.

        ``run_id`` is supplied by the caller and is the hash of §11.2: identity is
        the runner's rule, not the store's, and a store that derived it here would
        be a second place for that rule to drift.

        Raises:
            RunConflict: the id exists with different inputs, which means the
                §11.2 hash and the stored row disagree — a defect worth stopping
                for, not a cache hit.
        """
        with session_scope(self.factory) as session:
            existing = session.get(Run, run_id)
            if existing is not None:
                _require_same(
                    "run",
                    run_id,
                    existing,
                    {
                        "experiment_id": experiment_id,
                        "strategy_id": strategy_id,
                        "dataset_id": dataset_id,
                        "split_id": split_id,
                        "segment": segment,
                        "params_json": params_json,
                        "engine_name": engine_name,
                        "engine_version": engine_version,
                    },
                )
                return existing
            row = Run(
                run_id=run_id,
                experiment_id=experiment_id,
                strategy_id=strategy_id,
                dataset_id=dataset_id,
                split_id=split_id,
                segment=segment,
                params_json=params_json,
                engine_name=engine_name,
                engine_version=engine_version,
                cost_multiplier=float(cost_multiplier),
                status="pending",
                error_json=None,
                artifact_dir=artifact_dir,
                started_at=None,
                finished_at=None,
                created_at=self._now(),
            )
            session.add(row)
            return row

    def start_run(self, run_id: str) -> Run:
        """Mark a run as running and stamp ``started_at``."""
        with session_scope(self.factory) as session:
            row = self._require_row(session, Run, run_id, "run")
            if row.status in _TERMINAL_RUN_STATUS:
                raise RunConflict(
                    "a finished run cannot be restarted", run_id=run_id, status=row.status
                )
            row.status = "running"
            row.started_at = self._now()
            return row

    def finish_run(
        self,
        run_id: str,
        status: str,
        metrics: Mapping[str, float | None] | None = None,
        trades: Sequence[TradeRecord] | None = None,
        error: Mapping[str, Any] | None = None,
    ) -> Run:
        """Close a run and write everything it produced, in one transaction.

        Metrics, trades and the terminal status land together or not at all. A run
        marked ``ok`` whose metrics did not commit would be the worst possible
        outcome here: the cache of §11.2 would serve it forever as a finished run
        with nothing in it.

        Raises:
            RunConflict: the run is already in a terminal state. Finishing twice
                would either duplicate its metrics or silently discard the second
                set, and neither is a thing an audit trail should do quietly.
        """
        with session_scope(self.factory) as session:
            row = self._require_row(session, Run, run_id, "run")
            if row.status in _TERMINAL_RUN_STATUS:
                raise RunConflict(
                    "run is already finished", run_id=run_id, status=row.status, requested=status
                )
            for name, value in (metrics or {}).items():
                session.add(
                    Metric(
                        run_id=run_id,
                        name=name,
                        value=None if value is None else float(value),
                    )
                )
            for trade in trades or ():
                session.add(_trade_row(run_id, trade))
            row.status = status
            row.error_json = None if error is None else canonical_json(dict(error))
            row.finished_at = self._now()
            return row

    def metrics_for(self, run_id: str) -> dict[str, float | None]:
        """Every metric recorded for a run, by name."""
        with session_scope(self.factory) as session:
            rows = session.execute(select(Metric).where(Metric.run_id == run_id)).scalars().all()
        return {row.name: row.value for row in sorted(rows, key=lambda r: r.name)}

    def trades_for(self, run_id: str) -> list[Trade]:
        """A run's trade ledger, in execution order."""
        with session_scope(self.factory) as session:
            rows = session.execute(select(Trade).where(Trade.run_id == run_id)).scalars().all()
        return sorted(rows, key=lambda r: r.trade_no)

    def query_runs(self, **filters: Any) -> list[Run]:
        """Runs matching every supplied filter, oldest first.

        Raises:
            StoreError: on an unknown filter name. A typo that silently widened
                the query would be reported as a result, not as a mistake.
        """
        unknown = sorted(set(filters) - RUN_FILTERS)
        if unknown:
            raise StoreError("unknown run filter", unknown=unknown, allowed=sorted(RUN_FILTERS))
        statement = select(Run)
        for name, value in sorted(filters.items()):
            statement = statement.where(getattr(Run, name) == value)
        with session_scope(self.factory) as session:
            rows = session.execute(statement).scalars().all()
        return sorted(rows, key=lambda r: (r.created_at, r.run_id))

    def delete_run(self, run_id: str, *, confirm: Literal[True]) -> None:
        """Remove a failed run and everything it produced (spec section 6).

        The only deletion this store permits, and it is narrow on purpose: a
        failed run has no results anyone can be citing, and clearing one lets the
        §11.2 cache re-execute it. Anything else stays, because a record that can
        be removed when it becomes inconvenient is not evidence.

        Logged at WARNING: deletions in an append-only store should be visible in
        the log without anyone having to go looking for them.

        Raises:
            ValueError: ``confirm`` was not ``True``.
            StoreError: the run did not fail, or something still references it.
        """
        if confirm is not True:
            raise ValueError("delete_run requires confirm=True")
        with session_scope(self.factory) as session:
            row = self._require_row(session, Run, run_id, "run")
            if row.status != "failed":
                raise StoreError(
                    "only a failed run may be deleted", run_id=run_id, status=row.status
                )
            referrers = _run_referrers(session, run_id)
            if referrers:
                raise StoreError(
                    "the run is referenced by rows that are themselves append-only",
                    run_id=run_id,
                    referenced_by=referrers,
                )
            with _permit_deletion(session):
                session.execute(delete(Metric).where(Metric.run_id == run_id))
                session.execute(delete(Trade).where(Trade.run_id == run_id))
                session.delete(row)
                session.flush()
        log.warning(
            "deleted failed run",
            extra={"run_id": run_id, "artifact_dir": row.artifact_dir},
        )

    # -- verdicts, LLM calls, lockbox --------------------------------------
    def save_verdict(
        self,
        *,
        strategy_id: str,
        split_id: str,
        params_json: str,
        verdict: str,
        overfit_score: float,
        hard_gates_json: str,
        soft_checks_json: str,
        thresholds_json: str,
        n_trials_accounted: int,
        verdict_id: str | None = None,
    ) -> ValidationVerdict:
        """Record a validation verdict exactly as it was decided.

        The thresholds are stored with it: a verdict read a year later has to be
        interpretable against the rules that produced it, not against whatever
        the configuration says by then.
        """
        with session_scope(self.factory) as session:
            row = ValidationVerdict(
                verdict_id=verdict_id or _event_id(),
                strategy_id=strategy_id,
                split_id=split_id,
                params_json=params_json,
                verdict=verdict,
                overfit_score=float(overfit_score),
                hard_gates_json=hard_gates_json,
                soft_checks_json=soft_checks_json,
                thresholds_json=thresholds_json,
                n_trials_accounted=int(n_trials_accounted),
                created_at=self._now(),
            )
            session.add(row)
            return row

    def record_llm_interaction(
        self,
        *,
        campaign: str,
        provider: str,
        model: str,
        purpose: str,
        prompt_sha256: str,
        prompt_path: str,
        response_path: str,
        prompt_template_sha256: str,
        tokens_in: int,
        tokens_out: int,
        cost_eur: float,
        temperature: float,
        latency_ms: int,
        status: str,
        seed: int | None = None,
        interaction_id: str | None = None,
    ) -> LlmInteraction:
        """Record that a model was asked something, and what it cost.

        Paths, not contents: the prompt and response live in the artifact tree, so
        the database stays small and the redaction rules of §12 apply in one place.
        """
        with session_scope(self.factory) as session:
            row = LlmInteraction(
                interaction_id=interaction_id or _event_id(),
                campaign=campaign,
                provider=provider,
                model=model,
                purpose=purpose,
                prompt_sha256=prompt_sha256,
                prompt_path=prompt_path,
                response_path=response_path,
                prompt_template_sha256=prompt_template_sha256,
                tokens_in=int(tokens_in),
                tokens_out=int(tokens_out),
                cost_eur=float(cost_eur),
                temperature=float(temperature),
                seed=None if seed is None else int(seed),
                latency_ms=int(latency_ms),
                status=status,
                created_at=self._now(),
            )
            session.add(row)
            return row

    def record_lockbox_access(
        self,
        *,
        strategy_id: str,
        family_id: str,
        os_user: str,
        reason: str,
        run_id: str | None = None,
        access_id: str | None = None,
    ) -> LockboxAccess:
        """Record a look at the test partition. Written whether or not it passed.

        This table is the enforcement surface for the per-family and per-month
        budgets of §15: a look that went unrecorded would be a free one.
        """
        with session_scope(self.factory) as session:
            row = LockboxAccess(
                access_id=access_id or _event_id(),
                strategy_id=strategy_id,
                family_id=family_id,
                run_id=run_id,
                os_user=os_user,
                reason=reason,
                created_at=self._now(),
            )
            session.add(row)
            return row

    def lockbox_accesses(
        self, *, family_id: str | None = None, since_ts: int | None = None
    ) -> list[LockboxAccess]:
        """Recorded looks at the test partition, oldest first."""
        statement = select(LockboxAccess)
        if family_id is not None:
            statement = statement.where(LockboxAccess.family_id == family_id)
        if since_ts is not None:
            statement = statement.where(LockboxAccess.created_at >= int(since_ts))
        with session_scope(self.factory) as session:
            rows = session.execute(statement).scalars().all()
        return sorted(rows, key=lambda r: (r.created_at, r.access_id))

    # -- studies and paper sessions ----------------------------------------
    def record_optuna_study(
        self,
        *,
        study_id: str,
        experiment_id: str,
        strategy_id: str,
        segment: str,
        sampler: str,
        seed: int,
        n_trials: int,
        objective: str,
        best_trial_json: str,
        plateau_json: str,
        storage_path: str,
    ) -> OptunaStudy:
        """Record a completed parameter study and the trial count it contributed."""
        with session_scope(self.factory) as session:
            row = OptunaStudy(
                study_id=study_id,
                experiment_id=experiment_id,
                strategy_id=strategy_id,
                segment=segment,
                sampler=sampler,
                seed=int(seed),
                n_trials=int(n_trials),
                objective=objective,
                best_trial_json=best_trial_json,
                plateau_json=plateau_json,
                storage_path=storage_path,
                created_at=self._now(),
            )
            session.add(row)
            return row

    def create_paper_session(
        self,
        *,
        session_id: str,
        strategy_id: str,
        params_json: str,
        state_path: str,
        source_run_id: str | None = None,
    ) -> PaperSession:
        """Open a paper-trading session in ``running``."""
        with session_scope(self.factory) as session:
            row = PaperSession(
                session_id=session_id,
                strategy_id=strategy_id,
                params_json=params_json,
                source_run_id=source_run_id,
                started_at=self._now(),
                stopped_at=None,
                last_bar_ts=None,
                state_path=state_path,
                status="running",
            )
            session.add(row)
            return row

    def update_paper_session(
        self,
        session_id: str,
        *,
        status: str | None = None,
        last_bar_ts: int | None = None,
        stopped_at: int | None = None,
    ) -> PaperSession:
        """Advance a paper session's mutable state (spec section 6)."""
        with session_scope(self.factory) as session:
            row = self._require_row(session, PaperSession, session_id, "paper_session")
            if status is not None:
                row.status = status
            if last_bar_ts is not None:
                row.last_bar_ts = int(last_bar_ts)
            if stopped_at is not None:
                row.stopped_at = int(stopped_at)
            return row

    # -- internals ----------------------------------------------------------
    def _require_row(self, session: Session, model: type[_R], key: str, table: str) -> _R:
        row = session.get(model, key)
        if row is None:
            raise StoreError(f"no such {table}", table=table, key=key)
        return row


_R = TypeVar("_R")


def _event_id() -> str:
    """An id for a record of something that happened, rather than of an identity.

    Two verdicts on the same strategy, or two calls to a model with the same
    prompt, are distinct events and must not collapse into one row — so these ids
    are random rather than derived from content.
    """
    return uuid.uuid4().hex[:16]


def _require_same(table: str, key: str, row: Any, expected: Mapping[str, Any]) -> None:
    """Raise if a re-registration disagrees with what is already stored."""
    differing = {
        name: {"stored": getattr(row, name), "given": value}
        for name, value in expected.items()
        if getattr(row, name) != value
    }
    if differing:
        raise RunConflict(
            f"{table} id already exists with different inputs",
            table=table,
            key=key,
            differing=sorted(differing),
            detail=canonical_json(differing),
        )


def _policy_document(policy: SplitPolicy) -> str:
    """The canonical policy document, as stored in ``split_policy.wf_json``."""
    return policy.source_json or canonical_json(policy.to_dict())


def _policy_from_row(row: SplitPolicyRow) -> SplitPolicy:
    """Rebuild the core policy a row was written from.

    The stored document is used *verbatim* as ``source_json``, because
    ``split_id`` is a hash of it and the row is keyed by that hash. ``test_end_ts``
    is then taken from the column, which is the one field spec section 6 allows to
    change — so a frozen policy keeps the identity of the policy it froze. Folding
    the frozen end back into the document instead would give the reloaded policy a
    different ``split_id`` from the row it came from, and the next
    ``get_or_create_split`` would silently insert a second row for the same split.

    Raises:
        StoreError: the rebuilt policy does not hash to the row's key, which means
            the stored document no longer matches the id it is filed under.
    """
    document = json.loads(row.wf_json)
    policy = parse_split_policy(document, dataset_id=row.dataset_id, source_json=row.wf_json)
    if policy.split_id != row.split_id:
        raise StoreError(
            "stored split policy does not hash to its own id",
            split_id=row.split_id,
            recomputed=policy.split_id,
        )
    return replace(policy, test_end_ts=row.test_end_ts)


def _trade_row(run_id: str, trade: TradeRecord) -> Trade:
    return Trade(
        run_id=run_id,
        trade_no=int(trade.trade_no),
        side=str(trade.side),
        entry_ts=int(trade.entry_ts),
        entry_px=float(trade.entry_px),
        exit_ts=int(trade.exit_ts),
        exit_px=float(trade.exit_px),
        qty=float(trade.qty),
        fees=float(trade.fees),
        slippage_cost=float(trade.slippage_cost),
        pnl=float(trade.pnl),
        pnl_pct=float(trade.pnl_pct),
        bars_held=int(trade.bars_held),
        exit_reason=str(trade.exit_reason),
    )


def _run_referrers(session: Session, run_id: str) -> list[str]:
    """Tables holding an append-only reference to ``run_id``."""
    referrers: list[str] = []
    if session.execute(
        select(LockboxAccess.access_id).where(LockboxAccess.run_id == run_id).limit(1)
    ).first():
        referrers.append("lockbox_access")
    if session.execute(
        select(PaperSession.session_id).where(PaperSession.source_run_id == run_id).limit(1)
    ).first():
        referrers.append("paper_session")
    return referrers
