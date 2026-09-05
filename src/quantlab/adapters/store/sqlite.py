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

import logging
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

from sqlalchemy import Engine, create_engine, event, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from quantlab.adapters.store.models import TABLE_NAMES
from quantlab.core.errors import ConfigError, StoreError

__all__ = [
    "ALEMBIC_VERSION_TABLE",
    "IN_MEMORY_URL",
    "connection_url",
    "create_db_engine",
    "current_revision",
    "make_session_factory",
    "migrations_dir",
    "missing_tables",
    "session_scope",
    "upgrade_to_head",
]

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
    """Return a session factory bound to ``engine``."""
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


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
