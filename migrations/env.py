"""Alembic environment for QuantLab.

The database URL is resolved in this order:

1. a live :class:`~sqlalchemy.Engine` passed in ``config.attributes["connection"]``
   (used by :func:`quantlab.adapters.store.sqlite.upgrade_to_head`, and the only
   way an in-memory database can be migrated)
2. ``-x db_path=...`` on the alembic command line
3. ``project.db_path`` from the QuantLab configuration
"""

from __future__ import annotations

from pathlib import Path

from alembic import context
from sqlalchemy import Engine, engine_from_config, pool

from quantlab.adapters.store.models import Base
from quantlab.adapters.store.sqlite import connection_url

config = context.config
target_metadata = Base.metadata


def _resolve_url() -> str:
    x_args = context.get_x_argument(as_dictionary=True)
    if "db_path" in x_args:
        return connection_url(x_args["db_path"])

    configured = config.get_main_option("sqlalchemy.url")
    if configured:
        return configured

    from quantlab.core.config import DEFAULT_CONFIG_PATH, load_config

    if Path(DEFAULT_CONFIG_PATH).is_file():
        return connection_url(load_config().project.db_path)
    return connection_url("quantlab.db")


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting to a database."""
    context.configure(
        url=_resolve_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _run(connection) -> None:  # type: ignore[no-untyped-def]
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against a live connection."""
    existing = config.attributes.get("connection")

    if isinstance(existing, Engine):
        with existing.begin() as connection:
            _run(connection)
        return
    if existing is not None:
        _run(existing)
        return

    section = config.get_section(config.config_ini_section, {}) or {}
    section["sqlalchemy.url"] = _resolve_url()
    engine = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with engine.begin() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys = ON")
        _run(connection)
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
