"""The hidden training-environment audit table (Project Rome sections 18, 24).

One row per generation, recording the environment Rome's randomisation service
selected for it: the seed, the window, the capital, the slippage, the asset
universe and the two model versions. Rome section 24 requires that every
evaluation be reconstructible, and this is what a privileged auditor reads to do
it — ``core/environment.reproduce`` rebuilds the whole environment from
``seed_hex`` alone, and the remaining columns are what a reproduction is checked
against.

The table is append-only by default, like every other table without an entry in
``MUTABLE_COLUMNS``: an audit record that could be edited afterwards would not be
an audit record. ``UNIQUE (evolution_id, gen_index)`` is the second half of that
guarantee — a generation has exactly one environment, so a resumed run replays
the recorded one rather than drawing a second and quietly becoming a different
experiment.

The seed is stored here and nowhere else. It never enters a run row, an artifact,
a log line, a prompt or a strategy's inputs (Rome sections 6, 19).

As in ``0001`` and ``0002`` the DDL is emitted verbatim so the SQLite ``STRICT``
clause is exactly as intended; ``tests/unit/test_store.py`` asserts the
SQLAlchemy model stays in step with it.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-06
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


CREATE_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE training_environment (
      environment_id   TEXT PRIMARY KEY,
      evolution_id     TEXT NOT NULL REFERENCES evolution_run,
      gen_index        INTEGER NOT NULL,
      window_start_ts  INTEGER NOT NULL,
      window_end_ts    INTEGER NOT NULL,
      window_id        TEXT NOT NULL,
      starting_capital REAL NOT NULL,
      slippage_bps     REAL NOT NULL,
      asset_universe_json TEXT NOT NULL,
      seed_hex         TEXT NOT NULL,
      pool_id          TEXT NOT NULL,
      dataset_version  TEXT NOT NULL,
      execution_model_version TEXT NOT NULL,
      derivation_version TEXT NOT NULL,
      created_at       INTEGER NOT NULL,
      CONSTRAINT uq_training_environment_generation UNIQUE (evolution_id, gen_index)
    ) STRICT
    """,
    "CREATE INDEX ix_training_environment_window ON training_environment(window_id)",
)

DROP_STATEMENTS: tuple[str, ...] = (
    "DROP INDEX IF EXISTS ix_training_environment_window",
    "DROP TABLE IF EXISTS training_environment",
)


def upgrade() -> None:
    for statement in CREATE_STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    for statement in DROP_STATEMENTS:
        op.execute(statement)
