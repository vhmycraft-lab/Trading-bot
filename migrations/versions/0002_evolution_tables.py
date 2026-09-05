"""Evolution tables (master spec section 6, spec 1.1; task T21 amendment).

The five tables the evolutionary optimiser records itself in. Added in a second
migration rather than folded into ``0001`` because ``0001`` has already been
applied to real databases and an applied migration is a fact, not a draft.

As in ``0001`` the DDL is emitted verbatim so the SQLite ``STRICT`` clauses are
exactly as specified; ``tests/unit/test_store.py`` asserts the SQLAlchemy models
stay in step with it.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-05
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


CREATE_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE evolution_run (
      evolution_id   TEXT PRIMARY KEY,
      experiment_id  TEXT NOT NULL REFERENCES experiment,
      campaign       TEXT NOT NULL,
      dataset_id     TEXT NOT NULL REFERENCES dataset,
      split_id       TEXT NOT NULL REFERENCES split_policy,
      population_size INTEGER NOT NULL, n_survivors INTEGER NOT NULL,
      n_offspring INTEGER NOT NULL, n_immigrants INTEGER NOT NULL,
      max_generations INTEGER NOT NULL, seed INTEGER NOT NULL,
      fitness_config_json TEXT NOT NULL, mutation_config_json TEXT NOT NULL,
      diversity_config_json TEXT NOT NULL,
      n_evaluations  INTEGER NOT NULL DEFAULT 0,
      status         TEXT NOT NULL CHECK (status IN ('running','completed','stopped','failed')),
      stop_reason    TEXT,
      started_at INTEGER NOT NULL, finished_at INTEGER, created_at INTEGER NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE generation (
      generation_id  TEXT PRIMARY KEY,
      evolution_id   TEXT NOT NULL REFERENCES evolution_run,
      gen_index      INTEGER NOT NULL,
      best_fitness REAL, median_fitness REAL, mean_fitness REAL,
      diversity      REAL NOT NULL,
      n_evaluated INTEGER NOT NULL, n_cache_hits INTEGER NOT NULL,
      n_rejected_by_gate INTEGER NOT NULL, n_immigrants_used INTEGER NOT NULL,
      stats_json     TEXT NOT NULL, created_at INTEGER NOT NULL,
      CONSTRAINT uq_generation_evolution_index UNIQUE (evolution_id, gen_index)
    ) STRICT
    """,
    """
    CREATE TABLE candidate (
      candidate_id   TEXT PRIMARY KEY,
      evolution_id   TEXT NOT NULL REFERENCES evolution_run,
      generation_id  TEXT NOT NULL REFERENCES generation,
      gen_index      INTEGER NOT NULL,
      strategy_id    TEXT NOT NULL REFERENCES strategy_version,
      params_json    TEXT NOT NULL,
      genome_json    TEXT,
      kind           TEXT NOT NULL CHECK (kind IN ('genome','opaque')),
      origin         TEXT NOT NULL
                     CHECK (origin IN ('seed','survivor','mutant','immigrant','refined')),
      parent_candidate_id TEXT REFERENCES candidate,
      run_id         TEXT REFERENCES run,
      fitness REAL, base_score REAL, penalty_product REAL,
      components_json TEXT NOT NULL DEFAULT '{}',
      penalties_json  TEXT NOT NULL DEFAULT '{}',
      gate_failure   TEXT,
      rank           INTEGER,
      survived       INTEGER NOT NULL DEFAULT 0 CHECK (survived IN (0,1)),
      behaviour_hash TEXT,
      signature_json TEXT NOT NULL DEFAULT '{}',
      created_at     INTEGER NOT NULL
    ) STRICT
    """,
    "CREATE INDEX ix_candidate_evolution ON candidate(evolution_id, gen_index)",
    "CREATE INDEX ix_candidate_parent ON candidate(parent_candidate_id)",
    "CREATE INDEX ix_candidate_strategy ON candidate(strategy_id)",
    """
    CREATE TABLE mutation (
      mutation_id    TEXT PRIMARY KEY,
      candidate_id   TEXT NOT NULL REFERENCES candidate,
      parent_candidate_id TEXT NOT NULL REFERENCES candidate,
      seq            INTEGER NOT NULL,
      category       TEXT NOT NULL CHECK (category IN ('parameter','structural')),
      operator       TEXT NOT NULL,
      target         TEXT NOT NULL,
      before_json    TEXT NOT NULL, after_json TEXT NOT NULL,
      rng_seed       INTEGER NOT NULL,
      suggested_by   TEXT NOT NULL DEFAULT 'rng' CHECK (suggested_by IN ('rng','llm')),
      llm_interaction_id TEXT REFERENCES llm_interaction,
      created_at     INTEGER NOT NULL,
      CONSTRAINT uq_mutation_candidate_seq UNIQUE (candidate_id, seq)
    ) STRICT
    """,
    "CREATE INDEX ix_mutation_candidate ON mutation(candidate_id)",
    """
    CREATE TABLE candidate_promotion (
      promotion_id   TEXT PRIMARY KEY,
      candidate_id   TEXT NOT NULL REFERENCES candidate,
      evolution_id   TEXT NOT NULL REFERENCES evolution_run,
      gen_index      INTEGER NOT NULL,
      segment        TEXT NOT NULL,
      run_id         TEXT REFERENCES run,
      verdict_id     TEXT REFERENCES validation_verdict,
      reason         TEXT NOT NULL,
      created_at     INTEGER NOT NULL
    ) STRICT
    """,
    "CREATE INDEX ix_promotion_candidate ON candidate_promotion(candidate_id)",
)

#: Reverse creation order, so a child table never outlives its parent.
DROP_STATEMENTS: tuple[str, ...] = (
    "DROP TABLE IF EXISTS candidate_promotion",
    "DROP TABLE IF EXISTS mutation",
    "DROP TABLE IF EXISTS candidate",
    "DROP TABLE IF EXISTS generation",
    "DROP TABLE IF EXISTS evolution_run",
)


def upgrade() -> None:
    for statement in CREATE_STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    for statement in DROP_STATEMENTS:
        op.execute(statement)
