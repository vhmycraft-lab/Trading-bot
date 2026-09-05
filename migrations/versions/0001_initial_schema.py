"""Initial QuantLab schema.

The DDL below is the normative schema of master spec section 6, emitted
verbatim so the SQLite ``STRICT`` and ``WITHOUT ROWID`` clauses are exactly as
specified.  ``tests/unit/test_store.py`` asserts that the SQLAlchemy models in
``quantlab.adapters.store.models`` stay in step with this schema.

Revision ID: 0001
Revises:
Create Date: 2026-09-05
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


CREATE_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE dataset (
      dataset_id     TEXT PRIMARY KEY,
      exchange       TEXT NOT NULL, symbol TEXT NOT NULL, timeframe TEXT NOT NULL,
      start_ts INTEGER NOT NULL, end_ts INTEGER NOT NULL, n_bars INTEGER NOT NULL,
      manifest_json  TEXT NOT NULL, created_at INTEGER NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE split_policy (
      split_id       TEXT PRIMARY KEY,
      dataset_id     TEXT NOT NULL REFERENCES dataset,
      train_start_ts INTEGER NOT NULL, train_end_ts INTEGER NOT NULL,
      val_start_ts   INTEGER NOT NULL, val_end_ts   INTEGER NOT NULL,
      test_start_ts  INTEGER NOT NULL, test_end_ts  INTEGER,
      embargo_bars   INTEGER NOT NULL, wf_json TEXT NOT NULL, created_at INTEGER NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE strategy_family (
      family_id   TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE,
      origin      TEXT NOT NULL CHECK (origin IN ('human','llm')),
      description TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'open'
                  CHECK (status IN ('open','frozen','closed')),
      validation_touches INTEGER NOT NULL DEFAULT 0, created_at INTEGER NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE llm_interaction (
      interaction_id TEXT PRIMARY KEY, campaign TEXT NOT NULL, provider TEXT NOT NULL,
      model TEXT NOT NULL,
      purpose TEXT NOT NULL CHECK (purpose IN ('propose','repair','review')),
      prompt_sha256 TEXT NOT NULL, prompt_path TEXT NOT NULL, response_path TEXT NOT NULL,
      prompt_template_sha256 TEXT NOT NULL, tokens_in INTEGER NOT NULL, tokens_out INTEGER NOT NULL,
      cost_eur REAL NOT NULL, temperature REAL NOT NULL, seed INTEGER, latency_ms INTEGER NOT NULL,
      status TEXT NOT NULL CHECK (status IN ('ok','malformed','error')), created_at INTEGER NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE strategy_version (
      strategy_id        TEXT PRIMARY KEY,
      family_id          TEXT NOT NULL REFERENCES strategy_family,
      parent_strategy_id TEXT REFERENCES strategy_version,
      code_path          TEXT NOT NULL, code_sha256 TEXT NOT NULL,
      class_name         TEXT NOT NULL, param_schema_json TEXT NOT NULL,
      style              TEXT NOT NULL CHECK (style IN ('bar_loop','vectorized')),
      author             TEXT NOT NULL,
      llm_interaction_id TEXT REFERENCES llm_interaction,
      logic_lines        INTEGER NOT NULL, created_at INTEGER NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE experiment (
      experiment_id TEXT PRIMARY KEY, campaign TEXT NOT NULL,
      purpose TEXT NOT NULL CHECK (purpose IN
        ('smoke','train','optimize','validate','walkforward','lockbox','paper','baseline')),
      config_hash TEXT NOT NULL, config_json TEXT NOT NULL, seed INTEGER NOT NULL,
      git_commit TEXT NOT NULL, created_at INTEGER NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE run (
      run_id        TEXT PRIMARY KEY,
      experiment_id TEXT NOT NULL REFERENCES experiment,
      strategy_id   TEXT NOT NULL REFERENCES strategy_version,
      dataset_id    TEXT NOT NULL REFERENCES dataset,
      split_id      TEXT NOT NULL REFERENCES split_policy,
      segment       TEXT NOT NULL,
      params_json   TEXT NOT NULL, engine_name TEXT NOT NULL, engine_version TEXT NOT NULL,
      cost_multiplier REAL NOT NULL DEFAULT 1.0,
      status        TEXT NOT NULL CHECK (status IN ('pending','running','ok','failed','rejected')),
      error_json    TEXT, artifact_dir TEXT NOT NULL,
      started_at INTEGER, finished_at INTEGER, created_at INTEGER NOT NULL
    ) STRICT
    """,
    "CREATE INDEX ix_run_strategy ON run(strategy_id)",
    "CREATE INDEX ix_run_experiment ON run(experiment_id)",
    """
    CREATE TABLE metric (
      run_id TEXT NOT NULL REFERENCES run, name TEXT NOT NULL, value REAL,
      PRIMARY KEY (run_id, name)
    ) STRICT, WITHOUT ROWID
    """,
    """
    CREATE TABLE trade (
      run_id TEXT NOT NULL REFERENCES run, trade_no INTEGER NOT NULL,
      side TEXT NOT NULL CHECK (side IN ('long','short')),
      entry_ts INTEGER NOT NULL, entry_px REAL NOT NULL,
      exit_ts INTEGER NOT NULL, exit_px REAL NOT NULL,
      qty REAL NOT NULL, fees REAL NOT NULL, slippage_cost REAL NOT NULL,
      pnl REAL NOT NULL, pnl_pct REAL NOT NULL, bars_held INTEGER NOT NULL,
      exit_reason TEXT NOT NULL CHECK (exit_reason IN ('signal','end_of_data','stop')),
      PRIMARY KEY (run_id, trade_no)
    ) STRICT, WITHOUT ROWID
    """,
    """
    CREATE TABLE optuna_study (
      study_id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL REFERENCES experiment,
      strategy_id TEXT NOT NULL REFERENCES strategy_version, segment TEXT NOT NULL,
      sampler TEXT NOT NULL, seed INTEGER NOT NULL, n_trials INTEGER NOT NULL,
      objective TEXT NOT NULL, best_trial_json TEXT NOT NULL, plateau_json TEXT NOT NULL,
      storage_path TEXT NOT NULL, created_at INTEGER NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE validation_verdict (
      verdict_id TEXT PRIMARY KEY, strategy_id TEXT NOT NULL REFERENCES strategy_version,
      split_id TEXT NOT NULL REFERENCES split_policy, params_json TEXT NOT NULL,
      verdict TEXT NOT NULL CHECK (verdict IN
        ('REJECT','WEAK','CANDIDATE','LOCKBOX_PASS','LOCKBOX_FAIL')),
      overfit_score REAL NOT NULL, hard_gates_json TEXT NOT NULL, soft_checks_json TEXT NOT NULL,
      thresholds_json TEXT NOT NULL, n_trials_accounted INTEGER NOT NULL,
      created_at INTEGER NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE lockbox_access (
      access_id TEXT PRIMARY KEY, strategy_id TEXT NOT NULL REFERENCES strategy_version,
      family_id TEXT NOT NULL REFERENCES strategy_family, run_id TEXT REFERENCES run,
      os_user TEXT NOT NULL, reason TEXT NOT NULL, created_at INTEGER NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE paper_session (
      session_id TEXT PRIMARY KEY, strategy_id TEXT NOT NULL REFERENCES strategy_version,
      params_json TEXT NOT NULL, source_run_id TEXT REFERENCES run,
      started_at INTEGER NOT NULL, stopped_at INTEGER, last_bar_ts INTEGER,
      state_path TEXT NOT NULL, status TEXT NOT NULL CHECK (status IN
        ('running','stopped','crashed'))
    ) STRICT
    """,
)

DROP_ORDER: tuple[str, ...] = (
    "paper_session",
    "lockbox_access",
    "validation_verdict",
    "optuna_study",
    "trade",
    "metric",
    "run",
    "experiment",
    "strategy_version",
    "llm_interaction",
    "strategy_family",
    "split_policy",
    "dataset",
)


def upgrade() -> None:
    for statement in CREATE_STATEMENTS:
        op.execute(statement.strip())


def downgrade() -> None:
    for table in DROP_ORDER:
        op.execute(f"DROP TABLE IF EXISTS {table}")
