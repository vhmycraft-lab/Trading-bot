"""SQLAlchemy 2.0 models for the QuantLab store (master spec section 6).

Conventions, all enforced here and by the Alembic migration:

* ``TEXT`` ids are hex hashes or UUID4
* every timestamp is ``INTEGER`` milliseconds UTC
* JSON columns are ``TEXT`` holding canonical JSON
* tables are ``STRICT``; ``metric`` and ``trade`` are also ``WITHOUT ROWID``
* foreign keys are enforced and the journal mode is WAL (see :mod:`.sqlite`)

Rows are append-only except for the columns listed in :data:`MUTABLE_COLUMNS`.
"""

from __future__ import annotations

from typing import Final

from sqlalchemy import (
    REAL,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

__all__ = [
    "MUTABLE_COLUMNS",
    "TABLE_NAMES",
    "Base",
    "Dataset",
    "Experiment",
    "LlmInteraction",
    "LockboxAccess",
    "Metric",
    "OptunaStudy",
    "PaperSession",
    "Run",
    "SplitPolicy",
    "StrategyFamily",
    "StrategyVersion",
    "Trade",
    "ValidationVerdict",
]

_STRICT: Final[dict[str, object]] = {"sqlite_strict": True}
_STRICT_NO_ROWID: Final[dict[str, object]] = {"sqlite_strict": True, "sqlite_with_rowid": False}


class Base(DeclarativeBase):
    """Declarative base for every QuantLab table."""


class Dataset(Base):
    __tablename__ = "dataset"
    __table_args__ = _STRICT

    dataset_id: Mapped[str] = mapped_column(Text, primary_key=True)
    exchange: Mapped[str] = mapped_column(Text, nullable=False)
    symbol: Mapped[str] = mapped_column(Text, nullable=False)
    timeframe: Mapped[str] = mapped_column(Text, nullable=False)
    start_ts: Mapped[int] = mapped_column(Integer, nullable=False)
    end_ts: Mapped[int] = mapped_column(Integer, nullable=False)
    n_bars: Mapped[int] = mapped_column(Integer, nullable=False)
    manifest_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)


class SplitPolicy(Base):
    __tablename__ = "split_policy"
    __table_args__ = _STRICT

    split_id: Mapped[str] = mapped_column(Text, primary_key=True)
    dataset_id: Mapped[str] = mapped_column(Text, ForeignKey("dataset.dataset_id"), nullable=False)
    train_start_ts: Mapped[int] = mapped_column(Integer, nullable=False)
    train_end_ts: Mapped[int] = mapped_column(Integer, nullable=False)
    val_start_ts: Mapped[int] = mapped_column(Integer, nullable=False)
    val_end_ts: Mapped[int] = mapped_column(Integer, nullable=False)
    test_start_ts: Mapped[int] = mapped_column(Integer, nullable=False)
    # NULL until the lockbox is used for the first time, then frozen forever.
    test_end_ts: Mapped[int | None] = mapped_column(Integer, nullable=True)
    embargo_bars: Mapped[int] = mapped_column(Integer, nullable=False)
    wf_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)


class StrategyFamily(Base):
    __tablename__ = "strategy_family"
    __table_args__ = (
        CheckConstraint("origin IN ('human','llm')", name="ck_strategy_family_origin"),
        CheckConstraint("status IN ('open','frozen','closed')", name="ck_strategy_family_status"),
        _STRICT,
    )

    family_id: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    origin: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, server_default="''")
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="'open'")
    validation_touches: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)


class StrategyVersion(Base):
    __tablename__ = "strategy_version"
    __table_args__ = (
        CheckConstraint("style IN ('bar_loop','vectorized')", name="ck_strategy_version_style"),
        _STRICT,
    )

    strategy_id: Mapped[str] = mapped_column(Text, primary_key=True)
    family_id: Mapped[str] = mapped_column(
        Text, ForeignKey("strategy_family.family_id"), nullable=False
    )
    parent_strategy_id: Mapped[str | None] = mapped_column(
        Text, ForeignKey("strategy_version.strategy_id"), nullable=True
    )
    code_path: Mapped[str] = mapped_column(Text, nullable=False)
    code_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    class_name: Mapped[str] = mapped_column(Text, nullable=False)
    param_schema_json: Mapped[str] = mapped_column(Text, nullable=False)
    style: Mapped[str] = mapped_column(Text, nullable=False)
    author: Mapped[str] = mapped_column(Text, nullable=False)
    llm_interaction_id: Mapped[str | None] = mapped_column(
        Text, ForeignKey("llm_interaction.interaction_id"), nullable=True
    )
    logic_lines: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)


class Experiment(Base):
    __tablename__ = "experiment"
    __table_args__ = (
        CheckConstraint(
            "purpose IN ('smoke','train','optimize','validate','walkforward',"
            "'lockbox','paper','baseline')",
            name="ck_experiment_purpose",
        ),
        _STRICT,
    )

    experiment_id: Mapped[str] = mapped_column(Text, primary_key=True)
    campaign: Mapped[str] = mapped_column(Text, nullable=False)
    purpose: Mapped[str] = mapped_column(Text, nullable=False)
    config_hash: Mapped[str] = mapped_column(Text, nullable=False)
    config_json: Mapped[str] = mapped_column(Text, nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    git_commit: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)


class Run(Base):
    __tablename__ = "run"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending','running','ok','failed','rejected')", name="ck_run_status"
        ),
        Index("ix_run_strategy", "strategy_id"),
        Index("ix_run_experiment", "experiment_id"),
        _STRICT,
    )

    run_id: Mapped[str] = mapped_column(Text, primary_key=True)
    experiment_id: Mapped[str] = mapped_column(
        Text, ForeignKey("experiment.experiment_id"), nullable=False
    )
    strategy_id: Mapped[str] = mapped_column(
        Text, ForeignKey("strategy_version.strategy_id"), nullable=False
    )
    dataset_id: Mapped[str] = mapped_column(Text, ForeignKey("dataset.dataset_id"), nullable=False)
    split_id: Mapped[str] = mapped_column(Text, ForeignKey("split_policy.split_id"), nullable=False)
    #: 'train' | 'val' | 'test' | 'wf_is:<k>' | 'wf_oos:<k>' | 'custom:<start>-<end>'
    segment: Mapped[str] = mapped_column(Text, nullable=False)
    params_json: Mapped[str] = mapped_column(Text, nullable=False)
    engine_name: Mapped[str] = mapped_column(Text, nullable=False)
    engine_version: Mapped[str] = mapped_column(Text, nullable=False)
    cost_multiplier: Mapped[float] = mapped_column(REAL, nullable=False, server_default="1.0")
    status: Mapped[str] = mapped_column(Text, nullable=False)
    error_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_dir: Mapped[str] = mapped_column(Text, nullable=False)
    started_at: Mapped[int | None] = mapped_column(Integer, nullable=True)
    finished_at: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)


class Metric(Base):
    __tablename__ = "metric"
    __table_args__ = _STRICT_NO_ROWID

    run_id: Mapped[str] = mapped_column(Text, ForeignKey("run.run_id"), primary_key=True)
    name: Mapped[str] = mapped_column(Text, primary_key=True)
    #: NULL means "undefined", e.g. profit factor with no losing trades.
    value: Mapped[float | None] = mapped_column(REAL, nullable=True)


class Trade(Base):
    __tablename__ = "trade"
    __table_args__ = (
        CheckConstraint("side IN ('long','short')", name="ck_trade_side"),
        CheckConstraint(
            "exit_reason IN ('signal','end_of_data','stop')", name="ck_trade_exit_reason"
        ),
        _STRICT_NO_ROWID,
    )

    run_id: Mapped[str] = mapped_column(Text, ForeignKey("run.run_id"), primary_key=True)
    trade_no: Mapped[int] = mapped_column(Integer, primary_key=True)
    side: Mapped[str] = mapped_column(Text, nullable=False)
    entry_ts: Mapped[int] = mapped_column(Integer, nullable=False)
    entry_px: Mapped[float] = mapped_column(REAL, nullable=False)
    exit_ts: Mapped[int] = mapped_column(Integer, nullable=False)
    exit_px: Mapped[float] = mapped_column(REAL, nullable=False)
    qty: Mapped[float] = mapped_column(REAL, nullable=False)
    fees: Mapped[float] = mapped_column(REAL, nullable=False)
    slippage_cost: Mapped[float] = mapped_column(REAL, nullable=False)
    pnl: Mapped[float] = mapped_column(REAL, nullable=False)
    pnl_pct: Mapped[float] = mapped_column(REAL, nullable=False)
    bars_held: Mapped[int] = mapped_column(Integer, nullable=False)
    exit_reason: Mapped[str] = mapped_column(Text, nullable=False)


class OptunaStudy(Base):
    __tablename__ = "optuna_study"
    __table_args__ = _STRICT

    study_id: Mapped[str] = mapped_column(Text, primary_key=True)
    experiment_id: Mapped[str] = mapped_column(
        Text, ForeignKey("experiment.experiment_id"), nullable=False
    )
    strategy_id: Mapped[str] = mapped_column(
        Text, ForeignKey("strategy_version.strategy_id"), nullable=False
    )
    segment: Mapped[str] = mapped_column(Text, nullable=False)
    sampler: Mapped[str] = mapped_column(Text, nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    n_trials: Mapped[int] = mapped_column(Integer, nullable=False)
    objective: Mapped[str] = mapped_column(Text, nullable=False)
    best_trial_json: Mapped[str] = mapped_column(Text, nullable=False)
    plateau_json: Mapped[str] = mapped_column(Text, nullable=False)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)


class ValidationVerdict(Base):
    __tablename__ = "validation_verdict"
    __table_args__ = (
        CheckConstraint(
            "verdict IN ('REJECT','WEAK','CANDIDATE','LOCKBOX_PASS','LOCKBOX_FAIL')",
            name="ck_validation_verdict_verdict",
        ),
        _STRICT,
    )

    verdict_id: Mapped[str] = mapped_column(Text, primary_key=True)
    strategy_id: Mapped[str] = mapped_column(
        Text, ForeignKey("strategy_version.strategy_id"), nullable=False
    )
    split_id: Mapped[str] = mapped_column(Text, ForeignKey("split_policy.split_id"), nullable=False)
    params_json: Mapped[str] = mapped_column(Text, nullable=False)
    verdict: Mapped[str] = mapped_column(Text, nullable=False)
    overfit_score: Mapped[float] = mapped_column(REAL, nullable=False)
    hard_gates_json: Mapped[str] = mapped_column(Text, nullable=False)
    soft_checks_json: Mapped[str] = mapped_column(Text, nullable=False)
    thresholds_json: Mapped[str] = mapped_column(Text, nullable=False)
    n_trials_accounted: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)


class LlmInteraction(Base):
    __tablename__ = "llm_interaction"
    __table_args__ = (
        CheckConstraint(
            "purpose IN ('propose','repair','review')", name="ck_llm_interaction_purpose"
        ),
        CheckConstraint("status IN ('ok','malformed','error')", name="ck_llm_interaction_status"),
        _STRICT,
    )

    interaction_id: Mapped[str] = mapped_column(Text, primary_key=True)
    campaign: Mapped[str] = mapped_column(Text, nullable=False)
    provider: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(Text, nullable=False)
    purpose: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_path: Mapped[str] = mapped_column(Text, nullable=False)
    response_path: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_template_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    tokens_in: Mapped[int] = mapped_column(Integer, nullable=False)
    tokens_out: Mapped[int] = mapped_column(Integer, nullable=False)
    cost_eur: Mapped[float] = mapped_column(REAL, nullable=False)
    temperature: Mapped[float] = mapped_column(REAL, nullable=False)
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)


class LockboxAccess(Base):
    __tablename__ = "lockbox_access"
    __table_args__ = _STRICT

    access_id: Mapped[str] = mapped_column(Text, primary_key=True)
    strategy_id: Mapped[str] = mapped_column(
        Text, ForeignKey("strategy_version.strategy_id"), nullable=False
    )
    family_id: Mapped[str] = mapped_column(
        Text, ForeignKey("strategy_family.family_id"), nullable=False
    )
    run_id: Mapped[str | None] = mapped_column(Text, ForeignKey("run.run_id"), nullable=True)
    os_user: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)


class PaperSession(Base):
    __tablename__ = "paper_session"
    __table_args__ = (
        CheckConstraint(
            "status IN ('running','stopped','crashed')", name="ck_paper_session_status"
        ),
        _STRICT,
    )

    session_id: Mapped[str] = mapped_column(Text, primary_key=True)
    strategy_id: Mapped[str] = mapped_column(
        Text, ForeignKey("strategy_version.strategy_id"), nullable=False
    )
    params_json: Mapped[str] = mapped_column(Text, nullable=False)
    source_run_id: Mapped[str | None] = mapped_column(Text, ForeignKey("run.run_id"), nullable=True)
    started_at: Mapped[int] = mapped_column(Integer, nullable=False)
    stopped_at: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_bar_ts: Mapped[int | None] = mapped_column(Integer, nullable=True)
    state_path: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)


#: Every table in the schema, in creation order.
TABLE_NAMES: Final[tuple[str, ...]] = (
    "dataset",
    "split_policy",
    "strategy_family",
    "llm_interaction",
    "strategy_version",
    "experiment",
    "run",
    "metric",
    "trade",
    "optuna_study",
    "validation_verdict",
    "lockbox_access",
    "paper_session",
)

#: Columns exempt from the append-only rule (master spec section 6).
MUTABLE_COLUMNS: Final[dict[str, frozenset[str]]] = {
    "run": frozenset({"status", "error_json", "started_at", "finished_at"}),
    "strategy_family": frozenset({"status", "validation_touches"}),
    "split_policy": frozenset({"test_end_ts"}),
    "paper_session": frozenset({"started_at", "stopped_at", "status", "last_bar_ts"}),
}
