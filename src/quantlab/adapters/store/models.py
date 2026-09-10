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
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

__all__ = [
    "EVOLUTION_TABLES",
    "MUTABLE_COLUMNS",
    "TABLE_NAMES",
    "Base",
    "Candidate",
    "CandidatePromotion",
    "Dataset",
    "EvolutionRun",
    "Experiment",
    "Generation",
    "LlmInteraction",
    "LockboxAccess",
    "Metric",
    "Mutation",
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
    #: Which formula produced ``n_trials_accounted``. Defaulted to the superseded
    #: one so that every row written before migration ``0004`` is marked as having
    #: been computed under it, rather than silently inheriting today's meaning.
    m_formula_version: Mapped[str] = mapped_column(
        Text, nullable=False, server_default="'m1-bar-count'"
    )
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


class TrainingEnvironmentRow(Base):
    """One generation's hidden training environment (Project Rome sections 18, 24).

    The audit record, and the only place a seed is ever stored. Append-only like
    every table without a ``MUTABLE_COLUMNS`` entry — an audit record that could
    be edited afterwards would not be one — and unique per generation, so a
    resumed run replays the environment that was recorded rather than drawing a
    second one and silently becoming a different experiment.
    """

    __tablename__ = "training_environment"
    __table_args__ = (
        UniqueConstraint("evolution_id", "gen_index", name="uq_training_environment_generation"),
        _STRICT,
    )

    environment_id: Mapped[str] = mapped_column(Text, primary_key=True)
    evolution_id: Mapped[str] = mapped_column(
        Text, ForeignKey("evolution_run.evolution_id"), nullable=False
    )
    gen_index: Mapped[int] = mapped_column(Integer, nullable=False)
    window_start_ts: Mapped[int] = mapped_column(Integer, nullable=False)
    window_end_ts: Mapped[int] = mapped_column(Integer, nullable=False)
    window_id: Mapped[str] = mapped_column(Text, nullable=False)
    starting_capital: Mapped[float] = mapped_column(REAL, nullable=False)
    slippage_bps: Mapped[float] = mapped_column(REAL, nullable=False)
    asset_universe_json: Mapped[str] = mapped_column(Text, nullable=False)
    #: The secret. Recorded here and passed to nothing (Rome sections 6, 19).
    seed_hex: Mapped[str] = mapped_column(Text, nullable=False)
    pool_id: Mapped[str] = mapped_column(Text, nullable=False)
    dataset_version: Mapped[str] = mapped_column(Text, nullable=False)
    execution_model_version: Mapped[str] = mapped_column(Text, nullable=False)
    derivation_version: Mapped[str] = mapped_column(Text, nullable=False)
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


class EvolutionRun(Base):
    __tablename__ = "evolution_run"
    __table_args__ = (
        CheckConstraint(
            "status IN ('running','completed','stopped','failed')", name="ck_evolution_run_status"
        ),
        _STRICT,
    )

    evolution_id: Mapped[str] = mapped_column(Text, primary_key=True)
    experiment_id: Mapped[str] = mapped_column(
        Text, ForeignKey("experiment.experiment_id"), nullable=False
    )
    campaign: Mapped[str] = mapped_column(Text, nullable=False)
    dataset_id: Mapped[str] = mapped_column(Text, ForeignKey("dataset.dataset_id"), nullable=False)
    split_id: Mapped[str] = mapped_column(Text, ForeignKey("split_policy.split_id"), nullable=False)
    population_size: Mapped[int] = mapped_column(Integer, nullable=False)
    n_survivors: Mapped[int] = mapped_column(Integer, nullable=False)
    n_offspring: Mapped[int] = mapped_column(Integer, nullable=False)
    n_immigrants: Mapped[int] = mapped_column(Integer, nullable=False)
    max_generations: Mapped[int] = mapped_column(Integer, nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    fitness_config_json: Mapped[str] = mapped_column(Text, nullable=False)
    mutation_config_json: Mapped[str] = mapped_column(Text, nullable=False)
    diversity_config_json: Mapped[str] = mapped_column(Text, nullable=False)
    #: Feeds the deflated Sharpe ratio's ``M`` (spec section 14.4): the honest
    #: count of how many things were tried.
    n_evaluations: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    status: Mapped[str] = mapped_column(Text, nullable=False)
    stop_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[int] = mapped_column(Integer, nullable=False)
    finished_at: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)


class Generation(Base):
    __tablename__ = "generation"
    __table_args__ = (
        UniqueConstraint("evolution_id", "gen_index", name="uq_generation_evolution_index"),
        _STRICT,
    )

    generation_id: Mapped[str] = mapped_column(Text, primary_key=True)
    evolution_id: Mapped[str] = mapped_column(
        Text, ForeignKey("evolution_run.evolution_id"), nullable=False
    )
    gen_index: Mapped[int] = mapped_column(Integer, nullable=False)
    best_fitness: Mapped[float | None] = mapped_column(REAL, nullable=True)
    median_fitness: Mapped[float | None] = mapped_column(REAL, nullable=True)
    mean_fitness: Mapped[float | None] = mapped_column(REAL, nullable=True)
    #: ``1 - mean pairwise similarity`` (spec section 13.5).
    diversity: Mapped[float] = mapped_column(REAL, nullable=False)
    n_evaluated: Mapped[int] = mapped_column(Integer, nullable=False)
    n_cache_hits: Mapped[int] = mapped_column(Integer, nullable=False)
    n_rejected_by_gate: Mapped[int] = mapped_column(Integer, nullable=False)
    n_immigrants_used: Mapped[int] = mapped_column(Integer, nullable=False)
    stats_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)


class Candidate(Base):
    __tablename__ = "candidate"
    __table_args__ = (
        CheckConstraint("kind IN ('genome','opaque')", name="ck_candidate_kind"),
        CheckConstraint(
            "origin IN ('seed','survivor','mutant','immigrant','refined')",
            name="ck_candidate_origin",
        ),
        CheckConstraint("survived IN (0,1)", name="ck_candidate_survived"),
        Index("ix_candidate_evolution", "evolution_id", "gen_index"),
        Index("ix_candidate_parent", "parent_candidate_id"),
        Index("ix_candidate_strategy", "strategy_id"),
        _STRICT,
    )

    candidate_id: Mapped[str] = mapped_column(Text, primary_key=True)
    evolution_id: Mapped[str] = mapped_column(
        Text, ForeignKey("evolution_run.evolution_id"), nullable=False
    )
    generation_id: Mapped[str] = mapped_column(
        Text, ForeignKey("generation.generation_id"), nullable=False
    )
    gen_index: Mapped[int] = mapped_column(Integer, nullable=False)
    strategy_id: Mapped[str] = mapped_column(
        Text, ForeignKey("strategy_version.strategy_id"), nullable=False
    )
    params_json: Mapped[str] = mapped_column(Text, nullable=False)
    #: NULL for ``opaque`` candidates, which admit parameter mutations only.
    genome_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    origin: Mapped[str] = mapped_column(Text, nullable=False)
    parent_candidate_id: Mapped[str | None] = mapped_column(
        Text, ForeignKey("candidate.candidate_id"), nullable=True
    )
    #: The train-segment evaluation. INV-9: never a validation or test run.
    run_id: Mapped[str | None] = mapped_column(Text, ForeignKey("run.run_id"), nullable=True)
    fitness: Mapped[float | None] = mapped_column(REAL, nullable=True)
    base_score: Mapped[float | None] = mapped_column(REAL, nullable=True)
    penalty_product: Mapped[float | None] = mapped_column(REAL, nullable=True)
    components_json: Mapped[str] = mapped_column(Text, nullable=False, server_default="'{}'")
    penalties_json: Mapped[str] = mapped_column(Text, nullable=False, server_default="'{}'")
    gate_failure: Mapped[str | None] = mapped_column(Text, nullable=True)
    rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    survived: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    #: Digest of the ``position_frac`` series (spec section 13.5).
    behaviour_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    signature_json: Mapped[str] = mapped_column(Text, nullable=False, server_default="'{}'")
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)


class Mutation(Base):
    """One recorded edit. Never mutable: it is the record INV-10 replays."""

    __tablename__ = "mutation"
    __table_args__ = (
        CheckConstraint("category IN ('parameter','structural')", name="ck_mutation_category"),
        CheckConstraint("suggested_by IN ('rng','llm')", name="ck_mutation_suggested_by"),
        UniqueConstraint("candidate_id", "seq", name="uq_mutation_candidate_seq"),
        Index("ix_mutation_candidate", "candidate_id"),
        _STRICT,
    )

    mutation_id: Mapped[str] = mapped_column(Text, primary_key=True)
    candidate_id: Mapped[str] = mapped_column(
        Text, ForeignKey("candidate.candidate_id"), nullable=False
    )
    parent_candidate_id: Mapped[str] = mapped_column(
        Text, ForeignKey("candidate.candidate_id"), nullable=False
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    category: Mapped[str] = mapped_column(Text, nullable=False)
    operator: Mapped[str] = mapped_column(Text, nullable=False)
    #: Genome path, e.g. ``entry.conditions[1].right``.
    target: Mapped[str] = mapped_column(Text, nullable=False)
    before_json: Mapped[str] = mapped_column(Text, nullable=False)
    after_json: Mapped[str] = mapped_column(Text, nullable=False)
    #: Makes the mutation replayable (INV-10).
    rng_seed: Mapped[int] = mapped_column(Integer, nullable=False)
    suggested_by: Mapped[str] = mapped_column(Text, nullable=False, server_default="'rng'")
    llm_interaction_id: Mapped[str | None] = mapped_column(
        Text, ForeignKey("llm_interaction.interaction_id"), nullable=True
    )
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)


class CandidatePromotion(Base):
    """The only route to the validation segment (INV-9), written before the run."""

    __tablename__ = "candidate_promotion"
    __table_args__ = (
        Index("ix_promotion_candidate", "candidate_id"),
        _STRICT,
    )

    promotion_id: Mapped[str] = mapped_column(Text, primary_key=True)
    candidate_id: Mapped[str] = mapped_column(
        Text, ForeignKey("candidate.candidate_id"), nullable=False
    )
    evolution_id: Mapped[str] = mapped_column(
        Text, ForeignKey("evolution_run.evolution_id"), nullable=False
    )
    gen_index: Mapped[int] = mapped_column(Integer, nullable=False)
    #: ``'val'`` or ``'wf_oos:<k>'``.
    segment: Mapped[str] = mapped_column(Text, nullable=False)
    run_id: Mapped[str | None] = mapped_column(Text, ForeignKey("run.run_id"), nullable=True)
    verdict_id: Mapped[str | None] = mapped_column(
        Text, ForeignKey("validation_verdict.verdict_id"), nullable=True
    )
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    #: Written BEFORE the run executes (INV-9).
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)


#: The five tables migration 0002 adds (spec 1.1, task T21 amendment).
EVOLUTION_TABLES: Final[tuple[str, ...]] = (
    "evolution_run",
    "generation",
    "candidate",
    "mutation",
    "candidate_promotion",
)

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
    *EVOLUTION_TABLES,
    "training_environment",
)

#: Columns exempt from the append-only rule (master spec section 6).
MUTABLE_COLUMNS: Final[dict[str, frozenset[str]]] = {
    "run": frozenset({"status", "error_json", "started_at", "finished_at"}),
    "strategy_family": frozenset({"status", "validation_touches"}),
    "split_policy": frozenset({"test_end_ts"}),
    "paper_session": frozenset({"started_at", "stopped_at", "status", "last_bar_ts"}),
    "evolution_run": frozenset({"status", "stop_reason", "n_evaluations", "finished_at"}),
    "candidate": frozenset(
        {
            "run_id",
            "fitness",
            "base_score",
            "penalty_product",
            "components_json",
            "penalties_json",
            "gate_failure",
            "rank",
            "survived",
            "behaviour_hash",
        }
    ),
    "candidate_promotion": frozenset({"run_id", "verdict_id"}),
    # `mutation` is deliberately absent: a mutation row is never mutable, because
    # it is the record INV-10 replays to reproduce a child from its parent.
}
