"""Persistence ports (master spec section 11.1).

The record types are declared structurally rather than imported from the store
adapter: a caller needs to read a run's status and a strategy's parentage, and
nothing more. Declaring only that keeps ``core`` and every consumer free of
SQLAlchemy, so the store could be replaced without touching them (INV-8).

The evolution methods of section 11.1 (`create_evolution_run`, `add_candidate`,
`add_mutations`, `record_promotion`, `ancestry`, ...) join this protocol with the
tables they carry, in migration 0002. Growing a ``Protocol`` is safe: adapters
satisfy it structurally, so nothing already written needs to change.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import pandas as pd

from quantlab.core.splits import SplitPolicy
from quantlab.core.types import Trade

__all__ = [
    "ArtifactStore",
    "DatasetRecord",
    "ExperimentRecord",
    "ExperimentStore",
    "FamilyRecord",
    "RunRecord",
    "SourceStore",
    "StrategyVersionRecord",
    "VerdictRecord",
]


@runtime_checkable
class DatasetRecord(Protocol):
    """The identity and extent of one stored dataset."""

    dataset_id: str
    exchange: str
    symbol: str
    timeframe: str
    start_ts: int
    end_ts: int
    n_bars: int


@runtime_checkable
class FamilyRecord(Protocol):
    """A line of enquiry, and how much of the validation budget it has spent."""

    family_id: str
    name: str
    origin: str
    status: str
    validation_touches: int


@runtime_checkable
class StrategyVersionRecord(Protocol):
    """One immutable version of a strategy's source."""

    strategy_id: str
    family_id: str
    parent_strategy_id: str | None
    #: Where the immutable copy lives, as a key into a :class:`SourceStore`.
    code_path: str
    code_sha256: str
    class_name: str
    style: str
    logic_lines: int


@runtime_checkable
class ExperimentRecord(Protocol):
    """One campaign, one purpose, one configuration."""

    experiment_id: str
    campaign: str
    purpose: str
    config_hash: str
    seed: int


@runtime_checkable
class RunRecord(Protocol):
    """One evaluation of one strategy on one segment."""

    run_id: str
    experiment_id: str
    strategy_id: str
    dataset_id: str
    split_id: str
    segment: str
    status: str
    artifact_dir: str


@runtime_checkable
class VerdictRecord(Protocol):
    """What validation decided, and the thresholds it decided against."""

    verdict_id: str
    strategy_id: str
    split_id: str
    verdict: str
    overfit_score: float


@runtime_checkable
class ExperimentStore(Protocol):
    """Append-only record of everything an experiment depended on.

    Rows are append-only except for the columns spec section 6 names as mutable;
    an update outside those raises ``ImmutableRowError``. The single exception is
    :meth:`delete_run`, which removes a *failed* run and logs at WARNING.
    """

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
    ) -> DatasetRecord:
        """Register a dataset, returning the existing row if the id is known."""
        ...

    def get_or_create_split(self, policy: SplitPolicy) -> SplitPolicy:
        """Register a split policy, returning the stored one if it already exists."""
        ...

    def freeze_test_end(self, split_id: str, end_ts: int) -> SplitPolicy:
        """Pin a split's open test end the first time the lockbox is used."""
        ...

    # -- strategies ---------------------------------------------------------
    def create_family(
        self, *, name: str, origin: str, description: str = ..., family_id: str | None = ...
    ) -> FamilyRecord:
        """Register a strategy family."""
        ...

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
        parent_strategy_id: str | None = ...,
        llm_interaction_id: str | None = ...,
    ) -> StrategyVersionRecord:
        """Record one immutable version of a strategy's source."""
        ...

    def get_strategy_version(self, strategy_id: str) -> StrategyVersionRecord | None:
        """One registered version, or ``None``.

        Distinct from :meth:`lineage`: asking for a single row should not walk a
        parent chain, and a caller that only wants to know whether an id is
        registered should not have to interpret an ancestry to find out.
        """
        ...

    def lineage(self, strategy_id: str) -> Sequence[StrategyVersionRecord]:
        """Every ancestor of ``strategy_id``, oldest first, ending with itself."""
        ...

    def find_family(self, family_id: str) -> Any:
        """The strategy family with this id, or ``None``."""
        ...

    def record_verdict(
        self,
        *,
        verdict_id: str,
        strategy_id: str,
        split_id: str,
        params_json: str,
        verdict: str,
        overfit_score: float,
        hard_gates_json: str,
        soft_checks_json: str,
        thresholds_json: str,
        n_trials_accounted: int,
    ) -> Any:
        """Record one validation verdict. Append-only (spec section 6)."""
        ...

    def verdicts_for(self, strategy_id: str) -> Sequence[Any]:
        """Every verdict recorded for a strategy version, oldest first."""
        ...

    def increment_validation_touches(self, family_id: str) -> int:
        """Count one more look at the validation partition; return the new total."""
        ...

    def set_family_status(self, family_id: str, status: str) -> FamilyRecord:
        """Open, freeze or close a family."""
        ...

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
        experiment_id: str | None = ...,
    ) -> ExperimentRecord:
        """Register an experiment."""
        ...

    def metrics_for(self, run_id: str) -> Mapping[str, float | None]:
        """Every metric recorded for a run, by name.

        The way a *cached* run's numbers are read back: section 11.2 returns the
        stored record without re-executing, so a caller that needs the metrics of
        a run it did not just produce reads them from here rather than simulating
        again to find out what it already knows.
        """
        ...

    def trades_for(self, run_id: str) -> Sequence[Any]:
        """A run's trade ledger, in execution order."""
        ...

    def find_run(self, run_id: str) -> RunRecord | None:
        """The run with this id, or ``None``.  The cache lookup of section 11.2."""
        ...

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
        cost_multiplier: float = ...,
    ) -> RunRecord:
        """Open a run in ``pending``.  ``run_id`` is the caller's, per section 11.2."""
        ...

    def finish_run(
        self,
        run_id: str,
        status: str,
        metrics: Mapping[str, float | None] | None = ...,
        trades: Sequence[Trade] | None = ...,
        error: Mapping[str, Any] | None = ...,
    ) -> RunRecord:
        """Close a run and write everything it produced, in one transaction."""
        ...

    def query_runs(self, **filters: Any) -> Sequence[RunRecord]:
        """Runs matching every supplied filter, oldest first."""
        ...

    def delete_run(self, run_id: str, *, confirm: Literal[True]) -> None:
        """Remove a *failed* run and its results.  Logs at WARNING (section 6)."""
        ...

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
        verdict_id: str | None = ...,
    ) -> VerdictRecord:
        """Record a validation verdict, with the thresholds it was decided against."""
        ...

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
        seed: int | None = ...,
        interaction_id: str | None = ...,
    ) -> Any:
        """Record that a model was asked something, and what it cost."""
        ...

    # -- evolution (spec section 13) ---------------------------------------
    # Spelled out rather than declared as ``**fields``: a protocol that accepts
    # anything checks nothing, and these are the calls that write the lineage
    # INV-10 replays and the promotions INV-9 gates on.
    def create_evolution_run(
        self,
        *,
        evolution_id: str,
        experiment_id: str,
        campaign: str,
        dataset_id: str,
        split_id: str,
        population_size: int,
        n_survivors: int,
        n_offspring: int,
        n_immigrants: int,
        max_generations: int,
        seed: int,
        fitness_config_json: str,
        mutation_config_json: str,
        diversity_config_json: str,
    ) -> Any:
        """Open an evolution run, recording the search it is about to perform."""
        ...

    def finish_evolution_run(
        self, evolution_id: str, status: str, stop_reason: str | None = ...
    ) -> Any:
        """Close an evolution run and say why it stopped."""
        ...

    def count_evaluation(self, evolution_id: str, n: int = ...) -> int:
        """Add to the run's evaluation count; return the new total (section 14.4's ``M``)."""
        ...

    def add_generation(
        self,
        *,
        generation_id: str,
        evolution_id: str,
        gen_index: int,
        diversity: float,
        n_evaluated: int,
        n_cache_hits: int,
        n_rejected_by_gate: int,
        n_immigrants_used: int,
        stats_json: str = ...,
        best_fitness: float | None = ...,
        median_fitness: float | None = ...,
        mean_fitness: float | None = ...,
    ) -> Any:
        """Record one completed generation. Append-only."""
        ...

    def add_candidate(
        self,
        *,
        candidate_id: str,
        evolution_id: str,
        generation_id: str,
        gen_index: int,
        strategy_id: str,
        params_json: str,
        kind: str,
        origin: str,
        genome_json: str | None = ...,
        parent_candidate_id: str | None = ...,
        signature_json: str = ...,
    ) -> Any:
        """Record a candidate as it enters the population, before evaluation."""
        ...

    def score_candidate(
        self,
        candidate_id: str,
        *,
        run_id: str | None = ...,
        fitness: float | None = ...,
        base_score: float | None = ...,
        penalty_product: float | None = ...,
        components_json: str | None = ...,
        penalties_json: str | None = ...,
        gate_failure: str | None = ...,
        rank: int | None = ...,
        survived: bool | None = ...,
        behaviour_hash: str | None = ...,
    ) -> Any:
        """Write a candidate's evaluation into the columns section 6 allows to change."""
        ...

    def add_mutations(self, candidate_id: str, mutations: Sequence[Mapping[str, Any]]) -> None:
        """Record the edits that produced a child. Append-only; never updated (INV-10)."""
        ...

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
    ) -> Any:
        """Record a completed parameter study and the trials it contributed to ``M``."""
        ...

    def record_promotion(
        self,
        *,
        candidate_id: str,
        evolution_id: str,
        gen_index: int,
        segment: str,
        reason: str,
        promotion_id: str | None = ...,
    ) -> Any:
        """Record a promotion **before** the run it authorises executes (INV-9)."""
        ...

    def promotions_for(self, candidate_id: str) -> Sequence[Any]:
        """Every promotion recorded for a candidate, oldest first.

        What a validation run consults before it is created: a candidate with no
        row here has not been through the recorded, budgeted act of section 13.7
        (INV-9).
        """
        ...

    def candidates_for(self, evolution_id: str, gen_index: int | None = ...) -> Sequence[Any]:
        """A run's candidates, deterministically ordered."""
        ...

    def find_evolution_run(self, evolution_id: str) -> Any:
        """The evolution run with this id, or ``None``."""
        ...

    def generations_for(self, evolution_id: str) -> Sequence[Any]:
        """A run's generations, oldest first."""
        ...

    def mutations_for(self, candidate_id: str) -> Sequence[Any]:
        """The edits that produced a candidate, in application order (INV-10)."""
        ...

    def ancestry(self, candidate_id: str) -> Sequence[Any]:
        """Every ancestor of a candidate, oldest first, ending with itself."""
        ...

    def descendants(self, candidate_id: str) -> Sequence[Any]:
        """Every candidate reachable from this one by following parent links down."""
        ...

    def record_lockbox_access(
        self,
        *,
        strategy_id: str,
        family_id: str,
        os_user: str,
        reason: str,
        run_id: str | None = ...,
        access_id: str | None = ...,
    ) -> Any:
        """Record a look at the test partition, whether or not it passed."""
        ...

    # -- hidden training environments (Project Rome sections 18, 24) --------
    def record_training_environment(
        self, *, evolution_id: str, environment: Mapping[str, Any]
    ) -> Any:
        """File one generation's environment, before that generation runs."""
        ...

    def find_training_environment(self, evolution_id: str, gen_index: int) -> Any | None:
        """The environment a generation was recorded with, or ``None``.

        Declared on the port because the evolution loop reads it on resume and
        ``evolution/`` may not import an adapter (INV-8). A resumed generation
        replays the environment it was recorded with; drawing a fresh one would
        make the resumed run a different search.
        """
        ...

    def training_environments_for(self, evolution_id: str) -> Sequence[Any]:
        """Every environment recorded for a run, in generation order."""
        ...


@runtime_checkable
class ArtifactStore(Protocol):
    """Files produced by a run, addressed by run id."""

    def dir_for(self, run_id: str) -> Path:
        """Return (creating if needed) the directory holding a run's artifacts."""
        ...

    def write_parquet(self, run_id: str, name: str, df: pd.DataFrame) -> Path:
        """Write a DataFrame as Parquet with fixed options, for byte-stable goldens."""
        ...

    def write_json(self, run_id: str, name: str, obj: Any) -> Path:
        """Write an object as canonical JSON."""
        ...

    def read_json(self, run_id: str, name: str) -> Any:
        """Read back an object written by :meth:`write_json`."""
        ...


@runtime_checkable
class SourceStore(Protocol):
    """The immutable copy of a strategy's source, addressed by ``strategy_id``.

    Separate from :class:`ArtifactStore`, which is keyed by ``run_id`` and holds
    what a run *produced* (spec section 11.3). Strategy source is an input, it
    outlives every run that cites it, and its name is a hash of its own bytes —
    filing it under ``artifacts/runs/`` would make it look like a run that never
    happened.

    A strategy version is append-only (section 6), so writing the same id twice
    must be idempotent and writing *different* bytes under an existing id is a
    contradiction the implementation is required to refuse.
    """

    def write_source(self, strategy_id: str, source: str) -> str:
        """Store ``source`` under ``strategy_id`` and return its ``code_path``."""
        ...

    def read_source(self, strategy_id: str) -> str:
        """Return the stored source, or raise if there is none."""
        ...

    def path_for(self, strategy_id: str) -> Path:
        """The file the source occupies, whether or not it exists yet."""
        ...

    def exists(self, strategy_id: str) -> bool:
        """Whether a source is stored under ``strategy_id``."""
        ...
