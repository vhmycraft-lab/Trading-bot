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

    def lineage(self, strategy_id: str) -> list[StrategyVersionRecord]:
        """Every ancestor of ``strategy_id``, oldest first, ending with itself."""
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

    def query_runs(self, **filters: Any) -> list[RunRecord]:
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

    def record_llm_interaction(self, **fields: Any) -> Any:
        """Record that a model was asked something, and what it cost."""
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
