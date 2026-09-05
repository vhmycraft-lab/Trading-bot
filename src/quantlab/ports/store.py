"""Persistence ports (master spec section 11.1).

Only the dataset and split methods are declared so far: they are the ones the
data subsystem needs, and their record types exist.  The run, metric, verdict
and LLM methods of spec section 11.1 join this protocol in the store phase,
once the types they carry are defined.  Growing a ``Protocol`` is safe —
adapters satisfy it structurally, so nothing needs to be rewritten.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from quantlab.core.splits import SplitPolicy

__all__ = ["ArtifactStore", "DatasetRecord", "ExperimentStore"]


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
class ExperimentStore(Protocol):
    """Append-only record of everything an experiment depended on."""

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
