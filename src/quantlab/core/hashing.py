"""Canonical JSON and SHA-256 identity helpers (master spec section 1.3).

Fixed decisions encoded here:

* canonical JSON is ``json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)``
* hashing is SHA-256, hex
* ids are the first 16 hex characters of the digest

Every id in the database is produced by one of the functions below so that ids
are reproducible from stored inputs alone (INV-7).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

__all__ = [
    "ID_LENGTH",
    "canonical_json",
    "dataset_id",
    "file_sha256",
    "run_id",
    "sha256_hex",
    "short_id",
    "split_id",
    "strategy_id",
]

#: Number of hex characters kept from a digest when forming an id.
ID_LENGTH: Final[int] = 16

_READ_CHUNK: Final[int] = 1024 * 1024


def canonical_json(obj: Any) -> str:
    """Serialise ``obj`` deterministically.

    Key order, separators and unicode handling are fixed so that two logically
    equal objects always produce the same string, and therefore the same hash.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(data: str | bytes) -> str:
    """Return the full hex SHA-256 digest of ``data`` (UTF-8 encoded if a string)."""
    payload = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(payload).hexdigest()


def short_id(data: str | bytes) -> str:
    """Return the first :data:`ID_LENGTH` hex characters of ``sha256(data)``."""
    return sha256_hex(data)[:ID_LENGTH]


def file_sha256(path: str | Path) -> str:
    """Return the full hex SHA-256 digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_READ_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_id(files: Iterable[tuple[str, str]], symbol: str, timeframe: str) -> str:
    """Dataset id per spec section 7.2.

    ``files`` yields ``(relative_path, sha256_hex)`` pairs; they are sorted by
    path so that discovery order cannot change the id.
    """
    joined = "|".join(f"{path}:{digest}" for path, digest in sorted(files))
    return short_id(f"{joined}|{symbol}|{timeframe}")


def split_id(policy_yaml: str, dataset: str) -> str:
    """Split id per spec section 6: ``sha256(canonical yaml + dataset_id)[:16]``."""
    return short_id(f"{policy_yaml}{dataset}")


def strategy_id(code: str | bytes) -> str:
    """Strategy version id per spec section 6: ``sha256(code bytes)[:16]``."""
    return short_id(code)


def run_id(
    *,
    strategy: str,
    params: Mapping[str, Any],
    dataset: str,
    split: str,
    segment: str,
    backtest_config_hash: str,
    seed: int,
    engine_name: str,
    engine_version: str,
) -> str:
    """Run id per spec section 11.2.

    The identity is the full set of inputs a run depends on, so an identical
    request hits the cache and a changed input necessarily produces a new run.
    """
    parts: Sequence[str] = (
        strategy,
        canonical_json(dict(params)),
        dataset,
        split,
        segment,
        backtest_config_hash,
        str(seed),
        engine_name,
        engine_version,
    )
    return short_id("|".join(parts))
