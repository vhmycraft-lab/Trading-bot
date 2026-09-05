"""Files produced by a run, addressed by run id (master spec sections 11.1, 11.3).

Every run gets one directory, ``<artifacts_dir>/runs/<run_id>/``, and everything
it produced lives there: parameters, the equity and position series, the trade
ledger, metrics, the engine log, the environment it ran in. The database records
*that* a run happened and what it scored; this records what it actually did.

Two properties matter more than convenience here:

**Byte stability.** Parquet is written with fixed options and JSON in canonical
form, so re-running an identical backtest produces identical files. INV-7 is
checked by comparing metrics, but a golden test that compares bytes is a stronger
statement, and it only works if nothing incidental — dictionary order, compression
level, a pandas index — varies between writes.

**Containment.** A run id is a hash and a name is chosen by the platform, but both
are checked before they reach the filesystem. A store that concatenates untrusted
strings into paths is one bad identifier away from writing outside its own tree.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import pandas as pd

from quantlab.core.errors import StoreError
from quantlab.core.hashing import canonical_json

__all__ = ["ARTIFACT_PARQUET_KWARGS", "RUNS_SUBDIR", "FileArtifactStore"]

#: Fixed for byte stability. Changing any of these invalidates every golden file.
ARTIFACT_PARQUET_KWARGS: Final[dict[str, Any]] = {
    "engine": "pyarrow",
    "compression": "zstd",
    "index": False,
}

RUNS_SUBDIR: Final[str] = "runs"


def _safe_component(value: str, *, what: str) -> str:
    """Reject anything that is not a single, ordinary path segment.

    Run ids are hashes and artefact names are platform constants, so this should
    never fire. That is the point: it costs one comparison and removes a whole
    class of "how did that file get there" from consideration.
    """
    text = str(value)
    if not text or text in (".", ".."):
        raise StoreError(f"{what} must not be empty or a directory reference", value=text)
    if "/" in text or "\\" in text or "\x00" in text:
        raise StoreError(f"{what} must be a single path component", value=text)
    if Path(text).is_absolute() or Path(text).name != text:
        raise StoreError(f"{what} must be a single path component", value=text)
    return text


class FileArtifactStore:
    """A directory tree of run artefacts. Implements :class:`ports.store.ArtifactStore`."""

    __slots__ = ("root",)

    def __init__(self, artifacts_dir: str | Path) -> None:
        self.root = Path(artifacts_dir).expanduser().resolve() / RUNS_SUBDIR

    def __repr__(self) -> str:
        return f"FileArtifactStore(root={str(self.root)!r})"

    def dir_for(self, run_id: str) -> Path:
        """Return the run's directory, creating it if it does not exist."""
        directory = self.root / _safe_component(run_id, what="run_id")
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def path_for(self, run_id: str, name: str) -> Path:
        """Return the path one artefact would occupy, without creating the file."""
        return self.dir_for(run_id) / _safe_component(name, what="artifact name")

    def write_parquet(self, run_id: str, name: str, df: pd.DataFrame) -> Path:
        """Write ``df`` as Parquet with the fixed options."""
        path = self.path_for(run_id, name)
        df.to_parquet(path, **ARTIFACT_PARQUET_KWARGS)
        return path

    def read_parquet(self, run_id: str, name: str) -> pd.DataFrame:
        """Read back a frame written by :meth:`write_parquet`."""
        return pd.read_parquet(self._existing(run_id, name))

    def write_json(self, run_id: str, name: str, obj: Any) -> Path:
        """Write ``obj`` as canonical JSON, with a trailing newline."""
        path = self.path_for(run_id, name)
        path.write_text(canonical_json(obj) + "\n", encoding="utf-8")
        return path

    def read_json(self, run_id: str, name: str) -> Any:
        """Read back an object written by :meth:`write_json`."""
        return json.loads(self._existing(run_id, name).read_text(encoding="utf-8"))

    def write_text(self, run_id: str, name: str, text: str) -> Path:
        """Write plain text, e.g. ``engine_log.txt`` (spec section 11.3)."""
        path = self.path_for(run_id, name)
        path.write_text(text, encoding="utf-8")
        return path

    def read_text(self, run_id: str, name: str) -> str:
        return self._existing(run_id, name).read_text(encoding="utf-8")

    def exists(self, run_id: str, name: str) -> bool:
        return self.path_for(run_id, name).is_file()

    def listdir(self, run_id: str) -> tuple[str, ...]:
        """Artefact names present for a run, sorted."""
        return tuple(sorted(p.name for p in self.dir_for(run_id).iterdir() if p.is_file()))

    def _existing(self, run_id: str, name: str) -> Path:
        path = self.path_for(run_id, name)
        if not path.is_file():
            raise StoreError("artifact not found", run_id=run_id, name=name, path=str(path))
        return path
