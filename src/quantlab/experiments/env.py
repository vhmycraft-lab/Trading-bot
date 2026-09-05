"""What the machine looked like when a run happened (master spec section 11.3).

``env.json`` is the part of a run's record that nothing else can reconstruct
later. A metric can be recomputed from stored inputs; the version of NumPy that
produced it cannot, and neither can the fact that the working tree had uncommitted
changes. INV-7 asks whether a run reproduces, and this is the file that says what
it would have to reproduce *on*.

A dirty tree is allowed and flagged rather than refused: research happens on
dirty trees, and a platform that refused to record one would simply be lied to.
"""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from datetime import UTC, datetime
from typing import Any, Final

__all__ = ["INTERESTING_PACKAGES", "environment_report", "git_state"]

#: Packages whose version can change a number. Recorded exactly; everything else
#: in the environment is noise that would make two identical runs look different.
INTERESTING_PACKAGES: Final[tuple[str, ...]] = (
    "numpy",
    "pandas",
    "pyarrow",
    "sqlalchemy",
    "pydantic",
    "quantlab",
)

_GIT_TIMEOUT_S: Final[float] = 5.0


def _git(*args: str) -> str | None:
    """Run one git command, or return ``None`` if git cannot answer.

    Never raises: an environment with no git, no repository, or a slow disk still
    has to be able to record a run. The absence is recorded as an absence.
    """
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", *args],  # noqa: S607 - resolved from PATH deliberately
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def git_state() -> tuple[str, bool]:
    """The current commit and whether the working tree has uncommitted changes.

    Returns ``("", False)`` when git cannot answer — an unknown commit is recorded
    as unknown rather than guessed at.
    """
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return "", False
    status = _git("status", "--porcelain")
    return commit, bool(status)


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in INTERESTING_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:  # pragma: no cover - all are installed here
            continue
    return versions


def environment_report(*, engine_version: str, created_at: int | None = None) -> dict[str, Any]:
    """The contents of ``env.json`` (spec section 11.3).

    ``created_at`` is when the environment was recorded, in milliseconds UTC — the
    wall clock, not anything derived from the bars: it says when this machine
    looked like this, which is the only question the file answers.
    """
    commit, dirty = git_state()
    stamp = int(datetime.now(UTC).timestamp() * 1000) if created_at is None else int(created_at)
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _package_versions(),
        "git_commit": commit,
        "git_dirty": dirty,
        "engine_version": engine_version,
        "created_at": stamp,
    }
