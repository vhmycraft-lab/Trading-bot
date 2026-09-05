"""INV-2: no API key, secret, token or password literal in the repository.

This complements the ``gitleaks`` pre-commit hook, which developers can skip
with ``--no-verify``; ``make check`` cannot be skipped.

Nothing in this file may contain a realistic-looking credential.  The probe
values used to prove the scanner works are assembled at run time.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCANNED_DIRS = ("src", "configs", "tests")
SCANNED_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".json", ".md", ".cfg", ".ini", ".env"}

#: An assignment of a long opaque value to a credential-shaped name.
# No leading \b: credential names are usually embedded in a longer identifier
# such as GLM_API_KEY, where "_" is a word character and \b would never match.
ASSIGNMENT_RE = re.compile(
    r"""(?ix)
    (api[_-]?key|apikey|secret|token|password|passwd|authorization|bearer)
    \s*[:=]\s*
    ["']?
    (?P<value>[A-Za-z0-9+/_\-]{20,})
    """
)

#: A provider-style credential, matching the redaction rule of spec section 17.
CREDENTIAL_RE = re.compile(r"(?i)\b(sk|key|token)[-_][a-z0-9]{20,}\b")

#: Values that look long and opaque but are structural, not secret.
PLACEHOLDER_VALUES = frozenset(
    {
        "your_key_here",
        "changeme",
        "REDACTED",
        "None",
        "null",
        "true",
        "false",
    }
)


def _scanned_files() -> list[Path]:
    files: list[Path] = []
    for directory in SCANNED_DIRS:
        root = REPO_ROOT / directory
        if not root.is_dir():
            continue
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix in SCANNED_SUFFIXES
            and "__pycache__" not in path.parts
        )
    return sorted(files)


def _findings(text: str) -> list[str]:
    hits: list[str] = []
    for match in ASSIGNMENT_RE.finditer(text):
        value = match.group("value")
        if value in PLACEHOLDER_VALUES:
            continue
        hits.append(match.group(0))
    hits.extend(match.group(0) for match in CREDENTIAL_RE.finditer(text))
    return hits


def test_repository_contains_no_credential_literals() -> None:
    offenders: list[str] = []
    for path in _scanned_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for hit in _findings(text):
            offenders.append(f"{path.relative_to(REPO_ROOT)}: {hit[:40]}")
    assert not offenders, "INV-2: credential-shaped literal found: " + "; ".join(offenders)


def test_scanner_detects_a_synthetic_credential() -> None:
    """The scan above is only meaningful if it can actually fire."""
    synthetic_key = "sk-" + ("0a1b" * 8)
    assert _findings(synthetic_key)

    name = "GLM_API" + "_KEY"
    assignment = f"{name}={'9f' * 20}"
    assert _findings(assignment)


def test_scanner_ignores_ordinary_source() -> None:
    assert not _findings("fee_bps = 10.0  # taker fee in basis points")
    assert not _findings("api_key: ''")


def test_env_example_has_no_values() -> None:
    """`.env.example` documents names only; every value must be empty."""
    example = REPO_ROOT / ".env.example"
    assert example.is_file()
    for line in example.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name, _, value = stripped.partition("=")
        assert value == "", f"{name} in .env.example must have an empty value"


def test_dotenv_is_ignored_and_not_tracked() -> None:
    ignore_text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in ignore_text.split()

    try:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", ".env"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:  # pragma: no cover - git is present in CI
        pytest.skip("git is not available")
    assert tracked.returncode != 0, ".env must never be tracked by git"
