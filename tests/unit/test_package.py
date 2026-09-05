"""Package metadata and the repository layout promised by the spec."""

from __future__ import annotations

import tomllib
from pathlib import Path

import quantlab

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_version_is_declared_once() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert quantlab.__version__ == pyproject["project"]["version"] == "0.1.0"


def test_python_requirement_is_pinned_to_312() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["requires-python"] == ">=3.12,<3.13"


def test_console_script_points_at_the_typer_app() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["scripts"]["quantlab"] == "quantlab.cli:app"


def test_declared_dependencies_match_the_spec() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = {
        dep.split("~=")[0].split("[")[0].strip() for dep in pyproject["project"]["dependencies"]
    }
    expected = {
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "pydantic",
        "pydantic-settings",
        "pyyaml",
        "sqlalchemy",
        "alembic",
        "typer",
        "rich",
        "structlog",
        "httpx",
        "tenacity",
        "optuna",
        "ccxt",
        "keyring",
        "python-dotenv",
        "matplotlib",
        "websockets",
    }
    assert declared == expected, "spec section 0.3: no dependency change without an ADR"


def test_required_top_level_files_exist() -> None:
    for name in (
        ".env.example",
        ".gitignore",
        ".pre-commit-config.yaml",
        "Makefile",
        "README.md",
        "pyproject.toml",
        "uv.lock",
        "alembic.ini",
    ):
        assert (REPO_ROOT / name).is_file(), name


def test_required_directories_exist() -> None:
    for name in (
        "configs/splits",
        "docs/DECISIONS",
        "migrations/versions",
        "scripts",
        "src/quantlab/core",
        "src/quantlab/ports",
        "src/quantlab/adapters/secrets",
        "src/quantlab/adapters/store",
        "src/quantlab/cli",
        "tests/unit",
        "tests/property",
        "tests/golden",
        "tests/leakage",
        "tests/synthetic",
        "tests/integration",
    ):
        assert (REPO_ROOT / name).is_dir(), name


def test_documentation_is_present() -> None:
    for name in (
        "docs/ARCHITECTURE.md",
        "docs/SECURITY.md",
        "docs/DECISIONS/0001-record-architecture-decisions.md",
    ):
        assert (REPO_ROOT / name).is_file(), name


def test_gitignore_covers_the_required_paths() -> None:
    text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    for entry in (".env", "*.db", "data/*", "artifacts/*", "strategies/generated/*"):
        assert entry in text, entry
