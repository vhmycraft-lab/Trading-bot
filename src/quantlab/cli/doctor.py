"""``quantlab doctor`` — check that this machine can run QuantLab.

Reports on the interpreter, configuration, working directories, secrets and the
database, then exits with the code of the most severe problem found (spec
section 18.2):  2 for configuration/secret problems, 7 if a Binance key with
trading permissions is detected (spec section 21.2), 0 when everything is fine.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rich.console import Console
from rich.table import Table

from quantlab.adapters.store.sqlite import create_db_engine, current_revision, missing_tables
from quantlab.core.config import AppConfig
from quantlab.core.errors import EXIT_CODES
from quantlab.ports.secrets import OPTIONAL_SECRETS, REQUIRED_SECRETS, Secrets

__all__ = ["Check", "Status", "run_checks", "worst_exit_code"]

Status = Literal["ok", "warn", "fail"]

_STYLES: dict[Status, str] = {"ok": "green", "warn": "yellow", "fail": "red"}
_SYMBOLS: dict[Status, str] = {"ok": "OK", "warn": "WARN", "fail": "FAIL"}

MIN_PYTHON = (3, 12)
MAX_PYTHON_EXCLUSIVE = (3, 13)


@dataclass(frozen=True, slots=True)
class Check:
    """One diagnostic line."""

    name: str
    status: Status
    detail: str
    #: Exit code to use if this is the worst check; ``None`` means "do not fail".
    exit_code: int | None = None


def _check_python() -> Check:
    version = sys.version_info
    text = platform.python_version()
    if MIN_PYTHON <= (version.major, version.minor) < MAX_PYTHON_EXCLUSIVE:
        return Check("python", "ok", f"{text} on {platform.system()}")
    return Check(
        "python",
        "fail",
        f"{text}; QuantLab requires >=3.12,<3.13",
        EXIT_CODES["ConfigError"],
    )


def _check_config(config: AppConfig) -> Check:
    return Check("config", "ok", f"config_hash={config.config_hash[:16]}...")


def _check_directory(label: str, path: Path) -> Check:
    if not path.exists():
        return Check(label, "warn", f"{path} does not exist yet (it will be created on demand)")
    if not path.is_dir():
        return Check(
            label, "fail", f"{path} exists but is not a directory", EXIT_CODES["ConfigError"]
        )
    probe = path / ".quantlab-write-probe"
    try:
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return Check(label, "fail", f"{path} is not writable: {exc}", EXIT_CODES["ConfigError"])
    return Check(label, "ok", str(path))


def _check_secret(secrets: Secrets, name: str, *, required: bool) -> Check:
    present = secrets.try_get(name) is not None
    if present:
        # The value itself is never read into a message.
        return Check(f"secret:{name}", "ok", "found")
    if required:
        return Check(
            f"secret:{name}",
            "fail",
            "missing; add it to the macOS Keychain (service 'quantlab') or to .env",
            EXIT_CODES["ConfigError"],
        )
    return Check(f"secret:{name}", "warn", "not set (optional)")


def _check_trade_permissions(secrets: Secrets) -> Check | None:
    """Refuse to proceed if a configured Binance key can trade (spec section 21.2)."""
    if secrets.try_get("BINANCE_API_KEY") is None:
        return None
    try:
        import ccxt
    except ImportError:  # pragma: no cover - ccxt is a hard dependency
        return Check("binance:permissions", "warn", "ccxt unavailable; cannot verify")

    try:
        client = ccxt.binance(
            {
                "apiKey": secrets.get("BINANCE_API_KEY"),
                "secret": secrets.try_get("BINANCE_API_SECRET") or "",
                "enableRateLimit": True,
            }
        )
        restrictions = client.sapi_get_account_apirestrictions()
    except Exception as exc:
        return Check(
            "binance:permissions",
            "warn",
            f"could not verify API restrictions ({type(exc).__name__}); "
            "verify manually that trading is disabled",
        )

    if restrictions.get("enableSpotAndMarginTrading"):
        return Check(
            "binance:permissions",
            "fail",
            "the configured Binance key has spot/margin TRADING enabled; "
            "QuantLab refuses to run with a key that can place orders",
            EXIT_CODES["LockboxViolation"],
        )
    return Check("binance:permissions", "ok", "read-only key")


def _check_database(config: AppConfig) -> list[Check]:
    db_path = Path(config.project.db_path)
    if str(db_path) != ":memory:" and not db_path.exists():
        return [
            Check(
                "database",
                "warn",
                f"{db_path} does not exist yet; run `quantlab db upgrade`",
            )
        ]
    engine = create_db_engine(db_path)
    try:
        revision = current_revision(engine)
        absent = missing_tables(engine)
    finally:
        engine.dispose()

    if revision is None:
        return [Check("database", "warn", f"{db_path} is not managed by alembic yet")]
    checks = [Check("database", "ok", f"{db_path} at revision {revision}")]
    if absent:
        checks.append(
            Check(
                "database:schema",
                "warn",
                f"missing tables: {', '.join(absent)}; run `quantlab db upgrade`",
            )
        )
    else:
        checks.append(Check("database:schema", "ok", "all tables present"))
    return checks


def run_checks(config: AppConfig, secrets: Secrets) -> list[Check]:
    """Run every diagnostic and return the results in display order."""
    checks: list[Check] = [_check_python(), _check_config(config)]
    checks.append(_check_directory("dir:data", Path(config.project.data_dir)))
    checks.append(_check_directory("dir:artifacts", Path(config.project.artifacts_dir)))
    checks.extend(_check_database(config))
    for name in REQUIRED_SECRETS:
        checks.append(_check_secret(secrets, name, required=True))
    for name in OPTIONAL_SECRETS:
        checks.append(_check_secret(secrets, name, required=False))
    permissions = _check_trade_permissions(secrets)
    if permissions is not None:
        checks.append(permissions)
    return checks


def worst_exit_code(checks: list[Check]) -> int:
    """Return the exit code of the most severe failing check (0 if none fail)."""
    codes = [c.exit_code for c in checks if c.status == "fail" and c.exit_code is not None]
    if not codes:
        return 0
    # Lockbox/permission problems outrank plain configuration problems.
    return max(codes)


def render(checks: list[Check], console: Console) -> None:
    """Print the checks as a table."""
    table = Table(title="quantlab doctor", header_style="bold", show_lines=False)
    table.add_column("check", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("detail", overflow="fold")
    for check in checks:
        table.add_row(
            check.name,
            f"[{_STYLES[check.status]}]{_SYMBOLS[check.status]}[/]",
            check.detail,
        )
    console.print(table)

    failures = [c for c in checks if c.status == "fail"]
    if failures:
        console.print(f"[red]{len(failures)} check(s) failed.[/] Fix them and run doctor again.")
    else:
        console.print("[green]All required checks passed.[/]")
