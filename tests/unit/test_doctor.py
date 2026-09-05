"""Diagnostics logic (master spec sections 18.2 and 21.2)."""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console

from quantlab.cli import doctor as doctor_module
from quantlab.cli.doctor import Check, run_checks, worst_exit_code
from quantlab.core.config import AppConfig, load_config
from quantlab.core.errors import EXIT_CODES


class StubSecrets:
    def __init__(self, values: dict[str, str] | None = None) -> None:
        self.values = values or {}

    def try_get(self, name: str) -> str | None:
        return self.values.get(name)

    def get(self, name: str) -> str:
        return self.values[name]


def _config(tmp_path: Path, default_config_path: Path) -> AppConfig:
    return load_config(
        [],
        [
            f"project.db_path={tmp_path / 'quantlab.db'}",
            f"project.data_dir={tmp_path / 'data'}",
            f"project.artifacts_dir={tmp_path / 'artifacts'}",
        ],
        default_path=default_config_path,
    )


def _by_name(checks: list[Check]) -> dict[str, Check]:
    return {check.name: check for check in checks}


def test_missing_required_secret_fails_with_exit_code_two(
    tmp_path: Path, default_config_path: Path
) -> None:
    checks = run_checks(_config(tmp_path, default_config_path), StubSecrets())
    secret = _by_name(checks)["secret:GLM_API_KEY"]
    assert secret.status == "fail"
    assert secret.exit_code == EXIT_CODES["ConfigError"]
    assert worst_exit_code(checks) == 2


def test_present_secret_passes_without_revealing_the_value(
    tmp_path: Path, default_config_path: Path
) -> None:
    marker = "q" * 20
    checks = run_checks(
        _config(tmp_path, default_config_path), StubSecrets({"GLM_API_KEY": marker})
    )
    secret = _by_name(checks)["secret:GLM_API_KEY"]
    assert secret.status == "ok"
    assert marker not in secret.detail


def test_optional_secrets_only_warn(tmp_path: Path, default_config_path: Path) -> None:
    checks = _by_name(run_checks(_config(tmp_path, default_config_path), StubSecrets()))
    assert checks["secret:BINANCE_API_KEY"].status == "warn"
    assert checks["secret:BINANCE_API_SECRET"].status == "warn"


def test_python_and_config_checks_pass(tmp_path: Path, default_config_path: Path) -> None:
    config = _config(tmp_path, default_config_path)
    checks = _by_name(run_checks(config, StubSecrets()))
    assert checks["python"].status == "ok"
    assert checks["config"].status == "ok"
    assert config.config_hash[:16] in checks["config"].detail


def test_absent_directories_warn_rather_than_fail(
    tmp_path: Path, default_config_path: Path
) -> None:
    checks = _by_name(run_checks(_config(tmp_path, default_config_path), StubSecrets()))
    assert checks["dir:data"].status == "warn"
    assert checks["dir:artifacts"].status == "warn"


def test_existing_writable_directory_passes(tmp_path: Path, default_config_path: Path) -> None:
    (tmp_path / "data").mkdir()
    checks = _by_name(run_checks(_config(tmp_path, default_config_path), StubSecrets()))
    assert checks["dir:data"].status == "ok"


def test_a_file_where_a_directory_belongs_fails(tmp_path: Path, default_config_path: Path) -> None:
    (tmp_path / "data").write_text("not a directory", encoding="utf-8")
    checks = _by_name(run_checks(_config(tmp_path, default_config_path), StubSecrets()))
    assert checks["dir:data"].status == "fail"


def test_database_states(tmp_path: Path, default_config_path: Path) -> None:
    config = _config(tmp_path, default_config_path)
    checks = _by_name(run_checks(config, StubSecrets()))
    assert checks["database"].status == "warn"

    from quantlab.adapters.store.sqlite import create_db_engine, upgrade_to_head

    engine = create_db_engine(config.project.db_path)
    try:
        upgrade_to_head(engine)
    finally:
        engine.dispose()

    checks = _by_name(run_checks(config, StubSecrets()))
    assert checks["database"].status == "ok"
    assert "0001" in checks["database"].detail
    assert checks["database:schema"].status == "ok"


def test_unmanaged_database_warns(tmp_path: Path, default_config_path: Path) -> None:
    config = _config(tmp_path, default_config_path)
    Path(config.project.db_path).write_bytes(b"")
    checks = _by_name(run_checks(config, StubSecrets()))
    assert checks["database"].status == "warn"
    assert "alembic" in checks["database"].detail


# --- Binance trading-permission guard (spec section 21.2) ------------------
def test_no_binance_key_means_no_permission_check(
    tmp_path: Path, default_config_path: Path
) -> None:
    checks = _by_name(run_checks(_config(tmp_path, default_config_path), StubSecrets()))
    assert "binance:permissions" not in checks


def test_trading_enabled_key_is_refused(
    tmp_path: Path, default_config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeClient:
        def sapi_get_account_apirestrictions(self) -> dict[str, bool]:
            return {"enableSpotAndMarginTrading": True}

    monkeypatch.setattr("ccxt.binance", lambda *_args, **_kwargs: FakeClient())
    secrets = StubSecrets({"BINANCE_API_KEY": "k", "BINANCE_API_SECRET": "s"})
    checks = run_checks(_config(tmp_path, default_config_path), secrets)

    permission = _by_name(checks)["binance:permissions"]
    assert permission.status == "fail"
    assert permission.exit_code == EXIT_CODES["LockboxViolation"] == 7
    assert worst_exit_code(checks) == 7


def test_read_only_key_passes(
    tmp_path: Path, default_config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeClient:
        def sapi_get_account_apirestrictions(self) -> dict[str, bool]:
            return {"enableSpotAndMarginTrading": False, "enableReading": True}

    monkeypatch.setattr("ccxt.binance", lambda *_args, **_kwargs: FakeClient())
    secrets = StubSecrets({"BINANCE_API_KEY": "k"})
    checks = _by_name(run_checks(_config(tmp_path, default_config_path), secrets))
    assert checks["binance:permissions"].status == "ok"


def test_unreachable_exchange_warns_instead_of_failing(
    tmp_path: Path, default_config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args: object, **_kwargs: object) -> object:
        raise ConnectionError("offline")

    monkeypatch.setattr("ccxt.binance", boom)
    secrets = StubSecrets({"BINANCE_API_KEY": "k"})
    checks = _by_name(run_checks(_config(tmp_path, default_config_path), secrets))
    assert checks["binance:permissions"].status == "warn"


# --- reporting -------------------------------------------------------------
def test_worst_exit_code_prefers_the_most_severe_failure() -> None:
    checks = [
        Check("a", "ok", ""),
        Check("b", "fail", "", 2),
        Check("c", "fail", "", 7),
        Check("d", "warn", ""),
    ]
    assert worst_exit_code(checks) == 7


def test_worst_exit_code_is_zero_without_failures() -> None:
    assert worst_exit_code([Check("a", "ok", ""), Check("b", "warn", "")]) == 0


def test_render_prints_every_check() -> None:
    console = Console(record=True, width=120)
    doctor_module.render(
        [Check("python", "ok", "3.12.0"), Check("x", "fail", "broken", 2)], console
    )
    text = console.export_text()
    assert "python" in text
    assert "broken" in text
    assert "1 check(s) failed" in text


def test_render_reports_success() -> None:
    console = Console(record=True, width=120)
    doctor_module.render([Check("python", "ok", "3.12.0")], console)
    assert "All required checks passed" in console.export_text()
