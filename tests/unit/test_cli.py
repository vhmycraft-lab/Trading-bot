"""CLI entry point: version, config, doctor, db (master spec sections 18.2, 22 T01/T04)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quantlab import __version__
from quantlab.cli import app
from quantlab.core.errors import EXIT_CODES


@pytest.fixture
def project(tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway working directory with the real configs available."""
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "default.yaml").write_text(
        (repo_root / "configs" / "default.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "configs" / "splits").mkdir()
    (tmp_path / "configs" / "splits" / "btcusdt_1h.yaml").write_text(
        (repo_root / "configs" / "splits" / "btcusdt_1h.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


# --- version ---------------------------------------------------------------
def test_version_flag(cli: CliRunner) -> None:
    result = cli.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__ == "0.1.0"


def test_version_command(cli: CliRunner, project: Path) -> None:
    result = cli.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.0"


def test_help_lists_the_commands(cli: CliRunner) -> None:
    result = cli.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("config", "db", "doctor", "version"):
        assert command in result.stdout


def test_bare_invocation_shows_help(cli: CliRunner) -> None:
    assert cli.invoke(app, []).exit_code != 0


# --- config ----------------------------------------------------------------
def test_config_show_emits_json(cli: CliRunner, project: Path) -> None:
    result = cli.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["market"]["symbol"] == "BTC/USDT"
    assert payload["optimize"]["objective"] == "sortino_dd"


def test_config_show_canonical(cli: CliRunner, project: Path) -> None:
    result = cli.invoke(app, ["config", "show", "--indent", "0"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["backtest"]["fee_bps"] == 10.0


def test_config_hash_is_stable(cli: CliRunner, project: Path) -> None:
    first = cli.invoke(app, ["config", "hash"])
    second = cli.invoke(app, ["config", "hash"])
    assert first.exit_code == 0
    assert first.stdout.strip() == second.stdout.strip()
    assert len(first.stdout.strip()) == 64


def test_set_override_reaches_the_config(cli: CliRunner, project: Path) -> None:
    result = cli.invoke(app, ["--set", "optimize.n_trials=7", "config", "show"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["optimize"]["n_trials"] == 7


def test_config_file_option_is_merged(cli: CliRunner, project: Path) -> None:
    extra = project / "extra.yaml"
    extra.write_text("backtest:\n  fee_bps: 12.0\n", encoding="utf-8")
    result = cli.invoke(app, ["--config", str(extra), "config", "show"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["backtest"]["fee_bps"] == 12.0


# --- error handling --------------------------------------------------------
def test_forbidden_objective_exits_with_the_config_code(cli: CliRunner, project: Path) -> None:
    result = cli.invoke(app, ["--set", "optimize.objective=net_return", "config", "show"])
    assert result.exit_code == EXIT_CODES["ConfigError"] == 2


def test_unknown_key_exits_with_the_config_code(cli: CliRunner, project: Path) -> None:
    result = cli.invoke(app, ["--set", "market.symbl=X", "config", "show"])
    assert result.exit_code == 2


def test_unknown_profile_exits_with_the_config_code(cli: CliRunner, project: Path) -> None:
    result = cli.invoke(app, ["--profile", "production", "db", "info"])
    assert result.exit_code == 2


# --- db --------------------------------------------------------------------
def test_db_info_before_and_after_upgrade(cli: CliRunner, project: Path) -> None:
    before = cli.invoke(app, ["db", "info"])
    assert before.exit_code == 0
    assert "not created yet" in before.stdout

    upgraded = cli.invoke(app, ["db", "upgrade"])
    assert upgraded.exit_code == 0, upgraded.output
    assert "0001" in upgraded.stdout

    after = cli.invoke(app, ["db", "info"])
    assert after.exit_code == 0
    assert "0001" in after.stdout
    assert "all present" in after.stdout


def test_db_upgrade_is_idempotent(cli: CliRunner, project: Path) -> None:
    assert cli.invoke(app, ["db", "upgrade"]).exit_code == 0
    assert cli.invoke(app, ["db", "upgrade"]).exit_code == 0


# --- doctor ----------------------------------------------------------------
def test_doctor_without_secrets_lists_them_and_exits_two(cli: CliRunner, project: Path) -> None:
    result = cli.invoke(app, ["doctor"])
    assert result.exit_code == EXIT_CODES["ConfigError"] == 2
    assert "GLM_API_KEY" in result.stdout
    assert "missing" in result.stdout


def test_doctor_passes_once_the_secret_is_present(cli: CliRunner, project: Path) -> None:
    (project / ".env").write_text("GLM_API_KEY=" + ("x" * 12) + "\n", encoding="utf-8")
    (project / "data").mkdir()
    (project / "artifacts").mkdir()
    assert cli.invoke(app, ["db", "upgrade"]).exit_code == 0

    result = cli.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.stdout
    assert "All required checks passed" in result.stdout


def test_doctor_never_prints_a_secret_value(cli: CliRunner, project: Path) -> None:
    marker = "z" * 24
    (project / ".env").write_text(f"GLM_API_KEY={marker}\n", encoding="utf-8")
    result = cli.invoke(app, ["doctor"])
    assert marker not in result.stdout
