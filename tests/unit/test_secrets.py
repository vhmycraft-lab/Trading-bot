"""Secret resolution order and hygiene (master spec section 21.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from quantlab.adapters.secrets import ChainedSecrets, DotEnvSecrets, KeychainSecrets
from quantlab.core.errors import ConfigError
from quantlab.ports.secrets import (
    KEYCHAIN_SERVICE,
    OPTIONAL_SECRETS,
    REQUIRED_SECRETS,
    Secrets,
)


class FakeSecrets:
    """A minimal structural implementation of the port."""

    def __init__(self, values: dict[str, str]) -> None:
        self.values = values

    def try_get(self, name: str) -> str | None:
        return self.values.get(name)

    def get(self, name: str) -> str:
        value = self.try_get(name)
        if value is None:
            raise ConfigError(f"missing secret {name}; run quantlab doctor")
        return value


def test_port_is_structural() -> None:
    assert isinstance(FakeSecrets({}), Secrets)
    assert isinstance(DotEnvSecrets(".env"), Secrets)
    assert isinstance(KeychainSecrets(), Secrets)


def test_names_and_service_are_declared() -> None:
    assert REQUIRED_SECRETS == ("GLM_API_KEY",)
    assert OPTIONAL_SECRETS == ("BINANCE_API_KEY", "BINANCE_API_SECRET")
    assert KEYCHAIN_SERVICE == "quantlab"


# --- dotenv ----------------------------------------------------------------
def _write_env(path: Path, name: str, value: str) -> Path:
    env = path / ".env"
    env.write_text(f"{name}={value}\n", encoding="utf-8")
    return env


def test_dotenv_reads_the_file(tmp_path: Path) -> None:
    env = _write_env(tmp_path, "GLM_API_KEY", "from-file")
    assert DotEnvSecrets(env).get("GLM_API_KEY") == "from-file"


def test_dotenv_file_beats_the_process_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _write_env(tmp_path, "GLM_API_KEY", "from-file")
    monkeypatch.setenv("GLM_API_KEY", "from-environ")
    assert DotEnvSecrets(env).get("GLM_API_KEY") == "from-file"


def test_dotenv_falls_back_to_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GLM_API_KEY", "from-environ")
    assert DotEnvSecrets(tmp_path / "absent.env").get("GLM_API_KEY") == "from-environ"


def test_dotenv_can_ignore_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLM_API_KEY", "from-environ")
    backend = DotEnvSecrets(tmp_path / "absent.env", use_environ=False)
    assert backend.try_get("GLM_API_KEY") is None


def test_dotenv_treats_an_empty_value_as_missing(tmp_path: Path) -> None:
    env = _write_env(tmp_path, "GLM_API_KEY", "")
    assert DotEnvSecrets(env, use_environ=False).try_get("GLM_API_KEY") is None


def test_dotenv_reports_existence(tmp_path: Path) -> None:
    assert not DotEnvSecrets(tmp_path / ".env").exists
    assert DotEnvSecrets(_write_env(tmp_path, "A", "b")).exists


def test_missing_secret_message_names_the_doctor(tmp_path: Path) -> None:
    backend = DotEnvSecrets(tmp_path / "absent.env", use_environ=False)
    with pytest.raises(ConfigError, match=r"missing secret GLM_API_KEY; run quantlab doctor"):
        backend.get("GLM_API_KEY")


# --- keychain --------------------------------------------------------------
def test_keychain_returns_the_stored_password(monkeypatch: pytest.MonkeyPatch) -> None:
    import keyring

    monkeypatch.setattr(
        keyring, "get_password", lambda service, name: "kc" if service == "quantlab" else None
    )
    assert KeychainSecrets().get("GLM_API_KEY") == "kc"


def test_keychain_backend_failure_is_not_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    import keyring
    from keyring.errors import KeyringError

    def boom(service: str, name: str) -> str:
        raise KeyringError("no backend")

    monkeypatch.setattr(keyring, "get_password", boom)
    assert KeychainSecrets().try_get("GLM_API_KEY") is None
    with pytest.raises(ConfigError):
        KeychainSecrets().get("GLM_API_KEY")


# --- chain -----------------------------------------------------------------
def test_chain_prefers_the_first_backend_that_answers() -> None:
    chain = ChainedSecrets(
        [FakeSecrets({}), FakeSecrets({"A": "second"}), FakeSecrets({"A": "third"})]
    )
    assert chain.get("A") == "second"


def test_chain_reports_the_missing_name_only() -> None:
    chain = ChainedSecrets([FakeSecrets({}), FakeSecrets({})])
    with pytest.raises(ConfigError) as excinfo:
        chain.get("GLM_API_KEY")
    assert "GLM_API_KEY" in str(excinfo.value)
    assert chain.try_get("GLM_API_KEY") is None


def test_chain_exposes_its_backends() -> None:
    backends = [FakeSecrets({}), FakeSecrets({})]
    assert ChainedSecrets(backends).backends == tuple(backends)
