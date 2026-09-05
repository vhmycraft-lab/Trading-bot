"""Composition root behaviour (master spec section 2.2)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from quantlab.container import PROFILES, Container, build_container, build_secrets
from quantlab.core.config import AppConfig, load_config
from quantlab.core.errors import ConfigError
from quantlab.ports.secrets import Secrets


def _config(tmp_path: Path, default_config_path: Path) -> AppConfig:
    return load_config(
        [], [f"project.db_path={tmp_path / 'quantlab.db'}"], default_path=default_config_path
    )


def test_profiles_are_the_four_documented_ones() -> None:
    assert set(PROFILES) == {"research", "lockbox", "paper", "test"}


@pytest.mark.parametrize("profile", PROFILES)
def test_every_profile_builds(profile: str, tmp_path: Path, default_config_path: Path) -> None:
    container = build_container(_config(tmp_path, default_config_path), profile=profile)  # type: ignore[arg-type]
    try:
        assert isinstance(container, Container)
        assert isinstance(container.secrets, Secrets)
        assert container.profile == profile
    finally:
        container.db_engine.dispose()


def test_unknown_profile_is_rejected(tmp_path: Path, default_config_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown container profile"):
        build_container(_config(tmp_path, default_config_path), profile="production")  # type: ignore[arg-type]


def test_only_the_lockbox_profile_may_read_the_test_partition(
    tmp_path: Path, default_config_path: Path
) -> None:
    """INV-5, expressed at the level of the composition root."""
    for profile in PROFILES:
        container = build_container(_config(tmp_path, default_config_path), profile=profile)  # type: ignore[arg-type]
        try:
            assert container.may_read_test_partition == (profile == "lockbox")
        finally:
            container.db_engine.dispose()


def test_test_profile_uses_an_in_memory_database(tmp_path: Path, default_config_path: Path) -> None:
    container = build_container(_config(tmp_path, default_config_path), profile="test")
    try:
        assert container.db_engine.url.database == ":memory:"
    finally:
        container.db_engine.dispose()


def test_other_profiles_use_the_configured_path(tmp_path: Path, default_config_path: Path) -> None:
    config = _config(tmp_path, default_config_path)
    container = build_container(config, profile="research")
    try:
        assert container.db_engine.url.database == str(config.project.db_path)
    finally:
        container.db_engine.dispose()


def test_container_is_frozen(tmp_path: Path, default_config_path: Path) -> None:
    container = build_container(_config(tmp_path, default_config_path), profile="test")
    try:
        assert dataclasses.is_dataclass(container)
        with pytest.raises(dataclasses.FrozenInstanceError):
            container.profile = "lockbox"  # type: ignore[misc]
    finally:
        container.db_engine.dispose()


def test_secret_chain_order_is_keychain_then_dotenv() -> None:
    chain = build_secrets()
    names = [type(backend).__name__ for backend in chain.backends]  # type: ignore[attr-defined]
    assert names == ["KeychainSecrets", "DotEnvSecrets"]
