"""Composition root (master spec section 2.2).

This is the only module in the package that instantiates adapters.  Everything
else depends on ports.  To swap an implementation, add an adapter and one branch
here; ``tests/unit/test_architecture.py`` fails on any other import of
``quantlab.adapters`` outside ``container.py``, ``cli/`` and ``tests/``.

Profiles:

``research``
    Normal work.  The market data source is wrapped in a partition guard that
    refuses the test partition (INV-5).
``lockbox``
    The only profile whose data source may read the test partition.
``paper``
    Live paper trading against a websocket feed and the paper broker.
``test``
    Mock LLM provider and an in-memory database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, get_args

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from quantlab.adapters.secrets import ChainedSecrets, DotEnvSecrets, KeychainSecrets
from quantlab.adapters.store.sqlite import create_db_engine, make_session_factory
from quantlab.core.config import AppConfig
from quantlab.core.errors import ConfigError
from quantlab.ports.secrets import Secrets

__all__ = ["PROFILES", "Container", "Profile", "build_container", "build_secrets"]

Profile = Literal["research", "lockbox", "paper", "test"]

#: Every valid profile name, for validation and CLI help.
PROFILES: tuple[str, ...] = get_args(Profile)


@dataclass(frozen=True, slots=True)
class Container:
    """One instance per port, wired for a single profile.

    Ports that belong to phases not yet implemented are absent rather than
    stubbed, so a premature use is an :class:`AttributeError` at development
    time rather than a silent no-op at run time.
    """

    config: AppConfig
    profile: Profile
    secrets: Secrets
    db_engine: Engine
    session_factory: sessionmaker[Session] = field(repr=False)

    @property
    def may_read_test_partition(self) -> bool:
        """True only for the ``lockbox`` profile (INV-5)."""
        return self.profile == "lockbox"


def build_secrets(*, env_file: str = ".env") -> Secrets:
    """Build the secret resolution chain: macOS Keychain, then ``.env`` (spec section 21.1)."""
    return ChainedSecrets([KeychainSecrets(), DotEnvSecrets(env_file)])


def build_container(config: AppConfig, *, profile: Profile) -> Container:
    """Wire every port for ``profile``.

    Raises:
        ConfigError: if ``profile`` is not one of :data:`PROFILES`.
    """
    if profile not in PROFILES:
        raise ConfigError("unknown container profile", profile=profile, allowed=list(PROFILES))

    db_path = ":memory:" if profile == "test" else config.project.db_path
    engine = create_db_engine(db_path)

    return Container(
        config=config,
        profile=profile,
        secrets=build_secrets(),
        db_engine=engine,
        session_factory=make_session_factory(engine),
    )
