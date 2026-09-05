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
from pathlib import Path
from typing import Literal, get_args

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from quantlab.adapters.data.binance_archive import ParquetBarStore
from quantlab.adapters.data.guard import PartitionGuard
from quantlab.adapters.secrets import ChainedSecrets, DotEnvSecrets, KeychainSecrets
from quantlab.adapters.store.sqlite import create_db_engine, make_session_factory
from quantlab.core.config import AppConfig
from quantlab.core.errors import ConfigError
from quantlab.core.splits import SplitPolicy, load_split_policy
from quantlab.ports.data import MarketDataSource
from quantlab.ports.secrets import Secrets

__all__ = [
    "PROFILES",
    "Container",
    "Profile",
    "build_container",
    "build_market_data",
    "build_secrets",
]

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
    #: Historical bars.  Guarded in every profile except ``lockbox`` (INV-5).
    market_data: MarketDataSource | None = None
    #: The split this container's guard enforces; ``None`` if no policy is configured.
    split_policy: SplitPolicy | None = None

    @property
    def may_read_test_partition(self) -> bool:
        """True only for the ``lockbox`` profile (INV-5)."""
        return self.profile == "lockbox"


def build_secrets(*, env_file: str = ".env") -> Secrets:
    """Build the secret resolution chain: macOS Keychain, then ``.env`` (spec section 21.1)."""
    return ChainedSecrets([KeychainSecrets(), DotEnvSecrets(env_file)])


def build_market_data(
    config: AppConfig, *, profile: Profile
) -> tuple[MarketDataSource, SplitPolicy | None]:
    """Build the market data source for ``profile``, with its partition guard.

    Every profile except ``lockbox`` gets a
    :class:`~quantlab.adapters.data.guard.PartitionGuard`, which raises
    :class:`~quantlab.core.errors.LockboxViolation` rather than serving a bar
    from the held-out test partition (INV-5).  This is the one place the guard
    is attached, so no caller can obtain an unguarded source by accident.

    A missing or non-matching split policy leaves the source unguarded only in
    the ``lockbox`` profile; in every other profile it is an error, because an
    unguarded research source is exactly the failure this design exists to
    prevent.
    """
    store = ParquetBarStore(config.project.data_dir, exchange=config.market.exchange)
    if profile == "lockbox":
        return store, None

    policy_path = Path(config.splits.policy_file)
    if not policy_path.is_file():
        raise ConfigError(
            "a split policy is required outside the lockbox profile; "
            "without one the test partition cannot be protected",
            profile=profile,
            policy_file=str(policy_path),
        )
    policy = load_split_policy(policy_path)
    return PartitionGuard(store, policy), policy


def build_container(config: AppConfig, *, profile: Profile) -> Container:
    """Wire every port for ``profile``.

    Raises:
        ConfigError: if ``profile`` is not one of :data:`PROFILES`, or if a
            guarded profile has no usable split policy.
    """
    if profile not in PROFILES:
        raise ConfigError("unknown container profile", profile=profile, allowed=list(PROFILES))

    db_path = ":memory:" if profile == "test" else config.project.db_path
    engine = create_db_engine(db_path)
    market_data, policy = build_market_data(config, profile=profile)

    return Container(
        config=config,
        profile=profile,
        secrets=build_secrets(),
        db_engine=engine,
        session_factory=make_session_factory(engine),
        market_data=market_data,
        split_policy=policy,
    )
