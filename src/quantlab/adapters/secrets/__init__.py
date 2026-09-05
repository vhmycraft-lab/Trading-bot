"""Secrets adapters and the resolution chain (master spec section 21.1)."""

from __future__ import annotations

from collections.abc import Sequence

from quantlab.adapters.secrets.dotenv import DotEnvSecrets
from quantlab.adapters.secrets.keychain import KeychainSecrets
from quantlab.core.errors import ConfigError
from quantlab.ports.secrets import Secrets

__all__ = ["ChainedSecrets", "DotEnvSecrets", "KeychainSecrets"]


class ChainedSecrets:
    """Try each backend in order and return the first hit.

    The chain never reports *which* backend answered and never includes a value
    in an exception message.
    """

    def __init__(self, backends: Sequence[Secrets]) -> None:
        self._backends = tuple(backends)

    @property
    def backends(self) -> tuple[Secrets, ...]:
        return self._backends

    def try_get(self, name: str) -> str | None:
        for backend in self._backends:
            value = backend.try_get(name)
            if value:
                return value
        return None

    def get(self, name: str) -> str:
        value = self.try_get(name)
        if value is None:
            raise ConfigError(f"missing secret {name}; run quantlab doctor", secret=name)
        return value
