"""Secrets port (master spec section 21.1).

Resolution order for every secret:

1. macOS Keychain (``keyring``, service ``quantlab``, account = the name)
2. ``.env`` in the repository root (git-ignored)
3. :class:`~quantlab.core.errors.ConfigError` naming the missing secret

Secrets are read in :mod:`quantlab.container` only and handed to adapters as
constructor arguments.  They are never CLI flags, never in config files, never
in artifacts, and never logged.
"""

from __future__ import annotations

from typing import Final, Protocol, runtime_checkable

__all__ = ["KEYCHAIN_SERVICE", "OPTIONAL_SECRETS", "REQUIRED_SECRETS", "Secrets"]

#: Keychain service name used by the macOS adapter.
KEYCHAIN_SERVICE: Final[str] = "quantlab"

#: Secrets the LLM researcher needs (spec phase H).
REQUIRED_SECRETS: Final[tuple[str, ...]] = ("GLM_API_KEY",)

#: Read-only Binance credentials; used only to raise public rate limits.
OPTIONAL_SECRETS: Final[tuple[str, ...]] = ("BINANCE_API_KEY", "BINANCE_API_SECRET")


@runtime_checkable
class Secrets(Protocol):
    """A read-only source of named secrets."""

    def get(self, name: str) -> str:
        """Return the secret ``name``.

        Raises:
            ConfigError: if the secret cannot be found in any backend.
        """
        ...

    def try_get(self, name: str) -> str | None:
        """Return the secret ``name`` or ``None`` if it is not configured."""
        ...
