"""macOS Keychain secrets backend (``keyring``)."""

from __future__ import annotations

from quantlab.core.errors import ConfigError
from quantlab.ports.secrets import KEYCHAIN_SERVICE

__all__ = ["KeychainSecrets"]


class KeychainSecrets:
    """Read secrets from the OS keyring under the service ``quantlab``.

    A keyring that is unavailable (no backend, locked, not macOS) is not an
    error: :meth:`try_get` returns ``None`` so the chain falls through to
    ``.env``.  Only :meth:`get` on a genuinely missing secret raises.
    """

    def __init__(self, service: str = KEYCHAIN_SERVICE) -> None:
        self.service = service

    def try_get(self, name: str) -> str | None:
        try:
            import keyring
            from keyring.errors import KeyringError
        except ImportError:  # pragma: no cover - keyring is a hard dependency
            return None
        try:
            return keyring.get_password(self.service, name)
        except KeyringError:
            return None

    def get(self, name: str) -> str:
        value = self.try_get(name)
        if value is None:
            raise ConfigError(f"missing secret {name}; run quantlab doctor", secret=name)
        return value
