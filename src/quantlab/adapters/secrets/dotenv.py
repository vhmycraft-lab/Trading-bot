"""``.env`` secrets backend.

The file is git-ignored (INV-2).  Values are read from the file and from the
process environment, with the file taking precedence so that a developer's
``.env`` is authoritative for their checkout.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values

from quantlab.core.errors import ConfigError

__all__ = ["DEFAULT_ENV_FILE", "DotEnvSecrets"]

DEFAULT_ENV_FILE = Path(".env")


class DotEnvSecrets:
    """Read secrets from a ``.env`` file, falling back to ``os.environ``."""

    def __init__(self, path: str | Path = DEFAULT_ENV_FILE, *, use_environ: bool = True) -> None:
        self.path = Path(path)
        self.use_environ = use_environ

    @property
    def exists(self) -> bool:
        return self.path.is_file()

    def _from_file(self, name: str) -> str | None:
        if not self.exists:
            return None
        values = dotenv_values(self.path)
        value = values.get(name)
        return value or None

    def try_get(self, name: str) -> str | None:
        value = self._from_file(name)
        if value:
            return value
        if self.use_environ:
            return os.environ.get(name) or None
        return None

    def get(self, name: str) -> str:
        value = self.try_get(name)
        if value is None:
            raise ConfigError(f"missing secret {name}; run quantlab doctor", secret=name)
        return value
