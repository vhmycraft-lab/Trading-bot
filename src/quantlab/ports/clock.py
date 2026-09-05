"""Clock port (master spec sections 0.3 and 16.1).

Time is injected, never read ambiently.  ``core/``, ``adapters/engine/`` and
every strategy are forbidden from calling :func:`datetime.datetime.now` or
:func:`time.time`, because a backtest that depends on the wall clock is not
reproducible and INV-7 requires that every stored run reproduces exactly.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["Clock"]


@runtime_checkable
class Clock(Protocol):
    """A source of the current time, in milliseconds since the Unix epoch, UTC."""

    def now_ms(self) -> int:
        """Return the current time in milliseconds since the Unix epoch, UTC."""
        ...

    async def sleep(self, seconds: float) -> None:
        """Wait for ``seconds``.  A simulated clock may return immediately."""
        ...
