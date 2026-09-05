"""Backtest engine port (master spec section 8.3).

One method, and a deliberately narrow one: given a strategy, bars, parameters
and a configuration, produce a :class:`~quantlab.core.types.BacktestResult`.

An engine has no access to a broker, an exchange, or a network, and there is no
method by which it could acquire one — the port simply does not expose the idea
of sending an order anywhere (INV-1).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from quantlab.core.strategy import Strategy
from quantlab.core.types import BacktestConfig, BacktestResult, BarFrame

__all__ = ["BacktestEngine"]


@runtime_checkable
class BacktestEngine(Protocol):
    """Simulates a strategy over historical bars, deterministically."""

    name: str
    version: str

    def run(
        self,
        strategy: Strategy,
        bars: BarFrame,
        params: Mapping[str, Any],
        config: BacktestConfig,
    ) -> BacktestResult:
        """Simulate ``strategy`` over ``bars``.

        Implementations MUST be deterministic: the same inputs produce the same
        result, bit for bit.  ``version`` MUST be bumped whenever any fill or
        accounting rule changes, so the golden tests fail loudly rather than a
        historical comparison silently shifting.
        """
        ...
