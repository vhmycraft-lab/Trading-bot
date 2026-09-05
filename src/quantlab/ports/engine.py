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

__all__ = ["BacktestEngine", "StrategyEvaluator"]


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


@runtime_checkable
class StrategyEvaluator(Protocol):
    """Evaluates one already-identified strategy over whatever bars it is given.

    Narrower than :class:`BacktestEngine`, and deliberately so. An engine is
    handed a strategy *object*; an evaluator has already been bound to a
    particular strategy and only needs a segment. That is what lets the same
    interface serve a trusted in-process engine and an untrusted strategy running
    in a sandbox child, where no strategy object exists on this side of the
    process boundary at all (INV-4).

    ``engine_name`` and ``engine_version`` are declared rather than discovered
    because the run id of spec section 11.2 is built from them *before* anything
    runs — they are part of the cache key, not of the result.
    """

    @property
    def engine_name(self) -> str:
        """The engine this evaluator runs, as it appears in the run id."""
        ...

    @property
    def engine_version(self) -> str:
        """Its version. Bumped whenever a fill or accounting rule changes."""
        ...

    def evaluate(
        self, bars: BarFrame, params: Mapping[str, Any], config: BacktestConfig
    ) -> BacktestResult:
        """Simulate the bound strategy over ``bars``.

        Implementations MUST be deterministic: identical inputs produce identical
        results, bit for bit (INV-7).
        """
        ...
