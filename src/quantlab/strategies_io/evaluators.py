"""Binding a loaded strategy to something that can run it (spec sections 8.3, 11.2).

The runner needs to evaluate a strategy over a segment. It cannot import the
sandbox (INV-8), and the strategy is untrusted, so it must not be executed in the
process that decides what to trust (INV-4). This module is where those two facts
are reconciled: it lives in the one package permitted to reach both the sandbox
and the strategy contract, and it hands the runner a plain
:class:`~quantlab.ports.engine.StrategyEvaluator`.

Everything goes through the sandbox, including the platform's own baselines. A
second, faster, in-process path would be a second set of numbers, and the first
time the two disagreed nobody would know which had produced the stored result.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from quantlab.core.errors import StrategyRuntimeError
from quantlab.core.types import BacktestConfig, BacktestResult, BarFrame
from quantlab.sandbox.runner import SandboxRunner

__all__ = ["SandboxEvaluator"]


@dataclass(frozen=True, slots=True)
class SandboxEvaluator:
    """Evaluates one strategy's source in a child process, once per call.

    ``engine_name`` and ``engine_version`` are declared by the caller because the
    run id of section 11.2 is computed *before* anything executes: they are part
    of the cache key, so they cannot be read off a result that does not exist yet.
    The evaluator checks the child's answer against them, so a mismatch is a loud
    failure rather than a run filed under an engine that did not produce it.
    """

    source: str
    engine_module: str
    engine_class: str
    engine_name: str
    engine_version: str
    runner: SandboxRunner = field(default_factory=SandboxRunner)
    #: Already checked by the loader; re-running it per evaluation buys nothing.
    ast_check: bool = False

    def evaluate(
        self, bars: BarFrame, params: Mapping[str, Any], config: BacktestConfig
    ) -> BacktestResult:
        """Run the strategy over ``bars`` and return what the child produced.

        Raises:
            SandboxError: the child failed, timed out or hit a limit. Raised
                rather than returned, so the runner records a failed run instead
                of an empty successful one.
            StrategyRuntimeError: the child ran a different engine from the one
                the run id was computed against.
        """
        outcome = self.runner.run(
            self.source,
            bars,
            engine_module=self.engine_module,
            engine_class=self.engine_class,
            params=dict(params),
            config=config,
            ast_check=self.ast_check,
        )
        result = outcome.require()
        if (result.engine_name, result.engine_version) != (self.engine_name, self.engine_version):
            raise StrategyRuntimeError(
                "the sandbox ran a different engine from the one this run is identified by",
                expected=f"{self.engine_name}:{self.engine_version}",
                actual=f"{result.engine_name}:{result.engine_version}",
            )
        return result
