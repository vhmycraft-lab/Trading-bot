"""Running the leakage probe on code nobody has vouched for yet.

The probe of spec section 14.2 compares what a strategy decided across several
evaluations of the same segment. Someone has to run those evaluations, and for a
strategy that has just arrived from a file — or from a model — that someone must
not be this process (INV-4). This module supplies the missing half: an
``Evaluate`` that sends each segment through the sandbox child and hands the
result back, so :func:`~quantlab.core.validation.leakage.probe_evaluations` can do
the comparing without ever holding the strategy object.

The cost is one child process per evaluation — nine for the default probe — which
is why the probe runs once, at load, and its verdict is recorded rather than
recomputed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from quantlab.core.errors import StrategyLoadError
from quantlab.core.types import BacktestConfig, BacktestResult, BarFrame
from quantlab.core.validation.leakage import Evaluate
from quantlab.sandbox.runner import SandboxRunner

__all__ = ["sandbox_evaluator"]


def sandbox_evaluator(
    runner: SandboxRunner,
    source: str,
    *,
    engine_module: str,
    engine_class: str,
    params: Mapping[str, Any] | None = None,
    config: BacktestConfig | None = None,
) -> Evaluate:
    """An :data:`Evaluate` that runs ``source`` in the sandbox, once per segment.

    The AST check is not repeated per evaluation — the loader has already run it,
    and re-running it nine times would only slow the probe down. Every other
    control the child installs still applies to every evaluation.

    Raises:
        StrategyLoadError: an evaluation did not produce a result. Fail closed: a
            probe that treated a crashed evaluation as agreement would pass exactly
            the strategies most likely to be broken.
    """
    settings = config or BacktestConfig()
    supplied = dict(params or {})

    def evaluate(segment: BarFrame) -> BacktestResult:
        outcome = runner.run(
            source,
            segment,
            engine_module=engine_module,
            engine_class=engine_class,
            params=supplied,
            config=settings,
            ast_check=False,
        )
        if not outcome.ok or outcome.result is None:
            raise StrategyLoadError(
                "the strategy could not be evaluated for the leakage probe",
                status=outcome.status,
                n_bars=segment.n_bars,
                error=outcome.response.error_message[:500],
            )
        return outcome.result

    return evaluate
