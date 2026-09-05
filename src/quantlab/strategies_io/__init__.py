"""Loading strategy source into registered, reproducible strategy versions.

The only package besides ``sandbox`` that handles untrusted strategy text, and it
handles it as *text*: it checks, hashes, copies and registers, and never executes.
"""

from __future__ import annotations

from quantlab.strategies_io.evaluators import SandboxEvaluator
from quantlab.strategies_io.loader import SUPPORTED_STYLES, LoadedStrategy, StrategyLoader

__all__ = ["SUPPORTED_STYLES", "LoadedStrategy", "SandboxEvaluator", "StrategyLoader"]
