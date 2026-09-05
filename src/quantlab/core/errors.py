"""Exception hierarchy (master spec section 18.1).

Every CLI command catches :class:`QuantLabError` at the top, prints a one-line
human message plus an ``error_id``, logs the traceback, and exits with the exit
code of the error family (section 18.2).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "EXIT_CODES",
    "BudgetExceeded",
    "ConfigError",
    "DataError",
    "DataGapError",
    "DataValidationError",
    "EngineError",
    "GateFailure",
    "ImmutableRowError",
    "LLMError",
    "LLMMalformedResponse",
    "LLMTransportError",
    "LeakageDetected",
    "LockboxViolation",
    "LookaheadError",
    "ManifestMismatchError",
    "QuantLabError",
    "RunConflict",
    "SandboxError",
    "SandboxProtocolError",
    "SandboxResourceLimit",
    "SandboxTimeout",
    "StoreError",
    "StrategyError",
    "StrategyLoadError",
    "StrategyRuntimeError",
    "StrategySafetyError",
    "ValidationError_",
    "VerdictError",
    "exit_code_for",
]


class QuantLabError(Exception):
    """Base class for every error this application raises deliberately."""

    #: Process exit code used when this family reaches the CLI top level.
    exit_code: int = 1

    def __init__(self, message: str, /, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = dict(context)

    def __str__(self) -> str:
        if not self.context:
            return self.message
        rendered = ", ".join(f"{k}={v!r}" for k, v in sorted(self.context.items()))
        return f"{self.message} ({rendered})"


# --- config ----------------------------------------------------------------
class ConfigError(QuantLabError):
    """Malformed configuration, missing secret, or a forbidden setting."""

    exit_code = 2


# --- data ------------------------------------------------------------------
class DataError(QuantLabError):
    exit_code = 3


class DataValidationError(DataError):
    """Bars violate the canonical schema of spec section 7.2/7.3."""


class DataGapError(DataError):
    """A gap longer than the auto-fill limit was found and ``--allow-gaps`` was not given."""


class ManifestMismatchError(DataError):
    """A stored Parquet file no longer hashes to its manifest entry."""


# --- lockbox (INV-5) -------------------------------------------------------
class LockboxViolation(QuantLabError):
    """The test partition was touched outside the ``lockbox`` profile.

    MUST NOT be caught anywhere except the CLI top level.
    """

    exit_code = 7


# --- strategies ------------------------------------------------------------
class StrategyError(QuantLabError):
    exit_code = 4


class StrategyLoadError(StrategyError):
    """The strategy module could not be loaded or does not satisfy the contract."""


class StrategySafetyError(StrategyError):
    """The AST checker rejected the source."""

    def __init__(
        self,
        message: str,
        /,
        violations: list[str] | None = None,
        codes: list[str] | None = None,
        **context: Any,
    ) -> None:
        super().__init__(message, **context)
        #: Rendered violations, one per line, for a human.
        self.violations: list[str] = list(violations or [])
        #: The bare violation codes, for a caller that wants to react to a rule
        #: rather than parse a message.
        self.codes: list[str] = list(codes or [])


class StrategyRuntimeError(StrategyError):
    """The strategy raised while the engine was running it."""


class LookaheadError(StrategyError):
    """A strategy tried to read a bar at an index greater than the current one (INV-3)."""


class LeakageDetected(StrategyError):
    """The truncation probe proved the strategy is not causal."""

    def __init__(self, message: str, /, probe: Any = None, **context: Any) -> None:
        super().__init__(message, **context)
        self.probe = probe


# --- sandbox ---------------------------------------------------------------
class SandboxError(QuantLabError):
    exit_code = 5


class SandboxTimeout(SandboxError):
    """The child process exceeded its wall-clock budget."""


class SandboxResourceLimit(SandboxError):
    """The child process exceeded a CPU or memory rlimit."""


class SandboxProtocolError(SandboxError):
    """The child process produced output the parent cannot parse."""


# --- engine ----------------------------------------------------------------
class EngineError(QuantLabError):
    """An accounting invariant broke.  Always a bug in the engine; never swallowed."""

    exit_code = 8


# --- store -----------------------------------------------------------------
class StoreError(QuantLabError):
    exit_code = 1


class RunConflict(StoreError):
    """A run id already exists with incompatible inputs."""


class ImmutableRowError(StoreError):
    """An append-only row was updated in a column that is not mutable (spec section 6)."""


# --- LLM -------------------------------------------------------------------
class LLMError(QuantLabError):
    exit_code = 6


class LLMTransportError(LLMError):
    """Network, timeout or non-retryable HTTP status from the provider."""


class LLMMalformedResponse(LLMError):
    """The response did not validate against the requested pydantic model."""


class BudgetExceeded(LLMError):
    """The request would cross the configured cost cap; it was not sent."""


# --- validation ------------------------------------------------------------
class ValidationError_(QuantLabError):
    """Renamed with a trailing underscore to avoid clashing with ``pydantic.ValidationError``."""

    exit_code = 1


class GateFailure(ValidationError_):
    """A hard gate failed."""


class VerdictError(ValidationError_):
    """A verdict could not be computed or is inconsistent with the stored thresholds."""


#: Exit code per error family, for documentation and tests (spec section 18.2).
EXIT_CODES: dict[str, int] = {
    "ConfigError": 2,
    "DataError": 3,
    "StrategyError": 4,
    "SandboxError": 5,
    "LLMError": 6,
    "LockboxViolation": 7,
    "EngineError": 8,
    "other": 1,
}


def exit_code_for(exc: BaseException) -> int:
    """Return the process exit code for ``exc`` (1 for anything unrecognised)."""
    if isinstance(exc, QuantLabError):
        return exc.exit_code
    return 1
