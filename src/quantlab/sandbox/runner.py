"""Parent side of the sandbox: spawn a child, bound it, read what it produced.

Untrusted strategy code never runs in this process (INV-4). It runs in a child
that is given a temporary directory, a scrubbed environment, hard resource
limits and a wall-clock deadline, and that talks back only in Parquet and JSON.
The child is started with ``-I`` (isolated mode): no user site-packages, no
``PYTHONPATH``, no ``sitecustomize`` — the interpreter that runs untrusted code
must not be configurable from outside the request.

The engine is *named*, not imported. The sandbox layer sits beside ``core`` and
may not depend on an adapter (INV-8), and the caller holding an engine already
knows which one it wants.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from quantlab.core.errors import (
    SandboxError,
    SandboxProtocolError,
    SandboxResourceLimit,
    SandboxTimeout,
)
from quantlab.core.types import BacktestConfig, BacktestResult, BarFrame
from quantlab.sandbox.protocol import (
    EXIT_OK,
    EXIT_STATUS,
    SandboxRequest,
    SandboxResponse,
    SandboxStatus,
    read_response,
    read_result,
    write_request,
)

__all__ = ["CHILD_MODULE", "SandboxLimits", "SandboxOutcome", "SandboxRunner"]

CHILD_MODULE: Final[str] = "quantlab.sandbox.child_main"

#: Everything the child's environment holds. ``PYTHONHASHSEED`` is pinned because
#: a strategy that iterates a set must produce the same order on every run
#: (INV-7); the rest is absent so that nothing configured in this shell — proxies,
#: credentials, library paths — is visible to untrusted code.
_BASE_ENV: Final[dict[str, str]] = {
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "MPLBACKEND": "Agg",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}

#: Signals a child dies from when it hits a limit. Python reports these as a
#: negative return code.
_SIGXCPU: Final[int] = 24
_SIGKILL: Final[int] = 9


@dataclass(frozen=True, slots=True)
class SandboxLimits:
    """The box, in numbers (spec §21.3, config ``sandbox:``)."""

    cpu_seconds: int = 120
    memory_mb: int = 2048
    wall_clock_s: int = 300
    allowed_imports: tuple[str, ...] = (
        "numpy",
        "pandas",
        "math",
        "statistics",
        "dataclasses",
        "typing",
        "quantlab.core.strategy",
        "quantlab.core.indicators",
        "quantlab.core.types",
    )

    def __post_init__(self) -> None:
        for name in ("cpu_seconds", "memory_mb", "wall_clock_s"):
            if getattr(self, name) < 1:
                raise ValueError(f"SandboxLimits.{name} must be >= 1")


@dataclass(frozen=True, slots=True)
class SandboxOutcome:
    """What one child run produced, successful or not."""

    response: SandboxResponse
    result: BacktestResult | None = None
    exit_code: int = 0
    stderr: str = ""
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.result is not None and self.response.status == "ok"

    @property
    def status(self) -> SandboxStatus:
        return self.response.status

    def require(self) -> BacktestResult:
        """The result, or the reason there is none.

        Raises:
            SandboxTimeout: the child exceeded its wall clock.
            SandboxResourceLimit: the child exceeded CPU or memory.
            SandboxError: anything else, carrying the child's own message.
        """
        if self.result is not None and self.response.status == "ok":
            return self.result
        if self.status in ("timeout", "resource_limit"):
            error = SandboxTimeout if self.status == "timeout" else SandboxResourceLimit
            raise error(
                self.response.error_message or f"the sandbox child hit a {self.status}",
                stderr=self.stderr[-2000:],
            )
        raise SandboxError(
            f"the sandbox child failed: {self.response.error_message or self.status}",
            status=self.status,
            exit_code=self.exit_code,
            error_type=self.response.error_type,
        )


@dataclass(frozen=True, slots=True)
class SandboxRunner:
    """Runs one strategy, once, in a child process.

    Stateless on purpose: two runs share nothing but this object's limits, so a
    strategy cannot leave anything behind for the next one to find.
    """

    limits: SandboxLimits = field(default_factory=SandboxLimits)
    #: Kept for inspection when a run fails; deleted otherwise.
    keep_failed_dirs: bool = False
    python: str = field(default_factory=lambda: sys.executable)

    def run(
        self,
        source: str,
        bars: BarFrame,
        *,
        engine_module: str,
        engine_class: str,
        params: Mapping[str, Any] | None = None,
        config: BacktestConfig | None = None,
        strategy_name: str = "",
        ast_check: bool = True,
    ) -> SandboxOutcome:
        """Execute ``source`` against ``bars`` in a child process.

        Never raises for a strategy that misbehaves — that is an outcome, and the
        caller (an evolutionary generation, say) wants to record it and move on.
        Call :meth:`SandboxOutcome.require` to turn a failure into an exception.
        """
        request = SandboxRequest(
            symbol=bars.symbol,
            timeframe=bars.timeframe,
            dataset_id=bars.dataset_id,
            strategy_name=strategy_name,
            params=dict(params or {}),
            config=config or BacktestConfig(),
            engine_module=engine_module,
            engine_class=engine_class,
            allowed_imports=self.limits.allowed_imports,
            ast_check=ast_check,
        )
        directory = Path(tempfile.mkdtemp(prefix="quantlab-sandbox-"))
        try:
            write_request(directory, request, bars, source)
            outcome = self._spawn(directory)
        finally:
            keep = self.keep_failed_dirs and not _succeeded(locals().get("outcome"))
            if not keep:
                shutil.rmtree(directory, ignore_errors=True)
        return outcome

    # -- child process -----------------------------------------------------
    def _spawn(self, directory: Path) -> SandboxOutcome:
        completed, timed_out = self._invoke(directory)
        exit_code = -_SIGKILL if completed is None else completed.returncode
        stderr = "" if completed is None else completed.stderr

        if timed_out:
            return SandboxOutcome(
                response=SandboxResponse(
                    status="timeout",
                    error_type="SandboxTimeout",
                    error_message=(f"no result within {self.limits.wall_clock_s}s of wall clock"),
                ),
                exit_code=exit_code,
                stderr=stderr,
                timed_out=True,
            )

        response = self._response_for(directory, exit_code, stderr)
        result = None
        if response.status == "ok" and exit_code == EXIT_OK:
            result = read_result(directory, response)
        return SandboxOutcome(response=response, result=result, exit_code=exit_code, stderr=stderr)

    def _invoke(self, directory: Path) -> tuple[subprocess.CompletedProcess[str] | None, bool]:
        command = [self.python, "-I", "-m", CHILD_MODULE, str(directory)]
        try:
            completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
                command,
                cwd=directory,
                env=self._child_env(),
                capture_output=True,
                text=True,
                timeout=self.limits.wall_clock_s,
                check=False,
                preexec_fn=self._apply_limits,
            )
        except subprocess.TimeoutExpired:
            return None, True
        return completed, False

    def _child_env(self) -> dict[str, str]:
        env = dict(_BASE_ENV)
        env["PATH"] = os.environ.get("PATH", "/usr/bin:/bin")
        # -I ignores PYTHONPATH, so the child finds quantlab the same way this
        # process did: through the installed package, not through the shell.
        return env

    def _apply_limits(self) -> None:  # pragma: no cover - runs in the forked child
        """Set hard limits between fork and exec.

        Deliberately not conditional on success: on a platform where a limit
        cannot be set, the run continues with the limits that could be, and the
        wall clock still applies.
        """
        import resource

        cpu = self.limits.cpu_seconds
        memory = self.limits.memory_mb * 1024 * 1024
        for which, limit in (
            (resource.RLIMIT_CPU, (cpu, cpu + 1)),
            (resource.RLIMIT_AS, (memory, memory)),
            (resource.RLIMIT_CORE, (0, 0)),
        ):
            with contextlib.suppress(ValueError, OSError):
                resource.setrlimit(which, limit)

    # -- response ----------------------------------------------------------
    def _response_for(self, directory: Path, exit_code: int, stderr: str) -> SandboxResponse:
        """The child's own response, or one synthesised from how it died."""
        try:
            response = read_response(directory)
        except SandboxProtocolError:
            return _killed_response(exit_code, stderr, self.limits)
        if exit_code != EXIT_OK and response.status == "ok":
            # A child that reported success but exited non-zero was killed after
            # writing. Believe the exit code.
            return response.model_copy(
                update={
                    "status": EXIT_STATUS.get(exit_code, "internal"),
                    "error_type": "ExitCodeMismatch",
                    "error_message": f"child exited {exit_code} after reporting success",
                }
            )
        return response


def _succeeded(outcome: object) -> bool:
    return isinstance(outcome, SandboxOutcome) and outcome.ok


def _killed_response(exit_code: int, stderr: str, limits: SandboxLimits) -> SandboxResponse:
    """Describe a child that died before it could describe itself.

    A kernel-killed process leaves only its exit status, so the diagnosis has to
    come from the signal: ``SIGXCPU`` is the CPU limit, ``SIGKILL`` is the OOM
    killer, and a ``MemoryError`` on stderr is ``RLIMIT_AS`` refusing an
    allocation the interpreter then reported normally.
    """
    if exit_code == -_SIGXCPU:
        return SandboxResponse(
            status="resource_limit",
            error_type="SandboxResourceLimit",
            error_message=f"child exceeded its CPU limit of {limits.cpu_seconds}s",
        )
    if exit_code == -_SIGKILL or "MemoryError" in stderr:
        return SandboxResponse(
            status="resource_limit",
            error_type="SandboxResourceLimit",
            error_message=f"child exceeded its memory limit of {limits.memory_mb} MB",
        )
    return SandboxResponse(
        status=EXIT_STATUS.get(exit_code, "internal"),
        error_type="SandboxProtocolError",
        error_message=f"child exited {exit_code} without writing a response",
    )
