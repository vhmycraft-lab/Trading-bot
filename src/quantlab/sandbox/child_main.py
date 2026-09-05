"""Entry point executed in the sandbox child process (master spec §21.3).

``python -I -m quantlab.sandbox.child_main <directory>``. Nothing imports this
module; the parent spawns it, and :mod:`quantlab.sandbox.runner` is the only
thing that knows how.

Order matters more than anything else in this file:

1. read the request and resolve the **engine** — trusted infrastructure, loaded
   while imports are still unrestricted;
2. check the strategy source statically (§9.2), so obviously unsafe code never
   reaches step 4;
3. install the run-time guards — imports and sockets — irreversibly;
4. only then execute the untrusted source.

Reversing 3 and 4 by even one statement would hand a strategy an unguarded
interpreter, which is the whole of what this process is for.

This module and :mod:`quantlab.sandbox.ast_check` are the only places in the
codebase permitted to call ``exec``/``compile`` or use ``importlib`` (INV-4).
"""

from __future__ import annotations

import builtins
import contextlib
import importlib
import resource
import sys
import traceback
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, UnionType
from typing import Any

from quantlab.core.errors import SandboxProtocolError, StrategyLoadError, StrategySafetyError
from quantlab.core.strategy import Strategy
from quantlab.core.types import BacktestResult
from quantlab.sandbox.ast_check import FORBIDDEN_NAMES, require_safe_source
from quantlab.sandbox.guards import (
    ForbiddenImport,
    NetworkBlocked,
    block_network,
    install_import_guard,
)
from quantlab.sandbox.protocol import (
    ENGINE_PACKAGE,
    EXIT_BAD_REQUEST,
    EXIT_FORBIDDEN_IMPORT,
    EXIT_INTERNAL,
    EXIT_NETWORK,
    EXIT_OK,
    EXIT_RESOURCE_LIMIT,
    EXIT_STRATEGY_LOAD,
    EXIT_STRATEGY_RUNTIME,
    SandboxRequest,
    SandboxResponse,
    SandboxStatus,
    read_request,
    write_response,
    write_result,
)

__all__ = ["classify_failure", "failure_response", "main", "ok_response"]

#: Name given to the strategy module. Not importable, and not a real package:
#: nothing may import the strategy back by name.
_MODULE_NAME = "quantlab_sandboxed_strategy"


def _rusage() -> tuple[float, float]:
    """CPU seconds and peak RSS in MiB for this process."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    cpu = usage.ru_utime + usage.ru_stime
    # ru_maxrss is KiB on Linux and bytes on macOS.
    rss_mb = usage.ru_maxrss / 1024 if sys.platform != "darwin" else usage.ru_maxrss / (1024 * 1024)
    return cpu, rss_mb


def safe_builtins() -> dict[str, Any]:
    """``builtins`` with the §9.2 names removed.

    The AST check already refuses source that mentions them, so this is the second
    lock on the same door: it also stops a name reached indirectly, and costs one
    dict copy.

    ``__import__`` is the exception and must stay: the ``import`` statement itself
    resolves through it, so removing it would not restrict the strategy's imports
    but abolish them. By the time this is called the name is bound to the guarded
    version, which is the control that actually applies.
    """
    namespace = dict(vars(builtins))
    for name in FORBIDDEN_NAMES - {"__import__"}:
        namespace.pop(name, None)
    return namespace


def _load_engine(request: SandboxRequest) -> Any:
    """Import and construct the engine the parent named.

    Called before the guards go up: the engine is trusted code chosen by the
    caller, not by the strategy. The sandbox layer must not name an adapter
    itself (INV-8), so the name travels in the request — which makes the name the
    one thing an attacker would want to control, and the reason it is checked
    twice.

    :class:`SandboxRequest` validates the name on construction; this re-checks it
    against the same constant at the moment of use, so a request built by any
    route that skips validation (``model_construct``, a future refactor) still
    cannot turn this call into "import whatever I say". The resolved object then
    has to look like an engine before it is used as one.

    Raises:
        SandboxProtocolError: the name is outside the engine package, or what it
            resolves to is not an engine. Fail closed: nothing is instantiated
            speculatively to find out.
    """
    name = request.engine_module
    if not name.startswith(f"{ENGINE_PACKAGE}.") or not all(
        part.isidentifier() and not part.startswith("_")
        for part in name[len(ENGINE_PACKAGE) + 1 :].split(".")
    ):
        raise SandboxProtocolError(
            "engine_module is outside the engine package", module=name, allowed=ENGINE_PACKAGE
        )
    if not request.engine_class.isidentifier() or request.engine_class.startswith("_"):
        raise SandboxProtocolError("engine_class is not a public name", name=request.engine_class)

    module = importlib.import_module(name)
    engine_type = getattr(module, request.engine_class, None)
    if not isinstance(engine_type, type):
        raise SandboxProtocolError(
            "engine_class does not name a class in the engine module",
            module=name,
            name=request.engine_class,
        )
    engine = engine_type()
    # Structural, not nominal: quantlab.ports is off-limits to this layer (INV-8),
    # so the contract is checked by shape. An object that cannot run a backtest
    # must not reach the point where untrusted code is already loaded.
    if not callable(getattr(engine, "run", None)) or not isinstance(
        getattr(engine, "name", None), str
    ):
        raise SandboxProtocolError(
            "the named class is not a backtest engine", module=name, name=request.engine_class
        )
    return engine


def load_strategy(source: str) -> Strategy:
    """Execute the strategy source in a namespace of its own and return the class.

    Every guard is already installed when this runs.
    """
    module = ModuleType(_MODULE_NAME)
    module.__dict__["__builtins__"] = safe_builtins()
    code = compile(source, "strategy.py", "exec")
    exec(code, module.__dict__)  # noqa: S102 - the point of this process
    exported = module.__dict__.get("STRATEGY")
    if exported is None:
        raise StrategyLoadError("the strategy module set no STRATEGY")
    instance = exported()
    return instance  # type: ignore[no-any-return]


def _run(directory: Path) -> int:
    request, bars, source = read_request(directory)

    engine = _load_engine(request)

    if request.ast_check:
        require_safe_source(source, allowed_imports=request.allowed_imports)

    block_network()
    install_import_guard(request.allowed_imports, (_MODULE_NAME,))

    strategy = load_strategy(source)
    result: BacktestResult = engine.run(strategy, bars, dict(request.params), request.config)

    cpu, rss = _rusage()
    write_result(directory, result)
    write_response(directory, ok_response(result, cpu, rss))
    return EXIT_OK


def chain(exc: BaseException) -> Iterator[BaseException]:
    """``exc`` and everything it was raised from.

    The engine wraps whatever a strategy raises in a ``StrategyRuntimeError`` so
    it can say which bar failed. A guard tripping deep inside that call is still a
    guard tripping, and must be reported as one rather than as ordinary strategy
    misbehaviour — the two mean very different things about the candidate.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        yield current
        seen.add(id(current))
        current = current.__cause__ or current.__context__


def caused_by(exc: BaseException, kind: type[BaseException] | UnionType) -> bool:
    return any(isinstance(link, kind) for link in chain(exc))


def classify_failure(exc: BaseException) -> tuple[SandboxStatus, int]:
    """Say what kind of failure ``exc`` is, and which exit code reports it.

    The distinctions matter to the caller: a guard that tripped says something
    about the *candidate*, a resource limit says something about the *run*, and a
    strategy that raised says something about the code. Collapsing them into one
    "it failed" would leave an evolutionary generation unable to tell a hostile
    candidate from an expensive one.
    """
    if caused_by(exc, MemoryError):
        # RLIMIT_AS refuses an allocation and the interpreter reports it as an
        # ordinary MemoryError, often wrapped by whatever was allocating. That is
        # the memory ceiling doing its job, not a defect in the strategy's logic.
        return "resource_limit", EXIT_RESOURCE_LIMIT
    if caused_by(exc, ForbiddenImport):
        return "forbidden_import", EXIT_FORBIDDEN_IMPORT
    if caused_by(exc, NetworkBlocked):
        return "network", EXIT_NETWORK
    if caused_by(exc, SandboxProtocolError):
        return "bad_request", EXIT_BAD_REQUEST
    if caused_by(exc, StrategySafetyError | StrategyLoadError | SyntaxError):
        return "strategy_load", EXIT_STRATEGY_LOAD
    if isinstance(exc, Exception):
        return "strategy_runtime", EXIT_STRATEGY_RUNTIME
    return "internal", EXIT_INTERNAL


def failure_response(exc: BaseException) -> SandboxResponse:
    """Describe a failure for the parent, without letting anything execute there.

    Only the exception's *text* crosses: the traceback stays in the child's
    stderr, so nothing the strategy authored is re-interpreted on the other side.
    """
    status, _ = classify_failure(exc)
    codes: tuple[str, ...] = ()
    for link in chain(exc):
        codes = tuple(getattr(link, "codes", ()) or ())
        if codes:
            break
    cpu, rss = _rusage()
    return SandboxResponse(
        status=status,
        error_type=type(exc).__name__,
        error_message=str(exc)[:4000],
        error_codes=codes,
        cpu_seconds=cpu,
        peak_rss_mb=rss,
    )


def ok_response(result: BacktestResult, cpu: float, rss: float) -> SandboxResponse:
    """Everything about a successful run that is not one of the three tables."""
    return SandboxResponse(
        status="ok",
        engine_name=result.engine_name,
        engine_version=result.engine_version,
        warmup_bars=result.warmup_bars,
        n_bars=result.n_bars,
        bars_per_year=result.bars_per_year,
        cost_summary=dict(result.cost_summary),
        log=tuple(result.log),
        ruined=result.ruined,
        cpu_seconds=cpu,
        peak_rss_mb=rss,
    )


def _report_failure(directory: Path, exc: BaseException) -> int:
    """Write the failure down and return its exit code.

    Writing is best effort: a child killed by the kernel writes nothing at all,
    and the parent falls back to the exit code and the signal.
    """
    _, code = classify_failure(exc)
    # The directory may be gone; the exit code still carries the verdict.
    with contextlib.suppress(OSError):
        write_response(directory, failure_response(exc))
    traceback.print_exception(exc, file=sys.stderr)
    return code


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m quantlab.sandbox.child_main <directory>", file=sys.stderr)
        return EXIT_BAD_REQUEST
    directory = Path(args[0])
    try:
        return _run(directory)
    except BaseException as exc:  # the child must never leave a bare traceback
        return _report_failure(directory, exc)


if __name__ == "__main__":  # pragma: no cover - exercised as a child process
    raise SystemExit(main())
