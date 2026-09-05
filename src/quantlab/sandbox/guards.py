"""Run-time guards installed in the sandbox child before untrusted code loads.

The AST check (§9.2) is a *parse-time* control and can only see what the source
says. These are the run-time controls that catch what a source can arrange to do
without naming it: an import resolved through a string, a socket opened by a
library the strategy legitimately imports, a subprocess spawned from inside a
callback. They cost nothing when nothing is trying, and the process they protect
holds no secrets and no network credentials in the first place — this is the
second of four layers, not the only one.

Nothing here may be installed in the parent process. Both functions are
irreversible by design: there is no uninstall, because a strategy that could ask
for one would ask.
"""

from __future__ import annotations

import builtins
import contextlib
import sys
from collections.abc import Callable, Sequence
from typing import Any, Final

__all__ = [
    "STDLIB_ALLOWED",
    "ForbiddenImport",
    "NetworkBlocked",
    "apply_network_block",
    "block_network",
    "import_is_allowed",
    "install_import_guard",
    "make_guarded_import",
    "network_raiser",
]


class ForbiddenImport(ImportError):
    """A module outside the allow-list was imported at run time."""


class NetworkBlocked(OSError):
    """Something in the child tried to open a socket."""


#: The stdlib a strategy may reach without being named in ``allowed_imports``.
#: Deliberately tiny: pure computation and the modules numpy/pandas pull in on
#: their own paths. Anything that touches the world outside the process is absent.
STDLIB_ALLOWED: Final[frozenset[str]] = frozenset(
    {
        "__future__",
        "abc",
        "array",
        "bisect",
        "cmath",
        "collections",
        "contextlib",
        "copy",
        "copyreg",
        "dataclasses",
        "decimal",
        "enum",
        "fractions",
        "functools",
        "heapq",
        "itertools",
        "json",
        "keyword",
        "math",
        "numbers",
        "operator",
        "re",
        "reprlib",
        "statistics",
        "string",
        "textwrap",
        "types",
        "typing",
        "typing_extensions",
        "unicodedata",
        "warnings",
        "weakref",
    }
)


def import_is_allowed(name: str, allowed: frozenset[str]) -> bool:
    """Whether ``name`` may be imported, given the request's allow-list.

    A submodule of an allowed module is allowed (``numpy.linalg`` follows
    ``numpy``); a parent of one is not (``quantlab`` is not opened up by
    ``quantlab.core.strategy``), because the allow-list names full module paths.
    """
    if name in STDLIB_ALLOWED or name in allowed:
        return True
    return any(name.startswith(f"{prefix}.") for prefix in (*allowed, *STDLIB_ALLOWED))


def make_guarded_import(
    real_import: Callable[..., Any], allowed: Sequence[str], sandboxed_modules: Sequence[str]
) -> Callable[..., Any]:
    """Build the replacement for ``builtins.__import__``.

    Separate from installing it so the rule can be tested directly: installation
    is irreversible by design, and a test process that installed it could not go
    on to test anything else.

    The rule applies to imports whose *caller* is one of ``sandboxed_modules``,
    identified by the ``__name__`` in the calling frame's globals. That is the
    boundary the threat model actually draws: the question is what the strategy
    may reach, not what NumPy does inside its own machinery. A blanket rule breaks
    trusted libraries, which import lazily all the time — pydantic reaching for
    ``pydantic_core`` mid-validation is not an escape attempt — and a sandbox that
    has to be switched off to get work done protects nothing.

    A strategy cannot forge its way out: ``__import__``, ``exec``, ``eval`` and
    ``globals`` are rejected by the AST check (§9.2) and removed from the namespace
    it executes in, and a function it defines still carries its own module globals
    wherever it is later called from.
    """
    permitted = frozenset(allowed)
    sandboxed = frozenset(sandboxed_modules)

    def guarded_import(
        name: str,
        globals_: dict[str, Any] | None = None,
        locals_: dict[str, Any] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> Any:
        caller = str((globals_ or {}).get("__name__", ""))
        if caller in sandboxed:
            if level != 0:
                raise ForbiddenImport(f"relative import of {name!r} is not allowed in the sandbox")
            if not import_is_allowed(name, permitted):
                raise ForbiddenImport(
                    f"import of {name!r} is not allowed in the sandbox; "
                    f"allowed: {sorted(permitted)}"
                )
        return real_import(name, globals_, locals_, fromlist, level)

    return guarded_import


def install_import_guard(allowed: Sequence[str], sandboxed_modules: Sequence[str]) -> None:
    """Refuse every import the *sandboxed* modules make outside ``allowed``.

    Wraps ``builtins.__import__`` rather than adding a meta-path finder: a finder
    is consulted only for modules not already in ``sys.modules``, so it would
    silently permit anything the interpreter loaded during start-up — including
    the modules this guard exists to keep out of reach.

    There is no uninstall, because a strategy that could ask for one would.
    """
    builtins.__import__ = make_guarded_import(builtins.__import__, allowed, sandboxed_modules)


def network_raiser(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001 - a stand-in must
    # accept whatever the real callable accepts, and use none of it.
    raise NetworkBlocked("network access is not allowed in the sandbox")


def apply_network_block(module: Any) -> None:
    """Make socket creation raise for one already-imported module.

    Two steps, because either alone leaves a hole. Replacing the module
    attributes (``socket.socket`` and friends) catches the ordinary call. Patching
    the *class* catches a library that bound ``socket.socket`` to a local name
    before this ran — rebinding the module attribute would leave that reference
    pointing at a working original.
    """
    socket_type = getattr(module, "socket", None)
    if isinstance(socket_type, type):
        for method in ("__init__", "connect", "connect_ex", "bind"):
            if hasattr(socket_type, method):
                with contextlib.suppress(TypeError, AttributeError):
                    setattr(socket_type, method, network_raiser)
    for attribute in ("socket", "socketpair", "create_connection", "create_server", "getaddrinfo"):
        if hasattr(module, attribute):
            setattr(module, attribute, network_raiser)


def block_network(modules: Sequence[str] = ("socket",)) -> None:
    """Block socket creation in every named module that is already imported.

    The import guard is the primary control; this closes the case where a module
    the strategy is allowed to use opens the connection on its behalf.
    """
    for name in modules:
        module = sys.modules.get(name)
        if module is not None:
            apply_network_block(module)
