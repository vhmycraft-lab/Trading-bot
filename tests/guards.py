"""The scanners behind the structural guards, extracted so they can be tested.

Three guards in this suite check a *property of the source* rather than of a
running program: the dashboard writes nothing to the store, the dashboard opens
no database of its own, and the paper CLI offers no way to go live.

All three began as text searches over the source files, and all three fired on
their own documentation — the comment naming `create_db_engine`, the docstring
saying there is no `--live` flag, the fixtures in the redaction tests. Each was
then rewritten to look at structure instead: the AST for the first two, the
rendered Typer surface for the third.

That rewrite is exactly the moment a guard can quietly stop working. A scanner
narrowed until it no longer trips on prose can be narrowed a little too far and
trip on nothing at all, and nothing about the suite would look different: the
guard still runs, still passes, still reads like protection.

So the logic lives here, as functions over strings, and
``tests/unit/test_guard_meta.py`` feeds each one a violation it must catch and a
piece of prose it must ignore. A guard is only evidence if both directions are
pinned.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable

__all__ = [
    "declared_option_names",
    "engine_builder_uses",
    "write_method_calls",
]


def write_method_calls(source: str, methods: Iterable[str]) -> list[str]:
    """Every call in ``source`` to an attribute named in ``methods``.

    Matched on the attribute name, so ``store.record_verdict(...)`` is caught
    however the store was obtained or aliased. A bare name (``record_verdict()``
    with no receiver) is *not* matched: at that point it is a local function, and
    a module defining its own helper called ``create_run`` has not written to the
    store.
    """
    wanted = frozenset(methods)
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in wanted
        ):
            found.append(f"{node.lineno}:{node.func.attr}")
    return found


def engine_builder_uses(source: str, names: Iterable[str]) -> list[str]:
    """Every call to, or import of, a name in ``names``.

    Both forms matter and they fail differently: a module that *calls*
    ``create_db_engine`` has built its own connection, and a module that merely
    *imports* it is one line away from doing so while looking clean in a diff.
    """
    wanted = frozenset(names)
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in wanted:
                found.append(f"{node.lineno}:{name}()")
        elif isinstance(node, ast.ImportFrom):
            found.extend(
                f"{node.lineno}:import {alias.name}" for alias in node.names if alias.name in wanted
            )
    return found


def declared_option_names(app: object) -> set[str]:
    """Every ``--option`` a Typer app declares, across its subcommands.

    Read from Click's parameter objects, not from rendered help text and not
    from the module source. Both of those carry prose: the source has the
    docstring saying there is no ``--live`` flag, and Typer *renders that same
    docstring into the help output*, so scanning the help re-introduces exactly
    the false positive that scanning the source had. Only the parameter list
    knows what the command will actually accept.
    """
    import typer.main

    command = typer.main.get_command(app)  # type: ignore[arg-type]
    options: set[str] = set()

    def collect(cmd: object) -> None:
        for param in getattr(cmd, "params", []):
            for opt in list(getattr(param, "opts", [])) + list(
                getattr(param, "secondary_opts", [])
            ):
                if isinstance(opt, str) and opt.startswith("--"):
                    options.add(opt)
        for sub in getattr(cmd, "commands", {}).values():
            collect(sub)

    collect(command)
    return options
