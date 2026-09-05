"""Static safety check for strategy source (master spec section 9.2).

The first of four layers between an untrusted strategy and this process. It runs
before the code is executed anywhere, so a rejected file never reaches an
interpreter: the source is parsed, not run, and :mod:`ast` cannot be tricked into
executing what it parses.

It is deliberately **conservative**. A rule here can reject a legitimate strategy,
which costs one rewrite; a rule that is too permissive lets a search process
discover that it can reach the filesystem, which costs everything. Every rejection
carries a machine-readable code so the researcher — human or model — is told what
to change rather than merely that something was wrong.

This is not the only defence. The sandbox's import hook (§21.3) blocks at run
time what this blocks at parse time, and the leakage probe (§14.2) catches what
neither can see.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Final

from quantlab.core.errors import StrategySafetyError

__all__ = [
    "ALWAYS_ALLOWED_IMPORTS",
    "DEFAULT_ALLOWED_IMPORTS",
    "DEFAULT_MAX_LOGIC_LINES",
    "DEFAULT_MAX_PARAMS",
    "FORBIDDEN_MODULES",
    "FORBIDDEN_NAMES",
    "MAX_SOURCE_LINES",
    "AstReport",
    "Violation",
    "check_source",
    "count_logic_lines",
    "require_safe_source",
]

#: Names a strategy may never mention, anywhere (spec section 9.2).
FORBIDDEN_NAMES: Final[frozenset[str]] = frozenset(
    {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "breakpoint",
        "input",
        "exit",
        "quit",
    }
)

#: Modules a strategy may never import, however it spells the import.
FORBIDDEN_MODULES: Final[frozenset[str]] = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "socket",
        "http",
        "urllib",
        "requests",
        "httpx",
        "pathlib",
        "shutil",
        "pickle",
        "marshal",
        "ctypes",
        "threading",
        "multiprocessing",
        "asyncio",
        "importlib",
        "inspect",
        "builtins",
        "time",
        "datetime",
        "random",
    }
)

#: The default allow-list, mirroring ``sandbox.allowed_imports`` in the config.
DEFAULT_ALLOWED_IMPORTS: Final[tuple[str, ...]] = (
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

#: Always importable regardless of configuration: ``__future__`` is a compiler
#: directive with no run-time surface, and forbidding it would reject the
#: ``from __future__ import annotations`` line every modern module carries.
ALWAYS_ALLOWED_IMPORTS: Final[frozenset[str]] = frozenset({"__future__"})

#: Source files longer than this are rejected (spec section 9.2).
MAX_SOURCE_LINES: Final[int] = 400

#: Load-time parameter ceiling: ``validation.max_free_params + 4`` (spec section 9.2).
#: The soft complexity penalty applies from ``max_free_params`` upwards; this is the
#: hard limit at which a strategy is refused outright.
PARAM_LIMIT_SLACK: Final[int] = 4
DEFAULT_MAX_PARAMS: Final[int] = 6 + PARAM_LIMIT_SLACK

#: Above this the complexity penalty of spec section 14 starts to bite. Not a
#: rejection — a long strategy can still be sound — but the author is told.
DEFAULT_MAX_LOGIC_LINES: Final[int] = 150

#: Module-level assignments that are not mutable state.
_ALLOWED_MODULE_ASSIGNMENTS: Final[frozenset[str]] = frozenset({"STRATEGY", "__all__"})

#: Container literals that would be shared across every instance if module-level.
_MUTABLE_LITERALS: Final[tuple[type[ast.AST], ...]] = (ast.List, ast.Dict, ast.Set)


@dataclass(frozen=True, slots=True)
class Violation:
    """One rule a source file broke."""

    code: str
    message: str
    line: int = 0

    def __str__(self) -> str:
        where = f" (line {self.line})" if self.line else ""
        return f"{self.code}: {self.message}{where}"


@dataclass(frozen=True, slots=True)
class AstReport:
    """What the checker found, whether or not it rejected the file."""

    violations: tuple[Violation, ...] = ()
    warnings: tuple[Violation, ...] = ()
    class_name: str = ""
    logic_lines: int = 0
    n_params: int = 0
    n_lines: int = 0

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(violation.code for violation in self.violations)

    def describe(self) -> str:
        lines = [str(violation) for violation in self.violations]
        lines.extend(f"warning {violation}" for violation in self.warnings)
        return "\n".join(lines)


def count_logic_lines(source: str) -> int:
    """Non-blank, non-comment, non-import lines (spec section 9.2)."""
    total = 0
    for raw in source.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("import ", "from ")):
            continue
        total += 1
    return total


def _module_root(name: str) -> str:
    return name.split(".")[0]


def _import_is_allowed(name: str, allowed: frozenset[str]) -> bool:
    if name in ALWAYS_ALLOWED_IMPORTS or name in allowed:
        return True
    # ``numpy.linalg`` is allowed by ``numpy``; ``quantlab.core.strategy`` is not
    # allowed by ``quantlab``, because the allow-list names full module paths.
    return any(name.startswith(f"{prefix}.") for prefix in allowed)


class _Checker(ast.NodeVisitor):
    """Walks the module once, collecting violations."""

    def __init__(self, allowed: frozenset[str], max_params: int) -> None:
        self.allowed = allowed
        self.max_params = max_params
        self.violations: list[Violation] = []
        self.warnings: list[Violation] = []
        self._in_class_body = 0

    def add(self, code: str, message: str, node: ast.AST) -> None:
        self.violations.append(Violation(code, message, getattr(node, "lineno", 0)))

    # -- imports -----------------------------------------------------------
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._check_import(alias.name, node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            self.add("E_RELATIVE_IMPORT", "relative imports are not allowed", node)
        else:
            # An absolute `from ... import ...` always names a module; the grammar
            # only leaves `module` empty for the relative form handled above.
            module = node.module or ""
            if any(alias.name == "*" for alias in node.names):
                self.add("E_STAR_IMPORT", f"`from {module} import *` is not allowed", node)
            self._check_import(module, node)
        self.generic_visit(node)

    def _check_import(self, name: str, node: ast.AST) -> None:
        if _module_root(name) in FORBIDDEN_MODULES:
            self.add("E_FORBIDDEN_IMPORT", f"module {name!r} is forbidden", node)
        elif not _import_is_allowed(name, self.allowed):
            self.add(
                "E_IMPORT_NOT_ALLOWED",
                f"module {name!r} is not in the allow-list {sorted(self.allowed)}",
                node,
            )

    # -- names and attributes ----------------------------------------------
    def visit_Name(self, node: ast.Name) -> None:
        if node.id in FORBIDDEN_NAMES:
            self.add("E_FORBIDDEN_NAME", f"name {node.id!r} is forbidden", node)
        elif node.id in FORBIDDEN_MODULES:
            self.add("E_FORBIDDEN_NAME", f"name {node.id!r} refers to a forbidden module", node)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            self.add(
                "E_DUNDER_ACCESS",
                f"attribute {node.attr!r} reaches into interpreter internals",
                node,
            )
        self._check_intrabar_stop(node)
        self.generic_visit(node)

    def _check_intrabar_stop(self, node: ast.Attribute) -> None:
        """Reject the shape of a hand-rolled stop (spec section 9.1).

        Comparing a bar's own ``high``/``low`` against the position's entry price
        is how a strategy tries to simulate an intrabar stop. It cannot be right:
        the bar's range says a level was touched, not when or in what order, and
        assuming an order is exactly the optimism the engine refuses in §8.6. A
        strategy that wants a stop declares a ``RiskSpec`` and lets the engine
        apply it pessimistically.

        The rule is narrow on purpose. Reading a closed bar's high or low is
        perfectly legitimate — a breakout strategy needs it — so only a comparison
        that *also* mentions an entry price is refused.
        """
        if node.attr not in ("high", "low"):
            return
        if getattr(node, "_quantlab_compare_peer", False):
            self.add(
                "E_INTRABAR_STOP",
                "comparing a bar's own high/low against the entry price is a hand-rolled "
                "intrabar stop; declare a RiskSpec and let the engine apply it (spec 8.6)",
                node,
            )

    # -- module-level structure --------------------------------------------
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._in_class_body += 1
        self.generic_visit(node)
        self._in_class_body -= 1


def _mark_intrabar_comparisons(tree: ast.Module) -> None:
    """Tag ``bars.high``/``bars.low`` attributes compared against an entry price."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        mentions_entry = any(
            isinstance(part, ast.Attribute) and "entry" in part.attr
            for operand in operands
            for part in ast.walk(operand)
        )
        if not mentions_entry:
            continue
        for operand in operands:
            for part in ast.walk(operand):
                if isinstance(part, ast.Attribute) and part.attr in ("high", "low"):
                    part._quantlab_compare_peer = True  # type: ignore[attr-defined]


def _check_module_body(
    tree: ast.Module, report: list[Violation], exported_name: str = ""
) -> ast.ClassDef | None:
    """Module level may hold imports, constants, one class, and ``STRATEGY``."""
    classes: list[ast.ClassDef] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # a docstring
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if not isinstance(target, ast.Name):
                    report.append(
                        Violation(
                            "E_MODULE_CODE",
                            "module level allows simple assignments only",
                            getattr(node, "lineno", 0),
                        )
                    )
                    continue
                if target.id in _ALLOWED_MODULE_ASSIGNMENTS:
                    continue
                if not target.id.isupper():
                    report.append(
                        Violation(
                            "E_MODULE_STATE",
                            f"module-level name {target.id!r} must be a CONSTANT; "
                            "strategies keep state on the instance",
                            getattr(node, "lineno", 0),
                        )
                    )
                elif node.value is not None and isinstance(node.value, _MUTABLE_LITERALS):
                    report.append(
                        Violation(
                            "E_MUTABLE_MODULE_STATE",
                            f"module-level {target.id!r} is mutable and would be shared "
                            "across every run",
                            getattr(node, "lineno", 0),
                        )
                    )
        else:
            report.append(
                Violation(
                    "E_MODULE_CODE",
                    f"module level may not contain {type(node).__name__}",
                    getattr(node, "lineno", 0),
                )
            )

    if not classes:
        report.append(Violation("E_NO_CLASS", "no strategy class found", 0))
        return None
    if len(classes) > 1:
        report.append(
            Violation(
                "E_MULTIPLE_CLASSES",
                f"exactly one class is allowed, found {len(classes)}",
                classes[1].lineno,
            )
        )
    # With more than one class the file is rejected either way, but the remaining
    # checks should describe the class the module actually exports rather than
    # whichever happened to be typed first — otherwise one mistake reports as five.
    for cls in classes:
        if cls.name == exported_name:
            return cls
    return classes[0]


def _find_class_attribute(cls: ast.ClassDef, name: str) -> ast.expr | None:
    for node in cls.body:
        if isinstance(node, ast.Assign):
            targets: list[ast.expr] = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id == name:
                return node.value
    return None


def _check_params(cls: ast.ClassDef, max_params: int, report: list[Violation]) -> int:
    """``params`` must exist, be a dict literal, and every entry must be bounded."""
    value = _find_class_attribute(cls, "params")
    if value is None:
        report.append(Violation("E_NO_PARAMS", "the class must declare `params`", cls.lineno))
        return 0
    if not isinstance(value, ast.Dict):
        report.append(
            Violation("E_PARAMS_NOT_LITERAL", "`params` must be a dict literal", value.lineno)
        )
        return 0

    n_params = len(value.keys)
    if n_params > max_params:
        report.append(
            Violation(
                "E_TOO_MANY_PARAMS",
                f"{n_params} parameters declared; the limit is {max_params}",
                value.lineno,
            )
        )

    for key, entry in zip(value.keys, value.values, strict=True):
        label = key.value if isinstance(key, ast.Constant) else "?"
        if not (
            isinstance(entry, ast.Call)
            and isinstance(entry.func, ast.Name)
            and entry.func.id == "ParamSpec"
        ):
            report.append(
                Violation(
                    "E_PARAM_NOT_SPEC",
                    f"parameter {label!r} must be a ParamSpec(...) call",
                    entry.lineno,
                )
            )
            continue
        supplied = {kw.arg for kw in entry.keywords}
        kind_node = next((kw.value for kw in entry.keywords if kw.arg == "kind"), None)
        kind = kind_node.value if isinstance(kind_node, ast.Constant) else None
        if kind in ("int", "float") and not {"low", "high"} <= supplied:
            report.append(
                Violation(
                    "E_UNBOUNDED_PARAM",
                    f"parameter {label!r} must declare both `low` and `high`; an "
                    "unbounded parameter cannot be searched or mutated",
                    entry.lineno,
                )
            )
        if kind == "categorical" and "choices" not in supplied:
            report.append(
                Violation(
                    "E_UNBOUNDED_PARAM",
                    f"categorical parameter {label!r} must declare `choices`",
                    entry.lineno,
                )
            )
    return n_params


def _check_warmup(cls: ast.ClassDef, report: list[Violation]) -> None:
    value = _find_class_attribute(cls, "warmup_bars")
    if value is None:
        report.append(Violation("E_NO_WARMUP", "the class must declare `warmup_bars`", cls.lineno))
        return
    try:
        literal: object = ast.literal_eval(value)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        literal = None
    if not isinstance(literal, int) or isinstance(literal, bool):
        report.append(
            Violation(
                "E_WARMUP_NOT_INT",
                "`warmup_bars` must be a literal int, so it can be checked before the "
                "strategy runs",
                value.lineno,
            )
        )
    elif literal < 0:
        report.append(
            Violation("E_WARMUP_NEGATIVE", "`warmup_bars` must not be negative", value.lineno)
        )


def _check_export(tree: ast.Module, cls: ast.ClassDef | None, report: list[Violation]) -> None:
    exported = _find_module_assignment(tree, "STRATEGY")
    if exported is None:
        report.append(Violation("E_NO_STRATEGY_EXPORT", "the module must set `STRATEGY`", 0))
        return
    if cls is not None and not (isinstance(exported, ast.Name) and exported.id == cls.name):
        report.append(
            Violation(
                "E_STRATEGY_MISMATCH",
                f"`STRATEGY` must name the declared class {cls.name!r}",
                exported.lineno,
            )
        )


def _find_module_assignment(tree: ast.Module, name: str) -> ast.expr | None:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node.value
    return None


def check_source(
    source: str,
    *,
    allowed_imports: tuple[str, ...] = DEFAULT_ALLOWED_IMPORTS,
    max_params: int = DEFAULT_MAX_PARAMS,
    max_lines: int = MAX_SOURCE_LINES,
    max_logic_lines: int = DEFAULT_MAX_LOGIC_LINES,
) -> AstReport:
    """Check strategy source against spec section 9.2 and report every violation.

    Never raises for a *rejected* strategy — the report carries the reasons, so a
    caller can show all of them at once instead of one per attempt. A file that
    will not parse is the one exception, since there is nothing to report on.
    """
    lines = source.splitlines()
    if len(lines) > max_lines:
        return AstReport(
            violations=(
                Violation(
                    "E_FILE_TOO_LONG",
                    f"{len(lines)} lines; the limit is {max_lines}",
                    max_lines + 1,
                ),
            ),
            n_lines=len(lines),
        )

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return AstReport(
            violations=(
                Violation("E_SYNTAX", f"source does not parse: {exc.msg}", exc.lineno or 0),
            ),
            n_lines=len(lines),
        )

    violations: list[Violation] = []
    exported = _find_module_assignment(tree, "STRATEGY")
    exported_name = exported.id if isinstance(exported, ast.Name) else ""
    cls = _check_module_body(tree, violations, exported_name)
    if cls is not None:
        _check_params(cls, max_params, violations)
        _check_warmup(cls, violations)
    _check_export(tree, cls, violations)

    _mark_intrabar_comparisons(tree)
    checker = _Checker(frozenset(allowed_imports), max_params)
    checker.visit(tree)
    violations.extend(checker.violations)

    n_params = 0
    if cls is not None:
        params_node = _find_class_attribute(cls, "params")
        if isinstance(params_node, ast.Dict):
            n_params = len(params_node.keys)

    logic_lines = count_logic_lines(source)
    if logic_lines > max_logic_lines:
        checker.warnings.append(
            Violation(
                "W_LOGIC_LINES",
                f"{logic_lines} logic lines; above {max_logic_lines} the complexity "
                "penalty of spec section 14 applies",
                0,
            )
        )

    violations.sort(key=lambda v: (v.line, v.code))
    return AstReport(
        violations=tuple(violations),
        warnings=tuple(checker.warnings),
        class_name=cls.name if cls else "",
        logic_lines=logic_lines,
        n_params=n_params,
        n_lines=len(lines),
    )


def require_safe_source(
    source: str,
    *,
    allowed_imports: tuple[str, ...] = DEFAULT_ALLOWED_IMPORTS,
    max_params: int = DEFAULT_MAX_PARAMS,
    max_lines: int = MAX_SOURCE_LINES,
    max_logic_lines: int = DEFAULT_MAX_LOGIC_LINES,
) -> AstReport:
    """Check ``source`` and raise if it is not safe.

    Raises:
        StrategySafetyError: carrying every violation code, so the caller can act
            on all of them rather than discovering them one rejection at a time.
    """
    report = check_source(
        source,
        allowed_imports=allowed_imports,
        max_params=max_params,
        max_lines=max_lines,
        max_logic_lines=max_logic_lines,
    )
    if not report.ok:
        raise StrategySafetyError(
            f"strategy source rejected by the AST checker:\n{report.describe()}",
            violations=[str(violation) for violation in report.violations],
            codes=list(report.codes),
        )
    return report
