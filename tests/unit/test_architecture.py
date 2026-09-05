"""Invariants INV-1, INV-4 and INV-8, enforced by static analysis of ``src/``.

These tests must never be deleted or weakened (master spec section 0.2).  They
are the reason a reviewer can trust, without reading the whole tree, that this
codebase cannot place an order and cannot execute untrusted code in-process.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "quantlab"

# --- INV-1 ------------------------------------------------------------------
FORBIDDEN_CLASS_NAMES = frozenset({"LiveBroker", "RealBroker", "ExchangeBroker"})
FORBIDDEN_ORDER_TOKENS = (
    "create" + "_order",
    "create" + "Order",
    "create" + "_market_order",
    "private" + "_post_order",
)

# --- INV-4 ------------------------------------------------------------------
FORBIDDEN_BUILTINS = frozenset({"exec", "eval", "compile", "__import__"})
FORBIDDEN_MODULES = frozenset({"importlib", "pickle", "marshal", "shelve"})
SANDBOX_PACKAGE = "quantlab.sandbox"

# --- INV-8 ------------------------------------------------------------------
#: module prefix -> the quantlab sub-packages it may import
ALLOWED_IMPORTS: dict[str, frozenset[str]] = {
    "quantlab.core": frozenset({"quantlab.core"}),
    "quantlab.ports": frozenset({"quantlab.core", "quantlab.ports"}),
    "quantlab.sandbox": frozenset({"quantlab.core", "quantlab.sandbox"}),
    "quantlab.adapters": frozenset({"quantlab.core", "quantlab.ports", "quantlab.adapters"}),
    "quantlab.research": frozenset({"quantlab.core", "quantlab.ports", "quantlab.research"}),
    "quantlab.optimize": frozenset({"quantlab.core", "quantlab.ports", "quantlab.optimize"}),
    "quantlab.walkforward": frozenset(
        {"quantlab.core", "quantlab.ports", "quantlab.walkforward", "quantlab.optimize"}
    ),
    "quantlab.paper": frozenset({"quantlab.core", "quantlab.ports", "quantlab.paper"}),
    "quantlab.reporting": frozenset({"quantlab.core", "quantlab.ports", "quantlab.reporting"}),
    "quantlab.experiments": frozenset({"quantlab.core", "quantlab.ports", "quantlab.experiments"}),
    "quantlab.strategies_io": frozenset(
        {"quantlab.core", "quantlab.ports", "quantlab.sandbox", "quantlab.strategies_io"}
    ),
}

#: Only these modules may import ``quantlab.adapters`` (spec section 2.3).
#: ``quantlab.adapters`` itself is listed because an adapter package may wire
#: its own submodules together; the layer test above still bounds what it reaches.
ADAPTER_IMPORTERS = ("quantlab.container", "quantlab.cli", "quantlab.adapters")


def _python_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def _module_name(path: Path) -> str:
    relative = path.relative_to(SRC.parent).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imported_modules(tree: ast.Module) -> Iterator[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module


def _quantlab_prefix(module: str) -> str | None:
    parts = module.split(".")
    if parts[0] != "quantlab" or len(parts) < 2:
        return None
    return ".".join(parts[:2])


def _layer_of(module: str) -> str | None:
    for prefix in ALLOWED_IMPORTS:
        if module == prefix or module.startswith(prefix + "."):
            return prefix
    return None


def test_source_tree_is_not_empty() -> None:
    """Guards against the scans below passing vacuously."""
    assert len(_python_files()) >= 10


# ---------------------------------------------------------------------------
# INV-1: no real-money order path
# ---------------------------------------------------------------------------
def test_inv1_no_live_broker_class() -> None:
    offenders: list[str] = []
    for path in _python_files():
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.ClassDef) and node.name in FORBIDDEN_CLASS_NAMES:
                offenders.append(f"{path}:{node.lineno} class {node.name}")
    assert not offenders, "INV-1: real-broker class defined: " + "; ".join(offenders)


def test_inv1_no_exchange_order_calls() -> None:
    offenders: list[str] = []
    for path in _python_files():
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_ORDER_TOKENS:
            if token in text:
                offenders.append(f"{path}: {token}")
    assert not offenders, "INV-1: exchange order API referenced: " + "; ".join(offenders)


# ---------------------------------------------------------------------------
# INV-4: untrusted code never runs in the main process
# ---------------------------------------------------------------------------
def _is_sandbox(module: str) -> bool:
    return module == SANDBOX_PACKAGE or module.startswith(SANDBOX_PACKAGE + ".")


def test_inv4_no_dynamic_execution_outside_sandbox() -> None:
    offenders: list[str] = []
    for path in _python_files():
        module = _module_name(path)
        if _is_sandbox(module):
            continue
        tree = _parse(path)
        for node in ast.walk(tree):
            # A bare name call: exec(...), eval(...), compile(...).
            # `re.compile` is an attribute access and is deliberately not matched.
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in FORBIDDEN_BUILTINS
            ):
                offenders.append(f"{path}:{node.lineno} {node.func.id}()")
        for imported in _imported_modules(tree):
            root = imported.split(".")[0]
            if root in FORBIDDEN_MODULES:
                offenders.append(f"{path}: import {imported}")
    assert not offenders, "INV-4: dynamic execution outside sandbox/: " + "; ".join(offenders)


# ---------------------------------------------------------------------------
# INV-8: import boundaries
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", _python_files(), ids=_module_name)
def test_inv8_import_boundaries(path: Path) -> None:
    module = _module_name(path)
    layer = _layer_of(module)
    if layer is None:
        return  # container.py, cli/, dashboard/ and the package root are unrestricted
    allowed = ALLOWED_IMPORTS[layer]
    violations = [
        imported
        for imported in _imported_modules(_parse(path))
        if (prefix := _quantlab_prefix(imported)) is not None and prefix not in allowed
    ]
    assert not violations, (
        f"INV-8: {module} (layer {layer}) may import {sorted(allowed)} "
        f"but imports {sorted(set(violations))}"
    )


def test_inv8_only_container_and_cli_import_adapters() -> None:
    offenders: list[str] = []
    for path in _python_files():
        module = _module_name(path)
        if module.startswith(ADAPTER_IMPORTERS):
            continue
        for imported in _imported_modules(_parse(path)):
            if imported == "quantlab.adapters" or imported.startswith("quantlab.adapters."):
                offenders.append(f"{module} -> {imported}")
    assert not offenders, (
        "INV-8: only container.py, cli/ and tests/ may import adapters: " + "; ".join(offenders)
    )


def test_inv8_core_imports_only_core() -> None:
    """The rule that keeps the domain layer portable, stated as its own test."""
    for path in _python_files():
        module = _module_name(path)
        if not module.startswith("quantlab.core"):
            continue
        for imported in _imported_modules(_parse(path)):
            prefix = _quantlab_prefix(imported)
            assert prefix in (None, "quantlab.core"), f"{module} imports {imported}"
