"""Sandboxing for untrusted strategy code (INV-4, master spec section 21.3).

This is the only package permitted to use ``exec``, ``eval``, ``compile`` or
``importlib``: everywhere else those are forbidden and
``tests/unit/test_architecture.py`` enforces it. Untrusted code is loaded here,
in a child process, behind an import hook — never in the process that decides
what to trust.
"""

from __future__ import annotations

from quantlab.sandbox.ast_check import (
    DEFAULT_ALLOWED_IMPORTS,
    DEFAULT_MAX_LOGIC_LINES,
    DEFAULT_MAX_PARAMS,
    MAX_SOURCE_LINES,
    AstReport,
    Violation,
    check_source,
    count_logic_lines,
    require_safe_source,
)

__all__ = [
    "DEFAULT_ALLOWED_IMPORTS",
    "DEFAULT_MAX_LOGIC_LINES",
    "DEFAULT_MAX_PARAMS",
    "MAX_SOURCE_LINES",
    "AstReport",
    "Violation",
    "check_source",
    "count_logic_lines",
    "require_safe_source",
]
