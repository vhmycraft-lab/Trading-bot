"""Exit-code mapping shared by the research CLI and the lockbox CLI.

It lives in its own module for one reason: §14.6 forbids the research CLI from
importing ``cli/lockbox.py``, and the lockbox CLI should not have to import the
research CLI to get error handling. A shared leaf both depend on keeps the two
entry points genuinely independent of each other.
"""

from __future__ import annotations

import logging
import traceback
import uuid
from typing import Any

from rich.console import Console
from typer.core import TyperGroup

from quantlab.core.errors import QuantLabError, exit_code_for

__all__ = ["QuantLabGroup"]

err_console = Console(stderr=True)


class QuantLabGroup(TyperGroup):
    """Root command group that maps QuantLab errors onto the documented exit codes.

    The translation lives here rather than in a ``Typer.__call__`` override so
    that it applies identically to the installed console script, to
    ``python -m``, and to ``typer.testing.CliRunner`` in the test suite.
    """

    # `ctx` is a click Context, but Typer vendors its own click build, so the
    # precise type lives in a private module; Any keeps the override compatible.
    def invoke(self, ctx: Any) -> Any:
        try:
            return super().invoke(ctx)
        except QuantLabError as exc:
            error_id = uuid.uuid4().hex[:12]
            code = exit_code_for(exc)
            err_console.print(f"[red]error[/] {exc}  [dim](error_id={error_id})[/]")
            logging.getLogger("quantlab.cli").debug(
                "unhandled QuantLabError %s\n%s", error_id, traceback.format_exc()
            )
            raise SystemExit(code) from exc
