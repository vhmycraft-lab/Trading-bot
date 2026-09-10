"""``quantlab paper`` — running a validated strategy forward (spec section 16.3).

Paper, and only paper. There is no ``--live`` flag, no broker selection, and no
credential this command reads: the session it starts holds a
:class:`~quantlab.paper.broker.PaperBroker`, which is the only implementation of
the broker port there is (INV-1).

Two things this module does that are easy to leave out and hard to add later:

* **It refuses to start from a strategy that was never validated.** A paper
  session is where a person starts believing a number, and starting one from an
  unvalidated strategy makes the belief older than the evidence.
* **It records the session before the first bar.** Same shape as the lockbox's
  access record and for the same reason: a session that registered itself on
  completion would be unregistered for exactly as long as it was running, which
  is the whole time anybody might want to know it exists.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from quantlab.core.errors import ConfigError, QuantLabError

__all__ = ["app", "notify", "startup_guard"]

app = typer.Typer(help="Run a validated strategy forward on live data (paper only).")
console = Console()

#: Verdicts a strategy may hold and still be worth running forward. A `WEAK`
#: strategy is deliberately included — watching one fail in public costs
#: nothing and is often more informative than the verdict was — but a `REJECT`
#: or a closed family is not, because those are decisions and running one
#: forward is a way of relitigating them.
RUNNABLE_VERDICTS: tuple[str, ...] = ("CANDIDATE", "WEAK", "LOCKBOX_PASS")


def startup_guard(exchange: Any) -> None:
    """Abort if the configured key could place a real order (section 16.2).

    The guard is not "are we in paper mode?" — the code cannot leave paper mode.
    It is narrower and more useful: if a key happens to be configured on this
    machine, and that key has spot trading enabled, then a mistake anywhere in
    the stack has somewhere to land. Refusing to start removes the landing site.

    A missing key is not an error. The public stream needs none, and demanding
    one would push people towards creating a key they do not need.
    """
    if exchange is None:
        return
    try:
        restrictions = exchange.sapi_get_account_apirestrictions()
    except Exception as exc:  # a key that cannot be queried is not a safe key
        raise ConfigError(
            "an exchange key is configured but its permissions could not be read; "
            f"refusing to start a session next to a key of unknown scope ({exc})"
        ) from exc

    if restrictions.get("enableSpotAndMarginTrading"):
        raise ConfigError(
            "the configured exchange key has spot and margin trading enabled. "
            "This platform never places an order, but a trading-enabled key on the "
            "same machine is a hazard with no upside here — remove the permission, "
            "or remove the key, and start again.",
        )


def notify(message: str) -> None:
    """Best-effort macOS notification (section 16.2).

    Never fatal, and deliberately so: a paper session that killed itself because
    a notification could not be delivered would be less useful than one that
    kept trading and wrote the alert to the log. Everything here is guarded,
    including whether ``osascript`` exists at all.
    """
    binary = shutil.which("osascript")
    if binary is None:
        return
    body = message.replace('"', "'")[:200]
    try:
        subprocess.run(  # noqa: S603 - fixed binary, no shell, quoted payload
            [binary, "-e", f'display notification "{body}" with title "QuantLab paper"'],
            check=False,
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return


@app.command("status")
def status(ctx: typer.Context) -> None:
    """List every paper session this store knows about."""
    state = ctx.obj
    store = state.container.store
    sessions = list(store.paper_sessions())

    table = Table(title="paper sessions")
    for column in ("session", "strategy", "status", "last bar"):
        table.add_column(column)
    for session in sessions:
        table.add_row(
            str(getattr(session, "session_id", "")),
            str(getattr(session, "strategy_id", "")),
            str(getattr(session, "status", "")),
            str(getattr(session, "last_bar_ts", "") or "-"),
        )
    console.print(table)
    if not sessions:
        console.print("[dim]no sessions recorded[/dim]")


@app.command("report")
def report(
    ctx: typer.Context,
    session_id: Annotated[str, typer.Argument(help="Session to report on.")],
) -> None:
    """Print what a session has done so far, from its persisted state."""
    state = ctx.obj
    state_dir = Path(state.config.paper.state_dir)
    path = state_dir / f"{session_id}.json"
    if not path.is_file():
        raise QuantLabError(
            f"no state file for session {session_id!r} at {path}; "
            "`quantlab paper status` lists the sessions this store knows about"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    console.print_json(json.dumps(payload, indent=2, sort_keys=True))
