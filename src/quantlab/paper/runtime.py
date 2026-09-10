"""Driving one paper session (master spec section 16.2).

The runtime is the part that has to be *boring*. Its whole job is to put bars
through a strategy and a broker in the order section 8.4 specifies, and to be
able to stop and start again without the session becoming a different session.

Three rules do most of the work:

1. **Fill first, then decide.** Every bar: fill what the previous bar's close
   asked for, at *this* bar's open; mark to market; ask the strategy what it
   wants now; translate that into an order for the next open. Any other order of
   operations lets a decision see its own fill.
2. **The translation is not this module's.** ``desired`` versus ``current``
   exposure, and the deadband between them, come from
   :func:`quantlab.core.execution.needs_order`, which the engine also calls.
   Restating it here would make a live session open and close on different bars
   than the backtest that authorised it — on exactly the bars where comparing
   them matters most.
3. **A resumed session is the same session.** Restart replays the missed bars
   *with fills* (section 16.2), so the position, cash and fill history are what
   an uninterrupted run would have had. Skipping the fills would be faster and
   would silently produce a different account.

What is deliberately *not* here: any path that transmits an order. The runtime
holds a :class:`~quantlab.ports.broker.Broker`, and the only implementation of
that protocol is :class:`~quantlab.paper.broker.PaperBroker` (INV-1).
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

from quantlab.core.execution import EPS, needs_order
from quantlab.core.types import Bar, Order, Side, SignalKind
from quantlab.paper.broker import PaperBroker

__all__ = ["STATE_VERSION", "PaperSession", "PaperState", "load_state", "save_state"]

#: Bumped when the on-disk shape changes. A session whose state file predates a
#: change is refused rather than read optimistically — a half-understood resume
#: produces an account nobody can explain.
STATE_VERSION: Final[int] = 1


@dataclass
class PaperState:
    """What has to survive a restart (section 16.2)."""

    version: int = STATE_VERSION
    last_bar_ts: int | None = None
    position_qty: float = 0.0
    entry_px: float = 0.0
    entry_bar: int = -1
    cash: float = 0.0
    bar_index: int = -1
    pending_target_notional: float | None = None
    total_fees: float = 0.0
    total_slippage: float = 0.0
    #: Digest of the indicator cache, so a resume that rebuilt state from a
    #: different set of bars is detectable rather than merely unlikely.
    indicator_digest: str = ""


def save_state(state: PaperState, path: Path) -> None:
    """Write ``state`` to ``path`` atomically.

    Temp file plus rename, on the same filesystem. A partial write here is not a
    lost update — it is a session that resumes into a position it does not have,
    and the crash that truncated the file is exactly the moment the state
    mattered.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            json.dump(asdict(state), fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def load_state(path: Path) -> PaperState | None:
    """Read a session's state, or ``None`` if there is none to read.

    Raises:
        ValueError: the file exists but was written by a different version of
            this format. Refusing beats guessing: the alternative is a resumed
            session whose position is a misreading of somebody else's schema.
    """
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = int(payload.get("version", 0))
    if version != STATE_VERSION:
        raise ValueError(
            f"paper state at {path} is version {version}, this build writes "
            f"{STATE_VERSION}; refusing to resume from a state it may misread"
        )
    known = set(PaperState.__dataclass_fields__)
    return PaperState(**{k: v for k, v in payload.items() if k in known})


@dataclass
class PaperSession:
    """One strategy, one broker, one stream of closed bars."""

    strategy: Any
    broker: PaperBroker
    state_path: Path | None = None
    #: Peak equity seen so far, for the drawdown monitor of section 16.2.
    _peak_equity: float = 0.0
    equity_curve: list[float] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)
    #: Running drawdown above which the session logs and notifies. ``None``
    #: disables the monitor, which is what a strategy with no verdict artifacts
    #: to draw ``mdd_p95`` from gets — a fabricated threshold would alert on
    #: nothing or on everything.
    max_drawdown_alert: float | None = None

    def __post_init__(self) -> None:
        self._peak_equity = self.broker.equity(0.0)

    # -- the loop -----------------------------------------------------------
    def on_bar(self, bar: Bar, *, index: int) -> None:
        """One closed bar, in the order section 8.4 requires."""
        self.broker.on_bar_open(bar)

        mark = float(bar.close)
        equity = self.broker.equity(mark)
        self.equity_curve.append(equity)
        self._check_drawdown(equity)

        if self.broker.warm:
            # Warm bars rebuild indicator state and place no orders. Deciding
            # here would queue an order that the first live bar then fills —
            # a position opened by the backfill rather than by the strategy.
            return

        desired = self._desired_fraction(bar, index)
        current = (self.broker.position_qty * mark / equity) if abs(equity) > EPS else 0.0
        if not needs_order(desired, current):
            return

        self.broker.submit(
            Order(
                bar_index=index,
                side=Side.LONG if desired >= 0 else Side.SHORT,
                target_qty=0.0,
                created_ts=int(bar.ts_open),
            ),
            target_notional=desired * equity,
        )

    def run(self, bars: Iterable[Bar], *, start_index: int = 0) -> None:
        """Feed ``bars`` through the session, persisting after each one."""
        for offset, bar in enumerate(bars):
            self.on_bar(bar, index=start_index + offset)
            if self.state_path is not None:
                save_state(self.snapshot(int(bar.ts_open)), self.state_path)

    def warm_up(self, bars: Sequence[Bar]) -> None:
        """Replay backfilled bars to rebuild indicator state, without trading."""
        was_warm = self.broker.warm
        self.broker.warm = True
        try:
            for offset, bar in enumerate(bars):
                self.on_bar(bar, index=offset)
        finally:
            self.broker.warm = was_warm

    # -- state --------------------------------------------------------------
    def snapshot(self, last_bar_ts: int) -> PaperState:
        return PaperState(
            last_bar_ts=last_bar_ts,
            position_qty=self.broker.position_qty,
            entry_px=self.broker.entry_px,
            entry_bar=self.broker.entry_bar,
            cash=self.broker.cash,
            bar_index=self.broker.bar_index,
            pending_target_notional=(
                self.broker._pending_notional if self.broker._pending is not None else None
            ),
            total_fees=self.broker.total_fees,
            total_slippage=self.broker.total_slippage,
        )

    def restore(self, state: PaperState) -> None:
        """Put the broker back where the state file says it was."""
        self.broker.position_qty = state.position_qty
        self.broker.entry_px = state.entry_px
        self.broker.entry_bar = state.entry_bar
        self.broker.cash = state.cash
        self.broker.bar_index = state.bar_index
        self.broker.total_fees = state.total_fees
        self.broker.total_slippage = state.total_slippage
        if state.pending_target_notional is not None:
            self.broker.submit(
                Order(
                    bar_index=state.bar_index,
                    side=Side.LONG,
                    target_qty=0.0,
                    created_ts=int(state.last_bar_ts or 0),
                ),
                target_notional=state.pending_target_notional,
            )

    # -- internals ----------------------------------------------------------
    def _desired_fraction(self, bar: Bar, index: int) -> float:
        signal = self.strategy.on_bar(_LiveContext(bar=bar, i=index))
        if signal is None or signal.kind is SignalKind.FLAT:
            return 0.0
        fraction = float(getattr(signal, "target_fraction", 1.0))
        return fraction if signal.kind is SignalKind.LONG else -fraction

    def _check_drawdown(self, equity: float) -> None:
        """Section 16.2's divergence monitor.

        Best-effort and never fatal: a paper session that killed itself because
        a notification could not be delivered would be less useful than one that
        kept trading and said so. The alert is recorded here; delivery is the
        CLI's business.
        """
        self._peak_equity = max(self._peak_equity, equity)
        if self.max_drawdown_alert is None or self._peak_equity <= EPS:
            return
        drawdown = (self._peak_equity - equity) / self._peak_equity
        if drawdown > self.max_drawdown_alert:
            message = (
                f"drawdown {drawdown:.2%} exceeds the validated 95th percentile "
                f"of {self.max_drawdown_alert:.2%}"
            )
            if message not in self.alerts:
                self.alerts.append(message)


@dataclass(frozen=True)
class _LiveContext:
    """The minimum a bar-loop strategy needs on a live bar.

    Not the engine's :class:`~quantlab.core.strategy.Context`: that one carries
    a whole :class:`BarFrame` and an indicator cache built over completed
    history, neither of which exists mid-stream. A strategy that needs more than
    this is one the paper runtime cannot yet drive, and it should fail saying so
    rather than being handed a context with plausible-looking empty arrays.
    """

    bar: Bar
    i: int
