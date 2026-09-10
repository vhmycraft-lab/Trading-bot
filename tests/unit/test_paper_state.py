"""A restarted paper session is the same session (spec section 16.2).

Section 16.2 asks for something stronger than "the process comes back up": on
restart the runtime backfills the missed bars and **replays them, fills
included**, so that the account is what an uninterrupted run would have had. The
tempting shortcut — restore the position and skip straight to live bars — is
faster, looks correct, and silently produces a different account, because the
fills that would have happened in the gap simply never do.

That is the same guarantee the evolution resume test makes about candidate
identity, and it is tested the same way: run one session straight through, run
another with an interruption in the middle, and require the two to be
indistinguishable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.helpers import ScriptedStrategy, flat_frame

from quantlab.core.types import BacktestConfig, Bar, SlippageConfig
from quantlab.paper.broker import PaperBroker
from quantlab.paper.runtime import (
    STATE_VERSION,
    PaperSession,
    PaperState,
    load_state,
    save_state,
)

COSTS = BacktestConfig(
    initial_equity=10_000.0,
    fee_bps=10.0,
    slippage=SlippageConfig(model="fixed_bps", fixed_bps=5.0),
)

PRICES = [100.0, 104.0, 99.0, 103.0, 110.0, 104.0, 108.0, 115.0, 109.0, 112.0]
SCRIPT = ["long", "long", "flat", "long", "long", "flat", "long", "long", "flat", "flat"]


def _bars() -> list[Bar]:
    frame = flat_frame(PRICES)
    return [
        Bar(
            ts_open=int(frame.ts_open[i]),
            open=float(frame.open[i]),
            high=float(frame.high[i]),
            low=float(frame.low[i]),
            close=float(frame.close[i]),
            volume=float(frame.volume[i]),
        )
        for i in range(len(frame.close))
    ]


def _session(path: Path | None = None) -> PaperSession:
    return PaperSession(
        strategy=ScriptedStrategy(SCRIPT),
        broker=PaperBroker(config=COSTS, cash=COSTS.initial_equity),
        state_path=path,
    )


# ---------------------------------------------------------------------------
# the guarantee
# ---------------------------------------------------------------------------
def test_an_interrupted_session_matches_one_that_ran_straight_through(tmp_path: Path) -> None:
    """The whole point of persisting anything."""
    bars = _bars()

    whole = _session()
    whole.run(bars)

    state_path = tmp_path / "session.json"
    first = _session(state_path)
    first.run(bars[:5])

    resumed = _session(state_path)
    state = load_state(state_path)
    assert state is not None
    resumed.restore(state)
    resumed.run(bars[5:], start_index=5)

    assert resumed.broker.position_qty == whole.broker.position_qty
    assert resumed.broker.cash == whole.broker.cash
    assert resumed.broker.total_fees == whole.broker.total_fees


def test_the_fixture_would_notice_a_divergence(tmp_path: Path) -> None:
    """Guards the test above.

    If the script never traded after the interruption point, the two halves
    would agree for a reason that has nothing to do with resume. This asserts
    the second half is where most of the trading happens.
    """
    whole = _session()
    whole.run(_bars())
    late = [f for f in whole.broker.fills if f.bar_index >= 5]
    assert len(late) >= 2, "the interruption point must have trading on both sides of it"


def test_resuming_restores_the_pending_order(tmp_path: Path) -> None:
    """The order decided at the last bar before the crash has not filled yet.

    Dropping it loses a trade the strategy asked for; the next bar's open was
    always going to be its fill, and a resumed session that skipped it would be
    flat where the uninterrupted one is long.
    """
    bars = _bars()
    path = tmp_path / "s.json"
    first = _session(path)
    first.run(bars[:1])

    state = load_state(path)
    assert state is not None
    assert state.pending_target_notional is not None

    resumed = _session(path)
    resumed.restore(state)
    resumed.run(bars[1:2], start_index=1)
    assert resumed.broker.fills, "the pending order was dropped on resume"


# ---------------------------------------------------------------------------
# the file
# ---------------------------------------------------------------------------
def test_state_is_written_atomically(tmp_path: Path) -> None:
    """Temp file plus rename. A partial write here is not a lost update — it is
    a session that resumes into a position it does not hold."""
    path = tmp_path / "nested" / "s.json"
    save_state(PaperState(cash=123.0), path)
    assert json.loads(path.read_text())["cash"] == 123.0
    assert list(path.parent.glob("*.tmp")) == []


def test_a_state_file_from_another_format_version_is_refused(tmp_path: Path) -> None:
    """Refusing beats guessing: the alternative is a resumed session whose
    position is a misreading of somebody else's schema."""
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"version": STATE_VERSION + 1, "cash": 1.0}))
    with pytest.raises(ValueError, match="refusing to resume"):
        load_state(path)


def test_an_unknown_field_in_the_state_file_is_ignored(tmp_path: Path) -> None:
    """Forward compatibility in the one direction that is safe: a field this
    build does not know about cannot change what it does, and refusing on it
    would make every state file a version pin."""
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"version": STATE_VERSION, "cash": 5.0, "future_field": 1}))
    state = load_state(path)
    assert state is not None
    assert state.cash == 5.0


def test_no_state_file_is_not_an_error(tmp_path: Path) -> None:
    """A session that has never run has nothing to resume from, and that is the
    ordinary first-start case rather than a fault."""
    assert load_state(tmp_path / "absent.json") is None


# ---------------------------------------------------------------------------
# the drawdown monitor
# ---------------------------------------------------------------------------
def test_a_breach_of_the_validated_drawdown_is_recorded() -> None:
    """Section 16.2's divergence monitor. Best-effort and never fatal — a paper
    session that killed itself over an undeliverable notification would be less
    useful than one that kept trading and said so."""
    session = PaperSession(
        strategy=ScriptedStrategy(["long"] * 10),
        broker=PaperBroker(config=COSTS, cash=COSTS.initial_equity),
        max_drawdown_alert=0.02,
    )
    session.run(_bars())
    assert session.alerts, "a 10 % swing should have breached a 2 % threshold"


def test_no_threshold_means_no_alerts() -> None:
    """A strategy with no verdict artifacts has no ``mdd_p95`` to draw a
    threshold from, and a fabricated one would alert on nothing or everything."""
    session = PaperSession(
        strategy=ScriptedStrategy(["long"] * 10),
        broker=PaperBroker(config=COSTS, cash=COSTS.initial_equity),
        max_drawdown_alert=None,
    )
    session.run(_bars())
    assert session.alerts == []


def test_a_session_within_its_validated_drawdown_is_quiet() -> None:
    """Guards the alert test: a monitor that fired on every session would be
    ignored within a day."""
    session = PaperSession(
        strategy=ScriptedStrategy(["flat"] * 10),
        broker=PaperBroker(config=COSTS, cash=COSTS.initial_equity),
        max_drawdown_alert=0.02,
    )
    session.run(_bars())
    assert session.alerts == []


# ---------------------------------------------------------------------------
# warm-up
# ---------------------------------------------------------------------------
def test_warming_up_places_no_orders() -> None:
    """Backfill replay rebuilds indicator state. An order queued during warm-up
    would be filled by the first live bar — a position opened by the backfill
    rather than by the strategy."""
    session = _session()
    session.warm_up(_bars()[:6])
    assert session.broker.fills == []
    assert session.broker.cash == COSTS.initial_equity


def test_warm_up_leaves_the_broker_live_afterwards() -> None:
    """Guards the test above: a broker left warm would never trade again, and
    every paper session would report a flat, uneventful, wrong result."""
    session = _session()
    session.warm_up(_bars()[:3])
    assert not session.broker.warm
    session.run(_bars()[3:], start_index=3)
    assert session.broker.fills
