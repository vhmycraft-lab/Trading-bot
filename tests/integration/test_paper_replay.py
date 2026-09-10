"""A replayed stream must trade exactly as the backtest did (spec section 16.2).

This is the acceptance criterion for the whole paper layer, and it is worth
being precise about why it is stated as *equality* rather than as a tolerance.

A paper session exists to check a strategy against reality before any money is
involved. The comparison only means something if a disagreement between the
paper session and the backtest is evidence about the *market* — a fill that did
not happen, a bar that arrived late — and not about the platform. Every
basis point of drift between the two implementations is noise in that signal,
permanently, and it is noise nobody can subtract later because there is no
record of which half caused it.

So the two share their arithmetic (``core/execution.py``) and these tests assert
that the sharing actually holds. They are written to fail loudly if anybody
reintroduces a second implementation: not "close enough", but the same floats.
"""

from __future__ import annotations

import pytest
from tests.helpers import ScriptedStrategy, flat_frame

from quantlab.adapters.engine.simple_bar import SimpleBarEngine
from quantlab.core.types import (
    BacktestConfig,
    Bar,
    Order,
    Side,
    SlippageConfig,
)
from quantlab.paper.broker import PaperBroker
from quantlab.paper.runtime import PaperSession
from quantlab.ports.broker import Broker

#: The fixed model, because it is the one where a live session and a backtest are
#: identical by construction: a volatility- or volume-scaled rate needs the bar
#: after the fill to compute itself, which a live session does not have. The
#: limitation is documented on ``PaperBroker._price`` rather than papered over,
#: and pinning it here is what keeps this test measuring the platform rather
#: than measuring that known gap.
COSTS = BacktestConfig(
    initial_equity=10_000.0,
    fee_bps=10.0,
    slippage=SlippageConfig(model="fixed_bps", fixed_bps=5.0),
    lot_step=0.00001,
    min_notional=5.0,
)

PRICES = [100.0, 101.0, 99.0, 103.0, 107.0, 104.0, 108.0, 111.0, 109.0, 112.0]
SCRIPT = ["flat", "long", "long", "flat", "long", "long", "long", "flat", "flat", "flat"]


def _bars_of(frame: object) -> list[Bar]:
    """The frame, as the stream of closed bars a feed would yield."""
    return [
        Bar(
            ts_open=int(frame.ts_open[i]),  # type: ignore[attr-defined]
            open=float(frame.open[i]),  # type: ignore[attr-defined]
            high=float(frame.high[i]),  # type: ignore[attr-defined]
            low=float(frame.low[i]),  # type: ignore[attr-defined]
            close=float(frame.close[i]),  # type: ignore[attr-defined]
            volume=float(frame.volume[i]),  # type: ignore[attr-defined]
        )
        for i in range(len(frame.close))  # type: ignore[attr-defined]
    ]


def _replay(frame: object, script: list[str], config: BacktestConfig) -> PaperBroker:
    """Drive a session through :class:`PaperSession`, the way a live one runs.

    Through the real runtime rather than a hand-rolled loop, and that choice is
    the test. An earlier version of this file drove the broker directly and
    re-derived "does the strategy want a different position now?" inline — which
    looked equivalent and was not: it rebalanced on every bar, because a held
    position's fraction drifts with the mark price, and it disagreed with the
    engine within three bars. The rule that reconciles them
    (:func:`quantlab.core.execution.needs_order`) is called by the runtime and by
    the engine, and by nothing that restates it.
    """
    session = PaperSession(
        strategy=ScriptedStrategy(script),
        broker=PaperBroker(config=config, cash=config.initial_equity),
    )
    session.run(_bars_of(frame))
    return session.broker


# ---------------------------------------------------------------------------
# the acceptance criterion
# ---------------------------------------------------------------------------
def test_a_replayed_stream_fills_exactly_as_the_engine_did() -> None:
    """Section 16.2: "assert trades equal ``SimpleBarEngine`` trades on the same
    bars (bit-for-bit on qty and prices)"."""
    frame = flat_frame(PRICES)
    engine_result = SimpleBarEngine().run(ScriptedStrategy(SCRIPT), frame, {}, COSTS)
    broker = _replay(frame, SCRIPT, COSTS)

    engine_fills = list(engine_result.fills)
    assert engine_fills, "the fixture produced no fills; the comparison would be vacuous"
    assert len(broker.fills) == len(engine_fills)

    for paper, engine in zip(broker.fills, engine_fills, strict=True):
        assert paper.qty == engine.qty, (paper, engine)
        assert paper.fill_price == engine.fill_price, (paper, engine)
        assert paper.ref_price == engine.ref_price, (paper, engine)
        assert paper.fee == engine.fee, (paper, engine)
        assert paper.side == engine.side, (paper, engine)


def test_the_comparison_is_equality_and_not_a_tolerance() -> None:
    """Guards the test above against being quietly relaxed.

    ``==`` on floats is unusual enough in a test suite to look like a mistake
    and get "fixed" into ``pytest.approx``. It is not a mistake: the two sides
    call the same functions on the same inputs, so anything but exact equality
    means a second implementation has appeared.
    """
    frame = flat_frame(PRICES)
    engine_result = SimpleBarEngine().run(ScriptedStrategy(SCRIPT), frame, {}, COSTS)
    broker = _replay(frame, SCRIPT, COSTS)
    engine_prices = [f.fill_price for f in engine_result.fills]
    paper_prices = [f.fill_price for f in broker.fills]
    assert paper_prices == engine_prices
    assert repr(paper_prices) == repr(engine_prices)


def test_the_fixture_actually_trades_both_ways() -> None:
    """Vacuity guard. A script that never opened a position would make every
    assertion above pass on two empty lists."""
    broker = _replay(flat_frame(PRICES), SCRIPT, COSTS)
    assert {f.side for f in broker.fills} == {Side.LONG, Side.SHORT}


# ---------------------------------------------------------------------------
# the timing rule
# ---------------------------------------------------------------------------
def test_an_order_never_fills_on_the_bar_that_submitted_it() -> None:
    """Section 8.4, and most of what separates an honest paper session from one
    trading on information its backtest never had."""
    broker = PaperBroker(config=COSTS, cash=10_000.0)
    bars = _bars_of(flat_frame(PRICES))
    broker.on_bar_open(bars[0])
    broker.submit(
        Order(bar_index=0, side=Side.LONG, target_qty=0.0, created_ts=0), target_notional=10_000.0
    )
    assert broker.fills == []

    broker.on_bar_open(bars[1])
    assert len(broker.fills) == 1
    assert broker.fills[0].ref_price == pytest.approx(bars[1].open)


def test_a_second_order_on_the_same_bar_replaces_the_first() -> None:
    """A strategy is asked once per bar what it wants to hold, so two pending
    orders are two answers to one question and the older one is stale. Queueing
    them would fill both at the same open and double the position."""
    broker = PaperBroker(config=COSTS, cash=10_000.0)
    bars = _bars_of(flat_frame(PRICES))
    broker.on_bar_open(bars[0])
    order = Order(bar_index=0, side=Side.LONG, target_qty=0.0, created_ts=0)
    broker.submit(order, target_notional=10_000.0)
    broker.submit(order, target_notional=5_000.0)
    broker.on_bar_open(bars[1])
    assert len(broker.fills) == 1
    assert broker.fills[0].qty * broker.fills[0].fill_price < 6_000.0


# ---------------------------------------------------------------------------
# warm mode
# ---------------------------------------------------------------------------
def test_warm_bars_advance_the_clock_without_trading() -> None:
    """Section 16.2's backfill replay rebuilds indicator state. If it filled, a
    restarted session would open a position the original never had — and would
    then be a different session wearing the same id."""
    broker = PaperBroker(config=COSTS, cash=10_000.0, warm=True)
    bars = _bars_of(flat_frame(PRICES))
    for bar in bars[:4]:
        broker.submit(
            Order(bar_index=0, side=Side.LONG, target_qty=0.0, created_ts=0),
            target_notional=10_000.0,
        )
        broker.on_bar_open(bar)
    assert broker.fills == []
    assert broker.position().is_flat
    assert broker.cash == 10_000.0
    assert broker.bar_index == 3, "warm bars must still advance the index"


def test_leaving_warm_mode_resumes_trading() -> None:
    """Guards the test above: a broker stuck warm would silently never trade,
    and every paper session would report a flat, uneventful, wrong result."""
    broker = PaperBroker(config=COSTS, cash=10_000.0, warm=True)
    bars = _bars_of(flat_frame(PRICES))
    broker.on_bar_open(bars[0])
    broker.warm = False
    broker.submit(
        Order(bar_index=0, side=Side.LONG, target_qty=0.0, created_ts=0), target_notional=10_000.0
    )
    broker.on_bar_open(bars[1])
    assert len(broker.fills) == 1


# ---------------------------------------------------------------------------
# refusals the engine also makes
# ---------------------------------------------------------------------------
def test_a_fill_below_the_minimum_notional_is_dropped() -> None:
    broker = PaperBroker(config=COSTS, cash=10_000.0)
    bars = _bars_of(flat_frame(PRICES))
    broker.on_bar_open(bars[0])
    broker.submit(
        Order(bar_index=0, side=Side.LONG, target_qty=0.0, created_ts=0), target_notional=1.0
    )
    broker.on_bar_open(bars[1])
    assert broker.fills == []


def test_a_non_positive_open_is_refused_rather_than_filled_at_zero() -> None:
    """Bad data is not a trading opportunity, and the engine drops the order for
    the same reason — the two must agree even about what they refuse to do."""
    broker = PaperBroker(config=COSTS, cash=10_000.0)
    broker.submit(
        Order(bar_index=0, side=Side.LONG, target_qty=0.0, created_ts=0), target_notional=10_000.0
    )
    broker.on_bar_open(Bar(ts_open=0, open=0.0, high=1.0, low=0.0, close=1.0, volume=1.0))
    assert broker.fills == []


# ---------------------------------------------------------------------------
# INV-1
# ---------------------------------------------------------------------------
def test_the_paper_broker_is_the_only_broker() -> None:
    """INV-1 as a runtime fact to sit alongside the static scan: the protocol is
    satisfied, and by this class."""
    assert isinstance(PaperBroker(config=COSTS, cash=1.0), Broker)


def test_the_broker_has_no_way_to_send_anything() -> None:
    """The invariant is not "no exchange is configured" — it is that there is no
    method to configure one. Enumerated so that adding a transmitting method
    fails here rather than in review."""
    forbidden = {"send", "connect", "authenticate", "client", "session", "url", "api_key", "key"}
    surface = {name for name in dir(PaperBroker) if not name.startswith("_")}
    assert not (surface & forbidden), surface & forbidden
