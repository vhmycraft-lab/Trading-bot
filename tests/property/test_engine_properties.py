"""Engine invariants over generated inputs (master spec section 19).

The unit tests check specific execution cases; these check the properties that
must hold for *every* run. An accounting identity that holds on six hand-built
bars and fails on a thousand random ones is not an identity.
"""

from __future__ import annotations

import itertools

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from tests.helpers import ScriptedStrategy, make_bars, zero_cost_config

from quantlab.adapters.engine import SimpleBarEngine
from quantlab.core.metrics import compute_metrics, retention_after_removing_top_winners
from quantlab.core.types import BarFrame, RiskSpec, Side, SlippageConfig

SETTINGS = settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

SIGNALS = st.sampled_from(["long", "flat"])


def build(n: int, seed: int, drift: float = 0.0) -> BarFrame:
    return BarFrame(make_bars(n, seed=seed, drift=drift), symbol="BTC/USDT", timeframe="1h")


# ---------------------------------------------------------------------------
# accounting
# ---------------------------------------------------------------------------
@SETTINGS
@given(
    n=st.integers(min_value=3, max_value=60),
    seed=st.integers(min_value=0, max_value=40),
    fee=st.floats(min_value=0.0, max_value=50.0),
    slip=st.floats(min_value=0.0, max_value=50.0),
    script_seed=st.integers(min_value=0, max_value=1000),
)
def test_equity_change_equals_the_sum_of_trade_pnl(
    n: int, seed: int, fee: float, slip: float, script_seed: int
) -> None:
    """Nothing is created or destroyed: every unit of equity came from a trade."""
    rng = np.random.default_rng(script_seed)
    script = ["long" if rng.random() < 0.5 else "flat" for _ in range(n)]
    bars = build(n, seed)
    config = zero_cost_config(fee_bps=fee, slippage=SlippageConfig(fixed_bps=slip))

    result = SimpleBarEngine().run(ScriptedStrategy(script), bars, {}, config)
    change = float(result.equity.iloc[-1] - result.equity.iloc[0])
    assert change == np.float64(change)  # not NaN
    assert abs(change - sum(t.pnl for t in result.trades)) < 1e-6 * max(1.0, abs(change))


@SETTINGS
@given(
    n=st.integers(min_value=3, max_value=60),
    seed=st.integers(min_value=0, max_value=40),
    fee=st.floats(min_value=0.0, max_value=50.0),
)
def test_equity_is_always_finite_and_positive(n: int, seed: int, fee: float) -> None:
    bars = build(n, seed)
    result = SimpleBarEngine().run(
        ScriptedStrategy(["long", "flat"] * n), bars, {}, zero_cost_config(fee_bps=fee)
    )
    equity = result.equity.to_numpy()
    assert np.all(np.isfinite(equity))
    assert np.all(equity > 0.0)


@SETTINGS
@given(
    n=st.integers(min_value=3, max_value=60),
    seed=st.integers(min_value=0, max_value=40),
    cap=st.floats(min_value=0.05, max_value=1.0),
)
def test_the_cap_is_enforced_at_the_moment_of_sizing(n: int, seed: int, cap: float) -> None:
    """``max_position_fraction`` bounds the *target*, which is what the engine sets.

    Sizing happens at the close of the deciding bar; the fill is at the *next*
    open and the fraction is marked at that bar's close, so one bar of market
    movement sits between the target and the realised fraction. Correcting that
    on every bar would cost more than the drift does, so the engine rebalances
    only once the drift passes 1 %.

    The exact property is therefore about committed capital at the fill, not
    about the marked fraction afterwards.
    """
    bars = build(n, seed)
    result = SimpleBarEngine().run(
        ScriptedStrategy(["long"] * n),
        bars,
        {},
        zero_cost_config(fee_bps=10.0, max_position_fraction=cap),
    )
    equity = result.equity.to_numpy()
    for fill in result.fills:
        if fill.side is not Side.LONG or fill.bar_index == 0:
            continue  # entries only; an exit is not sized against the cap
        committed = fill.qty * fill.fill_price + fill.fee
        allowed = cap * equity[fill.bar_index - 1]
        assert committed <= allowed + 1e-6, "the committed capital respects the cap"

    fractions = result.position_frac.to_numpy()
    assert float(np.abs(fractions).max()) <= cap * 1.5 + 1e-9, "drift stays bounded"


@SETTINGS
@given(
    n=st.integers(min_value=3, max_value=50),
    seed=st.integers(min_value=0, max_value=40),
)
def test_costs_are_never_negative(n: int, seed: int) -> None:
    """A negative cost is free money, and a search would find it every time."""
    result = SimpleBarEngine().run(
        ScriptedStrategy(["long", "flat"] * n),
        build(n, seed),
        {},
        zero_cost_config(fee_bps=10.0, slippage=SlippageConfig(fixed_bps=5.0)),
    )
    for fill in result.fills:
        assert fill.fee >= 0.0
        assert fill.slippage_cost >= 0.0
        assert fill.qty > 0.0
        assert fill.fill_price > 0.0
    for value in result.cost_summary.values():
        assert value >= 0.0


@SETTINGS
@given(
    n=st.integers(min_value=3, max_value=50),
    seed=st.integers(min_value=0, max_value=40),
)
def test_slippage_always_moves_the_price_against_the_trader(n: int, seed: int) -> None:
    result = SimpleBarEngine().run(
        ScriptedStrategy(["long", "flat"] * n),
        build(n, seed),
        {},
        zero_cost_config(slippage=SlippageConfig(fixed_bps=25.0)),
    )
    for fill in result.fills:
        if fill.side is Side.LONG:
            assert fill.fill_price >= fill.ref_price - 1e-12
        else:
            assert fill.fill_price <= fill.ref_price + 1e-12


@SETTINGS
@given(
    n=st.integers(min_value=4, max_value=60),
    seed=st.integers(min_value=0, max_value=40),
)
def test_a_run_always_ends_flat(n: int, seed: int) -> None:
    result = SimpleBarEngine().run(
        ScriptedStrategy(["long"] * n), build(n, seed), {}, zero_cost_config()
    )
    assert float(result.position_frac.iloc[-1]) == 0.0


@SETTINGS
@given(
    n=st.integers(min_value=4, max_value=60),
    seed=st.integers(min_value=0, max_value=40),
)
def test_trades_are_numbered_and_ordered(n: int, seed: int) -> None:
    result = SimpleBarEngine().run(
        ScriptedStrategy(["long", "long", "flat"] * n), build(n, seed), {}, zero_cost_config()
    )
    for k, trade in enumerate(result.trades, start=1):
        assert trade.trade_no == k
        assert trade.entry_ts <= trade.exit_ts
        assert trade.bars_held >= 0
        assert trade.qty > 0


# ---------------------------------------------------------------------------
# no look-ahead, over generated series
# ---------------------------------------------------------------------------
@SETTINGS
@given(
    n=st.integers(min_value=10, max_value=40),
    extra=st.integers(min_value=1, max_value=30),
    seed=st.integers(min_value=0, max_value=40),
)
def test_extending_the_series_never_changes_the_prefix(n: int, extra: int, seed: int) -> None:
    """INV-3 for the engine, over generated inputs."""
    script = ["long", "long", "flat"] * (n + extra)
    config = zero_cost_config(
        fee_bps=10.0,
        slippage=SlippageConfig(fixed_bps=5.0),
        risk=RiskSpec(stop_loss_pct=0.05, take_profit_pct=0.10),
    )
    engine = SimpleBarEngine()
    short = engine.run(ScriptedStrategy(script), build(n, seed), {}, config)
    long = engine.run(ScriptedStrategy(script), build(n + extra, seed), {}, config)

    assert list(short.signals)[: n - 1] == list(long.signals)[: n - 1]
    assert [f for f in short.fills if f.bar_index < n - 1] == [
        f for f in long.fills if f.bar_index < n - 1
    ]


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------
@SETTINGS
@given(
    n=st.integers(min_value=3, max_value=50),
    seed=st.integers(min_value=0, max_value=40),
    run_seed=st.integers(min_value=0, max_value=10_000),
)
def test_a_run_is_reproducible(n: int, seed: int, run_seed: int) -> None:
    bars = build(n, seed)
    config = zero_cost_config(seed=run_seed, fee_bps=10.0, slippage=SlippageConfig(fixed_bps=5.0))
    script = ["long", "flat", "long"] * n
    a = SimpleBarEngine().run(ScriptedStrategy(script), bars, {}, config)
    b = SimpleBarEngine().run(ScriptedStrategy(script), bars, {}, config)
    assert list(a.equity) == list(b.equity)
    assert a.fills == b.fills
    assert a.trades == b.trades
    assert a.cost_summary == b.cost_summary


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
@SETTINGS
@given(
    n=st.integers(min_value=5, max_value=60),
    seed=st.integers(min_value=0, max_value=40),
)
def test_metric_ranges_hold(n: int, seed: int) -> None:
    result = SimpleBarEngine().run(
        ScriptedStrategy(["long", "flat"] * n),
        build(n, seed),
        {},
        zero_cost_config(fee_bps=10.0),
    )
    metrics = compute_metrics(result)
    assert 0.0 <= metrics.max_drawdown <= 1.0
    assert metrics.exposure is None or 0.0 <= metrics.exposure <= 1.0
    if metrics.win_rate is not None:
        assert 0.0 <= metrics.win_rate <= 1.0
    if metrics.top5_profit_share is not None:
        assert 0.0 <= metrics.top5_profit_share <= 1.0
    assert metrics.n_trades == len(result.trades)


@SETTINGS
@given(
    pnls=st.lists(
        st.floats(min_value=-500.0, max_value=500.0, allow_nan=False),
        min_size=1,
        max_size=30,
    )
)
def test_retention_is_non_increasing_in_k(pnls: list[float]) -> None:
    from tests.unit.test_metrics import make_trade

    trades = [make_trade(i, pnl) for i, pnl in enumerate(pnls, start=1)]
    retention = retention_after_removing_top_winners(trades, (1, 2, 3, 5, 10))
    values = [retention[k] for k in (1, 2, 3, 5, 10)]
    if values[0] is None:
        assert all(value is None for value in values)
        return
    for earlier, later in itertools.pairwise(values):
        assert later <= earlier + 1e-12


@SETTINGS
@given(
    equity=st.lists(
        st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=60,
    )
)
def test_drawdown_is_always_a_fraction(equity: list[float]) -> None:
    from quantlab.core.metrics import drawdown_series

    values = drawdown_series(np.array(equity, dtype="float64"))
    assert np.all(values >= -1e-12)
    assert np.all(values <= 1.0 + 1e-12)
