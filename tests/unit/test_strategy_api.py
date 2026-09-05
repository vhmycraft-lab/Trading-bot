"""The strategy contract (master spec section 9.1)."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError
from tests.helpers import make_bars

from quantlab.core.errors import ConfigError, StrategyLoadError
from quantlab.core.strategy import (
    INDICATORS,
    Context,
    IndicatorCache,
    ParamSpec,
    Strategy,
    resolve_params,
)
from quantlab.core.types import (
    BarFrame,
    Position,
    RiskSpec,
    Side,
    Signal,
    SignalKind,
    SizingSpec,
)


@pytest.fixture
def bars() -> BarFrame:
    return BarFrame(make_bars(80, seed=2), symbol="BTC/USDT", timeframe="1h")


# ---------------------------------------------------------------------------
# ParamSpec
# ---------------------------------------------------------------------------
def test_a_numeric_parameter_needs_bounds() -> None:
    """Unbounded parameters cannot be searched, mutated or reported."""
    with pytest.raises(ValidationError, match="both low and high"):
        ParamSpec(kind="int", default=10)
    with pytest.raises(ValidationError, match="both low and high"):
        ParamSpec(kind="float", default=1.0, low=0.0)


def test_bounds_must_be_ordered() -> None:
    with pytest.raises(ValidationError, match="strictly below"):
        ParamSpec(kind="int", default=10, low=100, high=5)


def test_the_default_must_lie_inside_the_bounds() -> None:
    with pytest.raises(ValidationError, match="within"):
        ParamSpec(kind="int", default=500, low=1, high=100)


def test_a_numeric_parameter_needs_a_numeric_default() -> None:
    with pytest.raises(ValidationError, match="numeric default"):
        ParamSpec(kind="float", default="ten", low=1, high=100)


def test_a_categorical_parameter_needs_choices() -> None:
    with pytest.raises(ValidationError, match="must declare choices"):
        ParamSpec(kind="categorical", default="a")
    with pytest.raises(ValidationError, match="one of choices"):
        ParamSpec(kind="categorical", default="z", choices=("a", "b"))


def test_a_numeric_parameter_must_not_declare_choices() -> None:
    with pytest.raises(ValidationError, match="must not declare choices"):
        ParamSpec(kind="int", default=5, low=1, high=10, choices=("a",))


def test_a_bool_parameter_needs_a_bool_default() -> None:
    with pytest.raises(ValidationError, match="boolean default"):
        ParamSpec(kind="bool", default=1)
    assert ParamSpec(kind="bool", default=True).default is True


def test_a_log_scaled_parameter_needs_a_positive_lower_bound() -> None:
    with pytest.raises(ValidationError, match="positive lower bound"):
        ParamSpec(kind="float", default=1.0, low=0.0, high=10.0, log=True)


def test_a_step_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="step must be positive"):
        ParamSpec(kind="int", default=5, low=1, high=10, step=0)


def test_a_param_spec_is_frozen() -> None:
    spec = ParamSpec(kind="int", default=10, low=1, high=100)
    with pytest.raises(ValidationError):
        spec.default = 20


# ---------------------------------------------------------------------------
# clamping and resolution
# ---------------------------------------------------------------------------
def test_clamp_snaps_into_the_declared_space() -> None:
    spec = ParamSpec(kind="int", default=10, low=1, high=100)
    assert spec.clamp(500) == 100
    assert spec.clamp(-5) == 1
    assert spec.clamp(42) == 42


def test_clamp_honours_the_step() -> None:
    spec = ParamSpec(kind="float", default=1.0, low=0.0, high=10.0, step=0.5)
    assert spec.clamp(3.3) == pytest.approx(3.5)


def test_clamp_of_a_categorical_falls_back_to_the_default() -> None:
    spec = ParamSpec(kind="categorical", default="a", choices=("a", "b"))
    assert spec.clamp("b") == "b"
    assert spec.clamp("z") == "a"


def test_resolve_params_fills_in_defaults() -> None:
    declared = {
        "fast": ParamSpec(kind="int", default=20, low=5, high=100),
        "slow": ParamSpec(kind="int", default=100, low=20, high=400),
    }
    assert resolve_params(declared) == {"fast": 20, "slow": 100}
    assert resolve_params(declared, {"fast": 30}) == {"fast": 30, "slow": 100}


def test_resolve_params_clamps_supplied_values() -> None:
    declared = {"fast": ParamSpec(kind="int", default=20, low=5, high=100)}
    assert resolve_params(declared, {"fast": 9999})["fast"] == 100


def test_resolve_params_rejects_an_undeclared_key() -> None:
    """A typo that silently did nothing would be a parameter nobody is tuning."""
    with pytest.raises(ConfigError, match="does not declare"):
        resolve_params({}, {"fastt": 3})


# ---------------------------------------------------------------------------
# the indicator cache
# ---------------------------------------------------------------------------
def test_the_cache_computes_each_series_once(bars: BarFrame) -> None:
    cache = IndicatorCache(bars)
    cache.value_at(50, "sma", n=10)
    cache.value_at(51, "sma", n=10)
    assert cache.size == 1
    cache.value_at(51, "sma", n=20)
    assert cache.size == 2


def test_the_cache_returns_scalars_for_single_valued_indicators(bars: BarFrame) -> None:
    assert isinstance(IndicatorCache(bars).value_at(50, "sma", n=10), float)


def test_the_cache_returns_named_tuples_for_multi_valued_indicators(
    bars: BarFrame,
) -> None:
    bands = IndicatorCache(bars).value_at(50, "bbands", n=20)
    assert bands.lower <= bands.middle <= bands.upper
    macd = IndicatorCache(bars).value_at(60, "macd", fast=5, slow=12, signal=4)
    assert macd.histogram == pytest.approx(macd.macd - macd.signal)


def test_the_cache_reads_the_right_columns(bars: BarFrame) -> None:
    """``atr`` needs high/low/close; ``highest`` needs high, not close."""
    from quantlab.core.indicators import atr, highest

    cache = IndicatorCache(bars)
    assert cache.value_at(40, "atr", n=14) == pytest.approx(
        float(atr(bars.high, bars.low, bars.close, 14)[40])
    )
    assert cache.value_at(40, "highest", n=10) == pytest.approx(float(highest(bars.high, 10)[40]))


def test_an_unknown_indicator_is_rejected(bars: BarFrame) -> None:
    with pytest.raises(StrategyLoadError, match="unknown indicator"):
        IndicatorCache(bars).get("telepathy")


def test_wrong_indicator_arguments_are_rejected(bars: BarFrame) -> None:
    with pytest.raises(StrategyLoadError, match="wrong arguments"):
        IndicatorCache(bars).get("sma", window=10)


def test_the_library_lists_every_indicator_of_the_spec() -> None:
    assert set(INDICATORS) == {
        "sma",
        "ema",
        "rsi",
        "atr",
        "bbands",
        "donchian",
        "returns",
        "log_returns",
        "rolling_vol",
        "zscore",
        "highest",
        "lowest",
        "roc",
        "macd",
    }


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------
def make_context(bars: BarFrame, i: int = 20, **overrides) -> Context:
    defaults = {
        "i": i,
        "bars": bars.window(i),
        "position": Position.flat(),
        "equity": 10_000.0,
        "cash": 10_000.0,
        "params": {"fast": 10},
        "cache": IndicatorCache(bars),
        "seed": 7,
    }
    defaults.update(overrides)
    return Context(**defaults)


def test_the_context_exposes_the_run_state(bars: BarFrame) -> None:
    ctx = make_context(bars)
    assert ctx.i == 20
    assert len(ctx.bars) == 21
    assert ctx.equity == 10_000.0
    assert ctx.params["fast"] == 10
    assert "i=20" in repr(ctx)


def test_the_context_is_immutable(bars: BarFrame) -> None:
    ctx = make_context(bars)
    for attribute in ("i", "equity", "cash", "params", "bars"):
        with pytest.raises(AttributeError, match="read-only"):
            setattr(ctx, attribute, None)


def test_the_context_indicator_is_causal(bars: BarFrame) -> None:
    from quantlab.core.indicators import sma

    ctx = make_context(bars, 30)
    assert ctx.ind("sma", n=10) == pytest.approx(float(sma(bars.close[:31], 10)[-1]))


def test_the_rng_is_seeded_from_the_run_seed_and_the_bar(bars: BarFrame) -> None:
    a = make_context(bars, 20, seed=7).rng.random()
    b = make_context(bars, 20, seed=7).rng.random()
    c = make_context(bars, 21, seed=7).rng.random()
    d = make_context(bars, 20, seed=8).rng.random()
    assert a == b, "same seed and bar: same draw"
    assert a != c, "a different bar draws differently"
    assert a != d, "a different run seed draws differently"


def test_the_rng_is_built_once_per_context(bars: BarFrame) -> None:
    ctx = make_context(bars)
    assert ctx.rng is ctx.rng


def test_the_rng_does_not_depend_on_how_many_bars_preceded_it() -> None:
    """What lets a truncated run reproduce a full one's draws at the same bar."""
    short = BarFrame(make_bars(30, seed=3), symbol="B/U", timeframe="1h")
    long = BarFrame(make_bars(90, seed=3), symbol="B/U", timeframe="1h")
    for i in (0, 5, 29):
        assert (
            make_context(short, i, seed=99).rng.random()
            == make_context(long, i, seed=99).rng.random()
        )


# ---------------------------------------------------------------------------
# Signal, RiskSpec, SizingSpec
# ---------------------------------------------------------------------------
def test_a_signal_tag_is_bounded() -> None:
    assert Signal(SignalKind.LONG, tag="x" * 32).tag == "x" * 32
    with pytest.raises(ValueError, match="32 characters"):
        Signal(SignalKind.LONG, tag="x" * 33)


def test_a_risk_spec_rejects_a_target_inside_the_stop() -> None:
    with pytest.raises(ValidationError, match="must exceed stop_loss_pct"):
        RiskSpec(stop_loss_pct=0.10, take_profit_pct=0.05)


def test_a_risk_spec_rejects_non_positive_percentages() -> None:
    for kwargs in ({"stop_loss_pct": 0.0}, {"take_profit_pct": -0.1}, {"time_stop_bars": 0}):
        with pytest.raises(ValidationError):
            RiskSpec(**kwargs)


def test_is_active_reports_whether_any_control_is_set() -> None:
    assert RiskSpec().is_active is False
    assert RiskSpec(stop_loss_pct=0.05).is_active is True
    assert RiskSpec(time_stop_bars=10).is_active is True


def test_a_sizing_spec_requires_its_mode_parameter() -> None:
    with pytest.raises(ValidationError, match="requires target_vol_annual"):
        SizingSpec(mode="volatility_target")
    with pytest.raises(ValidationError, match="requires atr_risk_pct"):
        SizingSpec(mode="atr_risk")


def test_a_sizing_spec_accepts_a_complete_declaration() -> None:
    spec = SizingSpec(mode="atr_risk", atr_risk_pct=0.01, atr_period=20)
    assert spec.atr_period == 20


# ---------------------------------------------------------------------------
# the protocol and the shipped strategies
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "module",
    [
        "strategies.TEMPLATE",
        "strategies.baselines.buy_and_hold",
        "strategies.baselines.sma_cross",
        "strategies.baselines.rsi_reversion",
        "strategies.baselines.random_entry",
    ],
)
def test_every_shipped_strategy_satisfies_the_protocol(module: str) -> None:
    import importlib

    strategy = importlib.import_module(module).STRATEGY()
    assert isinstance(strategy, Strategy)
    assert isinstance(strategy.risk, RiskSpec)
    assert isinstance(strategy.sizing, SizingSpec)


@pytest.mark.parametrize(
    "module",
    [
        "strategies.TEMPLATE",
        "strategies.baselines.sma_cross",
        "strategies.baselines.rsi_reversion",
        "strategies.baselines.random_entry",
    ],
)
def test_every_declared_parameter_is_bounded(module: str) -> None:
    import importlib

    for name, spec in importlib.import_module(module).STRATEGY.params.items():
        assert isinstance(spec, ParamSpec), name
        if spec.kind in ("int", "float"):
            assert spec.low is not None and spec.high is not None, name


def test_a_baseline_returns_a_signal(bars: BarFrame) -> None:
    from strategies.baselines.sma_cross import STRATEGY as SmaCross

    strategy = SmaCross()
    strategy.prepare(resolve_params(SmaCross.params, {"fast": 3, "slow": 8}))
    signal = strategy.on_bar(make_context(bars, 40))
    assert isinstance(signal, Signal)
    assert signal.kind in (SignalKind.LONG, SignalKind.FLAT)


def test_a_baseline_is_flat_during_warm_up(bars: BarFrame) -> None:
    from strategies.baselines.rsi_reversion import STRATEGY as RsiReversion

    strategy = RsiReversion()
    strategy.prepare(resolve_params(RsiReversion.params, {}))
    assert strategy.on_bar(make_context(bars, 1)).kind is SignalKind.FLAT


def test_random_entry_holds_for_the_configured_number_of_bars(bars: BarFrame) -> None:
    from strategies.baselines.random_entry import STRATEGY as RandomEntry

    strategy = RandomEntry()
    strategy.prepare(resolve_params(RandomEntry.params, {"p_enter": 0.5, "hold_bars": 3}))
    held = Position(side=Side.LONG, qty=1.0, entry_px=100.0, entry_bar=0)

    strategy.entered_at = 10
    assert strategy.on_bar(make_context(bars, 11, position=held)).kind is SignalKind.LONG
    assert strategy.on_bar(make_context(bars, 13, position=held)).kind is SignalKind.FLAT


def test_indicator_arrays_handed_out_are_read_only(bars: BarFrame) -> None:
    ctx = make_context(bars, 30)
    closes = ctx.bars.close
    assert not closes.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        closes[0] = np.nan
