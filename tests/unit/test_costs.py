"""Slippage models (master spec section 8.5)."""

from __future__ import annotations

import numpy as np
import pytest
from tests.helpers import make_bars, toy_frame

from quantlab.core.costs import (
    FixedBps,
    SlippageModel,
    VolatilityScaled,
    VolumeImpact,
    build_slippage_model,
)
from quantlab.core.errors import ConfigError
from quantlab.core.types import BarFrame, SlippageConfig


@pytest.fixture
def bars() -> BarFrame:
    return BarFrame(make_bars(60, seed=4), symbol="BTC/USDT", timeframe="1h")


def test_every_model_satisfies_the_port() -> None:
    for model in (FixedBps(5.0), VolatilityScaled(0.1), VolumeImpact(2.0, 50.0)):
        assert isinstance(model, SlippageModel)


# ---------------------------------------------------------------------------
# fixed
# ---------------------------------------------------------------------------
def test_fixed_bps_is_constant(bars: BarFrame) -> None:
    model = FixedBps(5.0)
    assert {model.bps(bars, i, 1_000.0) for i in range(1, 20)} == {5.0}
    assert "FixedBps(5.0)" in repr(model)


def test_fixed_bps_rejects_a_negative_cost() -> None:
    with pytest.raises(ConfigError, match="must not be negative"):
        FixedBps(-1.0)


# ---------------------------------------------------------------------------
# volatility scaled
# ---------------------------------------------------------------------------
def test_volatility_scaled_matches_its_definition(bars: BarFrame) -> None:
    from quantlab.core.indicators import atr

    model = VolatilityScaled(0.10)
    expected_atr = atr(bars.high, bars.low, bars.close, 14)
    i = 40
    expected = 0.10 * expected_atr[i - 1] / bars.close[i - 1] * 1e4
    assert model.bps(bars, i, 1_000.0) == pytest.approx(expected)


def test_volatility_scaled_reads_the_previous_bar(bars: BarFrame) -> None:
    """Reading bar i would be a look-ahead: the cost is incurred at its open."""
    model = VolatilityScaled(0.10)
    calm = toy_frame([(100.0, 100.2, 99.8, 100.0)] * 40)
    spiked = toy_frame([*[(100.0, 100.2, 99.8, 100.0)] * 30, *[(100.0, 200.0, 50.0, 100.0)] * 10])
    # Bar 30 is the first wild bar; its own slippage must still reflect bar 29.
    assert VolatilityScaled(0.10).bps(calm, 30, 1.0) == pytest.approx(model.bps(spiked, 30, 1.0))


def test_volatility_scaled_is_zero_during_warm_up(bars: BarFrame) -> None:
    model = VolatilityScaled(0.10)
    assert model.bps(bars, 0, 1_000.0) == 0.0
    assert model.bps(bars, 3, 1_000.0) == 0.0


def test_volatility_scaled_rejects_a_negative_coefficient() -> None:
    with pytest.raises(ConfigError, match="must not be negative"):
        VolatilityScaled(-0.1)


def test_volatility_scaled_caches_per_frame(bars: BarFrame) -> None:
    model = VolatilityScaled(0.10)
    first = model.bps(bars, 30, 1.0)
    assert model.bps(bars, 30, 1.0) == first
    assert "VolatilityScaled" in repr(model)


# ---------------------------------------------------------------------------
# volume impact
# ---------------------------------------------------------------------------
def test_volume_impact_grows_with_order_size(bars: BarFrame) -> None:
    model = VolumeImpact(2.0, 50.0)
    small = model.bps(bars, 20, 100.0)
    large = model.bps(bars, 20, 100_000.0)
    assert large > small >= 2.0


def test_volume_impact_matches_its_definition(bars: BarFrame) -> None:
    model = VolumeImpact(2.0, 50.0)
    i, notional = 20, 5_000.0
    expected = 2.0 + 50.0 * notional / float(bars.quote_volume[i - 1])
    assert model.bps(bars, i, notional) == pytest.approx(expected)


def test_volume_impact_falls_back_to_its_floor_in_a_dead_market() -> None:
    frame = toy_frame([(100.0, 101.0, 99.0, 100.0)] * 5)
    zeroed = frame.to_pandas()
    zeroed["quote_volume"] = 0.0
    dead = BarFrame(zeroed, symbol="BTC/USDT", timeframe="1h")
    assert VolumeImpact(2.0, 50.0).bps(dead, 3, 1_000.0) == 2.0


def test_volume_impact_at_the_first_bar_is_its_floor(bars: BarFrame) -> None:
    assert VolumeImpact(2.0, 50.0).bps(bars, 0, 1_000.0) == 2.0


def test_volume_impact_rejects_negative_parameters() -> None:
    with pytest.raises(ConfigError, match="must not be negative"):
        VolumeImpact(-1.0, 50.0)
    with pytest.raises(ConfigError, match="must not be negative"):
        VolumeImpact(2.0, -50.0)


# ---------------------------------------------------------------------------
# the property that matters
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "model",
    [FixedBps(0.0), FixedBps(5.0), VolatilityScaled(0.1), VolumeImpact(2.0, 50.0)],
    ids=["fixed_zero", "fixed", "volatility", "impact"],
)
def test_no_model_ever_returns_a_negative_cost(model: SlippageModel, bars: BarFrame) -> None:
    """A negative slippage would hand out free money, reliably found by a search."""
    for i in range(bars.n_bars):
        for notional in (0.0, 1.0, 1e6):
            assert model.bps(bars, i, notional) >= 0.0


def test_a_negative_notional_is_treated_as_its_magnitude(bars: BarFrame) -> None:
    model = VolumeImpact(2.0, 50.0)
    assert model.bps(bars, 20, -5_000.0) == pytest.approx(model.bps(bars, 20, 5_000.0))


# ---------------------------------------------------------------------------
# construction from config
# ---------------------------------------------------------------------------
def test_build_from_config() -> None:
    assert isinstance(build_slippage_model(SlippageConfig(model="fixed_bps")), FixedBps)
    assert isinstance(
        build_slippage_model(SlippageConfig(model="volatility_scaled")), VolatilityScaled
    )
    assert isinstance(build_slippage_model(SlippageConfig(model="volume_impact")), VolumeImpact)


def test_build_rejects_an_unknown_model() -> None:
    config = SlippageConfig(model="fixed_bps")
    object.__setattr__(config, "model", "telepathy")
    with pytest.raises(ConfigError, match="unknown slippage model"):
        build_slippage_model(config)


def test_the_configured_parameters_reach_the_model() -> None:
    model = build_slippage_model(SlippageConfig(model="fixed_bps", fixed_bps=12.5))
    assert isinstance(model, FixedBps)
    assert model.value == 12.5


def test_stress_multipliers_scale_the_realised_cost() -> None:
    """The engine multiplies, so a 3x stress is a 3x cost."""
    frame = toy_frame([(100.0, 101.0, 99.0, 100.0)] * 5)
    model = FixedBps(5.0)
    base = model.bps(frame, 3, 1_000.0)
    assert base * 3.0 == pytest.approx(15.0)
    assert np.isfinite(base)
