"""Slippage models (master spec section 8.5).

Every model returns basis points and every one of them is **non-negative**: the
engine always applies slippage against the trader, so a model that could return
a negative number would hand out free money in exactly the situations a search
process is best at finding.

All models read bar ``i-1`` rather than bar ``i``.  The decision to trade is made
at the close of bar ``i-1`` and filled at the open of bar ``i``, so ``i-1`` is the
last bar whose statistics are known when the cost is incurred.  Reading ``i``
would be a look-ahead — a small one, and a very convenient one, which is why it
is worth stating rather than assuming.
"""

from __future__ import annotations

from typing import Final, Protocol, runtime_checkable

import numpy as np

from quantlab.core.errors import ConfigError
from quantlab.core.indicators import atr
from quantlab.core.types import BarFrame, SlippageConfig

__all__ = [
    "ATR_PERIOD",
    "FixedBps",
    "SlippageModel",
    "VolatilityScaled",
    "VolumeImpact",
    "build_slippage_model",
]

#: Lookback used by :class:`VolatilityScaled`, fixed by spec section 8.5.
ATR_PERIOD: Final[int] = 14

_EPS: Final[float] = 1e-12


@runtime_checkable
class SlippageModel(Protocol):
    """Basis points of adverse price movement for a fill on bar ``i``."""

    name: str

    def bps(self, bars: BarFrame, i: int, notional: float) -> float:
        """Return slippage in basis points; never negative."""
        ...


class FixedBps:
    """A constant cost per side.  The default, and the only one that cannot surprise."""

    name = "fixed_bps"

    def __init__(self, bps: float) -> None:
        if bps < 0:
            raise ConfigError("fixed slippage must not be negative", bps=bps)
        self.value = float(bps)

    def bps(self, bars: BarFrame, i: int, notional: float) -> float:  # noqa: ARG002
        return self.value

    def __repr__(self) -> str:
        return f"FixedBps({self.value})"


class VolatilityScaled:
    """``k * ATR14[i-1] / close[i-1] * 1e4`` — wider spreads when the market moves.

    Falls back to zero while the ATR is still warming up: a made-up cost would be
    just as arbitrary, and the fallback direction is visible here rather than
    buried in an engine branch.
    """

    name = "volatility_scaled"

    def __init__(self, k: float, *, period: int = ATR_PERIOD) -> None:
        if k < 0:
            raise ConfigError("volatility slippage coefficient must not be negative", k=k)
        self.k = float(k)
        self.period = int(period)
        self._cache: tuple[int, np.ndarray] | None = None

    def _atr(self, bars: BarFrame) -> np.ndarray:
        # Cached per frame identity: the engine calls this once per fill and the
        # ATR over a whole segment is the same array every time.
        key = id(bars)
        if self._cache is None or self._cache[0] != key:
            self._cache = (key, atr(bars.high, bars.low, bars.close, self.period))
        return self._cache[1]

    def bps(self, bars: BarFrame, i: int, notional: float) -> float:  # noqa: ARG002
        if i <= 0:
            return 0.0
        values = self._atr(bars)
        previous = float(values[i - 1])
        close = float(bars.close[i - 1])
        if not np.isfinite(previous) or close <= _EPS:
            return 0.0
        return max(0.0, self.k * previous / close * 1e4)

    def __repr__(self) -> str:
        return f"VolatilityScaled(k={self.k}, period={self.period})"


class VolumeImpact:
    """``a_bps + b * notional / quote_volume[i-1]`` — larger orders cost more.

    When the previous bar reported no quote volume the impact term is undefined;
    the model returns the floor ``a_bps`` rather than infinity, and the engine's
    own minimum-notional rule is what stops an absurd order in a dead market.
    """

    name = "volume_impact"

    def __init__(self, a_bps: float, b: float) -> None:
        if a_bps < 0 or b < 0:
            raise ConfigError("volume impact parameters must not be negative", a_bps=a_bps, b=b)
        self.a_bps = float(a_bps)
        self.b = float(b)

    def bps(self, bars: BarFrame, i: int, notional: float) -> float:
        if i <= 0:
            return self.a_bps
        volume = float(bars.quote_volume[i - 1])
        if volume <= _EPS:
            return self.a_bps
        return max(0.0, self.a_bps + self.b * abs(float(notional)) / volume)

    def __repr__(self) -> str:
        return f"VolumeImpact(a_bps={self.a_bps}, b={self.b})"


def build_slippage_model(config: SlippageConfig) -> SlippageModel:
    """Build the configured model.

    Raises:
        ConfigError: if the model name is not one of the three in spec section 8.5.
    """
    if config.model == "fixed_bps":
        return FixedBps(config.fixed_bps)
    if config.model == "volatility_scaled":
        return VolatilityScaled(config.vol_k)
    if config.model == "volume_impact":
        return VolumeImpact(config.impact_a_bps, config.impact_b)
    raise ConfigError("unknown slippage model", model=config.model)
