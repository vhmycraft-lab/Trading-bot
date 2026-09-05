"""The strategy contract (master spec section 9.1).

A strategy sees exactly one thing: a :class:`Context` at bar ``i``, whose
:attr:`Context.bars` window stops at ``i``.  It returns a :class:`Signal`.  It
does not size positions, does not place orders, does not manage stops and cannot
read the future — all four are engine responsibilities, and moving any of them
into the strategy would require intrabar information the window refuses to give
(INV-3).

Indicators come from :meth:`Context.ind`, which reads a causal array the engine
computed once for the whole segment.  Because every function in
:mod:`quantlab.core.indicators` is causal, indexing that array at ``i`` returns
exactly what recomputing over ``bars[0..i]`` would have returned — proved bar by
bar in ``tests/unit/test_indicators.py`` — so the strategy gets the speed of a
vectorised computation with the guarantee of a windowed one.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

from quantlab.core.errors import ConfigError, StrategyLoadError
from quantlab.core.indicators import (
    atr,
    bbands,
    donchian,
    ema,
    highest,
    log_returns,
    lowest,
    macd,
    returns,
    roc,
    rolling_vol,
    rsi,
    sma,
    zscore,
)
from quantlab.core.types import (
    BarFrame,
    BarWindow,
    Position,
    RiskSpec,
    Signal,
    SignalKind,
    SizingSpec,
)

__all__ = [
    "INDICATORS",
    "BarWindow",
    "Context",
    "IndicatorCache",
    "ParamSpec",
    "Position",
    "RiskSpec",
    "Signal",
    "SignalKind",
    "SizingSpec",
    "Strategy",
    "StrategyBase",
    "VectorizedStrategy",
    "resolve_params",
]


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------
class ParamSpec(BaseModel):
    """A single tunable parameter, always bounded (spec section 9.1).

    Bounds are mandatory rather than optional: an unbounded parameter cannot be
    searched, cannot be mutated within a range, and cannot be reported as a
    fraction of its own space.  Rejecting it at declaration time is the only
    point where the cost of fixing it is small.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["int", "float", "categorical", "bool"]
    default: int | float | str | bool
    low: float | None = None
    high: float | None = None
    step: float | None = None
    choices: tuple[str, ...] | None = None
    log: bool = False

    @model_validator(mode="after")
    def _bounds_match_the_kind(self) -> ParamSpec:
        if self.kind in ("int", "float"):
            if self.low is None or self.high is None:
                raise ValueError(f"{self.kind} parameters must declare both low and high")
            if self.low >= self.high:
                raise ValueError("low must be strictly below high")
            if isinstance(self.default, bool) or not isinstance(self.default, (int, float)):
                raise ValueError(f"{self.kind} parameter needs a numeric default")
            if not self.low <= float(self.default) <= self.high:
                raise ValueError("default must lie within [low, high]")
            if self.step is not None and self.step <= 0:
                raise ValueError("step must be positive")
            if self.log and self.low <= 0:
                raise ValueError("log-scaled parameters need a positive lower bound")
            if self.choices is not None:
                raise ValueError("numeric parameters must not declare choices")
        elif self.kind == "categorical":
            if not self.choices:
                raise ValueError("categorical parameters must declare choices")
            if self.default not in self.choices:
                raise ValueError("default must be one of choices")
        elif self.kind == "bool" and not isinstance(self.default, bool):
            raise ValueError("bool parameter needs a boolean default")
        return self

    def clamp(self, value: Any) -> Any:
        """Snap ``value`` into this parameter's declared space."""
        if self.kind == "bool":
            return bool(value)
        if self.kind == "categorical":
            choices = self.choices or ()
            return value if value in choices else self.default
        low, high = float(self.low or 0.0), float(self.high or 0.0)
        number = float(value)
        if self.step:
            number = low + round((number - low) / self.step) * self.step
        number = min(max(number, low), high)
        return round(number) if self.kind == "int" else number


def resolve_params(
    declared: Mapping[str, ParamSpec], supplied: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Merge supplied values over declared defaults, clamped to their bounds.

    Raises:
        ConfigError: if a supplied key was never declared.  A typo that silently
            did nothing would be a parameter the search believes it is tuning.
    """
    values = dict(supplied or {})
    unknown = sorted(set(values) - set(declared))
    if unknown:
        raise ConfigError(
            "parameters were supplied that the strategy does not declare",
            unknown=unknown,
            declared=sorted(declared),
        )
    return {
        name: spec.clamp(values[name]) if name in values else spec.default
        for name, spec in declared.items()
    }


# ---------------------------------------------------------------------------
# indicators, computed once per segment
# ---------------------------------------------------------------------------
#: Indicators reachable through :meth:`Context.ind`, by the name a strategy uses.
INDICATORS: Mapping[str, Any] = {
    "sma": sma,
    "ema": ema,
    "rsi": rsi,
    "atr": atr,
    "bbands": bbands,
    "donchian": donchian,
    "returns": returns,
    "log_returns": log_returns,
    "rolling_vol": rolling_vol,
    "zscore": zscore,
    "highest": highest,
    "lowest": lowest,
    "roc": roc,
    "macd": macd,
}

#: Indicators whose first positional argument is not the close series.
_INPUTS: Mapping[str, tuple[str, ...]] = {
    "atr": ("high", "low", "close"),
    "donchian": ("high", "low"),
    "highest": ("high",),
    "lowest": ("low",),
}


class IndicatorCache:
    """Causal indicator arrays for one segment, computed on first use.

    Owned by the engine and shared across bars.  Computing over the whole
    segment is safe precisely because every indicator is causal; the cache exists
    so that a 47 000-bar backtest is linear rather than quadratic.
    """

    __slots__ = ("_bars", "_cache")

    def __init__(self, bars: BarFrame) -> None:
        self._bars = bars
        self._cache: dict[tuple[str, tuple[tuple[str, Any], ...]], Any] = {}

    def get(self, name: str, **kwargs: Any) -> Any:
        """Return the full causal array (or tuple of arrays) for ``name``."""
        try:
            function = INDICATORS[name]
        except KeyError:
            raise StrategyLoadError(
                "unknown indicator", indicator=name, available=sorted(INDICATORS)
            ) from None

        key = (name, tuple(sorted(kwargs.items())))
        if key not in self._cache:
            inputs = [self._bars.column(column) for column in _INPUTS.get(name, ("close",))]
            try:
                self._cache[key] = function(*inputs, **kwargs)
            except TypeError as exc:
                raise StrategyLoadError(
                    "indicator called with the wrong arguments", indicator=name, kwargs=dict(kwargs)
                ) from exc
        return self._cache[key]

    def value_at(self, i: int, name: str, **kwargs: Any) -> Any:
        """Return the indicator's value at bar ``i``, as a float or tuple of floats."""
        computed = self.get(name, **kwargs)
        if isinstance(computed, tuple):
            # Named tuples (BBands, Donchian, Macd) keep their field names, so a
            # strategy can write ``ctx.ind("bbands", n=20).upper``.
            values = [float(component[i]) for component in computed]
            make = getattr(type(computed), "_make", None)
            return make(values) if make is not None else tuple(values)
        return float(computed[i])

    @property
    def size(self) -> int:
        """Number of distinct indicator series computed so far."""
        return len(self._cache)


# ---------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------
class Context:
    """What a strategy sees at bar ``i``.  Constructed by the engine only.

    Immutable from the strategy's side: assigning to any attribute raises, so a
    strategy cannot smuggle state through the context between bars.  State that a
    strategy legitimately needs belongs on the strategy instance.
    """

    __slots__ = ("_cache", "_rng", "_seed", "bars", "cash", "equity", "i", "params", "position")

    i: int
    bars: BarWindow
    position: Position
    equity: float
    cash: float
    params: Mapping[str, Any]
    _cache: IndicatorCache
    _seed: int
    _rng: np.random.Generator | None

    def __init__(
        self,
        *,
        i: int,
        bars: BarWindow,
        position: Position,
        equity: float,
        cash: float,
        params: Mapping[str, Any],
        cache: IndicatorCache,
        seed: int = 0,
    ) -> None:
        for name, value in (
            ("i", i),
            ("bars", bars),
            ("position", position),
            ("equity", equity),
            ("cash", cash),
            ("params", params),
            ("_cache", cache),
            ("_seed", seed),
            ("_rng", None),
        ):
            object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Context is read-only")

    def ind(self, name: str, **kwargs: Any) -> Any:
        """Return the causal value of an indicator at the current bar.

        Single-valued indicators return a float; ``bbands``, ``donchian`` and
        ``macd`` return their named tuples of floats.
        """
        return self._cache.value_at(self.i, name, **kwargs)

    @property
    def rng(self) -> np.random.Generator:
        """The only randomness a strategy may use, seeded from ``(run seed, i)``.

        Deriving the stream from the bar index means a strategy's random draws are
        reproducible *and* independent of how many bars preceded it, so a
        truncated run makes the same draws as a full one at the same bar — which
        is what lets the leakage probe compare them.
        """
        if self._rng is None:
            object.__setattr__(self, "_rng", np.random.default_rng([self._seed, self.i]))
        rng = self._rng
        assert rng is not None
        return rng

    def __repr__(self) -> str:
        return f"Context(i={self.i}, equity={self.equity:.2f}, position={self.position})"


# ---------------------------------------------------------------------------
# the protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class Strategy(Protocol):
    """What the engine requires of a strategy (spec section 9.1)."""

    name: str
    version: str
    style: Literal["bar_loop", "vectorized"]
    params: ClassVar[dict[str, ParamSpec]]
    warmup_bars: int
    risk: ClassVar[RiskSpec]
    sizing: ClassVar[SizingSpec]

    def prepare(self, params: Mapping[str, Any]) -> None:
        """Receive the resolved parameters once, before the first bar."""
        ...

    def on_bar(self, ctx: Context) -> Signal:
        """Return the desired exposure at the close of bar ``ctx.i``."""
        ...


@runtime_checkable
class VectorizedStrategy(Protocol):
    """A strategy that emits its whole signal series at once (spec section 9.1)."""

    name: str
    version: str
    style: Literal["bar_loop", "vectorized"]
    params: ClassVar[dict[str, ParamSpec]]
    warmup_bars: int

    def prepare(self, params: Mapping[str, Any]) -> None: ...

    def signals(self, bars: pd.DataFrame, params: Mapping[str, Any]) -> pd.Series:
        """Return one :class:`SignalKind` per bar of the full segment."""
        ...


class StrategyBase:
    """Optional convenience base: defaults for everything the protocol requires.

    Subclassing is not required — the protocol is structural — but it removes the
    boilerplate from the common case and gives every strategy a sane
    ``prepare``.
    """

    name: str = "unnamed"
    version: str = "0"
    style: Literal["bar_loop", "vectorized"] = "bar_loop"
    params: ClassVar[dict[str, ParamSpec]] = {}
    warmup_bars: int = 0
    risk: ClassVar[RiskSpec] = RiskSpec()
    sizing: ClassVar[SizingSpec] = SizingSpec()

    def __init__(self) -> None:
        self.p: dict[str, Any] = {}

    def prepare(self, params: Mapping[str, Any]) -> None:
        self.p = dict(params)

    def on_bar(self, ctx: Context) -> Signal:  # pragma: no cover - overridden
        raise NotImplementedError
