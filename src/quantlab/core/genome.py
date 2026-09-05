"""The strategy genome (master spec section 9.6).

Structural mutation cannot operate safely on free-form Python: an edit to a token
stream produces invalid or subtly broken code far more often than it produces a
strategy. The genome is a declarative, typed, validated description of a strategy
from which module source is generated (``evolution/compiler.py``). Mutations act
on the genome, so a mutated candidate is valid **by construction**, and the
generated code still faces the AST check, the sandbox and the leakage probe as
defence in depth.

Nothing here executes anything, generates anything, or reads a bar. This module
holds the *representation* and the rules that make one valid; there is exactly one
of each, because a second spelling of either would let two candidates that are the
same thing carry different ids and be evaluated twice.

Three narrowings against section 9.6, all deliberate and all fail-closed. They
shrink the space of genomes the optimiser may reach; they never widen it, so a
genome valid here is valid under the full section 9.6 rules too:

* **Single-valued indicators only.** ``bbands``, ``donchian`` and ``macd`` return
  named tuples, so an operand naming one would need a field selector that section
  9.6's ``kwargs`` — "kwargs match that indicator's signature" — has nowhere to
  put. They are refused by name rather than mis-compiled.
* **Numeric parameters only.** Every genome parameter ends up on one side of a
  numeric comparison, so ``categorical`` and ``bool`` have no meaning here.
* **Long or flat.** The genome expresses an entry and an exit, which is a
  direction-free structure; shorting is a separate axis section 9.6 does not
  give the genome a field for.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator, model_validator

from quantlab.core.config import GenomeSettings
from quantlab.core.hashing import canonical_json, short_id
from quantlab.core.strategy import INDICATORS, ParamSpec
from quantlab.core.types import RiskSpec, SizingSpec

__all__ = [
    "COMPARISON_OPS",
    "CROSS_OPS",
    "GENOME_INDICATORS",
    "MULTI_OUTPUT_INDICATORS",
    "PRICE_COLUMNS",
    "PRICE_DOMAINS",
    "Condition",
    "ConditionTree",
    "Domain",
    "IndicatorSignature",
    "Operand",
    "StrategyGenome",
    "genome_id",
]


#: The units a value is measured in (see :attr:`IndicatorSignature.domain`).
Domain = Literal[
    "price", "range", "volume", "oscillator", "zscore", "ratio", "percent", "volatility"
]


@dataclass(frozen=True, slots=True)
class IndicatorSignature:
    """What a genome operand may say about one indicator, and what it costs.

    ``lookback`` is the number of bars the indicator needs before it produces a
    number, measured as ``first defined index + 1``. It is stated here rather than
    discovered, because a genome must be checkable without running anything —
    and ``tests/unit/test_genome.py`` holds every entry to the real implementation
    so the table cannot drift away from ``core/indicators.py``.
    """

    #: The one keyword argument the genome may supply: the lookback period.
    period: str = "n"
    #: Smallest period the underlying function accepts.
    min_period: int = 1
    #: Bars consumed beyond the period itself, e.g. the extra bar a return needs.
    extra: int = 0
    #: What the indicator's values are measured in.
    #:
    #: A fact about the indicator, recorded here because it is the only place
    #: that knows all of them. Genome *validity* ignores it — comparing a moving
    #: average against a fixed price level is a legitimate strategy — but the
    #: operator library of section 13.4 uses it to draw comparisons that mean
    #: something, rather than pairing an RSI against a Bitcoin price.
    domain: Domain = "price"

    def lookback(self, period: int) -> int:
        return int(period) + self.extra


#: Indicators a genome may name, and the warm-up each implies (spec section 9.6).
GENOME_INDICATORS: Final[Mapping[str, IndicatorSignature]] = {
    "atr": IndicatorSignature(extra=1, domain="range"),
    "ema": IndicatorSignature(domain="price"),
    "highest": IndicatorSignature(domain="price"),
    "log_returns": IndicatorSignature(extra=1, domain="ratio"),
    "lowest": IndicatorSignature(domain="price"),
    "returns": IndicatorSignature(extra=1, domain="ratio"),
    "roc": IndicatorSignature(extra=1, domain="percent"),
    "rolling_vol": IndicatorSignature(min_period=2, extra=1, domain="volatility"),
    "rsi": IndicatorSignature(extra=1, domain="oscillator"),
    "sma": IndicatorSignature(domain="price"),
    "zscore": IndicatorSignature(min_period=2, domain="zscore"),
}

#: Indicators that return a named tuple, and so cannot be a genome operand yet.
MULTI_OUTPUT_INDICATORS: Final[frozenset[str]] = frozenset({"bbands", "donchian", "macd"})

#: What a price column is measured in, by the same rule as :attr:`IndicatorSignature.domain`.
PRICE_DOMAINS: Final[Mapping[str, Domain]] = {
    "open": "price",
    "high": "price",
    "low": "price",
    "close": "price",
    "volume": "volume",
}

#: Bar columns an operand may read. Deliberately the price and volume columns
#: only: ``ts_open`` is a clock, and ``is_gap_filled`` is a data-quality flag that
#: a strategy has no business trading on.
PRICE_COLUMNS: Final[frozenset[str]] = frozenset({"open", "high", "low", "close", "volume"})

#: Operators comparing this bar's values.
COMPARISON_OPS: Final[frozenset[str]] = frozenset({"<", "<=", ">", ">="})

#: Operators that also need the previous bar's values.
CROSS_OPS: Final[frozenset[str]] = frozenset({"cross_above", "cross_below"})

#: Section 9.2's hard ceiling sits ``+4`` above the soft one of section 9.6.
PARAM_LIMIT_SLACK: Final[int] = 4


def _limits(info: ValidationInfo) -> GenomeSettings:
    """Structural limits for this validation, from the context or the defaults.

    A genome validated with no context uses ``GenomeSettings()`` — the same
    defaults ``configs/default.yaml`` ships — so a genome that breaks a limit
    cannot be constructed at all, which is what section 9.6 requires. A caller
    holding a tightened configuration passes it as
    ``StrategyGenome.model_validate(data, context={"limits": settings})``.
    """
    context = info.context if isinstance(info.context, dict) else None
    supplied = context.get("limits") if context else None
    return supplied if isinstance(supplied, GenomeSettings) else GenomeSettings()


class Operand(BaseModel):
    """One side of a comparison (spec section 9.6)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["indicator", "price", "constant", "param"]
    #: ``'sma'`` | ``'close'`` | ``''`` | ``'fast'``, by kind.
    name: str = ""
    #: Indicator arguments; a string value names a parameter instead of fixing one.
    kwargs: dict[str, int | float | str] = {}

    @field_validator("kwargs", mode="before")
    @classmethod
    def _no_booleans(cls, value: object) -> object:
        """Refuse ``True``/``False`` before pydantic can read them as ``1``/``0``.

        The genome is the canonical representation of a candidate, and a proposal
        that spelled a threshold as a boolean would be plausible-looking nonsense
        rather than a second spelling of something sensible. ``mode="before"`` is
        required: by the time the union has run, the bool is already an int.
        """
        if isinstance(value, dict) and any(isinstance(item, bool) for item in value.values()):
            raise ValueError("a boolean is not a number; kwargs take numbers or parameter names")
        return value

    @model_validator(mode="after")
    def _shape_matches_the_kind(self) -> Operand:
        getattr(self, f"_check_{self.kind}")()
        return self

    def _check_indicator(self) -> None:
        if self.name in MULTI_OUTPUT_INDICATORS:
            raise ValueError(
                f"{self.name!r} returns several series, and an operand has no way to say "
                "which one it means; use a single-valued indicator"
            )
        signature = GENOME_INDICATORS.get(self.name)
        if signature is None:
            raise ValueError(
                f"unknown indicator {self.name!r}; genome indicators are "
                f"{sorted(GENOME_INDICATORS)}"
            )
        if set(self.kwargs) != {signature.period}:
            raise ValueError(
                f"indicator {self.name!r} takes exactly one argument, {signature.period!r}"
            )
        period = self.kwargs[signature.period]
        if isinstance(period, str):
            if not period.isidentifier():
                raise ValueError("a parameter reference must be an identifier")
        elif isinstance(period, bool) or not isinstance(period, int):
            raise ValueError("a lookback period must be a whole number of bars")
        elif period < signature.min_period:
            raise ValueError(f"{self.name!r} needs a period of at least {signature.min_period}")

    def _check_price(self) -> None:
        if self.name not in PRICE_COLUMNS:
            raise ValueError(
                f"unknown price column {self.name!r}; use one of {sorted(PRICE_COLUMNS)}"
            )
        if self.kwargs:
            raise ValueError("a price operand takes no arguments")

    def _check_constant(self) -> None:
        if self.name:
            raise ValueError("a constant operand carries its value in kwargs, not in name")
        value = self.kwargs.get("value")
        if set(self.kwargs) != {"value"} or isinstance(value, (bool, str)):
            raise ValueError("a constant operand needs exactly one numeric kwarg, 'value'")

    def _check_param(self) -> None:
        if not self.name.isidentifier():
            raise ValueError("a parameter operand must name an identifier")
        if self.kwargs:
            raise ValueError("a parameter operand takes no arguments")

    # -- what it refers to --------------------------------------------------
    @property
    def value(self) -> float:
        """The literal a ``constant`` operand carries."""
        if self.kind != "constant":
            raise ValueError("only a constant operand has a value")
        return float(self.kwargs["value"])

    @property
    def period_param(self) -> str | None:
        """The parameter naming this indicator's period, if it is not fixed."""
        if self.kind != "indicator":
            return None
        period = self.kwargs[GENOME_INDICATORS[self.name].period]
        return period if isinstance(period, str) else None

    def param_references(self) -> frozenset[str]:
        """Every parameter this operand names, directly or as a lookback."""
        if self.kind == "param":
            return frozenset({self.name})
        reference = self.period_param
        return frozenset({reference}) if reference else frozenset()

    def lookback(self, params: Mapping[str, ParamSpec]) -> int:
        """Bars of history this operand needs before it produces a number.

        A period driven by a parameter is costed at that parameter's **upper
        bound**: warm-up is declared once for the genome, but the optimiser may
        set the parameter anywhere in its range, and a warm-up that only covers
        the default would leave the strategy reading NaN for part of its own
        search space.
        """
        if self.kind != "indicator":
            return 0
        signature = GENOME_INDICATORS[self.name]
        reference = self.period_param
        if reference is None:
            return signature.lookback(int(self.kwargs[signature.period]))
        spec = params.get(reference)
        if spec is None or spec.high is None:
            raise ValueError(f"parameter {reference!r} has no upper bound to cost a lookback at")
        return signature.lookback(int(spec.high))

    @property
    def domain(self) -> Domain | None:
        """What this operand's values are measured in, or ``None`` if unitless.

        A ``param`` or ``constant`` takes its meaning from whatever it is
        compared against, so neither carries a domain of its own.
        """
        if self.kind == "indicator":
            return GENOME_INDICATORS[self.name].domain
        if self.kind == "price":
            return PRICE_DOMAINS[self.name]
        return None

    @property
    def key(self) -> str:
        """A stable identity, so two spellings of one operand share a slot."""
        return canonical_json({"kind": self.kind, "name": self.name, "kwargs": self.kwargs})


class Condition(BaseModel):
    """One comparison between two operands (spec section 9.6)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    left: Operand
    op: Literal["<", "<=", ">", ">=", "cross_above", "cross_below"]
    right: Operand

    @model_validator(mode="after")
    def _compares_two_different_things(self) -> Condition:
        if self.left.key == self.right.key:
            raise ValueError("a condition that compares an operand to itself decides nothing")
        if self.left.kind == "constant" and self.right.kind == "constant":
            raise ValueError(
                "a condition between two constants has the same answer on every bar; it "
                "would occupy a condition slot and pay the complexity penalty for nothing"
            )
        return self

    @property
    def operands(self) -> tuple[Operand, Operand]:
        return (self.left, self.right)

    @property
    def key(self) -> str:
        return f"{self.left.key}|{self.op}|{self.right.key}"


class ConditionTree(BaseModel):
    """A conjunction or disjunction of conditions (spec section 9.6)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["all", "any"] = "all"
    conditions: tuple[Condition, ...] = ()

    @model_validator(mode="after")
    def _no_duplicate_condition(self) -> ConditionTree:
        keys = [condition.key for condition in self.conditions]
        if len(set(keys)) != len(keys):
            raise ValueError("a tree repeats a condition, which cannot change its answer")
        return self

    def __len__(self) -> int:
        return len(self.conditions)


class StrategyGenome(BaseModel):
    """A whole strategy, declaratively (spec section 9.6).

    Every rule of section 9.6 is enforced here rather than at compile time: a
    genome that reached the population invalid would be discovered only when its
    generated source failed the AST check, by which point it has already consumed
    a population slot and an evaluation the multiple-testing correction of section
    14.4 has to account for.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    version: str = "1"
    entry: ConditionTree
    exit: ConditionTree = ConditionTree()
    filters: tuple[Condition, ...] = ()
    risk: RiskSpec = RiskSpec()
    sizing: SizingSpec = SizingSpec()
    params: dict[str, ParamSpec] = {}
    warmup_bars: int = 0

    # -- structure ----------------------------------------------------------
    def conditions(self) -> Iterator[Condition]:
        """Every condition, in the order the compiler emits them."""
        yield from self.entry.conditions
        yield from self.exit.conditions
        yield from self.filters

    def operands(self) -> Iterator[Operand]:
        for condition in self.conditions():
            yield from condition.operands

    def indicator_operands(self) -> list[Operand]:
        """Distinct indicator operands, in first-mention order."""
        seen: dict[str, Operand] = {}
        for operand in self.operands():
            if operand.kind == "indicator":
                seen.setdefault(operand.key, operand)
        return list(seen.values())

    def param_references(self) -> frozenset[str]:
        references: frozenset[str] = frozenset()
        for operand in self.operands():
            references |= operand.param_references()
        return references

    def max_lookback(self) -> int:
        """The largest warm-up any operand implies (spec section 9.6 rule 6)."""
        return max((operand.lookback(self.params) for operand in self.operands()), default=0)

    # -- validity -----------------------------------------------------------
    @model_validator(mode="after")
    def _is_a_valid_genome(self, info: ValidationInfo) -> StrategyGenome:
        limits = _limits(info)
        self._check_identity()
        self._check_counts(limits)
        self._check_params(limits)
        self._check_warmup()
        return self

    def _check_identity(self) -> None:
        if not self.name.isidentifier():
            raise ValueError("a genome name must be usable as part of a class name")

    def _check_counts(self, limits: GenomeSettings) -> None:
        if not self.entry.conditions:
            raise ValueError("a genome with no entry condition never takes a position")
        for label, count, ceiling in (
            ("entry", len(self.entry), limits.max_conditions_entry),
            ("exit", len(self.exit), limits.max_conditions_exit),
            ("filters", len(self.filters), limits.max_filters),
            ("indicators", len(self.indicator_operands()), limits.max_indicators),
        ):
            if count > ceiling:
                raise ValueError(f"too many {label}: {count} > {ceiling}")
        keys = [condition.key for condition in self.filters]
        if len(set(keys)) != len(keys):
            raise ValueError("a filter is repeated, which cannot change its answer")

    def _check_params(self, limits: GenomeSettings) -> None:
        ceiling = limits.max_free_params + PARAM_LIMIT_SLACK
        if len(self.params) > ceiling:
            raise ValueError(f"too many parameters: {len(self.params)} > {ceiling}")
        for key, spec in self.params.items():
            if not key.isidentifier():
                raise ValueError(f"parameter name {key!r} is not an identifier")
            if spec.kind not in ("int", "float"):
                raise ValueError(
                    f"parameter {key!r} is {spec.kind!r}; a genome parameter always ends up "
                    "on one side of a numeric comparison"
                )

        referenced = self.param_references()
        unknown = sorted(referenced - set(self.params))
        if unknown:
            raise ValueError(f"operands name parameters that are not declared: {unknown}")
        unused = sorted(set(self.params) - referenced)
        if unused:
            raise ValueError(
                f"declared but never used: {unused}; an unused parameter inflates the free "
                "parameter count without changing a single decision"
            )
        for operand in self.operands():
            reference = operand.period_param
            if reference and self.params[reference].kind != "int":
                raise ValueError(
                    f"parameter {reference!r} sets a lookback period, so it must be an int"
                )

    def _check_warmup(self) -> None:
        needed = self.max_lookback()
        if self.warmup_bars < needed:
            raise ValueError(
                f"warmup_bars is {self.warmup_bars} but the indicators need {needed}; "
                "a strategy trading before its own indicators are defined is trading on NaN"
            )

    # -- identity -----------------------------------------------------------
    def canonical(self) -> str:
        """The canonical JSON this genome hashes to."""
        return canonical_json(self.model_dump(mode="json"))


def genome_id(genome: StrategyGenome) -> str:
    """``sha256(canonical genome)[:16]`` (spec section 9.6).

    The id of the *structure*, not of the source it compiles to. Both exist and
    they answer different questions: two genomes with the same id are the same
    candidate, and ``strategy_id`` — the hash of the compiled bytes — is what a
    run cites.
    """
    return short_id(genome.canonical())


def _every_indicator_is_classified() -> None:
    """Each indicator is either usable in a genome or excluded by name.

    Run at import so the two tables above cannot fall behind
    ``core/indicators.py``: adding an indicator there without deciding whether a
    genome may name it fails immediately, rather than leaving a capability that
    mutation can never reach and nobody notices is missing.
    """
    classified = set(GENOME_INDICATORS) | MULTI_OUTPUT_INDICATORS
    unclassified = sorted(set(INDICATORS) - classified)
    invented = sorted(classified - set(INDICATORS))
    if unclassified or invented:  # pragma: no cover - a constant, checked at import
        raise ValueError(
            "the genome indicator tables disagree with core.indicators: "
            f"unclassified={unclassified}, not real indicators={invented}"
        )


_every_indicator_is_classified()
