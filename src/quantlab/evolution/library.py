"""The operator library: where new genome material comes from (spec section 13.4).

Structural mutation appends conditions, replaces operands and draws whole fresh
genomes for immigrants. All of that needs a source of *sensible* genome material,
and this is it.

"Sensible" is a real constraint, not a nicety. A drawing rule that paired an RSI
against a Bitcoin price would fill the population with conditions that are either
always true or always false, and the search would spend its generations
rediscovering that. So every comparison this module draws is between operands
measured in the same units (:attr:`~quantlab.core.genome.IndicatorSignature.domain`),
and a constant is only ever drawn against a scale-free quantity, where a fixed
level means something.

Genome *validity* is deliberately looser: comparing a moving average against a
fixed price level is a legitimate strategy a person may write, and section 9.6
allows it. The library is what the search draws from, not what the type permits.

Everything here takes an explicit :class:`numpy.random.Generator`. Nothing reads
a global RNG, so a generation is reproducible from its seed (INV-7) and a replayed
mutation lands on the same value it landed on the first time (INV-10).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np
from pydantic import ValidationError

from quantlab.core.config import GenomeSettings
from quantlab.core.errors import StrategyError
from quantlab.core.genome import (
    COMPARISON_OPS,
    CROSS_OPS,
    GENOME_INDICATORS,
    PRICE_COLUMNS,
    PRICE_DOMAINS,
    Condition,
    ConditionTree,
    Domain,
    Operand,
    StrategyGenome,
)
from quantlab.core.strategy import ParamSpec
from quantlab.core.types import RiskSpec, SizingSpec

__all__ = [
    "CONSTANT_RANGES",
    "DEFAULT_PERIODS",
    "RISK_CONTROL_RANGES",
    "OperatorLibrary",
]

#: Lookback periods an operand may be drawn with. A geometric-ish ladder rather
#: than a uniform range: the difference between a 5-bar and a 10-bar average is
#: large, and between a 195-bar and a 200-bar average is nothing, so sampling
#: uniformly would spend most draws on distinctions the data cannot resolve.
DEFAULT_PERIODS: Final[tuple[int, ...]] = (5, 8, 12, 20, 30, 50, 80, 120, 200)

#: Where a constant may be drawn from, by the domain it is compared against.
#:
#: Absent domains are price-scaled: their numeric level depends on the asset and
#: the year, so a drawn constant would be meaningless. Those operands are only
#: ever compared against other operands.
CONSTANT_RANGES: Final[Mapping[Domain, tuple[float, float]]] = {
    "oscillator": (5.0, 95.0),
    "zscore": (-3.0, 3.0),
    "ratio": (-0.05, 0.05),
    "percent": (-5.0, 5.0),
    "volatility": (0.001, 0.05),
}

#: The range each risk control is drawn from when it is switched on (§13.4).
#: Every bound matches the one :class:`~quantlab.core.types.RiskSpec` enforces, so
#: a drawn value is always constructible.
RISK_CONTROL_RANGES: Final[Mapping[str, tuple[float, float]]] = {
    "stop_loss_pct": (0.005, 0.15),
    "take_profit_pct": (0.01, 0.40),
    "trailing_stop_pct": (0.005, 0.20),
    "time_stop_bars": (4.0, 240.0),
}

_INTEGER_RISK_CONTROLS: Final[frozenset[str]] = frozenset({"time_stop_bars"})


@dataclass(frozen=True, slots=True)
class OperatorLibrary:
    """Draws operands, conditions and whole genomes (spec section 13.4).

    Holds no state between draws: every method takes the generator it uses, so a
    library instance is safe to share across a generation and two draws from the
    same seeded generator are identical.
    """

    limits: GenomeSettings = field(default_factory=GenomeSettings)
    periods: tuple[int, ...] = DEFAULT_PERIODS
    #: Indicators the library may draw. Defaults to every one a genome can name.
    indicators: tuple[str, ...] = tuple(sorted(GENOME_INDICATORS))

    def __post_init__(self) -> None:
        unknown = sorted(set(self.indicators) - set(GENOME_INDICATORS))
        if unknown:
            raise StrategyError("library names indicators no genome can use", unknown=unknown)
        if not self.periods:
            raise StrategyError("the library needs at least one lookback period to draw")

    # -- primitives ---------------------------------------------------------
    def draw_period(self, rng: np.random.Generator, *, minimum: int = 1) -> int:
        """A lookback the drawn indicator accepts."""
        allowed = [period for period in self.periods if period >= minimum]
        if not allowed:  # pragma: no cover - every shipped minimum is 1 or 2
            raise StrategyError("no configured period satisfies the minimum", minimum=minimum)
        return int(rng.choice(allowed))

    def draw_indicator_operand(
        self, rng: np.random.Generator, *, domain: Domain | None = None
    ) -> Operand:
        """An indicator operand, optionally restricted to one domain."""
        names = [
            name
            for name in self.indicators
            if domain is None or GENOME_INDICATORS[name].domain == domain
        ]
        if not names:
            raise StrategyError("no indicator in this library has that domain", domain=domain)
        name = str(rng.choice(names))
        signature = GENOME_INDICATORS[name]
        period = self.draw_period(rng, minimum=signature.min_period)
        return Operand(kind="indicator", name=name, kwargs={signature.period: period})

    def draw_price_operand(
        self, rng: np.random.Generator, *, domain: Domain | None = None
    ) -> Operand:
        """A bar column, optionally restricted to one domain.

        The restriction matters: ``volume`` is a bar column but is not measured
        in the same units as ``close``, so a draw that ignored the domain would
        pair a moving average against a trade count.
        """
        columns = sorted(
            name for name in PRICE_COLUMNS if domain is None or PRICE_DOMAINS[name] == domain
        )
        if not columns:
            raise StrategyError("no price column has that domain", domain=domain)
        return Operand(kind="price", name=str(rng.choice(columns)))

    def draw_constant(self, rng: np.random.Generator, domain: Domain) -> Operand:
        """A fixed level, drawn from the range that domain makes meaningful.

        Raises:
            StrategyError: the domain is price-scaled. A constant compared against
                a price is an asset-specific and year-specific number, and drawing
                one at random would produce a condition that is true on every bar
                or on none.
        """
        span = CONSTANT_RANGES.get(domain)
        if span is None:
            raise StrategyError("no constant is meaningful against this domain", domain=domain)
        low, high = span
        return Operand(kind="constant", kwargs={"value": float(rng.uniform(low, high))})

    # -- conditions ---------------------------------------------------------
    def draw_condition(
        self,
        rng: np.random.Generator,
        *,
        params: Mapping[str, ParamSpec] | None = None,
        avoid: Sequence[Condition] = (),
        crossings: bool = True,
    ) -> Condition:
        """One comparison between two operands measured in the same units.

        ``avoid`` names conditions already present: a tree may not repeat one
        (section 9.6 rule 5), so drawing a duplicate would be an immediate,
        avoidable invalidity.

        Raises:
            StrategyError: no distinct condition could be drawn. The caller
                treats this as the failed attempt it is and redraws or falls back
                (section 13.4, layers 2 and 3).
        """
        taken = {condition.key for condition in avoid}
        operators = sorted(COMPARISON_OPS | CROSS_OPS) if crossings else sorted(COMPARISON_OPS)
        for _ in range(32):
            left = self.draw_indicator_operand(rng)
            domain = left.domain
            assert domain is not None  # an indicator always has one
            right = self._draw_peer(rng, domain, params)
            if right is None or left.key == right.key:
                continue
            candidate = Condition(left=left, op=str(rng.choice(operators)), right=right)  # type: ignore[arg-type]
            if candidate.key not in taken:
                return candidate
        raise StrategyError("could not draw a condition distinct from those already present")

    def _draw_peer(
        self,
        rng: np.random.Generator,
        domain: Domain,
        params: Mapping[str, ParamSpec] | None,
    ) -> Operand | None:
        """Something comparable with ``domain``: an operand, a constant or a param."""
        choices: list[str] = ["indicator"]
        if domain in PRICE_DOMAINS.values():
            choices.append("price")
        if domain in CONSTANT_RANGES:
            choices.append("constant")
        usable = sorted(
            name
            for name, spec in (params or {}).items()
            if spec.kind in ("int", "float") and not _is_period_like(spec)
        )
        if usable:
            choices.append("param")

        pick = str(rng.choice(choices))
        if pick == "indicator":
            try:
                return self.draw_indicator_operand(rng, domain=domain)
            except StrategyError:
                return None
        if pick == "price":
            return self.draw_price_operand(rng, domain=domain)
        if pick == "constant":
            return self.draw_constant(rng, domain)
        return Operand(kind="param", name=str(rng.choice(usable)))

    def compatible_indicators(self, operand: Operand) -> tuple[str, ...]:
        """Indicators this one may be swapped for (``replace_indicator``, §13.4).

        Compatible means: in this library, measured in the same units, accepting
        the operand's current period, and not the indicator already there. Same
        units is what keeps the surrounding comparison meaningful — swapping an
        RSI for a moving average would silently turn a threshold condition into a
        price comparison.
        """
        if operand.kind != "indicator":
            return ()
        signature = GENOME_INDICATORS[operand.name]
        period = operand.kwargs[signature.period]
        return tuple(
            name
            for name in self.indicators
            if name != operand.name
            and GENOME_INDICATORS[name].domain == signature.domain
            and (isinstance(period, str) or GENOME_INDICATORS[name].min_period <= int(period))
        )

    # -- risk and sizing ----------------------------------------------------
    def draw_risk_value(self, rng: np.random.Generator, control: str) -> float | int:
        """A value for one risk control, inside the range :class:`RiskSpec` allows."""
        span = RISK_CONTROL_RANGES.get(control)
        if span is None:
            raise StrategyError("unknown risk control", control=control)
        low, high = span
        drawn = float(rng.uniform(low, high))
        return round(drawn) if control in _INTEGER_RISK_CONTROLS else drawn

    # -- whole genomes ------------------------------------------------------
    def draw_genome(
        self,
        rng: np.random.Generator,
        *,
        name: str = "immigrant",
        version: str = "1",
        max_conditions: int = 2,
        attempts: int = 8,
    ) -> StrategyGenome:
        """A fresh random genome — the immigrant of section 13.5.

        Warm-up is *derived* from the conditions drawn rather than searched. Rule
        6 makes it a lower bound, so the honest value is the smallest one the
        genome's own indicators permit; anything larger would be a free parameter
        that only ever costs bars.

        A draw can still miss — two trees drawn independently may name more
        distinct indicators than ``max_indicators`` allows — so the draw is
        retried. That is section 13.4's layer 2 (redraw) applied at the source
        rather than after the fact.

        Raises:
            StrategyError: ``attempts`` draws all produced an invalid genome. The
                caller falls back rather than relaxing a rule (section 13.4,
                layer 3).
        """
        entry_ceiling = max(1, min(max_conditions, self.limits.max_conditions_entry))
        exit_ceiling = max(1, min(max_conditions, self.limits.max_conditions_exit))
        for attempt in range(max(1, attempts)):
            # Back off across attempts. A wide draw is likely to name more
            # distinct indicators than `max_indicators` allows, and retrying at
            # the same width would keep hitting the same ceiling; narrowing
            # converges on a genome the limits actually admit.
            backoff = attempt // 2
            entry: list[Condition] = []
            exits: list[Condition] = []
            try:
                for _ in range(int(rng.integers(1, max(1, entry_ceiling - backoff) + 1))):
                    entry.append(self.draw_condition(rng, avoid=entry))
                for _ in range(int(rng.integers(1, max(1, exit_ceiling - backoff) + 1))):
                    exits.append(self.draw_condition(rng, avoid=exits))
            except StrategyError:
                continue
            if _distinct_indicators(entry + exits) > self.limits.max_indicators:
                continue
            try:
                return StrategyGenome(
                    name=name,
                    version=version,
                    entry=ConditionTree(
                        mode=str(rng.choice(["all", "any"])),  # type: ignore[arg-type]
                        conditions=tuple(entry),
                    ),
                    exit=ConditionTree(conditions=tuple(exits)),
                    risk=RiskSpec(),
                    sizing=SizingSpec(),
                    params={},
                    # Derived, not drawn: the smallest warm-up rule 6 permits.
                    warmup_bars=required_warmup(entry + exits),
                )
            except ValidationError:
                continue
        raise StrategyError(
            "could not draw a valid genome", attempts=attempts, max_conditions=max_conditions
        )


def required_warmup(
    conditions: Sequence[Condition], params: Mapping[str, ParamSpec] | None = None
) -> int:
    """The smallest ``warmup_bars`` section 9.6 rule 6 permits for ``conditions``.

    Exposed because mutation needs it too: a structural edit can lengthen the
    longest lookback, and raising warm-up to match is not "repairing by relaxing a
    rule" — rule 6 is a lower bound, and a longer warm-up is strictly more
    conservative.
    """
    schema = dict(params or {})
    return max(
        (operand.lookback(schema) for condition in conditions for operand in condition.operands),
        default=0,
    )


def _distinct_indicators(conditions: Sequence[Condition]) -> int:
    return len(
        {
            operand.key
            for condition in conditions
            for operand in condition.operands
            if operand.kind == "indicator"
        }
    )


def _is_period_like(spec: ParamSpec) -> bool:
    """True for a parameter that looks like a lookback rather than a level.

    A period parameter belongs in an indicator's ``kwargs``, not on the far side
    of a comparison: ``rsi(14) > slow_period`` compares an oscillator against a
    bar count. The heuristic is deliberately crude — integer-valued and bounded
    above 3 — because it only steers what the library *draws*; a genome may
    still be written by hand or by an LLM either way.
    """
    return spec.kind == "int" and (spec.high or 0.0) > 3.0
