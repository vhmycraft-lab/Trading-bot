"""The operator library (master spec section 13.4, task T48).

The library is where new genome material comes from — appended conditions,
swapped indicators, and the whole fresh genomes section 13.5 calls immigrants.
Its job is not merely to produce *valid* genomes, which section 9.6 already
guarantees, but *sensible* ones: a drawing rule that paired an RSI against a
Bitcoin price would fill the population with conditions that are true on every
bar or on none, and the search would spend its generations rediscovering that.

So the tests below are mostly about meaning rather than validity.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from quantlab.core.config import GenomeSettings
from quantlab.core.errors import StrategyError
from quantlab.core.genome import (
    GENOME_INDICATORS,
    PRICE_COLUMNS,
    Condition,
    Operand,
    StrategyGenome,
    genome_id,
)
from quantlab.core.strategy import ParamSpec
from quantlab.core.types import RiskSpec
from quantlab.evolution.compiler import compile_genome
from quantlab.evolution.library import (
    CONSTANT_RANGES,
    DEFAULT_PERIODS,
    RISK_CONTROL_RANGES,
    OperatorLibrary,
    required_warmup,
)
from quantlab.sandbox.ast_check import require_safe_source

LIBRARY = OperatorLibrary()


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------
def test_a_library_cannot_name_an_indicator_no_genome_can_use() -> None:
    with pytest.raises(StrategyError, match="no genome can use"):
        OperatorLibrary(indicators=("sma", "bbands"))


def test_a_library_needs_at_least_one_period() -> None:
    with pytest.raises(StrategyError, match="at least one lookback period"):
        OperatorLibrary(periods=())


def test_the_default_periods_are_a_ladder_not_a_range() -> None:
    """Uniform sampling would spend most draws distinguishing a 195-bar average
    from a 200-bar one, which the data cannot resolve."""
    gaps = [b - a for a, b in itertools.pairwise(DEFAULT_PERIODS)]
    assert gaps == sorted(gaps)
    assert min(DEFAULT_PERIODS) >= 2


# ---------------------------------------------------------------------------
# operands
# ---------------------------------------------------------------------------
def test_a_drawn_indicator_always_gets_a_period_it_accepts() -> None:
    """``zscore`` and ``rolling_vol`` need two bars; a one-bar draw would be
    rejected by section 9.6 for a reason the library could have avoided."""
    for seed in range(200):
        operand = LIBRARY.draw_indicator_operand(rng(seed))
        signature = GENOME_INDICATORS[operand.name]
        assert int(operand.kwargs[signature.period]) >= signature.min_period


def test_an_indicator_can_be_drawn_from_one_domain() -> None:
    for seed in range(50):
        assert LIBRARY.draw_indicator_operand(rng(seed), domain="price").domain == "price"


def test_asking_for_a_domain_the_library_has_none_of_is_refused() -> None:
    narrow = OperatorLibrary(indicators=("sma", "ema"))
    with pytest.raises(StrategyError, match="no indicator in this library"):
        narrow.draw_indicator_operand(rng(), domain="oscillator")


def test_a_price_operand_is_a_tradable_column() -> None:
    for seed in range(30):
        assert LIBRARY.draw_price_operand(rng(seed)).name in PRICE_COLUMNS


def test_a_price_operand_can_be_restricted_to_one_domain() -> None:
    """``volume`` is a bar column but is not a price; pairing a moving average
    against a trade count is exactly the nonsense the domains exist to stop."""
    for seed in range(30):
        assert LIBRARY.draw_price_operand(rng(seed), domain="price").name != "volume"
    assert LIBRARY.draw_price_operand(rng(), domain="volume").name == "volume"


def test_a_constant_is_drawn_inside_the_range_its_domain_makes_meaningful() -> None:
    for domain, (low, high) in CONSTANT_RANGES.items():
        for seed in range(20):
            assert low <= LIBRARY.draw_constant(rng(seed), domain).value <= high


def test_no_constant_is_drawn_against_a_price() -> None:
    """A fixed number compared with a price is asset-specific and year-specific;
    drawing one at random would produce a condition with a constant answer."""
    for domain in ("price", "range", "volume"):
        with pytest.raises(StrategyError, match="no constant is meaningful"):
            LIBRARY.draw_constant(rng(), domain)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# conditions
# ---------------------------------------------------------------------------
def test_a_drawn_condition_compares_like_with_like() -> None:
    """The library's central claim, over two hundred draws."""
    for seed in range(200):
        condition = LIBRARY.draw_condition(rng(seed))
        left, right = condition.left.domain, condition.right.domain
        if left is not None and right is not None:
            assert left == right


def test_a_drawn_condition_never_repeats_one_already_present() -> None:
    """Section 9.6 rule 5 refuses a duplicate, so drawing one would be an
    avoidable invalidity rather than bad luck."""
    existing = [LIBRARY.draw_condition(rng(1))]
    for seed in range(60):
        drawn = LIBRARY.draw_condition(rng(seed), avoid=existing)
        assert drawn.key != existing[0].key


def test_crossings_can_be_excluded() -> None:
    for seed in range(40):
        assert LIBRARY.draw_condition(rng(seed), crossings=False).op in ("<", "<=", ">", ">=")


def test_a_parameter_may_stand_on_one_side_of_a_comparison() -> None:
    """A level parameter is a legitimate operand; a lookback parameter is not —
    ``rsi(14) > slow_period`` compares an oscillator against a bar count."""
    params = {
        "level": ParamSpec(kind="float", default=1.0, low=0.0, high=3.0),
        "slow": ParamSpec(kind="int", default=50, low=20, high=200),
    }
    named = set()
    for seed in range(200):
        condition = LIBRARY.draw_condition(rng(seed), params=params)
        named |= {operand.name for operand in condition.operands if operand.kind == "param"}
    assert "slow" not in named


def test_a_condition_that_cannot_be_drawn_raises_rather_than_looping() -> None:
    """The caller treats this as the failed attempt it is and redraws or falls
    back (section 13.4, layers 2 and 3)."""
    narrow = OperatorLibrary(indicators=("sma",), periods=(20,))
    existing = [narrow.draw_condition(rng(seed)) for seed in range(6)]
    with pytest.raises(StrategyError, match="could not draw a condition"):
        for seed in range(40):
            existing.append(narrow.draw_condition(rng(seed), avoid=existing))


# ---------------------------------------------------------------------------
# replace_indicator compatibility
# ---------------------------------------------------------------------------
def test_a_compatible_indicator_is_measured_in_the_same_units() -> None:
    operand = Operand(kind="indicator", name="sma", kwargs={"n": 20})
    options = LIBRARY.compatible_indicators(operand)
    assert "sma" not in options
    assert set(options) == {"ema", "highest", "lowest"}
    assert "rsi" not in options


def test_compatibility_respects_the_minimum_period() -> None:
    """``rolling_vol`` needs two bars, so it cannot replace a one-bar operand."""
    short = Operand(kind="indicator", name="returns", kwargs={"n": 1})
    assert "rolling_vol" not in LIBRARY.compatible_indicators(short)


def test_a_parameter_driven_period_stays_compatible() -> None:
    """The period is not known at compile time, so every same-domain indicator
    remains a candidate; section 9.6's warm-up rule catches an impossible one."""
    operand = Operand(kind="indicator", name="sma", kwargs={"n": "fast"})
    assert set(LIBRARY.compatible_indicators(operand)) == {"ema", "highest", "lowest"}


def test_only_an_indicator_has_compatible_replacements() -> None:
    assert LIBRARY.compatible_indicators(Operand(kind="price", name="close")) == ()


# ---------------------------------------------------------------------------
# risk controls
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("control", sorted(RISK_CONTROL_RANGES))
def test_a_drawn_risk_value_is_one_riskspec_accepts(control: str) -> None:
    """Every bound matches the one :class:`RiskSpec` enforces, so a drawn value
    is always constructible — the mutation never wastes a redraw on it."""
    for seed in range(40):
        value = LIBRARY.draw_risk_value(rng(seed), control)
        RiskSpec.model_validate({control: value})
        if control == "time_stop_bars":
            assert isinstance(value, int)


def test_an_unknown_risk_control_is_refused() -> None:
    with pytest.raises(StrategyError, match="unknown risk control"):
        LIBRARY.draw_risk_value(rng(), "max_loss_streak")


# ---------------------------------------------------------------------------
# whole genomes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_a_drawn_genome_is_valid_and_compiles(width: int) -> None:
    """Immigrants are drawn every generation; a draw that failed would silently
    shrink the population section 13.5 holds at a fixed size."""
    for seed in range(50):
        genome = LIBRARY.draw_genome(rng(seed), name=f"immigrant_{seed}", max_conditions=width)
        StrategyGenome.model_validate(genome.model_dump())
        require_safe_source(compile_genome(genome))


def test_a_drawn_genome_derives_its_warm_up_rather_than_guessing() -> None:
    """Rule 6 is a lower bound, so the honest value is the smallest one the
    genome's own indicators permit; a larger one only costs bars."""
    for seed in range(60):
        genome = LIBRARY.draw_genome(rng(seed))
        assert genome.warmup_bars == genome.max_lookback()


def test_drawn_genomes_differ() -> None:
    ids = {genome_id(LIBRARY.draw_genome(rng(seed))) for seed in range(40)}
    assert len(ids) > 30


def test_the_same_seed_draws_the_same_genome() -> None:
    assert LIBRARY.draw_genome(rng(7)).canonical() == LIBRARY.draw_genome(rng(7)).canonical()


def test_a_drawn_genome_respects_the_configured_limits() -> None:
    tight = OperatorLibrary(limits=GenomeSettings(max_indicators=4, max_conditions_entry=1))
    for seed in range(40):
        genome = tight.draw_genome(rng(seed), max_conditions=1)
        assert len(genome.indicator_operands()) <= 4
        assert len(genome.entry) == 1


def test_a_draw_that_cannot_succeed_raises_rather_than_returning_something_invalid() -> None:
    """Section 13.4's layer 3: the caller falls back, never relaxes a rule."""
    impossible = OperatorLibrary(limits=GenomeSettings(max_indicators=1), periods=(20, 50))
    with pytest.raises(StrategyError, match="could not draw a valid genome"):
        impossible.draw_genome(rng(0), max_conditions=4, attempts=2)


# ---------------------------------------------------------------------------
# required_warmup
# ---------------------------------------------------------------------------
def test_required_warmup_is_the_largest_lookback_any_operand_implies() -> None:
    condition = Condition(
        left=Operand(kind="indicator", name="sma", kwargs={"n": 50}),
        op=">",
        right=Operand(kind="indicator", name="rsi", kwargs={"n": 120}),
    )
    assert required_warmup([condition]) == 121  # rsi needs n + 1


def test_required_warmup_costs_a_parameter_at_its_upper_bound() -> None:
    condition = Condition(
        left=Operand(kind="indicator", name="sma", kwargs={"n": "slow"}),
        op=">",
        right=Operand(kind="price", name="close"),
    )
    params = {"slow": ParamSpec(kind="int", default=50, low=20, high=400)}
    assert required_warmup([condition], params) == 400


def test_required_warmup_of_nothing_is_zero() -> None:
    assert required_warmup([]) == 0
