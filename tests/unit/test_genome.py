"""The genome's validity rules (master spec section 9.6, task T46).

Section 9.6 lists seven rules and says a genome that breaks one "cannot be
constructed". That phrasing is the whole design: mutation (section 13.4) edits
genomes rather than source precisely so that an invalid candidate is impossible
rather than merely unlikely, so every rule below is tested at the constructor.

The indicator table is tested against ``core/indicators.py`` itself rather than
against a second copy of the numbers: a warm-up rule computed from a table that
had drifted would be a rule about nothing.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from quantlab.core.config import GenomeSettings
from quantlab.core.genome import (
    GENOME_INDICATORS,
    MULTI_OUTPUT_INDICATORS,
    Condition,
    ConditionTree,
    Operand,
    StrategyGenome,
    genome_id,
)
from quantlab.core.indicators import BBands, Donchian, Macd
from quantlab.core.strategy import INDICATORS, ParamSpec
from quantlab.core.types import RiskSpec, SizingSpec

# --- builders ---------------------------------------------------------------
FAST = Operand(kind="indicator", name="sma", kwargs={"n": "fast"})
SLOW = Operand(kind="indicator", name="sma", kwargs={"n": "slow"})
CLOSE = Operand(kind="price", name="close")
THIRTY = Operand(kind="constant", kwargs={"value": 30.0})

PARAMS = {
    "fast": ParamSpec(kind="int", default=50, low=5, high=200),
    "slow": ParamSpec(kind="int", default=200, low=20, high=400),
}


def crossover(**overrides: object) -> StrategyGenome:
    """A valid two-moving-average genome, with fields replaced for a negative test."""
    fields: dict[str, object] = {
        "name": "crossover",
        "entry": ConditionTree(conditions=(Condition(left=FAST, op=">", right=SLOW),)),
        "exit": ConditionTree(conditions=(Condition(left=FAST, op="<", right=SLOW),)),
        "params": dict(PARAMS),
        "warmup_bars": 400,
    }
    fields.update(overrides)
    return StrategyGenome(**fields)  # type: ignore[arg-type]


def refuses(match: str, **overrides: object) -> None:
    with pytest.raises(ValidationError, match=match):
        crossover(**overrides)


# --- the tables are honest --------------------------------------------------
def test_every_indicator_is_either_usable_or_excluded_by_name() -> None:
    """Adding an indicator to core must force a decision here, not be forgotten."""
    assert set(GENOME_INDICATORS) | MULTI_OUTPUT_INDICATORS == set(INDICATORS)
    assert not set(GENOME_INDICATORS) & MULTI_OUTPUT_INDICATORS


def test_the_excluded_indicators_are_exactly_the_multi_valued_ones() -> None:
    """The exclusion is a fact about their return type, not a preference."""
    values = np.linspace(100.0, 200.0, 120) + np.sin(np.arange(120))
    computed = {
        "bbands": INDICATORS["bbands"](values, 20),
        "donchian": INDICATORS["donchian"](values + 1, values - 1, 20),
        "macd": INDICATORS["macd"](values),
    }
    assert set(computed) == set(MULTI_OUTPUT_INDICATORS)
    for name, result in computed.items():
        assert isinstance(result, (BBands, Donchian, Macd)), name

    for name in GENOME_INDICATORS:
        single = INDICATORS[name](*([values] * _n_inputs(name)), 14)
        assert isinstance(single, np.ndarray), name


def _n_inputs(name: str) -> int:
    return 3 if name == "atr" else 1


@pytest.mark.parametrize("name", sorted(GENOME_INDICATORS))
@pytest.mark.parametrize("period", [5, 14, 30])
def test_the_declared_lookback_is_the_real_first_defined_bar(name: str, period: int) -> None:
    """Rule 6 costs warm-up from this table, so the table must match the code.

    ``lookback`` is defined as "bars needed before a number appears", i.e. one
    more than the index of the first non-NaN value.
    """
    signature = GENOME_INDICATORS[name]
    if period < signature.min_period:
        pytest.skip(f"{name} needs a period of at least {signature.min_period}")
    values = np.linspace(100.0, 300.0, 400) + np.sin(np.arange(400))
    computed = INDICATORS[name](*([values] * _n_inputs(name)), period)
    first_defined = int(np.flatnonzero(~np.isnan(computed))[0])
    assert signature.lookback(period) == first_defined + 1


# --- operands ---------------------------------------------------------------
def test_an_operand_may_not_name_an_indicator_that_does_not_exist() -> None:
    with pytest.raises(ValidationError, match="unknown indicator"):
        Operand(kind="indicator", name="hilbert_transform", kwargs={"n": 14})


@pytest.mark.parametrize("name", sorted(MULTI_OUTPUT_INDICATORS))
def test_a_multi_valued_indicator_is_refused_with_its_own_reason(name: str) -> None:
    """Not "unknown" — it exists, and an operand has no field to select a series."""
    with pytest.raises(ValidationError, match="which one it means"):
        Operand(kind="indicator", name=name, kwargs={"n": 20})


def test_an_indicator_operand_takes_exactly_its_period() -> None:
    with pytest.raises(ValidationError, match="exactly one argument"):
        Operand(kind="indicator", name="sma", kwargs={"n": 14, "k": 2})
    with pytest.raises(ValidationError, match="exactly one argument"):
        Operand(kind="indicator", name="sma", kwargs={})


def test_a_period_is_a_whole_number_of_bars_or_a_parameter_name() -> None:
    assert Operand(kind="indicator", name="sma", kwargs={"n": 14}).period_param is None
    assert Operand(kind="indicator", name="sma", kwargs={"n": "fast"}).period_param == "fast"
    with pytest.raises(ValidationError, match="whole number of bars"):
        Operand(kind="indicator", name="sma", kwargs={"n": 14.5})


def test_a_period_below_the_indicator_s_minimum_is_refused() -> None:
    """``zscore`` and ``rolling_vol`` need two bars to have any dispersion at all."""
    with pytest.raises(ValidationError, match="at least 2"):
        Operand(kind="indicator", name="zscore", kwargs={"n": 1})


def test_a_price_operand_names_a_tradable_column() -> None:
    assert Operand(kind="price", name="close").name == "close"
    with pytest.raises(ValidationError, match="unknown price column"):
        Operand(kind="price", name="ts_open")
    with pytest.raises(ValidationError, match="unknown price column"):
        Operand(kind="price", name="is_gap_filled")


def test_a_constant_carries_one_number_and_no_name() -> None:
    assert Operand(kind="constant", kwargs={"value": 30}).value == 30.0
    with pytest.raises(ValidationError, match="not in name"):
        Operand(kind="constant", name="thirty", kwargs={"value": 30})
    with pytest.raises(ValidationError, match="exactly one numeric kwarg"):
        Operand(kind="constant", kwargs={"value": "thirty"})
    with pytest.raises(ValidationError, match="a boolean is not a number"):
        Operand(kind="constant", kwargs={"value": True})


def test_only_a_constant_has_a_value() -> None:
    with pytest.raises(ValueError, match="only a constant"):
        _ = CLOSE.value


def test_two_spellings_of_one_operand_share_a_key() -> None:
    """The key is what deduplicates operands into slots, so it must ignore order."""
    left = Operand(kind="indicator", name="sma", kwargs={"n": 14})
    right = Operand(kind="indicator", name="sma", kwargs={"n": 14})
    assert left.key == right.key
    assert left.key != SLOW.key


# --- conditions -------------------------------------------------------------
def test_a_condition_may_not_compare_an_operand_to_itself() -> None:
    """Rule 5: the answer is fixed, so it is a slot spent on nothing."""
    with pytest.raises(ValidationError, match="compares an operand to itself"):
        Condition(left=FAST, op=">", right=FAST)


def test_a_condition_between_two_constants_is_refused() -> None:
    with pytest.raises(ValidationError, match="same answer on every bar"):
        Condition(
            left=Operand(kind="constant", kwargs={"value": 1}),
            op="<",
            right=Operand(kind="constant", kwargs={"value": 2}),
        )


def test_a_tree_may_not_repeat_a_condition() -> None:
    condition = Condition(left=FAST, op=">", right=SLOW)
    with pytest.raises(ValidationError, match="repeats a condition"):
        ConditionTree(conditions=(condition, condition))


# --- genome rules -----------------------------------------------------------
def test_the_reference_genome_is_valid() -> None:
    """Guards every negative test below against passing for the wrong reason."""
    genome = crossover()
    assert genome.max_lookback() == 400
    assert genome.param_references() == {"fast", "slow"}
    assert [operand.key for operand in genome.indicator_operands()] == [FAST.key, SLOW.key]


def test_a_genome_needs_at_least_one_entry_condition() -> None:
    """Rule 4: without one it never takes a position and cannot be evaluated."""
    refuses("never takes a position", entry=ConditionTree())


def test_an_exit_tree_may_be_empty() -> None:
    """A genome may leave every exit to the risk controls of section 8.6."""
    genome = crossover(exit=ConditionTree(), risk=RiskSpec(stop_loss_pct=0.05))
    assert len(genome.exit) == 0


def test_an_operand_may_not_name_an_undeclared_parameter() -> None:
    """Rule 2, first half."""
    refuses("not declared", params={"fast": PARAMS["fast"]})


def test_a_declared_parameter_must_be_used() -> None:
    """Rule 2, second half: an unused parameter inflates the free-parameter count."""
    refuses(
        "never used",
        params={**PARAMS, "spare": ParamSpec(kind="float", default=1.0, low=0.5, high=2.0)},
    )


def test_a_genome_parameter_must_be_numeric() -> None:
    """Every parameter ends up on one side of a comparison."""
    refuses(
        "on one side of a numeric comparison",
        entry=ConditionTree(
            conditions=(
                Condition(left=FAST, op=">", right=SLOW),
                Condition(left=CLOSE, op=">", right=Operand(kind="param", name="mode")),
            )
        ),
        params={**PARAMS, "mode": ParamSpec(kind="bool", default=True)},
    )


def test_a_parameter_driving_a_lookback_must_be_an_integer() -> None:
    """A period is a count of bars; a float one would be silently truncated."""
    refuses(
        "must be an int",
        params={
            "fast": ParamSpec(kind="float", default=50.0, low=5.0, high=200.0),
            "slow": PARAMS["slow"],
        },
        warmup_bars=400,
    )


def test_warmup_must_cover_the_largest_lookback() -> None:
    """Rule 6: trading before an indicator is defined is trading on NaN."""
    refuses("trading on NaN", warmup_bars=399)


def test_warmup_is_costed_at_the_parameter_s_upper_bound() -> None:
    """The optimiser may set the period anywhere in its range, so warm-up must
    cover the whole range rather than the default."""
    genome = crossover()
    assert genome.params["slow"].high == 400
    assert genome.params["slow"].default == 200
    assert genome.max_lookback() == 400


def test_a_fixed_period_is_costed_from_the_literal() -> None:
    rsi = Operand(kind="indicator", name="rsi", kwargs={"n": 14})
    genome = StrategyGenome(
        name="rsi_reversion",
        entry=ConditionTree(conditions=(Condition(left=rsi, op="<", right=THIRTY),)),
        params={},
        warmup_bars=15,
    )
    assert genome.max_lookback() == 15  # rsi needs n + 1


def test_the_condition_counts_respect_the_configured_limits() -> None:
    """Rule 4's ceilings come from ``evolution.genome`` (section 9.6)."""
    conditions = tuple(
        Condition(
            left=Operand(kind="indicator", name="sma", kwargs={"n": n}),
            op=">",
            right=CLOSE,
        )
        for n in (2, 3, 4, 5, 6)
    )
    with pytest.raises(ValidationError, match="too many entry"):
        StrategyGenome(
            name="wide", entry=ConditionTree(conditions=conditions), warmup_bars=10, params={}
        )


def test_a_tightened_limit_can_be_supplied_as_validation_context() -> None:
    """A campaign may narrow the structure; the defaults still apply without one."""
    data = crossover().model_dump(mode="json")
    assert StrategyGenome.model_validate(data).name == "crossover"
    with pytest.raises(ValidationError, match="too many indicators"):
        StrategyGenome.model_validate(data, context={"limits": GenomeSettings(max_indicators=1)})


def test_a_nonsense_context_falls_back_to_the_defaults() -> None:
    """A caller that passes the wrong thing gets the shipped limits, not no limits."""
    data = crossover().model_dump(mode="json")
    assert StrategyGenome.model_validate(data, context={"limits": "strict"}).name == "crossover"
    assert StrategyGenome.model_validate(data, context=["nonsense"]).name == "crossover"


def test_too_many_parameters_are_refused_at_the_section_9_2_ceiling() -> None:
    """Rule 7: ``max_free_params + 4``, matching the load-time limit."""
    limit = GenomeSettings().max_free_params + 4
    names = [f"p{i}" for i in range(limit + 1)]
    params = {name: ParamSpec(kind="float", default=1.0, low=0.0, high=2.0) for name in names}

    # Two parameters per condition, so the condition ceilings of rule 4 are not
    # what this test trips on.
    def pair(left: str, right: str) -> Condition:
        return Condition(
            left=Operand(kind="param", name=left), op=">", right=Operand(kind="param", name=right)
        )

    pairs = [pair(names[i], names[i + 1]) for i in range(0, len(names) - 1, 2)]
    last = Condition(left=CLOSE, op=">", right=Operand(kind="param", name=names[-1]))
    with pytest.raises(ValidationError, match="too many parameters"):
        StrategyGenome(
            name="inflated",
            entry=ConditionTree(conditions=tuple(pairs[:4])),
            exit=ConditionTree(conditions=(*pairs[4:], last)),
            params=params,
            warmup_bars=0,
        )


def test_a_genome_name_must_be_usable_as_a_class_name() -> None:
    refuses("part of a class name", name="not a name")


def test_a_genome_is_frozen_and_forbids_unknown_fields() -> None:
    genome = crossover()
    with pytest.raises(ValidationError):
        genome.warmup_bars = 10  # type: ignore[misc]
    with pytest.raises(ValidationError):
        crossover(edge="secret")


# --- identity ---------------------------------------------------------------
def test_the_genome_id_is_stable_and_structural() -> None:
    assert genome_id(crossover()) == genome_id(crossover())
    assert len(genome_id(crossover())) == 16
    assert genome_id(crossover()) != genome_id(crossover(version="2"))


def test_the_genome_id_ignores_how_the_genome_was_spelled() -> None:
    """Two objects that are the same structure are the same candidate."""
    other = crossover(params={"slow": PARAMS["slow"], "fast": PARAMS["fast"]})
    assert genome_id(other) == genome_id(crossover())


def test_the_genome_id_changes_when_the_structure_does() -> None:
    ids = {
        genome_id(crossover()),
        genome_id(crossover(warmup_bars=401)),
        genome_id(crossover(sizing=SizingSpec(fraction=0.5))),
        genome_id(crossover(risk=RiskSpec(stop_loss_pct=0.02))),
        genome_id(
            crossover(entry=ConditionTree(conditions=(Condition(left=FAST, op=">=", right=SLOW),)))
        ),
    }
    assert len(ids) == 5


def test_the_canonical_form_round_trips() -> None:
    genome = crossover()
    import json

    assert StrategyGenome.model_validate(json.loads(genome.canonical())) == genome
