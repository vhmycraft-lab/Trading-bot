"""Mutation operators and their four safety layers (spec section 13.4, task T48).

Section 22's acceptance criteria are the spine of this file: twenty seeded
mutations of a valid genome are all valid; a deliberately impossible operator
falls back to an immigrant after ``max_repair_attempts``; identical seeds produce
identical children; and an ``opaque`` candidate never receives a structural
mutation.

INV-10 is tested here too, ahead of the store that will persist it: re-applying a
child's recorded mutations to its parent must reproduce the child exactly. That
property is what makes a lineage a record rather than a story, and it is cheapest
to establish where the mutations are produced.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from quantlab.core.config import (
    MutationParameterSettings,
    MutationSettings,
    MutationStructuralSettings,
)
from quantlab.core.errors import StrategyError
from quantlab.core.genome import (
    Condition,
    ConditionTree,
    Operand,
    StrategyGenome,
    genome_id,
)
from quantlab.core.strategy import ParamSpec
from quantlab.core.types import RiskSpec, SizingSpec
from quantlab.evolution.compiler import compile_genome
from quantlab.evolution.library import OperatorLibrary
from quantlab.evolution.mutation import (
    PARAMETER_OPERATORS,
    STRUCTURAL_OPERATORS,
    Phenotype,
    mutate,
    replay,
)
from quantlab.sandbox.ast_check import require_safe_source

FAST = Operand(kind="indicator", name="sma", kwargs={"n": "fast"})
SLOW = Operand(kind="indicator", name="sma", kwargs={"n": "slow"})
RSI = Operand(kind="indicator", name="rsi", kwargs={"n": 14})
FIFTY = Operand(kind="constant", kwargs={"value": 50.0})

SETTINGS = MutationSettings()
LIBRARY = OperatorLibrary()


def parent_genome(**overrides: object) -> StrategyGenome:
    """A valid two-condition crossover, wide enough for every structural operator."""
    fields: dict[str, object] = {
        "name": "crossover",
        "entry": ConditionTree(
            conditions=(
                Condition(left=FAST, op=">", right=SLOW),
                Condition(left=RSI, op="<", right=FIFTY),
            )
        ),
        "exit": ConditionTree(conditions=(Condition(left=FAST, op="<=", right=SLOW),)),
        "params": {
            "fast": ParamSpec(kind="int", default=10, low=5, high=50),
            "slow": ParamSpec(kind="int", default=30, low=20, high=100),
        },
        "warmup_bars": 100,
    }
    fields.update(overrides)
    return StrategyGenome(**fields)  # type: ignore[arg-type]


def parent() -> Phenotype:
    return Phenotype.from_genome(parent_genome())


def children(n: int, *, start: int = 0, phenotype: Phenotype | None = None) -> list[Phenotype]:
    """Every child produced by seeds ``start .. start + n``, fallbacks skipped."""
    source = phenotype or parent()
    out = []
    for seed in range(start, start + n):
        result = mutate(source, SETTINGS, np.random.default_rng(seed), LIBRARY)
        if result.child is not None:
            out.append(result.child)
    return out


# ---------------------------------------------------------------------------
# validity by construction (layer 1)
# ---------------------------------------------------------------------------
def test_twenty_seeded_mutations_are_all_valid() -> None:
    """Section 22's first criterion, extended to what "valid" has to mean: the
    genome constructs, *and* the source it compiles to passes the section 9.2
    check that would otherwise reject it much later."""
    produced = children(20)
    assert len(produced) >= 18, "the fallback path should be rare on a healthy parent"
    for child in produced:
        assert child.genome is not None
        StrategyGenome.model_validate(child.genome.model_dump())
        require_safe_source(compile_genome(child.genome))


def test_a_child_differs_from_its_parent() -> None:
    """A mutation that changed nothing is a wasted evaluation, and the run cache
    of section 11.2 would serve the parent's numbers for it."""
    original = parent()
    for child in children(20):
        assert (child.genome, dict(child.params), child.risk, child.sizing) != (
            original.genome,
            dict(original.params),
            original.risk,
            original.sizing,
        )


def test_the_warm_up_follows_a_structural_edit() -> None:
    """A new condition can name a longer lookback than the parent's warm-up
    covers. Raising warm-up is not relaxing rule 6 — it is a lower bound, and a
    longer warm-up is strictly more conservative."""
    for child in children(40):
        assert child.genome is not None
        assert child.genome.warmup_bars >= child.genome.max_lookback()


# ---------------------------------------------------------------------------
# determinism (INV-7) and replay (INV-10)
# ---------------------------------------------------------------------------
def test_identical_seeds_produce_identical_children() -> None:
    """Section 22's third criterion. A generation must be reproducible from its
    seed, or nothing downstream of it is."""
    for seed in range(10):
        first = mutate(parent(), SETTINGS, np.random.default_rng(seed), LIBRARY)
        second = mutate(parent(), SETTINGS, np.random.default_rng(seed), LIBRARY)
        assert first.genome == second.genome
        assert first.mutations == second.mutations
        assert dict(first.failures) == dict(second.failures)


def test_different_seeds_explore() -> None:
    """Identity is ``genome_id``, not object identity: a genome holds dicts and
    is deliberately not hashable, so the id is what says two children are one."""
    genomes = {genome_id(child.genome) for child in children(30) if child.genome}
    assert len(genomes) > 20, "seeds should not collapse onto a handful of children"


def test_replaying_the_record_reproduces_the_child_exactly() -> None:
    """INV-10. Each record carries the value at its path *after* the edit, so
    replay is a pure application and needs none of the original randomness."""
    original = parent()
    for seed in range(60):
        result = mutate(original, SETTINGS, np.random.default_rng(seed), LIBRARY)
        if result.child is None:
            continue
        rebuilt = replay(original, result.mutations)
        assert rebuilt.genome == result.child.genome
        assert dict(rebuilt.params) == dict(result.child.params)
        assert rebuilt.risk == result.child.risk
        assert rebuilt.sizing == result.child.sizing


def test_replay_refuses_a_path_the_parent_does_not_have() -> None:
    """A corrupt lineage is not a mutation that failed; it must not be skipped."""
    result = mutate(parent(), SETTINGS, np.random.default_rng(0), LIBRARY)
    broken = result.mutations[0].__class__(
        category="structural",
        operator="modify_entry",
        target="nowhere.conditions[0].op",
        before_json="null",
        after_json='"<"',
        rng_seed=1,
    )
    with pytest.raises(StrategyError, match="unknown genome path"):
        replay(parent(), [broken])


# ---------------------------------------------------------------------------
# redraw and fall back (layers 2 and 3)
# ---------------------------------------------------------------------------
def test_an_impossible_operator_falls_back_after_the_repair_budget() -> None:
    """Section 22's second criterion.

    Every structural weight but ``remove_filter`` is zeroed, and the parent has no
    filter, so the only reachable operator can never succeed. After
    ``max_repair_attempts`` redraws the slot falls back rather than being patched
    into validity — which is section 13.4's layer 3, stated as a test.
    """
    impossible = MutationSettings(
        parameter_rate=0.0,
        structural_rate=1.0,
        max_repair_attempts=2,
        structural=MutationStructuralSettings(
            add_confirmation=0.0,
            remove_confirmation=0.0,
            modify_entry=0.0,
            modify_exit=0.0,
            add_filter=0.0,
            remove_filter=1.0,
            replace_indicator=0.0,
            change_tree_mode=0.0,
        ),
    )
    result = mutate(parent(), impossible, np.random.default_rng(0), LIBRARY)
    assert result.fell_back
    assert result.child is None
    assert result.attempts == impossible.max_repair_attempts + 1
    assert result.failures == {"remove_filter": 3}


def test_failures_are_counted_per_operator_for_the_generation_stats() -> None:
    """Section 13.4 records redraws in ``generation.stats_json``; they cannot be
    reported if nobody counts them."""
    counted: Counter[str] = Counter()
    for seed in range(40):
        counted.update(mutate(parent(), SETTINGS, np.random.default_rng(seed), LIBRARY).failures)
    assert counted, "a healthy parent still meets impossible draws sometimes"
    assert set(counted) <= set(PARAMETER_OPERATORS) | set(STRUCTURAL_OPERATORS)


def test_a_redraw_still_produces_a_recorded_child() -> None:
    """A failed attempt must not leave a half-applied edit behind."""
    for seed in range(40):
        result = mutate(parent(), SETTINGS, np.random.default_rng(seed), LIBRARY)
        if result.child is None:
            assert result.mutations == () or result.attempts > len(result.mutations)
        else:
            assert 1 <= len(result.mutations) <= SETTINGS.max_mutations_per_child


# ---------------------------------------------------------------------------
# opaque candidates
# ---------------------------------------------------------------------------
def opaque() -> Phenotype:
    """A free-form Python candidate: parameters, but no editable structure."""
    return Phenotype(
        kind="opaque",
        params={"lookback": 20, "threshold": 1.5, "invert": False},
        schema={
            "lookback": ParamSpec(kind="int", default=20, low=5, high=100),
            "threshold": ParamSpec(kind="float", default=1.5, low=0.5, high=3.0),
            "invert": ParamSpec(kind="bool", default=False),
        },
    )


def test_an_opaque_candidate_never_receives_a_structural_mutation() -> None:
    """Section 22's fourth criterion, and section 13.4's rule: a structural
    operator drawn for an opaque candidate is re-drawn as a parameter one."""
    structural_only = MutationSettings(parameter_rate=0.0, structural_rate=1.0)
    for seed in range(40):
        result = mutate(opaque(), structural_only, np.random.default_rng(seed), LIBRARY)
        assert result.child is not None
        assert result.child.genome is None
        for mutation in result.mutations:
            assert mutation.category == "parameter"
            assert mutation.operator in PARAMETER_OPERATORS


def test_an_opaque_candidate_still_mutates_its_parameters() -> None:
    changed = 0
    for seed in range(20):
        result = mutate(opaque(), SETTINGS, np.random.default_rng(seed), LIBRARY)
        if result.child is not None and dict(result.child.params) != dict(opaque().params):
            changed += 1
    assert changed >= 15


def test_an_opaque_candidate_may_not_carry_a_genome() -> None:
    with pytest.raises(StrategyError, match="no genome to carry"):
        Phenotype(kind="opaque", genome=parent_genome())
    with pytest.raises(StrategyError, match="must carry a genome"):
        Phenotype(kind="genome", genome=None)


def test_a_bool_parameter_is_only_toggled_where_one_exists() -> None:
    """``toggle_bool`` is drawn for the opaque candidate, which has a bool, and
    never for the genome candidate, whose parameters are numeric by construction
    (section 9.6). Drawing it there would spend the repair budget on the
    impossible."""
    genome_ops = {
        m.operator
        for seed in range(60)
        for m in mutate(parent(), SETTINGS, np.random.default_rng(seed), LIBRARY).mutations
    }
    assert "toggle_bool" not in genome_ops
    assert "resample_categorical" not in genome_ops

    opaque_ops = {
        m.operator
        for seed in range(60)
        for m in mutate(opaque(), SETTINGS, np.random.default_rng(seed), LIBRARY).mutations
    }
    assert "toggle_bool" in opaque_ops


# ---------------------------------------------------------------------------
# individual operators
# ---------------------------------------------------------------------------
def only(operator: str, **overrides: object) -> MutationSettings:
    """Settings that can draw exactly one structural operator."""
    weights = dict.fromkeys(STRUCTURAL_OPERATORS, 0.0)
    weights[operator] = 1.0
    fields: dict[str, object] = {
        "parameter_rate": 0.0,
        "structural_rate": 1.0,
        "max_mutations_per_child": 1,
        "structural": MutationStructuralSettings(**weights),  # type: ignore[arg-type]
    }
    fields.update(overrides)
    return MutationSettings(**fields)  # type: ignore[arg-type]


def first_success(operator: str, phenotype: Phenotype | None = None, seeds: int = 60):  # type: ignore[no-untyped-def]
    settings = only(operator)
    for seed in range(seeds):
        result = mutate(phenotype or parent(), settings, np.random.default_rng(seed), LIBRARY)
        if result.child is not None:
            return result
    raise AssertionError(f"{operator} never succeeded in {seeds} seeds")


def test_add_confirmation_appends_to_a_tree() -> None:
    result = first_success("add_confirmation")
    mutation = result.mutations[0]
    assert mutation.category == "structural"
    assert mutation.before_json == "null"
    assert result.genome is not None
    total = len(result.genome.entry) + len(result.genome.exit)
    assert total == len(parent_genome().entry) + len(parent_genome().exit) + 1


def test_remove_confirmation_never_empties_the_entry_tree() -> None:
    """Section 13.4 says so explicitly: a genome with no entry condition never
    takes a position, and section 9.6 refuses to construct one."""
    single_entry = Phenotype.from_genome(
        parent_genome(entry=ConditionTree(conditions=(Condition(left=FAST, op=">", right=SLOW),)))
    )
    settings = only("remove_confirmation")
    for seed in range(40):
        result = mutate(single_entry, settings, np.random.default_rng(seed), LIBRARY)
        if result.genome is not None:
            assert len(result.genome.entry) >= 1
            assert result.mutations[0].target.startswith("exit.")


def test_modify_entry_changes_one_field_of_one_condition() -> None:
    """The record names the field, not the whole condition, so replay sets
    exactly what was set and an audit reads one before/after pair."""
    result = first_success("modify_entry")
    target = result.mutations[0].target
    assert target.startswith("entry.conditions[")
    assert target.rsplit(".", 1)[1] in ("op", "left", "right")
    assert result.mutations[0].before_json != result.mutations[0].after_json


def test_replace_indicator_swaps_for_one_measured_in_the_same_units() -> None:
    """Swapping an RSI for a moving average would silently turn a threshold
    condition into a price comparison."""
    import json

    result = first_success("replace_indicator")
    mutation = result.mutations[0]
    before = json.loads(mutation.before_json)
    after = json.loads(mutation.after_json)
    assert before["kind"] == after["kind"] == "indicator"
    assert before["name"] != after["name"]
    assert Operand.model_validate(before).domain == Operand.model_validate(after).domain


def test_change_tree_mode_flips_all_and_any() -> None:
    result = first_success("change_tree_mode")
    mutation = result.mutations[0]
    assert mutation.target.endswith(".mode")
    assert {mutation.before_json, mutation.after_json} == {'"all"', '"any"'}


def test_change_tree_mode_is_refused_on_a_single_condition() -> None:
    """``all`` and ``any`` agree on one condition, so the edit would record a
    change that changes nothing."""
    narrow = Phenotype.from_genome(
        parent_genome(
            entry=ConditionTree(conditions=(Condition(left=FAST, op=">", right=SLOW),)),
            exit=ConditionTree(conditions=(Condition(left=FAST, op="<=", right=SLOW),)),
        )
    )
    settings = only("change_tree_mode", max_repair_attempts=4)
    assert mutate(narrow, settings, np.random.default_rng(0), LIBRARY).fell_back


def test_add_and_remove_filter_are_inverses_of_each_other() -> None:
    added = first_success("add_filter")
    assert added.genome is not None
    assert len(added.genome.filters) == 1

    with_filter = Phenotype.from_genome(added.genome)
    removed = first_success("remove_filter", with_filter)
    assert removed.genome is not None
    assert removed.genome.filters == ()


def test_a_numeric_parameter_stays_inside_its_own_bounds() -> None:
    """``perturb_numeric`` snaps to the spec's bounds and step (section 13.4)."""
    settings = MutationSettings(
        parameter_rate=1.0,
        structural_rate=0.0,
        max_mutations_per_child=3,
        parameter=MutationParameterSettings(perturb_pct=1.0),
    )
    for seed in range(40):
        result = mutate(parent(), settings, np.random.default_rng(seed), LIBRARY)
        if result.child is None:
            continue
        for name, value in result.child.params.items():
            spec = result.child.schema[name]
            assert spec.low is not None and spec.high is not None
            assert spec.low <= float(value) <= spec.high
            assert isinstance(value, int) or spec.kind == "float"


def test_a_risk_control_can_be_switched_on_and_off() -> None:
    """Section 13.4's ``toggle_risk_control``: ``None`` to a drawn value and back."""
    settings = MutationSettings(parameter_rate=1.0, structural_rate=0.0, max_mutations_per_child=1)
    switched_on = None
    for seed in range(60):
        result = mutate(parent(), settings, np.random.default_rng(seed), LIBRARY)
        if result.child is None:
            continue
        mutation = result.mutations[0]
        if mutation.operator == "toggle_risk_control" and mutation.before_json == "null":
            switched_on = result.child
            break
    assert switched_on is not None
    assert switched_on.risk.model_dump(exclude_none=True) != {}
    assert switched_on.genome is not None
    # The genome owns the risk spec, so the edit has to land there too or the
    # compiled source and the recorded controls would describe different things.
    assert switched_on.genome.risk == switched_on.risk


def test_the_genome_is_rebuilt_from_a_parameter_edit() -> None:
    """Sizing lives on the genome as well; an edit that only moved the phenotype
    would compile to a strategy that sizes differently from the record."""
    settings = MutationSettings(parameter_rate=1.0, structural_rate=0.0)
    for seed in range(30):
        result = mutate(parent(), settings, np.random.default_rng(seed), LIBRARY)
        if result.child is None or result.child.genome is None:
            continue
        assert result.child.genome.sizing == result.child.sizing
        assert result.child.genome.risk == result.child.risk


def test_a_phenotype_built_from_a_genome_takes_its_bounds_and_controls() -> None:
    genome = parent_genome(risk=RiskSpec(stop_loss_pct=0.03), sizing=SizingSpec(fraction=0.5))
    phenotype = Phenotype.from_genome(genome)
    assert phenotype.schema == genome.params
    assert phenotype.risk == genome.risk
    assert phenotype.sizing == genome.sizing
    assert phenotype.params == {"fast": 10, "slow": 30}


def test_supplied_parameter_values_override_the_declared_defaults() -> None:
    phenotype = Phenotype.from_genome(parent_genome(), {"fast": 7, "slow": 21})
    assert phenotype.params == {"fast": 7, "slow": 21}
