"""Assembling a generation (master spec section 13.2, task T50).

The population's shape is decided before anything is executed, so it can be
checked without a database, a sandbox or a bar. The identity
``n_survivors + n_offspring + n_immigrants == population_size`` is the thing this
file exists to defend: a generation that quietly resized would make every
downstream deflated Sharpe ratio wrong, because section 14.4 charges
``n_evaluations``.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantlab.core.config import DiversitySettings, EvolutionSettings
from quantlab.core.errors import StrategyError
from quantlab.core.genome import Condition, ConditionTree, Operand, StrategyGenome
from quantlab.evolution.diversity import CandidateView
from quantlab.evolution.library import OperatorLibrary
from quantlab.evolution.population import (
    CANDIDATE_ORIGINS,
    ScoredCandidate,
    draw_parents,
    plan_generation,
    rank_candidates,
    seed_population,
)

FAST = Operand(kind="indicator", name="sma", kwargs={"n": 10})
SLOW = Operand(kind="indicator", name="sma", kwargs={"n": 30})


def genome(op: str = ">") -> StrategyGenome:
    return StrategyGenome(
        name="pair",
        entry=ConditionTree(conditions=(Condition(left=FAST, op=op, right=SLOW),)),  # type: ignore[arg-type]
        warmup_bars=30,
    )


def scored(
    name: str,
    fitness: float,
    positions: list[float] | None = None,
    *,
    inner_oos: float = 0.0,
    n_free_params: int = 0,
) -> ScoredCandidate:
    view = CandidateView.from_genome(
        name, genome(), positions if positions is not None else [1.0, 0.0], fitness=fitness
    )
    return ScoredCandidate(view=view, inner_oos=inner_oos, n_free_params=n_free_params)


def distinct_population(n: int, settings: EvolutionSettings) -> list[ScoredCandidate]:
    """``n`` candidates whose behaviour differs enough to survive niching.

    One in-market bar each, at a different bar per candidate, so no two agree on
    any bar either is in the market — the strongest form of "not duplicates" the
    measure of section 13.5 recognises.
    """
    del settings
    return [
        scored(
            f"c{i:02d}",
            fitness=1.0 - i * 0.01,
            positions=[1.0 if bit == i else 0.0 for bit in range(n)],
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# ranking
# ---------------------------------------------------------------------------
def test_ranking_is_by_fitness_descending() -> None:
    population = [scored("low", 0.1), scored("high", 0.9), scored("mid", 0.5)]
    assert [c.candidate_id for c in rank_candidates(population)] == ["high", "mid", "low"]


def test_a_tie_breaks_on_generalisation_then_simplicity_then_id() -> None:
    """Section 13.2 step 3, all three tiers. Two candidates that performed
    identically are not equally good: the one with less to overfit with wins."""
    population = [
        scored("z", 0.5, inner_oos=0.1, n_free_params=2),
        scored("a", 0.5, inner_oos=0.1, n_free_params=2),
        scored("simpler", 0.5, inner_oos=0.1, n_free_params=1),
        scored("generalises", 0.5, inner_oos=0.9, n_free_params=4),
    ]
    assert [c.candidate_id for c in rank_candidates(population)] == [
        "generalises",
        "simpler",
        "a",
        "z",
    ]


def test_ranking_is_deterministic_to_the_last_term() -> None:
    population = [scored(f"c{i}", 0.5) for i in range(8)]
    first = [c.candidate_id for c in rank_candidates(population)]
    assert first == [c.candidate_id for c in rank_candidates(list(reversed(population)))]


def test_a_rejected_candidate_sorts_last() -> None:
    """``FITNESS_REJECTED`` is far below the attainable range, so no special case
    is needed at the comparison site."""
    from quantlab.core.fitness import FITNESS_REJECTED

    population = [scored("rejected", FITNESS_REJECTED), scored("worst_real", 0.0)]
    assert rank_candidates(population)[0].candidate_id == "worst_real"


# ---------------------------------------------------------------------------
# parent selection
# ---------------------------------------------------------------------------
def test_parents_are_drawn_without_replacement() -> None:
    survivors = [scored(f"s{i}", 1.0 - i * 0.1) for i in range(5)]
    drawn = draw_parents(survivors, 5, np.random.default_rng(0))
    assert sorted(c.candidate_id for c in drawn) == sorted(c.candidate_id for c in survivors)


def test_the_best_survivor_is_favoured_but_does_not_monopolise() -> None:
    """Section 13.2 step 5. Weights come from rank, not fitness, so the shape is
    the same in an early generation and a converged one."""
    survivors = [scored(f"s{i}", 1.0 - i * 0.001) for i in range(5)]
    counts: dict[str, int] = {}
    for seed in range(400):
        for parent in draw_parents(survivors, 1, np.random.default_rng(seed)):
            counts[parent.candidate_id] = counts.get(parent.candidate_id, 0) + 1
    assert counts["s0"] > counts["s4"]
    assert counts["s4"] > 0, "the worst survivor must still reproduce sometimes"


def test_more_children_than_survivors_reuses_every_survivor_first() -> None:
    survivors = [scored(f"s{i}", 1.0 - i * 0.1) for i in range(3)]
    drawn = draw_parents(survivors, 7, np.random.default_rng(1))
    assert len(drawn) == 7
    assert {c.candidate_id for c in drawn[:3]} == {"s0", "s1", "s2"}


def test_drawing_from_nobody_is_an_error_not_a_short_generation() -> None:
    with pytest.raises(StrategyError, match="empty survivor set"):
        draw_parents([], 3, np.random.default_rng(0))


def test_drawing_none_is_fine() -> None:
    assert draw_parents([], 0, np.random.default_rng(0)) == []


def test_the_same_seed_draws_the_same_parents() -> None:
    survivors = [scored(f"s{i}", 1.0 - i * 0.1) for i in range(5)]
    left = draw_parents(survivors, 3, np.random.default_rng(7))
    right = draw_parents(survivors, 3, np.random.default_rng(7))
    assert [c.candidate_id for c in left] == [c.candidate_id for c in right]


# ---------------------------------------------------------------------------
# the whole plan
# ---------------------------------------------------------------------------
def test_a_generation_always_totals_the_population_size() -> None:
    """Section 13.2 step 7, over a healthy population and a degenerate one."""
    settings = EvolutionSettings()
    healthy = distinct_population(settings.population_size, settings)
    assert plan_generation(healthy, settings, np.random.default_rng(0)).total == (
        settings.population_size
    )

    twins = [scored(f"t{i}", 1.0 - i * 0.01, [1.0, 1.0, 0.0]) for i in range(16)]
    assert plan_generation(twins, settings, np.random.default_rng(0)).total == (
        settings.population_size
    )


def test_a_healthy_population_keeps_its_survivor_quota() -> None:
    settings = EvolutionSettings()
    plan = plan_generation(
        distinct_population(settings.population_size, settings),
        settings,
        np.random.default_rng(0),
    )
    assert len(plan.survivors) == settings.n_survivors
    assert len(plan.offspring_parents) == settings.n_offspring
    assert plan.n_immigrants == settings.n_immigrants


def test_niching_shortfall_becomes_immigrants_not_offspring() -> None:
    """Section 13.2 step 4: there were not enough *distinct* survivors, and
    offspring are mutations of survivors."""
    settings = EvolutionSettings()
    twins = [scored(f"t{i}", 1.0 - i * 0.01, [1.0, 1.0, 0.0]) for i in range(16)]
    plan = plan_generation(twins, settings, np.random.default_rng(0))
    assert len(plan.survivors) == 1
    assert plan.n_immigrants > settings.n_immigrants
    assert plan.total == settings.population_size


def test_every_offspring_slot_names_its_own_parent() -> None:
    """One parent per slot, in order, so section 13.2's determinism rule —
    ``(seed, gen_index, slot_index, attempt)`` — lines up with the slot."""
    settings = EvolutionSettings()
    plan = plan_generation(
        distinct_population(settings.population_size, settings),
        settings,
        np.random.default_rng(3),
    )
    assert len(plan.offspring_parents) == plan.slots.n_offspring
    assert all(parent in plan.survivors for parent in plan.offspring_parents)


def test_low_diversity_boosts_the_immigrant_count() -> None:
    settings = EvolutionSettings()
    twins = [scored(f"t{i}", 1.0 - i * 0.01, [1.0, 1.0, 0.0]) for i in range(16)]
    plan = plan_generation(twins, settings, np.random.default_rng(0))
    assert plan.diversity < settings.diversity.min_population_diversity
    assert plan.slots.boosted


def test_a_diverse_population_is_reported_as_such() -> None:
    settings = EvolutionSettings()
    plan = plan_generation(
        distinct_population(settings.population_size, settings),
        settings,
        np.random.default_rng(0),
    )
    assert plan.diversity > settings.diversity.min_population_diversity
    assert not plan.slots.boosted


def test_planning_is_deterministic() -> None:
    settings = EvolutionSettings()
    population = distinct_population(settings.population_size, settings)
    left = plan_generation(population, settings, np.random.default_rng(11))
    right = plan_generation(population, settings, np.random.default_rng(11))
    assert [c.candidate_id for c in left.offspring_parents] == [
        c.candidate_id for c in right.offspring_parents
    ]
    assert left.diversity == right.diversity


def test_an_all_rejected_population_becomes_a_fresh_one() -> None:
    """Nothing survived, so there is nothing to mutate. The generation refills
    with immigrants rather than failing — which is what keeps a bad generation
    from ending the run."""
    settings = EvolutionSettings(
        n_survivors=1,
        n_offspring=1,
        n_immigrants=1,
        population_size=3,
        diversity=DiversitySettings(max_pairwise_similarity=0.99, max_immigrants=2),
    )
    empty: list[ScoredCandidate] = []
    plan = plan_generation(empty, settings, np.random.default_rng(0))
    assert plan.survivors == ()
    assert plan.offspring_parents == ()
    assert plan.n_immigrants == settings.population_size


# ---------------------------------------------------------------------------
# generation zero
# ---------------------------------------------------------------------------
def test_generation_zero_fills_in_the_specified_order() -> None:
    """Supplied candidates first — a person chose those — then LLM proposals,
    then random draws until the population is full."""
    settings = EvolutionSettings()
    members = seed_population(
        settings,
        OperatorLibrary(),
        np.random.default_rng(0),
        supplied=[genome(">"), genome("<")],
        proposed=[genome(">="), genome("<=")],
        max_seed_genomes=1,
    )
    assert len(members) == settings.population_size
    assert [m.source for m in members[:3]] == ["supplied", "supplied", "llm"]
    assert all(m.source == "random" for m in members[3:])
    assert {m.origin for m in members} <= set(CANDIDATE_ORIGINS)


def test_generation_zero_is_full_without_any_llm() -> None:
    """Which is what lets an evolution run offline (section 12's mock provider)."""
    settings = EvolutionSettings()
    members = seed_population(settings, OperatorLibrary(), np.random.default_rng(1))
    assert len(members) == settings.population_size
    assert all(m.source == "random" for m in members)
    assert all(m.genome is not None for m in members)


def test_proposals_never_displace_a_supplied_candidate() -> None:
    settings = EvolutionSettings(
        n_survivors=1,
        n_offspring=1,
        n_immigrants=1,
        population_size=3,
        diversity=DiversitySettings(max_immigrants=2),
    )
    members = seed_population(
        settings,
        OperatorLibrary(),
        np.random.default_rng(0),
        supplied=[genome(">"), genome("<"), genome(">=")],
        proposed=[genome("<=")],
        max_seed_genomes=8,
    )
    assert [m.source for m in members] == ["supplied"] * 3


def test_generation_zero_is_deterministic() -> None:
    settings = EvolutionSettings()
    library = OperatorLibrary()
    left = seed_population(settings, library, np.random.default_rng(5))
    right = seed_population(settings, library, np.random.default_rng(5))
    assert [m.genome.canonical() for m in left if m.genome] == [
        m.genome.canonical() for m in right if m.genome
    ]
