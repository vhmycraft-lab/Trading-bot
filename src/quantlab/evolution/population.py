"""Assembling one generation (master spec section 13.2, steps 3-7).

Ranking, survivor selection, reproduction and immigration — the shape of a
generation, decided before anything is executed or written down. Everything here
is pure: it takes scored candidates and a generator, and returns a plan. The loop
of T52 is what turns a plan into runs and rows.

That split is deliberate. The identity ``n_survivors + n_offspring +
n_immigrants == population_size`` is asserted at config load and again here, and a
generation that quietly resized would make every downstream deflated Sharpe ratio
wrong (section 14.4 charges ``n_evaluations``). Deciding the shape separately from
executing it is what lets that be checked without a database, a sandbox or a bar.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final, Literal

import numpy as np

from quantlab.core.config import EvolutionSettings
from quantlab.core.errors import StrategyError
from quantlab.core.genome import StrategyGenome
from quantlab.evolution.diversity import (
    CandidateView,
    SlotPlan,
    population_diversity,
    select_survivors,
    slot_plan,
)
from quantlab.evolution.library import OperatorLibrary

__all__ = [
    "CANDIDATE_ORIGINS",
    "GenerationPlan",
    "ScoredCandidate",
    "draw_parents",
    "plan_generation",
    "rank_candidates",
    "seed_population",
]

#: What ``candidate.origin`` may say (spec section 6).
CANDIDATE_ORIGINS: Final[tuple[str, ...]] = (
    "seed",
    "survivor",
    "mutant",
    "immigrant",
    "refined",
)


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    """One evaluated candidate, as ranking and selection see it."""

    view: CandidateView
    #: The ``inner_oos`` fitness component, which breaks ties before parameter count.
    inner_oos: float = 0.0
    n_free_params: int = 0

    @property
    def candidate_id(self) -> str:
        return self.view.candidate_id

    @property
    def fitness(self) -> float:
        return self.view.fitness

    def sort_key(self) -> tuple[float, float, int, str]:
        """Section 13.2 step 3's ordering, as a key that sorts ascending.

        Fitness descending, then inner-OOS descending, then *fewer* free
        parameters, then lower ``candidate_id``. Deterministic to the last term,
        and biased towards the simpler strategy — two candidates that performed
        identically are not equally good, and the one with less to overfit with
        is the one to keep.
        """
        return (-self.fitness, -self.inner_oos, self.n_free_params, self.candidate_id)


def rank_candidates(scored: Sequence[ScoredCandidate]) -> list[ScoredCandidate]:
    """Order a population by section 13.2 step 3."""
    return sorted(scored, key=lambda candidate: candidate.sort_key())


def draw_parents(
    survivors: Sequence[ScoredCandidate], n: int, rng: np.random.Generator
) -> list[ScoredCandidate]:
    """Draw ``n`` parents by rank-proportional selection without replacement.

    Section 13.2 step 5. Weights come from *rank*, not fitness: fitness is a score
    in ``[0, 1]`` whose spread varies wildly between generations, so weighting by
    it would make the best survivor monopolise an early generation and matter
    barely at all in a converged one. Rank weights are the same shape every time.

    Without replacement, up to the number of survivors. When more children are
    wanted than there are survivors, the draw restarts — every survivor is used
    once before any is used twice, which is the closest thing to "without
    replacement" that a larger request admits.

    Raises:
        StrategyError: there is nobody to draw from. A generation cannot produce
            offspring of an empty survivor set, and returning fewer children than
            asked for would silently resize the population.
    """
    if n <= 0:
        return []
    if not survivors:
        raise StrategyError("cannot draw parents from an empty survivor set", requested=n)

    ordered = rank_candidates(survivors)
    weights = np.arange(len(ordered), 0, -1, dtype="float64")
    weights /= weights.sum()

    drawn: list[ScoredCandidate] = []
    while len(drawn) < n:
        take = min(n - len(drawn), len(ordered))
        indices = rng.choice(len(ordered), size=take, replace=False, p=weights)
        drawn.extend(ordered[int(index)] for index in indices)
    return drawn


@dataclass(frozen=True, slots=True)
class GenerationPlan:
    """The shape of the next generation, before anything is built.

    ``offspring_parents`` names one parent per offspring slot, in order, so the
    caller mutates each in turn and the seeding of section 13.2's determinism
    rule — ``(evolution.seed, gen_index, slot_index, attempt)`` — lines up with
    the slot it fills.
    """

    survivors: tuple[ScoredCandidate, ...] = ()
    offspring_parents: tuple[ScoredCandidate, ...] = ()
    n_immigrants: int = 0
    diversity: float = 1.0
    slots: SlotPlan = field(default_factory=lambda: SlotPlan(0, 0, 0))

    @property
    def total(self) -> int:
        return len(self.survivors) + len(self.offspring_parents) + self.n_immigrants


def plan_generation(
    scored: Sequence[ScoredCandidate],
    settings: EvolutionSettings,
    rng: np.random.Generator,
) -> GenerationPlan:
    """Decide the next generation's shape (spec section 13.2, steps 3-7).

    Rank, select survivors under niching, divide the remaining slots, and draw a
    parent for each offspring slot. Nothing is executed and nothing is written.

    Raises:
        StrategyError: the plan does not total ``population_size``. That is step
            7's assertion, and it is an error rather than a silent resize.
    """
    ranked = rank_candidates(scored)
    survivors = select_survivors(
        [candidate.view for candidate in ranked],
        n_survivors=settings.n_survivors,
        settings=settings.diversity,
    )
    kept_ids = {view.candidate_id for view in survivors}
    kept = tuple(candidate for candidate in ranked if candidate.candidate_id in kept_ids)

    diversity = population_diversity([candidate.view for candidate in ranked], settings.diversity)
    slots = slot_plan(settings, diversity=diversity, n_survivors=len(kept))

    parents = draw_parents(kept, slots.n_offspring, rng) if kept else []
    if not kept and slots.n_offspring:
        # No survivor means nothing to mutate. Section 13.2 step 4 already turns a
        # survivor shortfall into immigrants; a *total* shortfall turns the
        # offspring slots into immigrants too, rather than failing the generation.
        slots = SlotPlan(
            n_survivors=0,
            n_offspring=0,
            n_immigrants=settings.population_size,
            boosted=slots.boosted,
        )

    plan = GenerationPlan(
        survivors=kept,
        offspring_parents=tuple(parents),
        n_immigrants=slots.n_immigrants,
        diversity=diversity,
        slots=slots,
    )
    if plan.total != settings.population_size:
        raise StrategyError(
            "a generation must hold exactly population_size candidates",
            planned=plan.total,
            population_size=settings.population_size,
        )
    return plan


@dataclass(frozen=True, slots=True)
class SeedMember:
    """One slot of generation 0, before it is registered or evaluated."""

    genome: StrategyGenome | None
    origin: Literal["seed", "immigrant"]
    source: Literal["supplied", "llm", "random"]


def seed_population(
    settings: EvolutionSettings,
    library: OperatorLibrary,
    rng: np.random.Generator,
    *,
    supplied: Sequence[StrategyGenome] = (),
    proposed: Sequence[StrategyGenome] = (),
    max_seed_genomes: int = 0,
) -> list[SeedMember]:
    """Fill generation 0 (spec section 13.2).

    In the order the specification gives: explicitly supplied seed candidates
    (baselines and any human strategies), then LLM-proposed genomes up to
    ``research.max_seed_genomes``, then random genomes from the operator library
    until the population is full.

    The order matters. Supplied candidates are the ones a person chose, so they
    are never displaced by a proposal or a draw; and the random tail means a
    generation 0 is always full even with no LLM configured, which is what lets
    an evolution run offline (the ``MockLLMProvider`` path of section 12).

    Raises:
        StrategyError: the library could not fill the remaining slots. A short
            generation 0 would silently resize the population.
    """
    members: list[SeedMember] = [
        SeedMember(genome=genome, origin="seed", source="supplied")
        for genome in supplied[: settings.population_size]
    ]
    for genome in proposed[: max(0, max_seed_genomes)]:
        if len(members) >= settings.population_size:
            break
        members.append(SeedMember(genome=genome, origin="seed", source="llm"))

    while len(members) < settings.population_size:
        index = len(members)
        drawn = library.draw_genome(rng, name=f"gen0_{index:02d}")
        members.append(SeedMember(genome=drawn, origin="immigrant", source="random"))

    if len(members) != settings.population_size:  # pragma: no cover - loop guarantees it
        raise StrategyError("generation 0 is not full", filled=len(members))
    return members
