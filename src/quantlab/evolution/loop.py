"""The generation loop (master spec section 13.2). 🔒

Eight steps per generation — evaluate, score, rank, select, reproduce, immigrate,
assemble, persist — and one rule that outranks all of them.

**INV-9, segment discipline.** Evolution reads the **train** segment and its own
inner folds, and nothing else. :func:`require_evolution_segment` asserts that for
every run this module creates, and it is the only place a run is created. The
validation segment is reachable solely through ``promote()``, which writes a
``candidate_promotion`` row *before* the run executes; the test partition is
reachable only through ``quantlab lockbox``. No validation-derived number is fed
back into fitness, selection, mutation or stopping — a search that could see the
validation set would optimise against it, and the number the platform exists to
produce would be worth nothing.

**Determinism (INV-7).** Every stochastic choice draws from a generator seeded
from ``(evolution.seed, gen_index, slot_index, attempt)``, so re-running an
evolution from its stored configuration reproduces the identical sequence of
candidate ids. That is what makes ``evolve resume`` correct: a run that resumed
into a different sequence would silently be a different search.

**Cost.** Every evaluation is an ordinary cached run (section 11.2). With one
fixed environment a survivor carried forward is a cache hit and costs nothing.
Under per-generation environments (Project Rome sections 4-19) it is not: the
environment is part of the run's identity, so a survivor is re-evaluated on the
new generation's history — which is the entire point, since a survivor that has
only ever been measured on one window has not been shown to survive anything. The
count that matters either way is ``n_evaluations``, which section 14.4 charges
into the deflated Sharpe ratio's ``M``, and it counts cache hits too, because a
candidate that was evaluated once and reused ten times was still one hypothesis
tested.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from quantlab.core.config import EvolutionSettings
from quantlab.core.environment import GenerationEnvironment
from quantlab.core.errors import EngineError, QuantLabError, StrategyError
from quantlab.core.fitness import (
    FITNESS_REJECTED,
    FitnessResult,
    InnerFoldReport,
    compute_fitness,
)
from quantlab.core.genome import StrategyGenome
from quantlab.core.hashing import canonical_json, short_id
from quantlab.core.logging import get_logger
from quantlab.core.metrics import MetricSet, benchmark_drawdown
from quantlab.core.types import BacktestConfig, BarFrame
from quantlab.evolution.diversity import CandidateView, behaviour_hash, genome_signature
from quantlab.evolution.library import OperatorLibrary
from quantlab.evolution.mutation import AppliedMutation, Phenotype, mutate
from quantlab.evolution.population import (
    GenerationPlan,
    ScoredCandidate,
    plan_generation,
    rank_candidates,
    seed_population,
)
from quantlab.ports.store import ExperimentStore

__all__ = [
    "EVOLUTION_SEGMENTS",
    "STOP_REASONS",
    "EnvironmentFor",
    "EvolutionResult",
    "GenerationEnvironment",
    "GenerationOutcome",
    "Member",
    "evolve",
    "require_evolution_segment",
    "resume_point",
]

log = get_logger(__name__)

#: The only segments evolution may read (spec section 13.7, INV-9).
EVOLUTION_SEGMENTS: Final[tuple[str, ...]] = ("train", "inner_is", "inner_oos")

#: Why a run stopped (``evolution_run.stop_reason``, section 6).
STOP_REASONS: Final[tuple[str, ...]] = (
    "max_generations",
    "max_evaluations",
    "max_wall_clock_s",
    "no_improvement",
)


def require_evolution_segment(segment: str) -> str:
    """Return ``segment``, or refuse it — INV-9's mechanical enforcement.

    Every run evolution creates passes through here. Removing this check is the
    demonstration section 22 asks for: with it gone, the loop would happily
    evaluate a population against the validation segment, and every verdict
    afterwards would be measuring a search that had already seen the answer.

    Raises:
        StrategyError: the segment is not train or an inner fold.
    """
    if not segment.startswith(EVOLUTION_SEGMENTS):
        raise StrategyError(
            "evolution may only read the train segment and its own inner folds; "
            "the validation segment is reachable only through a recorded promotion "
            "and the test partition only through the lockbox (INV-9)",
            segment=segment,
            allowed=list(EVOLUTION_SEGMENTS),
        )
    return segment


# ---------------------------------------------------------------------------
# what the loop needs from the outside
# ---------------------------------------------------------------------------
#: Registers a genome as a strategy version and returns ``(strategy_id, source)``.
Register = Callable[[StrategyGenome], tuple[str, str]]

#: Builds an evaluator bound to one strategy's source, for the runner.
EvaluatorFor = Callable[[str], Any]


#: Supplies one generation's environment. Injected, like ``inner_for``: the
#: selection must happen inside Rome's trusted infrastructure before the backtest
#: begins (Rome section 8), and ``evolution/`` may import neither an adapter nor
#: a randomisation service of its own (INV-8). ``None`` keeps the loop on one
#: fixed environment, which is what a walk-forward window or a replay wants.
EnvironmentFor = Callable[[int], GenerationEnvironment]

#: Measures generalisation inside train (section 13.6). Optional: without it the
#: ``inner_oos`` component scores 0 for every candidate, which is honest and
#: uniform, and section 13.3 does not also penalise the absence.
InnerFor = Callable[[StrategyGenome, Mapping[str, Any]], InnerFoldReport]


@dataclass(frozen=True, slots=True)
class Member:
    """One population slot: a candidate before it has been evaluated."""

    candidate_id: str
    phenotype: Phenotype
    origin: str
    parent_candidate_id: str | None = None
    mutations: tuple[AppliedMutation, ...] = ()

    @property
    def genome(self) -> StrategyGenome:
        genome = self.phenotype.genome
        if genome is None:  # pragma: no cover - the loop only builds genome members
            raise StrategyError("this member has no genome", candidate_id=self.candidate_id)
        return genome


@dataclass(frozen=True, slots=True)
class GenerationOutcome:
    """What one generation produced."""

    gen_index: int
    scored: tuple[ScoredCandidate, ...] = ()
    plan: GenerationPlan | None = None
    n_evaluated: int = 0
    n_cache_hits: int = 0
    n_rejected_by_gate: int = 0
    n_immigrants_used: int = 0
    diversity: float = 1.0
    stats: Mapping[str, Any] = field(default_factory=dict)

    @property
    def best_fitness(self) -> float | None:
        real = [c.fitness for c in self.scored if c.fitness > FITNESS_REJECTED]
        return max(real) if real else None


@dataclass(frozen=True, slots=True)
class EvolutionResult:
    """A finished (or stopped) evolution run."""

    evolution_id: str
    generations: tuple[GenerationOutcome, ...] = ()
    n_evaluations: int = 0
    stop_reason: str = "max_generations"
    candidate_ids: tuple[str, ...] = ()

    @property
    def best(self) -> ScoredCandidate | None:
        """The fittest candidate across every generation."""
        everyone = [candidate for outcome in self.generations for candidate in outcome.scored]
        ranked = rank_candidates(everyone)
        return ranked[0] if ranked and ranked[0].fitness > FITNESS_REJECTED else None


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------
def evolve(
    *,
    evolution_id: str,
    store: ExperimentStore,
    runner: Any,
    register: Register,
    evaluator_for: EvaluatorFor,
    bars_train: BarFrame,
    experiment_id: str,
    dataset_id: str,
    split_id: str,
    settings: EvolutionSettings,
    config: BacktestConfig,
    library: OperatorLibrary | None = None,
    seeds: Sequence[StrategyGenome] = (),
    inner_for: InnerFor | None = None,
    environment_for: EnvironmentFor | None = None,
    segment: str = "train",
    start_generation: int = 0,
    clock: Callable[[], float] = time.monotonic,
) -> EvolutionResult:
    """Run the generation loop of section 13.2.

    The caller opens the ``evolution_run`` row before calling: it owns the three
    configuration blobs section 6 stores verbatim, and every generation and
    candidate written here is a foreign key into it.

    Args:
        evolution_id: The run's id, of an ``evolution_run`` row that already
            exists. It seeds nothing — determinism comes from ``settings.seed`` —
            but it is what ``resume`` re-enters.
        store: Where candidates, generations and mutations are recorded.
        runner: An ``ExperimentRunner``. Every evaluation goes through it, so
            every one is a cached run (section 11.2).
        register: Turns a genome into a registered strategy version. Injected so
            this module needs no loader and no source store of its own.
        evaluator_for: Builds a sandboxed evaluator from a strategy's source.
        bars_train: The train segment's bars, and the only bars this loop sees.
        settings: ``evolution``.
        config: The backtest settings every candidate is evaluated under.
        seeds: Explicitly supplied genomes for generation 0 (section 13.2).
        inner_for: Optional inner walk-forward (section 13.6).
        environment_for: Optional per-generation environment (Project Rome
            sections 4-19). Called once per generation, **including** generations
            a resume is replaying — a replayed generation must be handed the
            environment it was recorded with, or the resumed run becomes a
            different search. When ``None`` every generation shares ``bars_train``
            and ``config``.
        segment: Checked by :func:`require_evolution_segment` before anything runs.
        start_generation: The first generation to **record**, for ``evolve
            resume``. Earlier generations are still recomputed — see below.
        clock: Injected so the wall-clock stop can be tested without waiting.

    Returns:
        An :class:`EvolutionResult` carrying every generation and the stop reason.
    """
    require_evolution_segment(segment)
    draw = library or OperatorLibrary(limits=settings.genome)
    started = clock()
    outcomes: list[GenerationOutcome] = []
    evaluations = 0
    all_ids: list[str] = []
    best_so_far = FITNESS_REJECTED
    stale = 0
    stop_reason = "max_generations"

    members = _generation_zero(settings, draw, seeds, 0)
    previous: GenerationOutcome | None = None

    # Resume recomputes every generation from zero and records only the ones that
    # are missing. It has to: a generation's shape depends on the previous
    # generation's *plan*, and niching reads position series that the store does
    # not keep (only their digest, section 13.5). Re-entering from a partial
    # reconstruction would silently plan differently and make the resumed run a
    # different search.
    #
    # This is cheap and is exactly what section 13.9 promises. Every evaluation of
    # an already-recorded generation is a cache hit (section 11.2), so nothing is
    # re-executed; what is repeated is the fitness arithmetic and the plan.
    for gen_index in range(settings.max_generations):
        if previous is not None:
            members = _next_generation(previous, settings, draw, gen_index)

        replaying = gen_index < start_generation
        environment = environment_for(gen_index) if environment_for is not None else None
        outcome, used = _run_generation(
            gen_index=gen_index,
            record=not replaying,
            members=members,
            store=store,
            runner=runner,
            register=register,
            evaluator_for=evaluator_for,
            bars=bars_train if environment is None else environment.bars,
            evolution_id=evolution_id,
            experiment_id=experiment_id,
            dataset_id=dataset_id,
            split_id=split_id,
            segment=segment if environment is None else environment.segment_for(segment),
            settings=settings,
            config=config if environment is None else environment.config,
            environment_id="" if environment is None else environment.environment_id,
            inner_for=inner_for,
        )
        if replaying:
            # Already recorded, already counted. Carry the plan forward and move
            # on: re-counting would inflate the ``M`` section 14.4 charges.
            previous = outcome
            continue

        evaluations += used
        store.count_evaluation(evolution_id, used)
        outcomes.append(outcome)
        all_ids.extend(candidate.candidate_id for candidate in outcome.scored)

        best = outcome.best_fitness
        if best is not None and best > best_so_far + settings.min_improvement:
            best_so_far, stale = best, 0
        else:
            stale += 1

        log.info(
            "generation_finished",
            evolution_id=evolution_id,
            gen_index=gen_index,
            best_fitness=best,
            diversity=outcome.diversity,
            n_evaluations=evaluations,
        )

        stop_reason = _stop_reason(
            settings, evaluations=evaluations, stale=stale, elapsed=clock() - started
        )
        if stop_reason != "max_generations" or gen_index == settings.max_generations - 1:
            break
        previous = outcome

    return EvolutionResult(
        evolution_id=evolution_id,
        generations=tuple(outcomes),
        n_evaluations=evaluations,
        stop_reason=stop_reason,
        candidate_ids=tuple(all_ids),
    )


def _stop_reason(
    settings: EvolutionSettings, *, evaluations: int, stale: int, elapsed: float
) -> str:
    """Which budget, if any, has run out (spec section 13.2).

    Checked in a fixed order so two runs that exhausted two budgets at once report
    the same reason. Stopping early does not change ``n_evaluations``, which is
    what section 14.4 charges.
    """
    if evaluations >= settings.max_evaluations:
        return "max_evaluations"
    if elapsed >= settings.max_wall_clock_s:
        return "max_wall_clock_s"
    if stale >= settings.stop_on_no_improvement_generations:
        return "no_improvement"
    return "max_generations"


# ---------------------------------------------------------------------------
# building a generation
# ---------------------------------------------------------------------------
def _rng(
    settings: EvolutionSettings, gen_index: int, slot: int, attempt: int = 0
) -> np.random.Generator:
    """The generator for one slot (spec section 13.2's determinism rule)."""
    return np.random.default_rng([settings.seed, gen_index, slot, attempt])


def _candidate_id(evolution_id: str, gen_index: int, slot: int, phenotype: Phenotype) -> str:
    """An id that depends on the candidate, not on when it was created.

    Built from the run, the slot and the candidate's own content, so the same
    search re-run from the same seed produces the identical sequence of ids —
    which is what ``test_evolution_replay.py`` checks.
    """
    genome = phenotype.genome
    body = "" if genome is None else genome.canonical()
    return short_id(
        f"{evolution_id}|{gen_index}|{slot}|{body}|{canonical_json(dict(phenotype.params))}"
    )


def _generation_zero(
    settings: EvolutionSettings,
    library: OperatorLibrary,
    seeds: Sequence[StrategyGenome],
    gen_index: int,
) -> list[Member]:
    """Fill the first generation (spec section 13.2)."""
    drawn = seed_population(settings, library, _rng(settings, gen_index, 0), supplied=seeds)
    return [
        Member(
            candidate_id="",  # filled in by `_run_generation`, which knows the run id
            phenotype=Phenotype.from_genome(member.genome),
            origin=member.origin,
        )
        for member in drawn
        if member.genome is not None
    ]


def _next_generation(
    previous: GenerationOutcome,
    settings: EvolutionSettings,
    library: OperatorLibrary,
    gen_index: int,
) -> list[Member]:
    """Survivors carried forward, offspring mutated, immigrants drawn.

    A mutation that exhausts its repair budget leaves the slot to an immigrant
    (section 13.4, layer 3) rather than being patched into validity, so the
    population still totals ``population_size``.
    """
    plan = previous.plan
    if plan is None:  # pragma: no cover - every finished generation carries one
        raise StrategyError("the previous generation has no plan to build on")

    members: list[Member] = []
    for candidate in plan.survivors:
        members.append(
            Member(
                candidate_id="",
                phenotype=_phenotype_of(candidate),
                origin="survivor",
                parent_candidate_id=candidate.candidate_id,
            )
        )

    shortfall = 0
    for slot, parent in enumerate(plan.offspring_parents, start=len(members)):
        result = mutate(
            _phenotype_of(parent), settings.mutation, _rng(settings, gen_index, slot), library
        )
        if result.child is None:
            shortfall += 1
            continue
        members.append(
            Member(
                candidate_id="",
                phenotype=result.child,
                origin="mutant",
                parent_candidate_id=parent.candidate_id,
                mutations=result.mutations,
            )
        )

    for slot in range(len(members), settings.population_size):
        genome = library.draw_genome(
            _rng(settings, gen_index, slot), name=f"gen{gen_index}_{slot:02d}"
        )
        members.append(
            Member(candidate_id="", phenotype=Phenotype.from_genome(genome), origin="immigrant")
        )
    del shortfall  # counted through the immigrant fill above
    return members


def _phenotype_of(candidate: ScoredCandidate) -> Phenotype:
    """A survivor's phenotype: what it is carried forward as, or mutated from."""
    if candidate.phenotype is None:  # pragma: no cover - the loop always attaches one
        raise StrategyError("candidate carries no phenotype", candidate_id=candidate.candidate_id)
    return candidate.phenotype


# ---------------------------------------------------------------------------
# running one generation
# ---------------------------------------------------------------------------
def _run_generation(
    *,
    gen_index: int,
    record: bool,
    members: Sequence[Member],
    store: ExperimentStore,
    runner: Any,
    register: Register,
    evaluator_for: EvaluatorFor,
    bars: BarFrame,
    evolution_id: str,
    experiment_id: str,
    dataset_id: str,
    split_id: str,
    segment: str,
    settings: EvolutionSettings,
    config: BacktestConfig,
    inner_for: InnerFor | None,
    environment_id: str = "",
) -> tuple[GenerationOutcome, int]:
    """Steps 1-4 of section 13.2, plus the plan for the next generation.

    Evaluation happens first and writes nothing to the evolution tables; the
    generation row and its candidates are written together at the end, by
    :func:`_persist_generation`. That ordering is forced, and it is the right one:
    ``candidate.generation_id`` is a foreign key, so the generation must exist
    first, while section 6 makes a generation row append-only with statistics that
    are only known once every candidate has been scored.

    A crash mid-generation therefore leaves the *runs* in the store and no
    candidates for the unfinished generation — which is exactly what ``resume``
    needs. Re-entering re-evaluates that generation and every evaluation is a
    cache hit (section 11.2), so nothing expensive is repeated and no candidate is
    recorded twice.

    ``record`` is false while a resume is recomputing generations that are already
    in the store. They are still evaluated, because the next generation's shape
    depends on this one's plan, but nothing is written.
    """
    require_evolution_segment(segment)

    identified: list[Member] = []
    results: list[_Evaluated] = []
    for slot, member in enumerate(members):
        named = Member(
            candidate_id=_candidate_id(evolution_id, gen_index, slot, member.phenotype),
            phenotype=member.phenotype,
            origin=member.origin,
            parent_candidate_id=member.parent_candidate_id,
            mutations=member.mutations,
        )
        identified.append(named)
        results.append(
            _evaluate_member(
                named,
                store=store,
                runner=runner,
                register=register,
                evaluator_for=evaluator_for,
                bars=bars,
                experiment_id=experiment_id,
                dataset_id=dataset_id,
                split_id=split_id,
                segment=segment,
                settings=settings,
                config=config,
                inner_for=inner_for,
            )
        )

    scored = [result.candidate for result in results]
    plan = plan_generation(scored, settings, _rng(settings, gen_index, settings.population_size))
    outcome = GenerationOutcome(
        gen_index=gen_index,
        scored=tuple(scored),
        plan=plan,
        n_evaluated=len(results),
        n_cache_hits=sum(1 for result in results if result.cache_hit),
        n_rejected_by_gate=sum(1 for result in results if result.fitness.rejected),
        n_immigrants_used=sum(1 for member in members if member.origin == "immigrant"),
        diversity=plan.diversity,
        stats={
            "n_survivors": len(plan.survivors),
            "n_offspring": len(plan.offspring_parents),
            "n_immigrants": plan.n_immigrants,
            "boosted": plan.slots.boosted,
            # The opaque id only. The window, seed, capital and slippage live in
            # `training_environment` and are read by a privileged audit, never by
            # anything that builds a prompt or a strategy's inputs (Rome section 6).
            "environment_id": environment_id,
        },
    )
    if record:
        _persist_generation(store, evolution_id, outcome, identified, results)
    return outcome, len(results)


@dataclass(frozen=True, slots=True)
class _Evaluated:
    candidate: ScoredCandidate
    fitness: FitnessResult
    cache_hit: bool
    strategy_id: str = ""
    run_id: str | None = None


def _evaluate_member(
    member: Member,
    *,
    store: ExperimentStore,
    runner: Any,
    register: Register,
    evaluator_for: EvaluatorFor,
    bars: BarFrame,
    experiment_id: str,
    dataset_id: str,
    split_id: str,
    segment: str,
    settings: EvolutionSettings,
    config: BacktestConfig,
    inner_for: InnerFor | None,
) -> _Evaluated:
    """Evaluate one candidate on train and score it (section 13.2, steps 1-2).

    A candidate whose run *fails* is scored ``FITNESS_REJECTED`` and kept, never
    silently dropped: section 13.2 says so, and a generation that quietly shrank
    when a strategy raised would resize the population.

    ``EngineError`` is the exception, and is re-raised. It means an accounting
    invariant broke inside the engine, which is a defect in the platform rather
    than in the candidate — recording it as a rejected candidate would hide a bug
    behind a plausible-looking result.
    """
    genome = member.genome
    strategy_id, source = register(genome)
    params = dict(member.phenotype.params)

    try:
        run = runner.run(
            evaluator_for(source),
            experiment_id=experiment_id,
            strategy_id=strategy_id,
            dataset_id=dataset_id,
            split_id=split_id,
            segment=require_evolution_segment(segment),
            bars=bars,
            params=params,
            config=config,
        )
    except EngineError:
        raise
    except QuantLabError as exc:
        return _rejected(member, genome, reason=type(exc).__name__, strategy_id=strategy_id)

    if run.record.status != "ok":
        # The runner files a failed run rather than raising (section 11.2), so
        # this is where most failures arrive. The reason comes from the record,
        # which is where section 13.2 says the candidate's error lives.
        return _rejected(member, genome, reason=_error_type(run.record), strategy_id=strategy_id)

    trades: Sequence[Any]
    if run.result is None or run.metrics is None:
        # A cache hit returns the stored record without re-executing, so the
        # numbers come from the store rather than from a fresh simulation.
        stored = _metrics_from_store(store, run.run_id)
        if stored is None:
            return _rejected(member, genome, reason="no_metrics", strategy_id=strategy_id)
        metrics, trades, positions = stored, store.trades_for(run.run_id), None
    else:
        metrics = run.metrics
        trades = run.result.trades
        positions = np.asarray(run.result.position_frac, dtype="float64")

    inner = InnerFoldReport() if inner_for is None else inner_for(genome, params)
    # Computed here, from the same ``bars`` this candidate was just run on, rather
    # than hoisted to the generation or cached by symbol and segment. With a
    # per-generation TrainingEnvironment the window moves while the symbol and the
    # segment name do not, so a hoisted value is one refactor away from being
    # silently stale — and a drawdown ceiling that is quietly measured against the
    # wrong window is exactly the kind of defect that produces no error. The cost
    # is one pass over the closes per candidate, against a full backtest. ADR 0012.
    fitness = compute_fitness(
        metrics,
        trades,
        inner,
        None,
        settings.fitness,
        n_free_params=len(genome.params),
        benchmark_drawdown=benchmark_drawdown(bars.close),
    )
    view = CandidateView(
        candidate_id=member.candidate_id,
        fitness=fitness.fitness,
        signature=genome_signature(genome),
        positions=positions,
        behaviour=None if positions is None else behaviour_hash(positions),
    )
    return _Evaluated(
        candidate=ScoredCandidate(
            view=view,
            inner_oos=float(fitness.components.get("inner_oos", 0.0)),
            n_free_params=len(genome.params),
            phenotype=member.phenotype,
        ),
        fitness=fitness,
        cache_hit=bool(run.cache_hit),
        strategy_id=strategy_id,
        run_id=run.run_id,
    )


def _error_type(record: Any) -> str:
    """The kind of failure a run recorded, for ``candidate.gate_failure``."""
    document = getattr(record, "error_json", None)
    if not document:
        return str(getattr(record, "status", "failed"))
    try:
        return str(json.loads(document).get("type", "failed"))
    except (TypeError, ValueError):  # pragma: no cover - the store writes canonical JSON
        return "failed"


def _metrics_from_store(store: ExperimentStore, run_id: str) -> MetricSet | None:
    """Rebuild a cached run's metric set from the ``metric`` rows."""
    stored = store.metrics_for(run_id)
    if not stored:
        return None
    return MetricSet.model_validate(
        {name: value for name, value in stored.items() if name in MetricSet.model_fields}
    )


def _rejected(
    member: Member, genome: StrategyGenome, *, reason: str, strategy_id: str
) -> _Evaluated:
    """A candidate that could not be evaluated, kept in the generation."""
    return _Evaluated(
        candidate=ScoredCandidate(
            view=CandidateView(
                candidate_id=member.candidate_id,
                fitness=FITNESS_REJECTED,
                signature=genome_signature(genome),
            ),
            n_free_params=len(genome.params),
            phenotype=member.phenotype,
        ),
        fitness=FitnessResult(fitness=FITNESS_REJECTED, gate_failure=reason),
        cache_hit=False,
        strategy_id=strategy_id,
    )


def _persist_generation(
    store: ExperimentStore,
    evolution_id: str,
    outcome: GenerationOutcome,
    members: Sequence[Member],
    results: Sequence[_Evaluated],
) -> None:
    """Write the generation and everything in it (section 13.2, step 8).

    The generation row first, because ``candidate.generation_id`` is a foreign
    key; then each candidate, the mutations that produced it, and its score. All
    of it after the generation is complete, because section 6 makes a generation
    row append-only and its statistics are only known once every candidate has
    been scored.
    """
    generation_id = short_id(f"{evolution_id}|{outcome.gen_index}")
    real = sorted(c.fitness for c in outcome.scored if c.fitness > FITNESS_REJECTED)
    median = None
    if real:
        middle = len(real) // 2
        median = real[middle] if len(real) % 2 else (real[middle - 1] + real[middle]) / 2.0

    store.add_generation(
        generation_id=generation_id,
        evolution_id=evolution_id,
        gen_index=outcome.gen_index,
        diversity=outcome.diversity,
        n_evaluated=outcome.n_evaluated,
        n_cache_hits=outcome.n_cache_hits,
        n_rejected_by_gate=outcome.n_rejected_by_gate,
        n_immigrants_used=outcome.n_immigrants_used,
        stats_json=json.dumps(dict(outcome.stats), sort_keys=True),
        best_fitness=outcome.best_fitness,
        median_fitness=median,
        mean_fitness=sum(real) / len(real) if real else None,
    )

    ranked = rank_candidates(list(outcome.scored))
    ranks = {candidate.candidate_id: position for position, candidate in enumerate(ranked, 1)}
    survivors = {c.candidate_id for c in (outcome.plan.survivors if outcome.plan else ())}

    for member, result in zip(members, results, strict=True):
        store.add_candidate(
            candidate_id=member.candidate_id,
            evolution_id=evolution_id,
            generation_id=generation_id,
            gen_index=outcome.gen_index,
            strategy_id=result.strategy_id,
            params_json=canonical_json(dict(member.phenotype.params)),
            kind="genome",
            origin=member.origin,
            genome_json=member.genome.canonical(),
            parent_candidate_id=member.parent_candidate_id,
            signature_json=genome_signature(member.genome).canonical(),
        )
        if member.mutations and member.parent_candidate_id is not None:
            store.add_mutations(
                member.candidate_id,
                [
                    {
                        "parent_candidate_id": member.parent_candidate_id,
                        "seq": index,
                        "category": mutation.category,
                        "operator": mutation.operator,
                        "target": mutation.target,
                        "before_json": mutation.before_json,
                        "after_json": mutation.after_json,
                        "rng_seed": mutation.rng_seed,
                    }
                    for index, mutation in enumerate(member.mutations)
                ],
            )
        store.score_candidate(
            member.candidate_id,
            run_id=result.run_id,
            fitness=result.fitness.fitness,
            base_score=result.fitness.base_score,
            penalty_product=result.fitness.penalty_product,
            components_json=canonical_json(dict(result.fitness.components)),
            penalties_json=canonical_json(dict(result.fitness.penalties)),
            gate_failure=result.fitness.gate_failure,
            rank=ranks.get(member.candidate_id),
            survived=member.candidate_id in survivors,
        )


def resume_point(store: ExperimentStore, evolution_id: str) -> int:
    """The first generation with no ``generation`` row (spec section 13.9).

    ``evolve resume`` re-enters here. Candidates already recorded stay recorded
    and their runs are served from the cache of section 11.2, so nothing is
    repeated and nothing is lost.
    """
    recorded = {int(row.gen_index) for row in store.generations_for(evolution_id)}
    index = 0
    while index in recorded:
        index += 1
    return index
