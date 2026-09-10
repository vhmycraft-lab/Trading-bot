"""Keeping the population from collapsing onto one strategy (spec section 13.5).

Without an explicit mechanism a population converges: the fittest candidate's
descendants fill every slot within a few generations, the search stops exploring,
and sixteen near-identical strategies produce sixteen near-identical — and
mutually uninformative — validation results.

Similarity combines structure and behaviour::

    similarity(a, b) = structural_weight · jaccard(signature(a), signature(b))
                     + behavioural_weight · agreement(position_frac(a), position_frac(b))

Behaviour is the more important half and is weighted accordingly. Two
structurally different strategies that enter and exit together are **not**
diverse, and their agreement is exactly what would mislead a reviewer into
thinking two independent methods had confirmed each other.

Three enforcement points, all here:

* :func:`select_survivors` — niching in step 4 of section 13.2. A candidate too
  similar to a fitter survivor already accepted is skipped, and the shortfall
  becomes extra immigrants rather than near-duplicates.
* :func:`population_diversity` and :func:`slot_plan` — the diversity floor.
  Below ``min_population_diversity`` the immigrant count rises and offspring fall
  to match. Survivors are never displaced.
* Reserved novelty — ``n_immigrants >= 1`` always, enforced at config load, so
  every generation contains at least one candidate that owes nothing to the
  current leader.
"""

from __future__ import annotations

import ast
import itertools
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from quantlab.core.config import DiversitySettings, EvolutionSettings
from quantlab.core.errors import StrategyError
from quantlab.core.genome import StrategyGenome
from quantlab.core.hashing import canonical_json, short_id

__all__ = [
    "BEHAVIOUR_QUANTUM",
    "STRUCTURAL_TRIPLE_FIELDS",
    "CandidateView",
    "Signature",
    "SlotPlan",
    "agreement",
    "behaviour_hash",
    "genome_signature",
    "jaccard",
    "population_diversity",
    "select_survivors",
    "signature_from_source",
    "similarity",
    "slot_plan",
]

#: What a structural triple names (spec section 13.5).
STRUCTURAL_TRIPLE_FIELDS: Final[tuple[str, str, str]] = ("kind", "name", "comparator")

#: ``position_frac`` is quantised before hashing, so two candidates whose sizing
#: differs in the twelfth decimal are still recognised as behavioural duplicates.
#: Coarse enough to absorb float noise, fine enough that half a position and a
#: whole one hash differently.
BEHAVIOUR_QUANTUM: Final[float] = 0.01

_EPS: Final[float] = 1e-12


@dataclass(frozen=True, slots=True)
class Signature:
    """A candidate's structure, as a multiset of triples plus its risk controls.

    Section 13.5: ``(component kind, indicator name, comparator)`` from the
    genome, plus the set of enabled risk controls. A *multiset* rather than a set
    — a strategy comparing two moving averages is structurally different from one
    comparing three, and collapsing the repeats would hide that.
    """

    triples: tuple[tuple[str, str, str], ...] = ()
    risk_controls: frozenset[str] = frozenset()

    def counts(self) -> Counter[tuple[str, ...]]:
        """Every element as one multiset, risk controls included.

        Risk controls join the same multiset rather than being scored separately:
        section 13.5 gives one Jaccard over "the triples plus the controls", and
        two similarities averaged some other way would be a different measure.
        """
        counter: Counter[tuple[str, ...]] = Counter(self.triples)
        counter.update(("risk", control) for control in self.risk_controls)
        return counter

    @property
    def is_empty(self) -> bool:
        return not self.triples and not self.risk_controls

    def canonical(self) -> str:
        """The stored form (``candidate.signature_json``, section 6)."""
        return canonical_json(
            {
                "triples": [list(triple) for triple in self.triples],
                "risk_controls": sorted(self.risk_controls),
            }
        )


def _enabled_risk_controls(genome: StrategyGenome) -> frozenset[str]:
    return frozenset(genome.risk.model_dump(exclude_none=True))


def genome_signature(genome: StrategyGenome) -> Signature:
    """The structural signature of a ``genome`` candidate (spec section 13.5).

    Every operand contributes one triple: what kind of component it is, what it
    names, and the comparator of the condition it appears in. Periods are
    deliberately absent — two moving-average crossovers at 20/50 and 25/60 are
    the same *structure*, and the behavioural half of the similarity is what
    tells them apart.
    """
    triples: list[tuple[str, str, str]] = []
    for condition in genome.conditions():
        for operand in condition.operands:
            triples.append((operand.kind, operand.name, condition.op))
    return Signature(triples=tuple(sorted(triples)), risk_controls=_enabled_risk_controls(genome))


def signature_from_source(source: str) -> Signature:
    """The structural signature of an ``opaque`` candidate, from its AST.

    Section 13.5 says an opaque candidate's signature "is derived from its AST".
    What is legible there is which indicators it calls and which comparators it
    uses, so those are what is read — by parsing, never by running it (INV-4).

    The two signature kinds are deliberately comparable: an opaque candidate that
    calls ``ctx.ind("sma", ...)`` inside a ``>`` comparison produces the same
    ``("indicator", "sma", ">")`` triple a genome would. That is what lets a
    population hold both kinds and still measure one diversity across it.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise StrategyError("cannot read a signature from source that does not parse") from exc

    bindings = _local_bindings(tree)
    triples: list[tuple[str, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or not node.ops:
            continue
        comparator = _OP_NAMES.get(type(node.ops[0]))
        if comparator is None:
            continue
        for operand in (node.left, *node.comparators):
            triples.extend(
                (kind, name, comparator) for kind, name in _operand_names(operand, bindings)
            )
    return Signature(triples=tuple(sorted(triples)), risk_controls=_risk_controls_in(tree))


def _local_bindings(tree: ast.Module) -> dict[str, list[tuple[str, str]]]:
    """Local names bound to an indicator or price read.

    A hand-written strategy almost never compares two calls directly — it reads
    ``fast = ctx.ind("sma", ...)`` and then compares ``fast > slow``. Without
    following that one hop, every opaque signature would come out empty and every
    pair of opaque candidates would look identical.

    One hop is enough and is where it stops: this is a similarity heuristic, not
    a dataflow analysis, and a wrong answer here costs a slightly mismeasured
    diversity rather than a wrong strategy.
    """
    bindings: dict[str, list[tuple[str, str]]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        found = _calls_in(node.value)
        if found:
            bindings[target.id] = found
    return bindings


#: Comparators that carry structural meaning. ``==`` and ``!=`` are absent on
#: purpose: the only place a strategy uses them is the ``x != x`` warm-up guard,
#: which says nothing about what the strategy trades.
_OP_NAMES: Final[Mapping[type[ast.cmpop], str]] = {
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}


def _operand_names(
    node: ast.expr, bindings: Mapping[str, list[tuple[str, str]]]
) -> list[tuple[str, str]]:
    """What one side of a comparison reads: a call, or a name bound to one."""
    if isinstance(node, ast.Name) and node.id in bindings:
        return list(bindings[node.id])
    return _calls_in(node)


def _calls_in(node: ast.expr) -> list[tuple[str, str]]:
    """Indicator calls and price reads inside an expression."""
    found: list[tuple[str, str]] = []
    for inner in ast.walk(node):
        if not isinstance(inner, ast.Call) or not isinstance(inner.func, ast.Attribute):
            continue
        first = inner.args[0] if inner.args else None
        if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
            continue
        if inner.func.attr == "ind":
            found.append(("indicator", first.value))
        elif inner.func.attr in ("value", "last", "column"):
            found.append(("price", first.value))
    found.extend(_random_draws(node))
    return found


def _random_draws(node: ast.expr) -> list[tuple[str, str]]:
    """Draws from ``ctx.rng`` — the one source of randomness section 9.1 allows.

    A strategy that decides by coin toss has a structure worth recording: without
    this, a random-entry strategy would present an empty signature and look
    structurally identical to any other strategy that compares nothing.
    """
    draws: list[tuple[str, str]] = []
    for inner in ast.walk(node):
        if (
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and isinstance(inner.func.value, ast.Attribute)
            and inner.func.value.attr == "rng"
        ):
            draws.append(("random", inner.func.attr))
    return draws


def _risk_controls_in(tree: ast.Module) -> frozenset[str]:
    """Risk controls a source declares as keyword arguments to ``RiskSpec``."""
    controls: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "RiskSpec"
        ):
            controls.update(
                keyword.arg
                for keyword in node.keywords
                if keyword.arg is not None
                and not (isinstance(keyword.value, ast.Constant) and keyword.value.value is None)
            )
    return frozenset(controls)


# ---------------------------------------------------------------------------
# the two halves of similarity
# ---------------------------------------------------------------------------
def jaccard(left: Signature, right: Signature) -> float:
    """Multiset Jaccard: ``Σ min(counts) / Σ max(counts)`` (spec section 13.5).

    The generalised form, because section 13.5 says *multiset*: on plain sets it
    is the ordinary Jaccard index, and on multisets it keeps the information that
    a strategy comparing three moving averages is not the one comparing two.

    Two empty signatures score 1.0 — they are structurally identical, having no
    structure — rather than dividing by zero.
    """
    a, b = left.counts(), right.counts()
    if not a and not b:
        return 1.0
    keys = set(a) | set(b)
    intersection = sum(min(a.get(key, 0), b.get(key, 0)) for key in keys)
    union = sum(max(a.get(key, 0), b.get(key, 0)) for key in keys)
    return intersection / union if union else 1.0


def agreement(left: Sequence[float] | np.ndarray, right: Sequence[float] | np.ndarray) -> float:
    """How often two candidates hold the same *sign* of exposure, while either is in.

    Sign, not size: two strategies both long throughout agree about the market
    even if one sizes at half the other's fraction, and section 13.5's concern is
    whether they would confirm each other, not whether they would earn the same.

    Bars on which **both are flat** are excluded from the numerator and the
    denominator. Section 13.5 names the quantity it wants — strategies that
    "enter and exit together" — and agreeing to stay out of the market is not
    that. Counting mutual inactivity puts a floor under the measure that rises
    with how selective the strategies are: on this repository's own baselines it
    scores two unrelated strategies at 0.66, which is nearer their score against
    themselves (1.0) than against a strategy they share nothing with.

    A candidate that is **never in the market at all** has not demonstrated a
    different behaviour; it has demonstrated no behaviour. Its agreement with
    anything is therefore *unmeasurable*, and unmeasurable is scored 1.0 —
    identical — for exactly the reason :func:`similarity` gives when neither
    candidate has been evaluated: assuming they differ would be an unearned claim
    to diversity.

    Scoring it 0.0 instead — as "maximally different" — let dead candidates prop
    up the population diversity measure. Eight behavioural clones score 0.0000,
    a total collapse; adding two structurally distinct do-nothing candidates
    lifted that to 0.3644, above the 0.35 floor, so the immigrant boost stopped
    firing on precisely the population it exists to rescue. The inactive
    candidates also survived niching, because nothing was similar to them.

    Two *active* candidates that are never both in the market still score 0.0:
    they genuinely traded differently, which is the case this measure is for.

    Raises:
        StrategyError: the two series have different lengths. Comparing a prefix
            would quietly report agreement over bars one of them never saw.
    """
    a = np.sign(np.asarray(left, dtype="float64"))
    b = np.sign(np.asarray(right, dtype="float64"))
    if a.shape != b.shape:
        raise StrategyError(
            "position series must cover the same bars", left=int(a.size), right=int(b.size)
        )
    if a.size == 0:
        return 1.0
    # An entirely flat series carries no behaviour to compare. This covers the
    # both-flat case too, which is why it is tested before `live`.
    if not np.any(a) or not np.any(b):
        return 1.0
    live = (a != 0) | (b != 0)
    n_live = int(np.count_nonzero(live))
    if n_live == 0:  # pragma: no cover - implied by the emptiness test above
        return 1.0
    return float(np.count_nonzero((a == b) & live) / n_live)


def behaviour_hash(
    positions: Sequence[float] | np.ndarray, *, quantum: float = BEHAVIOUR_QUANTUM
) -> str:
    """A digest of the quantised ``position_frac`` series (spec section 13.5).

    Lets exact behavioural duplicates be detected without an O(n²) scan: equal
    hashes mean equal quantised series, so the pair need never be compared.

    The implication runs one way only, and that is the useful direction. The hash
    is magnitude-sensitive while :func:`agreement` is sign-based, so an equal hash
    implies perfect agreement but perfect agreement does not imply an equal hash —
    a candidate sizing at a quarter and one sizing at a whole agree on every bar
    and hash differently. A cheap pre-filter must never claim two candidates are
    the same when they are not; missing a pair only costs the comparison it was
    trying to save.
    """
    values = np.asarray(positions, dtype="float64")
    if quantum <= _EPS:
        raise StrategyError("the behaviour quantum must be positive", quantum=quantum)
    quantised = np.round(values / quantum).astype("int64")
    return short_id(",".join(str(int(value)) for value in quantised))


# ---------------------------------------------------------------------------
# candidates, as diversity sees them
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CandidateView:
    """What diversity needs of a candidate: its rank, structure and behaviour.

    Deliberately not the store row. Similarity is a property of what a candidate
    *is* and *did*, and keeping it free of the database is what lets the whole
    mechanism be tested without one.
    """

    candidate_id: str
    fitness: float = 0.0
    signature: Signature = field(default_factory=Signature)
    positions: np.ndarray | None = None
    behaviour: str | None = None

    @classmethod
    def from_genome(
        cls,
        candidate_id: str,
        genome: StrategyGenome,
        positions: Sequence[float] | np.ndarray | None = None,
        *,
        fitness: float = 0.0,
    ) -> CandidateView:
        array = None if positions is None else np.asarray(positions, dtype="float64")
        return cls(
            candidate_id=candidate_id,
            fitness=fitness,
            signature=genome_signature(genome),
            positions=array,
            behaviour=None if array is None else behaviour_hash(array),
        )


def similarity(
    left: CandidateView, right: CandidateView, settings: DiversitySettings | None = None
) -> float:
    """Section 13.5's weighted combination of structure and behaviour.

    When neither candidate has a position series — before either has been
    evaluated — the behavioural half cannot be measured. It is then taken as
    *identical* rather than skipped: two unevaluated candidates have shown no
    difference, and niching on the structural half alone is the conservative
    reading. Assuming they differ would be an unearned claim to diversity.
    """
    weights = settings or DiversitySettings()
    structural = jaccard(left.signature, right.signature)
    behavioural = 1.0
    if left.positions is not None and right.positions is not None:
        behavioural = agreement(left.positions, right.positions)
    return weights.structural_weight * structural + weights.behavioural_weight * behavioural


# ---------------------------------------------------------------------------
# enforcement
# ---------------------------------------------------------------------------
def select_survivors(
    ranked: Sequence[CandidateView], *, n_survivors: int, settings: DiversitySettings | None = None
) -> list[CandidateView]:
    """Niching, step 4 of section 13.2.

    Walk the ranking and accept a candidate unless its similarity to a survivor
    already accepted exceeds ``max_pairwise_similarity``. ``ranked`` must already
    be in the order section 13.2 step 3 defines; this function preserves it and
    never reorders, so the fittest candidate is always accepted.

    Returns fewer than ``n_survivors`` when the quota cannot be filled without
    near-duplicates. That shortfall is intentional and becomes extra immigrants
    (section 13.2, step 4); silently topping it up with the duplicates just
    rejected would undo the whole mechanism.
    """
    limits = settings or DiversitySettings()
    survivors: list[CandidateView] = []
    for view in ranked:
        if len(survivors) >= n_survivors:
            break
        if any(
            similarity(view, accepted, limits) > limits.max_pairwise_similarity
            for accepted in survivors
        ):
            continue
        survivors.append(view)
    return survivors


def population_diversity(
    views: Sequence[CandidateView], settings: DiversitySettings | None = None
) -> float:
    """``1 - mean pairwise similarity`` (spec section 13.5), stored per generation.

    A population of one is perfectly diverse by this definition — there is no
    pair to disagree — which is the honest reading: the measure is about spread,
    and one point has none to report.
    """
    limits = settings or DiversitySettings()
    pairs = list(itertools.combinations(views, 2))
    if not pairs:
        return 1.0
    mean = sum(similarity(left, right, limits) for left, right in pairs) / len(pairs)
    return 1.0 - mean


@dataclass(frozen=True, slots=True)
class SlotPlan:
    """How one generation's non-survivor slots are divided (section 13.2, step 6)."""

    n_survivors: int
    n_offspring: int
    n_immigrants: int
    boosted: bool = False

    @property
    def total(self) -> int:
        return self.n_survivors + self.n_offspring + self.n_immigrants


def slot_plan(
    settings: EvolutionSettings, *, diversity: float, n_survivors: int | None = None
) -> SlotPlan:
    """Divide a generation's slots, boosting immigrants when diversity is low.

    Two adjustments, in this order:

    1. A survivor quota that niching could not fill (section 13.2, step 4)
       becomes extra immigrants — never extra offspring, because offspring are
       mutations *of survivors* and there were not enough distinct ones.
    2. Below ``min_population_diversity``, immigrants rise by ``immigrant_boost``
       and offspring fall to match, capped at ``max_immigrants``. Survivors are
       never displaced, which is why config load requires
       ``max_immigrants <= n_offspring + n_immigrants``.

    The population always totals ``population_size``: section 13.2 step 7 asserts
    it, and a generation that quietly resized would make every downstream
    deflated Sharpe ratio wrong.
    """
    survivors = settings.n_survivors if n_survivors is None else max(0, n_survivors)
    survivors = min(survivors, settings.n_survivors)
    open_slots = settings.population_size - survivors

    immigrants = settings.n_immigrants + (settings.n_survivors - survivors)
    boosted = diversity < settings.diversity.min_population_diversity
    if boosted:
        immigrants += settings.diversity.immigrant_boost
    immigrants = min(immigrants, settings.diversity.max_immigrants, open_slots)
    immigrants = max(immigrants, min(settings.n_immigrants, open_slots))

    return SlotPlan(
        n_survivors=survivors,
        n_offspring=open_slots - immigrants,
        n_immigrants=immigrants,
        boosted=boosted,
    )


def duplicate_groups(views: Iterable[CandidateView]) -> dict[str, list[str]]:
    """Candidates sharing a ``behaviour_hash``, by hash (spec section 13.5).

    The cheap half of duplicate detection: equal hashes mean equal quantised
    position series, so those pairs need no similarity computed at all.
    """
    groups: dict[str, list[str]] = {}
    for view in views:
        if view.behaviour is not None:
            groups.setdefault(view.behaviour, []).append(view.candidate_id)
    return {digest: ids for digest, ids in groups.items() if len(ids) > 1}
