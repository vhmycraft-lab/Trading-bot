"""Lineage: the record of how a candidate came to exist (spec section 13.6).

Every candidate except a seed names a parent, and every mutation that produced it
is stored in order. INV-10 makes that a *checkable* property rather than a
narrative: re-applying a child's recorded mutations to its parent's genome must
reproduce the child byte for byte. :func:`verify_run` is that check, over a whole
stored evolution run.

The check is what turns lineage from a reporting feature into a guarantee. A tree
nobody can replay is a picture of what someone believes happened; a tree that
replays is what happened.

Everything here reads. Nothing in this module writes to the store — the loop of
T52 records candidates and mutations as it produces them, and lineage is how the
record is read back, audited and drawn.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from quantlab.core.errors import StoreError, StrategyError
from quantlab.core.genome import StrategyGenome, genome_id
from quantlab.evolution.mutation import AppliedMutation, Phenotype
from quantlab.evolution.mutation import replay as replay_mutations
from quantlab.ports.store import ExperimentStore

__all__ = [
    "LineageNode",
    "LineageTree",
    "ReplayReport",
    "ancestry",
    "descendants",
    "genome_of",
    "lineage_tree",
    "mutations_of",
    "replay",
    "verify_candidate",
    "verify_run",
]


def ancestry(store: ExperimentStore, candidate_id: str) -> list[Any]:
    """Every ancestor of a candidate, oldest first, ending with itself.

    A generation-4 candidate returns five rows and the first of them is a seed —
    a candidate with no parent. That is INV-10's structural half: the chain
    reaches a seed, and it does so without a cycle.
    """
    return list(store.ancestry(candidate_id))


def descendants(store: ExperimentStore, candidate_id: str) -> list[Any]:
    """Every candidate reachable by following parent links down from this one."""
    return list(store.descendants(candidate_id))


def mutations_of(store: ExperimentStore, candidate_id: str) -> list[AppliedMutation]:
    """The stored edits that produced a candidate, as mutation objects.

    Rows come back as they were written — append-only, never revised — and are
    turned into the same type :func:`quantlab.evolution.mutation.mutate` produced,
    so replay runs the identical code path the child came from.
    """
    return [
        AppliedMutation(
            category=row.category,
            operator=row.operator,
            target=row.target,
            before_json=row.before_json,
            after_json=row.after_json,
            rng_seed=int(row.rng_seed),
        )
        for row in store.mutations_for(candidate_id)
    ]


def genome_of(candidate: Any) -> StrategyGenome:
    """The genome a candidate row carries.

    A stored genome that will not parse or will not validate is reported as a
    :class:`StrategyError`, which :func:`verify_run` records as a mismatch. It
    used to escape as a pydantic ``ValidationError`` and abort the audit — so
    INV-10's verifier crashed on exactly the corruption it exists to detect,
    and one bad row took the whole run's report with it. Corrupt is a verdict
    this function is allowed to reach, not an error it should die of.

    Raises:
        StoreError: the candidate is ``opaque`` and has none. Section 9.6 gives
            opaque candidates parameter mutation only, so asking one to replay a
            structural edit is a question about the wrong candidate.
        StrategyError: the stored genome is not a valid genome.
    """
    document = getattr(candidate, "genome_json", None)
    if not document:
        raise StoreError(
            "this candidate has no genome to replay",
            candidate_id=getattr(candidate, "candidate_id", "?"),
            kind=getattr(candidate, "kind", "?"),
        )
    try:
        return StrategyGenome.model_validate(json.loads(document))
    except (ValidationError, ValueError) as exc:
        raise StrategyError(
            "the stored genome is not a valid genome",
            candidate_id=getattr(candidate, "candidate_id", "?"),
        ) from exc


def _params_of(candidate: Any) -> Mapping[str, Any]:
    document = getattr(candidate, "params_json", None) or "{}"
    values = json.loads(document)
    return values if isinstance(values, dict) else {}


def _phenotype_of(candidate: Any) -> Phenotype:
    """A candidate row as mutation sees it (see :class:`Phenotype`)."""
    if getattr(candidate, "kind", "genome") == "opaque":
        return Phenotype(kind="opaque", params=_params_of(candidate))
    return Phenotype.from_genome(genome_of(candidate), _params_of(candidate))


def replay(store: ExperimentStore, candidate_id: str) -> StrategyGenome:
    """Rebuild a candidate's genome from its parent and its recorded mutations.

    Section 13.6's ``replay``. Nothing about the original run is consulted beyond
    the parent's genome and the edits: each mutation carries the value at its path
    *after* the edit, so this needs none of the randomness that produced them.

    Raises:
        StoreError: the candidate is unknown, is a seed (which has no parent to
            replay from), or is opaque.
    """
    chain = ancestry(store, candidate_id)
    if not chain:
        raise StoreError("no such candidate", candidate_id=candidate_id)
    candidate = chain[-1]
    parent_id = getattr(candidate, "parent_candidate_id", None)
    if parent_id is None:
        raise StoreError("a seed has no parent to replay from", candidate_id=candidate_id)
    if len(chain) < 2:  # pragma: no cover - a parent id implies a parent row
        raise StoreError("lineage is broken: the parent is missing", candidate_id=candidate_id)

    rebuilt = replay_mutations(_phenotype_of(chain[-2]), mutations_of(store, candidate_id))
    if rebuilt.genome is None:  # pragma: no cover - an opaque parent raises above
        raise StoreError("replay produced no genome", candidate_id=candidate_id)
    return rebuilt.genome


@dataclass(frozen=True, slots=True)
class ReplayReport:
    """What :func:`verify_run` found — INV-10, over a whole stored run."""

    checked: tuple[str, ...] = ()
    #: Candidates whose replay did not reproduce the stored genome.
    mismatched: tuple[str, ...] = ()
    #: Candidates skipped, and why: seeds, opaque candidates, missing mutations.
    skipped: Mapping[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.mismatched

    @property
    def n_checked(self) -> int:
        return len(self.checked)


def verify_candidate(store: ExperimentStore, candidate_id: str) -> bool:
    """True when replaying a candidate reproduces its stored genome (INV-10)."""
    chain = ancestry(store, candidate_id)
    if not chain:
        raise StoreError("no such candidate", candidate_id=candidate_id)
    try:
        stored = genome_of(chain[-1])
    except StrategyError:
        # A stored genome that will not validate cannot equal a replayed one.
        # Returning False is the answer to the question asked; raising here made
        # INV-10's verifier die on exactly the corruption it exists to detect.
        return False
    return genome_id(replay(store, candidate_id)) == genome_id(stored)


def verify_run(store: ExperimentStore, evolution_id: str) -> ReplayReport:
    """Check INV-10 across every genome candidate of one evolution run.

    Seeds and opaque candidates are *skipped with a reason* rather than passed
    silently: a report that counted them as verified would let a run of nothing
    but seeds claim a clean bill of health.
    """
    checked: list[str] = []
    mismatched: list[str] = []
    skipped: dict[str, str] = {}

    for candidate in store.candidates_for(evolution_id):
        candidate_id = candidate.candidate_id
        if getattr(candidate, "kind", "genome") == "opaque":
            skipped[candidate_id] = "opaque: parameter mutation only (section 9.6)"
            continue
        if getattr(candidate, "parent_candidate_id", None) is None:
            skipped[candidate_id] = "seed: nothing to replay from"
            continue
        if not store.mutations_for(candidate_id):
            skipped[candidate_id] = "no mutations recorded"
            continue
        checked.append(candidate_id)
        try:
            if not verify_candidate(store, candidate_id):
                mismatched.append(candidate_id)
        except (StoreError, StrategyError):
            mismatched.append(candidate_id)

    return ReplayReport(checked=tuple(checked), mismatched=tuple(mismatched), skipped=dict(skipped))


# ---------------------------------------------------------------------------
# the tree, for the dashboard (section 22, T56)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class LineageNode:
    """One candidate in the tree, with the ids of its children."""

    candidate_id: str
    gen_index: int
    origin: str
    kind: str
    fitness: float | None
    survived: bool
    parent_candidate_id: str | None
    children: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LineageTree:
    """A whole run's parent-to-child structure, plus per-generation summaries."""

    evolution_id: str
    nodes: Mapping[str, LineageNode]
    roots: tuple[str, ...]
    #: ``gen_index -> {best, median, n}`` over the generation's defined fitnesses.
    generations: Mapping[int, Mapping[str, float]]

    def path_to_root(self, candidate_id: str) -> list[str]:
        """The chain from a candidate up to its seed, child first.

        Raises:
            StoreError: the chain revisits a candidate. INV-10 requires the parent
                chain to reach a seed without cycles, and looping forever while
                drawing a dashboard is the worst way to discover it has not.
        """
        seen: set[str] = set()
        path: list[str] = []
        current: str | None = candidate_id
        while current is not None:
            if current in seen:
                raise StoreError("lineage contains a cycle", candidate_id=current)
            node = self.nodes.get(current)
            if node is None:
                raise StoreError("lineage names a candidate that is not here", candidate_id=current)
            seen.add(current)
            path.append(current)
            current = node.parent_candidate_id
        return path


def lineage_tree(store: ExperimentStore, evolution_id: str) -> LineageTree:
    """Build the parent-to-child tree of one run (spec section 13.6)."""
    rows = list(store.candidates_for(evolution_id))
    children: dict[str, list[str]] = {}
    for row in rows:
        parent = getattr(row, "parent_candidate_id", None)
        if parent is not None:
            children.setdefault(parent, []).append(row.candidate_id)

    nodes = {
        row.candidate_id: LineageNode(
            candidate_id=row.candidate_id,
            gen_index=int(row.gen_index),
            origin=row.origin,
            kind=row.kind,
            fitness=None if row.fitness is None else float(row.fitness),
            survived=bool(row.survived),
            parent_candidate_id=getattr(row, "parent_candidate_id", None),
            children=tuple(sorted(children.get(row.candidate_id, ()))),
        )
        for row in rows
    }
    roots = tuple(
        sorted(node.candidate_id for node in nodes.values() if node.parent_candidate_id is None)
    )
    return LineageTree(
        evolution_id=evolution_id,
        nodes=nodes,
        roots=roots,
        generations=_generation_summaries(rows),
    )


def _generation_summaries(rows: Sequence[Any]) -> dict[int, dict[str, float]]:
    """Best, median and count of *defined* fitness, per generation.

    Rejected candidates are excluded from best and median but counted: a
    generation of sixteen where twelve were rejected is a different picture from
    one where four were, and reporting only the survivors' median would hide it.
    """
    by_generation: dict[int, list[float]] = {}
    counts: dict[int, int] = {}
    for row in rows:
        index = int(row.gen_index)
        counts[index] = counts.get(index, 0) + 1
        if row.fitness is not None:
            by_generation.setdefault(index, []).append(float(row.fitness))

    summaries: dict[int, dict[str, float]] = {}
    for index, count in sorted(counts.items()):
        scores = sorted(by_generation.get(index, []))
        summary = {"n": float(count), "n_scored": float(len(scores))}
        if scores:
            summary["best"] = scores[-1]
            middle = len(scores) // 2
            summary["median"] = (
                scores[middle] if len(scores) % 2 else (scores[middle - 1] + scores[middle]) / 2.0
            )
        summaries[index] = summary
    return summaries
