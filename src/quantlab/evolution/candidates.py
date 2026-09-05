"""Turning a genome into something a run can cite (master spec sections 9.6, 13).

A genome is a structure; a run cites a ``strategy_version``. This module is the
one bridge between them, and it is deliberately the only one: the genome is
compiled exactly once, by :func:`~quantlab.evolution.compiler.compile_genome`, and
registered through the same :class:`~quantlab.strategies_io.loader.StrategyLoader`
that admits human-written strategies. Generated code gets no shortcut past the AST
check, the source store or the version row — the compiler makes valid strategies
by construction, and defence in depth means not trusting that claim (INV-4).

Section 9.6's two candidate kinds meet here. A ``genome`` candidate carries the
structure it was compiled from and admits both parameter and structural mutation.
An ``opaque`` candidate is free-form Python with no genome, and admits parameter
mutation only; it is registered through the loader directly and never reaches
this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from quantlab.core.genome import StrategyGenome, genome_id
from quantlab.evolution.compiler import compile_genome
from quantlab.strategies_io.loader import LoadedStrategy, StrategyLoader

__all__ = ["FAMILY_ORIGINS", "GenomeCandidate", "register_genome"]


@dataclass(frozen=True, slots=True)
class GenomeCandidate:
    """A registered strategy version and the genome it was compiled from.

    Both identities are kept because they answer different questions.
    ``genome_id`` is the identity of the *structure*, so two candidates that
    mutated to the same shape are recognisably the same candidate.
    ``strategy.strategy_id`` is the hash of the compiled bytes, and is what the
    run identity of section 11.2 is built from.
    """

    genome: StrategyGenome
    genome_id: str
    strategy: LoadedStrategy

    @property
    def genome_json(self) -> str:
        """The canonical document stored in ``candidate.genome_json`` (section 6)."""
        return self.genome.canonical()

    @property
    def strategy_id(self) -> str:
        return self.strategy.strategy_id


#: What ``strategy_family.origin`` may say (section 6).
FAMILY_ORIGINS: Final[frozenset[str]] = frozenset({"human", "llm"})


def register_genome(
    loader: StrategyLoader,
    genome: StrategyGenome,
    *,
    family: str | None = None,
    origin: str = "human",
    author: str = "evolution",
    parent_strategy_id: str | None = None,
    llm_interaction_id: str | None = None,
) -> GenomeCandidate:
    """Compile ``genome`` and register the result as a strategy version.

    Idempotent by construction: compilation is deterministic, so re-registering
    the same genome produces the same bytes, the same ``strategy_id`` and the same
    row — which is what lets a survivor be carried into the next generation
    without being re-evaluated (section 11.2).

    ``origin`` is the family's, and section 6 admits two values. The axis it
    records is whether a language model wrote the thing, which is the one the
    audit trail and section 12's review cares about — so a genome the operator
    library drew or mutated is ``'human'``, however machine-made it is, and only
    a genome an LLM proposed is ``'llm'``. The finer provenance — drawn, mutated,
    promoted — belongs on ``candidate.origin`` (section 6), which is per
    candidate rather than per family. ``author`` carries it on the version row.

    Raises:
        ValueError: ``origin`` is not one section 6 allows. The database would
            refuse it anyway, several statements later and with a message about a
            check constraint rather than about provenance.
    """
    if origin not in FAMILY_ORIGINS:
        raise ValueError(f"family origin must be one of {sorted(FAMILY_ORIGINS)}, not {origin!r}")
    source = compile_genome(genome)
    registered = loader.load_source(
        source,
        family=family or genome.name,
        origin=origin,
        author=author,
        parent_strategy_id=parent_strategy_id,
        llm_interaction_id=llm_interaction_id,
    )
    return GenomeCandidate(genome=genome, genome_id=genome_id(genome), strategy=registered)
