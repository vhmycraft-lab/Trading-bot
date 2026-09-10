"""What a model is allowed to propose (master spec section 12.4).

The model proposes **JSON**, never Python. That is a security property before it
is a convenience one: a genome is a closed vocabulary of indicators, comparisons
and bounded parameters (section 9.6), so a malformed or hostile proposal fails
pydantic and costs one round trip, where free-form source would have to be
parsed, AST-checked and sandboxed before anyone could say the same.

Every model here sets ``extra="forbid"``. A provider that returns a field nobody
asked for has misunderstood the request, and silently dropping it would hide
that — the repair attempt of section 12.2 exists precisely to be told.
"""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from quantlab.core.genome import StrategyGenome

__all__ = [
    "STRUCTURAL_OPERATORS",
    "GenerationReview",
    "GenomeProposal",
    "MutationSuggestion",
    "StrategyProposal",
]

#: The structural operators of section 13.4, which is the only vocabulary a
#: guided-mutation suggestion may name. Held here rather than imported from
#: ``evolution`` because ``research`` may not import that layer (INV-8); the
#: agreement between the two lists is asserted in
#: ``tests/unit/test_schemas.py`` rather than assumed.
STRUCTURAL_OPERATORS: Final[tuple[str, ...]] = (
    "add_confirmation",
    "remove_confirmation",
    "modify_entry",
    "modify_exit",
    "add_filter",
    "remove_filter",
    "replace_indicator",
    "change_tree_mode",
)


class _Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hypothesis: str = Field(min_length=20, max_length=1000)
    expected_trade_frequency: Literal["hours", "days", "weeks"]
    notes: str = ""

    @field_validator("hypothesis")
    @classmethod
    def _hypothesis_says_something(cls, value: str) -> str:
        """A hypothesis is a claim about *why* an edge should exist.

        The length bound alone is satisfied by twenty spaces, and a proposal
        whose stated reasoning is blank is one nobody can review after the fact
        — which is most of what the field is for.
        """
        if not value.strip():
            raise ValueError("hypothesis must not be blank")
        return value


class StrategyProposal(_Proposal):
    """A full strategy module, as text (spec 1.0's form).

    Retained because section 12.4 still lists it, but it is **not** what the
    evolutionary loop asks for: ``GenomeProposal`` replaced it, for the reasons
    at the top of this file. Anything reaching this type still goes through the
    AST check and the sandbox like any other source.
    """

    code: str = Field(max_length=20_000)


class GenomeProposal(_Proposal):
    """A genome, validated by section 9.6 before it costs anything."""

    genome: StrategyGenome


class MutationSuggestion(BaseModel):
    """A structural operator the model suggests for one offspring (section 13.4).

    A *suggestion*: it is applied through the same typed operator as any other
    mutation, subject to the same validity repair, and recorded with
    ``mutation.suggested_by = 'llm'``. The model does not get to edit a genome
    directly, so a suggestion that would produce an invalid child is repaired or
    discarded exactly as a drawn one would be.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    operator: str
    target: str
    reasoning: str = ""

    @field_validator("operator")
    @classmethod
    def _operator_is_structural(cls, value: str) -> str:
        if value not in STRUCTURAL_OPERATORS:
            raise ValueError(
                f"{value!r} is not a structural operator; expected one of "
                + ", ".join(STRUCTURAL_OPERATORS)
            )
        return value


class GenerationReview(BaseModel):
    """Optional commentary on a finished generation (section 12.5).

    Its inputs are the INV-11 generation report — training and inner-fold
    numbers only — and its outputs are advisory. Nothing here decides survival,
    ranking or stopping; those are fitness-driven and deterministic (section 12).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    observations: str = ""
    suggested_directions: tuple[str, ...] = ()
    new_genomes: tuple[GenomeProposal, ...] = ()
