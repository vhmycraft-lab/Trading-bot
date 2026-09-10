"""What a model may propose, and what it may be shown (spec section 12.4, 12.5).

The schemas are the first gate a proposal meets, and the cheapest: a genome that
fails pydantic has cost one round trip, where the same idea reaching the
population costs an evaluation the multiple-testing correction of section 14.4
then has to charge for. So the assertions here are mostly about *refusing*.

The templates are tested for two things the rendering cannot check for itself —
that the file set matches the one section 12.5 defines, and that the system
prompt actually says the things section 12.5 requires it to say. A prompt that
quietly stopped explaining the rejection reasons would still render, still hash,
and still produce proposals; it would just produce worse ones for a reason
nobody could see.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from quantlab.core.fitness import GATE_IDS
from quantlab.evolution.mutation import STRUCTURAL_OPERATORS as EVOLUTION_OPERATORS
from quantlab.research.prompts import (
    PROMPT_DIR,
    TEMPLATE_NAMES,
    known_templates,
    render,
    template_sha256,
    template_text,
)
from quantlab.research.redaction import scan_for_leaks
from quantlab.research.schemas import (
    STRUCTURAL_OPERATORS,
    GenerationReview,
    GenomeProposal,
    MutationSuggestion,
)


# ---------------------------------------------------------------------------
# the operator vocabulary
# ---------------------------------------------------------------------------
def test_the_suggestible_operators_are_exactly_the_structural_ones() -> None:
    """``research`` may not import ``evolution`` (INV-8), so the list is copied —
    and a copy that drifts is worse than no list at all: the model would be told
    to suggest an operator the loop cannot apply, and every such suggestion would
    be silently discarded while appearing to have been honoured."""
    assert STRUCTURAL_OPERATORS == EVOLUTION_OPERATORS


def test_an_operator_outside_the_vocabulary_is_refused() -> None:
    with pytest.raises(ValidationError, match="not a structural operator"):
        MutationSuggestion(operator="rewrite_everything", target="entry")


def test_a_parameter_operator_is_not_a_structural_one() -> None:
    """Guards the test above against a vocabulary that quietly widened: guided
    mutation is over *structure* (section 13.4), and a model allowed to nudge
    parameters would be running a second, unaccounted search."""
    with pytest.raises(ValidationError):
        MutationSuggestion(operator="perturb", target="params.period")


# ---------------------------------------------------------------------------
# proposals
# ---------------------------------------------------------------------------
def test_a_proposal_without_a_hypothesis_is_refused() -> None:
    with pytest.raises(ValidationError):
        GenomeProposal(hypothesis="too short", genome=None, expected_trade_frequency="days")  # type: ignore[arg-type]


def test_a_blank_hypothesis_that_clears_the_length_bound_is_still_refused() -> None:
    """Twenty spaces satisfy ``min_length=20``. A proposal whose stated reasoning
    is blank is one nobody can review after the fact, which is most of what the
    field is for."""
    with pytest.raises(ValidationError, match="blank"):
        GenomeProposal(
            hypothesis=" " * 40,
            genome=None,  # type: ignore[arg-type]
            expected_trade_frequency="days",
        )


def test_an_unexpected_field_is_refused() -> None:
    """``extra="forbid"`` everywhere. A provider that returns a field nobody
    asked for has misunderstood the request, and dropping it silently would hide
    exactly what the repair attempt of section 12.2 exists to be told."""
    with pytest.raises(ValidationError):
        GenerationReview(observations="fine", confidence=0.9)  # type: ignore[call-arg]


def test_a_review_defaults_to_saying_nothing() -> None:
    """The model does not decide survival, ranking or stopping (section 12), so
    an empty review is a valid one and must not be an error."""
    review = GenerationReview()
    assert review.new_genomes == ()
    assert review.suggested_directions == ()


# ---------------------------------------------------------------------------
# templates
# ---------------------------------------------------------------------------
def test_the_templates_on_disk_are_the_ones_the_spec_defines() -> None:
    """``review.md`` from spec 1.0 is gone along with ``ReviewDecision``."""
    assert known_templates() == tuple(sorted(TEMPLATE_NAMES))
    assert not (PROMPT_DIR / "review.md").exists()


def test_the_system_prompt_explains_every_rejection_reason() -> None:
    """Section 12.5: the system prompt MUST state "the plain-language meaning of
    each fitness gate and rejection reason". A model told only the gate's *name*
    cannot avoid it — ``F_CONCENTRATION`` is not self-explanatory to anything."""
    text = template_text("system")
    for gate in GATE_IDS:
        assert gate in text, f"the system prompt never mentions {gate}"
    assert "10 basis points" in text
    assert "six parameters" in text.lower()


def test_the_system_prompt_says_the_model_will_never_see_validation_or_test() -> None:
    """Section 12.5, and the reason INV-11 is worth stating to the model at all:
    a model that thinks the numbers are merely withheld will speculate about
    them, and speculation about a partition is a weak form of seeing it."""
    text = template_text("system").lower()
    assert "never" in text
    assert "validation" in text
    assert "test period" in text


def test_no_shipped_template_would_fail_its_own_leak_scan() -> None:
    """The templates are prompts too.

    Section 14.5's scan is normally applied to an assembled prompt, but the
    largest fixed chunk of every prompt is a file in this repository — and a
    sentence added to ``system.md`` in good faith is exactly how a partition
    name gets into every prompt at once. Scanned under the *strict* reading,
    since the system prompt is used during an active run.
    """
    for name in TEMPLATE_NAMES:
        problems = scan_for_leaks(template_text(name), forbid_validation=True, environ={})
        assert problems == [], f"{name}.md: {problems}"


def test_a_template_hash_is_stable_and_distinct() -> None:
    """``llm_interaction.prompt_template_sha256`` is only evidence if it changes
    when the template does and not otherwise."""
    hashes = {name: template_sha256(name) for name in TEMPLATE_NAMES}
    assert len(set(hashes.values())) == len(hashes)
    assert hashes["system"] == template_sha256("system")


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def test_rendering_fills_every_placeholder() -> None:
    text = render("repair", {"error": "field required: genome.entry"})
    assert "{{" not in text
    assert "field required" in text


def test_an_unfilled_placeholder_is_refused() -> None:
    """A ``{{market}}`` left in a sent prompt is a caller that forgot an
    argument, and the model would answer the literal braces — producing a
    proposal that looks fine and was made in the dark."""
    with pytest.raises(KeyError, match="unfilled placeholder"):
        render("propose_genome", {"n": "3"})


def test_a_value_for_a_placeholder_that_does_not_exist_is_refused() -> None:
    """The other direction, and the one a rename produces: the caller believes it
    supplied the market description, the template no longer asks for it, and
    substitution alone would drop it without a word."""
    with pytest.raises(KeyError, match="no placeholder for"):
        render("repair", {"error": "x", "market": "BTC/USDT"})


def test_a_substituted_value_cannot_smuggle_in_a_placeholder() -> None:
    """One pass, and substituted text is never re-scanned. Otherwise a
    population signature containing ``{{indicators}}`` — which a model could
    propose as a genome name — would expand on the next render."""
    text = render("repair", {"error": "{{indicators}}"})
    assert "{{indicators}}" in text


def test_an_unknown_template_is_refused() -> None:
    with pytest.raises(KeyError, match="not a prompt template"):
        render("review", {})
