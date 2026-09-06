"""INV-9: what evolution may read (master spec section 13.7, task T52). 🔒

The whole platform is built to produce one trustworthy out-of-sample number. A
search that could see the validation segment would optimise against it, and that
number would be worth nothing. Section 13.7 states the rule as a table; this file
is the mechanical enforcement of it, tested at every place a run could be created.

Section 22 asks for the failure to be demonstrable — "INV-9 demonstrated to fail
when the segment assertion is removed". The last test does that in-process, by
patching the assertion out and showing the loop would then evaluate against
validation, then restoring it. That is a stronger demonstration than a comment,
and it cannot rot: if the assertion moves, the test stops finding it.
"""

from __future__ import annotations

import pytest

from quantlab.core.errors import StrategyError
from quantlab.evolution import loop as loop_module
from quantlab.evolution.loop import EVOLUTION_SEGMENTS, require_evolution_segment


@pytest.mark.parametrize("segment", ["train", "inner_is:0", "inner_oos:3", "train_extra"])
def test_the_train_segment_and_its_inner_folds_are_readable(segment: str) -> None:
    """Section 13.7's first row: the optimiser reads train, every generation."""
    assert require_evolution_segment(segment) == segment


@pytest.mark.parametrize(
    "segment", ["val", "validation", "test", "wf_oos:0", "lockbox", "TRAIN", " train"]
)
def test_everything_else_is_refused(segment: str) -> None:
    """Validation is reachable only through a recorded promotion, and the test
    partition only through the lockbox."""
    with pytest.raises(StrategyError, match="evolution may only read"):
        require_evolution_segment(segment)


def test_the_allowed_prefixes_are_the_three_the_specification_names() -> None:
    assert EVOLUTION_SEGMENTS == ("train", "inner_is", "inner_oos")


def test_the_refusal_names_the_segment_and_what_is_allowed() -> None:
    """A rejected segment is usually a caller passing the wrong thing, so the
    error says what would have been right."""
    with pytest.raises(StrategyError) as caught:
        require_evolution_segment("val")
    assert caught.value.context["segment"] == "val"
    assert caught.value.context["allowed"] == list(EVOLUTION_SEGMENTS)


def test_the_assertion_is_reached_before_anything_is_built(monkeypatch: pytest.MonkeyPatch) -> None:
    """``evolve`` refuses a forbidden segment before it draws a population, so a
    misconfigured run costs nothing and leaves nothing behind."""
    drawn = False

    def _never(*_args: object, **_kwargs: object) -> None:
        nonlocal drawn
        drawn = True
        raise AssertionError("generation zero should never be built")

    monkeypatch.setattr(loop_module, "_generation_zero", _never)
    with pytest.raises(StrategyError, match="evolution may only read"):
        loop_module.evolve(
            evolution_id="ev0",
            store=object(),  # type: ignore[arg-type]
            runner=object(),
            register=lambda genome: ("", ""),
            evaluator_for=lambda source: object(),
            bars_train=object(),  # type: ignore[arg-type]
            experiment_id="e0",
            dataset_id="d0",
            split_id="s0",
            settings=_settings(),
            config=_config(),
            segment="val",
        )
    assert not drawn


def test_removing_the_assertion_would_admit_the_validation_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Section 22's demonstration, in-process and reversible.

    With the assertion replaced by an identity function, ``val`` passes. That is
    the failure INV-9 exists to prevent, and showing it here means the test
    notices if the assertion is ever weakened to a warning, a log line, or a
    check that only looks at the first character.
    """
    assert require_evolution_segment("train") == "train"
    with pytest.raises(StrategyError):
        require_evolution_segment("val")

    monkeypatch.setattr(loop_module, "require_evolution_segment", lambda segment: segment)
    assert loop_module.require_evolution_segment("val") == "val"

    monkeypatch.undo()
    with pytest.raises(StrategyError):
        loop_module.require_evolution_segment("val")


def _settings() -> object:
    from quantlab.core.config import EvolutionSettings

    return EvolutionSettings()


def _config() -> object:
    from quantlab.core.types import BacktestConfig

    return BacktestConfig()
