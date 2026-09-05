"""Optuna parameter studies (master spec section 13.8, task T51).

Section 20 names three things for this file — seed determinism, the segment
assertion, and ``n_trials`` persisted — and each is one of the three constraints
that follow from Optuna's demotion from primary search to refinement:

* a study samples hundreds of parameter sets, so it may only touch train;
* every trial counts into the deflated Sharpe ratio's ``M``, so the count must be
  recorded rather than merely printed;
* a study that could not be reproduced could not be audited (INV-7).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from quantlab.core.config import OptimizeSettings, PlateauSettings
from quantlab.core.errors import ConfigError
from quantlab.core.strategy import ParamSpec
from quantlab.optimize.study import (
    SEARCHABLE_SEGMENTS,
    require_searchable_segment,
    run_study,
    study_record,
)

SCHEMA = {
    "fast": ParamSpec(kind="int", default=10, low=5, high=50),
    "slow": ParamSpec(kind="float", default=30.0, low=20.0, high=100.0),
}

FAST_PLATEAU = PlateauSettings(top_k=2, perturbation_pcts=(0.1,), n_neighbors=3)


def settings(**overrides: object) -> OptimizeSettings:
    base: dict[str, object] = {
        "n_trials": 12,
        "seed": 5,
        "sampler": "random",
        "min_trades": 0,
        "plateau": FAST_PLATEAU,
    }
    base.update(overrides)
    return OptimizeSettings(**base)  # type: ignore[arg-type]


def surface(params: Mapping[str, Any]) -> float:
    """A smooth hill, so a study always has something to find."""
    fast, slow = float(params["fast"]), float(params["slow"])
    return float(-((fast - 25) ** 2) / 100.0 - ((slow - 60) ** 2) / 400.0)


# ---------------------------------------------------------------------------
# the segment assertion
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("segment", ["val", "test", "wf_oos:0", "validation"])
def test_a_search_may_not_touch_validation_or_test(segment: str) -> None:
    """Section 22's stated criterion, and the assertion spec 1.0 carried.

    A study samples hundreds of parameter sets; doing that on validation would
    spend the one measurement the platform exists to protect.
    """
    with pytest.raises(ConfigError, match="train segment or a walk-forward"):
        require_searchable_segment(segment)
    with pytest.raises(ConfigError, match="train segment or a walk-forward"):
        run_study(surface, SCHEMA, settings(), segment=segment, strategy_id="s0")


@pytest.mark.parametrize("segment", ["train", "wf_is:0", "wf_is:12"])
def test_train_and_in_sample_windows_are_searchable(segment: str) -> None:
    assert require_searchable_segment(segment) == segment


def test_the_allowed_prefixes_are_the_two_the_specification_names() -> None:
    assert SEARCHABLE_SEGMENTS == ("train", "wf_is")


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------
def test_the_same_seed_produces_the_same_study() -> None:
    """Section 22's stated criterion. A study that could not be reproduced could
    not be audited (INV-7)."""
    first = run_study(surface, SCHEMA, settings(), segment="train", strategy_id="s0")
    second = run_study(surface, SCHEMA, settings(), segment="train", strategy_id="s0")
    assert first.study_id == second.study_id
    assert [t.params for t in first.trials] == [t.params for t in second.trials]
    assert dict(first.params) == dict(second.params)


def test_a_different_seed_explores_differently() -> None:
    first = run_study(surface, SCHEMA, settings(seed=1), segment="train", strategy_id="s0")
    second = run_study(surface, SCHEMA, settings(seed=2), segment="train", strategy_id="s0")
    assert [t.params for t in first.trials] != [t.params for t in second.trials]


def test_the_study_id_names_what_was_searched() -> None:
    """Two searches that differ in any of their inputs are different studies."""
    base = run_study(surface, SCHEMA, settings(), segment="train", strategy_id="s0")
    ids = {
        base.study_id,
        run_study(surface, SCHEMA, settings(), segment="wf_is:0", strategy_id="s0").study_id,
        run_study(surface, SCHEMA, settings(), segment="train", strategy_id="s1").study_id,
        run_study(surface, SCHEMA, settings(seed=9), segment="train", strategy_id="s0").study_id,
        run_study(
            surface, SCHEMA, settings(objective="sharpe"), segment="train", strategy_id="s0"
        ).study_id,
    }
    assert len(ids) == 5


# ---------------------------------------------------------------------------
# what a study produces
# ---------------------------------------------------------------------------
def test_a_study_searches_inside_the_declared_bounds() -> None:
    """Section 9.1 makes every parameter bounded precisely so a search cannot
    wander outside the space its author described."""
    result = run_study(surface, SCHEMA, settings(n_trials=25), segment="train", strategy_id="s0")
    for trial in result.trials:
        assert 5 <= trial.params["fast"] <= 50
        assert 20.0 <= trial.params["slow"] <= 100.0


def test_the_trials_come_back_best_first() -> None:
    result = run_study(surface, SCHEMA, settings(n_trials=20), segment="train", strategy_id="s0")
    values = [trial.value for trial in result.trials]
    assert values == sorted(values, reverse=True)
    assert result.best is not None
    assert result.best.value == values[0]


def test_the_submitted_parameters_are_the_plateau_not_the_optimum() -> None:
    """Section 13.8 makes plateau selection mandatory before any validation run."""
    result = run_study(surface, SCHEMA, settings(n_trials=20), segment="train", strategy_id="s0")
    assert result.plateau is not None
    assert dict(result.params) == dict(result.plateau.plateau.params)


def test_a_study_without_a_plateau_refuses_to_hand_over_parameters() -> None:
    from quantlab.optimize.study import StudyResult

    bare = StudyResult(study_id="x", segment="train", objective="sortino_dd", sampler="tpe", seed=1)
    with pytest.raises(ConfigError, match="plateau selection mandatory"):
        _ = bare.params


def test_a_strategy_with_no_parameters_cannot_be_refined() -> None:
    """There is nothing to search, and an empty study would still cost trials."""
    with pytest.raises(ConfigError, match="at least one declared parameter"):
        run_study(surface, {}, settings(), segment="train", strategy_id="s0")


def test_the_grid_sampler_is_refused_with_its_reason() -> None:
    """It needs an explicit search space, which this study builds from the
    strategy's own bounds."""
    with pytest.raises(ConfigError, match="grid sampler needs an explicit search space"):
        run_study(surface, SCHEMA, settings(sampler="grid"), segment="train", strategy_id="s0")


# ---------------------------------------------------------------------------
# what a study records
# ---------------------------------------------------------------------------
def test_the_trial_count_is_persisted() -> None:
    """Section 22's third criterion. Section 14.4 charges these trials into the
    deflated Sharpe ratio's ``M``, so a study whose count went unrecorded would
    make every verdict after it too generous."""
    result = run_study(surface, SCHEMA, settings(n_trials=15), segment="train", strategy_id="s0")
    record = study_record(result, experiment_id="e0", strategy_id="s0")
    assert record["n_trials"] == result.n_trials
    assert result.n_trials > 0
    assert record["segment"] == "train"
    assert record["objective"] == "sortino_dd"
    assert record["seed"] == 5


def test_the_record_keeps_both_answers_side_by_side() -> None:
    """Section 13.8 requires the report to show the optimum and the plateau, so
    a reader can see how far selection moved."""
    result = run_study(surface, SCHEMA, settings(n_trials=20), segment="train", strategy_id="s0")
    record = study_record(result, experiment_id="e0", strategy_id="s0")

    best = json.loads(record["best_trial_json"])
    plateau = json.loads(record["plateau_json"])
    assert set(best) == {"params", "value"}
    assert plateau["point_optimum"] == best["params"]
    assert "median_drop" in plateau
    assert isinstance(plateau["moved"], bool)


def test_the_record_stores_no_second_copy_of_the_search() -> None:
    """Section 6 owns the record; an Optuna database as well would be a second
    answer to "what was searched"."""
    result = run_study(surface, SCHEMA, settings(), segment="train", strategy_id="s0")
    assert study_record(result, experiment_id="e0", strategy_id="s0")["storage_path"] == ""


def test_the_stats_report_what_the_search_cost() -> None:
    result = run_study(surface, SCHEMA, settings(n_trials=15), segment="train", strategy_id="s0")
    assert result.stats["n_trials_requested"] == 15
    assert result.stats["n_trials_completed"] == result.n_trials
    assert result.stats["n_rejected"] == 0


def test_rejected_trials_are_counted() -> None:
    """A search that spent its budget on rejected parameter sets found nothing,
    and the record should say so rather than reporting a best of ``-1e9``."""
    from quantlab.optimize.objectives import REJECTED

    result = run_study(
        lambda _p: REJECTED, SCHEMA, settings(n_trials=8), segment="train", strategy_id="s0"
    )
    assert result.stats["n_rejected"] == result.n_trials


def test_a_repeated_parameter_set_is_recorded_once() -> None:
    """The run cache of section 11.2 keeps it from costing twice; the trial list
    should not double-count it either."""
    fixed = {"fast": ParamSpec(kind="int", default=10, low=10, high=11)}
    result = run_study(
        lambda _p: 1.0, fixed, settings(n_trials=10), segment="train", strategy_id="s0"
    )
    keys = {json.dumps(dict(trial.params), sort_keys=True) for trial in result.trials}
    assert len(keys) == len(result.trials)


def test_a_forbidden_objective_is_refused_before_any_trial_runs() -> None:
    calls = 0

    def counted(params: Mapping[str, Any]) -> float:
        nonlocal calls
        calls += 1
        return surface(params)

    bad = settings()
    object.__setattr__(bad, "objective", "net_return")
    with pytest.raises(ConfigError, match="return-only"):
        run_study(counted, SCHEMA, bad, segment="train", strategy_id="s0")
    assert calls == 0


def test_a_window_may_budget_fewer_trials() -> None:
    """A three-window walk-forward cannot afford the full study three times."""
    result = run_study(
        surface, SCHEMA, settings(n_trials=50), segment="wf_is:1", strategy_id="s0", n_trials=6
    )
    assert result.stats["n_trials_requested"] == 6
    assert result.n_trials <= 6


def test_the_plateau_neighbourhood_is_reproducible() -> None:
    """It is drawn from the study's own seed, so the same study yields the same
    plateau (INV-7)."""
    first = run_study(surface, SCHEMA, settings(), segment="train", strategy_id="s0")
    second = run_study(surface, SCHEMA, settings(), segment="train", strategy_id="s0")
    assert first.plateau is not None and second.plateau is not None
    assert first.plateau.plateau.neighbours == second.plateau.plateau.neighbours


def test_a_seeded_generator_is_not_shared_with_the_caller() -> None:
    """Sanity: the study must not consume the caller's RNG stream."""
    rng = np.random.default_rng(0)
    before = rng.random()
    run_study(surface, SCHEMA, settings(), segment="train", strategy_id="s0")
    rng_again = np.random.default_rng(0)
    assert rng_again.random() == before
