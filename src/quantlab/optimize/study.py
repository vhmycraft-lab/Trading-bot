"""Optuna parameter studies, scoped to a promoted candidate (spec section 13.8).

Optuna was the primary search in spec 1.0 and is demoted here to *refinement*:
evolution (section 13) finds a region, and a study explores the parameters inside
it. Three constraints follow from that demotion, and all three are enforced here
rather than left to a caller's discipline:

* **Train only.** ``assert segment.startswith(("train", "wf_is"))``, as section
  13.8 keeps from spec 1.0. A study samples hundreds of parameter sets, and doing
  that on the validation segment would burn the one measurement the whole
  platform is built to protect (INV-5, section 13.7).
* **Every trial counts.** Trials add to ``n_trials_accounted``, which section
  14.4 charges into the deflated Sharpe ratio's ``M``. A study whose trials went
  unrecorded would make every verdict that followed it too generous.
* **Return-only objectives are refused**, by :mod:`quantlab.optimize.objectives`.

The study itself is in-memory and seeded. Optuna's own storage is not used for
persistence: section 6 gives that job to ``optuna_study``, and two records of one
search would be two answers to "what was searched".
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import optuna

from quantlab.core.config import OptimizeSettings
from quantlab.core.errors import ConfigError
from quantlab.core.hashing import canonical_json, short_id
from quantlab.core.strategy import ParamSpec
from quantlab.optimize.objectives import REJECTED, require_allowed_objective
from quantlab.optimize.plateau import PlateauReport, Trial, select_plateau

__all__ = [
    "SEARCHABLE_SEGMENTS",
    "StudyResult",
    "require_searchable_segment",
    "run_study",
    "study_record",
    "suggest_params",
]

#: The only segments a parameter search may touch (spec sections 13.7, 13.8).
SEARCHABLE_SEGMENTS: Final[tuple[str, ...]] = ("train", "wf_is")

#: Scores one parameter set. The caller supplies it, so this module needs no
#: engine, no sandbox and no bars.
Evaluate = Callable[[Mapping[str, Any]], float]


def require_searchable_segment(segment: str) -> str:
    """Return ``segment``, or refuse it (spec section 13.8).

    Raises:
        ConfigError: the segment is not ``train`` or a walk-forward in-sample
            window. This is the assertion spec 1.0 carried and 13.8 keeps: a
            search over hundreds of parameter sets on validation or test data
            would consume the measurement the platform exists to protect.
    """
    if not segment.startswith(SEARCHABLE_SEGMENTS):
        raise ConfigError(
            "a parameter search may only run on the train segment or a walk-forward "
            "in-sample window; searching validation or test data spends the one "
            "measurement the platform exists to protect",
            segment=segment,
            allowed=list(SEARCHABLE_SEGMENTS),
        )
    return segment


def _sampler(settings: OptimizeSettings) -> optuna.samplers.BaseSampler:
    """The configured sampler, seeded so a study is reproducible (INV-7)."""
    if settings.sampler == "random":
        return optuna.samplers.RandomSampler(seed=settings.seed)
    if settings.sampler == "grid":
        raise ConfigError(
            "the grid sampler needs an explicit search space, which this study builds "
            "from the strategy's own ParamSpec bounds; use tpe or random",
            sampler=settings.sampler,
        )
    return optuna.samplers.TPESampler(seed=settings.seed)


def suggest_params(trial: optuna.Trial, schema: Mapping[str, ParamSpec]) -> dict[str, Any]:
    """Ask Optuna for one parameter set, inside the strategy's own bounds.

    The bounds come from the strategy's ``ParamSpec`` declarations and nowhere
    else. Section 9.1 makes every parameter bounded precisely so that a search
    cannot wander outside the space its author described.
    """
    values: dict[str, Any] = {}
    for name, spec in sorted(schema.items()):
        if spec.kind == "int":
            values[name] = trial.suggest_int(
                name, int(spec.low or 0), int(spec.high or 0), step=int(spec.step or 1)
            )
        elif spec.kind == "float":
            values[name] = trial.suggest_float(
                name, float(spec.low or 0.0), float(spec.high or 0.0), log=spec.log
            )
        elif spec.kind == "bool":
            values[name] = trial.suggest_categorical(name, [False, True])
        else:
            values[name] = trial.suggest_categorical(name, list(spec.choices or ()))
    return values


@dataclass(frozen=True, slots=True)
class StudyResult:
    """What one study produced, and what it cost the multiple-testing budget."""

    study_id: str
    segment: str
    objective: str
    sampler: str
    seed: int
    #: Every completed trial, best first.
    trials: tuple[Trial, ...] = ()
    plateau: PlateauReport | None = None
    stats: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_trials(self) -> int:
        """Trials that completed — what section 14.4 charges into ``M``."""
        return len(self.trials)

    @property
    def best(self) -> Trial | None:
        return self.trials[0] if self.trials else None

    @property
    def params(self) -> Mapping[str, Any]:
        """The parameters to submit for validation: the plateau choice (§13.8)."""
        if self.plateau is None:
            raise ConfigError(
                "this study has no plateau choice; section 13.8 makes plateau "
                "selection mandatory before any validation run",
                study_id=self.study_id,
            )
        return self.plateau.plateau.params

    def best_trial_json(self) -> str:
        """The stored form of the point optimum (``optuna_study``, section 6)."""
        best = self.best
        return canonical_json(
            {} if best is None else {"params": dict(best.params), "value": best.value}
        )

    def plateau_json(self) -> str:
        """The stored form of the plateau choice, with both answers side by side.

        Section 13.8 requires the report to show the point optimum *and* the
        plateau, so the record keeps both: a reader can see how far selection
        moved, which is the whole evidence that the optimum was a spike.
        """
        if self.plateau is None:
            return "{}"
        return canonical_json(
            {
                "params": dict(self.plateau.plateau.params),
                "value": self.plateau.plateau.value,
                "median_neighbour": self.plateau.plateau.median_neighbour,
                "median_drop": self.plateau.plateau.median_drop,
                "n_neighbours": len(self.plateau.plateau.neighbours),
                "point_optimum": dict(self.plateau.point_optimum.params),
                "moved": self.plateau.moved,
            }
        )


def run_study(
    evaluate: Evaluate,
    schema: Mapping[str, ParamSpec],
    settings: OptimizeSettings,
    *,
    segment: str,
    strategy_id: str,
    n_trials: int | None = None,
) -> StudyResult:
    """Search ``schema`` on ``segment``, then choose the plateau (section 13.8).

    Args:
        evaluate: Scores one parameter set. Every call is a real backtest, and
            the run cache of section 11.2 is what keeps a repeated parameter set
            from costing twice.
        schema: The strategy's declared parameters and their bounds.
        settings: ``optimize``.
        segment: Must be ``train`` or ``wf_is:*`` — see
            :func:`require_searchable_segment`.
        strategy_id: The candidate being refined, for the study's id.
        n_trials: Overrides ``optimize.n_trials``, for a walk-forward window that
            budgets fewer.

    Returns:
        A :class:`StudyResult` whose ``params`` are the plateau choice, never the
        point optimum.
    """
    require_searchable_segment(segment)
    require_allowed_objective(settings.objective)
    if not schema:
        raise ConfigError(
            "a parameter study needs at least one declared parameter", strategy_id=strategy_id
        )

    budget = settings.n_trials if n_trials is None else max(1, int(n_trials))
    # Before `create_study`, which itself logs at INFO. A two-hundred-trial
    # refinement would otherwise bury the run's own structured log in Optuna's.
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=_sampler(settings))
    seen: dict[str, Trial] = {}

    def _objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, schema)
        value = float(evaluate(params))
        key = canonical_json(params)
        seen.setdefault(key, Trial(params=params, value=value))
        return value

    study.optimize(_objective, n_trials=budget, timeout=settings.timeout_s)

    trials = sorted(seen.values(), key=lambda trial: -trial.value)
    plateau = select_plateau(
        trials,
        schema,
        evaluate,
        settings.plateau,
        np.random.default_rng(settings.seed),
    )
    return StudyResult(
        study_id=short_id(f"{strategy_id}|{segment}|{settings.objective}|{settings.seed}|{budget}"),
        segment=segment,
        objective=settings.objective,
        sampler=settings.sampler,
        seed=settings.seed,
        trials=tuple(trials),
        plateau=plateau,
        stats={
            "n_trials_requested": budget,
            "n_trials_completed": len(trials),
            "n_rejected": sum(1 for trial in trials if trial.value <= REJECTED),
        },
    )


def study_record(result: StudyResult, *, experiment_id: str, strategy_id: str) -> dict[str, Any]:
    """The ``optuna_study`` row for a finished study (spec section 6).

    ``storage_path`` is empty by design: the study runs in memory and section 6
    owns the record, so a second copy in an Optuna database would be a second
    answer to "what was searched".
    """
    return {
        "study_id": result.study_id,
        "experiment_id": experiment_id,
        "strategy_id": strategy_id,
        "segment": result.segment,
        "sampler": result.sampler,
        "seed": result.seed,
        "n_trials": result.n_trials,
        "objective": result.objective,
        "best_trial_json": result.best_trial_json(),
        "plateau_json": result.plateau_json(),
        "storage_path": "",
    }
