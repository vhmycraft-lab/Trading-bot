"""Plateau selection (master spec section 13.8).

The point optimum of a parameter search is the single best trial. It is also, very
often, a spike: one combination that happened to line up with the training window
and falls off the moment the next month of data arrives. Section 13.8 makes
plateau selection **mandatory** before any validation run — "the parameters
submitted for validation are the plateau choice, never the point optimum" — and
requires the report to show both, so the difference is visible rather than
decided quietly.

The method is direct: take the best ``top_k`` trials, perturb each one's
parameters by each configured percentage, evaluate the neighbours, and prefer the
trial whose *neighbourhood* holds up. A spike has a good centre and poor
neighbours; a plateau has a slightly worse centre and neighbours nearly as good.

The same neighbourhood evidence feeds two other places: section 13.3's
``p_sensitivity`` penalty and section 14.4's soft check 4. It is computed once
here and handed to both as a
:class:`~quantlab.core.fitness.SensitivityReport`.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from quantlab.core.config import PlateauSettings
from quantlab.core.errors import ConfigError
from quantlab.core.fitness import SensitivityReport
from quantlab.core.strategy import ParamSpec

__all__ = [
    "Neighbourhood",
    "PlateauReport",
    "Trial",
    "neighbours_of",
    "select_plateau",
]

_EPS: Final[float] = 1e-12

#: Scores a parameter set. Supplied by the caller so this module needs no engine,
#: no sandbox and no bars, and can be tested against a synthetic surface.
Evaluate = Callable[[Mapping[str, Any]], float]


@dataclass(frozen=True, slots=True)
class Trial:
    """One evaluated parameter set from a search."""

    params: Mapping[str, Any]
    value: float


@dataclass(frozen=True, slots=True)
class Neighbourhood:
    """A trial and how the objective behaves around it."""

    params: Mapping[str, Any]
    value: float
    neighbours: tuple[float, ...] = ()

    @property
    def median_neighbour(self) -> float | None:
        """The middle neighbour's score, or ``None`` if none were evaluated."""
        return statistics.median(self.neighbours) if self.neighbours else None

    @property
    def median_drop(self) -> float | None:
        """How far the median neighbour falls below the centre, as a fraction.

        Relative to the *magnitude* of the centre, so a drop is comparable across
        objectives with different scales. ``None`` when there are no neighbours,
        or when the centre is ~0 and the ratio would be unbounded — reporting an
        arbitrary number there would set off the sensitivity penalty for a
        strategy whose objective merely happened to land near zero.
        """
        median = self.median_neighbour
        if median is None or abs(self.value) <= _EPS:
            return None
        return (self.value - median) / abs(self.value)

    @property
    def score(self) -> float:
        """What plateau selection ranks on: the median neighbour, not the centre.

        This is the whole idea in one line. A spike is a high ``value`` with poor
        neighbours; ranking on the neighbours picks the parameter set that is
        still good when the market moves slightly away from it.
        """
        median = self.median_neighbour
        return self.value if median is None else median


@dataclass(frozen=True, slots=True)
class PlateauReport:
    """Both answers, as section 13.8 requires the report to show them."""

    point_optimum: Neighbourhood
    plateau: Neighbourhood
    considered: tuple[Neighbourhood, ...] = ()

    @property
    def moved(self) -> bool:
        """True when plateau selection chose something other than the optimum."""
        return dict(self.plateau.params) != dict(self.point_optimum.params)

    def sensitivity(self) -> SensitivityReport:
        """The chosen parameters' neighbourhood, for section 13.3's penalty."""
        return SensitivityReport(
            median_drop=self.plateau.median_drop, n_neighbours=len(self.plateau.neighbours)
        )


def neighbours_of(
    params: Mapping[str, Any],
    schema: Mapping[str, ParamSpec],
    rng: np.random.Generator,
    *,
    n: int,
    pct: float,
) -> list[dict[str, Any]]:
    """Parameter sets within ``pct`` of ``params``, snapped to their own bounds.

    Every numeric parameter moves at once, each by its own draw: perturbing one at
    a time would explore the axes and miss the corners, and it is the corners that
    tell a plateau from a ridge.

    Non-numeric parameters are held fixed. A categorical has no neighbourhood —
    "10 % of `volatility_target`" is not a thing — and resampling one would
    measure a different strategy rather than the same one moved slightly.

    A draw that lands back on ``params`` after snapping is kept: it is a true
    statement about a coarse grid, and dropping it would quietly report a
    smoother surface than the one being searched.
    """
    if n <= 0:
        return []
    drawn: list[dict[str, Any]] = []
    for _ in range(n):
        moved = dict(params)
        for name, spec in schema.items():
            if spec.kind not in ("int", "float") or name not in moved:
                continue
            factor = 1.0 + float(rng.uniform(-pct, pct))
            moved[name] = spec.clamp(float(moved[name]) * factor)
        drawn.append(moved)
    return drawn


def select_plateau(
    trials: Sequence[Trial],
    schema: Mapping[str, ParamSpec],
    evaluate: Evaluate,
    settings: PlateauSettings,
    rng: np.random.Generator,
) -> PlateauReport:
    """Choose the parameters to submit for validation (spec section 13.8).

    Args:
        trials: Every evaluated parameter set from the search, in any order.
        schema: The bounds each parameter moves inside.
        evaluate: Scores one parameter set. Every call is a real evaluation, so
            the cost of this step is ``top_k * len(perturbation_pcts) *
            n_neighbors`` of them — which is why ``top_k`` is small.
        settings: ``optimize.plateau``.
        rng: Seeded, so the same search yields the same plateau (INV-7).

    Returns:
        Both the point optimum and the plateau choice.

    Raises:
        ConfigError: there are no trials to choose from. A search that produced
            nothing cannot have its parameters refined, and returning the empty
            choice would hand validation whatever the caller had lying around.
    """
    if not trials:
        raise ConfigError("plateau selection needs at least one trial")

    ranked = sorted(trials, key=lambda trial: (-trial.value, _key(trial.params)))
    considered: list[Neighbourhood] = []
    for trial in ranked[: max(1, settings.top_k)]:
        scores: list[float] = []
        for pct in settings.perturbation_pcts:
            scores.extend(
                evaluate(candidate)
                for candidate in neighbours_of(
                    trial.params, schema, rng, n=settings.n_neighbors, pct=pct
                )
            )
        considered.append(
            Neighbourhood(params=dict(trial.params), value=trial.value, neighbours=tuple(scores))
        )

    optimum = considered[0]
    plateau = optimum
    for hood in considered[1:]:
        # Strictly better, or selection stays put. A tie means nothing
        # distinguishes the two neighbourhoods, and moving on a lexicographic
        # tie-break would have selection wander for its own sake — away from the
        # trial the search itself ranked highest, for no reason a reader could
        # follow.
        if (hood.score, hood.value) > (plateau.score, plateau.value):
            plateau = hood
    return PlateauReport(point_optimum=optimum, plateau=plateau, considered=tuple(considered))


def _key(params: Mapping[str, Any]) -> str:
    """A deterministic tie-break for the trial ordering, so two trials that
    scored identically are always considered in the same order."""
    return repr(sorted((str(name), repr(value)) for name, value in params.items()))
