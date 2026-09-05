"""Plateau selection (master spec section 13.8, task T51).

Section 20 names the test: "spike-vs-plateau synthetic objective". A synthetic
surface is the right instrument here — the claim is about the *selection rule*,
and a real backtest would leave every failure ambiguous between the rule and the
market. The surface below has a narrow spike that scores higher than a broad
plateau, which is exactly the situation section 13.8 exists to handle: the
optimum is a spike, and validation must not be handed it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from quantlab.core.config import PlateauSettings
from quantlab.core.errors import ConfigError
from quantlab.core.strategy import ParamSpec
from quantlab.optimize.plateau import (
    Neighbourhood,
    Trial,
    neighbours_of,
    select_plateau,
)

SCHEMA = {
    "fast": ParamSpec(kind="int", default=10, low=1, high=60),
    "slow": ParamSpec(kind="int", default=30, low=1, high=120),
}


def surface(params: Mapping[str, Any]) -> float:
    """A tall narrow spike at (7, 21) and a lower broad plateau at (30, 70).

    The spike is the higher of the two, so a search that took the point optimum
    would take it — and it is one bar wide in each direction, so it is worth
    nothing the moment the market moves.
    """
    fast, slow = float(params["fast"]), float(params["slow"])
    spike = 3.0 * np.exp(-(((fast - 7) / 0.5) ** 2 + ((slow - 21) / 0.5) ** 2))
    plateau = 2.0 * np.exp(-(((fast - 30) / 15.0) ** 2 + ((slow - 70) / 25.0) ** 2))
    return float(max(spike, plateau))


def grid_trials() -> list[Trial]:
    """Both regions, evaluated on a coarse grid the search would have covered."""
    points = [(7, 21), (6, 21), (8, 21), (30, 70), (28, 68), (32, 72), (25, 60), (35, 80)]
    return [
        Trial(params={"fast": f, "slow": s}, value=surface({"fast": f, "slow": s}))
        for f, s in points
    ]


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# the central claim
# ---------------------------------------------------------------------------
def test_the_spike_wins_the_search_and_loses_the_selection() -> None:
    """Section 13.8's whole purpose in one test."""
    trials = grid_trials()
    report = select_plateau(trials, SCHEMA, surface, PlateauSettings(), rng())

    assert dict(report.point_optimum.params) == {"fast": 7, "slow": 21}
    assert report.point_optimum.value > report.plateau.value
    assert dict(report.plateau.params) == {"fast": 30, "slow": 70}
    assert report.moved


def test_the_spike_s_neighbourhood_collapses_and_the_plateau_s_does_not() -> None:
    """The evidence behind the choice, stated separately from the choice."""
    report = select_plateau(grid_trials(), SCHEMA, surface, PlateauSettings(), rng())
    assert report.point_optimum.median_drop is not None
    assert report.plateau.median_drop is not None
    assert report.point_optimum.median_drop > 0.9
    assert report.plateau.median_drop < 0.3


def test_selection_ranks_on_the_neighbours_not_the_centre() -> None:
    spike = Neighbourhood(params={"fast": 7}, value=3.0, neighbours=(0.0, 0.0, 0.0))
    broad = Neighbourhood(params={"fast": 30}, value=2.0, neighbours=(1.9, 2.0, 1.8))
    assert spike.value > broad.value
    assert broad.score > spike.score


def test_a_flat_surface_keeps_the_optimum() -> None:
    """Nothing to move to: selection must not wander for its own sake."""
    trials = [Trial(params={"fast": f, "slow": 30}, value=1.0) for f in (10, 20, 30)]
    report = select_plateau(trials, SCHEMA, lambda _p: 1.0, PlateauSettings(), rng())
    assert not report.moved
    assert report.plateau.median_drop == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# neighbourhoods
# ---------------------------------------------------------------------------
def test_every_numeric_parameter_moves_at_once() -> None:
    """One at a time would explore the axes and miss the corners, and it is the
    corners that tell a plateau from a ridge."""
    drawn = neighbours_of({"fast": 20, "slow": 60}, SCHEMA, rng(1), n=20, pct=0.25)
    assert len(drawn) == 20
    assert any(p["fast"] != 20 and p["slow"] != 60 for p in drawn)


def test_a_neighbour_stays_inside_the_declared_bounds() -> None:
    """The bounds are the strategy author's, and a search may not leave them."""
    at_edge = {"fast": 60, "slow": 120}
    for candidate in neighbours_of(at_edge, SCHEMA, rng(2), n=40, pct=1.0):
        assert 1 <= candidate["fast"] <= 60
        assert 1 <= candidate["slow"] <= 120


def test_a_non_numeric_parameter_is_held_fixed() -> None:
    """ "Ten per cent of `volatility_target`" is not a thing, and resampling one
    would measure a different strategy rather than the same one moved."""
    schema = {
        "fast": ParamSpec(kind="int", default=10, low=1, high=60),
        "mode": ParamSpec(kind="categorical", default="a", choices=("a", "b")),
        "invert": ParamSpec(kind="bool", default=False),
    }
    for candidate in neighbours_of(
        {"fast": 20, "mode": "a", "invert": False}, schema, rng(3), n=10, pct=0.5
    ):
        assert candidate["mode"] == "a"
        assert candidate["invert"] is False


def test_a_larger_perturbation_reaches_further() -> None:
    near = neighbours_of({"fast": 30, "slow": 60}, SCHEMA, rng(4), n=40, pct=0.05)
    far = neighbours_of({"fast": 30, "slow": 60}, SCHEMA, rng(4), n=40, pct=0.50)
    assert max(abs(p["fast"] - 30) for p in far) > max(abs(p["fast"] - 30) for p in near)


def test_asking_for_no_neighbours_gives_none() -> None:
    assert neighbours_of({"fast": 10}, SCHEMA, rng(), n=0, pct=0.1) == []


def test_neighbour_drawing_is_deterministic() -> None:
    left = neighbours_of({"fast": 20, "slow": 60}, SCHEMA, rng(9), n=8, pct=0.2)
    right = neighbours_of({"fast": 20, "slow": 60}, SCHEMA, rng(9), n=8, pct=0.2)
    assert left == right


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------
def test_the_report_shows_both_answers() -> None:
    """Section 13.8 requires it: the difference is the evidence that the optimum
    was a spike, and hiding it would make the choice unauditable."""
    report = select_plateau(grid_trials(), SCHEMA, surface, PlateauSettings(), rng())
    assert report.point_optimum is not report.plateau
    assert len(report.considered) == min(PlateauSettings().top_k, len(grid_trials()))
    assert report.point_optimum in report.considered
    assert report.plateau in report.considered


def test_the_report_feeds_the_sensitivity_penalty() -> None:
    """Computed once here, used by section 13.3's ``p_sensitivity`` and section
    14.4's soft check 4."""
    report = select_plateau(grid_trials(), SCHEMA, surface, PlateauSettings(), rng())
    sensitivity = report.sensitivity()
    assert sensitivity.median_drop == report.plateau.median_drop
    assert sensitivity.n_neighbours == len(report.plateau.neighbours)


def test_a_centre_at_zero_reports_no_drop_rather_than_an_arbitrary_one() -> None:
    """The ratio is unbounded there, and an arbitrary number would set off the
    sensitivity penalty for a strategy whose objective merely landed near zero."""
    hood = Neighbourhood(params={"fast": 1}, value=0.0, neighbours=(0.5, -0.5))
    assert hood.median_drop is None
    assert hood.score == 0.0


def test_a_neighbourhood_with_no_neighbours_scores_its_own_centre() -> None:
    hood = Neighbourhood(params={"fast": 1}, value=1.25)
    assert hood.median_neighbour is None
    assert hood.median_drop is None
    assert hood.score == 1.25


def test_selection_is_deterministic() -> None:
    trials = grid_trials()
    first = select_plateau(trials, SCHEMA, surface, PlateauSettings(), rng(7))
    second = select_plateau(trials, SCHEMA, surface, PlateauSettings(), rng(7))
    assert dict(first.plateau.params) == dict(second.plateau.params)
    assert first.plateau.neighbours == second.plateau.neighbours


def test_selection_needs_something_to_choose_from() -> None:
    """Returning the empty choice would hand validation whatever the caller had
    lying around."""
    with pytest.raises(ConfigError, match="at least one trial"):
        select_plateau([], SCHEMA, surface, PlateauSettings(), rng())


def test_top_k_bounds_the_cost() -> None:
    """Each considered trial costs ``len(perturbation_pcts) * n_neighbors`` real
    evaluations, which is why ``top_k`` is small."""
    calls = 0

    def counted(params: Mapping[str, Any]) -> float:
        nonlocal calls
        calls += 1
        return surface(params)

    settings = PlateauSettings(top_k=2, perturbation_pcts=(0.1,), n_neighbors=5)
    select_plateau(grid_trials(), SCHEMA, counted, settings, rng())
    assert calls == 2 * 1 * 5
