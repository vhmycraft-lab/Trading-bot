"""Parameter sensitivity (master spec section 14.4, check 4).

A result that holds only at one exact parameter set is a result about that
parameter set, not about the market. The neighbourhood test asks how far the
objective falls when the parameters are nudged: a plateau barely moves, a spike
collapses.

The neighbourhood itself is **computed once**, in
:mod:`quantlab.optimize.plateau`, as part of the plateau selection section 13.8
makes mandatory before any validation run. This module owns the report type and
the scoring of it, so the same evidence feeds section 13.3's ``p_sensitivity``
penalty during evolution and section 14.4's soft check 4 during validation
without either sampling its own neighbours and reaching a different answer.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SensitivityReport", "sensitivity_points"]


@dataclass(frozen=True, slots=True)
class SensitivityReport:
    """How far the objective falls around the chosen parameters (section 13.8).

    ``median_drop`` is the median neighbour's fall as a fraction of the centre's
    own magnitude, so it is comparable across objectives with different scales.
    ``None`` means it could not be measured — no neighbours were evaluated, or
    the centre sat at zero where the ratio is unbounded.
    """

    median_drop: float | None = None
    n_neighbours: int = 0

    @property
    def measured(self) -> bool:
        return self.median_drop is not None and self.n_neighbours > 0


def sensitivity_points(report: SensitivityReport | None, *, max_drop: float, points: int) -> int:
    """Section 14.4's check 4: charge when the neighbourhood collapses.

    An **unmeasured** neighbourhood charges nothing. That is the opposite of how
    the hard gates of section 14.3 treat a missing measurement, and deliberately
    so: a gate asks for evidence of safety and refuses without it, while a soft
    check charges for evidence of *fragility* and has none here. Charging for an
    absent measurement would penalise a strategy for a step the pipeline skipped.
    """
    if report is None or not report.measured:
        return 0
    return points if float(report.median_drop or 0.0) > max_drop else 0
