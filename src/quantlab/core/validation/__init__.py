"""Checks that decide whether a result may be believed (master spec section 14)."""

from __future__ import annotations

from quantlab.core.validation.leakage import (
    DEFAULT_CUT_FRACTIONS,
    REVERSED_TAIL_FRACTION,
    TAIL_MODES,
    Divergence,
    ProbeResult,
    default_cut_points,
    perturbed_tail,
    probe_evaluations,
    require_causal,
    reversed_tail,
    truncation_probe,
)

__all__ = [
    "DEFAULT_CUT_FRACTIONS",
    "REVERSED_TAIL_FRACTION",
    "TAIL_MODES",
    "Divergence",
    "ProbeResult",
    "default_cut_points",
    "perturbed_tail",
    "probe_evaluations",
    "require_causal",
    "reversed_tail",
    "truncation_probe",
]
