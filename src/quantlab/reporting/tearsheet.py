"""The equity and drawdown picture for a run (master spec T23).

Rendered offline, with a non-interactive backend chosen before pyplot is
imported: a report generated on a headless machine — which is every machine that
runs this in a batch — must not depend on a display existing.

Two panels, not ten. The equity curve says what happened; the drawdown says what
it cost to sit through. A tearsheet that showed twelve ratios would be read as
twelve pieces of evidence, when they are mostly the same one.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantlab.core.metrics import drawdown_series

__all__ = ["TEARSHEET_NAME", "write_tearsheet"]

TEARSHEET_NAME = "tearsheet.png"

_FIGSIZE = (10.0, 6.0)
_DPI = 110


def write_tearsheet(
    equity: pd.Series, path: str | Path, *, title: str = "", dpi: int = _DPI
) -> Path:
    """Write the equity and drawdown panels for ``equity`` to ``path``.

    Raises:
        ValueError: the series is empty. A tearsheet of nothing would be an image
            that looks like a result.
    """
    values = np.asarray(equity.to_numpy(), dtype="float64")
    if values.size == 0:
        raise ValueError("cannot draw a tearsheet for an empty equity curve")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    drawdown = np.asarray(drawdown_series(values), dtype="float64")
    x = np.arange(values.size)

    figure, (top, bottom) = plt.subplots(2, 1, figsize=_FIGSIZE, sharex=True, height_ratios=[2, 1])
    try:
        top.plot(x, values, linewidth=1.2, color="#1f77b4")
        top.set_ylabel("equity")
        top.grid(True, alpha=0.25)
        if title:
            top.set_title(title)

        bottom.fill_between(x, -drawdown, 0.0, color="#d62728", alpha=0.35)
        bottom.set_ylabel("drawdown")
        bottom.set_xlabel("bar")
        bottom.grid(True, alpha=0.25)

        figure.tight_layout()
        figure.savefig(target, dpi=dpi)
    finally:
        # Closed explicitly: a batch that rendered a few hundred reports would
        # otherwise hold every figure open until the process ended.
        plt.close(figure)
    return target
