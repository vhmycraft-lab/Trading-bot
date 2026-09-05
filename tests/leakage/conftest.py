"""Shared bars and fixture loading for the leakage suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import pytest

from quantlab.core.types import BarFrame

FIXTURES: Final[Path] = Path(__file__).resolve().parents[1] / "fixtures" / "strategies"
LEAKY_DIR: Final[Path] = FIXTURES / "leaky"
HONEST_DIR: Final[Path] = FIXTURES / "honest"
BASELINES: Final[Path] = Path(__file__).resolve().parents[2] / "strategies" / "baselines"

#: Named so the probe's verdicts are reproducible: the same bars every run, and a
#: shape with both trend and mean reversion so a strategy has something to react
#: to. A probe whose result depended on the day's random draw would be useless as
#: a gate.
BAR_SEED: Final[int] = 7
N_BARS: Final[int] = 400


def make_bars(n: int = N_BARS, seed: int = BAR_SEED) -> BarFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + rng.normal(0, 0.6, n).cumsum() + 8.0 * np.sin(np.arange(n) / 23.0)
    return BarFrame(
        pd.DataFrame(
            {
                "ts_open": (np.arange(n, dtype="int64") * 3_600_000),
                "open": close,
                "high": close + np.abs(rng.normal(0, 0.4, n)),
                "low": close - np.abs(rng.normal(0, 0.4, n)),
                "close": close,
                "volume": np.full(n, 10.0),
                "quote_volume": np.full(n, 1000.0),
                "trades": np.full(n, 5, dtype="int64"),
                "is_gap_filled": np.zeros(n, dtype=bool),
            }
        ),
        symbol="BTCUSDT",
        timeframe="1h",
        dataset_id="leakage-fixture",
    )


def load_strategy(path: Path) -> Any:
    """Instantiate a fixture strategy.

    These files are repository test fixtures, not untrusted input: INV-4 is about
    LLM-authored code, which reaches the engine only through the sandbox. Running
    the probe in-process here is what makes the suite fast enough to be run on
    every commit.
    """
    namespace: dict[str, Any] = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    return namespace["STRATEGY"]()


def default_params(strategy: Any) -> dict[str, Any]:
    return {name: spec.default for name, spec in getattr(strategy, "params", {}).items()}


@pytest.fixture(scope="session")
def bars() -> BarFrame:
    return make_bars()
