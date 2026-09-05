"""Never trades. The degenerate case: no positions, no fills, nothing to diverge."""

import pandas as pd

from quantlab.core.strategy import ParamSpec, SignalKind


class AlwaysFlat:
    name = "honest_always_flat"
    version = "1"
    style = "vectorized"

    params = {"n": ParamSpec(kind="int", default=1, low=1, high=2)}
    warmup_bars = 5

    def prepare(self, params):
        self.p = dict(params)

    def signals(self, bars: pd.DataFrame, params) -> pd.Series:
        return pd.Series([SignalKind.FLAT] * len(bars), index=bars.index)


STRATEGY = AlwaysFlat
