"""Causal momentum: this bar's close against the close ``n`` bars ago."""

import pandas as pd

from quantlab.core.strategy import ParamSpec, SignalKind


class MomentumVectorized:
    name = "honest_momentum"
    version = "1"
    style = "vectorized"

    params = {"n": ParamSpec(kind="int", default=12, low=2, high=100)}
    warmup_bars = 40

    def prepare(self, params):
        self.p = dict(params)

    def signals(self, bars: pd.DataFrame, params) -> pd.Series:
        n = int(params["n"])
        change = bars["close"].pct_change(n)
        long = change > 0
        return pd.Series(
            [SignalKind.LONG if flag else SignalKind.FLAT for flag in long], index=bars.index
        )


STRATEGY = MomentumVectorized
