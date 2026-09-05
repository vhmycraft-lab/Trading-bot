"""A causal vectorised moving-average cross.

``min_periods=n`` matters: without it the first values are means of fewer bars
than the parameter claims, which is not leakage but is a different strategy from
the one declared.
"""

import pandas as pd

from quantlab.core.strategy import ParamSpec, SignalKind


class SmaCrossVectorized:
    name = "honest_sma_cross"
    version = "1"
    style = "vectorized"

    params = {
        "fast": ParamSpec(kind="int", default=10, low=2, high=100),
        "slow": ParamSpec(kind="int", default=30, low=5, high=300),
    }
    warmup_bars = 60

    def prepare(self, params):
        self.p = dict(params)

    def signals(self, bars: pd.DataFrame, params) -> pd.Series:
        fast = int(params["fast"])
        slow = int(params["slow"])
        close = bars["close"]
        quick = close.rolling(fast, min_periods=fast).mean()
        slowly = close.rolling(slow, min_periods=slow).mean()
        long = quick > slowly
        return pd.Series(
            [SignalKind.LONG if flag else SignalKind.FLAT for flag in long], index=bars.index
        )


STRATEGY = SmaCrossVectorized
