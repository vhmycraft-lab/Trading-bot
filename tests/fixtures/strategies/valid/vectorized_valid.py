"""A vectorized strategy. The engine shifts the returned series by +1 bar."""

import numpy as np
import pandas as pd

from quantlab.core.strategy import ParamSpec, SignalKind


class VectorizedValid:
    name = "vectorized_valid"
    version = "1"
    style = "vectorized"

    params = {"n": ParamSpec(kind="int", default=50, low=10, high=200)}
    warmup_bars = 200

    def prepare(self, params):
        self.p = dict(params)

    def signals(self, bars: pd.DataFrame, params) -> pd.Series:
        n = int(params["n"])
        # min_periods=n so the first n-1 values are NaN rather than a partial mean
        # computed from fewer bars than the parameter claims.
        mean = bars["close"].rolling(n, min_periods=n).mean()
        long = bars["close"] > mean
        return pd.Series(np.where(long, SignalKind.LONG, SignalKind.FLAT), index=bars.index)


STRATEGY = VectorizedValid
