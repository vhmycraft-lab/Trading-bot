"""Causal but unbounded lookback: an expanding mean uses every bar so far.

Included because it is the shape most easily confused with the full-series
aggregation in ``leaky/full_series_zscore.py``. Expanding backwards is causal;
aggregating over the whole segment is not, and the probe must tell them apart.
"""

import pandas as pd

from quantlab.core.strategy import ParamSpec, SignalKind


class ExpandingMean:
    name = "honest_expanding_mean"
    version = "1"
    style = "vectorized"

    params = {"n": ParamSpec(kind="int", default=30, low=5, high=200)}
    warmup_bars = 30

    def prepare(self, params):
        self.p = dict(params)

    def signals(self, bars: pd.DataFrame, params) -> pd.Series:
        n = int(params["n"])
        mean = bars["close"].expanding(min_periods=n).mean()
        long = bars["close"] > mean
        return pd.Series(
            [SignalKind.LONG if flag else SignalKind.FLAT for flag in long], index=bars.index
        )


STRATEGY = ExpandingMean
