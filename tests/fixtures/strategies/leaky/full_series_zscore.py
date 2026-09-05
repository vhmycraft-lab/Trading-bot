"""An indicator computed on the full series, then indexed (spec section 14.2).

The mean and standard deviation are taken over every bar in the segment, so the
z-score at bar 0 already knows what the last bar did. Nothing about the
expression looks forward; the leak is entirely in the aggregation.
"""

import pandas as pd

from quantlab.core.strategy import ParamSpec, SignalKind


class FullSeriesZScore:
    name = "leaky_full_series_zscore"
    version = "1"
    style = "vectorized"

    params = {"threshold": ParamSpec(kind="float", default=0.5, low=0.1, high=3.0)}
    warmup_bars = 50

    def prepare(self, params):
        self.p = dict(params)

    def signals(self, bars: pd.DataFrame, params) -> pd.Series:
        close = bars["close"]
        z = (close - close.mean()) / (close.std() or 1.0)
        long = z < -float(params["threshold"])
        return pd.Series(
            [SignalKind.LONG if flag else SignalKind.FLAT for flag in long], index=bars.index
        )


STRATEGY = FullSeriesZScore
