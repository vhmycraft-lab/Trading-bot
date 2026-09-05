"""A centred rolling mean (spec section 14.2).

Half of every window lies in the future. It is the most innocent-looking leak in
the list, because ``center=True`` is one keyword and reads like a smoothing
choice rather than a time-travel one.
"""

import pandas as pd

from quantlab.core.strategy import ParamSpec, SignalKind


class CentredMean:
    name = "leaky_centred_mean"
    version = "1"
    style = "vectorized"

    params = {"n": ParamSpec(kind="int", default=21, low=3, high=99)}
    warmup_bars = 99

    def prepare(self, params):
        self.p = dict(params)

    def signals(self, bars: pd.DataFrame, params) -> pd.Series:
        n = int(params["n"])
        smooth = bars["close"].rolling(n, center=True, min_periods=1).mean()
        long = bars["close"] > smooth
        return pd.Series(
            [SignalKind.LONG if flag else SignalKind.FLAT for flag in long], index=bars.index
        )


STRATEGY = CentredMean
