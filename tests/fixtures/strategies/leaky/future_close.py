"""Reads tomorrow's close: ``close.shift(-1)`` (spec section 14.2).

The engine shifts a vectorised strategy's output by one bar (section 9.1), so
this exactly consumes the slack that shift is meant to provide. Truncation cannot
see it — the prefix computes the same values — but the reversed tail can: the
signal recorded at the first replaced bar was computed from that bar's close, and
that close is now a different number.
"""

import pandas as pd

from quantlab.core.strategy import ParamSpec, SignalKind


class FutureClose:
    name = "leaky_future_close"
    version = "1"
    style = "vectorized"

    params = {"n": ParamSpec(kind="int", default=5, low=2, high=50)}
    warmup_bars = 50

    def prepare(self, params):
        self.p = dict(params)

    def signals(self, bars: pd.DataFrame, params) -> pd.Series:
        tomorrow = bars["close"].shift(-1)
        long = tomorrow > bars["close"]
        return pd.Series(
            [SignalKind.LONG if flag else SignalKind.FLAT for flag in long], index=bars.index
        )


STRATEGY = FutureClose
