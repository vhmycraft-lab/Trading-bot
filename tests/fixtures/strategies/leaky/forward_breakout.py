"""A breakout channel that reaches forward.

Spec section 14.2 names "same-bar ``high`` breakout entry filled at same-bar
open". This engine cannot fill on the same bar — the fill rule is next-open
(section 8.4) — and a vectorised signal is shifted a further bar, so the literal
form is causal here and there would be nothing to detect. The leak that shape
becomes when written vectorised is this one: a channel whose window extends past
the bar it is evaluated at, so the entry is decided by the breakout it is
supposed to predict.
"""

import pandas as pd

from quantlab.core.strategy import ParamSpec, SignalKind


class ForwardBreakout:
    name = "leaky_forward_breakout"
    version = "1"
    style = "vectorized"

    params = {"n": ParamSpec(kind="int", default=10, low=2, high=50)}
    warmup_bars = 50

    def prepare(self, params):
        self.p = dict(params)

    def signals(self, bars: pd.DataFrame, params) -> pd.Series:
        n = int(params["n"])
        # rolling(...).max() over the *next* n bars: the channel a causal
        # strategy would only learn about afterwards.
        ahead = bars["high"].iloc[::-1].rolling(n, min_periods=1).max().iloc[::-1]
        # A selective threshold, so the entry is *driven* by the peeked window
        # rather than saturating at LONG regardless of it. A leak the strategy
        # barely acts on is not much of a leak, and would say little about the probe.
        long = ahead > bars["close"] * 1.02
        return pd.Series(
            [SignalKind.LONG if flag else SignalKind.FLAT for flag in long], index=bars.index
        )


STRATEGY = ForwardBreakout
