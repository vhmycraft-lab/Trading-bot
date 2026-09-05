"""Buy and hold: long from the first bar (master spec section 9.5).

The benchmark every strategy must beat to be interesting.  Most do not.
"""

from quantlab.core.strategy import Context, RiskSpec, Signal, SignalKind, SizingSpec


class BuyAndHold:
    name = "buy_and_hold"
    version = "1"
    style = "bar_loop"
    params = {}
    warmup_bars = 0

    # Declared explicitly: risk controls are engine behaviour (spec section 8.6),
    # and a candidate overrides these without touching this file.
    risk = RiskSpec()
    sizing = SizingSpec()

    def prepare(self, params):
        self.p = dict(params)

    def on_bar(self, ctx: Context) -> Signal:
        return Signal(SignalKind.LONG, tag="hold")


STRATEGY = BuyAndHold
