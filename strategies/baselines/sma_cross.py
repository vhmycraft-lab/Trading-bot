"""Moving-average crossover (master spec section 9.5)."""

from quantlab.core.strategy import Context, ParamSpec, RiskSpec, Signal, SignalKind, SizingSpec


class SmaCross:
    name = "sma_cross"
    version = "1"
    style = "bar_loop"

    params = {
        "fast": ParamSpec(kind="int", default=50, low=5, high=200),
        "slow": ParamSpec(kind="int", default=200, low=20, high=400),
    }
    warmup_bars = 400

    # Declared explicitly: risk controls are engine behaviour (spec section 8.6),
    # and a candidate overrides these without touching this file.
    risk = RiskSpec()
    sizing = SizingSpec()

    def prepare(self, params):
        self.p = dict(params)

    def on_bar(self, ctx: Context) -> Signal:
        fast = ctx.ind("sma", n=self.p["fast"])
        slow = ctx.ind("sma", n=self.p["slow"])
        if fast != fast or slow != slow:  # NaN during warm-up
            return Signal(SignalKind.FLAT)
        return Signal(SignalKind.LONG) if fast > slow else Signal(SignalKind.FLAT)


STRATEGY = SmaCross
