"""A plain bar-loop strategy: the shape spec section 9.4 prescribes."""

from quantlab.core.strategy import Context, ParamSpec, RiskSpec, Signal, SignalKind, SizingSpec


class SmaCrossValid:
    name = "sma_cross_valid"
    version = "1"
    style = "bar_loop"

    params = {
        "fast": ParamSpec(kind="int", default=20, low=5, high=100),
        "slow": ParamSpec(kind="int", default=100, low=20, high=400),
    }
    warmup_bars = 400
    risk = RiskSpec(stop_loss_pct=0.05, take_profit_pct=0.15)
    sizing = SizingSpec()

    def prepare(self, params):
        self.p = dict(params)

    def on_bar(self, ctx: Context) -> Signal:
        fast = ctx.ind("sma", n=self.p["fast"])
        slow = ctx.ind("sma", n=self.p["slow"])
        return Signal(SignalKind.LONG) if fast > slow else Signal(SignalKind.FLAT)


STRATEGY = SmaCrossValid
