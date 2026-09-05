"""Simulates an intrabar stop from the bar's own low, which needs an order of events the bar does not record."""

from quantlab.core.strategy import Context, ParamSpec, Signal, SignalKind

class Broken:
    name = "broken"
    version = "1"
    style = "bar_loop"
    params = {"n": ParamSpec(kind="int", default=20, low=5, high=100)}
    warmup_bars = 100

    def prepare(self, params):
        self.p = dict(params)

    def on_bar(self, ctx: Context) -> Signal:
        if ctx.position.entry_price and ctx.bars.low[-1] <= ctx.position.entry_price * 0.98:
            return Signal(SignalKind.FLAT)
        return Signal(SignalKind.LONG)


STRATEGY = Broken
