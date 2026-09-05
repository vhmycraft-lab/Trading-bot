"""Declares no params, so nothing can be searched or mutated."""

from quantlab.core.strategy import Context, ParamSpec, Signal, SignalKind

class Broken:
    name = "broken"
    version = "1"
    style = "bar_loop"
    warmup_bars = 100

    def prepare(self, params):
        self.p = dict(params)

    def on_bar(self, ctx: Context) -> Signal:
        fast = ctx.ind("sma", n=self.p["n"])
        return Signal(SignalKind.LONG) if fast > 0 else Signal(SignalKind.FLAT)


STRATEGY = Broken
