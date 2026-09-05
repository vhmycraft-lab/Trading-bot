"""Declares more free parameters than the load-time ceiling allows."""

from quantlab.core.strategy import Context, ParamSpec, Signal, SignalKind

class Broken:
    name = "broken"
    version = "1"
    style = "bar_loop"
    params = {
        "p0": ParamSpec(kind="int", default=10, low=1, high=50),
        "p1": ParamSpec(kind="int", default=10, low=1, high=50),
        "p2": ParamSpec(kind="int", default=10, low=1, high=50),
        "p3": ParamSpec(kind="int", default=10, low=1, high=50),
        "p4": ParamSpec(kind="int", default=10, low=1, high=50),
        "p5": ParamSpec(kind="int", default=10, low=1, high=50),
        "p6": ParamSpec(kind="int", default=10, low=1, high=50),
        "p7": ParamSpec(kind="int", default=10, low=1, high=50),
        "p8": ParamSpec(kind="int", default=10, low=1, high=50),
        "p9": ParamSpec(kind="int", default=10, low=1, high=50),
        "p10": ParamSpec(kind="int", default=10, low=1, high=50),
    }
    warmup_bars = 100

    def prepare(self, params):
        self.p = dict(params)

    def on_bar(self, ctx: Context) -> Signal:
        fast = ctx.ind("sma", n=self.p["n"])
        return Signal(SignalKind.LONG) if fast > 0 else Signal(SignalKind.FLAT)


STRATEGY = Broken
