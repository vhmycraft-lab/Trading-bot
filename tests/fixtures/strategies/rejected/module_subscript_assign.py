"""Mutates a module-level container at import time."""

from quantlab.core.strategy import Context, ParamSpec, Signal, SignalKind

CACHE = (1, 2)
LOOKUP = dict()
LOOKUP["seen"] = 0


class Broken:
    name = "broken"
    version = "1"
    style = "bar_loop"
    params = {"n": ParamSpec(kind="int", default=20, low=5, high=100)}
    warmup_bars = 100

    def prepare(self, params):
        self.p = dict(params)

    def on_bar(self, ctx: Context) -> Signal:
        fast = ctx.ind("sma", n=self.p["n"])
        return Signal(SignalKind.LONG) if fast > 0 else Signal(SignalKind.FLAT)


STRATEGY = Broken
