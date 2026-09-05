"""Walks the class hierarchy to reach objects it was never handed."""

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
        leaked = ctx.__class__
        return Signal(SignalKind.FLAT) if leaked else Signal(SignalKind.LONG)


STRATEGY = Broken
