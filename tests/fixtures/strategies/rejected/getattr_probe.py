"""Reads an attribute by name, which the BarWindow cannot guard."""

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
        future = getattr(ctx.bars, "close")
        return Signal(SignalKind.LONG) if future is not None else Signal(SignalKind.FLAT)


STRATEGY = Broken
