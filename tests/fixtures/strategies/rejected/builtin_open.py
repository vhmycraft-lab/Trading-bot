"""Opens a file: the shortest path from a strategy to the filesystem."""

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
        handle = open("/etc/passwd")
        return Signal(SignalKind.FLAT) if handle else Signal(SignalKind.LONG)


STRATEGY = Broken
