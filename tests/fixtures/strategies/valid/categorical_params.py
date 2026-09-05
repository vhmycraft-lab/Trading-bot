"""Exercises every ParamSpec kind, plus an __init__ definition.

`__init__` is a definition, not an attribute read, so it is the one dunder the
checker leaves alone (spec section 9.2).
"""

import math

from quantlab.core.strategy import Context, ParamSpec, Signal, SignalKind


class CategoricalParams:
    name = "categorical_params"
    version = "1"
    style = "bar_loop"

    params = {
        "n": ParamSpec(kind="int", default=14, low=2, high=60),
        "threshold": ParamSpec(kind="float", default=1.5, low=0.1, high=5.0),
        "smoother": ParamSpec(kind="categorical", default="sma", choices=["sma", "ema"]),
        "invert": ParamSpec(kind="bool", default=False),
    }
    warmup_bars = 60

    def __init__(self):
        self.p: dict = {}

    def prepare(self, params):
        self.p = dict(params)

    def on_bar(self, ctx: Context) -> Signal:
        value = ctx.ind(self.p["smoother"], n=self.p["n"])
        edge = math.copysign(1.0, value - self.p["threshold"])
        if self.p["invert"]:
            edge = -edge
        return Signal(SignalKind.LONG) if edge > 0 else Signal(SignalKind.FLAT)


STRATEGY = CategoricalParams
