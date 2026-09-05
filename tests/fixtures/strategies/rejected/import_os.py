"""Reaches for the filesystem through a forbidden module."""

from quantlab.core.strategy import Context, ParamSpec, Signal, SignalKind
import os

class Broken:
    name = "broken"
    version = "1"
    style = "bar_loop"
    params = {"n": ParamSpec(kind="int", default=20, low=5, high=100)}
    warmup_bars = 100

    def prepare(self, params):
        self.p = dict(params)

    def on_bar(self, ctx: Context) -> Signal:
        return Signal(SignalKind.FLAT)


STRATEGY = Broken
