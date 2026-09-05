"""Reads the bar's own high legitimately.

A Donchian breakout compares a closed bar's high against a channel computed from
earlier bars.  That is not an intrabar stop and must not be refused as one: the
comparison never mentions an entry price, and the engine's fill still happens at
the next open.
"""

from quantlab.core.strategy import Context, ParamSpec, RiskSpec, Signal, SignalKind


class BreakoutValid:
    name = "breakout_valid"
    version = "1"
    style = "bar_loop"

    params = {"n": ParamSpec(kind="int", default=55, low=10, high=200)}
    warmup_bars = 200
    risk = RiskSpec(trailing_stop_pct=0.08)

    def prepare(self, params):
        self.p = dict(params)

    def on_bar(self, ctx: Context) -> Signal:
        upper = ctx.ind("highest", n=self.p["n"])
        if ctx.bars.high[-1] >= upper:
            return Signal(SignalKind.LONG)
        return Signal(SignalKind.FLAT)


STRATEGY = BreakoutValid
