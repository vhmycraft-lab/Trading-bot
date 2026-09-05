"""Template for a new strategy (master spec section 9.4).

Copy this file, rename the class, and change ``on_bar``.  The contract is small
on purpose: decide, and return a signal.  Sizing, order placement, stops and
costs are the engine's job, and a strategy that tried to do them would need
intrabar prices the ``BarWindow`` refuses to provide.
"""

from quantlab.core.strategy import Context, ParamSpec, RiskSpec, Signal, SignalKind, SizingSpec


class MyStrategy:
    name = "my_strategy"
    version = "1"
    style = "bar_loop"

    params = {
        "fast": ParamSpec(kind="int", default=20, low=5, high=100),
        "slow": ParamSpec(kind="int", default=100, low=20, high=400),
    }
    warmup_bars = 400

    # Declared explicitly: risk controls are engine behaviour (spec section 8.6),
    # and a candidate overrides these without touching this file.
    risk = RiskSpec()
    sizing = SizingSpec()

    def prepare(self, params):
        self.p = dict(params)

    def on_bar(self, ctx: Context) -> Signal:
        fast = ctx.ind("sma", n=self.p["fast"])
        slow = ctx.ind("sma", n=self.p["slow"])
        return Signal(SignalKind.LONG) if fast > slow else Signal(SignalKind.FLAT)


STRATEGY = MyStrategy
