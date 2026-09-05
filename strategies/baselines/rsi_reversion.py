"""RSI mean reversion (master spec section 9.5).

Long below ``oversold``, flat above ``exit_level``.  The asymmetric thresholds
are deliberate: entering and exiting on the same level would trade on noise.
"""

from quantlab.core.strategy import Context, ParamSpec, RiskSpec, Signal, SignalKind, SizingSpec


class RsiReversion:
    name = "rsi_reversion"
    version = "1"
    style = "bar_loop"

    params = {
        "n": ParamSpec(kind="int", default=14, low=2, high=50),
        "oversold": ParamSpec(kind="float", default=30.0, low=5.0, high=45.0),
        "exit_level": ParamSpec(kind="float", default=55.0, low=46.0, high=90.0),
    }
    warmup_bars = 60

    # Declared explicitly: risk controls are engine behaviour (spec section 8.6),
    # and a candidate overrides these without touching this file.
    risk = RiskSpec()
    sizing = SizingSpec()

    def prepare(self, params):
        self.p = dict(params)

    def on_bar(self, ctx: Context) -> Signal:
        value = ctx.ind("rsi", n=self.p["n"])
        if value != value:  # NaN during warm-up
            return Signal(SignalKind.FLAT)
        if value < self.p["oversold"]:
            return Signal(SignalKind.LONG, tag="oversold")
        if value > self.p["exit_level"]:
            return Signal(SignalKind.FLAT, tag="recovered")
        # Between the thresholds, hold whatever is open.
        return (
            Signal(SignalKind.LONG, tag="hold")
            if ctx.position.side is not None
            else Signal(SignalKind.FLAT)
        )


STRATEGY = RsiReversion
