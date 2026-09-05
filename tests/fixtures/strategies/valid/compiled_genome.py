"""Machine-written source, as the genome compiler of spec section 9.6 emits it.

The compiler produces valid strategies by construction; this fixture holds the
checker to that claim, since generated code has a different shape from anything a
person types — an `__all__`, generated constants, and a flat condition list.
"""

from quantlab.core.strategy import Context, ParamSpec, RiskSpec, Signal, SignalKind, SizingSpec

__all__ = ["STRATEGY", "CompiledGenome"]

GENOME_ID = "g-0001"
GENERATION = 7


class CompiledGenome:
    name = "compiled_genome"
    version = "1"
    style = "bar_loop"

    params = {
        "p_0": ParamSpec(kind="int", default=12, low=4, high=60),
        "p_1": ParamSpec(kind="int", default=48, low=20, high=240),
        "p_2": ParamSpec(kind="float", default=30.0, low=5.0, high=45.0),
    }
    warmup_bars = 240
    risk = RiskSpec(stop_loss_pct=0.04, take_profit_pct=0.12, time_stop_bars=96)
    sizing = SizingSpec(mode="fixed_fraction", fraction=0.5)

    def prepare(self, params):
        self.p = dict(params)

    def on_bar(self, ctx: Context) -> Signal:
        op_0 = ctx.ind("ema", n=self.p["p_0"])
        op_1 = ctx.ind("ema", n=self.p["p_1"])
        op_2 = ctx.ind("rsi", n=14)
        entry = (op_0 > op_1) and (op_2 > self.p["p_2"])
        exit_ = op_0 < op_1
        if entry and not exit_:
            return Signal(SignalKind.LONG)
        return Signal(SignalKind.FLAT)


STRATEGY = CompiledGenome
