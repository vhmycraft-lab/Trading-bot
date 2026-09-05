"""Random entry (master spec section 9.5).

Not a strategy — a control.  Any candidate that cannot beat a coin flip with the
same trade frequency and the same costs has demonstrated nothing, and spec
section 14.4 check 9 compares against a distribution of these.

Randomness comes from ``ctx.rng``, seeded from ``(run seed, bar index)``, so the
draw at a given bar is the same whether the run covers 100 bars or 10 000 — which
is what lets the leakage probe compare a truncated run against a full one.
"""

from quantlab.core.strategy import Context, ParamSpec, RiskSpec, Signal, SignalKind, SizingSpec


class RandomEntry:
    name = "random_entry"
    version = "1"
    style = "bar_loop"

    params = {
        "p_enter": ParamSpec(kind="float", default=0.02, low=0.001, high=0.5),
        "hold_bars": ParamSpec(kind="int", default=24, low=1, high=500),
    }
    warmup_bars = 0

    # Declared explicitly: risk controls are engine behaviour (spec section 8.6),
    # and a candidate overrides these without touching this file.
    risk = RiskSpec()
    sizing = SizingSpec()

    def prepare(self, params):
        self.p = dict(params)
        self.entered_at = None

    def on_bar(self, ctx: Context) -> Signal:
        if ctx.position.side is not None and self.entered_at is not None:
            if ctx.i - self.entered_at < self.p["hold_bars"]:
                return Signal(SignalKind.LONG, tag="holding")
            self.entered_at = None
            return Signal(SignalKind.FLAT, tag="held_out")

        if float(ctx.rng.random()) < self.p["p_enter"]:
            self.entered_at = ctx.i
            return Signal(SignalKind.LONG, tag="random")
        return Signal(SignalKind.FLAT)


STRATEGY = RandomEntry
