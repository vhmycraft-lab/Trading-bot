# 12. The drawdown gate is relative to buy-and-hold over the same window

* **Status:** Accepted
* **Date:** 2026-09-11

## Context

`F_DRAWDOWN` rejected any candidate whose maximum drawdown exceeded an absolute
**0.50**. The first real campaign (`rome-first-run`, `afbc399cd3f39991`, 256
candidates on BTC/USDT 1h) made two things measurable about that number.

**Buy-and-hold fails it in every window.** Across the eight per-generation
training environments:

| gen | bars | B&H max drawdown |
|---:|---:|---:|
| 0 | 16,489 | 0.7041 |
| 1 | 23,554 | 0.7041 |
| 2 | 28,264 | 0.7041 |
| 3 | 16,489 | 0.7464 |
| 4 | 30,619 | 0.7041 |
| 5 | 30,619 | 0.7720 |
| 6 | 35,329 | 0.7405 |
| 7 | 18,844 | 0.7041 |

**8 of 8.** The asset itself could not clear the gate applied to strategies
trading it. A 0.50 ceiling on BTC does not express "manage risk"; it expresses
"be roughly twice as good on drawdown as the thing you are trading", which is a
much stronger demand than anyone wrote down.

**An absolute number cannot make the comparison that matters.** It scores 55%
drawdown in a period when the asset fell 80% as worse than 45% in a period when
it fell 30%, though the first is skill and the second is not. `G_BENCH` already
makes exactly this comparison at validation (against `sortino_bh` and
`drawdown_bh`), so the fitness gate was not merely stricter than validation — it
was asking a different question, and the wrong one.

### What the gate was actually rejecting

Drawdown in this population is a churn measurement, not a risk measurement.
Against the 256 candidates it correlates with trade count at Spearman **+0.922**
and with net return at **-0.784**. At the extremes:

| group | n | median trades | median net return | profitable |
|---|---:|---:|---:|---:|
| drawdown > 0.90 | 92 | 2,386 | **-99.96%** | **0** |
| drawdown <= 0.50 | 63 | **0** | +0.00% | 9 |

Bimodal, and neither mode is a strategy: the high mode is genomes churning
themselves to death on costs, the low mode never traded. The old gate mostly
passed non-traders and mostly rejected cost-bleeders. Of the 26 candidates that
cleared expectancy *and* the 30-trade floor — the population where drawdown is
genuinely the deciding gate — exactly **one** was under 0.50, against a median of
0.7657 and a benchmark of 0.7041.

## Decision

A candidate's drawdown may not exceed `max_drawdown_vs_benchmark` times
buy-and-hold's drawdown **over the window it was evaluated on**. The multiple is
**1.00**: no worse on drawdown than holding the asset.

1.00 was chosen against the measured alternatives rather than picked:

| ceiling | of the 26 real candidates | of all 256, passing every gate |
|---|---:|---:|
| absolute 0.50 (old) | 1 | 0 |
| <= 0.90 x B&H | 6 | 4 |
| **<= 1.00 x B&H** | **9** | **7** |
| <= 1.10 x B&H | 18 | 12 |
| <= 1.25 x B&H | 26 | 18 |

1.25x admits all 26 and is therefore not a filter at all. 1.00x is the line with
a meaning that can be stated in one sentence, and it is the same line `G_BENCH`
draws downstream.

### Window-local by construction, not by convention

The comparison is against the candidate's **own** bars. `benchmark_drawdown` is a
function of one close array, called in `_evaluate_member` on the same `BarFrame`
that was just handed to the engine. It is deliberately **not** hoisted to the
generation, cached by symbol, or stored against a segment name.

The reason is Project Rome. With a per-generation `TrainingEnvironment` the
window moves while the symbol and the segment name stay fixed, so any benchmark
keyed on those is stale — and a drawdown ceiling quietly measured against the
wrong window raises no error, changes no type, and silently alters what
qualifies. The cost of recomputing is one pass over the closes per candidate,
against a full backtest. That is not a trade worth making.

### One ceiling, not two

`p_drawdown` previously read `gates.max_drawdown` directly, so it ramped between
`drawdown_soft` and the same absolute value the gate used. Making the gate
relative would have left the penalty ramping toward a fallback the gate no longer
enforces — weaker penalties in exactly the windows where the gate got stricter.
`_penalties` now receives the effective ceiling that `_failed_gate` used. "Penalise
as you approach the line, reject at the line" is one threshold.

This was caught by the existing penalty tests failing, not by design. It is
recorded because the next relative threshold will have the same shape.

### The fallback is not a second gate

`gates.max_drawdown` survives only for a window too short to have a benchmark,
and is now **1.00**. `None` means "we could not measure the benchmark", and a
candidate must not be rejected for our failure to measure. It is not consulted
when a benchmark exists — otherwise the relative gate would be decorative
wherever the absolute one is tighter, and there is a test that says so.

## Consequences

The gate becomes consistent with `G_BENCH` rather than stricter along a different
axis. On the recorded campaign it moves the population passing every gate from
**0 to 7**.

**This does not by itself produce risk-managed candidates.** All 256 genomes used
`sizing.mode = "fixed_fraction"` and **238 of 256 sat at `fraction = 1.0`**, with
181 carrying no risk block at all. A strategy fully invested with no stop cannot
draw down much less than buy-and-hold while it holds. The wall was substantially a
*sizing* artefact; this ADR removes a barrier the search could not climb for
reasons unrelated to the gate, and the sizing work is what gives it something to
climb with.

`min_trades` is untouched — it is a statistical floor, not a preference. Nothing
in the validation layer is touched.

## References

* `tests/unit/test_relative_drawdown_gate.py` — window-locality asserted by
  judging one unchanged candidate against two windows
* `tests/unit/test_fitness.py::test_the_drawdown_penalty_follows_the_same_ceiling_the_gate_enforces`
* ADR 0011 (the deflation's population), spec section 13.3, section 14.1's `G_BENCH`
