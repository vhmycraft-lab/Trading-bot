# 8. What "undefined" scores as, everywhere it can happen

* **Status:** Accepted
* **Date:** 2026-09-10

## Context

`profit_factor = None` scored **1.0** — full credit — while `sortino = None`
scored **0.0**. Both mean "this ratio has no denominator". The same epistemic
state was given opposite answers, in the same function, and that inconsistency
is what made the Sortino cliff pay: adding one economically null loss moved a
component from 0.0 to 1.0 (ADR 0007's neighbour finding, commit `99d1a6a`).

So every undefined branch in fitness and in the gates was enumerated and each
was asked the same question: is that answer deliberate?

## The rule the module already follows

Reading the code as a whole, there is a coherent convention and it is stated in
`_ratio`'s own docstring:

> A missing metric scores 0. […] a component that could not be measured has not
> been demonstrated, and scoring it as absent is the difference between "no
> evidence" and "no problem".

paired with `_divergence_penalty`'s:

> `1.0` when the inner walk-forward has not run: this penalty punishes a
> *measured* divergence […] Charging for it twice would make an unmeasured
> candidate look worse than a demonstrably divergent one.

Together: **components give no credit for the unmeasured; penalties do not
double-charge for it.** Eight of ten components and all seven penalties follow
this. Two components did not.

## The audit

### Fitness components (`_components`)

| component | undefined when | scored | deliberate? |
| --- | --- | ---: | --- |
| `expectancy` | no trades | 0.0 | yes — convention |
| `risk_adjusted` | no measurable downside | 0.0 | yes — convention |
| `drawdown` | equity empty | 0.0 | yes — unreachable, gate rejects first |
| `profit_factor` | no losing trades | **1.0** | **yes, but see below** |
| `consistency` | series shorter than one window | 0.0 | yes — convention |
| `inner_oos` | inner walk-forward not run | 0.0 | yes — documented |
| `trades` | never undefined (`n_trades` is a float) | — | — |
| `win_rate` | no trades | 0.0 | yes — convention |
| `net_return` | CAGR undefined (span < 1 day) | 0.0 | yes — convention |
| `concentration` | **no winning trades** | **1.0 → 0.0** | **no — fixed** |

### Fitness penalties (`_penalties`)

Every penalty is `1.0` (no charge) when its input is undefined. All deliberate,
by the no-double-charge rule: each has a component that already scores 0 for the
same absence. `p_removal` is the exception and is now `0.0` when retention is
undefined *and* there were trades, because that state means total P&L was not
positive — decisive negative evidence, not a missing measurement (commit
`99d1a6a`).

Worth recording rather than changing: during evolution the loop passes
`sensitivity=None` always, so `p_sensitivity` is permanently `1.0` there.
Sensitivity is a validation-stage check; the penalty exists for that path.

### Validation gates

| gate | undefined input | verdict | deliberate? |
| --- | --- | --- | --- |
| `G_LEAK` | probe never run | FAIL | yes — documented |
| `G_MIN_TRADES` | counts are ints | — | — |
| `G_SANITY` | no trades, no positions | **PASS** (vacuous) | yes — `G_MIN_TRADES` refuses the candidate |
| `G_COST` | never measured under stress | FAIL | yes — documented |
| `G_PERM` | p-value never computed | FAIL | yes — documented |
| `G_DEGRADE` | `sharpe_val` undefined | FAIL | yes |
| `G_DEGRADE` | `sharpe_train` undefined or ≤ 0 | ratio **waived**, sign test only | yes — documented |
| `G_BENCH` | no buy-and-hold baseline | FAIL | yes — documented |

## What was not deliberate, and is fixed

**1. `concentration` scored 1.0 for a ledger with no winners.**
`_clip(1.0 - (top5_profit_share or 0.0))` gave `1.0` — maximum credit — when
`top5_profit_share` was `None`, which happens exactly when there are no winning
trades. That is the worst possible concentration, not the best. It was
unreachable only because `F_EXPECTANCY` rejects such a ledger first, and a
component must not depend on a distant gate for its safety. Now `0.0`.

**2. Sharpe carried the Sortino defect, and it mattered more.**
Sortino's guard was fixed to be relative to the return scale; Sharpe's was left
as an absolute `1e-12` on the same scale-dependent quantity. Sharpe is what the
**deflated Sharpe ratio deflates**:

| equity curve | annualised vol | Sharpe | `deflated_sharpe` at M=1000 |
| --- | ---: | ---: | ---: |
| drifting 1e-6/bar | 5.4e-08 | **162,313** | **1.0** |

The single defence against multiple testing returned maximum confidence for a
curve that is flat to within float noise. Both metrics now share one
`_MIN_DISPERSION_FRACTION` guard, relative to the RMS of the returns.

**3. `+inf` actively passed two gates.**
NaN failed closed only by the accident that every NaN comparison is False.
`+inf` compares greater than any threshold, so an infinite validation Sharpe
passed `G_DEGRADE` and an infinite Sortino beat buy-and-hold at `G_BENCH`. A
`_measured()` helper now funnels NaN and ±inf into `None`, giving them the
treatment this module already gives an absent measurement, and a non-finite
trade return is mapped to infinity in `G_SANITY` so it reads as an impossible
trade rather than being skipped.

## What is deliberate but has an unbuilt half

`profit_factor = None → 1.0` is spec §13.3, and its docstring justifies it as
"scores as 1.0 **while forcing the low-trade warning**". The warning half was
never built: `low_trade_warning` is computed in `metrics.py`, stored, and read
by **no gate and no score anywhere in the codebase**.

Left as 1.0 rather than changed, because the protection the warning was meant to
provide is in fact supplied by control flow: `F_TRADES` runs before any component
is computed, so `profit_factor is None` at scoring time always means "at least
`min_trades` round trips, not one of them a loss". That is a genuinely remarkable
ledger rather than a degenerate one. Recorded here because the docstring cites a
control that does not exist, and a future change to gate ordering would remove
the real protection without touching the sentence that claims it.

## Consequences

* One convention now holds throughout: unmeasured earns no credit in a
  component, and no double charge in a penalty.
* `deflated_sharpe` can no longer be handed an unbounded Sharpe by a flat curve.
* `tests/unit/test_gate_thresholds.py` sweeps every configured gate across its
  threshold from both sides and asserts every gate's response to `None`, NaN,
  ±inf and zero-length input, so the next such asymmetry fails a test rather
  than paying a candidate.
