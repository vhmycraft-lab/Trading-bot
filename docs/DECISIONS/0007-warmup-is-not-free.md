# 7. A declared warm-up selects the period a candidate is scored on

* **Status:** Proposed — the ratchet is fixed; the bound is a decision for the project owner
* **Date:** 2026-09-10

## Context

Warm-up bars are excluded from every metric (`compute_metrics` slices
`equity_full[warmup:]`). `warmup_bars` is declared by the candidate. Spec §9.6
rule 6 bounds it from **below** only — it must cover the indicators' lookback —
and `required_warmup`'s docstring justified that by calling a longer warm-up
"strictly more conservative".

It is not conservative. It is period selection. Same run, same trades, same
equity curve — a curve that falls for 1000 bars and then rises for 1000 — with
only the declared warm-up changed:

| declared warm-up | bars scored | net return | max drawdown | fitness |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2000 | +1.7162 | 0.3932 | 0.078220 |
| 250 | 1750 | +2.0780 | 0.3124 | 0.141456 |
| 500 | 1500 | +2.4879 | 0.2209 | 0.247454 |
| 900 | 1100 | +3.2604 | 0.0483 | **0.295525** |

Fitness rises 3.8× and the drawdown almost vanishes, because the declining half
is simply deleted from the measurement.

It was reachable without intent. `MutationPlan._compile` set

```python
"warmup_bars": max(self.genome.warmup_bars, required_warmup(conditions, schema))
```

— a **ratchet**. A lineage that once held a long-lookback indicator kept its
warm-up after mutating that indicator away, and was thereafter scored on a
shorter, later window than its competitors. Evolution selects for whichever
lineage has ratcheted its warm-up onto the most flattering start point, and the
ratchet only ever turns one way.

## Decision

Two halves, and only the first is taken here.

**Fixed:** the ratchet. `_compile` now recomputes `warmup_bars` from the
conditions actually present, so warm-up follows the structure in both
directions and cannot drift upward across generations.

**Not taken:** making rule 6 an exact bound. It is a one-line change in
`StrategyGenome._check_warmup`, and it fully removes the free parameter for
evolved candidates. But `strategies/baselines/rsi_reversion.py` declares a round
`warmup_bars = 60` where `rsi(n<=50)` needs 51, and it is a **golden baseline**.
Exactness therefore forces one of:

1. change the baseline to 51 — which changes its results and requires
   regenerating a golden baseline; or
2. leave the baseline at 60 and let it diverge from its compiled genome twin —
   which breaks the bar-for-bar parity test that exists to prove the compiler is
   faithful.

Both are decisions about how candidates are scored, not bug fixes, and the
second would weaken a real test. Recorded rather than taken unilaterally.

## Options for the owner

* **(a) Exact bound.** Warm-up becomes fully derived; the free parameter is gone
  for every evolved candidate. Cost: regenerate the `rsi_reversion` golden, with
  the change stated as warm-up-only and the equivalence shown.
* **(b) Uniform scoring window.** Exclude the *population's* maximum warm-up from
  every candidate's metrics in a generation, so all candidates are measured over
  identical bars. Strictly the most correct — comparability is the property
  fitness needs — and it leaves hand-written warm-ups alone. Larger change:
  fitness becomes a function of the population, not of one run.
* **(c) Score the warm-up bars.** Include them with flat equity. Declaring a
  longer warm-up then costs flat bars instead of deleting bad ones, so the
  incentive inverts with no threshold anywhere. Changes every existing metric.

Recommendation: **(a)** now, because it is small and closes the reachable path,
with **(b)** if hand-written strategies ever compete directly against evolved
ones in the same ranking.

## Consequences

* The ratchet is gone, so the exploit is no longer produced by ordinary drift.
  A candidate must now declare an over-long warm-up in its own source to get it,
  which for a genome means a hand-authored genome rather than an evolved one.
* Until (a), (b) or (c) is taken, a hand-written strategy can still buy score by
  declaring warm-up it does not need, and `tests/unit/test_fitness_adversarial.py`
  records that as a known-open hole rather than asserting it is closed.
