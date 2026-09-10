# 7. A declared warm-up selects the period a candidate is scored on

* **Status:** Accepted — both halves taken (option (a))
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

**Both halves are taken.** Option (a): the bound is exact.

1. `MutationPlan._compile` recomputes `warmup_bars` from the conditions actually
   present instead of `max(existing, required)`, so warm-up follows structure in
   both directions and cannot ratchet upward across generations.
2. `StrategyGenome._check_warmup` refuses a warm-up above the structural need as
   well as below it. Warm-up is derived, not chosen.
3. `strategies/baselines/rsi_reversion.py` declared a round `warmup_bars = 60`
   where `rsi(n<=50)` needs 51. It is now 51, and its compiled genome twin is 51,
   so the bar-for-bar parity test compares two exact values rather than two
   arbitrary ones.

### Regenerating the rsi_reversion golden

The old golden froze the 60. That is a preservation order around a value we now
know was wrong, not a regression guard, so it was regenerated — and the change
was shown to be confined before it was accepted:

* **No trade moved.** All five golden trade ledgers, old bytes against new, are
  identical under `assert_frame_equal(check_exact=True)`. The first
  `rsi_reversion` trade occurs well after bar 60, so measuring from bar 51
  cannot reach it.
* **All five `trades.parquet` files changed bytes** — that is pyarrow 23
  re-encoding what pyarrow 17 wrote (ADR 0006), not a logic change, which is why
  the logical comparison above was required to accept it.
* **Only `rsi_reversion/metrics.json` changed**, in 8 of 45 fields, every one of
  them a function of the bar count:

  | field | 60 | 51 |
  | --- | ---: | ---: |
  | `ann_volatility` | 0.12192561693209647 | 0.12152240784022499 |
  | `cagr` | -0.008694863327745939 | -0.008637741024603018 |
  | `calmar` | -0.1632892833299457 | -0.16221652811910772 |
  | `exposure` | 0.11283185840707964 | 0.11208791208791209 |
  | `sharpe` | -0.010603998397598067 | -0.010568982384311188 |
  | `sharpe_ci_low` | -0.06385044759192161 | -0.06363946457239664 |
  | `sharpe_ci_high` | 0.042642450796725476 | 0.04250149980377427 |
  | `sortino` | -0.014405879089447746 | -0.014358273704336335 |

  `net_return`, `max_drawdown`, `n_trades` and 34 other fields are unchanged:
  equity is flat until the first trade, so moving the start from bar 60 to bar 51
  changes the denominator without changing the endpoints.

`engine_version` is **not** bumped. The engine's fill and accounting rules are
untouched; what changed is one strategy's declared warm-up and the window its
metrics are computed over.

### What is not closed

Opaque, hand-written strategies declare `warmup_bars` as a class attribute, and
no lookback is computable from arbitrary source, so `ast_check` can only require
that it is a non-negative int literal. A hand-written strategy can therefore
still over-declare. Evolution produces genomes, so the search — the threat model
here — is closed; a hand-written strategy over-declaring is a review matter.

## Consequences

* No genome, evolved or hand-authored, can buy score by declaring warm-up it
  does not need. The adversarial suite asserts the refusal in both directions,
  and separately keeps measuring what the choice used to be worth (fitness 0.078
  against 0.296) so the prize is on record if a route to setting it reappears.
* Option (b), a population-uniform scoring window, remains the stronger answer if
  hand-written strategies ever compete directly against evolved ones in one
  ranking. It is not needed while they do not.
