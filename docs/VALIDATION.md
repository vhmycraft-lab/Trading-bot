# Validation

*Normative source: `CLAUDE_CODE_MASTER_SPEC.md` §14. Where this document and the
specification disagree, the specification wins.*

Validation is the point of the platform. Everything before it — data, engine,
sandbox, search — exists so that this step can produce one number nobody has
already optimised against.

That is also why validation is *expensive on purpose*. Every look at held-out
data costs some of its power to say anything, so the budget is finite, the
spending is recorded, and the rules that decide a verdict travel with the verdict.

---

## 1. The shape of a verdict

A verdict has two halves and they do different jobs.

**Hard gates (§14.3)** ask whether a result is admissible at all. Seven absolute
rules; failing any one is a `REJECT`, and no amount of good performance elsewhere
compensates. That is what makes them gates rather than another weighted term.

**Soft checks (§14.4)** ask how much of the result is likely to be overfitting.
Eleven checks, each charging points for a specific symptom. The total is the
`overfit_score`.

> **Points are charges.** A higher score is *worse*. Every other number in this
> platform runs the other way, and a reader who assumes otherwise reads every
> verdict backwards.

Then:

| condition | verdict |
|---|---|
| every gate passes and `score <= candidate_max` (30) | `CANDIDATE` |
| every gate passes and `score <= weak_max` (60) | `WEAK` |
| anything else | `REJECT` |

`CANDIDATE` does not mean "trade this". It means: nothing in the evidence
gathered so far says this is overfitting. That is a much smaller claim, and the
lockbox (§14.6) is what turns it into a larger one — once, per family, for ever.

---

## 2. Missing evidence

Gates and checks treat an absent measurement in **opposite** ways, and the
asymmetry is deliberate.

* **A gate that was never measured fails.** `G_LEAK` with no probe, `G_COST` with
  no cost stress, `G_PERM` with no permutation test, `G_BENCH` with no baseline —
  each fails. A gate asks for evidence of safety, and "not checked" is not
  evidence of anything.
* **A check that was never measured charges nothing**, and is recorded as
  unmeasured. A check charges for evidence of *fragility*, and there is none;
  charging for a step the pipeline skipped would penalise a strategy for the
  pipeline's own gaps.

The consequence is the thing to watch for: a `CANDIDATE` with a long
`unmeasured` list is **not** a clean strategy. It is a strategy nobody finished
checking, and `quantlab validate` says so in yellow rather than letting the
number stand for evidence that was never gathered.

---

## 3. The seven gates (§14.3)

| gate | refuses |
|---|---|
| `G_LEAK` | a strategy that is not causal, or was never probed |
| `G_MIN_TRADES` | too few trades on **either** segment to mean anything |
| `G_SANITY` | an impossible single trade, or a position beyond the configured cap |
| `G_COST` | an edge that disappears when costs double |
| `G_PERM` | a result indistinguishable from the same strategy on permuted returns |
| `G_DEGRADE` | non-positive validation performance, or a collapse against train |
| `G_BENCH` | failing to beat buy-and-hold on **either** return or risk |

Three of these are worth a sentence each.

**`G_MIN_TRADES` needs both segments.** A strategy with two hundred training
trades and four validation trades has not been validated; it has been observed
four times.

**`G_DEGRADE` expects degradation.** Some fall from train to validation is what
out-of-sample *means*, so the ratio permits it while the sign test refuses a
strategy that only worked in the past. When the train Sharpe is not positive,
only the sign test applies — dividing by it would let a strategy that lost money
on both segments pass by losing slightly less on one.

**`G_BENCH` is a disjunction on purpose.** A strategy that earns less than
buy-and-hold but halves the drawdown is a different and legitimate product, and a
gate demanding both would reject it for succeeding at one.

**`G_SANITY` carries slack.** `max_position_fraction` bounds the *target*
fraction at the deciding bar; the realised fraction then drifts with the market
until the next rebalance, so `sanity_drift_allowance` (default 0.5) keeps the
gate from failing correctly-behaved strategies. The exact invariant — committed
capital at a fill never exceeds `max_position_fraction × equity` at the deciding
bar — belongs to the engine and is checked by its property tests.

---

## 4. The eleven checks (§14.4)

| # | check | charges | for |
|---|---|---|---|
| 1 | deflated Sharpe | 25 | a result the size of the search cannot support |
| 2 | PBO | 25 / 10 | the in-sample winner falling below median out of sample |
| 3 | permutation | 10 | getting close to indistinguishable from noise |
| 4 | sensitivity | 20 | an objective that collapses when parameters move |
| 5 | concentration | 10 | profit carried by the top few trades |
| 6 | walk-forward | 20 / 10 / 5 | low efficiency, few profitable windows, unstable parameters |
| 7 | complexity | 5 each | free parameters and logic lines above their budgets |
| 8 | regime | 10 | more than 90 % of profit inside one calendar quarter |
| 9 | random baseline | 20 | sitting inside the random-entry distribution |
| 10 | trade removal | 25 / 15 / 10 | profit that does not survive losing its best trades |
| 11 | provenance | 10 | a wide search that never generalised in-sample |

**Check 2 is banded; checks 6 and 10 are cumulative.** Both of check 2's
thresholds read one quantity, and `PBO > 0.5` implies `PBO > 0.3`, so charging
both would make the higher band worth 35 rather than the 25 §14.4 assigns it.
Checks 6 and 10 each read three *different* quantities — three separate
symptoms — so their charges add.

**Check 11 needs both conditions.** A wide search that *did* generalise is
exactly what the platform is for; a narrow search that did not is charged by
other checks.

---

## 5. The statistics

### Deflated Sharpe ratio (check 1)

Bailey and López de Prado (2014). A Sharpe ratio selected as the best of `M`
trials is biased upwards by the selection itself. The deflated ratio measures the
result against `SR₀` — the Sharpe one would *expect* the best of `M` trials to
reach by luck alone — rather than against zero, and corrects the standard error
for skew and kurtosis.

`M` counts **every evaluation that led here**: `evolution_run.n_evaluations +
optuna trials + family.validation_touches`. A search that tried more things has to
clear a higher bar, and that is the whole point.

> **Units.** Every function takes a **per-period** Sharpe ratio, because `T`
> counts periods. `MetricSet.sharpe` is annualised; `deannualise` converts it.
> Passing the annualised figure with a bar count overstates the ratio by roughly
> `√bars_per_year` — about 94× on hourly bars — and returns 1.0 for everything,
> with no exception and no obviously wrong number. `kurtosis` is the **non-excess**
> fourth moment, so a normal distribution is 3.

### Probability of backtest overfitting (check 2)

Combinatorially symmetric cross-validation. Split the period into `n_blocks`
blocks; for each way of choosing half as in-sample, find the best trial there and
look up its rank over the blocks left out. If the search is finding real
structure the winner keeps winning; if it is fitting noise its out-of-sample rank
is a coin flip and PBO approaches 0.5.

The splits are **sampled uniformly** when there are more than `max_combinations`.
Taking a lexicographic prefix is not sampling: every split it yields puts the
early blocks in-sample and the late ones out, which makes the estimate a
statement about the tail of the period. Measured on pure noise, truncation
returned PBO from 0.01 to 0.82 across five seeds.

### Market permutation (check 3, `G_PERM`)

Resample the return series with a stationary block bootstrap — blocks of
geometrically distributed length, so the marginal distribution and short-range
autocorrelation survive while the long-range structure a strategy might be
fitting does not — and re-run the strategy on each.

```
p = (1 + #(sortino_perm >= sortino_real)) / (1 + n)
```

The `1 +` on both sides makes the p-value achievable-but-never-zero, which is the
honest statement when a finite number of permutations were drawn. A permutation
the strategy could not be scored on is **counted, not dropped**: shrinking the
denominator would make the p-value look more significant the more often the
strategy failed.

### Trade shuffling

The same trades in a different order produce a different equity path.
`mdd_p95` — the 95th percentile of the shuffled drawdowns — is a fairer statement
of risk than the one ordering history happened to deal, and is what a paper
trading report compares a realised drawdown against.

---

## 6. The budget

Section 14.1's step 8, and the reason the whole design holds:

* Every validation **increments `strategy_family.validation_touches`**.
* At `family_max_validation_touches` (20) the family **freezes**, and step 0
  refuses to validate a family that is not `open`. The twenty-first validation
  cannot happen rather than merely being noticed.
* Touches are counted **per family**, not per strategy version: the budget is
  about how often one *idea* has been measured, and a re-parameterised version of
  it is the same idea.
* The touch is charged **after** the judging, so a pipeline that crashed
  mid-judgement does not spend a look it never used.

A family also closes permanently on a lockbox failure (§14.6), and step 0 refuses
that too — so a lockbox failure cannot be walked back by re-validating.

---

## 7. Direction of information

This is the end of the line.

* Evolution reads the **train** segment and its own inner folds, and nothing else
  (INV-9, enforced at the one place a run is created).
* The **validation** segment is reachable only through a recorded promotion
  (§13.7) or a direct `quantlab validate`.
* The **test** partition is reachable only through `quantlab lockbox` (INV-5).
* Nothing computed in §14 is read by the search. `tests/unit/test_architecture.py`
  forbids any module under `evolution/` or `optimize/` from importing a validation
  judgement, so the shortcut is unavailable as well as unproductive.

A verdict informs a person. That is all it does.

---

## 8. Running it

```bash
quantlab validate run <strategy_id> --params '{"fast": 20}'
quantlab validate run <strategy_id> --permutations 200      # check 3 and G_PERM
```

The permutation test is opt-in because it costs `n` full backtests. Skipping it
leaves `G_PERM` **failing** and check 3 unmeasured — which is the correct
asymmetry, not an inconsistency: the gate wants evidence and has none.

Every verdict is recorded in `validation_verdict` with its gates, its checks and
a snapshot of every threshold it was reached under. Thresholds change; a verdict
read a year later must be interpretable against the rules that produced it.
