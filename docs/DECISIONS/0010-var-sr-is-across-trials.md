# 10. `var_sr` is the dispersion of the trials, not of one run's returns

* **Status:** Accepted
* **Date:** 2026-09-10

## Context

The deflated Sharpe ratio (spec section 14.4, check 1) deflates a result by the
size of the search that produced it. Bailey and López de Prado's equation 5:

```
SR0 = sqrt(V) * ((1 - g) * Z(1 - 1/M) + g * Z(1 - 1/(M*e)))
```

It takes **two** inputs from the search. `M` is how many trials ran. `V` is the
variance of those trials' Sharpe ratios — how widely they scored, which is how
much luck was available to the winner.

ADR 0003 §11 defined `M` and commit `28c4bf6` made the code compute it, replacing an `m1-bar-count` stand-in that read `max(1, n_bars // 100)`. `V`
was never wired up at all. `cli/validate.py::_deflated` passed:

```python
var_sr=float(np.var(returns, ddof=1)),
```

— the variance of the **validation run's own per-bar returns**. Its docstring
said:

> The variance of the trial Sharpe ratios is not recoverable from one run, so it
> is estimated from the per-bar returns themselves — a conservative stand-in that
> is stated here rather than hidden: with a wider spread the deflation would be
> harsher, never gentler.

The second clause is true and irrelevant. `SR0` does rise with `V`. What the
docstring does not establish — and what turns out to be false — is that *this*
substitute is on the high side of the quantity it stands in for.

## Measurement

> **Amended.** The table below was first published computed at
> `bars_per_year = 35040`, a 15-minute assumption. The data is **1h**
> (`bars_per_year = 8760`). Per-bar variance is annualised variance divided by
> `bars_per_year`, so the error moved both quantities and the ratio between them.
> The numbers here are the corrected ones; the direction, the magnitude class and
> the decision are unchanged. See "Reconciling the timeframe" below.

Measured across the 704 candidates of the two campaigns on BTC/USDT **1h**
(`rome-first-run` `afbc399cd3f39991`, and the crashed `first-real-campaign`
`e2391416ead7b09f`), 643 of which recorded a finite Sharpe:

| | per-bar variance | sd, annualised | `SR0` at M=704 |
|---|---:|---:|---:|
| variance of the trial Sharpes | 1.45e-03 | 3.559 | **11.22** |
| variance of one run's returns | 4.16e-05 | 0.604 | **1.90** |

The substitute is **34.8x too small** in variance and puts the benchmark
**5.9x lower** than the trials themselves say. It errs toward passing, which is
the one direction this correction exists to prevent.

### Reconciling the timeframe

The campaign report accompanying this work said 15m and this ADR inherited it.
Neither was checked against the store. Three artifacts were then consulted, and
all three agree on **1h**:

* the `dataset` row bound into every run of both campaigns
  (`aec45c9df07a9a94`, `timeframe = '1h'`);
* its own span against its bar count — 47,107 hours over 47,108 bars, exactly
  60.00 minutes per bar;
* the loaded bars' `ts_open` deltas, a constant 3,600,000 ms.

The **`dataset` row is the authoritative one**, and `configs/default.yaml` is
not. The config is mutable after the fact; the `dataset_id` is a foreign key on
every `run`, so a run can always be traced to the bars it actually executed on
no matter what the config says today. A reproducibility claim that rests on a
config file is a claim about the present, not about the run.

**No shipped code was affected.** `_deflated` takes `bars_per_year` from
`result.bars_per_year`, which the engine derives from the dataset's timeframe, so
the implementation always used 8760. The error was confined to hand-computed
figures in prose — which is its own lesson about arithmetic that never passes
through a test.

They are not two estimates of one number. A variance of returns and a variance
of Sharpe ratios are different quantities in different units; the ratio between
them is a property of whatever search happened to run, not a constant that could
be divided out. There was never a factor that would have made the stand-in
right.

## Decision

`var_sr` is computed from the search's own trial Sharpe ratios.

`ExperimentStore.trial_sharpes_for_family` returns the annualised Sharpe of every
trial counted into that family's `M`, resolving the runs **the same way**
`search_trials_for_family` does — every evolution run holding a candidate of one
of the family's versions. Equation 5 multiplies a function of `M` by `sqrt(V)`;
if the two came from different populations of trials the product would describe
no search that ever ran.

A run reached through section 11.2's cache backs more than one candidate, and
its Sharpe is returned once per candidate. `n_evaluations` counts a cache hit as
an evaluation, so dropping the repeat would leave `M` describing a larger
population than `V` does — the same desynchronisation, entered from the other
end. `DISTINCT` here would be a defect.

`cli/validate.py::_trial_sharpes` de-annualises them, because `MetricSet.sharpe`
is annualised (section 10) and the deflation works per bar. The conversion lives
next to the `n_periods` it has to agree with — mixing the two scales raises every
trial Sharpe by `sqrt(bars_per_year)` and is the error `deannualise` was written
to name.

**Fewer than two trials leaves check 1 unmeasured.** `_deflated` returns `None`,
`_deflated_sharpe` reports `C01_DEFLATED_SHARPE` absent, it charges nothing, and
the verdict prints "this verdict rests on partial evidence". The old stand-in
also charged nothing in that situation — it just did so silently, behind a number
that looked measured. This follows ADR 0008: undefined gets one answer, and the
answer is to say so.

Non-finite trial Sharpes are dropped, not clamped. One infinity makes `V`
infinite and `SR0` with it — the safe direction, and still wrong, because check 1
would then fail forever for a reason no operator could read off a verdict.

### What was not done

The threshold was not touched. `validation.dsr_threshold` is unchanged. This is a
correction to an input, not a re-tuning of the gate — raising a threshold moves
an exploit, it does not close one.

## Consequences

Check 1 becomes materially harder to pass, by roughly the factor above for any
search with real dispersion across its trials. That is the correction working:
the campaign that produced 218 trials ranging from Sharpe -16.4 to +1.78 offered
its winner a great deal of luck, and the bar should say so.

Verdicts recorded before this change would have been computed against a benchmark
about 7.7x too low. **There were none.** `validation_verdict` held zero rows when
this landed — the first real campaign rejected all 256 candidates at the fitness
gates, so promotion refused and the validation segment was never reached. Nothing
required invalidating, and no `m_formula_version`-style stamp is needed to tell
old verdicts from new ones because there are no old ones. A later change of this
kind will not be so lucky.

### The pattern this is the latest instance of

This is not the first docstring in this project to assert a conservatism the
arithmetic did not support, and the recurrence is the part worth recording.

* **ADR 0007** — `warmup_bars` above the true lookback was held to be "strictly
  more conservative"; it is not, because warm-up bars are excluded from every
  metric, so a longer warm-up *selects* the period the candidate is scored on.
  Worth 3.8x fitness. The belief was written into two docstrings
  (`core/genome.py`, `evolution/library.py`) and into a `max()` ratchet in
  `evolution/mutation.py`.
* **ADR 0008** — "a missing metric scores 0. That is the conservative reading",
  beside `profit_factor = None` scoring 1.0 and a `concentration` default of 1.0.
* **This ADR** — "harsher, never gentler", 60x too small.

In each case the docstring argued about the *direction* of an effect and left
the *magnitude* unstated, and in each case the magnitude was where the defect
lived. A claim that a substitution is conservative is a claim about a number, and
it is not established until the number is computed. The regression test for this
ADR therefore pins the **unit** rather than the value: the two quantities respond
to different inputs, so invariance under a rescaling of the validation returns,
and sensitivity to the spread of the trials, together say the input is measured
along the trial axis and no other. A test on the value alone would pass again the
moment someone reintroduced the substitution with a different constant.

## References

* `tests/unit/test_deflated_sharpe_var_sr.py` — the unit-pinning fixture
* `tests/unit/test_store.py::test_the_trial_sharpes_come_from_the_same_runs_that_m_counts`
* ADR 0003 §11 (`M` includes every evolution evaluation), ADR 0007, ADR 0008
* Bailey, D. and López de Prado, M. (2014), *The Deflated Sharpe Ratio*
