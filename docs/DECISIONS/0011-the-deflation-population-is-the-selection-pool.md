# 11. The deflation's population is the selection pool, not the whole search

* **Status:** Accepted
* **Date:** 2026-09-11

## Context

ADR 0010 fixed `var_sr` to be the dispersion of the trial Sharpe ratios rather
than of one run's returns, and left both `M` and `V` computed over **every
evaluation the search made**. Measured across the two campaigns' 704 evaluations
on BTC/USDT 1h:

| population | M | sd(SR) annualised | `SR0` annualised |
|---|---:|---:|---:|
| (a) all evaluations — ADR 0010 as shipped | 704 | 3.559 | **11.224** |
| (b) gate-passing candidates only | 7 | 0.365 | **0.507** |
| (c) finite Sharpe only | 643 | 3.559 | 11.129 |
| (d) gate-passing **and** finite | 7 | 0.365 | **0.507** |

Under (a) a candidate needs an **annualised Sharpe of 12.50** to clear
`dsr_threshold`. Real strategies live at 1–2. Check 1 was therefore unreachable
for anything real — a constant 25-point charge rather than a measurement, which
also distorts the whole overfit scale, since CANDIDATE at `<= 30` becomes "five
points away from everything else".

**The cause.** The 704 include the 38 zero-trade candidates, the 103 under the
trade floor, and every degenerate high-turnover genome. Their Sharpes are extreme
and they dominate the variance — but they were never *selectable*. They fail hard
gates before selection happens. The deflation asks "given you picked the best of
`M`, how likely is it real?", and nobody was ever picking among those.

This is ADR 0010's own principle applied to one input and not the other. `M` was
restricted to nothing and `V` to nothing, so they matched — but they matched on a
population that does not correspond to any act of selection.

## Decision

**Both `M` and `V` are computed over the selection pool**: the candidates that
passed the hard fitness gates and could therefore have been chosen.

`trial_sharpes_for_family` filters on `candidate.gate_failure IS NULL`.
`_selection_pool_m` makes the evolution term `len(pool)`. The Optuna and
validation-touch terms carry through unchanged — both are genuine selection
events, neither contributes a candidate to the evolution pool, so neither is
double-counted. They leave `M` marginally larger than the population `V` covers,
which is the conservative direction and is recorded rather than rounded away.

### The non-finite filter is removed, not kept alongside

ADR 0010 dropped non-finite Sharpes explicitly. That is now redundant and it is
**deleted rather than retained**: across both campaigns, of the 61 candidates
with no finite Sharpe, **zero** passed the gates. A candidate cannot both clear
`F_EXPECTANCY` and have too little dispersion to define a Sharpe. Two filters
where one suffices reads to the next person as though both were load-bearing.

The claim is kept as a test rather than as a comment, so that if gate-passing
ever stops implying finite it fails there rather than silently returning an
infinity into a variance.

Note also that (a) vs (c) above is negligible — 11.224 to 11.129. The extremes
doing the damage were *finite*, merely enormous. The non-finite filter was never
the lever, which is the other reason to stop implying it was.

## The small-pool problem, and why the answer is a bound

Restricting to the pool makes `M` small — **7** on the recorded campaigns. A
variance estimated from seven points has a large standard error, and `SR0` is
proportional to `sigma`, so the benchmark inherits that noise.

The obvious answer is a minimum pool size. It is the wrong one. A floor high
enough to be comfortable — twenty, thirty — leaves check 1 permanently unmeasured
on realistic pools, and **"permanently unmeasured" carries exactly as much
information as "permanently failing"**. It replaces one non-informative check
with a differently non-informative check.

Instead the dispersion is a one-sided chi-square **upper confidence bound**:

```
sigma_upper^2 = (n - 1) * s^2 / chi2_ppf(alpha, n - 1)
```

A small pool produces a *harsher* benchmark rather than an absent one. No cliff,
errs toward rejecting, and tightens on its own as pools grow. Measured at
`alpha = 0.05`:

| n | sigma inflation | | n | sigma inflation |
|---:|---:|---|---:|---:|
| 2 | x15.95 | | 30 | x1.28 |
| 3 | x4.42 | | 50 | x1.20 |
| 5 | x2.37 | | 100 | x1.13 |
| 7 | **x1.92** | | 300 | x1.07 |
| 10 | x1.65 | | 1000 | x1.04 |

On the real pool of 7 this moves `SR0` from 0.507 to 0.970 annualised, and the
Sharpe a candidate must reach from **1.78 to 2.24**. Both are in the range real
strategies occupy, so the check is informative either way and the bound buys
genuine conservatism without buying unreachability.

**The hard floor stays at 2**, and it is arithmetic rather than policy: below two
there is no variance to bound. One trial reports unmeasured per ADR 0008 — never
a fall back to the unfiltered population, which would put this whole problem
straight back.

`alpha` is `validation.dsr_dispersion_alpha`, default 0.05.

## Consequences

Check 1 becomes a measurement again. On the recorded campaigns it moves from
"nothing can pass" to a bar at annualised Sharpe 2.24.

`dsr_threshold` is **untouched**. This corrects an input, exactly as ADR 0010
did. Raising a threshold moves a problem; it does not fix one.

**This campaign's pool is empty**, because every one of its 256 candidates was
gate-rejected. `M = 0` there, check 1 reports unmeasured, and that is correct: a
search that selected nothing offers no selection to correct for. The pool stops
being empty now that ADR 0012's relative drawdown gate admits 7 of the same 256.

### On the direction of these two ADRs

ADR 0010 made check 1 much stricter; this makes it less strict. That reads like a
retreat and is not one. ADR 0010 corrected the *quantity* and left the
*population* wrong; this corrects the population. Neither moved a threshold, and
the second was found by asking what the first's number meant rather than by
finding it inconvenient — the measurement table above was produced before the
change, precisely so the definition could not be chosen for being gentler.

## References

* `tests/unit/test_deflated_sharpe_var_sr.py` — the bound's monotone relaxation,
  its direction, and the two-trial floor
* `tests/unit/test_store.py::test_the_trial_sharpes_are_the_selection_pool_not_every_evaluation`
* `tests/unit/test_store.py::test_there_is_no_second_non_finite_filter_because_the_gates_subsume_it`
* ADR 0010 (what `var_sr` measures), ADR 0008 (undefined reports unmeasured),
  ADR 0012 (the gate that makes the pool non-empty)
