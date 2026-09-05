# Evolutionary optimisation — design notes

Normative source: `CLAUDE_CODE_MASTER_SPEC.md` §13 (and §8.6, §9.6, §14.4). This
document is **non-normative**: it explains why the design is shaped the way it
is, and works through the parts that are easy to get subtly wrong. Where the two
disagree, the spec wins.

*Status: specified, not implemented. Lands in phase F′ (T45–T54).*

---

## 1. Why a population

The original design ran a sequential loop: the model proposes a strategy, the
platform backtests it, the model reads the result and proposes the next one.
That has three problems, and only the third is obvious.

1. **One line of enquiry at a time.** A promising idea that happens to score
   badly on its first parameterisation is abandoned, because there is nowhere for
   it to wait while something else is tried.
2. **The searcher becomes the bias.** Every proposal is conditioned on the last
   result, so the search inherits whatever the model finds salient. Two runs of
   the same campaign explore the same corridor.
3. **The search is uncountable.** This is the serious one. Overfitting is a
   function of how many things you tried, and a sequential loop with a model in
   it has no principled count. The deflated Sharpe ratio needs an honest `M`.

A population fixes all three. Sixteen candidates advance together, so several
regions stay alive; selection is mechanical, so the model's taste is one input
rather than the control flow; and every evaluation is a countable trial that
§14.4 charges against the significance bar.

That last point deserves emphasis, because it inverts the usual intuition: a
*wider* search is not free. Running 30 generations of 16 candidates means 480
trials, and a strategy that emerges from it must clear a substantially higher
deflated-Sharpe threshold than one found in 20. The optimiser makes it easy to
search harder, and the statistics make you pay for it. Both halves are necessary;
either alone is a trap.

---

## 2. The shape of a generation

```
                    ┌──────────────────────────────────────────┐
   generation g     │  16 candidates                           │
                    └───────────────┬──────────────────────────┘
                                    │  evaluate on TRAIN only (cached runs)
                                    v
                        fitness = gates → weighted score → penalties
                                    │
                                    v
                              rank, then select
                                    │
                    ┌───────────────┼───────────────┐
                    v               v               v
              12 survivors     3 offspring     1 immigrant
              (niched, so      (mutations of   (novel genome;
               no near-        survivors)       more when
               duplicates)                      diversity is low)
                    └───────────────┼───────────────┘
                                    v
                    ┌──────────────────────────────────────────┐
   generation g+1   │  16 candidates                           │
                    └──────────────────────────────────────────┘
```

Two details that are easy to miss:

* **Survivors are re-evaluated, and it is free.** A survivor's
  `(strategy_id, params, dataset, split, segment, config, seed, engine)` tuple is
  unchanged, so `run_id` is unchanged, so §11.2's cache returns the stored run
  without executing anything. Elitism costs nothing, and `n_evaluations` counts
  only genuinely new work.
* **12 + 3 + 1 = 16 is checked, not assumed.** A configuration where the three
  numbers do not sum to the population size is a `ConfigError` at load. Silently
  resizing the population would make `n_evaluations` — and therefore every
  deflated Sharpe downstream — quietly wrong.

### Tuning note

The shipped default (12 survivors, 3 offspring, 1 immigrant) is deliberately
conservative: high elitism, low churn. It converges slowly and is unlikely to
lose a good candidate to noise. If a campaign stagnates — diversity at the floor,
best fitness flat for several generations — the lever is *more churn*, not more
generations: try 8 / 6 / 2. All four numbers are configuration, and changing them
does not require an ADR (only changing a *threshold* does).

---

## 3. Fitness, and the two traps it avoids

### Trap 1: optimising net profit

Ranking by return selects for luck, leverage and drawdown tolerance. The design
gives `net_return` a weight of **0.02** — present so that a profitable strategy
outranks an unprofitable one with otherwise identical statistics, and small
enough that it can never be the reason a candidate wins.

### Trap 2: optimising win rate

Win rate is seductive and nearly useless on its own: a strategy that takes a
0.1 % profit ninety times and a 20 % loss ten times has a 90 % win rate and
destroys capital. So win rate carries weight **0.04**, and — more importantly —
it sits *behind two hard gates*:

```
if expectancy_pct <= 0:      fitness = REJECTED     # F_EXPECTANCY
if max_drawdown > 0.50:      fitness = REJECTED     # F_DRAWDOWN
```

Gates run before the weighted score exists. There is no arithmetic by which a
high win rate can offset negative expectancy or an unacceptable drawdown, which
is the property the design was asked for, expressed as control flow rather than
as a weight.

### What the design actually prefers

"Frequent, consistent, profitable trades rather than a few huge winners" is
encoded four times over, deliberately redundantly:

| mechanism | weight / effect |
|---|---|
| `trades` component — `log1p(n)/log1p(200)` | +0.06, rewards frequency with diminishing returns |
| `consistency` component — share of 30-day windows in profit | +0.12 |
| `concentration` component — `1 − top5_profit_share` | +0.02 |
| `p_removal` penalty — the trade-removal test | multiplicative, can approach 0 |

The penalty is the sharp instrument. The components nudge; `p_removal` can
multiply a strategy's fitness by nearly zero.

### The trade-removal test

Remove the top `k` winning trades by pnl and recompute:

```
retention_k = Σ pnl(remaining) / Σ pnl(all)
```

Computed on the trade list, exactly and without re-simulation — cheap enough to
run for all 16 candidates every generation. Interpretation:

| observation | reading |
|---|---|
| `retention_1 ≈ 0.9` | profit is broadly distributed; the edge is plausible |
| `retention_1 ≈ 0.5` | half the profit came from one trade |
| `retention_1 ≤ 0` | **rejected**: the strategy loses money without its single best trade |

The `retention_1 ≤ 0` case is a hard gate rather than a penalty, because there is
no honest reading of it. The `k = 3` and `k = 5` results grade the penalty.

One caveat worth stating: this measures concentration, not causation. A genuine
trend-following strategy legitimately earns most of its money in a few large
moves, and will score poorly here. That is a known and accepted bias of this
configuration — the platform is tuned for frequent, consistent edges, and a
trend follower should be evaluated with different `removal_floor` values, set
deliberately and recorded in an ADR.

---

## 4. Mutation without invalid strategies

The requirement is that mutation cannot produce an invalid strategy. Editing
Python source cannot satisfy it — most edits to a token stream produce code that
does not run, and the dangerous minority produce code that runs and is wrong.

So mutation operates on a **genome**: a typed, validated description of entry
conditions, exit conditions, filters, risk controls, sizing and parameters, from
which module source is generated deterministically.

```
StrategyGenome ──mutate──▶ StrategyGenome ──compile──▶ Python source
   (validated)              (validated)                    │
                                                           ▼
                                          ast_check → sandbox → leakage probe
```

Four layers, in order:

1. **Validity by construction.** Operators emit genome edits, and pydantic
   validators reject a genome that names an unknown indicator, leaves a parameter
   unbounded, empties the entry condition, or declares a warm-up shorter than its
   longest lookback. An invalid genome cannot be *constructed*, so it cannot
   enter the population.
2. **Redraw.** An operator that fails validation is retried with a fresh RNG draw,
   up to `max_repair_attempts`.
3. **Fall back to an immigrant.** If every attempt fails, the slot is filled by a
   novel candidate. The child is never salvaged by relaxing a rule — that path
   ends with a population of strategies that only *nearly* satisfy the contract.
4. **Defence in depth.** The compiled source still passes the AST checker, still
   runs in the sandbox, and still clears the leakage probe. Layer 1 should make
   these redundant; they exist because "should" is not a guarantee.

Free-form Python strategies (human- or LLM-authored) remain first-class as
`opaque` candidates. They compete on the same fitness and admit parameter
mutation; they simply cannot be mutated structurally, because their structure is
not addressable.

### Why risk controls moved into the engine

Stop-loss, take-profit and trailing stops are listed as mutable parameters, which
means they need a fixed, typed surface to mutate. They also cannot be implemented
inside a strategy: a stop is an *intrabar* event, and `BarWindow` deliberately
refuses to expose intrabar information (INV-3). A strategy that tried would
either look ahead or approximate.

So they became engine behaviour (§8.6), declared per candidate in `RiskSpec`.
The engine applies them pessimistically: fills at the worse of trigger price and
bar open, stop-loss wins a same-bar tie with take-profit, and nothing fires on a
gap-filled bar. Every one of those choices costs the strategy money, which is the
correct direction for a rule chosen under uncertainty — otherwise evolution finds
and exploits the optimism, reliably and invisibly.

---

## 5. Diversity, and why behaviour matters more than structure

Left alone, a population converges. The leader's descendants fill every slot
within a few generations and the search stops exploring. Worse, the *appearance*
of confirmation survives: sixteen near-identical strategies produce sixteen
agreeing validation results, which reads like independent evidence and is not.

Similarity is therefore measured on two axes and weighted towards the one that
matters:

```
similarity = 0.4 · jaccard(structural signature)
           + 0.6 · agreement(position_frac series)
```

Behaviour dominates on purpose. Two strategies built from different indicators
that enter and exit on the same bars are the same strategy wearing two hats, and
treating them as diverse is precisely the error that makes a population look
better-explored than it is.

Three mechanisms, layered:

1. **Niching in selection** — a candidate too similar to a fitter survivor is
   skipped in favour of the next one down.
2. **Diversity floor** — while `1 − mean pairwise similarity` is below the floor,
   immigrants increase and offspring decrease.
3. **Reserved novelty** — at least one candidate per generation owes nothing to
   the current leader.

---

## 6. Where the segments are

This is the part that has to be right.

```
┌──────────────── TRAIN ────────────────┐  ┌── VAL ──┐  ┌── TEST ──┐
│  inner folds: IS | OOS | IS | OOS ... │  │         │  │  locked  │
└───────────────────────────────────────┘  └─────────┘  └──────────┘
        ▲                                       ▲             ▲
        │ every generation, all 16              │             │
        │                                       │             │
   evolution                            promotion only,   quantlab
   (fitness, selection,                 recorded, budgeted,  lockbox
    mutation, stopping)                 counted as a       only
                                        validation touch
```

**The inner walk-forward is the key move.** Fitness must reward generalisation,
but measuring generalisation on the validation segment would burn it sixteen
times per generation and turn it into training data by the fourth. So train is
split into its own in-sample/out-of-sample folds, and the `inner_oos` fitness
component (weight 0.12) and the `p_divergence` penalty are computed there.

The validation segment is reached only through `promote()`, which:

* is limited to `n_promote` candidates, chosen with a similarity constraint so
  the budget is not spent three times on one strategy;
* writes the `candidate_promotion` row **before** the run executes, so a crash
  cannot hide a touch;
* increments `strategy_family.validation_touches`, which feeds `M` in §14.4.

And no validation-derived number is ever fed back — not into fitness, not into
selection, not into stopping, and (INV-11) not into an LLM prompt. Promotion
results inform the human and the final report. If they informed the loop, the
validation segment would be a second training set with a good reputation.

The test partition is never reachable during evolution: the container is built
with `profile="research"`, and `PartitionGuard` raises rather than returning
clipped data.

---

## 7. Lineage

Every candidate except a seed records its parent and the ordered list of
mutations that produced it, each with the RNG seed that drew it. That makes
INV-10 checkable:

> re-applying a child's recorded mutations to its parent's genome reproduces the
> child's genome, byte for byte.

Which converts "we can visualise how a strategy evolved" from a reporting feature
into a property the build enforces. A lineage that cannot be replayed is a bug,
not a missing chart.

```
seed:sma_cross
  └─ gen1 mutant  (perturb_numeric fast 50→43)
       ├─ gen2 mutant  (add_confirmation rsi < 70)
       │    └─ gen3 mutant  (perturb_risk stop_loss 0.02→0.031)   ← promoted
       └─ gen2 mutant  (change_tree_mode all→any)                 ← died gen 3
```

---

## 8. Known limitations

Recorded so they are chosen rather than discovered.

* **No crossover.** Mutation only. Recombining two genomes is a much harder
  correctness problem (which conditions are compatible? what happens to the
  parameter namespace?), and mutation-only search with a population is a
  well-understood evolution strategy. Adding crossover later requires an ADR.
* **The concentration bias.** As noted in §3, this fitness configuration
  systematically disfavours trend-following. That is a deliberate choice for the
  default profile, not an oversight.
* **Walk-forward cost.** Re-searching each of 23 windows with 8 generations of 16
  candidates is ~2 900 evaluations. This is the dominant cost of the pipeline,
  and it raises the deflated-Sharpe bar accordingly — correctly, but it must be
  budgeted for.
* **Fitness weights are not themselves validated.** They encode a preference, not
  a fact. They are configuration, and a campaign that tunes them until a
  favourite strategy wins has simply moved the overfitting up one level. The
  honest use is to fix them before a campaign and record them in the verdict's
  `thresholds_json`, which §14.4 already does.
