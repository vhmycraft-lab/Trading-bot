# 3. Replace the sequential research loop with an evolutionary optimiser

* **Status:** Accepted
* **Date:** 2026-09-05
* **Supersedes:** spec §13 (v1.0), §12.4 `ReviewDecision`, task T36
* **Amends:** spec §0.2, §2.1, §3, §5, §6, §8, §9, §11.1, §12, §14, §15, §19, §20, §22, §23
* **Companion:** `docs/EVOLUTION.md` (non-normative)

## Context

Spec 1.0 searched strategy space with a sequential loop: the LLM proposes a
strategy, the platform backtests it, the LLM reads a redacted report and proposes
the next one (`ReviewDecision` with `MODIFY / NEW_IDEA / ABANDON_FAMILY / STOP`).
Parameter search within a strategy was a separate Optuna study (§13).

Three problems, in increasing order of severity:

1. **One line of enquiry at a time.** A promising idea whose first
   parameterisation scores badly is abandoned; there is nowhere for it to wait.
2. **The searcher is the bias.** Each proposal is conditioned on the last result,
   so the model's priors determine which corridor of strategy space is explored,
   and the exploration is not reproducible in any meaningful sense.
3. **The size of the search is uncountable.** This is the one that matters.
   Overfitting scales with how many things were tried, and §14.4 needs an honest
   `M` for the deflated Sharpe ratio. A loop with a language model in the control
   flow supplies no principled count, so the platform's central statistical claim
   rested on a number it could not justify.

The project owner has additionally specified requirements that spec 1.0 has no
place to put: mutable stop-loss / take-profit / trailing-stop / position-sizing
parameters (the strategy contract has none), structural mutation of entry and
exit logic (there is no addressable structure — strategies are free-form
Python), and a robustness test based on removing the best trades.

## Decision

Replace the sequential loop with a **population-based evolutionary optimiser**,
specified in full as §13 of spec 1.1.

1. **Population.** 16 candidates; each generation keeps 12 survivors, produces 3
   offspring by mutation and 1 immigrant. All four numbers are configuration; the
   identity `survivors + offspring + immigrants == population_size` is validated
   at load rather than silently corrected.

2. **Fitness (§13.3) is gated, weighted and penalised, in that order.** Ten
   components, of which `net_return` carries weight 0.02 and `win_rate` 0.04.
   Hard gates on expectancy, trade count, drawdown and single-trade dependence run
   *before* the weighted score exists, which is how "win rate must never override
   negative expectancy or excessive drawdown" becomes control flow rather than a
   hopeful weighting. Robustness penalties multiply, so independent fragilities
   compound.

3. **The genome (§9.6).** Structural mutation cannot operate safely on free-form
   Python. Candidates carry a typed, validated declarative genome that compiles
   deterministically to module source. Mutations edit the genome, so a mutated
   candidate is valid by construction; the compiled source still passes the AST
   check, the sandbox and the leakage probe. Free-form strategies remain as
   `opaque` candidates with parameter mutation only.

4. **Risk controls become engine behaviour (§8.6).** Stops are intrabar events,
   and `BarWindow` refuses intrabar information (INV-3), so a strategy cannot
   implement one without either looking ahead or approximating. `RiskSpec` is
   declared per candidate and applied by the engine, pessimistically: fills at
   the worse of trigger and open, stop-loss wins a same-bar tie with take-profit,
   nothing fires on a gap-filled bar.

5. **Diversity (§13.5)** is measured as a weighted combination of structural
   signature and *behavioural* agreement, with behaviour weighted higher —
   because two strategies that trade identically are one strategy, and treating
   them as two is what makes a converged population look well-explored.

6. **The inner walk-forward (§13.6).** Fitness must reward generalisation, but
   measuring it on the validation segment would consume that segment 16 times per
   generation. Train is therefore split into its own IS/OOS folds; the `inner_oos`
   component and the divergence penalty are computed there. Validation is reached
   only through a recorded, budgeted promotion.

7. **Lineage (§6, §13.6)** is persisted as `candidate` + `mutation` rows carrying
   each operator, its genome path, its before/after values and its RNG seed.

8. **The LLM's role changes (§12)** from author to proposer: seed genomes,
   immigrants, and optionally a suggested mutation operator. It proposes *JSON
   genomes*, never code, and it never decides survival, ranking or stopping.

9. **Optuna is retained but demoted (§13.8)** to parameter refinement of promoted
   candidates. Plateau selection becomes mandatory before any validation run.

10. **New invariants INV-9 (segment discipline), INV-10 (replayable lineage) and
    INV-11 (no validation data in prompts during evolution)**, each with a named
    enforcing test, in the same style as INV-1…INV-8.

11. **`M` in the deflated Sharpe ratio (§14.4) now includes
    `evolution_run.n_evaluations`.** A wider search must clear a higher bar. This
    is the change that makes the whole design statistically honest, and it is
    non-negotiable: without it, evolution is a machine for manufacturing
    plausible-looking overfits.

## Alternatives considered

* **Keep the sequential loop, add Optuna breadth.** Cheapest, but leaves the
  uncountable-search problem untouched and provides nowhere to put structural
  mutation or mutable risk controls.
* **Genetic programming over Python ASTs.** Maximum expressiveness, but validity
  becomes a filtering problem rather than a construction property, and the
  fraction of valid-and-meaningful mutants is low enough to dominate the compute
  budget. Rejected in favour of the constrained genome.
* **Add crossover.** Deferred, not rejected. Recombining two genomes raises
  questions (which conditions compose? whose parameter namespace wins?) that
  mutation-only search does not. Adding it later requires its own ADR.
* **Multi-objective Pareto ranking (NSGA-II) instead of a scalar fitness.**
  Genuinely attractive, and closer to the truth than any weighting. Rejected for
  v1 because a Pareto front gives no single ordering for survival selection, and
  the tie-breaking policy that would supply one is a scalarisation in disguise —
  with the weights hidden instead of written down. Revisit once the scalar
  version has produced a campaign's worth of evidence.

## Consequences

**Cost.** Phase F grows from three tasks to ten (T45–T54). Walk-forward with
per-window re-search becomes the dominant runtime cost of the pipeline
(~2 900 evaluations for the default 23-window configuration).

**Benefit.** The search is countable, reproducible from a seed, and auditable
candidate by candidate. Every ranking decision can be reconstructed from stored
components and penalties, and every candidate's ancestry can be replayed.

**Risk accepted.** The fitness weights encode a preference, not a fact, and this
configuration systematically disfavours trend-following (see `docs/EVOLUTION.md`
§3). Tuning the weights until a favourite strategy wins would move the
overfitting up one level; the mitigation is that weights are fixed before a
campaign and snapshotted into each verdict's `thresholds_json`.

**Migration.** No implemented code changes in this ADR — the specification and
architecture are updated, implementation lands in phase F′. `configs/default.yaml`
and `core/config.py` do **not** yet carry the `evolution:` section; that gap is
closed by amended task T04 and is the first thing phase F′ must do.
