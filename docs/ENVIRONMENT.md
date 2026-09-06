# Hidden randomised training environments

*Project Rome sections 4–19, 26, 35, 36, 43, 47. Implemented on top of the
existing QuantLab architecture rather than replacing it (Rome §45).*

Evolutionary search run for hundreds of generations against one fixed slice of
history will find that slice's accidents. This subsystem is the countermeasure:
**every generation is evaluated inside a freshly selected training environment**,
drawn by Rome's own infrastructure, hidden from the strategy and the LLM, and
recorded for audit.

Rome §47 asks that each requirement come with its implementation location, its
tests, the result, and its known limitations. That is what the tables below are.

---

## 1. What varies, and what deliberately does not

| Quantity | Varies? | Where | Why |
|---|---|---|---|
| Training window | **yes** | `EnvironmentSettings.window_*`, `stride_fraction` | The point of the exercise (§8). |
| Starting capital | **yes**, 5 000–50 000 | `min/max_starting_capital` | §10. A strategy dependent on one account size has not been tested. |
| Slippage level | **yes**, 2–8 bps | `min/max_slippage_bps` | §11. Tests dependence on one exact execution assumption. |
| Asset subset | **yes**, when a universe is configured | `asset_universe`, `min_assets` | §15. |
| **Commission** | **no** | — | §12. A structural property of the venue, not a source of uncertainty. |
| **Execution timing / fill rule** | **no** | — | §13. Same reason; varying it manufactures difficulty. |
| **Slippage *model*** | **no** | — | §39. Changing *how* slippage is computed is a change of execution model, which is versioned, not sampled. |

`TrainingEnvironment` has no field for a commission or a delay. That is the
strongest available form of §12 and §13: not a rule someone must follow, but a
value that has nowhere to live. `tests/unit/test_environment.py::
test_no_field_of_the_environment_is_a_commission_or_an_execution_delay` asserts
it, and `tests/integration/test_environment_evolution.py::
test_the_environment_varies_capital_and_slippage_but_never_commission` checks the
settings the loop was actually handed.

Every band is documented in `configs/default.yaml` and bounded. Nothing here
makes conditions worse than the venue exhibits (§9).

---

## 2. How a window is chosen

```
split policy  ──►  build_window_pool  ──►  pool of valid windows
                                              │
                     secrets.token_bytes(32) ──┤  (OS CSPRNG, per generation)
                                              ▼
                          HMAC-SHA256 expansion  ──►  TrainingEnvironment
                                              │
                    recorded to training_environment
                                              │
                                              ▼
                    bars sliced, BacktestConfig derived, generation runs
```

**Validity is structural.** The pool is enumerated from the *training segment*
before any drawing happens, so a window that could reach the validation segment
or the test partition is never a candidate. There is no post-hoc check to forget
to run.

**Independence.** Nothing about the previous generation, the generation index,
the population, the clock, a strategy id or a code hash enters the draw. The
search cannot learn "next generation = next historical period", and the sequence
of window starts steps backwards about as often as forwards (measured: 47 back /
51 forward over 100 draws).

**Unpredictability.** The seed is 32 bytes from `secrets.token_bytes`, which is
`os.urandom`. Every subsequent choice is an HMAC-SHA256 expansion of it, with
rejection sampling so there is no modulo bias toward one end of history.

**Reproducibility for privileged audit.** Because the expansion is a pure
function of the seed, `core.environment.reproduce` rebuilds an environment
exactly from its recorded row. It refuses a record from a different derivation
version rather than approximating it.

HMAC rather than a seeded PRNG, for two reasons: the seed is secret and HMAC's
output reveals nothing about the key (a Mersenne twister's state is recoverable
from its output), and HMAC is stable across interpreter versions — an audit
reproduction that changed with the Python release would be no reproduction.

---

## 3. What is hidden, and what honestly is not

Rome §7 is explicit that the historical period cannot be hidden from a strategy
that receives timestamped candles, and this implementation does not pretend
otherwise: **no timestamp is ever rewritten.** What is hidden is the *selection* —
which environment Rome chose and why — and that hiding is structural:

| Boundary | Mechanism | Test |
|---|---|---|
| Strategy cannot receive the environment | `SandboxRequest` has no field for it and `extra="forbid"` | `test_the_sandbox_wire_format_has_no_field_for_the_environment` |
| Strategy cannot receive the seed | `Context.__slots__` carries none of it; `BacktestConfig` has no seed field of Rome's kind | `test_a_strategy_context_exposes_nothing_about_the_environment` |
| Sandbox cannot reach the selector | no `quantlab.sandbox.*` module imports `core.environment` | `test_no_sandbox_module_can_import_the_environment_selector` |
| Only three modules hold the selector at all | asserted as an exact list | `test_the_selector_is_reachable_from_only_the_layers_that_must_hold_it` |
| The loop holds no seed | `GenerationEnvironment` is exactly `{bars, config, environment_id}` | `test_the_loop_receives_the_value_type_and_not_the_selector` |
| LLM cannot specify a window or seed | `select` takes one argument and there is no other entry point | `test_the_selector_has_no_parameter_that_could_carry_a_preference` |
| No prompt/report module can reach it | import guard, written to fail when `research/` appears | `test_no_prompt_or_research_module_can_reach_the_environment` |
| `seed_hex` appears nowhere else | exact-set assertion over the whole source tree | `test_no_module_outside_the_audit_path_mentions_the_seed_field` |
| Identifiers are opaque | `environment_id = sha256(seed)[:16]` | `test_the_environment_id_does_not_leak_the_seed_or_the_window` |

`Context.rng` exists so a strategy can be stochastic reproducibly; it is seeded
from `BacktestConfig.seed`, which is run configuration, **not** the environment
secret. Conflating the two would hand the secret to any strategy that asked for a
random number, and `test_a_strategys_own_rng_seed_is_not_the_environment_seed`
holds them apart.

---

## 4. The audit record

`training_environment` (migration `0003`), one row per generation:

`environment_id`, `evolution_id`, `gen_index`, `window_start_ts`,
`window_end_ts`, `window_id`, `starting_capital`, `slippage_bps`,
`asset_universe_json`, `seed_hex`, `pool_id`, `dataset_version`,
`execution_model_version`, `derivation_version`, `created_at`.

Two properties matter as much as the columns:

* **Written before the generation runs.** A record filed afterwards would be
  missing exactly the generations that crashed, which are the ones an auditor
  most wants to reconstruct. (`test_the_environment_is_recorded_before_its_
  generation_runs` makes a generation fail and asserts the row survives.)
* **Append-only and unique per generation.** The table has no `MUTABLE_COLUMNS`
  entry, so the store's append-only guard refuses any update; `UNIQUE
  (evolution_id, gen_index)` makes a second draw for one generation a database
  error rather than a silently different experiment.

The seed is stored here and nowhere else.

---

## 5. Resume, and the invariant this feature endangers

Per-generation randomisation and replaying a run from the store are in direct
tension. If `evolve resume` re-drew, every resumed run would be a different
search wearing the same run id.

`GenerationEnvironmentProvider` resolves it by **reading before it writes**: a
generation that already has a recorded environment is replayed under it, and only
genuinely new generations draw. It also checks the reproduction against the
stored row, because a derivation that changed since the run was recorded would
otherwise replay the run on different history while claiming not to.

INV-7's evolution clause therefore reads, under Rome:

> A run replayed from its stored seed **and its stored environments** produces the
> identical sequence of `candidate_id`s.

Held by `test_a_resumed_run_reproduces_the_candidates_of_the_run_it_resumes` and
`test_a_resumed_generation_replays_its_recorded_environment`.

**A consequence, measured rather than assumed:** a survivor carried forward is no
longer a run-cache hit, because the environment is part of a run's identity. That
is the point — a survivor only ever measured on one window has not been shown to
survive anything — and it means a campaign costs more evaluations than it did
under a fixed environment. `test_a_survivor_is_re_evaluated_on_the_next_
generations_history` measures the difference rather than trusting this paragraph.

The environment id also qualifies the *segment* (`train:9f2c…`). Without that,
two generations that drew different windows but the same capital and slippage
would share a run id — a run's identity does not include the bar range — and the
second would be served the first one's results. Silently wrong backtests, from a
cache doing its job.

---

## 6. Exposure tracking (§16, §18, §43)

`sampling_report` answers "how often has evolutionary pressure seen this bar?",
and gets there through two decisions:

* **Observations, not windows.** Windows overlap. Two disjoint windows drawn once
  each and one window drawn twice are very different exposures and identical
  window counts.
* **Against the pool, not a flat line.** Windows are contiguous, so uniform draws
  still cover the middle of history far more than its edges — a window can
  overlap the middle from either side and the first bucket from only one.
  Measured on the default settings, 400 genuinely uniform draws reach **1.6× the
  flat share.** A flat baseline would therefore report every healthy campaign as
  concentrated and its warning would be worthless. The baseline is the exposure
  the pool itself produces, which isolates what the campaign did from the pool's
  geometry, which the campaign did not choose.

Measured behaviour:

| Situation | `concentration` | Flagged? |
|---|---|---|
| 50 uniform draws | 1.24 | no |
| 400 uniform draws | 1.06 | no |
| 1000 uniform draws | 1.09 | no |
| One window drawn 50 times | ≫ 1.75 | **yes** (over-used *and* unsampled buckets) |
| Nothing drawn yet | 0.00 | **yes** (everything unsampled) |

Threshold: `overuse_factor: 1.75`. A flagged report is *reported*, never
corrected: Rome §43 forbids "solving" concentration by excluding history, and
re-weighting the pool would make the draw no longer uniform over valid windows.
`quantlab evolve run` and `resume` print it at the end of every run.

---

## 7. Known limitations

Stated plainly, per Rome §47.

1. **Asset-universe randomisation is implemented but unexercised.** The split
   policy is single-symbol, so the default universe is the split's own symbol and
   every environment gets the same one. The draw is tested against a synthetic
   four-asset universe; what is *not* implemented is §15's requirement that an
   asset actually existed during the selected window, because there is no
   multi-asset dataset to check listing dates against. Configuring
   `environment.asset_universe` today would sample assets without that check.
2. **Regime classification is not implemented.** §16 asks for statistics on how
   often each *regime* was sampled. Exposure is tracked by historical period,
   which is §43's question; labelling periods bull/bear/sideways would need a
   regime model this platform does not have, and inventing one would put an
   unvalidated classifier in the audit trail.
3. **One environment per generation, not per strategy.** §17 permits either and
   requires the policy be documented: this is the per-generation policy. Every
   candidate in a generation faces the same history, which keeps the generation's
   ranking a like-for-like comparison — the ranking is what selection consumes,
   and ranking candidates measured on different windows would make selection
   partly a lottery over environments.
4. **Funding, leverage, margin and liquidation (§14) are not modelled at all.**
   The engine is spot, long-or-flat. This is a gap in the platform, not in the
   randomisation; nothing here fabricates a funding rate.
5. **The seed is stored in plaintext in the local SQLite file.** It is protected
   from the strategy (which runs in a sandbox with no database access) and from
   the LLM (which has no path to it), not from someone who already has the
   database file. Rome §19's "store the seed securely for reproducibility" is met
   at the level this single-user local platform operates at, and no further.
