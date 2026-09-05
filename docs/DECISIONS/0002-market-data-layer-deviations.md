# 2. Deviations in the market-data layer

* **Status:** Accepted
* **Date:** 2026-09-05
* **Covers:** spec tasks T05-T09 (phase B)

## Context

ADR 0001 requires a record for any deviation from the file tree, phase order or
declared signatures of `CLAUDE_CODE_MASTER_SPEC.md`. Building the historical
market-data subsystem produced five, each driven by the project owner's phase
brief — which stated that *"the system must make it impossible for the
backtester to accidentally use future data when generating a historical
signal"* — or by a contradiction inside the specification itself.

No threshold, split boundary, cost default or metric definition was changed.

## Decisions

### 1. `BarWindow` lives in `core/types.py`, not `core/strategy.py`

Spec §9.1 lists `BarWindow` alongside the strategy contract, which is phase C.
The owner's phase-2 brief makes look-ahead prevention a phase-2 deliverable, and
`BarWindow` is the mechanism that provides it. It is also a property of a
`BarFrame` rather than of a strategy: it is a causal view over bars, and it is
meaningful with no strategy in sight.

`core/strategy.py` will import it from `core/types.py` in phase C. Nothing about
the contract in §9.1 changes.

### 2. `PartitionGuard` is built now, not in phase G

Spec §22 assigns `PartitionGuard` to T32. It is the second half of "cannot use
future data": `BarWindow` stops a strategy reading bar `t+1` inside a run, while
the guard stops the whole system reading the held-out partition between runs.
Shipping one without the other would leave the brief half-met, and the guard is
small and fully testable against `core/splits.py`, which phase B delivers anyway.

`build_container` attaches it for every profile except `lockbox`, so no caller
can obtain an unguarded research source. A guarded profile with no usable split
policy now **fails to build** rather than falling back to an unguarded source.

### 3. `validate_bars` keeps its signature; `normalise_bars` does the repairing

Spec §7.3 declares `validate_bars(df, timeframe) -> ValidationReport` and, in the
same paragraph, requires that gaps be filled — which that signature cannot
express, since a report is not a frame.

Rather than change the declared signature, the two jobs are separated:

* `validate_bars(df, timeframe) -> ValidationReport` — checks only, never repairs
* `normalise_bars(df, timeframe, *, allow_gaps) -> (DataFrame, ValidationReport)`
  — sorts, de-duplicates, fills short gaps, then calls `validate_bars`

Every ingestion path goes through `normalise_bars`, so archive bars and REST tail
bars are validated identically.

### 4. Real bars supersede gap fills

Not covered by the spec, and discovered by a test. A gap fill is a placeholder,
not an observation. When the exchange later publishes the real bar for a filled
timestamp — a backfilled archive month, a REST tail — the two would otherwise
look like a contradiction and ingestion would refuse to proceed.

Synthetic rows are therefore dropped in favour of real ones at the same
timestamp, and the count is reported as `n_gap_fills_superseded`. Two
*conflicting real* observations still raise, which is the case that genuinely
needs a human.

### 5. `CcxtRestSource` deliberately does not implement `MarketDataSource`

Spec §7.4 defines the port as returning stored history addressed by a
`dataset_id`. Live REST output has no content hash and is not reproducible, so
implementing the port would let an unreproducible run be handed to a backtest as
if it were reproducible history — INV-7 in reverse.

The adapter exposes `fetch_range` and `fetch_tail` instead. Tail bars become
history by being merged into the Parquet dataset, where they acquire a
`dataset_id` like everything else.

### 6. `ports/store.py` declares only the dataset and split methods

Spec §11.1 lists the full `ExperimentStore` protocol, most of which carries
record types that do not exist until phases C-E. Declaring them now with
placeholder types would guarantee a rewrite in T21.

`ArtifactStore` is complete; `ExperimentStore` declares the dataset and split
methods, whose types exist. Protocols are structural, so growing one later
breaks nothing.

## Consequences

* `alembic.ini`, `migrations/`, `core/types.py`, `core/data_validation.py`,
  `core/splits.py`, `adapters/data/*` and `ports/{clock,data,store}.py` exist at
  or ahead of their phase letters in spec §3.
* Phase C inherits `BarWindow` and `Bar*` types rather than defining them, which
  reduces the work in T05/T12 and removes the risk of two causal-window
  implementations disagreeing.
* The `lockbox` profile is now the single, explicit route to test-partition data,
  and every other route fails loudly.
