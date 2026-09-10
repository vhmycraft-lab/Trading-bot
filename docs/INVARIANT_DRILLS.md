# Invariant drills — evidence

Non-normative. `CLAUDE_CODE_MASTER_SPEC.md` is the specification.

Section 23's second acceptance criterion says every invariant has been
**demonstrated to fail when broken**. Until now that was partly a claim: some
invariants had a named test and no record that anyone had ever watched it go
red, which makes the definition of done unverifiable. A test that has only ever
passed is indistinguishable from a test that cannot fail.

This file records the drill for each of INV-1 to INV-11: what was broken, which
test went red, and what the failure looked like.

**Date of this run:** 2026-09-10
**Tree:** `86b84ee` (working tree clean before and after; every edit reverted)

## Method

Each drill is mechanical and identical in shape:

1. Run the named test and confirm it is **green**. A drill that starts red
   proves nothing.
2. Apply a single, minimal edit that removes the protection — not a
   cosmetic change, and not one that breaks the module in some unrelated way.
   The edit is asserted to be a real change to the file, so a patch that
   silently failed to apply cannot be recorded as a pass.
3. Run the test again and confirm it is **red**, and record the failure.
4. Revert, and confirm the test is **green** again.

An invariant is CONFIRMED only if all three observations hold. The whole run
was driven from a script rather than by hand, and the working tree was clean
before and after — the revert is part of the drill, not a tidy-up afterwards.

### What this evidence is not

It shows each guard fires against **the specific defect drilled**. It does not
show the guard catches every possible violation of the invariant, and no
finite set of drills could. Where a drill is weaker than the invariant it
stands for, the entry says so.

---

### INV-1 — CONFIRMED

> No code path can sign, send, or simulate sending a real-money order.

**How it was broken.** Defined a class named LiveBroker in the paper package.  
Edited `src/quantlab/paper/broker.py`.

**Test run.** `tests/unit/test_architecture.py::test_inv1_no_live_broker_class`

**Result.** Green before · **red when broken** · green after revert.  
Failure observed: `FAILED tests/unit/test_architecture.py::test_inv1_no_live_broker_class - Asse...`

### INV-2 — CONFIRMED

> No API key, secret, token or password literal in the repository.

**How it was broken.** Assigned a long opaque value to a name the INV-2 scanner watches.  
Edited `src/quantlab/core/execution.py`.

**Test run.** `tests/unit/test_no_secrets.py::test_repository_contains_no_credential_literals`

**Result.** Green before · **red when broken** · green after revert.  
Failure observed: `FAILED tests/unit/test_no_secrets.py::test_repository_contains_no_credential_literals`

### INV-3 — CONFIRMED

> A strategy can never observe bar t+1 when deciding at bar t.

**How it was broken.** Dropped the scaled tail replacements, leaving only reversal — the weaker check amendment 1.1.7(a) replaced.  
Edited `src/quantlab/core/validation/leakage.py`.

**Test run.** `tests/leakage/test_leakage_probe.py`

**Result.** Green before · **red when broken** · green after revert.  
Failure observed: `tests/leakage/test_leakage_probe.py:76: AssertionError | FAILED tests/leakage/test_leakage_probe.py::test_every_leaky_fixture_is_detected[future_close.py]`

### INV-4 — CONFIRMED

> Untrusted code never executes in the main process.

**How it was broken.** Called eval() in core/, outside sandbox/.  
Edited `src/quantlab/core/execution.py`.

**Test run.** `tests/unit/test_architecture.py::test_inv4_no_dynamic_execution_outside_sandbox`

**Result.** Green before · **red when broken** · green after revert.  
Failure observed: `tests/unit/test_architecture.py:198: AssertionError | FAILED tests/unit/test_architecture.py::test_inv4_no_dynamic_execution_outside_sandbox`

### INV-5 — CONFIRMED

> The test partition is never loaded outside the lockbox profile.

**How it was broken.** Removed the PartitionGuard wrapper so every profile reads unguarded bars.  
Edited `src/quantlab/container.py`.

**Test run.** `tests/unit/test_lockbox.py::test_the_research_profile_refuses_a_test_partition_range tests/unit/test_lockbox.py::test_the_lockbox_profile_is_the_only_one_that_may_read_the_test_partition`

**Result.** Green before · **red when broken** · green after revert.  
Failure observed: `src/quantlab/adapters/data/binance_archive.py:405: DataError | FAILED tests/unit/test_lockbox.py::test_the_research_profile_refuses_a_test_partition_range`

### INV-6 — CONFIRMED

> Nothing derived from the test partition is ever placed in an LLM prompt.

**How it was broken.** Emptied the test-partition scan loop so a prompt naming test metrics passes.  
Edited `src/quantlab/research/redaction.py`.

**Test run.** `tests/unit/test_redaction.py`

**Result.** Green before · **red when broken** · green after revert.  
Failure observed: `FAILED tests/unit/test_redaction.py::test_the_scanner_is_capable_of_refusing_something`

### INV-7 — CONFIRMED

> Every stored run reproduces; a replayed evolution run yields the identical candidate sequence.

**How it was broken.** Restored the wall-clock term in candidates_for — the original baseline defect.  
Edited `src/quantlab/adapters/store/sqlite.py`.

**Test run.** `tests/unit/test_store.py::test_the_candidate_order_does_not_move_with_the_clock`

**Result.** Green before · **red when broken** · green after revert.  
Failure observed: `FAILED tests/unit/test_store.py::test_the_candidate_order_does_not_move_with_the_clock`

### INV-8 — CONFIRMED

> Import boundaries between packages.

**How it was broken.** Imported an adapter from core/.  
Edited `src/quantlab/core/execution.py`.

**Test run.** `tests/unit/test_architecture.py::test_inv8_core_imports_only_core tests/unit/test_architecture.py::test_inv8_only_container_and_cli_import_adapters`

**Result.** Green before · **red when broken** · green after revert.  
Failure observed: `tests/unit/test_architecture.py:244: AssertionError | FAILED tests/unit/test_architecture.py::test_inv8_core_imports_only_core - As...`

### INV-9 — CONFIRMED

> Evolution evaluates on train only; validation is reachable only via the recorded promotion path.

**How it was broken.** Made require_evolution_segment return unconditionally, admitting the validation segment.  
Edited `src/quantlab/evolution/loop.py`.

**Test run.** `tests/unit/test_evolution_segments.py`

**Result.** Green before · **red when broken** · green after revert.  
Failure observed: `FAILED tests/unit/test_evolution_segments.py::test_everything_else_is_refused[val]`

### INV-10 — CONFIRMED

> Lineage is complete and replayable; re-applying a child's mutations reproduces its genome.

**How it was broken.** Dropped the recorded mutations during replay, so a child rebuilds as its parent.  
Edited `src/quantlab/evolution/lineage.py`.

**Test run.** `tests/unit/test_lineage.py`

**Result.** Green before · **red when broken** · green after revert.  
Failure observed: `FAILED tests/unit/test_lineage.py::test_inv10_holds_over_a_stored_five_generation_run`

### INV-11 — CONFIRMED

> While a run is active, no validation- or test-derived quantity enters an LLM prompt.

**How it was broken.** Emptied the validation scan loop so a generation prompt may carry val metrics.  
Edited `src/quantlab/research/redaction.py`.

**Test run.** `tests/unit/test_redaction.py`

**Result.** Green before · **red when broken** · green after revert.  
Failure observed: `FAILED tests/unit/test_redaction.py::test_a_validation_quantity_is_refused_while_a_run_is_active[the val_sharpe of the leader is 1.9]`

---

## Notes on individual drills

**INV-5** is the one drill whose failure mode is worth reading closely. With
the `PartitionGuard` in place, a research-profile request for a test-partition
range raises `LockboxViolation` before any data is touched. With the guard
removed, the same call falls through to the data layer and raises `DataError`
instead — because this machine holds no bars for that range. The test fails
either way, which is what the drill needs to show, but the *reason* it fails
changes: green is "refused at the boundary", red is "reached the data layer at
all". That is the property INV-5 actually asserts, and it is visible in the
drill precisely because the guard short-circuits so early.

**INV-3**'s drill weakens the probe rather than deleting it: `TAIL_MODES` drops
the two scaled tail replacements and keeps only reversal, which is the weaker
check that amendment 1.1.7(a) replaced. That is a more useful drill than
removing the probe altogether — it shows the *scaled* replacements are load
bearing, and the one-bar peek in `future_close.py` is what escapes without
them.

**INV-7** is drilled by restoring the original defect: putting `created_at`
back into the `candidates_for` sort key. It is the only drill here that
reproduces a bug this repository actually shipped, and it was found by
`make check` being red on a fresh clone rather than by a drill.

## Re-running

The drills are scripted rather than manual, but the scripts are scratch
artefacts and are not committed — they patch source files in place, and a
committed tool that edits `src/` on purpose is a hazard with no other use. To
reproduce, follow the four steps in **Method** for the file and target named in
each entry.

Anyone repeating this should expect the tree to be clean at the end. If it is
not, a drill did not revert, and the next `make check` is measuring a
deliberately broken tree.
