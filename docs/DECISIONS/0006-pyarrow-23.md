# 6. pyarrow 17.0.0 -> 23.0.1

* **Status:** Accepted
* **Date:** 2026-09-10

## Context

`pip-audit` reported PYSEC-2026-113 (CVE-2026-25087, GHSA-rgxp-2hwp-jwgg)
against pyarrow 17.0.0: a use-after-free in Apache Arrow C++, affecting 15.0.0
onwards, fixed in 23.0.1.

Spec §0.3 requires an ADR for any dependency change, and pyarrow writes every
byte of the processed dataset, so the bump had to be shown not to disturb data
already on disk.

## Was this pipeline ever exposed?

No, on two independent grounds:

1. The advisory states the functionality is not reachable from the language
   bindings: "The functionality is not exposed in language bindings (Python,
   Ruby, C GLib), so these bindings are not vulnerable." QuantLab is a pure
   Python consumer.
2. The defect is in the Arrow **IPC** reader, not Parquet. QuantLab never
   imports pyarrow directly — it is only pandas' `engine="pyarrow"` for Parquet.
   `git grep` across `src/` and `scripts/` finds no `import pyarrow`, no
   `pyarrow.ipc`, no `RecordBatchFileReader`, no `read_feather` / `to_feather`,
   and no `pre_buffer` usage.

Either ground alone is sufficient. Exposure was nil, and the bump is hygiene
rather than remediation.

## Decision

Bump to `pyarrow~=23.0`. Dataset files were **not** re-ingested or rewritten.

Verification, in order, against Parquet written by pyarrow 17.0.0:

| step | result |
| --- | --- |
| `quantlab data validate` (strict, no policy) | `dataset is consistent`, 47,108 bars |
| `dataset_id` before / after | `aec45c9df07a9a94` / `aec45c9df07a9a94` |
| sha256 of all six Parquet files | unchanged — reading does not rewrite |
| `make check`, incl. golden baselines | OK, **no byte-compare failure** |
| `quantlab report reproduce b8574b7935e74c58` | `reproduces exactly` |
| `pip-audit` | `No known vulnerabilities found` |

The golden baselines did not fail, so no baseline was regenerated and no
encoding-equivalence argument was needed. Run `b8574b7935e74c58` was recorded
under pyarrow 17.0.0 specifically so that INV-7 could be tested *across* the
bump rather than merely after it.

## Consequences

* INV-7 (a stored run reproduces exactly) holds across the pyarrow major bump,
  demonstrated on a run written by the previous version.
* Old Parquet stays readable; nothing on disk was rewritten, so the manifest
  hashes and `dataset_id` are untouched and prior runs remain reproducible.
* No metric, threshold, split boundary or cost default changed.
  `engine_version` is untouched.
