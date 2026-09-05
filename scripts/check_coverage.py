#!/usr/bin/env python3
"""Enforce the per-area coverage thresholds of the master spec (section 19).

    core/, sandbox/, research/redaction.py, cli/lockbox.py  >= 90 %
    overall                                                  >= 75 %

Reads ``coverage.json`` (produced by ``pytest --cov-report=json``).  Areas whose
files do not exist yet are reported as "not present yet" and do not fail the
gate; the moment a file appears it is held to its threshold.  Thresholds are
raised only by amending the spec, never lowered to make a build pass.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

COVERAGE_JSON = Path("coverage.json")

# (label, path prefixes relative to the repo root, required percent)
AREAS: list[tuple[str, tuple[str, ...], float]] = [
    ("core/", ("src/quantlab/core/",), 90.0),
    ("sandbox/", ("src/quantlab/sandbox/",), 90.0),
    ("research/redaction.py", ("src/quantlab/research/redaction.py",), 90.0),
    ("cli/lockbox.py", ("src/quantlab/cli/lockbox.py",), 90.0),
]
OVERALL_MIN = 75.0


def _norm(path: str) -> str:
    return path.replace("\\", "/")


def _percent(covered: int, total: int) -> float:
    return 100.0 if total == 0 else 100.0 * covered / total


def main() -> int:
    if not COVERAGE_JSON.exists():
        print(f"check_coverage: {COVERAGE_JSON} not found; run pytest --cov-report=json")
        return 1

    report = json.loads(COVERAGE_JSON.read_text(encoding="utf-8"))
    files: dict[str, dict] = report["files"]

    failures: list[str] = []
    lines: list[str] = []

    for label, prefixes, required in AREAS:
        covered = total = 0
        n_files = 0
        for path, entry in files.items():
            if not _norm(path).startswith(prefixes):
                continue
            n_files += 1
            summary = entry["summary"]
            covered += summary["covered_lines"] + summary.get("covered_branches", 0)
            total += summary["num_statements"] + summary.get("num_branches", 0)
        if n_files == 0:
            lines.append(f"  {label:<24} not present yet (threshold {required:.0f}%)")
            continue
        pct = _percent(covered, total)
        ok = pct + 1e-9 >= required
        lines.append(
            f"  {label:<24} {pct:6.2f}%  (>= {required:.0f}%, {n_files} file(s))"
            f"  {'OK' if ok else 'FAIL'}"
        )
        if not ok:
            failures.append(f"{label}: {pct:.2f}% < {required:.0f}%")

    totals = report["totals"]
    overall = _percent(
        totals["covered_lines"] + totals.get("covered_branches", 0),
        totals["num_statements"] + totals.get("num_branches", 0),
    )
    ok = overall + 1e-9 >= OVERALL_MIN
    lines.append(
        f"  {'overall':<24} {overall:6.2f}%  (>= {OVERALL_MIN:.0f}%)  {'OK' if ok else 'FAIL'}"
    )
    if not ok:
        failures.append(f"overall: {overall:.2f}% < {OVERALL_MIN:.0f}%")

    print("coverage gates:")
    print("\n".join(lines))

    if failures:
        print("\ncheck_coverage: FAILED")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\ncheck_coverage: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
