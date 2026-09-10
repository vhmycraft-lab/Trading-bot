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
from datetime import UTC, datetime, timedelta
from pathlib import Path

COVERAGE_JSON = Path("coverage.json")

#: How old ``coverage.json`` may be before the gate refuses to read it.
#:
#: ``pytest --cov-report=json`` writes the report at the very end of the run this
#: script is invoked from, so under ``make test`` it is always seconds old. An
#: hour is loose enough that no honest invocation trips it and tight enough that a
#: file left over from an earlier session cannot stand in for one.
#:
#: Without this the gate reports on whatever ``coverage.json`` happens to be on
#: disk. A stale report passes silently and is indistinguishable from a real pass:
#: the failure being guarded against is not a wrong number but a *green tick
#: nobody earned*, which is the one failure a coverage gate cannot survive.
MAX_REPORT_AGE = timedelta(hours=1)

#: Where the measured code lives. Every file the report names must exist here — a
#: report describing modules that are not in the tree was not produced by running
#: this suite against this working tree.
SOURCE_ROOT = Path("src/quantlab")

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


def check_provenance(report: dict, *, now: datetime | None = None) -> list[str]:
    """Reasons this report cannot be read as evidence of the current tree.

    Thresholds answer "is the number high enough". This answers the question that
    has to come first — *is this a number at all?* Three ways a report can look
    perfectly healthy while measuring nothing that just ran:

    1. **Age.** A leftover from an earlier session. Its percentages are real, they
       are simply about code that has since changed.
    2. **Files that do not exist.** A report naming modules absent from the tree
       was produced somewhere else, or by hand.
    3. **Line-only coverage.** The gate's arithmetic sums statements *and*
       branches; against a report gathered without ``branch = true`` the same
       thresholds silently measure something easier.
    """
    problems: list[str] = []
    meta = report.get("meta", {})

    stamp = meta.get("timestamp")
    if not stamp:
        problems.append("the report carries no meta.timestamp, so its age cannot be established")
    else:
        try:
            written = datetime.fromisoformat(str(stamp))
        except ValueError:
            problems.append(f"meta.timestamp {stamp!r} is not a readable timestamp")
        else:
            # coverage.py stamps naive *local* time. Reading it as UTC would shift
            # the age by the machine's offset, which west of Greenwich makes a
            # fresh report look hours old and east of it makes a stale one look
            # fresh. `astimezone()` on a naive value attaches the local zone,
            # which is what the writer meant.
            if written.tzinfo is None:
                written = written.astimezone()
            age = (now or datetime.now(UTC)) - written
            if age > MAX_REPORT_AGE:
                problems.append(
                    f"the report is {age} old (written {written.isoformat(timespec='seconds')}); "
                    f"it is stale, not evidence of this tree. Re-run `make test`"
                )

    if not meta.get("branch_coverage", False):
        problems.append(
            "the report was gathered without branch coverage, but the gate's "
            "thresholds are computed over statements and branches together"
        )

    missing = sorted(path for path in report.get("files", {}) if not Path(_norm(path)).is_file())
    if missing:
        shown = ", ".join(missing[:5]) + (
            f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        )
        problems.append(f"the report measures files that are not in the tree: {shown}")

    if not SOURCE_ROOT.is_dir():
        problems.append(f"{SOURCE_ROOT} does not exist; this is not the repository root")

    return problems


def main() -> int:
    if not COVERAGE_JSON.exists():
        print(f"check_coverage: {COVERAGE_JSON} not found; run pytest --cov-report=json")
        return 1

    report = json.loads(COVERAGE_JSON.read_text(encoding="utf-8"))

    # Before any threshold: is this report about the code that just ran? A gate
    # that reads a stale file reports a pass nobody earned, which is worse than
    # no gate at all — it is a green tick with a bug behind it.
    stale = check_provenance(report)
    if stale:
        print("check_coverage: FAILED -- the coverage report is not usable as evidence")
        for problem in stale:
            print(f"  - {problem}")
        return 1

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
