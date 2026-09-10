#!/usr/bin/env python
"""Nightly audit (master spec T44).

Four checks, run against the tree as it stands rather than against the tree as
it was when somebody last looked:

1. **Reproduce three random stored runs.** INV-7 says every stored run is
   reproducible from its stored inputs. That is checked in tests against runs
   the tests just made; this checks it against runs the *platform* made, on
   whatever data is really on disk, under today's dependency versions.
2. **Re-hash every dataset manifest.** Section 21.6. A Parquet file that
   changed a byte since ingestion invalidates every run that cites it, and
   nothing else in the system would notice.
3. **`pip-audit`.** A dependency with a published vulnerability.
4. **The coverage report's provenance.** The same gate `make check` runs, so a
   nightly that reported coverage from a stale `coverage.json` cannot claim a
   pass nobody earned.

Runs are chosen at **random**, not "the three most recent". Recent runs are the
ones a person has just been looking at and is most likely to have noticed
breaking; the risk is an old run quietly becoming unreproducible while nobody
is watching. The seed is printed so a failure can be repeated exactly.

Exit code is the number of checks that failed, capped at 125, so a CI job fails
loudly and a human can see how much is wrong at a glance.
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE = 3


@dataclass
class Check:
    name: str
    ok: bool
    detail: str

    def render(self) -> str:
        return f"[{'OK' if self.ok else 'FAIL'}] {self.name}\n     {self.detail}"


def _run(command: list[str], *, timeout: int = 3600) -> tuple[int, str]:
    result = subprocess.run(
        command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout, check=False
    )
    return result.returncode, (result.stdout + result.stderr).strip()


def reproduce_random_runs(sample: int, seed: int) -> Check:
    """Re-run ``sample`` stored runs and compare every metric (INV-7)."""
    try:
        from quantlab.adapters.store.sqlite import SqliteExperimentStore, make_session_factory
        from quantlab.container import create_db_engine
    except ImportError as exc:  # pragma: no cover - import failure is itself the finding
        return Check("reproduce", False, f"could not import the platform: {exc}")

    db = REPO_ROOT / "quantlab.db"
    if not db.is_file():
        return Check(
            "reproduce", True, "no database on this machine; nothing to reproduce (skipped)"
        )

    try:
        store = SqliteExperimentStore(make_session_factory(create_db_engine(str(db))))
        runs = [r for r in store.query_runs(status="ok") if getattr(r, "run_id", None)]
    except Exception as exc:
        # An unreadable or unmigrated database is a *finding*, not a crash. A
        # nightly that died here would page somebody with a traceback where the
        # useful sentence is "run `quantlab db upgrade`".
        return Check(
            "reproduce",
            False,
            f"{db} could not be read ({type(exc).__name__}: {exc}). "
            "If this is a fresh machine, run `quantlab db upgrade`.",
        )
    if not runs:
        return Check("reproduce", True, "the database holds no successful runs (skipped)")

    chosen = random.Random(seed).sample(runs, min(sample, len(runs)))
    failures: list[str] = []
    for run in chosen:
        code, output = _run(["uv", "run", "quantlab", "report", "reproduce", str(run.run_id)])
        if code != 0:
            failures.append(f"{run.run_id}: {output.splitlines()[-1] if output else 'no output'}")

    ids = ", ".join(str(r.run_id) for r in chosen)
    if failures:
        return Check(
            "reproduce", False, f"seed={seed} sampled {ids}\n     " + "\n     ".join(failures)
        )
    return Check("reproduce", True, f"seed={seed} reproduced {len(chosen)} run(s): {ids}")


def rehash_manifests() -> Check:
    """Every Parquet file must still hash to its manifest entry (section 21.6)."""
    from quantlab.core.hashing import file_sha256

    data_dir = REPO_ROOT / "data"
    manifests = sorted(data_dir.rglob("manifest.json")) if data_dir.is_dir() else []
    if not manifests:
        return Check("manifests", True, "no datasets on this machine (skipped)")

    import json

    problems: list[str] = []
    checked = 0
    for manifest_path in manifests:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            problems.append(f"{manifest_path}: unreadable ({exc})")
            continue
        root = manifest_path.parent
        for entry in payload.get("files", []):
            name, expected = entry.get("path"), entry.get("sha256")
            if not name or not expected:
                problems.append(f"{manifest_path}: an entry has no path or hash")
                continue
            target = root / name
            checked += 1
            if not target.is_file():
                problems.append(f"{target}: listed in the manifest and missing")
            elif file_sha256(target) != expected:
                problems.append(f"{target}: does not match its manifest hash")

    if problems:
        return Check("manifests", False, "\n     ".join(problems[:10]))
    return Check("manifests", True, f"{checked} file(s) across {len(manifests)} manifest(s) match")


def pip_audit() -> Check:
    code, output = _run(["uv", "run", "pip-audit"], timeout=900)
    if code == 0:
        return Check("pip-audit", True, "no known vulnerabilities")
    return Check("pip-audit", False, "\n     ".join(output.splitlines()[-12:]))


def coverage_provenance() -> Check:
    """The same freshness gate `make check` applies."""
    code, output = _run(["uv", "run", "python", "scripts/check_coverage.py"], timeout=300)
    if code == 0:
        return Check("coverage", True, "report is fresh and thresholds hold")
    return Check("coverage", False, "\n     ".join(output.splitlines()[-8:]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for run selection. Omit for a fresh random one, printed on every run.",
    )
    parser.add_argument("--skip-coverage", action="store_true")
    args = parser.parse_args(argv)

    seed = args.seed if args.seed is not None else random.randrange(2**31)

    checks = [
        reproduce_random_runs(args.sample, seed),
        rehash_manifests(),
        pip_audit(),
    ]
    if not args.skip_coverage:
        checks.append(coverage_provenance())

    print("QuantLab nightly audit\n")
    for check in checks:
        print(check.render())
    failed = [c for c in checks if not c.ok]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    if failed:
        print("failed: " + ", ".join(c.name for c in failed))
    return min(len(failed), 125)


if __name__ == "__main__":
    sys.exit(main())
