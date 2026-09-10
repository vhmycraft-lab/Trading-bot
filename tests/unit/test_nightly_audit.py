"""The nightly audit (spec T44).

T44's acceptance criterion has two halves and the second is the one that
matters: "job green; **an altered Parquet byte makes it fail**". A nightly that
only ever reports success is a cron job, not an audit — and the way it decays is
silent, because a green result and a check that stopped looking are the same
line of output.

So the manifest check is tested against a dataset that really is corrupt, and
the reproduce check is tested for the property that would hollow it out: it must
not report success when it has silently checked nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import nightly_audit
from nightly_audit import (
    Check,
    coverage_provenance,
    rehash_manifests,
    reproduce_random_runs,
)


def a_dataset(root: Path) -> Path:
    """A dataset directory with one Parquet file and an honest manifest."""
    from quantlab.core.hashing import file_sha256

    folder = root / "data" / "binance" / "BTCUSDT" / "1h"
    folder.mkdir(parents=True)
    bars = folder / "bars.parquet"
    bars.write_bytes(b"PAR1" + b"\x00\x01\x02\x03" * 64)
    (folder / "manifest.json").write_text(
        json.dumps({"files": [{"path": "bars.parquet", "sha256": file_sha256(bars)}]}),
        encoding="utf-8",
    )
    return folder


# ---------------------------------------------------------------------------
# T44's acceptance criterion
# ---------------------------------------------------------------------------
def test_an_intact_dataset_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stated first: the audit must not cry wolf, or it gets muted."""
    a_dataset(tmp_path)
    monkeypatch.setattr(nightly_audit, "REPO_ROOT", tmp_path)
    result = rehash_manifests()
    assert result.ok, result.detail
    assert "1 file(s)" in result.detail


def test_an_altered_parquet_byte_makes_it_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T44, verbatim. One byte, in a file nothing else would re-read.

    A dataset that changed after ingestion invalidates every run that cites it,
    and no other part of the platform would notice until somebody tried to
    reproduce one and got a different number with no explanation.
    """
    folder = a_dataset(tmp_path)
    monkeypatch.setattr(nightly_audit, "REPO_ROOT", tmp_path)

    bars = folder / "bars.parquet"
    payload = bytearray(bars.read_bytes())
    payload[10] ^= 0x01  # exactly one bit, in one byte
    bars.write_bytes(bytes(payload))

    result = rehash_manifests()
    assert not result.ok
    assert "does not match its manifest hash" in result.detail


def test_a_missing_file_is_reported_too(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Deletion and corruption are the same failure to a run that cites the
    dataset, and only one of them changes a hash."""
    folder = a_dataset(tmp_path)
    monkeypatch.setattr(nightly_audit, "REPO_ROOT", tmp_path)
    (folder / "bars.parquet").unlink()
    result = rehash_manifests()
    assert not result.ok
    assert "missing" in result.detail


def test_an_unreadable_manifest_is_a_failure_not_a_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manifest that will not parse is of unknown provenance. Skipping it
    would let a corrupted dataset pass by corrupting its manifest too."""
    folder = a_dataset(tmp_path)
    monkeypatch.setattr(nightly_audit, "REPO_ROOT", tmp_path)
    (folder / "manifest.json").write_text("{not json", encoding="utf-8")
    result = rehash_manifests()
    assert not result.ok
    assert "unreadable" in result.detail


# ---------------------------------------------------------------------------
# the checks must not pass vacuously
# ---------------------------------------------------------------------------
def test_no_datasets_reads_as_skipped_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A machine with no data has nothing to audit, and that is an ordinary
    state — but the output has to say "skipped" rather than "OK", or a broken
    ingest looks like a clean audit."""
    monkeypatch.setattr(nightly_audit, "REPO_ROOT", tmp_path)
    result = rehash_manifests()
    assert result.ok
    assert "skipped" in result.detail


def test_no_runs_reads_as_skipped_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(nightly_audit, "REPO_ROOT", tmp_path)
    result = reproduce_random_runs(3, seed=1)
    assert result.ok
    assert "skipped" in result.detail


def test_every_skip_is_visible_in_the_rendered_line() -> None:
    """The detail is not decoration: it is how a reader tells a check that
    passed from a check that had nothing to look at."""
    rendered = Check("x", True, "nothing to reproduce (skipped)").render()
    assert "OK" in rendered
    assert "skipped" in rendered


# ---------------------------------------------------------------------------
# reproducibility of the audit itself
# ---------------------------------------------------------------------------
def test_an_unreadable_database_is_a_finding_not_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unmigrated database used to raise ``OperationalError`` out of the
    audit. A nightly that dies here pages somebody with a stack trace whose one
    useful sentence is "run `quantlab db upgrade`" — so it says that instead."""
    monkeypatch.setattr(nightly_audit, "REPO_ROOT", tmp_path)
    (tmp_path / "quantlab.db").write_bytes(b"")
    result = reproduce_random_runs(3, seed=20260910)
    assert not result.ok
    assert "db upgrade" in result.detail


def test_both_sampling_outcomes_report_the_seed() -> None:
    """Runs are sampled at random rather than "the three most recent": recent
    runs are the ones somebody has just been looking at, and the risk is an
    *old* run going quietly unreproducible. Random sampling is only usable if a
    failure can be replayed exactly, so the seed has to appear whether the
    sample passed or failed.

    Asserted over the source rather than by running it, because reaching either
    branch needs a migrated database with stored runs and real bars behind them
    — which this machine does not have, and which a unit test should not build.
    """
    import inspect

    source = inspect.getsource(reproduce_random_runs)
    body = source[source.index("chosen = random.Random") :]
    returns = [line for line in body.splitlines() if "return Check(" in line]
    assert len(returns) == 2, returns
    assert body.count("seed={seed}") == 2, "a sampling outcome does not report its seed"


def test_the_exit_code_counts_the_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CI job needs a non-zero exit; a human reading the log wants to know how
    much is wrong without counting lines."""
    monkeypatch.setattr(nightly_audit, "reproduce_random_runs", lambda s, d: Check("r", False, "x"))
    monkeypatch.setattr(nightly_audit, "rehash_manifests", lambda: Check("m", False, "y"))
    monkeypatch.setattr(nightly_audit, "pip_audit", lambda: Check("p", True, "z"))
    assert nightly_audit.main(["--skip-coverage"]) == 2


def test_a_clean_audit_exits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nightly_audit, "reproduce_random_runs", lambda s, d: Check("r", True, "x"))
    monkeypatch.setattr(nightly_audit, "rehash_manifests", lambda: Check("m", True, "y"))
    monkeypatch.setattr(nightly_audit, "pip_audit", lambda: Check("p", True, "z"))
    assert nightly_audit.main(["--skip-coverage"]) == 0


def test_the_coverage_check_is_the_same_gate_make_check_runs() -> None:
    """Not a second implementation of freshness. The nightly shells out to
    `scripts/check_coverage.py`, so a nightly reporting coverage from a stale
    report is impossible for the same reason `make check` cannot."""
    import inspect

    source = inspect.getsource(coverage_provenance)
    assert "scripts/check_coverage.py" in source
