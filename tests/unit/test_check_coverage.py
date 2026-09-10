"""The coverage gate's own provenance checks (spec section 19).

``scripts/check_coverage.py`` is what stands between "the thresholds are met" and
"we believe the thresholds are met". Until these tests it had none of its own,
and it read whatever ``coverage.json`` happened to be on disk: a report from an
earlier session, or one written by hand, produced a clean green pass naming
modules that do not exist. A gate that can be satisfied by a stale file is not a
gate, and the failure is invisible precisely because it looks like success.

The thresholds themselves are exercised through ``make test`` on every run. What
is asserted here is the question that comes before them — *is this report
evidence of the tree that was just tested?*
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from check_coverage import (
    MAX_REPORT_AGE,
    check_provenance,
)

#: A file that really is in the tree, so the existence check has something honest
#: to accept. Chosen because it is load-bearing enough never to be deleted.
REAL_FILE = "src/quantlab/core/validation/deflated_sharpe.py"

NOW = datetime(2026, 9, 9, 12, 0, 0, tzinfo=UTC)


def report(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "meta": {
            "format": 3,
            "version": "7.16.0",
            "timestamp": (NOW - timedelta(seconds=30)).isoformat(),
            "branch_coverage": True,
        },
        "files": {REAL_FILE: {"summary": {}}},
        "totals": {},
    }
    meta = overrides.pop("meta", None)
    if meta is not None:
        base["meta"].update(meta)
    base.update(overrides)
    return base


def test_a_report_from_the_run_that_just_finished_is_accepted() -> None:
    """The honest case, stated first: nothing here may reject a real run."""
    assert check_provenance(report(), now=NOW) == []


def test_a_stale_report_is_refused() -> None:
    """The defect this gate was added for.

    A ``coverage.json`` left over from a previous session carries real, plausible
    percentages — about code that has since changed. Reading it produces a pass
    that no test run earned, and nothing in the output says so.
    """
    old = (NOW - MAX_REPORT_AGE - timedelta(minutes=1)).isoformat()
    problems = check_provenance(report(meta={"timestamp": old}), now=NOW)
    assert any("stale" in problem for problem in problems), problems


def test_a_report_just_inside_the_age_limit_is_still_accepted() -> None:
    """The boundary, so the limit is a decision rather than an accident."""
    edge = (NOW - MAX_REPORT_AGE + timedelta(seconds=1)).isoformat()
    assert check_provenance(report(meta={"timestamp": edge}), now=NOW) == []


def test_a_report_naming_files_that_do_not_exist_is_refused() -> None:
    """Age alone is not enough: a fabricated report can carry a current timestamp.

    What it cannot easily do is name only modules that are really in the tree, so
    the file set is checked against the working directory.
    """
    problems = check_provenance(
        report(files={REAL_FILE: {}, "src/quantlab/core/not_a_module.py": {}}), now=NOW
    )
    assert any("not in the tree" in problem for problem in problems), problems
    assert any("not_a_module.py" in problem for problem in problems), problems


def test_a_line_only_report_is_refused() -> None:
    """The gate sums statements *and* branches. Measured without branch coverage,
    the same thresholds quietly describe an easier quantity."""
    problems = check_provenance(report(meta={"branch_coverage": False}), now=NOW)
    assert any("branch coverage" in problem for problem in problems), problems


@pytest.mark.parametrize("stamp", ["", "not-a-date", None])
def test_a_report_whose_age_cannot_be_established_is_refused(stamp: str | None) -> None:
    """Fail closed. A report that will not say when it was written is of unknown
    provenance, and unknown provenance is what this check exists to stop."""
    meta = {"timestamp": stamp} if stamp is not None else {}
    payload = report(meta=meta)
    if stamp is None:
        payload["meta"].pop("timestamp")
    assert check_provenance(payload, now=NOW) != []


def test_every_problem_names_what_to_do_or_what_was_wrong() -> None:
    """A refusal a reader cannot act on gets worked around rather than fixed."""
    problems = check_provenance(
        report(meta={"timestamp": "2020-01-01T00:00:00", "branch_coverage": False}), now=NOW
    )
    assert len(problems) >= 2
    assert all(len(problem) > 30 for problem in problems), problems


def test_a_naive_timestamp_is_read_as_local_time() -> None:
    """The shape coverage.py actually writes.

    ``meta.timestamp`` carries no offset — it is ``datetime.now()`` on the machine
    that ran the suite. Reading it as UTC would shift every age by that machine's
    offset: west of Greenwich a report written seconds ago looks hours old and the
    gate blocks an honest run, east of it a stale report looks fresh and the gate
    waves through exactly what it exists to catch.
    """
    local_now = datetime.now().astimezone()
    fresh = report(meta={"timestamp": local_now.replace(tzinfo=None).isoformat()})
    assert "timestamp" not in str(check_provenance(fresh, now=local_now))
    assert check_provenance(fresh, now=local_now) == []

    old_naive = (local_now - MAX_REPORT_AGE - timedelta(minutes=5)).replace(tzinfo=None)
    problems = check_provenance(report(meta={"timestamp": old_naive.isoformat()}), now=local_now)
    assert any("stale" in problem for problem in problems), problems
