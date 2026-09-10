"""The drill record stays true (spec section 23, criterion 2). 🔒

``docs/INVARIANT_DRILLS.md`` is the evidence that every invariant has been
watched going red. Evidence rots: a test gets renamed, a module moves, an
invariant is added, and the document keeps asserting a drill against a target
that no longer exists — while reading exactly as convincing as it did the day
it was written.

So the document is checked against the tree. Nothing here re-runs the drills;
they patch source files in place and belong in a deliberate session, not in
``make check``. What is checked is weaker and still worth having: that the
record covers every invariant the spec defines, and that every file and test it
names is real.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DRILLS = ROOT / "docs" / "INVARIANT_DRILLS.md"
SPEC = ROOT / "CLAUDE_CODE_MASTER_SPEC.md"

TEXT = DRILLS.read_text(encoding="utf-8")

#: Every ``INV-n`` the spec's invariant table defines.
SPEC_INVARIANTS = sorted(
    set(re.findall(r"^\| (INV-\d+) \|", SPEC.read_text(encoding="utf-8"), re.MULTILINE)),
    key=lambda name: int(name.split("-")[1]),
)

#: Every file path the drills claim to have edited.
EDITED_FILES = sorted(set(re.findall(r"Edited `([^`]+)`", TEXT)))

#: Every pytest target the drills claim to have run.
TARGETS = sorted(
    {
        target
        for line in re.findall(r"\*\*Test run\.\*\* `([^`]+)`", TEXT)
        for target in line.split()
    }
)


def test_the_document_exists_and_was_parsed() -> None:
    """Guards every test below: an empty parse passes them all."""
    assert DRILLS.is_file()
    assert len(SPEC_INVARIANTS) >= 11, SPEC_INVARIANTS
    assert EDITED_FILES, "no drill entries were parsed"
    assert TARGETS, "no test targets were parsed"


@pytest.mark.parametrize("invariant", SPEC_INVARIANTS)
def test_every_invariant_the_spec_defines_has_a_drill(invariant: str) -> None:
    """Parameterised over the *spec*, not over the document.

    An invariant added to section 3 and never drilled fails here, which is the
    only direction that matters — the document cannot be made complete by
    leaving something out of it.
    """
    assert re.search(rf"^### {re.escape(invariant)} — ", TEXT, re.MULTILINE), (
        f"{invariant} has no entry in docs/INVARIANT_DRILLS.md"
    )


@pytest.mark.parametrize("invariant", SPEC_INVARIANTS)
def test_every_drill_is_recorded_as_confirmed(invariant: str) -> None:
    """A drill that came out INCONCLUSIVE is a guard nobody has seen work.

    This test failing is not a reason to edit the document — it is a reason to
    fix the guard, or to say plainly in the entry why the invariant cannot be
    drilled mechanically.
    """
    match = re.search(rf"^### {re.escape(invariant)} — (\w+)", TEXT, re.MULTILINE)
    assert match is not None, f"{invariant} has no entry"
    assert match.group(1) == "CONFIRMED", f"{invariant} is recorded as {match.group(1)}"


@pytest.mark.parametrize("path", EDITED_FILES)
def test_every_file_a_drill_edited_still_exists(path: str) -> None:
    """A drill against a module that has since moved proves nothing about the
    tree as it stands."""
    assert (ROOT / path).is_file(), f"{path} no longer exists; its drill is stale"


def test_every_test_a_drill_ran_still_exists() -> None:
    """Collected, not executed. A renamed test would leave the document claiming
    a guard was watched failing when nothing by that name is left to fail."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "--no-cov", *TARGETS],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        "a target named in docs/INVARIANT_DRILLS.md no longer collects:\n"
        + result.stdout[-1500:]
        + result.stderr[-500:]
    )


def test_the_document_says_what_the_evidence_does_not_cover() -> None:
    """A record of eleven passing drills reads as "the invariants hold". It
    supports something narrower — each guard fires against the one defect
    drilled — and the difference has to be on the page, not in the reader."""
    assert "does not show" in TEXT or "is not" in TEXT
    assert "finite set of drills" in TEXT


def test_the_document_records_when_it_was_run() -> None:
    """Undated evidence is unfalsifiable. A drill run against a tree from six
    months ago is a historical note, and a reader has to be able to tell."""
    assert re.search(r"\*\*Date of this run:\*\* \d{4}-\d{2}-\d{2}", TEXT)
    assert re.search(r"\*\*Tree:\*\* `[0-9a-f]{7,}`", TEXT)
