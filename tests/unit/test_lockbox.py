"""The lockbox: eligibility, budget, verdict and isolation (spec section 14.6, INV-5). 🔒

INV-5 says the test partition "is never loaded by any code path other than
``quantlab lockbox``", and this file is what enforces it. Three layers, tested
separately because each fails differently:

* the **guard** — a research container physically refuses a test-partition range
* the **rules** — who may open the lockbox, how often, and on what evidence
* the **verdict** — what a failure does to the family, for ever

The rules are pure functions over stored rows, so they are tested against fakes
rather than a database: the question "is this family allowed a second look?" has
nothing to do with SQLite, and a test that needed a migration to ask it would be
testing the wrong thing. The end-to-end ordering — *access recorded before the
first test bar is read* — is in ``tests/integration/test_lockbox_cli.py``, where
there is a real store to observe it in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from tests.unit.test_gates import passing

from quantlab.container import PROFILES, build_container
from quantlab.core.config import AppConfig, LockboxSettings, ValidationSettings
from quantlab.core.errors import LockboxViolation
from quantlab.core.metrics import MetricSet
from quantlab.core.validation.gates import GATE_IDS
from quantlab.core.validation.lockbox import (
    CLOSED_FAMILY_STATUS,
    LOCKBOX_GATES,
    MIN_REASON_CHARS,
    decide,
    lockbox_gates,
    require_budget,
    require_candidate,
    require_reason,
)

DAY_MS = 24 * 60 * 60 * 1000
NOW = 1_700_000_000_000


@dataclass(frozen=True)
class FakeVerdict:
    verdict: str


@dataclass(frozen=True)
class FakeAccess:
    family_id: str
    created_at: int


# ---------------------------------------------------------------------------
# who may open it: the CANDIDATE requirement
# ---------------------------------------------------------------------------
def test_a_candidate_verdict_opens_the_lockbox() -> None:
    latest = FakeVerdict("CANDIDATE")
    assert require_candidate([FakeVerdict("WEAK"), latest]) is latest


def test_a_strategy_that_was_never_validated_is_refused() -> None:
    """The lockbox confirms an edge already argued for; it is not where one is
    looked for. Admitting an unvalidated strategy would make the test partition
    part of the search."""
    with pytest.raises(LockboxViolation, match="validate the strategy first"):
        require_candidate([])


@pytest.mark.parametrize("verdict", ["REJECT", "WEAK", "LOCKBOX_FAIL"])
def test_anything_but_candidate_is_refused(verdict: str) -> None:
    with pytest.raises(LockboxViolation):
        require_candidate([FakeVerdict(verdict)])


def test_a_superseded_candidate_is_not_authorisation() -> None:
    """The *latest* verdict decides. If a stale CANDIDATE could open the lockbox,
    re-validating until one came out favourably would be a way in — which is the
    search over the validation segment that section 14.1's budget exists to stop,
    laundered into a search over the test segment."""
    with pytest.raises(LockboxViolation, match="superseded"):
        require_candidate([FakeVerdict("CANDIDATE"), FakeVerdict("REJECT")])


# ---------------------------------------------------------------------------
# the written reason
# ---------------------------------------------------------------------------
def test_a_reason_is_required_and_is_not_whitespace() -> None:
    with pytest.raises(LockboxViolation, match="written reason"):
        require_reason(" " * 40)


def test_a_reason_at_the_threshold_is_accepted() -> None:
    text = "x" * MIN_REASON_CHARS
    assert require_reason(f"  {text}  ") == text


def test_a_reason_one_character_short_is_refused() -> None:
    """Guards the boundary in the direction that matters: a rule that rounded up
    would let the shortest reasons through."""
    with pytest.raises(LockboxViolation):
        require_reason("x" * (MIN_REASON_CHARS - 1))


# ---------------------------------------------------------------------------
# the budget
# ---------------------------------------------------------------------------
SETTINGS = LockboxSettings(max_per_family=1, max_per_month=3)


def test_a_family_with_no_history_may_look() -> None:
    require_budget([], SETTINGS, family_id="fam", now_ms=NOW)


def test_a_second_evaluation_of_a_family_is_refused() -> None:
    """Section 22's acceptance criterion. A family is one *idea*: the second look
    would be at an idea the first one already informed."""
    accesses = [FakeAccess("fam", NOW - DAY_MS)]
    with pytest.raises(LockboxViolation, match="already used its lockbox access"):
        require_budget(accesses, SETTINGS, family_id="fam", now_ms=NOW)


def test_another_family_is_unaffected_by_the_first_ones_access() -> None:
    """Guards the test above against a per-family cap that is really a global one."""
    require_budget([FakeAccess("other", NOW - DAY_MS)], SETTINGS, family_id="fam", now_ms=NOW)


def test_the_monthly_cap_bounds_the_platform_across_families() -> None:
    """Twenty families looked at in a week is a search over the test partition
    however it is spelled."""
    accesses = [FakeAccess(f"fam{i}", NOW - i * DAY_MS) for i in range(3)]
    with pytest.raises(LockboxViolation, match="thirty days"):
        require_budget(accesses, SETTINGS, family_id="fresh", now_ms=NOW)


def test_the_month_is_a_rolling_thirty_days_not_a_calendar_one() -> None:
    """A calendar month could be doubled by looking on the 31st and again on the
    1st; accesses older than thirty days fall out of the window instead."""
    accesses = [FakeAccess(f"fam{i}", NOW - 31 * DAY_MS) for i in range(3)]
    require_budget(accesses, SETTINGS, family_id="fresh", now_ms=NOW)


def test_an_access_exactly_thirty_days_old_still_counts() -> None:
    """The boundary is closed on the side that refuses. An off-by-one here would
    hand the platform a free look every thirty days."""
    accesses = [FakeAccess(f"fam{i}", NOW - 30 * DAY_MS) for i in range(3)]
    with pytest.raises(LockboxViolation):
        require_budget(accesses, SETTINGS, family_id="fresh", now_ms=NOW)


# ---------------------------------------------------------------------------
# the gates
# ---------------------------------------------------------------------------
def test_only_the_four_gates_of_section_14_6_are_applied() -> None:
    """``G_LEAK``, ``G_SANITY`` and ``G_PERM`` are properties of the strategy that
    validation already established; re-running them would spend the test
    partition re-answering answered questions."""
    report = lockbox_gates(passing(), ValidationSettings())
    assert tuple(result.gate_id for result in report.results) == LOCKBOX_GATES
    assert set(LOCKBOX_GATES) < set(GATE_IDS)


def test_the_lockbox_gate_list_is_a_real_subset_and_not_a_typo() -> None:
    """Guards the test above: a name misspelled in ``LOCKBOX_GATES`` would silently
    apply *fewer* gates than section 14.6 requires."""
    assert set(LOCKBOX_GATES) <= set(GATE_IDS)
    assert len(LOCKBOX_GATES) == 4


def test_a_strategy_failing_only_a_skipped_gate_still_passes_the_lockbox() -> None:
    """The subset is deliberate, so it has to be observable: a probe failure fails
    ``G_LEAK`` in validation and is not re-litigated here."""
    report = lockbox_gates(passing(probe_passed=False), ValidationSettings())
    assert report.passed


def test_a_strategy_failing_a_lockbox_gate_fails() -> None:
    report = lockbox_gates(
        passing(metrics_val=MetricSet(n_trades=1.0, sortino=1.1, net_return=0.2)),
        ValidationSettings(),
    )
    assert not report.passed
    assert "G_MIN_TRADES" in report.failed


# ---------------------------------------------------------------------------
# the verdict, and what a failure costs
# ---------------------------------------------------------------------------
def test_a_clean_run_passes_and_leaves_the_family_open() -> None:
    decision = decide(lockbox_gates(passing(), ValidationSettings()))
    assert decision.verdict == "LOCKBOX_PASS"
    assert decision.passed
    assert not decision.family_closed


def test_a_failure_closes_the_family() -> None:
    """Section 22's acceptance criterion, and the harshest rule in the platform.
    The family has been measured on the only data nobody could optimise against
    and did not hold; leaving it open would invite exactly the loop the lockbox
    exists to prevent — optimise more, come back, look again."""
    decision = decide(
        lockbox_gates(passing(metrics_val=MetricSet(n_trades=1.0)), ValidationSettings())
    )
    assert decision.verdict == "LOCKBOX_FAIL"
    assert decision.family_closed
    assert CLOSED_FAMILY_STATUS == "closed"


# ---------------------------------------------------------------------------
# INV-5: the guard, structurally
# ---------------------------------------------------------------------------
def _config(tmp_path: Any, repo_root: Any) -> AppConfig:
    from quantlab.core.config import load_config

    return load_config([repo_root / "configs" / "default.yaml"], [])


def test_the_research_profile_refuses_a_test_partition_range(
    tmp_path: Any, repo_root: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Section 22's acceptance criterion: "research profile raises LockboxViolation
    on test range". The guard is attached in ``container.py`` and nowhere else, so
    this is a property of every research code path at once rather than of the one
    that happens to be under test."""
    monkeypatch.chdir(repo_root)
    config = _config(tmp_path, repo_root)
    container = build_container(config, profile="research")
    assert container.split_policy is not None
    assert not container.may_read_test_partition
    test_segment = container.split_policy.segment("test")
    assert container.market_data is not None
    with pytest.raises(LockboxViolation):
        container.market_data.load(
            symbol=container.split_policy.symbol,
            timeframe=container.split_policy.timeframe,
            start_ts=test_segment.start_ts,
            end_ts=test_segment.start_ts + container.split_policy.bar_ms,
        )


def test_the_lockbox_profile_is_the_only_one_that_may_read_the_test_partition(
    repo_root: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Enumerated over every profile rather than asserted about one, so a profile
    added later is either guarded or fails this test."""
    monkeypatch.chdir(repo_root)
    config = _config(None, repo_root)
    unguarded = [
        profile
        for profile in PROFILES
        if build_container(config, profile=profile).may_read_test_partition  # type: ignore[arg-type]
    ]
    assert unguarded == ["lockbox"]
