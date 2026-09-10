"""What a model may be shown (spec section 14.5; INV-6, INV-11). 🔒

The invariant table names this file for two invariants, and it did not exist
until now — so the first thing asserted here is that the scan is capable of
failing at all. A leak check that cannot be made to fire is indistinguishable
from one that always passes, and this is exactly the class of test where that
goes unnoticed: prompts are usually clean, so a broken scanner is green every
day until the one day it matters.

Each section therefore has the same shape — a clean case, then a case that must
be refused, then the boundary between them.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from quantlab.core.errors import LockboxViolation
from quantlab.research.redaction import (
    ENVIRONMENT_ALLOWLIST,
    MIN_SECRET_LENGTH,
    REDACTED_FIELDS,
    RedactedReport,
    assert_prompt_is_clean,
    build_generation_report,
    build_redacted_report,
    scan_for_leaks,
)

#: Real epoch-millisecond boundaries, so the ISO dates the scan derives are the
#: ones a real split would produce rather than ones invented to match.
TRAIN_END = int(datetime(2023, 12, 31, tzinfo=UTC).timestamp() * 1000)
VAL_START = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000)
VAL_END = int(datetime(2024, 6, 30, tzinfo=UTC).timestamp() * 1000)
TEST_START = int(datetime(2024, 7, 1, tzinfo=UTC).timestamp() * 1000)
TEST_END = int(datetime(2024, 12, 31, tzinfo=UTC).timestamp() * 1000)


class FakeSplit:
    train_end_ts = TRAIN_END
    val_start_ts = VAL_START
    val_end_ts = VAL_END
    test_start_ts = TEST_START
    test_end_ts = TEST_END


class FakeSecrets:
    def __init__(self, **values: str) -> None:
        self._values = values

    def get(self, name: str) -> str:
        return self._values[name]

    def try_get(self, name: str) -> str | None:
        return self._values.get(name)


#: Credential-shaped fixtures, assembled rather than written out. The
#: repository-wide scanner of INV-2 (``tests/unit/test_no_secrets.py``) refuses
#: a credential literal anywhere in the tree, and it is right to — a test file
#: is as greppable as any other. The same idiom is used in
#: ``test_architecture.py`` for the forbidden order-API names.
FAKE_KEY = "sk-" + "live-" + "abcdefghijklmnop"
FAKE_VALUE = "abcdefghij" + "klmnopqrst"
FAKE_AWS = "AKIA" + "IOSFODNN7EXAMPLE"
FAKE_SESSION = "verylongsession" + "tokenvalue"

CLEAN = """
Propose a genome. The population currently explores momentum and mean reversion.
Training Sharpe of the best surviving candidate was 1.4 over 8000 bars.
"""


# ---------------------------------------------------------------------------
# the scanner can fail — asserted before anything relies on it passing
# ---------------------------------------------------------------------------
def test_an_honest_prompt_is_clean() -> None:
    """Stated first, because nothing below may reject a real prompt."""
    assert scan_for_leaks(CLEAN, split=FakeSplit(), environ={}) == []


def test_the_scanner_is_capable_of_refusing_something() -> None:
    """Guards every other test in this file.

    If ``scan_for_leaks`` returned ``[]`` unconditionally — a bad refactor, a
    swallowed exception, a regex that stopped compiling — every assertion of
    cleanliness above and below would still pass. This one would not.
    """
    assert scan_for_leaks("the test partition metrics were strong", environ={}) != []


# ---------------------------------------------------------------------------
# INV-6: nothing from the test partition
# ---------------------------------------------------------------------------
def test_a_test_period_date_is_refused() -> None:
    """The specific form section 14.5 asks for: "assert none of the test-period
    ISO dates appear". A date is how a leak survives paraphrase — a model told
    "results through 2024-12-31 were strong" has been told the test window."""
    text = f"{CLEAN}\nResults through {datetime(2024, 12, 31, tzinfo=UTC).date()} held up."
    problems = scan_for_leaks(text, split=FakeSplit(), environ={})
    assert any("test-period date" in p for p in problems), problems


def test_naming_the_test_partition_is_refused() -> None:
    problems = scan_for_leaks("the test_segment returned 2.1 Sharpe", environ={})
    assert any("INV-6" in p for p in problems), problems


def test_naming_the_lockbox_is_refused() -> None:
    """``lockbox`` is a segment name as much as ``test`` is, and a model that
    knows a lockbox run happened knows the family reached one."""
    problems = scan_for_leaks("the lockbox metrics are attached", environ={})
    assert any("INV-6" in p for p in problems), problems


def test_a_validation_date_is_allowed_when_no_run_is_active() -> None:
    """INV-6 and INV-11 are different invariants and this is where they differ.

    Outside an evolution run, validation numbers may reach a *person* through a
    report, and a report is not a prompt — but nothing here forbids a prompt
    that happens to mention a validation date either, because INV-11's stricter
    reading is scoped to an active run. Asserting the difference keeps a later
    "simplification" from collapsing the two into whichever is easier.
    """
    text = f"validation ran through {datetime(2024, 6, 30, tzinfo=UTC).date()}"
    assert scan_for_leaks(text, split=FakeSplit(), forbid_validation=False, environ={}) == []


# ---------------------------------------------------------------------------
# INV-11: nothing from validation, while a run is active
# ---------------------------------------------------------------------------
def test_a_validation_date_is_refused_while_a_run_is_active() -> None:
    text = f"validation ran through {datetime(2024, 6, 30, tzinfo=UTC).date()}"
    problems = scan_for_leaks(text, split=FakeSplit(), forbid_validation=True, environ={})
    assert any("INV-11" in p for p in problems), problems


@pytest.mark.parametrize(
    "text",
    [
        "the val_sharpe of the leader is 1.9",
        "wf_oos_return came out positive",
        "val segment metrics attached",
    ],
)
def test_a_validation_quantity_is_refused_while_a_run_is_active(text: str) -> None:
    """Section 12: the loop sees training and inner-fold numbers only. A
    validation Sharpe reaching a proposer is the model optimising against the
    partition the platform reserved to check it — laundered through a genome."""
    assert scan_for_leaks(text, forbid_validation=True, environ={}) != []


def test_a_training_quantity_is_not_refused_while_a_run_is_active() -> None:
    """The other half of the same rule, and the one that makes the guard usable:
    a generation report is *supposed* to carry training numbers."""
    assert (
        scan_for_leaks("train_sharpe 1.4, inner-fold OOS 0.9", forbid_validation=True, environ={})
        == []
    )


# ---------------------------------------------------------------------------
# amendment 1.3(f): secrets and the environment
# ---------------------------------------------------------------------------
def test_a_configured_secret_value_is_refused() -> None:
    """The direction the original scan did not cover.

    Prompts are written to ``artifacts/llm/<id>/prompt.json`` unencrypted and
    kept. A report assembled by formatting an exception or a config dump can
    carry the API key into that file, where it outlives the run.
    """
    secrets = FakeSecrets(GLM_API_KEY=FAKE_KEY)
    text = f"the call failed with key {FAKE_KEY}"
    problems = scan_for_leaks(text, secrets=secrets, environ={})
    assert problems, problems


def test_a_credential_shape_is_refused_even_when_nothing_is_configured() -> None:
    """A key belonging to a *different* service is still a key. Matching only
    values this platform knows about would miss every one it does not."""
    assert scan_for_leaks(f"{FAKE_AWS} is in the log", environ={}) != []


def test_a_secret_shaped_assignment_is_refused() -> None:
    problems = scan_for_leaks(f'api_key = "{FAKE_VALUE}"', environ={})
    assert any("secret-shaped name" in p for p in problems), problems


def test_a_short_value_under_a_secret_name_is_not_refused() -> None:
    """The boundary, in the direction that keeps the guard usable. Placeholders
    are short; a scan that fired on ``token: none`` would be switched off."""
    assert scan_for_leaks("api_key = none", environ={}) == []


def test_the_boundary_length_is_the_one_that_refuses() -> None:
    value = "x" * MIN_SECRET_LENGTH
    assert scan_for_leaks(f'password = "{value}"', environ={}) != []
    shorter = "x" * (MIN_SECRET_LENGTH - 1)
    assert scan_for_leaks(f'password = "{shorter}"', environ={}) == []


def test_an_environment_value_is_refused() -> None:
    """A prompt built from a formatted traceback or a subprocess dump carries
    whatever was in ``os.environ`` at the time, which on a developer's machine
    is usually more than they would choose to send."""
    environ = {"AWS_SESSION_TOKEN": FAKE_SESSION}
    problems = scan_for_leaks(f"context: {FAKE_SESSION}", environ=environ)
    assert any("AWS_SESSION_TOKEN" in p for p in problems), problems


def test_an_allowlisted_environment_value_is_not_refused() -> None:
    """``PATH`` appears in half the tracebacks ever printed. Refusing on it
    would make the guard fire constantly and get disabled."""
    environ = {"PATH": "/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin"}
    assert "PATH" in ENVIRONMENT_ALLOWLIST
    assert scan_for_leaks(f"PATH is {environ['PATH']}", environ=environ) == []


def test_a_broken_secrets_backend_does_not_open_the_gate() -> None:
    """Fail closed. A keyring that raises must not turn the secret scan off."""

    class Exploding:
        def try_get(self, name: str) -> str | None:
            raise RuntimeError("keychain is locked")

    assert scan_for_leaks("the test partition metrics", secrets=Exploding(), environ={}) != []


# ---------------------------------------------------------------------------
# the raising form
# ---------------------------------------------------------------------------
def test_a_clean_prompt_passes_through_unchanged() -> None:
    assert assert_prompt_is_clean(CLEAN, split=FakeSplit(), environ={}) == CLEAN


def test_a_leaking_prompt_raises_a_lockbox_violation() -> None:
    """Not an ``LLMError``. Section 14.6 says a ``LockboxViolation`` MUST NOT be
    caught outside the CLI top level, so an adapter cannot catch this, log it as
    a transport hiccup and retry — which is how a leak becomes a warning."""
    with pytest.raises(LockboxViolation, match="may not be sent"):
        assert_prompt_is_clean("the test_partition results", environ={})


def test_every_problem_is_reported_not_just_the_first() -> None:
    """A caller that fixes one leak and resends should not discover the second
    one round trip later — each of those costs money and writes an artifact."""
    text = f"test_segment sharpe on {datetime(2024, 12, 31, tzinfo=UTC).date()}"
    assert len(scan_for_leaks(text, split=FakeSplit(), environ={})) >= 2


# ---------------------------------------------------------------------------
# the report is a whitelist
# ---------------------------------------------------------------------------
def test_the_report_refuses_a_field_nobody_whitelisted() -> None:
    """Section 14.5 lists the fields "exclusively". ``extra="forbid"`` is what
    makes a field added to the store later absent from a prompt until somebody
    puts it here deliberately — the safe default for a boundary."""
    with pytest.raises(ValidationError, match="extra"):
        RedactedReport(strategy_id="s0", test_metrics={"sharpe": 3.0})  # type: ignore[call-arg]


def test_the_whitelist_and_the_model_agree() -> None:
    """Guards the test above: a name in :data:`REDACTED_FIELDS` that no longer
    exists on the model would document a protection nothing enforces."""
    assert set(REDACTED_FIELDS) == set(RedactedReport.model_fields)


def test_a_serialised_report_names_no_forbidden_segment() -> None:
    """Section 14.5's own test, stated in its own terms: serialise and scan."""
    report = RedactedReport(strategy_id="s0", metrics_train={"sharpe": 1.4})
    assert scan_for_leaks(report.model_dump_json(), split=FakeSplit(), environ={}) == []


# ---------------------------------------------------------------------------
# building a report from a store
# ---------------------------------------------------------------------------
class FakeRun:
    def __init__(self, run_id: str, segment: str, status: str = "ok") -> None:
        self.run_id = run_id
        self.segment = segment
        self.status = status


class FakeStore:
    """The narrowest store the report builders actually touch.

    Written out rather than mocked so that the *absence* of certain calls is
    visible: there is no ``lockbox_access`` here and no ``split_policy``, and a
    builder that reached for either would fail with an ``AttributeError`` naming
    the thing it should not have wanted.
    """

    def __init__(self, runs: list[FakeRun], metrics: dict[str, dict[str, float]]) -> None:
        self.runs = runs
        self.metrics = metrics
        self.asked_for: list[str] = []

    def query_runs(self, **filters: Any) -> list[FakeRun]:
        return list(self.runs)

    def metrics_for(self, run_id: str) -> dict[str, float]:
        self.asked_for.append(run_id)
        return dict(self.metrics.get(run_id, {}))

    def verdicts_for(self, strategy_id: str) -> list[Any]:
        return [type("V", (), {"overfit_score": 12.0, "verdict": "CANDIDATE"})()]

    def lineage(self, strategy_id: str) -> list[Any]:
        return [
            type("S", (), {"strategy_id": "seed"})(),
            type("S", (), {"strategy_id": strategy_id})(),
        ]

    def get_strategy_version(self, strategy_id: str) -> Any:
        return type("V", (), {"family_id": "fam0", "strategy_id": strategy_id})()

    def find_family(self, family_id: str) -> Any:
        return type("F", (), {"validation_touches": 3})()

    def find_evolution_run(self, evolution_id: str) -> Any:
        return type("E", (), {"evolution_id": evolution_id, "status": "running"})()

    def candidates_for(self, evolution_id: str, gen_index: int | None = None) -> list[Any]:
        return [type("C", (), {"candidate_id": "c0", "strategy_id": "s0"})()]


ALL_SEGMENTS = [
    FakeRun("r_train", "train"),
    FakeRun("r_val", "val"),
    FakeRun("r_wf", "wf_oos:0"),
    FakeRun("r_test", "test"),
    FakeRun("r_lock", "lockbox"),
]
METRICS = {
    "r_train": {"sharpe": 1.4},
    "r_val": {"sharpe": 0.9},
    "r_wf": {"sharpe": 0.7},
    "r_test": {"sharpe": 3.3},
    "r_lock": {"sharpe": 3.1},
}


def test_the_report_never_reads_a_test_segment_run() -> None:
    """INV-6, structurally rather than by scanning the output: the test-segment
    run is in the store, and the builder does not ask for its metrics at all."""
    store = FakeStore(ALL_SEGMENTS, METRICS)
    report = build_redacted_report(store, "s0")
    assert "r_test" not in store.asked_for
    assert "r_lock" not in store.asked_for
    assert report.metrics_train == {"sharpe": 1.4}
    assert report.metrics_val == {"sharpe": 0.9}


def test_the_report_carries_the_verdict_and_the_touch_count() -> None:
    report = build_redacted_report(FakeStore(ALL_SEGMENTS, METRICS), "s0")
    assert report.verdict == "CANDIDATE"
    assert report.overfit_score == 12.0
    assert report.validation_touches == 3
    assert report.lineage == ("seed", "s0")


def test_a_failed_run_is_not_evidence() -> None:
    """A run that errored has no metrics worth reporting, and reporting the
    partial ones it does have would put a number nobody stands behind in front
    of a model."""
    runs = [FakeRun("r_train", "train", status="failed"), FakeRun("r_ok", "train")]
    store = FakeStore(runs, {"r_train": {"sharpe": 9.9}, "r_ok": {"sharpe": 1.0}})
    assert build_redacted_report(store, "s0").metrics_train == {"sharpe": 1.0}


def test_a_strategy_with_no_runs_yields_an_empty_report() -> None:
    """Not an error. A strategy that has been registered and not yet evaluated
    is an ordinary state, and a builder that raised on it would make the first
    proposal of every campaign a special case."""
    report = build_redacted_report(FakeStore([], {}), "s0")
    assert report.metrics_train == {}
    assert scan_for_leaks(report.model_dump_json(), environ={}) == []


# ---------------------------------------------------------------------------
# the generation report is stricter (INV-11)
# ---------------------------------------------------------------------------
def test_a_generation_report_excludes_validation_as_well_as_test() -> None:
    """The difference between INV-6 and INV-11, at the point it is enforced.
    A generation report carries training and inner-fold numbers, and the inner
    folds are cut from the training segment — so nothing in it has ever touched
    validation."""
    store = FakeStore(ALL_SEGMENTS, METRICS)
    report = build_generation_report(store, "evo0", 2)
    assert report.metrics_val == {}
    assert report.metrics_train == {"sharpe": 1.4}
    assert "r_val" not in store.asked_for
    assert "r_test" not in store.asked_for


def test_a_generation_report_passes_the_strict_scan() -> None:
    """The end-to-end statement of INV-11: what the loop may see, scanned under
    the reading that applies while a run is active."""
    report = build_generation_report(FakeStore(ALL_SEGMENTS, METRICS), "evo0", 2)
    problems = scan_for_leaks(
        report.model_dump_json(), split=FakeSplit(), forbid_validation=True, environ={}
    )
    assert problems == [], problems


def test_a_generation_report_for_a_run_that_does_not_exist_is_refused() -> None:
    """Building one from nothing would produce an empty, clean-looking report —
    and an empty report passes every leak check there is."""

    class NoRuns(FakeStore):
        def find_evolution_run(self, evolution_id: str) -> Any:
            return None

    with pytest.raises(LockboxViolation, match="no such evolution run"):
        build_generation_report(NoRuns([], {}), "missing", 0)


def test_a_candidate_with_no_runs_is_skipped_rather_than_reported_empty() -> None:
    """A candidate that has not been evaluated contributes nothing, and listing
    it with blank metrics would read as a measurement of zero."""

    class NoRunsForCandidate(FakeStore):
        def query_runs(self, **filters: Any) -> list[FakeRun]:
            return []

    report = build_generation_report(NoRunsForCandidate([], {}), "evo0", 0)
    assert report.lineage == ()
