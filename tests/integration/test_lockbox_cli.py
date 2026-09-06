"""``quantlab-lockbox evaluate`` end to end (spec section 14.6, task T32). 🔒

Section 22 names four acceptance criteria for T32, and three of them are about
*ordering and consequences* rather than about arithmetic, so they need a real
store to be observable:

* the access row is written **before** the run;
* a second evaluation of a family is refused;
* a failure closes the family;
* the research profile raises on a test range (that one is in
  ``tests/unit/test_lockbox.py``, where it is a property of the container).

The first is the one worth being careful about. "Written before" cannot be tested
by observing a successful run — both orderings leave the same row behind. It is
tested by making the run *fail* and asserting the row is there anyway: if the
access were recorded afterwards, a crash would be a free look, and "crash, read
the traceback, try again" is a search over the test partition.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from quantlab.adapters.store.sqlite import (
    SqliteExperimentStore,
    create_db_engine,
    make_session_factory,
)
from quantlab.cli import app
from quantlab.cli.lockbox import app as lockbox_app
from quantlab.cli.lockbox import evaluate_lockbox
from quantlab.core.config import load_config
from quantlab.core.errors import ConfigError, LockboxViolation, StoreError
from quantlab.core.hashing import short_id

pytestmark = pytest.mark.slow

RUN_ID = re.compile(r"run ([0-9a-f]{16})")
REASON = "confirming the momentum family before paper trading"


def _store(project: Path) -> SqliteExperimentStore:
    engine = create_db_engine(project / "quantlab.db", create_parents=False)
    return SqliteExperimentStore(make_session_factory(engine))


def _config(project: Path) -> Any:
    return load_config([project / "configs" / "default.yaml"], [])


def _registered(cli: CliRunner, project: Path, strategy_file: Path) -> tuple[str, str]:
    """Run the strategy on train, returning ``(strategy_id, run_id)``.

    Going through the CLI rather than writing rows directly: ``--params`` names a
    run whose parameters the platform already measured, and a hand-written row
    would not have the ``params.json`` artifact the lockbox reads.
    """
    result = cli.invoke(
        app, ["backtest", "run", "--strategy", str(strategy_file), "--segment", "train"]
    )
    assert result.exit_code == 0, result.output
    match = RUN_ID.search(result.output)
    assert match, result.output
    run_id = match.group(1)
    row = _store(project).find_run(run_id)
    assert row is not None
    return str(row.strategy_id), run_id


def _policy(project: Path) -> Any:
    """The research split policy, stamped with the dataset it is measured on.

    ``split_policy.dataset_id`` is a foreign key, so a policy registered without
    one cannot be stored; ``cli/validate.py`` stamps it the same way.
    """
    from dataclasses import replace

    from quantlab.container import build_container

    container = build_container(_config(project), profile="research")
    assert container.split_policy is not None
    assert container.market_data is not None
    dataset_id = container.market_data.dataset_id(
        container.split_policy.symbol, container.split_policy.timeframe
    )
    return replace(container.split_policy, dataset_id=dataset_id)


def _candidate(project: Path, strategy_id: str) -> None:
    """Record the CANDIDATE verdict that authorises a lockbox access."""
    store = _store(project)
    version = store.get_strategy_version(strategy_id)
    assert version is not None
    policy = store.get_or_create_split(_policy(project))
    store.record_verdict(
        verdict_id=short_id(f"candidate|{strategy_id}"),
        strategy_id=strategy_id,
        split_id=policy.split_id,
        params_json="{}",
        verdict="CANDIDATE",
        overfit_score=10.0,
        hard_gates_json="{}",
        soft_checks_json="{}",
        thresholds_json="{}",
        n_trials_accounted=1,
    )


@pytest.fixture
def eligible(cli: CliRunner, project: Path, strategy_file: Path) -> tuple[str, str]:
    """A strategy that has been measured on train and validated as a CANDIDATE."""
    strategy_id, run_id = _registered(cli, project, strategy_file)
    _candidate(project, strategy_id)
    return strategy_id, run_id


# ---------------------------------------------------------------------------
# the ordering that makes the budget real
# ---------------------------------------------------------------------------
def test_the_access_is_recorded_before_the_run(
    project: Path, eligible: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Section 22: "access row written before run".

    The run is made to fail after eligibility has passed. A row written
    afterwards would not exist; one written before does, and the family has spent
    its look on a crash — which is the correct and deliberately unforgiving
    behaviour, because the alternative is a free retry loop over the test
    partition.
    """
    strategy_id, run_id = eligible

    from quantlab.strategies_io import evaluators

    def explode(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("the run failed after the access was recorded")

    monkeypatch.setattr(evaluators.SandboxEvaluator, "evaluate", explode)

    with pytest.raises(RuntimeError, match="after the access was recorded"):
        evaluate_lockbox(
            _config(project),
            strategy_id=strategy_id,
            run_id=run_id,
            reason=REASON,
            os_user="tester",
        )

    accesses = _store(project).lockbox_accesses()
    assert len(accesses) == 1
    assert accesses[0].strategy_id == strategy_id
    assert accesses[0].reason == REASON


def test_a_refused_access_records_nothing(project: Path, eligible: tuple[str, str]) -> None:
    """The mirror of the test above, and what keeps it honest: a refusal must not
    charge the family. A rule that recorded first and checked second would spend
    the budget on requests it never granted."""
    strategy_id, run_id = eligible
    with pytest.raises(LockboxViolation):
        evaluate_lockbox(
            _config(project),
            strategy_id=strategy_id,
            run_id=run_id,
            reason="too short",
            os_user="tester",
        )
    assert _store(project).lockbox_accesses() == []


# ---------------------------------------------------------------------------
# the budget, against a real store
# ---------------------------------------------------------------------------
def test_a_second_evaluation_of_a_family_is_refused(
    project: Path, eligible: tuple[str, str]
) -> None:
    """Section 22: "second evaluation of a family refused"."""
    strategy_id, run_id = eligible
    first = evaluate_lockbox(
        _config(project),
        strategy_id=strategy_id,
        run_id=run_id,
        reason=REASON,
        os_user="tester",
        baselines=False,
    )
    assert first.verdict in {"LOCKBOX_PASS", "LOCKBOX_FAIL"}

    with pytest.raises(LockboxViolation):
        evaluate_lockbox(
            _config(project),
            strategy_id=strategy_id,
            run_id=run_id,
            reason=REASON,
            os_user="tester",
            baselines=False,
        )
    assert len(_store(project).lockbox_accesses()) == 1


def test_an_evaluation_writes_a_lockbox_verdict(project: Path, eligible: tuple[str, str]) -> None:
    strategy_id, run_id = eligible
    decision = evaluate_lockbox(
        _config(project),
        strategy_id=strategy_id,
        run_id=run_id,
        reason=REASON,
        os_user="tester",
        baselines=False,
    )
    verdicts = [row.verdict for row in _store(project).verdicts_for(strategy_id)]
    assert verdicts[-1] == decision.verdict
    assert verdicts[-1].startswith("LOCKBOX_")


def test_a_failure_closes_the_family_and_locks_it_out_for_good(
    project: Path, eligible: tuple[str, str]
) -> None:
    """Section 22: "family closed on FAIL", and what closure then means.

    The fast ``sma_cross`` fixture does not clear the gates on this synthetic
    series — it fails all four on its own merits rather than being forced to — so
    the assertion is unconditional. A version of this test that branched on the
    verdict would stop testing closure the moment the fixture changed, and say
    nothing about it.

    The second half is the part that matters: a closed family is refused *before*
    the budget is even consulted, so no amount of further optimisation buys
    another look. That is what stops the lockbox from being a retry loop.
    """
    strategy_id, run_id = eligible
    decision = evaluate_lockbox(
        _config(project),
        strategy_id=strategy_id,
        run_id=run_id,
        reason=REASON,
        os_user="tester",
        baselines=False,
    )
    assert decision.verdict == "LOCKBOX_FAIL"
    assert decision.family_closed

    store = _store(project)
    version = store.get_strategy_version(strategy_id)
    assert version is not None
    family = store.find_family(str(version.family_id))
    assert family is not None
    assert family.status == "closed"

    with pytest.raises(LockboxViolation, match="closed"):
        evaluate_lockbox(
            _config(project),
            strategy_id=strategy_id,
            run_id=run_id,
            reason=REASON,
            os_user="tester",
            baselines=False,
        )


# ---------------------------------------------------------------------------
# eligibility, against a real store
# ---------------------------------------------------------------------------
def test_a_strategy_without_a_verdict_cannot_open_the_lockbox(
    cli: CliRunner, project: Path, strategy_file: Path
) -> None:
    strategy_id, run_id = _registered(cli, project, strategy_file)
    with pytest.raises(LockboxViolation, match="validate the strategy first"):
        evaluate_lockbox(
            _config(project),
            strategy_id=strategy_id,
            run_id=run_id,
            reason=REASON,
            os_user="tester",
        )
    assert _store(project).lockbox_accesses() == []


def test_an_unknown_run_is_refused_rather_than_run_with_empty_parameters(
    project: Path, eligible: tuple[str, str]
) -> None:
    """``--params`` names a run whose parameters were already measured. Falling
    back to ``{}`` would quietly evaluate a *different* strategy configuration on
    the one segment that cannot be re-measured."""
    strategy_id, _ = eligible
    with pytest.raises(StoreError, match="no recorded parameters"):
        evaluate_lockbox(
            _config(project),
            strategy_id=strategy_id,
            run_id="0" * 16,
            reason=REASON,
            os_user="tester",
        )
    assert _store(project).lockbox_accesses() == []


# ---------------------------------------------------------------------------
# the window, frozen
# ---------------------------------------------------------------------------
def test_the_test_window_is_frozen_by_the_first_access(
    project: Path, eligible: tuple[str, str]
) -> None:
    """An open-ended test segment could be waited out: a family that failed in
    March would return in June to a strictly longer period. The end is pinned
    once, to the last bar the dataset holds."""
    strategy_id, run_id = eligible
    store = _store(project)
    policy = _policy(project)
    assert policy.test_end_ts is None, "the fixture policy is open-ended by design"

    evaluate_lockbox(
        _config(project),
        strategy_id=strategy_id,
        run_id=run_id,
        reason=REASON,
        os_user="tester",
        baselines=False,
    )
    stored = store.get_or_create_split(policy)
    assert stored.test_end_ts is not None
    assert stored.test_end_ts >= policy.test_start_ts


def test_the_access_reason_and_user_are_recorded_verbatim(
    project: Path, eligible: tuple[str, str]
) -> None:
    """The record is the only account of why a family's one look was spent."""
    strategy_id, run_id = eligible
    evaluate_lockbox(
        _config(project),
        strategy_id=strategy_id,
        run_id=run_id,
        reason=f"  {REASON}  ",
        os_user="alice",
        baselines=False,
    )
    row = _store(project).lockbox_accesses()[0]
    assert row.reason == REASON
    assert row.os_user == "alice"


# ---------------------------------------------------------------------------
# the command itself
# ---------------------------------------------------------------------------
def test_the_command_reports_the_verdict_and_exits_non_zero_on_failure(
    project: Path, eligible: tuple[str, str]
) -> None:
    """``quantlab-lockbox`` is its own binary, so it is invoked as its own app.

    The exit code carries the verdict: a lockbox failure that exited zero would
    be indistinguishable from a pass to anything scripting this command, and this
    is the one command whose result cannot be re-derived by running it again.
    """
    strategy_id, run_id = eligible
    result = CliRunner().invoke(
        lockbox_app,
        [
            "evaluate",
            strategy_id,
            "--params",
            run_id,
            "--reason",
            REASON,
            "--config",
            str(project / "configs" / "default.yaml"),
            "--no-baselines",
        ],
    )
    assert result.exit_code == 1, result.output
    assert "LOCKBOX_FAIL" in result.output
    assert "G_MIN_TRADES" in result.output
    assert _store(project).lockbox_accesses() != []


def test_the_benchmark_gate_runs_buy_and_hold_on_the_test_segment(
    project: Path, eligible: tuple[str, str]
) -> None:
    """``G_BENCH`` needs a baseline measured on the same segment, and it goes
    through the loader and the sandbox like the strategy under test (INV-4). The
    other tests here disable it for speed, so this one keeps that path live."""
    strategy_id, run_id = eligible
    decision = evaluate_lockbox(
        _config(project),
        strategy_id=strategy_id,
        run_id=run_id,
        reason=REASON,
        os_user="tester",
        baselines=True,
    )
    bench = next(r for r in decision.gates.results if r.gate_id == "G_BENCH")
    assert "unmeasured" not in bench.reason.lower()


def test_an_unknown_strategy_is_refused_before_anything_is_recorded(
    project: Path, eligible: tuple[str, str]
) -> None:
    with pytest.raises(StoreError, match="no such strategy version"):
        evaluate_lockbox(
            _config(project),
            strategy_id="0" * 16,
            run_id=eligible[1],
            reason=REASON,
            os_user="tester",
        )
    assert _store(project).lockbox_accesses() == []


def test_an_already_frozen_window_is_not_re_frozen(
    project: Path, eligible: tuple[str, str]
) -> None:
    """The freeze happens once. A second access to the same split — a different
    family, later — must see the window the first one pinned, not today's last
    bar, or the test period would creep forward with every evaluation."""
    strategy_id, run_id = eligible
    store = _store(project)
    policy = _policy(project)
    stored = store.get_or_create_split(policy)
    pinned = store.freeze_test_end(stored.split_id, policy.test_start_ts + policy.bar_ms * 10)

    evaluate_lockbox(
        _config(project),
        strategy_id=strategy_id,
        run_id=run_id,
        reason=REASON,
        os_user="tester",
        baselines=False,
    )
    assert store.get_or_create_split(policy).test_end_ts == pinned.test_end_ts


def test_an_unmigrated_database_is_refused_rather_than_created_on_the_way_past(
    tmp_path: Path, project: Path, eligible: tuple[str, str]
) -> None:
    """Fail closed. Creating the schema here would make the *first* lockbox access
    of a fresh checkout the one that also has no access history to check against."""
    config = _config(project)
    fresh = config.project.model_copy(update={"db_path": tmp_path / "empty.db"})
    with pytest.raises(ConfigError, match="not migrated"):
        evaluate_lockbox(
            config.model_copy(update={"project": fresh}),
            strategy_id=eligible[0],
            run_id=eligible[1],
            reason=REASON,
            os_user="tester",
        )
