"""``quantlab backtest`` end to end (master spec sections 11, 20; task T22).

Section 20 states the scenario: "`quantlab backtest` on fixture → run row,
artifacts, report". Everything below runs offline against generated bars, and the
strategy is executed in a sandbox child exactly as it would be in a real run —
mocking that away would leave the one path that matters untested.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quantlab.adapters.store.artifacts import FileArtifactStore
from quantlab.adapters.store.sqlite import (
    SqliteExperimentStore,
    create_db_engine,
    make_session_factory,
)
from quantlab.cli import app
from quantlab.core.errors import EXIT_CODES
from quantlab.experiments.runner import ARTIFACT_NAMES

pytestmark = pytest.mark.slow

RUN_ID = re.compile(r"run ([0-9a-f]{16})")


def _run_id(output: str) -> str:
    match = RUN_ID.search(output)
    assert match, f"no run id in output:\n{output}"
    return match.group(1)


def _store(project: Path) -> SqliteExperimentStore:
    engine = create_db_engine(project / "quantlab.db", create_parents=False)
    return SqliteExperimentStore(make_session_factory(engine))


def _backtest(cli: CliRunner, strategy_file: Path, *extra: str):  # type: ignore[no-untyped-def]
    return cli.invoke(
        app,
        ["backtest", "run", "--strategy", str(strategy_file), "--segment", "train", *extra],
    )


def test_a_backtest_records_a_run(cli: CliRunner, project: Path, strategy_file: Path) -> None:
    result = _backtest(cli, strategy_file)
    assert result.exit_code == 0, result.output

    run_id = _run_id(result.output)
    row = _store(project).find_run(run_id)
    assert row is not None
    assert row.status == "ok"
    assert row.segment == "train"
    assert row.engine_name == "simple_bar"


def test_the_report_names_the_run_and_whether_it_was_cached(
    cli: CliRunner, project: Path, strategy_file: Path
) -> None:
    result = _backtest(cli, strategy_file)
    assert "cache_hit" in result.output
    assert "false" in result.output
    assert "net_return" in result.output


def test_every_artifact_the_spec_names_is_written(
    cli: CliRunner, project: Path, strategy_file: Path
) -> None:
    """Section 11.3 lists ten files. A run missing one is a run that cannot be
    reproduced or reported on later."""
    result = _backtest(cli, strategy_file)
    run_id = _run_id(result.output)
    artifacts = FileArtifactStore(project / "artifacts")
    assert set(ARTIFACT_NAMES) <= set(artifacts.listdir(run_id))


def test_metrics_and_trades_reach_the_database(
    cli: CliRunner, project: Path, strategy_file: Path
) -> None:
    result = _backtest(cli, strategy_file)
    run_id = _run_id(result.output)
    store = _store(project)
    assert store.metrics_for(run_id)
    assert "net_return" in store.metrics_for(run_id)


def test_the_same_backtest_twice_hits_the_cache(
    cli: CliRunner, project: Path, strategy_file: Path
) -> None:
    """Section 11.2's acceptance criterion, through the command a person types."""
    first = _backtest(cli, strategy_file)
    second = _backtest(cli, strategy_file)
    assert first.exit_code == second.exit_code == 0

    assert _run_id(first.output) == _run_id(second.output)
    assert "true" in second.output.split("cache_hit")[1][:40]
    assert len(_store(project).query_runs()) == 1


def test_force_re_executes_without_creating_a_second_run(
    cli: CliRunner, project: Path, strategy_file: Path
) -> None:
    """Section 6 permits no rewrite of a successful run, so ``--force`` verifies
    rather than overwrites: it executes again and reports fresh numbers, and the
    stored record is left exactly as it was."""
    first = _backtest(cli, strategy_file)
    forced = _backtest(cli, strategy_file, "--force")
    assert forced.exit_code == 0
    assert "false" in forced.output.split("cache_hit")[1][:40]
    assert _run_id(forced.output) == _run_id(first.output)
    assert len(_store(project).query_runs()) == 1


def test_the_strategy_version_is_registered_by_the_loader(
    cli: CliRunner, project: Path, strategy_file: Path
) -> None:
    """The run cites a ``strategy_id`` the loader derived from the source bytes,
    and the immutable copy is on disk under that id."""
    result = _backtest(cli, strategy_file)
    run_id = _run_id(result.output)
    row = _store(project).find_run(run_id)
    assert row is not None
    assert (project / "strategies" / "generated" / f"{row.strategy_id}.py").is_file()


def test_the_test_partition_is_refused_outside_the_lockbox_profile(
    cli: CliRunner, project: Path, strategy_file: Path
) -> None:
    """INV-5, through the CLI.

    The container attaches a ``PartitionGuard`` to the market-data source in every
    profile but ``lockbox``, so the refusal happens when the bars are requested —
    before a strategy has been loaded, let alone run.
    """
    result = cli.invoke(
        app,
        ["backtest", "run", "--strategy", str(strategy_file), "--segment", "test"],
    )
    assert result.exit_code == EXIT_CODES["LockboxViolation"]
    assert _store(project).query_runs() == []


def test_the_validation_segment_is_a_different_run(
    cli: CliRunner, project: Path, strategy_file: Path
) -> None:
    """Validation is reachable — it is the *test* partition that is locked — and
    it is a different run identity, so its numbers can never be confused with the
    training ones."""
    train = _backtest(cli, strategy_file)
    val = cli.invoke(app, ["backtest", "run", "--strategy", str(strategy_file), "--segment", "val"])
    assert val.exit_code == 0, val.output
    assert _run_id(val.output) != _run_id(train.output)
    assert {row.segment for row in _store(project).query_runs()} == {"train", "val"}


def test_a_missing_strategy_file_is_reported(cli: CliRunner, project: Path) -> None:
    result = cli.invoke(
        app, ["backtest", "run", "--strategy", "does-not-exist.py", "--segment", "train"]
    )
    assert result.exit_code == EXIT_CODES["StrategyError"]
