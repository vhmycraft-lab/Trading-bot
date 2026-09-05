"""INV-7 end to end: a recorded run can be rebuilt from what was stored.

Master spec section 20 names two cases for this file — "INV-7 on golden runs, and
on a run executed in a fresh process". Both are here, and the second one matters
most: a reproduction that only holds inside the process that produced it proves
nothing about the record on disk. So the fresh-process tests shell out to
``python -m quantlab`` and let the CLI rebuild everything from the database, the
immutable source copy and the artifact tree, with no state carried across.

The file also covers section 20's other T23 line, "report Markdown + PNG created
offline": the tearsheet is drawn through matplotlib's Agg backend, so it works on
a headless machine and never reaches for a display or the network.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import text
from typer.testing import CliRunner

from quantlab.adapters.store.sqlite import (
    SqliteExperimentStore,
    create_db_engine,
    make_session_factory,
)
from quantlab.cli import app
from quantlab.cli.report import REPORT_NAME
from quantlab.core.errors import EXIT_CODES
from quantlab.reporting.tearsheet import TEARSHEET_NAME

pytestmark = pytest.mark.slow

RUN_ID = re.compile(r"run ([0-9a-f]{16})")


def _run_id(output: str) -> str:
    match = RUN_ID.search(output)
    assert match, f"no run id in output:\n{output}"
    return match.group(1)


def _store(project: Path) -> SqliteExperimentStore:
    engine = create_db_engine(project / "quantlab.db", create_parents=False)
    return SqliteExperimentStore(make_session_factory(engine))


def _quantlab(project: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Invoke the CLI in a *new* interpreter rooted at the project directory.

    ``sys.executable`` is the interpreter running the tests, so the subprocess
    sees the same installed package without needing the console script on PATH.
    ``COLUMNS`` keeps Rich from wrapping a run id in half.
    """
    return subprocess.run(
        [sys.executable, "-m", "quantlab", *args],
        cwd=project,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": str(project), "COLUMNS": "220"},
        check=False,
        timeout=600,
    )


@pytest.fixture
def golden_run(cli: CliRunner, project: Path, strategy_file: Path) -> str:
    """One recorded, successful run — the thing every test below tries to rebuild."""
    result = cli.invoke(
        app, ["backtest", "run", "--strategy", str(strategy_file), "--segment", "train"]
    )
    assert result.exit_code == 0, result.output
    return _run_id(result.output)


def test_a_golden_run_reproduces_exactly(cli: CliRunner, project: Path, golden_run: str) -> None:
    """Section 11.4: every metric the run defined comes back within 1e-9."""
    result = cli.invoke(app, ["report", "reproduce", golden_run])
    assert result.exit_code == 0, result.output
    assert "reproduces exactly" in result.output


def test_a_golden_run_reproduces_in_a_fresh_process(project: Path, golden_run: str) -> None:
    """The second case section 20 asks for.

    Nothing survives from the run that produced these numbers: the strategy is
    re-read from its immutable source copy, the costs from ``backtest_config.json``
    and the bars from the parquet store, all inside a new interpreter.
    """
    done = _quantlab(project, "report", "reproduce", golden_run)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "reproduces exactly" in done.stdout


def test_the_run_identity_is_stable_across_processes(
    project: Path, strategy_file: Path, golden_run: str
) -> None:
    """The §11.2 hash depends on inputs only, never on process state.

    Asking for the same backtest from a new interpreter has to land on the same
    id and be served from the cache — if it did not, every restart would re-derive
    numbers the platform already has and inflate the evaluation count that section
    14.4 deflates for multiple testing.
    """
    done = _quantlab(
        project, "backtest", "run", "--strategy", str(strategy_file), "--segment", "train"
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert _run_id(done.stdout) == golden_run
    assert "true" in done.stdout.split("cache_hit")[1][:40]
    assert len(_store(project).query_runs()) == 1


def test_a_tampered_metric_fails_reproduction(
    cli: CliRunner, project: Path, golden_run: str
) -> None:
    """The check has to be able to *fail*, or it certifies nothing.

    The metric is changed with raw SQL, behind the append-only guard, because the
    guard is a Python-level rule and the corruption INV-7 exists to catch is one
    that did not come through it.
    """
    engine = create_db_engine(project / "quantlab.db", create_parents=False)
    with engine.begin() as connection:
        changed = connection.execute(
            text("UPDATE metric SET value = value + 0.5 WHERE run_id = :r AND name = 'net_return'"),
            {"r": golden_run},
        ).rowcount
    engine.dispose()
    assert changed == 1

    result = cli.invoke(app, ["report", "reproduce", golden_run])
    assert result.exit_code == 1
    assert "did NOT reproduce" in result.output
    assert "net_return" in result.output


def test_reproduce_refuses_a_run_it_has_never_heard_of(cli: CliRunner, project: Path) -> None:
    result = cli.invoke(app, ["report", "reproduce", "0" * 16])
    assert result.exit_code == EXIT_CODES["other"]


def test_reproduce_refuses_a_run_that_did_not_succeed(
    cli: CliRunner, project: Path, golden_run: str
) -> None:
    """A failed run has no numbers to agree with, so there is nothing to verify."""
    store = _store(project)
    row = store.find_run(golden_run)
    assert row is not None
    failed_id = "f" * 16
    store.create_run(
        run_id=failed_id,
        experiment_id=row.experiment_id,
        strategy_id=row.strategy_id,
        dataset_id=row.dataset_id,
        split_id=row.split_id,
        segment=row.segment,
        params_json='{"note":"deliberately failed"}',
        engine_name=row.engine_name,
        engine_version=row.engine_version,
        artifact_dir=row.artifact_dir,
    )
    store.finish_run(failed_id, "failed", error={"type": "StrategyError", "message": "boom"})

    result = cli.invoke(app, ["report", "reproduce", failed_id])
    assert result.exit_code == EXIT_CODES["other"]
    assert "only a successful run" in result.output


def test_report_show_writes_markdown_and_a_png_offline(
    cli: CliRunner, project: Path, golden_run: str
) -> None:
    """Section 20's other T23 line, in a headless process with no network."""
    out = project / "out"
    done = _quantlab(project, "report", "show", golden_run, "--out", str(out))
    assert done.returncode == 0, done.stdout + done.stderr

    report = (out / REPORT_NAME).read_text(encoding="utf-8")
    assert golden_run in report
    assert "net_return" in report

    png = out / TEARSHEET_NAME
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_report_names_the_source_of_every_number(
    cli: CliRunner, project: Path, golden_run: str
) -> None:
    """A report that did not cite its inputs could not be checked against them."""
    result = cli.invoke(app, ["report", "show", golden_run])
    assert result.exit_code == 0, result.output

    row = _store(project).find_run(golden_run)
    assert row is not None
    report = (project / "artifacts" / "runs" / golden_run / REPORT_NAME).read_text(encoding="utf-8")
    for cited in (row.strategy_id, row.dataset_id, row.split_id, row.segment):
        assert cited in report
