"""Rendering a recorded run (master spec sections 11.3, 11.4; task T23).

A report is evidence or it is decoration. The tests here are mostly about the
difference: that an undefined metric stays undefined rather than becoming zero,
that a run made on a dirty tree says so, and that the same record always renders
the same bytes so two reports can be diffed against each other.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantlab.cli.report import ABS_TOL, REL_TOL, compare_metrics
from quantlab.reporting.markdown import (
    HEADLINE_METRICS,
    RunReport,
    format_metric,
    render_run_report,
)
from quantlab.reporting.tearsheet import TEARSHEET_NAME, write_tearsheet


def _report(**overrides: object) -> RunReport:
    base: dict[str, object] = {
        "run_id": "abc123",
        "strategy_id": "s0",
        "segment": "train",
        "status": "ok",
        "engine": "simple_bar:1",
        "dataset_id": "d0",
        "split_id": "sp0",
        "params": {"fast": 20, "slow": 100},
        "metrics": {"net_return": 0.42, "n_trades": 7, "profit_factor": None},
        "environment": {"python": "3.12.11", "git_dirty": False, "packages": {"numpy": "2.1.0"}},
        "n_trades": 7,
        "artifact_dir": "artifacts/runs/abc123",
    }
    base.update(overrides)
    return RunReport(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# undefined is not zero
# ---------------------------------------------------------------------------
def test_an_undefined_metric_renders_as_undefined() -> None:
    """A profit factor with no losing trades is undefined, not zero. Printing it
    as a number would make a report that reads plausibly and says something
    false."""
    assert format_metric(None) == "—"
    assert format_metric(0.0) == "0"
    assert "—" in render_run_report(_report())


@pytest.mark.parametrize(
    ("value", "rendered"),
    [(1.0, "1"), (7, "7"), (0.123456789, "0.123457"), (True, "yes"), (False, "no"), (-2.5, "-2.5")],
)
def test_metrics_render_readably(value: float | bool, rendered: str) -> None:
    assert format_metric(value) == rendered


def test_a_boolean_metric_is_not_shown_as_a_number() -> None:
    """``ruined`` is a fact about the run, not a quantity."""
    assert format_metric(True) == "yes"
    text = render_run_report(_report(metrics={"ruined": True}))
    assert "| ruined | yes |" in text


# ---------------------------------------------------------------------------
# what the report says about itself
# ---------------------------------------------------------------------------
def test_the_report_names_everything_needed_to_find_the_run_again() -> None:
    text = render_run_report(_report())
    for expected in ("abc123", "s0", "train", "simple_bar:1", "d0", "sp0"):
        assert expected in text


def test_a_dirty_working_tree_is_flagged() -> None:
    """Section 11.3: a run with ``git_dirty=true`` is allowed but flagged. The
    recorded commit does not fully describe the code that ran."""
    text = render_run_report(
        _report(environment={"python": "3.12", "git_dirty": True, "packages": {}})
    )
    assert "Read with care" in text
    assert "uncommitted changes" in text


def test_a_clean_tree_is_not_flagged() -> None:
    assert "Read with care" not in render_run_report(_report())


def test_a_failed_run_says_so_at_the_top() -> None:
    text = render_run_report(_report(status="failed"))
    assert "Read with care" in text
    assert "`failed`" in text


def test_headline_metrics_come_first() -> None:
    """Drawdown buried in an alphabetical list is drawdown that goes unread."""
    text = render_run_report(
        _report(metrics={"zzz_last": 1.0, "max_drawdown": 0.2, "net_return": 0.4})
    )
    assert text.index("net_return") < text.index("max_drawdown") < text.index("zzz_last")
    assert "net_return" in HEADLINE_METRICS


def test_remaining_metrics_are_alphabetical_so_the_output_is_stable() -> None:
    text = render_run_report(_report(metrics={"beta": 1.0, "alpha": 2.0}))
    assert text.index("alpha") < text.index("beta")


def test_the_same_record_always_renders_the_same_bytes() -> None:
    """A report that varied between runs could not be diffed against an earlier
    one, and a difference that means nothing is worse than no difference."""
    assert render_run_report(_report()) == render_run_report(_report())


def test_the_report_ends_with_exactly_one_newline() -> None:
    text = render_run_report(_report())
    assert text.endswith("\n")
    assert not text.endswith("\n\n")


def test_an_empty_report_still_renders() -> None:
    text = render_run_report(
        RunReport(
            run_id="r",
            strategy_id="s",
            segment="train",
            status="ok",
            engine="e",
            dataset_id="d",
            split_id="sp",
        )
    )
    assert "# Run `r`" in text


def test_the_tearsheet_is_linked_when_one_was_drawn() -> None:
    assert "![equity](tearsheet.png)" in render_run_report(_report(tearsheet=TEARSHEET_NAME))
    assert "![equity]" not in render_run_report(_report())


# ---------------------------------------------------------------------------
# the tearsheet
# ---------------------------------------------------------------------------
def test_a_tearsheet_is_written_offline(tmp_path: Path) -> None:
    """Rendered with a non-interactive backend: every machine that runs this in a
    batch is headless, and a report that needed a display would never be made."""
    equity = pd.Series(10_000.0 * np.cumprod(1 + np.linspace(-0.001, 0.002, 200)))
    path = write_tearsheet(equity, tmp_path / TEARSHEET_NAME, title="run abc")
    assert path.is_file()
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_a_tearsheet_creates_its_directory(tmp_path: Path) -> None:
    equity = pd.Series([1.0, 2.0, 1.5])
    path = write_tearsheet(equity, tmp_path / "nested" / "deeper" / TEARSHEET_NAME)
    assert path.is_file()


def test_an_empty_equity_curve_is_refused(tmp_path: Path) -> None:
    """An image of nothing would still look like a result."""
    with pytest.raises(ValueError, match="empty equity curve"):
        write_tearsheet(pd.Series(dtype="float64"), tmp_path / TEARSHEET_NAME)


def test_a_flat_equity_curve_still_draws(tmp_path: Path) -> None:
    """The degenerate case: a strategy that never traded is a result too."""
    path = write_tearsheet(pd.Series([10_000.0] * 50), tmp_path / TEARSHEET_NAME)
    assert path.stat().st_size > 0


# ---------------------------------------------------------------------------
# reproduction comparison (spec section 11.4)
# ---------------------------------------------------------------------------
def test_identical_metrics_reproduce() -> None:
    metrics = {"net_return": 0.42, "sharpe": 1.1, "n_trades": 7}
    assert compare_metrics(metrics, dict(metrics)) == []


def test_the_tolerance_is_the_one_the_spec_names() -> None:
    assert (REL_TOL, ABS_TOL) == (1e-9, 1e-12)


def test_a_difference_beyond_the_tolerance_is_reported() -> None:
    differences = compare_metrics({"net_return": 0.42}, {"net_return": 0.42 * (1 + 1e-6)})
    assert [name for name, _, _ in differences] == ["net_return"]


def test_a_difference_within_the_tolerance_is_not() -> None:
    """Two runs of a deterministic engine agree bit for bit; the tolerance exists
    for the last bit of a float, not for a different answer."""
    assert compare_metrics({"net_return": 0.42}, {"net_return": 0.42 * (1 + 1e-13)}) == []


def test_a_metric_that_was_undefined_is_not_required_to_stay_undefined() -> None:
    """Demanding that would fail a reproduction for agreeing about nothing."""
    assert compare_metrics({"profit_factor": None}, {"profit_factor": 3.0}) == []


def test_a_metric_that_became_undefined_is_a_difference() -> None:
    """The other direction is not symmetric: the first run defined it, so a
    reproduction that cannot is a reproduction that failed."""
    differences = compare_metrics({"profit_factor": 3.0}, {"profit_factor": None})
    assert differences == [("profit_factor", 3.0, None)]


def test_a_missing_metric_is_a_difference() -> None:
    assert compare_metrics({"sharpe": 1.0}, {}) == [("sharpe", 1.0, None)]


def test_boolean_metrics_are_compared_as_booleans() -> None:
    """``ruined`` differing is a categorically different run, not a small number."""
    assert compare_metrics({"ruined": False}, {"ruined": False}) == []
    assert compare_metrics({"ruined": False}, {"ruined": True}) == [("ruined", False, True)]


def test_differences_are_reported_in_a_stable_order() -> None:
    reference = {"b": 1.0, "a": 2.0, "c": 3.0}
    observed = {"b": 9.0, "a": 9.0, "c": 9.0}
    assert [name for name, _, _ in compare_metrics(reference, observed)] == ["a", "b", "c"]
