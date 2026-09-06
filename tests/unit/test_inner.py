"""The inner walk-forward (master spec section 13.6, task T52).

Generalisation has to be measured somewhere, and section 13.7 forbids the obvious
place: sixteen candidates a generation for thirty generations would spend the
validation set four hundred and eighty times, after which it measures nothing. So
train is split into its own folds, and the ``inner_oos`` fitness component,
``p_instability`` and ``p_divergence`` all come from here.

The evaluator is injected, so these tests drive the fold arithmetic directly with
a scripted one rather than through an engine — a failure here is then a failure in
the splitting, not in a strategy.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import Any

import pytest
from tests.helpers import make_bars, zero_cost_config

from quantlab.core.config import InnerWalkForwardSettings
from quantlab.core.errors import ConfigError, StrategyError
from quantlab.core.types import BacktestConfig, BacktestResult, BarFrame
from quantlab.evolution.inner import inner_report, inner_windows

EPOCH = 1_600_000_000_000


def bars(n: int = 400) -> BarFrame:
    return BarFrame(
        make_bars(n, start_ts=EPOCH, seed=5, drift=0.05), symbol="BTC/USDT", timeframe="1h"
    )


# ---------------------------------------------------------------------------
# splitting
# ---------------------------------------------------------------------------
def test_the_folds_are_disjoint_and_ordered() -> None:
    windows = inner_windows(400, InnerWalkForwardSettings(n_folds=4, embargo_bars=0))
    assert len(windows) == 4
    for window in windows:
        assert window.is_end <= window.oos_start
        assert window.oos_start < window.oos_end
    for earlier, later in itertools.pairwise(windows):
        assert earlier.oos_start < later.oos_start
        assert earlier.is_end < later.is_end


def test_the_embargo_separates_in_sample_from_out_of_sample() -> None:
    """The same reason the outer split has one (section 7.3): a position open at
    the boundary would otherwise be closed on data the fit already saw."""
    windows = inner_windows(400, InnerWalkForwardSettings(n_folds=4, embargo_bars=24))
    for window in windows:
        assert window.oos_start - window.is_end == 24


def test_rolling_folds_move_their_in_sample_stretch() -> None:
    windows = inner_windows(400, InnerWalkForwardSettings(n_folds=4, scheme="rolling"))
    assert [window.is_start for window in windows] == sorted(window.is_start for window in windows)
    assert windows[0].is_start == 0
    assert windows[-1].is_start > 0


def test_anchored_folds_grow_from_the_start_of_train() -> None:
    windows = inner_windows(400, InnerWalkForwardSettings(n_folds=4, scheme="anchored"))
    assert {window.is_start for window in windows} == {0}
    assert [window.is_bars for window in windows] == sorted(w.is_bars for w in windows)


def test_a_segment_too_short_to_fold_is_refused() -> None:
    """Better here than as an inner-OOS component that is silently zero for every
    candidate in the run."""
    with pytest.raises(ConfigError, match="too short to split into inner folds"):
        inner_windows(10, InnerWalkForwardSettings(n_folds=4, embargo_bars=24))


def test_a_fold_must_outlast_the_candidate_s_warm_up() -> None:
    """A window shorter than the warm-up produces no signals at all, and a fold
    reporting "no trades" for that reason would read as a strategy that declined
    to trade."""
    settings = InnerWalkForwardSettings(n_folds=4, embargo_bars=0)
    assert inner_windows(400, settings, warmup_bars=50)
    with pytest.raises(ConfigError, match="too short"):
        inner_windows(400, settings, warmup_bars=200)


def test_more_folds_make_each_one_shorter() -> None:
    few = inner_windows(600, InnerWalkForwardSettings(n_folds=2, embargo_bars=0))
    many = inner_windows(600, InnerWalkForwardSettings(n_folds=5, embargo_bars=0))
    assert few[0].oos_bars > many[0].oos_bars


def test_no_fold_reaches_past_the_segment() -> None:
    for n_bars in (200, 333, 400, 601):
        for window in inner_windows(n_bars, InnerWalkForwardSettings(n_folds=3, embargo_bars=5)):
            assert window.oos_end <= n_bars
            assert window.is_start >= 0


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------
class Scripted:
    """An evaluator that returns a fixed result and records what it was asked."""

    def __init__(self, result: BacktestResult) -> None:
        self.result = result
        self.seen: list[int] = []

    def __call__(
        self, window: BarFrame, params: Mapping[str, Any], config: BacktestConfig
    ) -> BacktestResult:
        self.seen.append(window.n_bars)
        return self.result


def a_result(frame: BarFrame) -> BacktestResult:
    """A real backtest result, so the metrics computed from it are real."""
    from tests.helpers import ScriptedStrategy

    from quantlab.adapters.engine.simple_bar import SimpleBarEngine

    script = ["long" if index % 20 < 10 else "flat" for index in range(frame.n_bars)]
    return SimpleBarEngine().run(ScriptedStrategy(script), frame, {}, zero_cost_config())


def test_both_sides_of_every_fold_are_evaluated() -> None:
    """The in-sample Sortino is what the out-of-sample one is compared against; a
    report carrying only the OOS half could not tell a candidate that generalised
    from one that was mediocre everywhere."""
    frame = bars()
    evaluator = Scripted(a_result(frame))
    windows = inner_windows(frame.n_bars, InnerWalkForwardSettings(n_folds=3, embargo_bars=0))

    report = inner_report(evaluator, frame, {}, zero_cost_config(), windows)
    assert len(report.folds) == 3
    # Two evaluations per fold for the Sortino pair, plus one more for the OOS
    # return: the report asks for three numbers per fold.
    assert len(evaluator.seen) == 3 * 3


def test_the_report_derives_the_three_aggregates_fitness_needs() -> None:
    frame = bars()
    windows = inner_windows(frame.n_bars, InnerWalkForwardSettings(n_folds=3, embargo_bars=0))
    report = inner_report(Scripted(a_result(frame)), frame, {}, zero_cost_config(), windows)

    assert report.is_sortino is not None
    assert report.oos_sortino is not None
    # Every fold returned the same scripted result, so there is no spread.
    assert report.return_cv == pytest.approx(0.0)


def test_a_fold_that_fails_is_unmeasured_rather_than_zero() -> None:
    """Section 13.3 already treats an unmeasured input as undemonstrated; letting
    the failure propagate would end the generation over one bad slice."""

    def explode(*_args: Any, **_kwargs: Any) -> BacktestResult:
        raise StrategyError("this fold cannot be evaluated")

    frame = bars()
    windows = inner_windows(frame.n_bars, InnerWalkForwardSettings(n_folds=3, embargo_bars=0))
    report = inner_report(explode, frame, {}, zero_cost_config(), windows)

    assert len(report.folds) == 3
    assert all(fold.oos_sortino is None for fold in report.folds)
    assert report.oos_sortino is None
    assert report.return_cv is None


def test_the_evaluator_only_ever_sees_a_slice_of_train() -> None:
    """INV-9 again, one level down: an inner fold is inside train by construction,
    so no window can reach a bar the outer split protects."""
    frame = bars()
    evaluator = Scripted(a_result(frame))
    windows = inner_windows(frame.n_bars, InnerWalkForwardSettings(n_folds=4, embargo_bars=10))
    inner_report(evaluator, frame, {}, zero_cost_config(), windows)
    assert evaluator.seen
    assert all(0 < seen <= frame.n_bars for seen in evaluator.seen)


def test_the_folds_are_a_property_of_the_bars_not_the_candidate() -> None:
    """Which is why they are computed once per generation: with four folds this
    doubles what a generation costs, and recomputing per candidate would make it
    worse for no gain."""
    settings = InnerWalkForwardSettings(n_folds=4, embargo_bars=12)
    assert inner_windows(500, settings) == inner_windows(500, settings)
