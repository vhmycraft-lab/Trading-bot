"""Walk-forward testing (master spec section 15, task T54).

Section 20 names three things for this file — window generation, stitching, and
WFE — and the middle one carries the most weight. Each window's backtest starts
from the same initial equity, so concatenating the levels would reset the account
at every boundary and hide every compounding effect, including the one that
matters: a drawdown that carries across a window edge.

The optimiser and the evaluator are both injected, so these tests drive the
arithmetic with scripted ones. A failure here is then a failure in the
walk-forward, not in a strategy or a search.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
import pytest
from tests.helpers import ScriptedStrategy, make_bars, zero_cost_config

from quantlab.adapters.engine.simple_bar import SimpleBarEngine
from quantlab.core.config import WalkForwardSettings
from quantlab.core.errors import ConfigError
from quantlab.core.metrics import MetricSet
from quantlab.core.splits import SplitPolicy, walk_forward_windows
from quantlab.core.types import BarFrame
from quantlab.walkforward.runner import (
    WindowResult,
    WindowSearch,
    parameter_stability,
    stitch_equity,
    walk_forward,
)

EPOCH = 1_600_000_000_000
HOUR = 3_600_000


def policy(n_train: int = 300, n_val: int = 200) -> SplitPolicy:
    return SplitPolicy(
        symbol="BTC/USDT",
        timeframe="1h",
        train_start_ts=EPOCH,
        train_end_ts=EPOCH + (n_train - 1) * HOUR,
        embargo_bars=0,
        val_start_ts=EPOCH + n_train * HOUR,
        val_end_ts=EPOCH + (n_train + n_val - 1) * HOUR,
        test_start_ts=EPOCH + (n_train + n_val) * HOUR,
        test_end_ts=None,
        source_json="{}",
        dataset_id="d0",
    )


def bars(n: int = 500) -> BarFrame:
    return BarFrame(
        make_bars(n, start_ts=EPOCH, seed=5, drift=0.05), symbol="BTC/USDT", timeframe="1h"
    )


def settings(**overrides: Any) -> WalkForwardSettings:
    base: dict[str, Any] = {"is_bars": 200, "oos_bars": 100, "step_bars": 100}
    base.update(overrides)
    return WalkForwardSettings(**base)


def scripted_evaluate(frame: BarFrame, params: Mapping[str, Any], segment: str) -> Any:
    """Run a simple alternating strategy, so every window has real trades."""
    del segment
    hold = int(params.get("hold", 10))
    script = ["long" if (index // hold) % 2 == 0 else "flat" for index in range(frame.n_bars)]
    return SimpleBarEngine().run(ScriptedStrategy(script), frame, {}, zero_cost_config())


# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------
def test_windows_cover_train_and_validation_without_straddling_the_embargo() -> None:
    """Section 15.1. The embargo is excluded from the timeline, so a window
    cannot span it — a position open at the boundary would otherwise be closed on
    data the in-sample fit already saw."""
    guarded = policy()
    windows = walk_forward_windows(guarded, is_bars=200, oos_bars=100, step_bars=100)
    assert windows
    for window in windows:
        assert window.is_start_ts < window.is_end_ts < window.oos_start_ts < window.oos_end_ts
        assert window.oos_end_ts <= guarded.val_end_ts


def test_a_timeline_too_short_for_one_window_is_refused() -> None:
    """Reported here rather than as an empty result, which a caller would read as
    "the walk-forward found nothing" instead of "it never ran"."""
    with pytest.raises(ConfigError, match="too short for one walk-forward window"):
        walk_forward(
            bars(),
            policy(n_train=50, n_val=20),
            settings(is_bars=200, oos_bars=100),
            optimizer=lambda frame, k: WindowSearch(params={}),
            evaluate=scripted_evaluate,
            config=zero_cost_config(),
        )


# ---------------------------------------------------------------------------
# stitching
# ---------------------------------------------------------------------------
def test_stitching_chains_returns_rather_than_concatenating_levels() -> None:
    """Section 15.3's ``E_{k+1,0} = E_{k,end}``.

    Two windows that each doubled from 100 must stitch to 400, not to a curve
    that fell back to 100 at the boundary.
    """
    first = pd.Series([100.0, 150.0, 200.0], index=[0, 1, 2])
    second = pd.Series([100.0, 150.0, 200.0], index=[3, 4, 5])
    stitched = stitch_equity([first, second])
    assert list(stitched) == [100.0, 150.0, 200.0, 300.0, 400.0]


def test_a_drawdown_carries_across_a_window_edge() -> None:
    """The compounding effect that concatenating levels would hide."""
    up = pd.Series([100.0, 200.0])
    down = pd.Series([100.0, 50.0])
    stitched = stitch_equity([up, down])
    assert list(stitched) == [100.0, 200.0, 100.0]


def test_the_boundary_point_is_not_counted_twice() -> None:
    first = pd.Series([100.0, 110.0])
    second = pd.Series([100.0, 120.0])
    assert len(stitch_equity([first, second])) == 3


def test_stitching_nothing_gives_nothing() -> None:
    assert stitch_equity([]).empty
    assert stitch_equity([pd.Series(dtype="float64")]).empty


def test_one_window_stitches_to_itself() -> None:
    only = pd.Series([100.0, 120.0, 90.0])
    assert list(stitch_equity([only])) == [100.0, 120.0, 90.0]


# ---------------------------------------------------------------------------
# the whole run
# ---------------------------------------------------------------------------
def test_a_walk_forward_produces_one_result_per_window() -> None:
    result = walk_forward(
        bars(),
        policy(),
        settings(),
        optimizer=lambda frame, k: WindowSearch(params={"hold": 10 + k}, n_evaluations=5),
        evaluate=scripted_evaluate,
        config=zero_cost_config(),
    )
    expected = walk_forward_windows(policy(), is_bars=200, oos_bars=100, step_bars=100)
    assert len(result.windows) == len(expected)
    assert [window.k for window in result.windows] == list(range(len(expected)))


def test_each_window_is_searched_on_its_own_in_sample_bars() -> None:
    """The out-of-sample window is judged by a search that never saw it."""
    seen: list[tuple[int, int, int]] = []

    def optimizer(frame: BarFrame, k: int) -> WindowSearch:
        seen.append((k, int(frame.ts_open[0]), int(frame.ts_open[-1])))
        return WindowSearch(params={"hold": 10})

    walk_forward(
        bars(),
        policy(),
        settings(),
        optimizer=optimizer,
        evaluate=scripted_evaluate,
        config=zero_cost_config(),
    )
    windows = walk_forward_windows(policy(), is_bars=200, oos_bars=100, step_bars=100)
    assert [k for k, _, _ in seen] == [window.k for window in windows]
    for (_, start, end), window in zip(seen, windows, strict=True):
        assert start >= window.is_start_ts
        assert end <= window.is_end_ts
        assert end < window.oos_start_ts


def test_the_segments_are_named_as_the_specification_requires() -> None:
    """``wf_is:k`` and ``wf_oos:k`` (section 15.2), because section 13.8's
    segment assertion admits ``wf_is`` and nothing else admits ``wf_oos``."""
    segments: list[str] = []

    def evaluate(frame: BarFrame, params: Mapping[str, Any], segment: str) -> Any:
        segments.append(segment)
        return scripted_evaluate(frame, params, segment)

    walk_forward(
        bars(),
        policy(),
        settings(),
        optimizer=lambda frame, k: WindowSearch(params={"hold": 10}),
        evaluate=evaluate,
        config=zero_cost_config(),
    )
    assert segments[:4] == ["wf_is:0", "wf_oos:0", "wf_is:1", "wf_oos:1"]
    assert all(name.startswith(("wf_is:", "wf_oos:")) for name in segments)


def test_every_window_s_evaluations_are_charged() -> None:
    result = walk_forward(
        bars(),
        policy(),
        settings(),
        optimizer=lambda frame, k: WindowSearch(params={"hold": 10}, n_evaluations=7),
        evaluate=scripted_evaluate,
        config=zero_cost_config(),
    )
    assert result.n_trials_accounted == 7 * len(result.windows)


def test_the_walk_forward_efficiency_compares_stitched_out_of_sample_to_in_sample() -> None:
    """Section 15.4: ``cagr_oos_stitched / mean_k(cagr_is_k)``."""
    result = walk_forward(
        bars(),
        policy(),
        settings(),
        optimizer=lambda frame, k: WindowSearch(params={"hold": 10}),
        evaluate=scripted_evaluate,
        config=zero_cost_config(),
    )
    is_cagrs = [w.is_metrics.cagr for w in result.windows if w.is_metrics.cagr is not None]
    mean_is = sum(is_cagrs) / len(is_cagrs)
    if mean_is > 0 and result.stitched.cagr is not None:
        assert result.wfe == pytest.approx(result.stitched.cagr / mean_is)
    else:
        assert result.wfe is None


def test_the_efficiency_is_undefined_rather_than_infinite() -> None:
    """A mean in-sample CAGR of zero or below would make the ratio report a
    number about nothing."""
    result = walk_forward(
        bars(),
        policy(),
        settings(),
        optimizer=lambda frame, k: WindowSearch(params={"hold": 10}),
        evaluate=lambda frame, params, segment: _flat_result(frame),
        config=zero_cost_config(),
    )
    assert all(
        window.is_metrics.cagr is None or window.is_metrics.cagr <= 0.0 for window in result.windows
    )
    assert result.wfe is None


def _flat_result(frame: BarFrame) -> Any:
    """A run that neither gained nor lost, through the real engine."""
    return SimpleBarEngine().run(
        ScriptedStrategy(["flat"] * frame.n_bars), frame, {}, zero_cost_config()
    )


def test_the_profitable_share_counts_windows_not_bars() -> None:
    result = walk_forward(
        bars(),
        policy(),
        settings(),
        optimizer=lambda frame, k: WindowSearch(params={"hold": 10}),
        evaluate=scripted_evaluate,
        config=zero_cost_config(),
    )
    expected = sum(1 for w in result.windows if w.profitable) / len(result.windows)
    assert result.profitable_oos_share == pytest.approx(expected)
    assert 0.0 <= result.profitable_oos_share <= 1.0


# ---------------------------------------------------------------------------
# parameter stability
# ---------------------------------------------------------------------------
def test_a_parameter_that_never_moved_has_no_variation() -> None:
    result = walk_forward(
        bars(),
        policy(),
        settings(),
        optimizer=lambda frame, k: WindowSearch(params={"hold": 10}),
        evaluate=scripted_evaluate,
        config=zero_cost_config(),
    )
    assert result.param_cv["hold"] == pytest.approx(0.0)


def test_a_parameter_re_chosen_wildly_shows_it() -> None:
    """Section 14.4's soft check 6 charges for this: a parameter the data does
    not determine is one the search is guessing at."""
    result = walk_forward(
        bars(),
        policy(),
        settings(),
        optimizer=lambda frame, k: WindowSearch(params={"hold": 5 + k * 40}),
        evaluate=scripted_evaluate,
        config=zero_cost_config(),
    )
    assert result.param_cv["hold"] > 0.5


def test_only_numeric_parameters_are_measured() -> None:
    """A categorical has no coefficient of variation, and an undefined one is
    omitted rather than reported as zero."""
    window = walk_forward_windows(policy(), is_bars=200, oos_bars=100, step_bars=100)[0]
    made_up = [
        WindowResult(
            k=index,
            window=window,
            params={"hold": 10 + index, "mode": "fast", "invert": True},
            is_metrics=MetricSet(),
            oos_metrics=MetricSet(),
            oos_equity=pd.Series(dtype="float64"),
        )
        for index in range(3)
    ]
    assert set(parameter_stability(made_up)) == {"hold"}


# ---------------------------------------------------------------------------
# what the result says about itself
# ---------------------------------------------------------------------------
def test_a_fixed_parameter_run_says_it_is_weaker_evidence() -> None:
    """Section 15.2 requires the report to state it plainly: reusing fixed
    parameters never tests whether the *search* generalises."""
    searched = walk_forward(
        bars(),
        policy(),
        settings(reoptimize_each_window=True),
        optimizer=lambda frame, k: WindowSearch(params={"hold": 10}, n_evaluations=3),
        evaluate=scripted_evaluate,
        config=zero_cost_config(),
    )
    fixed = walk_forward(
        bars(),
        policy(),
        settings(reoptimize_each_window=False),
        optimizer=lambda frame, k: WindowSearch(params={"hold": 10}, searched=False),
        evaluate=scripted_evaluate,
        config=zero_cost_config(),
    )
    assert "re-searched each window" in searched.evidence
    assert "does not test whether the search generalises" in fixed.evidence
    assert fixed.n_trials_accounted == 0


def test_the_artifact_document_carries_every_window() -> None:
    """Section 15.5's ``walkforward.json``."""
    result = walk_forward(
        bars(),
        policy(),
        settings(),
        optimizer=lambda frame, k: WindowSearch(params={"hold": 10 + k}, n_evaluations=4),
        evaluate=scripted_evaluate,
        config=zero_cost_config(),
    )
    document = result.as_document()
    assert document["n_windows"] == len(result.windows)
    assert len(document["windows"]) == len(result.windows)
    assert document["n_trials_accounted"] == result.n_trials_accounted
    assert set(document["metrics_wf_oos"]) >= {"net_return", "sharpe", "max_drawdown"}
    assert document["evidence"]


def test_the_stitched_metrics_use_the_one_metric_implementation() -> None:
    """Rather than a second one that computes "walk-forward Sharpe" slightly
    differently (section 10)."""
    result = walk_forward(
        bars(),
        policy(),
        settings(),
        optimizer=lambda frame, k: WindowSearch(params={"hold": 10}),
        evaluate=scripted_evaluate,
        config=zero_cost_config(),
    )
    assert not result.stitched_equity.empty
    assert result.stitched.net_return is not None
    assert result.stitched.n_trades == sum(w.oos_metrics.n_trades for w in result.windows)
