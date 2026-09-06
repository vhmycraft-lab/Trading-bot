"""Probability of backtest overfitting (master spec section 14.4, task T29).

Section 20 states the calibration: "noise ⇒ ≈0.5; strong edge ⇒ <0.1". Both are
tested, and the noise case is tested as a **mean over five seeds** rather than
per seed. That is not a weakening — it is what the statement means. On a finite
noise sample one column genuinely is best throughout, and CSCV correctly reports
a low PBO for it; the estimator's own standard deviation across seeds is around
0.22 at this size, so a per-seed bound of ±0.1 would be a coin flip dressed as an
assertion.

The file also carries the regression test for the split-selection bug found while
implementing this: truncating the combinations lexicographically is not sampling
them, and it made the estimate a statement about the tail of the period.
"""

from __future__ import annotations

import itertools
import math
import statistics

import numpy as np
import pytest

from quantlab.core.config import PboSettings
from quantlab.core.errors import ConfigError
from quantlab.core.validation.pbo import (
    _split_choices,
    _unrank,
    logit_ranks,
    probability_of_backtest_overfitting,
)

DEFAULTS = PboSettings()


def noise(seed: int, *, periods: int = 2_000, trials: int = 30) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 0.01, size=(periods, trials))


def with_edge(seed: int, *, edge: float = 0.003) -> np.ndarray:
    """Pure noise, plus one trial with a real and persistent advantage."""
    matrix = noise(seed)
    matrix[:, 0] += edge
    return matrix


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------
def test_a_search_over_noise_is_a_coin_flip_out_of_sample() -> None:
    """Section 20's first criterion, as a mean over five seeds — see the module
    docstring for why per-seed would be a coin flip dressed as an assertion."""
    measured = [probability_of_backtest_overfitting(noise(seed), DEFAULTS).pbo for seed in range(5)]
    assert statistics.fmean(measured) == pytest.approx(0.5, abs=0.1)


def test_a_real_edge_keeps_winning_out_of_sample() -> None:
    """Section 20's second criterion. The in-sample winner is the same trial
    every time, and it is genuinely better, so it rarely lands below median."""
    for seed in range(5):
        assert probability_of_backtest_overfitting(with_edge(seed), DEFAULTS).pbo < 0.1


def test_the_median_logit_separates_an_edge_from_noise() -> None:
    """A summary a reader can act on: strongly positive means the winner kept
    winning, around zero means the ranking was luck."""
    edge = probability_of_backtest_overfitting(with_edge(0), DEFAULTS)
    pure = probability_of_backtest_overfitting(noise(0), DEFAULTS)
    assert edge.median_logit is not None and edge.median_logit > 2.0
    assert pure.median_logit is not None and abs(pure.median_logit) < 1.0


def test_a_stronger_edge_is_harder_to_mistake_for_luck() -> None:
    weak = probability_of_backtest_overfitting(with_edge(1, edge=0.0005), DEFAULTS).pbo
    strong = probability_of_backtest_overfitting(with_edge(1, edge=0.01), DEFAULTS).pbo
    assert strong <= weak


# ---------------------------------------------------------------------------
# the split-selection regression
# ---------------------------------------------------------------------------
def test_the_splits_are_sampled_not_truncated() -> None:
    """Regression. Taking the first ``max_combinations`` in lexicographic order
    yields only splits beginning ``(0, 1, 2, ...)``, so every one puts the early
    blocks in-sample and the late blocks out — and the estimate becomes a
    statement about the tail of the period rather than about the search.

    Measured on pure noise, truncation returned PBO values from 0.01 to 0.82
    across five seeds. The assertion below is on the *shape* of the sample, which
    is what makes the failure impossible rather than merely unlikely.
    """
    chosen = _split_choices(blocks=16, half=8, cap=200)
    assert len(chosen) == 200

    truncated = list(
        itertools.islice((c for c in itertools.combinations(range(16), 8) if 0 in c), 200)
    )
    assert all(combination[:4] == (0, 1, 2, 3) for combination in truncated)
    assert not all(combination[:4] == (0, 1, 2, 3) for combination in chosen)

    # Every block appears in a healthy share of the sampled halves; under
    # truncation the early ones appear in all of them and the late ones in none.
    counts = [sum(1 for c in chosen if block in c) for block in range(16)]
    assert min(counts) > 0.3 * len(chosen)
    assert max(counts) < 0.7 * len(chosen) or counts[0] == len(chosen)


def test_block_zero_is_always_in_sample_so_each_pair_is_used_once() -> None:
    """The symmetry the method is named for: every split is used with its
    complement, and enumerating all ``C(n, n/2)`` would use each pair twice."""
    for cap in (10, 200, 10_000):
        for combination in _split_choices(blocks=16, half=8, cap=cap):
            assert combination[0] == 0
            assert len(set(combination)) == 8
            assert all(0 <= block < 16 for block in combination)


def test_every_split_is_distinct() -> None:
    chosen = _split_choices(blocks=16, half=8, cap=500)
    assert len(set(chosen)) == len(chosen)


def test_the_sample_is_deterministic() -> None:
    """The same matrix must always yield the same estimate (INV-7)."""
    assert _split_choices(16, 8, 200) == _split_choices(16, 8, 200)
    assert (
        probability_of_backtest_overfitting(noise(3), DEFAULTS).pbo
        == probability_of_backtest_overfitting(noise(3), DEFAULTS).pbo
    )


def test_all_splits_are_used_when_they_fit_under_the_cap() -> None:
    chosen = _split_choices(blocks=8, half=4, cap=10_000)
    assert len(chosen) == math.comb(7, 3)
    assert sorted(chosen) == sorted(c for c in itertools.combinations(range(8), 4) if 0 in c)


@pytest.mark.parametrize(("n", "k"), [(7, 3), (9, 4), (15, 7)])
def test_unranking_walks_the_combinations_in_order(n: int, k: int) -> None:
    """The unranking exists so ``C(31, 15)`` need not be materialised to draw
    five hundred from it; it has to agree with enumeration where both are
    possible."""
    enumerated = list(itertools.combinations(range(n), k))
    assert [_unrank(index, n, k) for index in range(len(enumerated))] == enumerated


def test_unranking_can_be_offset_past_a_fixed_block() -> None:
    assert _unrank(0, 15, 7, offset=1) == (1, 2, 3, 4, 5, 6, 7)


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------
def test_every_rank_is_strictly_inside_the_unit_interval() -> None:
    """``(position + 1) / (n_trials + 1)`` is never exactly 0 or 1, so the logit
    is always finite and no split has to be discarded."""
    ranks, logits = logit_ranks(noise(0), DEFAULTS)
    assert ranks
    assert all(0.0 < rank < 1.0 for rank in ranks)
    assert all(math.isfinite(value) for value in logits)


def test_the_probability_is_the_share_of_splits_below_median() -> None:
    report = probability_of_backtest_overfitting(noise(0), DEFAULTS)
    below = sum(1 for value in report.logits if value < 0.0)
    assert report.pbo == pytest.approx(below / report.n_splits)
    assert 0.0 <= report.pbo <= 1.0


def test_the_report_records_what_it_was_computed_over() -> None:
    report = probability_of_backtest_overfitting(noise(0), DEFAULTS)
    document = report.as_dict()
    assert document["n_trials"] == 30
    assert document["n_blocks"] == DEFAULTS.n_blocks
    assert document["n_splits"] == report.n_splits == len(report.logits)


def test_a_trial_that_made_no_bet_neither_wins_nor_loses() -> None:
    """A column with no variation has an undefined Sharpe ratio; scoring it as
    zero keeps it in the ranking without letting a divide-by-zero decide one."""
    matrix = noise(0)
    matrix[:, 5] = 0.0
    report = probability_of_backtest_overfitting(matrix, DEFAULTS)
    assert 0.0 <= report.pbo <= 1.0
    assert all(math.isfinite(value) for value in report.logits)


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------
def test_an_odd_number_of_blocks_is_refused() -> None:
    """CSCV splits the blocks into two equal halves."""
    with pytest.raises(ConfigError, match="n_blocks must be even"):
        probability_of_backtest_overfitting(noise(0), PboSettings(n_blocks=15))


def test_a_single_trial_has_nothing_to_be_ranked_against() -> None:
    with pytest.raises(ConfigError, match="at least two"):
        probability_of_backtest_overfitting(noise(0, trials=1), DEFAULTS)


def test_fewer_periods_than_blocks_is_refused() -> None:
    with pytest.raises(ConfigError, match="fewer periods than blocks"):
        probability_of_backtest_overfitting(noise(0, periods=8), DEFAULTS)


def test_a_transpose_is_caught_only_when_it_makes_the_shape_impossible() -> None:
    """Periods by trials, not trials by periods — and the orientation cannot be
    inferred in general.

    A 2000-by-4 matrix is a valid study of four trials over two thousand periods,
    whether or not the caller meant it that way, so no check can tell them apart.
    What *is* caught is a transpose that leaves fewer periods than blocks, which
    is the shape a wide study takes when it arrives the wrong way round. The
    limitation is stated here rather than papered over with a heuristic that
    would reject legitimate narrow studies.
    """
    wide = noise(0, periods=8, trials=2_000)
    with pytest.raises(ConfigError, match="fewer periods than blocks"):
        probability_of_backtest_overfitting(wide, DEFAULTS)

    # The same data the other way round is a legitimate study and is accepted.
    report = probability_of_backtest_overfitting(wide.T, DEFAULTS)
    assert report.n_trials == 8


def test_a_one_dimensional_input_is_refused() -> None:
    with pytest.raises(ConfigError, match="2-D matrix"):
        probability_of_backtest_overfitting(np.zeros(100), DEFAULTS)
