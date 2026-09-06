"""The deflated Sharpe ratio (master spec section 14.4, task T28).

Section 20 names three checks — a reference example, ``DSR = 0.5`` at
``SR = SR0``, and monotonicity in ``M`` — and each pins a different part of the
formula. The middle one is the sharpest: it holds only if the benchmark is
actually the expected maximum rather than zero, so it fails immediately if the
selection correction is dropped.

The unit trap is tested too. ``MetricSet.sharpe`` is annualised and ``T`` counts
bars; passing the former with the latter overstates the ratio by roughly
``sqrt(bars_per_year)`` and turns every deflated Sharpe into 1.0 with no
exception and no obviously wrong number.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from quantlab.core.errors import ConfigError
from quantlab.core.validation.deflated_sharpe import (
    EULER_MASCHERONI,
    deannualise,
    deflated_sharpe,
    expected_max_sharpe,
    moments,
    probabilistic_sharpe,
    sharpe_variance,
)


# ---------------------------------------------------------------------------
# the expected maximum
# ---------------------------------------------------------------------------
def test_the_expected_maximum_reproduces_the_published_formula() -> None:
    """Bailey and López de Prado (2014), equation 5, computed independently here.

    A reference example in the strict sense: the same inputs through the paper's
    algebra, written out rather than called.
    """
    trials, variance = 100, 0.04
    expected = math.sqrt(variance) * (
        (1.0 - EULER_MASCHERONI) * float(stats.norm.ppf(1.0 - 1.0 / trials))
        + EULER_MASCHERONI * float(stats.norm.ppf(1.0 - 1.0 / (trials * math.e)))
    )
    assert expected_max_sharpe(trials, variance) == pytest.approx(expected, abs=1e-12)
    assert expected > 0.0


def test_the_expected_maximum_grows_with_the_number_of_trials() -> None:
    """Run enough strategies on noise and one of them looks excellent; the
    benchmark has to rise to say so."""
    values = [expected_max_sharpe(m, 0.04) for m in (2, 10, 100, 1_000, 10_000)]
    assert values == sorted(values)
    assert values[0] < values[-1]


def test_a_single_trial_carries_no_selection_bias() -> None:
    """The formula is undefined at ``M = 1`` — ``Z(0)`` is negative infinity — so
    the case is handled rather than left to produce a nan."""
    assert expected_max_sharpe(1, 0.04) == 0.0
    assert not math.isnan(expected_max_sharpe(1, 0.04))


def test_trials_that_all_scored_alike_offered_nothing_to_select_from() -> None:
    assert expected_max_sharpe(1_000, 0.0) == 0.0


def test_a_wider_spread_of_trials_raises_the_bar() -> None:
    assert expected_max_sharpe(100, 0.09) > expected_max_sharpe(100, 0.01)


def test_zero_trials_is_refused() -> None:
    with pytest.raises(ConfigError, match="at least one trial"):
        expected_max_sharpe(0, 0.04)


def test_the_trial_variance_needs_more_than_one_trial() -> None:
    """A variance of zero would make the selection correction vanish — right for
    a single trial, and wrong as a default."""
    with pytest.raises(ConfigError, match="at least two trials"):
        sharpe_variance([1.0])
    assert sharpe_variance([1.0, 2.0, 3.0]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# the probabilistic Sharpe ratio
# ---------------------------------------------------------------------------
def test_a_sharpe_equal_to_its_benchmark_is_a_coin_flip() -> None:
    """Section 20's stated criterion. It holds only if the benchmark really is
    the expected maximum, so it fails the moment the selection correction is
    dropped."""
    assert probabilistic_sharpe(0.05, benchmark=0.05, n_periods=1_000) == pytest.approx(0.5)


def test_the_deflated_ratio_is_a_coin_flip_at_the_expected_maximum() -> None:
    benchmark = expected_max_sharpe(500, 0.04)
    assert deflated_sharpe(benchmark, n_trials=500, var_sr=0.04, n_periods=5_000) == pytest.approx(
        0.5
    )


def test_more_data_makes_the_same_estimate_more_convincing() -> None:
    short = probabilistic_sharpe(0.05, benchmark=0.0, n_periods=100)
    long = probabilistic_sharpe(0.05, benchmark=0.0, n_periods=10_000)
    assert 0.5 < short < long < 1.0


def test_negative_skew_and_fat_tails_lower_confidence() -> None:
    """Both widen the standard error of a Sharpe ratio, so the same point
    estimate says less."""
    normal = probabilistic_sharpe(0.05, benchmark=0.0, n_periods=2_000, skew=0.0, kurtosis=3.0)
    skewed = probabilistic_sharpe(0.05, benchmark=0.0, n_periods=2_000, skew=-1.5, kurtosis=3.0)
    fat = probabilistic_sharpe(0.05, benchmark=0.0, n_periods=2_000, skew=0.0, kurtosis=12.0)
    assert skewed < normal
    assert fat < normal


def test_excess_kurtosis_passed_by_mistake_is_caught_where_it_can_be() -> None:
    """``kurtosis`` is the non-excess fourth moment, so a normal is 3. A value
    that drives the estimator's variance non-positive cannot have come from one
    sample, and refusing beats returning a confident number from the square root
    of a negative."""
    with pytest.raises(ConfigError, match="non-excess fourth moment"):
        probabilistic_sharpe(2.0, benchmark=0.0, n_periods=1_000, skew=0.0, kurtosis=-5.0)


def test_a_series_too_short_to_judge_is_refused() -> None:
    with pytest.raises(ConfigError, match="at least two periods"):
        probabilistic_sharpe(0.05, benchmark=0.0, n_periods=1)


# ---------------------------------------------------------------------------
# the deflation itself
# ---------------------------------------------------------------------------
def test_the_deflated_ratio_falls_as_the_search_widens() -> None:
    """Section 20's stated criterion, and section 14.4's reason for counting
    every evaluation: a search that tried more things must clear a higher bar."""
    values = [
        deflated_sharpe(0.06, n_trials=m, var_sr=0.04, n_periods=5_000)
        for m in (2, 10, 100, 1_000, 10_000)
    ]
    assert values == sorted(values, reverse=True)
    assert values[0] > values[-1]


def test_a_wide_search_can_deflate_a_good_looking_result_to_nothing() -> None:
    """The point of the whole check.

    ``var_sr`` is the *per-period* dispersion of the trial Sharpe ratios, and its
    magnitude matters as much as ``M``: a search whose trials ranged over 0.02
    per-period Sharpe offered that much luck to its winner. Two trials of that
    spread set almost no bar; a hundred thousand set one this result cannot
    clear.
    """
    narrow = deflated_sharpe(0.08, n_trials=2, var_sr=0.0004, n_periods=5_000)
    wide = deflated_sharpe(0.08, n_trials=100_000, var_sr=0.0004, n_periods=5_000)
    assert narrow > 0.9
    assert wide < 0.5


def test_the_result_is_always_a_probability() -> None:
    for sharpe in (-0.5, -0.05, 0.0, 0.05, 0.5):
        for trials in (1, 10, 5_000):
            value = deflated_sharpe(sharpe, n_trials=trials, var_sr=0.02, n_periods=2_000)
            assert 0.0 <= value <= 1.0


def test_a_single_trial_reduces_to_the_probabilistic_sharpe_ratio() -> None:
    assert deflated_sharpe(0.05, n_trials=1, var_sr=0.04, n_periods=1_000) == pytest.approx(
        probabilistic_sharpe(0.05, benchmark=0.0, n_periods=1_000)
    )


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------
def test_annualised_and_per_period_sharpe_are_not_interchangeable() -> None:
    """The one error here that produces no exception and no obviously wrong
    number — just a deflated Sharpe of 1.0 for everything."""
    annualised, bars_per_year = 1.5, 8_760
    per_period = deannualise(annualised, bars_per_year)
    assert per_period == pytest.approx(annualised / math.sqrt(bars_per_year))

    honest = deflated_sharpe(per_period, n_trials=1_000, var_sr=0.04, n_periods=8_760)
    mistaken = deflated_sharpe(annualised, n_trials=1_000, var_sr=0.04, n_periods=8_760)
    assert mistaken == pytest.approx(1.0)
    assert honest < mistaken


def test_a_nonsense_bar_count_is_refused() -> None:
    with pytest.raises(ConfigError, match="bars_per_year must be positive"):
        deannualise(1.5, 0)


# ---------------------------------------------------------------------------
# moments
# ---------------------------------------------------------------------------
def test_the_four_moments_come_from_one_sample() -> None:
    """Which is the mistake that drives the estimator's variance negative: four
    numbers computed from different slices of data."""
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0005, 0.01, size=4_000)
    sharpe, skew, kurtosis, n_periods = moments(returns)

    assert n_periods == 4_000
    assert sharpe == pytest.approx(returns.mean() / returns.std(ddof=1))
    assert skew == pytest.approx(0.0, abs=0.15)
    assert kurtosis == pytest.approx(3.0, abs=0.3)
    assert (
        deflated_sharpe(
            sharpe, n_trials=100, var_sr=0.01, n_periods=n_periods, skew=skew, kurtosis=kurtosis
        )
        >= 0.0
    )


def test_moments_of_a_flat_series_report_no_edge_rather_than_dividing_by_zero() -> None:
    sharpe, _skew, _kurtosis, n_periods = moments([0.0] * 100)
    assert sharpe == 0.0
    assert n_periods == 100


def test_moments_ignore_undefined_returns() -> None:
    """A NaN return is a bar the strategy could not be measured on, not a zero."""
    sharpe, _s, _k, n_periods = moments([0.01, float("nan"), 0.02, float("inf")])
    assert n_periods == 2
    assert math.isfinite(sharpe)


def test_moments_of_too_short_a_series_are_refused() -> None:
    with pytest.raises(ConfigError, match="at least two returns"):
        moments([0.01])
