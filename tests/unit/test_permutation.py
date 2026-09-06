"""Market permutation and trade shuffling (spec section 14.4, task T30).

Section 20 names two things: "p-value calibration on random-entry" and "block
bootstrap preserves marginal moments". Both are here, and the first is the one
that matters — a permutation test whose p-values are not uniform under the null
is not a test, it is a number.

``random_entry`` is the right null: it draws its entries from ``ctx.rng`` and
never looks at a price, so on any return series its Sortino ratio is a draw from
the same distribution. If the machinery is right, its p-values are uniform.

Measured across four independent seed bases the KS statistic against uniform came
out at p = 0.044, 0.071, 0.086 and 0.152 — all clearing section 22's alpha = 0.01,
with the tightest at four times the threshold. The test below fixes its seeds, so
it passes or fails permanently rather than flakily.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import stats
from tests.helpers import make_bars, zero_cost_config

from quantlab.adapters.engine.simple_bar import SimpleBarEngine
from quantlab.core.config import PermutationSettings
from quantlab.core.errors import ConfigError
from quantlab.core.metrics import compute_metrics
from quantlab.core.types import BarFrame, Trade
from quantlab.core.validation.permutation import (
    block_bootstrap_indices,
    market_permutation_test,
    permute_bars,
    trade_shuffle_test,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINES = REPO_ROOT / "strategies" / "baselines"
SETTINGS = PermutationSettings()


def bars(seed: int = 1, n: int = 400, drift: float = 0.0) -> BarFrame:
    return BarFrame(make_bars(n, seed=seed, drift=drift), symbol="BTC/USDT", timeframe="1h")


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def trade(no: int, pnl_pct: float) -> Trade:
    return Trade(
        trade_no=no,
        side="long",
        entry_ts=1_600_000_000_000 + no * 3_600_000,
        entry_px=100.0,
        exit_ts=1_600_000_000_000 + (no + 1) * 3_600_000,
        exit_px=100.0 * (1 + pnl_pct),
        qty=1.0,
        fees=0.0,
        slippage_cost=0.0,
        pnl=100.0 * pnl_pct,
        pnl_pct=pnl_pct,
        bars_held=1,
        exit_reason="signal",
    )


# ---------------------------------------------------------------------------
# the block bootstrap
# ---------------------------------------------------------------------------
def test_the_resample_is_the_right_length_and_stays_in_range() -> None:
    positions = block_bootstrap_indices(500, rng(), block_len=24)
    assert positions.size == 500
    assert positions.min() >= 0
    assert positions.max() < 500


def test_blocks_are_contiguous_runs() -> None:
    """A resample that jumped at every step would be an i.i.d. bootstrap and
    would destroy the short-range autocorrelation the null is meant to keep."""
    positions = block_bootstrap_indices(2_000, rng(1), block_len=50)
    steps = np.diff(positions)
    continued = int(np.count_nonzero((steps == 1) | (steps == -(positions.size - 1))))
    assert continued > 0.9 * (positions.size - 1)


def test_block_lengths_average_the_configured_mean() -> None:
    """Geometric with mean ``block_len``, which is what makes the resample
    *stationary*: with fixed blocks the distribution depends on where in a block
    a point falls."""
    positions = block_bootstrap_indices(20_000, rng(2), block_len=25)
    steps = np.diff(positions)
    breaks = int(np.count_nonzero((steps != 1) & (steps != -(positions.size - 1))))
    assert 20_000 / 35 < breaks < 20_000 / 18


def test_every_position_is_equally_likely_to_be_drawn() -> None:
    """Wrapping at the end is what makes this true. Under truncation the last
    bars appear less often than the first, and the resample is quietly biased
    towards early history."""
    counts = np.zeros(200, dtype="int64")
    for seed in range(60):
        counts += np.bincount(block_bootstrap_indices(200, rng(seed), block_len=10), minlength=200)
    first_half, second_half = counts[:100].sum(), counts[100:].sum()
    assert abs(first_half - second_half) < 0.15 * counts.sum()


def test_an_empty_series_cannot_be_resampled() -> None:
    with pytest.raises(ConfigError, match="cannot resample an empty series"):
        block_bootstrap_indices(0, rng(), block_len=10)


def test_a_zero_block_length_is_refused() -> None:
    with pytest.raises(ConfigError, match="at least one bar"):
        block_bootstrap_indices(100, rng(), block_len=0)


# ---------------------------------------------------------------------------
# permuted markets
# ---------------------------------------------------------------------------
def test_a_permuted_market_keeps_its_marginal_moments() -> None:
    """Section 20's stated criterion. The permutation changes the *history*, not
    the character of the market — otherwise the p-value would be a statement
    about a different asset."""
    original = bars(seed=3, n=2_000)
    returns = np.diff(np.log(original.close))

    means, deviations = [], []
    for seed in range(20):
        permuted = permute_bars(original, rng(seed), block_len=24)
        permuted_returns = np.diff(np.log(permuted.close))
        means.append(float(permuted_returns.mean()))
        deviations.append(float(permuted_returns.std(ddof=1)))

    assert float(np.mean(means)) == pytest.approx(float(returns.mean()), abs=2e-4)
    assert float(np.mean(deviations)) == pytest.approx(float(returns.std(ddof=1)), rel=0.1)


def test_a_permuted_market_is_a_different_history() -> None:
    original = bars(seed=3)
    permuted = permute_bars(original, rng(0), block_len=24)
    assert permuted.n_bars == original.n_bars
    assert not np.allclose(permuted.close, original.close)
    assert float(permuted.close[0]) == pytest.approx(float(original.close[0]))


def test_a_permuted_bar_keeps_its_own_geometry() -> None:
    """``high >= max(open, close) >= min(open, close) >= low``. A permutation
    that broke these would be rejected by the validation of section 7.3 rather
    than tested against."""
    permuted = permute_bars(bars(seed=4), rng(1), block_len=24)
    highs, lows = permuted.high, permuted.low
    opens, closes = permuted.open, permuted.close
    assert np.all(highs >= np.maximum(opens, closes) - 1e-9)
    assert np.all(lows <= np.minimum(opens, closes) + 1e-9)
    assert np.all(lows > 0.0)


def test_volume_and_gap_flags_are_carried_across_unpermuted() -> None:
    """They are properties of the venue rather than of the price path; permuting
    them would change two things at once and make the p-value a statement about
    neither."""
    original = bars(seed=4)
    permuted = permute_bars(original, rng(1), block_len=24)
    assert np.array_equal(permuted.volume, original.volume)
    assert np.array_equal(permuted.is_gap_filled, original.is_gap_filled)
    assert np.array_equal(permuted.ts_open, original.ts_open)


def test_permutation_is_deterministic() -> None:
    original = bars(seed=5)
    assert np.array_equal(
        permute_bars(original, rng(7), block_len=24).close,
        permute_bars(original, rng(7), block_len=24).close,
    )


def test_a_series_of_one_bar_cannot_be_permuted() -> None:
    with pytest.raises(ConfigError, match="at least two bars"):
        permute_bars(bars(n=1), rng(), block_len=10)


# ---------------------------------------------------------------------------
# the p-value
# ---------------------------------------------------------------------------
def test_the_p_value_is_never_zero() -> None:
    """``(1 + beaten) / (1 + n)``. Reporting 0 would claim more certainty than a
    finite number of permutations can supply."""
    settings = PermutationSettings(n_market_permutations=10)
    report = market_permutation_test(bars(), 99.0, lambda frame: -1.0, settings, seed=0)
    assert report.p_value == pytest.approx(1 / 11)
    assert report.p_value > 0.0


def test_a_result_no_permutation_beats_is_as_significant_as_the_count_allows() -> None:
    settings = PermutationSettings(n_market_permutations=199)
    report = market_permutation_test(bars(), 99.0, lambda frame: -1.0, settings, seed=0)
    assert report.p_value == pytest.approx(1 / 200)
    assert report.beaten_by == 0


def test_a_result_every_permutation_beats_is_not_significant_at_all() -> None:
    settings = PermutationSettings(n_market_permutations=10)
    report = market_permutation_test(bars(), -99.0, lambda frame: 1.0, settings, seed=0)
    assert report.p_value == pytest.approx(1.0)


def test_an_unmeasured_result_cannot_beat_anything() -> None:
    settings = PermutationSettings(n_market_permutations=5)
    report = market_permutation_test(bars(), None, lambda frame: 1.0, settings, seed=0)
    assert report.p_value == 1.0
    assert report.observed is None


def test_a_permutation_the_strategy_failed_on_is_counted_not_dropped() -> None:
    """Silently shrinking the denominator would make the p-value look more
    significant the more often the strategy failed."""
    settings = PermutationSettings(n_market_permutations=10)
    report = market_permutation_test(bars(), 1.0, lambda frame: None, settings, seed=0)
    assert report.n_failed == 10
    assert report.n_permutations == 10
    assert report.p_value == pytest.approx(1 / 11)


def test_the_report_records_what_it_measured() -> None:
    settings = PermutationSettings(n_market_permutations=8)
    document = market_permutation_test(bars(), 0.5, lambda frame: 0.1, settings, seed=0).as_dict()
    assert document["n_permutations"] == 8
    assert document["observed_sortino"] == 0.5
    assert document["beaten_by"] == 0


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_random_entry_p_values_are_uniform() -> None:
    """Section 22's stated criterion, and the one that makes this a test rather
    than a number.

    ``random_entry`` draws from ``ctx.rng`` and never looks at a price, so under
    the null its Sortino ratio on a permuted market is a draw from the same
    distribution as on the real one. Non-uniform p-values here would mean the
    permutation is changing something the strategy can see.
    """
    source = (BASELINES / "random_entry.py").read_text(encoding="utf-8")
    namespace: dict[str, Any] = {}
    exec(compile(source, "random_entry.py", "exec"), namespace)
    strategy_class = namespace["STRATEGY"]

    settings = PermutationSettings(n_market_permutations=60, block_len_bars=24)
    config = zero_cost_config(bars_per_year=8_760)

    p_values = []
    for seed in range(50):
        market = bars(seed=1_000 + seed)

        def sortino(frame: BarFrame, run_seed: int = seed) -> float | None:
            result = SimpleBarEngine().run(
                strategy_class(), frame, {}, config.model_copy(update={"seed": run_seed})
            )
            return compute_metrics(result).sortino

        p_values.append(
            market_permutation_test(market, sortino(market), sortino, settings, seed=seed).p_value
        )

    assert len(p_values) == 50
    assert stats.kstest(p_values, "uniform").pvalue > 0.01


# ---------------------------------------------------------------------------
# the trade shuffle
# ---------------------------------------------------------------------------
def test_the_shuffle_reports_a_worse_drawdown_than_the_median() -> None:
    """``mdd_p95`` is what section 20's paper report compares a realised drawdown
    against, so it must sit above the typical ordering rather than at it."""
    ledger = [trade(i, 0.02) for i in range(1, 21)] + [trade(i, -0.03) for i in range(21, 31)]
    report = trade_shuffle_test(ledger, PermutationSettings(n_trade_shuffles=400), seed=0)
    assert report.mdd_p95 is not None and report.mdd_median is not None
    assert report.mdd_p95 > report.mdd_median > 0.0
    assert 0.0 <= report.mdd_p95 <= 1.0


def test_a_lucky_ordering_says_so() -> None:
    """The same trades in a different sequence produce a different path; a
    realised drawdown better than the median ordering was partly luck."""
    ledger = [trade(i, 0.02) for i in range(1, 21)] + [trade(i, -0.03) for i in range(21, 31)]
    report = trade_shuffle_test(
        ledger, PermutationSettings(n_trade_shuffles=200), seed=0, observed_max_drawdown=0.001
    )
    assert report.was_lucky

    unlucky = trade_shuffle_test(
        ledger, PermutationSettings(n_trade_shuffles=200), seed=0, observed_max_drawdown=0.99
    )
    assert not unlucky.was_lucky


def test_returns_are_compounded_not_summed() -> None:
    """A 20 % loss after a 20 % gain does not leave the account where it started,
    and a drawdown on a summed path would understate exactly the sequences that
    matter."""
    ledger = [trade(1, 0.5), trade(2, -0.5)]
    report = trade_shuffle_test(ledger, PermutationSettings(n_trade_shuffles=50), seed=0)
    assert report.mdd_p95 is not None
    assert report.mdd_p95 == pytest.approx(0.5, abs=1e-9)


def test_shuffling_is_deterministic() -> None:
    ledger = [trade(i, 0.01 * (-1) ** i) for i in range(1, 15)]
    settings = PermutationSettings(n_trade_shuffles=100)
    assert (
        trade_shuffle_test(ledger, settings, seed=3).mdd_p95
        == trade_shuffle_test(ledger, settings, seed=3).mdd_p95
    )


def test_an_empty_ledger_has_no_drawdown_distribution() -> None:
    report = trade_shuffle_test([], PermutationSettings(), seed=0)
    assert report.mdd_p95 is None
    assert report.n_shuffles == 0
