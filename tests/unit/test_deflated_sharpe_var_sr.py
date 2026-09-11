"""``var_sr`` must be across-trial dispersion, not within-run return variance (ADR 0010).

The two quantities differ in *units*: one is a variance of Sharpe ratios, the
other a variance of returns. A test that pins the number alone would pass again
the moment someone reintroduced the substitution with a rescaled constant, so
these tests pin the unit instead, through two invariances that only the
across-trial reading satisfies:

* **Scale invariance.** Multiplying the validation run's returns by a constant
  changes the variance of those returns by the square of it, and changes the run's
  Sharpe ratio not at all. ``SR0`` is a property of the *search*, so the deflated
  Sharpe must not move. Under the old stand-in it moves a great deal.
* **Trial sensitivity.** Widening the spread of the trial Sharpes — while leaving
  the validation run untouched — must raise ``SR0`` and lower the deflated Sharpe.
  Under the old stand-in nothing happens at all: the trials were never read.

Together they say the input is measured along the trial axis and no other. The
first fails if dispersion is read from the run; the second fails if it is read
from anywhere except the trials.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from quantlab.cli.validate import _deflated, _trial_sharpes
from quantlab.core.validation.deflated_sharpe import deannualise

BARS_PER_YEAR = 365 * 24 * 4


@dataclass(frozen=True)
class _Result:
    """The two attributes ``_deflated`` reads off a backtest result."""

    equity: np.ndarray
    bars_per_year: int = BARS_PER_YEAR


def _run(*, scale: float = 1.0, n: int = 4000, seed: int = 7) -> _Result:
    """An equity curve whose per-bar returns are ``scale`` times a fixed series.

    Built multiplicatively so that scaling the returns leaves the Sharpe ratio,
    the skew and the kurtosis of the return series exactly where they were.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0004, 0.004, size=n) * scale
    return _Result(equity=10_000.0 * np.cumprod(1.0 + returns))


def _trials(*, spread: float, n: int = 200, seed: int = 11) -> list[float]:
    """``n`` per-bar trial Sharpes with a controlled dispersion."""
    rng = np.random.default_rng(seed)
    return [float(v) for v in rng.normal(0.0, spread, size=n)]


# ---------------------------------------------------------------------------
# the unit
# ---------------------------------------------------------------------------
def test_rescaling_the_validation_returns_does_not_move_the_deflated_sharpe() -> None:
    """The dispersion belongs to the search, so this run's volatility cannot set it.

    Ten times the returns is ten times the volatility, the same Sharpe ratio, and
    a hundred times the return variance. If ``var_sr`` were that variance, ``SR0``
    would be ten times higher and the deflated Sharpe would collapse.
    """
    trials = _trials(spread=0.02)
    quiet = _deflated(_run(scale=1.0), 256, trials)
    loud = _deflated(_run(scale=10.0), 256, trials)

    assert quiet is not None and loud is not None
    assert loud == pytest.approx(quiet, rel=1e-9), (
        "the deflated Sharpe moved when only the run's return scale changed; "
        "var_sr is being read from the run rather than from the trials"
    )


def test_widening_the_trial_spread_lowers_the_deflated_sharpe() -> None:
    """The search is the only thing that changes here, and it must be felt.

    A search whose trials scored alike offered little to select from; one whose
    trials ranged widely offered a great deal, and the winner has to clear a
    higher bar for it.
    """
    result = _run()
    tight = _deflated(result, 256, _trials(spread=0.002))
    wide = _deflated(result, 256, _trials(spread=0.05))

    assert tight is not None and wide is not None
    assert wide < tight, (
        "widening the trial Sharpes left the deflated Sharpe alone; "
        "the trials are not reaching var_sr at all"
    )


def test_the_stand_in_that_shipped_is_not_a_scaled_version_of_the_right_answer() -> None:
    """Why the two invariances above are both needed.

    The old code passed ``np.var(returns, ddof=1)``. This asserts the substitution
    is not merely mis-scaled but differently *shaped*: fixing it with any constant
    factor is impossible, because the two quantities respond to different inputs.
    On the first real campaign it was sixty times too small; the factor is not a
    constant of the code, it is a property of whatever search happened to run.
    """
    trials = _trials(spread=0.02)
    result = _run()
    returns = np.diff(result.equity) / result.equity[:-1]

    within_run = float(np.var(returns, ddof=1))
    across_trials = float(np.var(np.asarray(trials), ddof=1))

    assert within_run < across_trials, "fixture no longer reproduces the direction of the bug"
    # And the ratio is a property of the fixture, not a constant to divide out:
    louder = np.diff(_run(scale=10.0).equity) / _run(scale=10.0).equity[:-1]
    assert float(np.var(louder, ddof=1)) == pytest.approx(within_run * 100.0, rel=1e-6)
    assert across_trials == pytest.approx(across_trials)  # unmoved by the same change


# ---------------------------------------------------------------------------
# undefined rather than guessed
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("trials", [[], [0.01]])
def test_a_search_with_under_two_trials_leaves_check_one_unmeasured(trials: list[float]) -> None:
    """No dispersion to measure means no benchmark, and the pipeline says so.

    Charging nothing *and saying nothing* is what the old stand-in did. Returning
    ``None`` charges nothing and is listed under ``unmeasured``, which prints
    "this verdict rests on partial evidence".
    """
    assert _deflated(_run(), 256, trials) is None


# ---------------------------------------------------------------------------
# the annualisation the unit trap lives in
# ---------------------------------------------------------------------------
class _FakeStore:
    def __init__(self, annualised: list[float]) -> None:
        self._annualised = annualised

    def trial_sharpes_for_family(self, family_id: str) -> list[float]:
        return list(self._annualised)


def test_trial_sharpes_are_deannualised_before_they_reach_the_variance() -> None:
    """The store keeps them annualised; the deflation works per bar.

    Mixing the two scales raises every trial Sharpe by ``sqrt(bars_per_year)`` and
    the variance by ``bars_per_year`` — inflating ``SR0`` by about 187x here, with
    no exception raised.
    """
    annualised = [2.0, -1.0, 0.5]
    got = _trial_sharpes(_FakeStore(annualised), "fam", BARS_PER_YEAR)

    assert got == pytest.approx([deannualise(v, BARS_PER_YEAR) for v in annualised])
    assert got == pytest.approx([v / math.sqrt(BARS_PER_YEAR) for v in annualised])


def test_a_nonsense_bar_count_yields_no_trials_rather_than_a_division() -> None:
    assert _trial_sharpes(_FakeStore([2.0, -1.0]), "fam", 0) == []


# ---------------------------------------------------------------------------
# the dispersion is bounded from above, not point-estimated (ADR 0011)
# ---------------------------------------------------------------------------
def test_a_small_pool_produces_a_harsher_benchmark_not_an_absent_one() -> None:
    """The reason this is a bound and not a minimum pool size.

    A hard floor high enough to be comfortable — twenty, thirty — would leave
    check 1 permanently unmeasured on realistic pools, and "permanently
    unmeasured" carries exactly as much information as "permanently failing".
    The bound has no cliff: it errs toward rejecting and relaxes on its own as the
    pool grows.
    """
    from quantlab.core.validation.deflated_sharpe import (
        sharpe_variance,
        sharpe_variance_upper_bound,
    )

    rng = np.random.default_rng(3)
    sigma = 0.02
    inflation = []
    for n in (5, 10, 30, 100, 400):
        sample = list(rng.normal(0.0, sigma, size=n))
        point = sharpe_variance(sample)
        bounded = sharpe_variance_upper_bound(sample)
        assert bounded > point, "the upper bound must exceed the point estimate"
        inflation.append(math.sqrt(bounded / point))

    assert inflation == sorted(inflation, reverse=True), (
        f"the penalty for a small pool must shrink monotonically as it grows: {inflation}"
    )
    assert inflation[0] > 2.0, "a five-trial pool should be charged substantially"
    assert inflation[-1] < 1.10, "a four-hundred-trial pool should be charged almost nothing"


def test_the_bound_makes_the_deflation_stricter_never_gentler() -> None:
    """Direction, asserted rather than argued — the failure mode of ADR 0010."""
    from quantlab.core.validation.deflated_sharpe import deflated_sharpe as dsr
    from quantlab.core.validation.deflated_sharpe import (
        sharpe_variance,
        sharpe_variance_upper_bound,
    )

    trials = _trials(spread=0.02, n=8)
    point = dsr(
        deannualise(2.0, BARS_PER_YEAR),
        n_trials=8,
        var_sr=sharpe_variance(trials),
        n_periods=15_000,
    )
    bounded = dsr(
        deannualise(2.0, BARS_PER_YEAR),
        n_trials=8,
        var_sr=sharpe_variance_upper_bound(trials),
        n_periods=15_000,
    )
    assert bounded < point


def test_two_trials_is_the_hard_floor_and_it_is_arithmetic_not_policy() -> None:
    """Below two there is no variance to bound, so there is nothing to be
    conservative *with*. One trial reports unmeasured (ADR 0008), never a
    fabricated dispersion and never a fall back to the unfiltered population."""
    from quantlab.core.errors import ConfigError
    from quantlab.core.validation.deflated_sharpe import sharpe_variance_upper_bound

    with pytest.raises(ConfigError):
        sharpe_variance_upper_bound([0.01])
    assert _deflated(_run(), 256, [0.01]) is None


def test_a_smaller_alpha_is_more_conservative() -> None:
    """The knob points the way its name says, so a future operator turning it
    down cannot accidentally loosen the check."""
    from quantlab.core.validation.deflated_sharpe import sharpe_variance_upper_bound

    trials = _trials(spread=0.02, n=12)
    assert sharpe_variance_upper_bound(trials, alpha=0.01) > sharpe_variance_upper_bound(
        trials, alpha=0.10
    )
