"""The deflated Sharpe ratio (master spec section 14.4, check 1).

Bailey and López de Prado (2014). A Sharpe ratio is an estimate, and an estimate
selected as the best of many trials is biased upwards by the selection itself:
run enough strategies on enough noise and one of them will look excellent. The
deflated Sharpe ratio asks the only question that matters after a search — *given
that this was the best of M trials, how likely is it that its true Sharpe ratio
is above zero?*

Two corrections, applied together:

* **Selection.** The benchmark is not zero but ``SR0``, the Sharpe ratio one
  would *expect* the best of ``M`` independent trials to reach by luck alone. It
  grows with ``M``, which is why every evaluation the platform makes is counted
  (section 14.4: ``M = evolution_run.n_evaluations + optuna trials +
  family.validation_touches``). A search that tried more things has to clear a
  higher bar, and that is the whole point.
* **Non-normality.** Returns are skewed and fat-tailed, and the standard error of
  a Sharpe ratio depends on both. Negative skew and high kurtosis widen it, which
  lowers confidence in the same point estimate.

**Units.** Every function here takes a **per-period** Sharpe ratio, because ``T``
is a count of periods. :func:`deannualise` converts the annualised figure
``MetricSet.sharpe`` carries. Passing an annualised Sharpe with a bar count would
overstate the ratio by roughly ``sqrt(bars_per_year)`` — about 94x on hourly bars
— and quietly turn every deflated Sharpe into 1.0.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

import numpy as np
from scipy import stats

from quantlab.core.errors import ConfigError

__all__ = [
    "EULER_MASCHERONI",
    "M_FORMULA_VERSION",
    "SUPERSEDED_M_FORMULA",
    "deannualise",
    "deflated_sharpe",
    "expected_max_sharpe",
    "moments",
    "probabilistic_sharpe",
    "sharpe_variance",
]

#: Which formula produced a recorded verdict's ``M``. Stamped on every
#: ``validation_verdict`` row so a stored verdict says how it was computed, rather
#: than leaving a reader to assume it matches today's code.
#:
#: ``M`` is the number of trials the deflation charges against, and understating
#: it lowers the bar a search-fitted strategy has to clear. A verdict computed
#: under an older, weaker formula is therefore *overstated*, and nothing about the
#: number itself distinguishes it from a sound one — which is why the formula is
#: recorded beside it, and why
#: :func:`~quantlab.core.validation.lockbox.require_candidate` refuses to open the
#: lockbox on a verdict not stamped with the current version.
M_FORMULA_VERSION: Final[str] = "m2-search-size"

#: What every row written before the correction carries (migration ``0004``).
#: ``m1`` computed ``M`` as ``max(1, n_bars // 100)`` — a function of the
#: validation segment's length rather than of the search — so it deflated a
#: forty-thousand-evaluation campaign exactly as gently as a single backtest.
#: Such rows are not deleted: §6 forbids rewriting a recorded verdict, and what
#: the platform once believed is itself part of the audit trail. They are marked,
#: and must be recomputed before they are trusted again.
SUPERSEDED_M_FORMULA: Final[str] = "m1-bar-count"

#: Euler-Mascheroni constant, from the expected-maximum formula of Bailey and
#: López de Prado (2014), equation 5.
EULER_MASCHERONI: Final[float] = 0.5772156649015329

_EPS: Final[float] = 1e-12


def deannualise(annualised: float, bars_per_year: int) -> float:
    """Convert an annualised Sharpe ratio to the per-period one used here.

    ``MetricSet.sharpe`` is annualised (section 10); ``T`` in the deflation
    formula is a count of *periods*. Mixing them is the one error in this module
    that produces no exception and no obviously wrong number — just a deflated
    Sharpe of 1.0 for everything.
    """
    if bars_per_year <= 0:
        raise ConfigError("bars_per_year must be positive", bars_per_year=bars_per_year)
    return float(annualised) / math.sqrt(float(bars_per_year))


def sharpe_variance(trial_sharpes: Sequence[float] | np.ndarray) -> float:
    """Variance of the Sharpe ratios a search produced.

    The dispersion of the trials is what says how much luck was available to the
    winner: a search whose trials all scored alike offered little to select from,
    and one whose trials ranged widely offered a great deal.

    Raises:
        ConfigError: fewer than two trials. One trial has no dispersion, and a
            variance of zero would make the selection correction vanish — which
            is right for a single trial and wrong as a default.
    """
    values = np.asarray(list(trial_sharpes), dtype="float64")
    if values.size < 2:
        raise ConfigError(
            "the variance of the trial Sharpe ratios needs at least two trials",
            n_trials=int(values.size),
        )
    return float(values.var(ddof=1))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """``SR0``: the Sharpe ratio the best of ``n_trials`` reaches by luck alone.

    Bailey and López de Prado (2014), equation 5::

        SR0 = sqrt(V) * ((1 - g) * Z(1 - 1/M) + g * Z(1 - 1/(M*e)))

    where ``g`` is Euler-Mascheroni, ``Z`` the inverse standard normal CDF, and
    ``V`` the variance of the trial Sharpe ratios.

    ``M = 1`` returns 0: with a single trial there was no selection, so there is
    no selection bias to remove and the deflated ratio reduces to the
    probabilistic Sharpe ratio against a benchmark of zero. The formula itself is
    undefined there — ``Z(0)`` is negative infinity — so the case is handled
    rather than left to produce a nan.
    """
    trials = int(n_trials)
    if trials < 1:
        raise ConfigError("a deflation needs at least one trial", n_trials=trials)
    if trials == 1 or var_sr <= _EPS:
        return 0.0

    scale = math.sqrt(float(var_sr))
    upper = float(stats.norm.ppf(1.0 - 1.0 / trials))
    lower = float(stats.norm.ppf(1.0 - 1.0 / (trials * math.e)))
    return scale * ((1.0 - EULER_MASCHERONI) * upper + EULER_MASCHERONI * lower)


def probabilistic_sharpe(
    sharpe: float,
    *,
    benchmark: float,
    n_periods: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """The probability that the true Sharpe ratio exceeds ``benchmark``.

    Bailey and López de Prado (2012)::

        PSR = Phi( (SR - SR*) * sqrt(T - 1) / sqrt(1 - g3*SR + (g4 - 1)/4 * SR^2) )

    ``kurtosis`` is the **non-excess** fourth moment, so a normal distribution is
    3 and the term ``(g4 - 1)/4`` reduces to ``0.5``. Passing excess kurtosis
    would understate the standard error and inflate every result.

    Returns 0.5 exactly when ``sharpe == benchmark``: the estimate carries no
    information either way, which is the honest answer and the invariant the
    tests pin.
    """
    periods = int(n_periods)
    if periods < 2:
        raise ConfigError(
            "the probabilistic Sharpe ratio needs at least two periods", n_periods=periods
        )

    variance = 1.0 - skew * sharpe + 0.25 * (kurtosis - 1.0) * sharpe * sharpe
    if variance <= _EPS:
        # The estimator's own variance has gone non-positive, which means the
        # moments supplied cannot have come from one sample. Refusing beats
        # returning a confident number derived from a square root of a negative.
        raise ConfigError(
            "the Sharpe estimator's variance is not positive; check that kurtosis is "
            "the non-excess fourth moment and that the Sharpe ratio is per-period",
            sharpe=sharpe,
            skew=skew,
            kurtosis=kurtosis,
            variance=variance,
        )

    statistic = (sharpe - benchmark) * math.sqrt(periods - 1) / math.sqrt(variance)
    return float(stats.norm.cdf(statistic))


def deflated_sharpe(
    sharpe: float,
    *,
    n_trials: int,
    var_sr: float,
    n_periods: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """The deflated Sharpe ratio (spec section 14.4, check 1).

    The probabilistic Sharpe ratio measured against ``SR0`` rather than zero, so
    the benchmark rises with the number of trials the search made.

    Args:
        sharpe: The **per-period** Sharpe ratio (see :func:`deannualise`).
        n_trials: ``M`` — every evaluation the search made, per section 14.4.
        var_sr: Variance of the trial Sharpe ratios (:func:`sharpe_variance`).
        n_periods: ``T``, in the same periods as ``sharpe``.
        skew: Skewness of the per-period returns.
        kurtosis: **Non-excess** kurtosis of the per-period returns.

    Returns:
        A probability in ``[0, 1]``. Section 14.4 charges 25 points when it falls
        below ``validation.dsr_threshold``.
    """
    benchmark = expected_max_sharpe(n_trials, var_sr)
    return probabilistic_sharpe(
        sharpe, benchmark=benchmark, n_periods=n_periods, skew=skew, kurtosis=kurtosis
    )


def moments(returns: Sequence[float] | np.ndarray) -> tuple[float, float, float, int]:
    """Per-period Sharpe, skew, non-excess kurtosis and count of a return series.

    Everything the deflation needs, computed once from one array so the four
    numbers cannot come from different samples — which is the mistake that makes
    the estimator variance go negative and the result meaningless.
    """
    values = np.asarray(list(returns), dtype="float64")
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ConfigError("moments need at least two returns", n_returns=int(values.size))

    deviation = float(values.std(ddof=1))
    sharpe = 0.0 if deviation <= _EPS else float(values.mean()) / deviation
    return (
        sharpe,
        float(stats.skew(values, bias=False)),
        float(stats.kurtosis(values, fisher=False, bias=False)),
        int(values.size),
    )
