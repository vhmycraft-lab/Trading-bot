"""Causal indicator library (master spec section 9.3).

Every function returns a full-length ``float64`` array in which element ``i``
depends **only** on inputs at indices ``<= i``.  That property is what lets the
engine compute an indicator once over a whole segment and hand a strategy the
value at bar ``i`` without any risk of look-ahead: a causal array indexed at
``i`` is by definition the same number a recomputation over ``bars[0..i]`` would
have produced.  ``tests/unit/test_indicators.py`` proves that equality at every
bar for every indicator here, which is what makes the optimisation safe rather
than merely plausible.

Warm-up positions are ``NaN`` rather than a filled-forward guess: a strategy that
acts on an indicator before it has enough history should fail a comparison, not
silently trade on a fabricated value.
"""

from __future__ import annotations

from typing import Final, NamedTuple

import numpy as np

from quantlab.core.errors import DataValidationError

__all__ = [
    "BBands",
    "Donchian",
    "Macd",
    "atr",
    "bbands",
    "donchian",
    "ema",
    "highest",
    "log_returns",
    "lowest",
    "macd",
    "returns",
    "roc",
    "rolling_vol",
    "rsi",
    "sma",
    "zscore",
]

#: Denominators below this are treated as zero.
_EPS: Final[float] = 1e-12


class BBands(NamedTuple):
    lower: np.ndarray
    middle: np.ndarray
    upper: np.ndarray


class Donchian(NamedTuple):
    lower: np.ndarray
    upper: np.ndarray


class Macd(NamedTuple):
    macd: np.ndarray
    signal: np.ndarray
    histogram: np.ndarray


def _as_float(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype="float64")
    if array.ndim != 1:
        raise DataValidationError("indicator input must be one-dimensional", ndim=array.ndim)
    return array


def _check_period(n: int, *, name: str = "n", minimum: int = 1) -> int:
    period = int(n)
    if period < minimum:
        raise DataValidationError(f"{name} must be >= {minimum}", **{name: period})
    return period


def _empty_like(values: np.ndarray) -> np.ndarray:
    return np.full(values.shape, np.nan, dtype="float64")


def _rolling_view(values: np.ndarray, n: int) -> np.ndarray:
    """Return a (len - n + 1, n) sliding window view; empty when too short."""
    if values.size < n:
        return np.empty((0, n), dtype="float64")
    return np.lib.stride_tricks.sliding_window_view(values, n)


# ---------------------------------------------------------------------------
# moving averages
# ---------------------------------------------------------------------------
def sma(values: np.ndarray, n: int) -> np.ndarray:
    """Simple moving average over ``n`` bars."""
    array = _as_float(values)
    period = _check_period(n)
    out = _empty_like(array)
    if array.size >= period:
        windows = _rolling_view(array, period)
        out[period - 1 :] = windows.mean(axis=1)
    return out


def ema(values: np.ndarray, n: int) -> np.ndarray:
    """Exponential moving average, seeded with the first complete SMA.

    ``alpha = 2 / (n + 1)``.  Seeding with an SMA rather than the first value
    keeps the early output from being dominated by a single observation.
    """
    array = _as_float(values)
    period = _check_period(n)
    out = _empty_like(array)
    if array.size < period:
        return out

    alpha = 2.0 / (period + 1.0)
    current = float(array[:period].mean())
    out[period - 1] = current
    for i in range(period, array.size):
        current = alpha * float(array[i]) + (1.0 - alpha) * current
        out[i] = current
    return out


def _wilder(values: np.ndarray, n: int, *, seed: float) -> np.ndarray:
    """Wilder's smoothing, seeded at index ``n - 1`` with ``seed``."""
    out = np.full(values.shape, np.nan, dtype="float64")
    current = seed
    out[n - 1] = current
    for i in range(n, values.size):
        current = (current * (n - 1) + float(values[i])) / n
        out[i] = current
    return out


# ---------------------------------------------------------------------------
# oscillators
# ---------------------------------------------------------------------------
def rsi(values: np.ndarray, n: int) -> np.ndarray:
    """Relative strength index with Wilder's smoothing.

    Returns 100.0 when there are no losses in the window, which is the limit of
    ``100 - 100/(1 + RS)`` as the average loss goes to zero.
    """
    array = _as_float(values)
    period = _check_period(n)
    out = _empty_like(array)
    if array.size <= period:
        return out

    delta = np.diff(array)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    avg_gain = float(gains[:period].mean())
    avg_loss = float(losses[:period].mean())
    for i in range(period, array.size):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + float(gains[i - 1])) / period
            avg_loss = (avg_loss * (period - 1) + float(losses[i - 1])) / period
        if avg_loss <= _EPS:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - 100.0 / (1.0 + rs)
    return out


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    """Average true range with Wilder's smoothing."""
    highs, lows, closes = _as_float(high), _as_float(low), _as_float(close)
    if not highs.size == lows.size == closes.size:
        raise DataValidationError("atr inputs must be the same length")
    period = _check_period(n)
    if highs.size < period + 1:
        return _empty_like(closes)

    previous_close = np.concatenate([[np.nan], closes[:-1]])
    true_range = np.maximum(
        highs - lows,
        np.maximum(
            np.abs(highs - previous_close),
            np.abs(lows - previous_close),
        ),
    )
    true_range[0] = highs[0] - lows[0]
    # Wilder seeds at index n with the mean of true ranges 1..n, so the first
    # value never depends on the incomplete range at index 0.
    out = _empty_like(closes)
    current = float(true_range[1 : period + 1].mean())
    out[period] = current
    for i in range(period + 1, closes.size):
        current = (current * (period - 1) + float(true_range[i])) / period
        out[i] = current
    return out


# ---------------------------------------------------------------------------
# bands and channels
# ---------------------------------------------------------------------------
def bbands(values: np.ndarray, n: int, k: float = 2.0) -> BBands:
    """Bollinger bands: ``sma(n) +/- k * population std(n)``."""
    array = _as_float(values)
    period = _check_period(n, minimum=2)
    middle = sma(array, period)
    deviation = _empty_like(array)
    if array.size >= period:
        deviation[period - 1 :] = _rolling_view(array, period).std(axis=1, ddof=0)
    return BBands(middle - float(k) * deviation, middle, middle + float(k) * deviation)


def donchian(high: np.ndarray, low: np.ndarray, n: int) -> Donchian:
    """Donchian channel over the last ``n`` bars, inclusive of the current bar."""
    return Donchian(lowest(low, n), highest(high, n))


def highest(values: np.ndarray, n: int) -> np.ndarray:
    """Rolling maximum over ``n`` bars, inclusive of the current bar."""
    array = _as_float(values)
    period = _check_period(n)
    out = _empty_like(array)
    if array.size >= period:
        out[period - 1 :] = _rolling_view(array, period).max(axis=1)
    return out


def lowest(values: np.ndarray, n: int) -> np.ndarray:
    """Rolling minimum over ``n`` bars, inclusive of the current bar."""
    array = _as_float(values)
    period = _check_period(n)
    out = _empty_like(array)
    if array.size >= period:
        out[period - 1 :] = _rolling_view(array, period).min(axis=1)
    return out


# ---------------------------------------------------------------------------
# returns and dispersion
# ---------------------------------------------------------------------------
def returns(values: np.ndarray, n: int = 1) -> np.ndarray:
    """Simple return over ``n`` bars: ``x[i] / x[i-n] - 1``."""
    array = _as_float(values)
    period = _check_period(n)
    out = _empty_like(array)
    if array.size > period:
        previous = array[:-period]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[period:] = np.where(
                np.abs(previous) > _EPS, array[period:] / previous - 1.0, np.nan
            )
    return out


def log_returns(values: np.ndarray, n: int = 1) -> np.ndarray:
    """Log return over ``n`` bars: ``log(x[i] / x[i-n])``."""
    array = _as_float(values)
    period = _check_period(n)
    out = _empty_like(array)
    if array.size > period:
        previous = array[:-period]
        current = array[period:]
        valid = (previous > _EPS) & (current > _EPS)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[period:] = np.where(valid, np.log(np.where(valid, current / previous, 1.0)), np.nan)
    return out


def rolling_vol(values: np.ndarray, n: int) -> np.ndarray:
    """Sample standard deviation of one-bar simple returns over ``n`` bars."""
    array = _as_float(values)
    period = _check_period(n, minimum=2)
    one_bar = returns(array, 1)
    out = _empty_like(array)
    if one_bar.size >= period:
        windows = _rolling_view(one_bar, period)
        values_out = windows.std(axis=1, ddof=1)
        out[period - 1 :] = values_out
    out[:period] = np.nan  # the first return is NaN, so the first window is too
    return out


def zscore(values: np.ndarray, n: int) -> np.ndarray:
    """``(x - sma(n)) / population std(n)``; NaN where the deviation is zero."""
    array = _as_float(values)
    period = _check_period(n, minimum=2)
    middle = sma(array, period)
    deviation = _empty_like(array)
    if array.size >= period:
        deviation[period - 1 :] = _rolling_view(array, period).std(axis=1, ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(deviation) > _EPS, (array - middle) / deviation, np.nan)


def roc(values: np.ndarray, n: int) -> np.ndarray:
    """Rate of change in percent: ``returns(n) * 100``."""
    return returns(values, n) * 100.0


def macd(values: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Macd:
    """MACD line, its signal line, and their difference.

    Raises:
        DataValidationError: if ``fast`` is not shorter than ``slow``.
    """
    array = _as_float(values)
    fast_n = _check_period(fast, name="fast")
    slow_n = _check_period(slow, name="slow")
    signal_n = _check_period(signal, name="signal")
    if fast_n >= slow_n:
        raise DataValidationError(
            "macd fast period must be shorter than slow", fast=fast_n, slow=slow_n
        )

    line = ema(array, fast_n) - ema(array, slow_n)
    # The signal line is an EMA of the MACD line, which is NaN during warm-up;
    # seed it from the first defined value so the signal is causal too.
    signal_line = _empty_like(array)
    defined = np.flatnonzero(~np.isnan(line))
    if defined.size >= signal_n:
        start = int(defined[0])
        signal_line[start:] = ema(line[start:], signal_n)
    return Macd(line, signal_line, line - signal_line)
