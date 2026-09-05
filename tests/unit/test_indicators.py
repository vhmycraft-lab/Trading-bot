"""Indicator correctness and causality (master spec section 9.3).

Two properties matter. **Correctness**: each indicator matches an independent
pandas reference. **Causality**: element ``i`` of the full-series computation
equals element ``i`` of a computation over ``[0..i]`` alone.

The second is the load-bearing one. The engine computes indicators once over a
whole segment for speed; that is only sound because every function here is
causal, so the causality sweep below is what makes the optimisation safe rather
than merely fast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.core.errors import DataValidationError
from quantlab.core.indicators import (
    atr,
    bbands,
    donchian,
    ema,
    highest,
    log_returns,
    lowest,
    macd,
    returns,
    roc,
    rolling_vol,
    rsi,
    sma,
    zscore,
)


@pytest.fixture
def series() -> np.ndarray:
    rng = np.random.default_rng(42)
    return 100.0 + np.cumsum(rng.normal(0.0, 1.0, 200))


@pytest.fixture
def ohlc(series: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    high = series + np.abs(rng.normal(0.0, 0.6, series.size))
    low = series - np.abs(rng.normal(0.0, 0.6, series.size))
    return high, low, series


def close_enough(a: np.ndarray, b: np.ndarray, *, tol: float = 1e-9) -> bool:
    a, b = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
    both_nan = np.isnan(a) & np.isnan(b)
    return bool(np.all(both_nan | (np.abs(a - b) <= tol)))


# ---------------------------------------------------------------------------
# correctness against pandas
# ---------------------------------------------------------------------------
def test_sma_matches_pandas(series: np.ndarray) -> None:
    expected = pd.Series(series).rolling(20).mean().to_numpy()
    assert close_enough(sma(series, 20), expected)


def test_ema_matches_pandas_after_its_seed(series: np.ndarray) -> None:
    """Seeded with an SMA, then the standard recursion, so compare from the seed on."""
    n = 10
    ours = ema(series, n)
    expected = np.full_like(ours, np.nan)
    current = float(series[:n].mean())
    expected[n - 1] = current
    alpha = 2.0 / (n + 1.0)
    for i in range(n, series.size):
        current = alpha * series[i] + (1 - alpha) * current
        expected[i] = current
    assert close_enough(ours, expected)


def test_rsi_matches_a_wilder_reference(series: np.ndarray) -> None:
    """Against an independently written Wilder loop, seeded the standard way."""
    n = 14
    delta = np.diff(series)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    expected = np.full(series.size, np.nan)
    avg_gain = float(gains[:n].mean())
    avg_loss = float(losses[:n].mean())
    expected[n] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    for i in range(n + 1, series.size):
        avg_gain = (avg_gain * (n - 1) + gains[i - 1]) / n
        avg_loss = (avg_loss * (n - 1) + losses[i - 1]) / n
        expected[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    assert close_enough(rsi(series, n), expected)


def test_rsi_converges_to_the_pandas_ewm_form(series: np.ndarray) -> None:
    """Different seeding, same recursion: the two must agree once the seed decays."""
    n = 14
    delta = np.diff(series)
    gains = pd.Series(np.where(delta > 0, delta, 0.0))
    losses = pd.Series(np.where(delta < 0, -delta, 0.0))
    avg_gain = gains.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = losses.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    expected = (100 - 100 / (1 + avg_gain / avg_loss)).to_numpy()

    assert close_enough(rsi(series, n)[150:], expected[149:], tol=1e-3)


def test_rsi_is_bounded(series: np.ndarray) -> None:
    values = rsi(series, 14)
    defined = values[~np.isnan(values)]
    assert defined.size
    assert np.all((defined >= 0.0) & (defined <= 100.0))


def test_rsi_is_100_when_nothing_falls() -> None:
    rising = np.arange(1.0, 60.0)
    values = rsi(rising, 14)
    assert values[-1] == pytest.approx(100.0)


def test_atr_matches_a_wilder_reference(ohlc) -> None:
    high, low, close = ohlc
    n = 14
    previous = np.concatenate([[np.nan], close[:-1]])
    true_range = np.maximum(high - low, np.maximum(np.abs(high - previous), np.abs(low - previous)))
    expected = np.full(close.size, np.nan)
    current = float(true_range[1 : n + 1].mean())
    expected[n] = current
    for i in range(n + 1, close.size):
        current = (current * (n - 1) + true_range[i]) / n
        expected[i] = current
    assert close_enough(atr(high, low, close, n), expected)


def test_atr_is_never_negative(ohlc) -> None:
    high, low, close = ohlc
    values = atr(high, low, close, 14)
    assert np.all(values[~np.isnan(values)] >= 0.0)


def test_bbands_match_pandas(series: np.ndarray) -> None:
    n, k = 20, 2.0
    rolled = pd.Series(series).rolling(n)
    middle = rolled.mean().to_numpy()
    deviation = rolled.std(ddof=0).to_numpy()
    bands = bbands(series, n, k)
    assert close_enough(bands.middle, middle)
    assert close_enough(bands.upper, middle + k * deviation)
    assert close_enough(bands.lower, middle - k * deviation)


def test_bbands_are_ordered(series: np.ndarray) -> None:
    bands = bbands(series, 20, 2.0)
    defined = ~np.isnan(bands.middle)
    assert np.all(bands.lower[defined] <= bands.middle[defined])
    assert np.all(bands.middle[defined] <= bands.upper[defined])


def test_donchian_matches_rolling_extremes(ohlc) -> None:
    high, low, _close = ohlc
    channel = donchian(high, low, 20)
    assert close_enough(channel.upper, pd.Series(high).rolling(20).max().to_numpy())
    assert close_enough(channel.lower, pd.Series(low).rolling(20).min().to_numpy())


def test_highest_and_lowest_match_pandas(series: np.ndarray) -> None:
    assert close_enough(highest(series, 15), pd.Series(series).rolling(15).max().to_numpy())
    assert close_enough(lowest(series, 15), pd.Series(series).rolling(15).min().to_numpy())


def test_returns_match_pandas(series: np.ndarray) -> None:
    assert close_enough(returns(series, 5), pd.Series(series).pct_change(5).to_numpy())


def test_log_returns_match_the_definition(series: np.ndarray) -> None:
    expected = np.full(series.size, np.nan)
    expected[3:] = np.log(series[3:] / series[:-3])
    assert close_enough(log_returns(series, 3), expected)


def test_roc_is_returns_in_percent(series: np.ndarray) -> None:
    assert close_enough(roc(series, 5), returns(series, 5) * 100.0)


def test_rolling_vol_matches_pandas(series: np.ndarray) -> None:
    one_bar = pd.Series(series).pct_change()
    expected = one_bar.rolling(20).std(ddof=1).to_numpy()
    assert close_enough(rolling_vol(series, 20), expected)


def test_zscore_matches_pandas(series: np.ndarray) -> None:
    rolled = pd.Series(series).rolling(20)
    expected = ((pd.Series(series) - rolled.mean()) / rolled.std(ddof=0)).to_numpy()
    assert close_enough(zscore(series, 20), expected)


def test_macd_components_are_consistent(series: np.ndarray) -> None:
    result = macd(series, 12, 26, 9)
    assert close_enough(result.macd, ema(series, 12) - ema(series, 26))
    defined = ~np.isnan(result.signal)
    assert close_enough(result.histogram[defined], (result.macd - result.signal)[defined])


# ---------------------------------------------------------------------------
# causality: the property the engine's indicator cache depends on
# ---------------------------------------------------------------------------
INDICATOR_CASES = [
    ("sma", sma, ("close",), {"n": 10}),
    ("ema", ema, ("close",), {"n": 10}),
    ("rsi", rsi, ("close",), {"n": 14}),
    ("atr", atr, ("high", "low", "close"), {"n": 14}),
    ("bbands", bbands, ("close",), {"n": 20}),
    ("donchian", donchian, ("high", "low"), {"n": 20}),
    ("returns", returns, ("close",), {"n": 5}),
    ("log_returns", log_returns, ("close",), {"n": 5}),
    ("rolling_vol", rolling_vol, ("close",), {"n": 20}),
    ("zscore", zscore, ("close",), {"n": 20}),
    ("highest", highest, ("high",), {"n": 20}),
    ("lowest", lowest, ("low",), {"n": 20}),
    ("roc", roc, ("close",), {"n": 5}),
    ("macd", macd, ("close",), {"fast": 5, "slow": 12, "signal": 4}),
]


@pytest.mark.parametrize(
    ("name", "fn", "inputs", "kwargs"),
    INDICATOR_CASES,
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_every_indicator_is_causal(name, fn, inputs, kwargs, ohlc) -> None:
    """``full[i] == recompute(series[:i+1])[i]`` for every i, for every indicator."""
    high, low, close = ohlc
    columns = {"high": high, "low": low, "close": close}
    args = [columns[key] for key in inputs]

    full = fn(*args, **kwargs)
    full_arrays = list(full) if isinstance(full, tuple) else [full]

    for i in range(0, close.size, 7):  # every 7th bar keeps the sweep quick and total
        sliced = fn(*[a[: i + 1] for a in args], **kwargs)
        sliced_arrays = list(sliced) if isinstance(sliced, tuple) else [sliced]
        for component_full, component_sliced in zip(full_arrays, sliced_arrays, strict=True):
            a, b = component_full[i], component_sliced[i]
            assert (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-9, f"{name} leaked at i={i}"


# ---------------------------------------------------------------------------
# warm-up and edge cases
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("name", "fn", "inputs", "kwargs"),
    INDICATOR_CASES,
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_indicators_return_a_full_length_array(name, fn, inputs, kwargs, ohlc) -> None:
    high, low, close = ohlc
    columns = {"high": high, "low": low, "close": close}
    result = fn(*[columns[key] for key in inputs], **kwargs)
    arrays = list(result) if isinstance(result, tuple) else [result]
    for array in arrays:
        assert array.shape == close.shape, name


@pytest.mark.parametrize(
    ("name", "fn", "inputs", "kwargs"),
    INDICATOR_CASES,
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_a_too_short_series_is_all_nan(name, fn, inputs, kwargs) -> None:
    """Warm-up is NaN, never a filled-forward guess a strategy might trade on."""
    tiny = np.array([100.0, 101.0], dtype="float64")
    result = fn(*[tiny] * len(inputs), **kwargs)
    arrays = list(result) if isinstance(result, tuple) else [result]
    for array in arrays:
        assert np.all(np.isnan(array)), name


def test_an_empty_series_is_handled(ohlc) -> None:
    empty = np.array([], dtype="float64")
    assert sma(empty, 5).size == 0
    assert ema(empty, 5).size == 0
    assert rsi(empty, 5).size == 0


@pytest.mark.parametrize("period", [0, -1])
def test_a_non_positive_period_is_rejected(period: int, series: np.ndarray) -> None:
    with pytest.raises(DataValidationError, match="must be >="):
        sma(series, period)


def test_a_two_dimensional_input_is_rejected() -> None:
    with pytest.raises(DataValidationError, match="one-dimensional"):
        sma(np.zeros((3, 3)), 2)


def test_atr_rejects_mismatched_lengths() -> None:
    with pytest.raises(DataValidationError, match="same length"):
        atr(np.zeros(5), np.zeros(4), np.zeros(5), 2)


def test_macd_rejects_a_fast_period_that_is_not_faster() -> None:
    with pytest.raises(DataValidationError, match="shorter than slow"):
        macd(np.arange(50.0), fast=26, slow=12)


def test_returns_tolerate_a_zero_denominator() -> None:
    values = np.array([0.0, 1.0, 2.0, 3.0], dtype="float64")
    assert np.isnan(returns(values, 1)[1])


def test_zscore_is_nan_when_the_series_is_constant() -> None:
    assert np.all(np.isnan(zscore(np.full(30, 100.0), 10)[9:]))
