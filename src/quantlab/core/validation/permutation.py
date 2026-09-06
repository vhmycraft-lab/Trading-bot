"""Market permutation and trade shuffling (spec section 14.4, check 3).

Two questions, both asked by destroying something and seeing whether the result
survives.

**Market permutation.** Would this strategy have looked as good on a market that
had the same statistical character but a different history? The return series is
resampled with a stationary block bootstrap — blocks of geometrically distributed
length, so the marginal distribution and the short-range autocorrelation survive
while the long-range structure a strategy might be fitting does not — and the
strategy is re-run on each. The p-value is the share of permutations that matched
or beat the real result::

    p = (1 + #(sortino_perm >= sortino_real)) / (1 + n)

The ``1 +`` on both sides is not a rounding convenience: it makes the p-value
achievable-but-never-zero, which is the honest statement when a finite number of
permutations were drawn. Reporting ``0`` would claim more certainty than 200
resamples can supply.

**Trade shuffling.** The market permutation asks whether the *edge* is real; this
asks how bad the *path* could have been. The same trades in a different order
produce a different drawdown, and ``mdd_p95`` — the 95th percentile of the
shuffled drawdowns — is what section 20 compares a live paper drawdown against.
A strategy whose realised drawdown was lucky in its ordering will say so here.

Both are seeded, so a verdict is reproducible (INV-7).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from quantlab.core.config import PermutationSettings
from quantlab.core.errors import ConfigError
from quantlab.core.metrics import drawdown_series
from quantlab.core.types import BarFrame, Trade

__all__ = [
    "PermutationReport",
    "ShuffleReport",
    "block_bootstrap_indices",
    "market_permutation_test",
    "permute_bars",
    "trade_shuffle_test",
]

_EPS: Final[float] = 1e-12


# ---------------------------------------------------------------------------
# the block bootstrap
# ---------------------------------------------------------------------------
def block_bootstrap_indices(n: int, rng: np.random.Generator, *, block_len: int) -> np.ndarray:
    """Positions for one stationary block bootstrap resample of length ``n``.

    Politis and Romano (1994). Blocks start at a uniformly drawn position and
    continue with probability ``1 - 1/block_len`` at each step, so block lengths
    are geometrically distributed with mean ``block_len``. Wrapping at the end
    keeps every position equally likely to be drawn, which a truncating variant
    does not — under truncation the last bars appear less often than the first,
    and the resample is quietly biased towards early history.

    Geometric rather than fixed lengths is what makes the resample *stationary*:
    with fixed blocks the joint distribution depends on where in a block a point
    falls, and the bootstrap inherits a periodicity the data does not have.
    """
    if n <= 0:
        raise ConfigError("cannot resample an empty series", n=n)
    if block_len < 1:
        raise ConfigError("the mean block length must be at least one bar", block_len=block_len)

    continuation = 1.0 - 1.0 / float(block_len)
    positions = np.empty(n, dtype="int64")
    current = int(rng.integers(0, n))
    for index in range(n):
        if index > 0 and rng.random() >= continuation:
            current = int(rng.integers(0, n))
        positions[index] = current
        current = (current + 1) % n
    return positions


def permute_bars(bars: BarFrame, rng: np.random.Generator, *, block_len: int) -> BarFrame:
    """A synthetic market with the same character and a different history.

    Close-to-close returns are resampled by block bootstrap and compounded into a
    new price path from the original opening price. Each bar's whole OHLC is
    scaled by the same factor as its close, so ``high >= max(open, close)`` and
    the rest of the bar's internal geometry survive — a permutation that broke
    those would be rejected by the data validation of section 7.3 rather than
    tested against.

    Volume and the gap flags are carried across unpermuted. They are properties of
    the *venue* rather than of the price path, and permuting them would change two
    things at once and make the p-value a statement about neither.
    """
    frame = bars.to_pandas()
    closes = frame["close"].to_numpy(dtype="float64")
    if closes.size < 2:
        raise ConfigError("a permutation needs at least two bars", n_bars=int(closes.size))

    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.where(closes[:-1] > _EPS, closes[1:] / closes[:-1], 1.0)
    returns = np.nan_to_num(returns, nan=1.0, posinf=1.0, neginf=1.0)

    drawn = returns[block_bootstrap_indices(returns.size, rng, block_len=block_len)]
    path = np.concatenate([[closes[0]], closes[0] * np.cumprod(drawn)])
    # The factor that turns each original bar into its synthetic counterpart.
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(np.abs(closes) > _EPS, path / closes, 1.0)

    permuted = frame.copy()
    for column in ("open", "high", "low", "close"):
        permuted[column] = frame[column].to_numpy(dtype="float64") * scale
    return BarFrame(permuted, symbol=bars.symbol, timeframe=bars.timeframe)


# ---------------------------------------------------------------------------
# the market permutation test
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PermutationReport:
    """What the market permutation found (spec section 14.4, check 3)."""

    p_value: float
    observed: float | None
    permuted: tuple[float, ...] = ()
    n_permutations: int = 0
    n_failed: int = 0

    @property
    def beaten_by(self) -> int:
        """How many permutations matched or beat the real result."""
        if self.observed is None:
            return len(self.permuted)
        return sum(1 for value in self.permuted if value >= self.observed)

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "p_value": self.p_value,
            "observed_sortino": self.observed,
            "n_permutations": self.n_permutations,
            "n_failed": self.n_failed,
            "beaten_by": self.beaten_by,
        }


def market_permutation_test(
    bars: BarFrame,
    observed_sortino: float | None,
    evaluate: Callable[[BarFrame], float | None],
    settings: PermutationSettings,
    *,
    seed: int,
) -> PermutationReport:
    """Re-run a strategy on permuted markets (spec section 14.4, check 3).

    Args:
        bars: The segment the real result came from.
        observed_sortino: The real Sortino ratio. ``None`` — never measured —
            yields a p-value of 1.0: an unmeasured result cannot beat anything.
        evaluate: Runs the strategy on one permuted market and returns its
            Sortino ratio, or ``None`` if it could not be measured. Injected, so
            this module needs no engine of its own.
        settings: ``validation.permutation``.
        seed: Fixed per section 14.4, so a verdict is reproducible (INV-7).

    Returns:
        A report whose ``p_value`` is ``(1 + beaten) / (1 + n)``. Section 14.3's
        ``G_PERM`` rejects above ``alpha``; section 14.4 charges 10 points above
        ``alpha / 2``.
    """
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    failed = 0
    for _ in range(settings.n_market_permutations):
        permuted = permute_bars(bars, rng, block_len=settings.block_len_bars)
        value = evaluate(permuted)
        if value is None:
            # A permutation the strategy could not be scored on. Counted, not
            # dropped: silently shrinking the denominator would make the p-value
            # look more significant the more often the strategy failed.
            failed += 1
            continue
        scores.append(float(value))

    if observed_sortino is None:
        return PermutationReport(
            p_value=1.0,
            observed=None,
            permuted=tuple(scores),
            n_permutations=settings.n_market_permutations,
            n_failed=failed,
        )

    beaten = sum(1 for value in scores if value >= observed_sortino)
    return PermutationReport(
        p_value=(1 + beaten) / (1 + settings.n_market_permutations),
        observed=float(observed_sortino),
        permuted=tuple(scores),
        n_permutations=settings.n_market_permutations,
        n_failed=failed,
    )


# ---------------------------------------------------------------------------
# the trade shuffle
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ShuffleReport:
    """How bad the drawdown could have been in a different order."""

    mdd_p95: float | None
    mdd_observed: float | None
    mdd_median: float | None = None
    n_shuffles: int = 0

    @property
    def was_lucky(self) -> bool:
        """True when the realised drawdown was better than the median ordering."""
        return (
            self.mdd_observed is not None
            and self.mdd_median is not None
            and self.mdd_observed < self.mdd_median
        )

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "mdd_p95": self.mdd_p95,
            "mdd_observed": self.mdd_observed,
            "mdd_median": self.mdd_median,
            "n_shuffles": self.n_shuffles,
        }


def trade_shuffle_test(
    trades: Sequence[Trade],
    settings: PermutationSettings,
    *,
    seed: int,
    initial_equity: float = 10_000.0,
    observed_max_drawdown: float | None = None,
) -> ShuffleReport:
    """The drawdown distribution over reorderings of the same trades.

    The same trades in a different sequence produce a different equity path, and
    the worst of those paths is a fairer statement of the strategy's risk than
    the one ordering history happened to deal. ``mdd_p95`` is what section 20's
    paper-trading report compares a realised drawdown against.

    Returns are compounded rather than summed, because a 20 % loss after a 20 %
    gain does not leave the account where it started, and a drawdown computed on
    a summed path would understate exactly the sequences that matter.
    """
    if not trades:
        return ShuffleReport(mdd_p95=None, mdd_observed=observed_max_drawdown, n_shuffles=0)

    returns = np.array([trade.pnl_pct for trade in trades], dtype="float64")
    rng = np.random.default_rng(seed)
    drawdowns = np.empty(settings.n_trade_shuffles, dtype="float64")
    for index in range(settings.n_trade_shuffles):
        order = rng.permutation(returns.size)
        equity = initial_equity * np.cumprod(1.0 + returns[order])
        drawdowns[index] = float(drawdown_series(np.concatenate([[initial_equity], equity])).max())

    return ShuffleReport(
        mdd_p95=float(np.percentile(drawdowns, 95)),
        mdd_observed=observed_max_drawdown,
        mdd_median=float(np.median(drawdowns)),
        n_shuffles=settings.n_trade_shuffles,
    )
