"""Probability of backtest overfitting, via CSCV (spec section 14.4, check 2).

Bailey, Borwein, López de Prado and Zhu (2015). The deflated Sharpe ratio of
check 1 asks whether *one* result survives the size of the search. This asks a
different and more searching question about the search *procedure*: **if I pick
the best trial in-sample, how often does it turn out to be below median out of
sample?**

The method is combinatorially symmetric cross-validation. Split the period into
``n_blocks`` blocks; for each way of choosing half of them as in-sample, find the
trial that did best there, and look up where that same trial ranks over the
blocks left out. If the procedure is finding real structure, the in-sample winner
keeps winning; if it is fitting noise, its out-of-sample rank is a coin flip and
the fraction of splits where it lands below median approaches one half.

Symmetric is the operative word: every split is used with its complement, so the
estimate cannot be an artefact of which half happened to come first. The blocks
are contiguous and stay in order within each half, which preserves the
autocorrelation a shuffled resample would destroy.

The input is a matrix of per-period performance, one column per trial. Nothing
here runs a strategy or reads a segment; the caller supplies what the search
produced.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from quantlab.core.config import PboSettings
from quantlab.core.errors import ConfigError

__all__ = ["PboReport", "logit_ranks", "probability_of_backtest_overfitting"]

#: Fixed, so the same matrix always yields the same estimate (INV-7).
_SPLIT_SEED: Final[int] = 20140228

_EPS: Final[float] = 1e-12


@dataclass(frozen=True, slots=True)
class PboReport:
    """What CSCV found (``soft_checks_json``, spec section 6)."""

    pbo: float
    #: One relative rank per split, in ``(0, 1)``: where the in-sample winner
    #: landed out of sample.
    ranks: tuple[float, ...] = ()
    #: The logits of those ranks. Negative means "below median out of sample".
    logits: tuple[float, ...] = ()
    n_splits: int = 0
    n_trials: int = 0
    n_blocks: int = 0

    @property
    def median_logit(self) -> float | None:
        """The middle split's logit; a summary of how badly the winner degrades."""
        return float(np.median(self.logits)) if self.logits else None

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "pbo": self.pbo,
            "median_logit": self.median_logit,
            "n_splits": self.n_splits,
            "n_trials": self.n_trials,
            "n_blocks": self.n_blocks,
        }


def _performance(matrix: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Each trial's performance over a subset of periods.

    The Sharpe ratio of the subset rather than its mean return: CSCV ranks
    trials, and ranking on raw return would let a trial win a half by being the
    most leveraged rather than the best. A column with no variation scores 0 —
    it made no bet, so it neither wins nor loses the ranking.
    """
    window = matrix[rows, :]
    means = window.mean(axis=0)
    deviations = window.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.where(deviations > _EPS, means / deviations, 0.0)
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


def _split_choices(blocks: int, half: int, cap: int) -> list[tuple[int, ...]]:
    """Which halves to take as in-sample, at most ``cap`` of them.

    Only combinations containing block 0 are considered, with the complement
    taken as the other half: every split is then used exactly once *with* its
    complement, which is the symmetry the method is named for. Enumerating all
    ``C(n, n/2)`` would use each pair twice and change nothing but the runtime.

    When there are more such halves than ``cap``, they are **sampled uniformly**
    rather than truncated. A lexicographic prefix is not a sample: its entries
    all begin ``(0, 1, 2, ...)``, so every split it yields puts the early blocks
    in-sample and the late ones out, and the estimate becomes a statement about
    the tail of the period rather than about the search. Measured on pure noise,
    truncation returned PBO values from 0.01 to 0.82 across five seeds; sampling
    returns the ~0.5 the method is calibrated to.

    The sampling seed is fixed, so the same matrix always yields the same
    estimate (INV-7).
    """
    total = math.comb(blocks - 1, half - 1)
    if total <= cap:
        return [
            combination
            for combination in itertools.combinations(range(blocks), half)
            if 0 in combination
        ]
    # A reproducibility seed, not a secret: `random.sample` draws k indices from a
    # range of hundreds of millions without materialising it, which is exactly
    # what is wanted here and what `numpy.random.Generator.choice` cannot do
    # without allocating the whole population.
    chooser = random.Random(_SPLIT_SEED)  # noqa: S311 - statistical sampling, not cryptography
    # Block 0 is always in-sample, so the free choice is `half - 1` of the
    # remaining `blocks - 1`.
    return [
        (0, *_unrank(index, blocks - 1, half - 1, offset=1))
        for index in sorted(chooser.sample(range(total), cap))
    ]


def _unrank(index: int, n: int, k: int, *, offset: int = 0) -> tuple[int, ...]:
    """The ``index``-th ``k``-subset of ``range(n)`` in lexicographic order.

    Used rather than materialising every combination: ``n_blocks`` is
    configurable, and ``C(31, 15)`` is three hundred million tuples nobody needs
    in memory in order to draw five hundred of them.
    """
    chosen: list[int] = []
    remaining = index
    current = 0
    for position in range(k):
        while True:
            block = math.comb(n - current - 1, k - position - 1)
            if remaining < block:
                break
            remaining -= block
            current += 1
        chosen.append(current + offset)
        current += 1
    return tuple(chosen)


def logit_ranks(
    matrix: np.ndarray, settings: PboSettings | None = None
) -> tuple[list[float], list[float]]:
    """The out-of-sample rank of each split's in-sample winner, and its logit.

    Returns ``(ranks, logits)`` where a rank is ``(position + 1) / (n_trials + 1)``
    in ``(0, 1)`` — never exactly 0 or 1, so the logit is always finite.
    """
    limits = settings or PboSettings()
    periods, trials = matrix.shape
    blocks = limits.n_blocks
    if blocks % 2:
        raise ConfigError(
            "CSCV splits the blocks into two equal halves, so n_blocks must be even",
            n_blocks=blocks,
        )
    if periods < blocks:
        raise ConfigError(
            "there are fewer periods than blocks to divide them into",
            n_periods=periods,
            n_blocks=blocks,
        )
    if trials < 2:
        raise ConfigError(
            "the probability of backtest overfitting compares trials against each "
            "other, so it needs at least two",
            n_trials=trials,
        )

    partitions = [
        np.asarray(part, dtype="int64") for part in np.array_split(np.arange(periods), blocks)
    ]
    half = blocks // 2

    ranks: list[float] = []
    logits: list[float] = []
    for combination in _split_choices(blocks, half, max(1, limits.max_combinations)):
        chosen = set(combination)
        in_sample = np.concatenate([partitions[index] for index in sorted(chosen)])
        out_of_sample = np.concatenate(
            [partitions[index] for index in range(blocks) if index not in chosen]
        )

        winner = int(np.argmax(_performance(matrix, in_sample)))
        out_scores = _performance(matrix, out_of_sample)
        # `argsort(argsort(x))` is the ascending rank of each element; the
        # winner's is how many trials it beat out of sample.
        position = int(np.argsort(np.argsort(out_scores))[winner])
        relative = (position + 1) / (trials + 1)
        ranks.append(relative)
        logits.append(math.log(relative / (1.0 - relative)))
    return ranks, logits


def probability_of_backtest_overfitting(
    trial_returns: Sequence[Sequence[float]] | np.ndarray,
    settings: PboSettings | None = None,
) -> PboReport:
    """Estimate PBO by CSCV (spec section 14.4, check 2).

    Args:
        trial_returns: Per-period returns, one **column** per trial and one row
            per period. Every trial must cover the same periods, because CSCV
            compares them on the same blocks.
        settings: ``validation.pbo``.

    Returns:
        A :class:`PboReport` whose ``pbo`` is the fraction of splits on which the
        in-sample winner finished below median out of sample. Section 14.4
        charges 25 points above 0.5 and 10 above 0.3.

    Raises:
        ConfigError: the matrix is the wrong shape, or the blocks cannot be
            halved. Reported rather than worked around: a silently transposed
            matrix would produce a plausible number about nothing.
    """
    matrix = np.asarray(trial_returns, dtype="float64")
    if matrix.ndim != 2:
        raise ConfigError(
            "trial returns must be a 2-D matrix of periods by trials", shape=list(matrix.shape)
        )

    ranks, logits = logit_ranks(matrix, settings)
    limits = settings or PboSettings()
    below_median = sum(1 for value in logits if value < 0.0)
    return PboReport(
        pbo=below_median / len(logits) if logits else 0.0,
        ranks=tuple(ranks),
        logits=tuple(logits),
        n_splits=len(logits),
        n_trials=int(matrix.shape[1]),
        n_blocks=limits.n_blocks,
    )
