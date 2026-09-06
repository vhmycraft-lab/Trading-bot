"""Per-generation hidden training environments (Project Rome sections 4-19). 🔒

Hundreds of generations of evolutionary pressure against one fixed slice of
history will find that slice's accidents. This module is the countermeasure: each
generation is evaluated inside a **freshly selected training environment** — a
window of the training partition, a starting capital, a slippage level and an
asset subset — drawn by Rome's own infrastructure and recorded for audit.

Four properties, and each is a design decision rather than a detail:

**Valid.** Candidate windows are enumerated from the *training segment only*,
before any drawing happens. A window that could reach the validation segment or
the test partition is not rejected after the fact; it is never in the pool. That
is what makes "validation periods can never be selected" a property of the
enumeration rather than of a check somebody has to remember to run.

**Independent.** Generation *n+1*'s environment is not a function of generation
*n*'s. Nothing about the previous window, the generation index, the population,
or the clock enters the draw. The evolutionary algorithm therefore cannot learn
"next generation = next historical period", which a rolling window would teach it
within a handful of generations.

**Unpredictable.** The seed comes from the operating system's CSPRNG
(:func:`secrets.token_bytes`) and nothing else — not the generation number, not a
strategy id, not a code hash, not a timestamp, not anything an LLM or a user can
reach. Every subsequent choice is an HMAC-SHA256 expansion of that seed, so the
whole environment is a deterministic function of 32 secret bytes.

**Reproducible for privileged audit.** Because of that expansion,
:func:`reproduce` rebuilds an environment exactly from its recorded seed. A
researcher with the audit record can reconstruct precisely what a generation ran
against; a strategy or a prompt, which never sees the seed, cannot.

**Not harder, just different.** Every randomised quantity moves inside a
documented, realistic band (Rome section 9). Commission is not randomised at all
(section 12) and neither is execution timing (section 13) — both are structural
properties of the venue, and varying them would be manufacturing difficulty
rather than exposure. What varies is *which history*, *how much capital*, and
*how much slippage within the range the venue actually exhibits*.

**What is hidden.** Rome section 7 is explicit that the *dates* cannot be hidden
from a strategy that receives timestamped candles, and this module does not
pretend otherwise: it never rewrites a timestamp. What is hidden is the
*selection* — which environment was chosen and why — and that hiding is
structural, since :class:`TrainingEnvironment` is never placed in a strategy's
inputs or an LLM's context by any code path.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from quantlab.core.config import EnvironmentSettings
from quantlab.core.errors import ConfigError
from quantlab.core.hashing import canonical_json, sha256_hex
from quantlab.core.splits import SplitPolicy
from quantlab.core.types import BacktestConfig, BarFrame

__all__ = [
    "ENVIRONMENT_MODEL_VERSION",
    "SEED_BYTES",
    "EnvironmentSelector",
    "GenerationEnvironment",
    "SamplingReport",
    "TrainingEnvironment",
    "TrainingWindow",
    "WindowPool",
    "build_window_pool",
    "reproduce",
    "sampling_report",
]

#: Bytes of operating-system entropy behind every environment. 32 is the width of
#: the HMAC-SHA256 key that expands it; less would narrow the seed space for no
#: saving worth having.
SEED_BYTES: Final[int] = 32

#: Bumped whenever the *derivation* changes — a different expansion from the same
#: seed is a different environment, and an audit record that did not say so would
#: reproduce the wrong thing.
ENVIRONMENT_MODEL_VERSION: Final[str] = "env/1"

#: Cents. Capital is money, and money that is not quantised produces position
#: sizes no venue would accept.
_CAPITAL_STEP: Final[float] = 0.01

#: Tenths of a basis point. Finer than any venue's quoted spread resolution and
#: coarse enough that a recorded value round-trips through JSON unchanged.
_SLIPPAGE_STEP: Final[float] = 0.1


# ---------------------------------------------------------------------------
# the pool of valid windows
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True, order=True)
class TrainingWindow:
    """A closed range of bar open times inside the training segment."""

    start_ts: int
    end_ts: int

    def n_bars(self, bar_ms: int) -> int:
        return (self.end_ts - self.start_ts) // bar_ms + 1

    @property
    def window_id(self) -> str:
        """A stable id for this window, for sampling statistics.

        Derived from the bounds, so the same window drawn in two generations is
        recognisably the same window. This is an *internal* statistic key, never
        an identifier handed outward — Rome section 34 forbids exposing a
        ``2019_04_01_2020_08_14``-shaped name, and :attr:`TrainingEnvironment.
        environment_id` is the opaque one that leaves this module.
        """
        return sha256_hex(f"window|{self.start_ts}|{self.end_ts}")[:16]


@dataclass(frozen=True, slots=True)
class WindowPool:
    """Every window a generation may be evaluated on, and nothing else.

    Built once from the split policy. Its ``windows`` are all inside
    ``[train_start_ts, train_end_ts]`` by construction, which is why no selection
    path needs a validation-or-holdout check: there is nothing to check.
    """

    windows: tuple[TrainingWindow, ...]
    train_start_ts: int
    train_end_ts: int
    bar_ms: int

    def __post_init__(self) -> None:
        if not self.windows:
            raise ConfigError(
                "no valid training window could be built; the training segment is "
                "shorter than the shortest window the settings permit",
                train_start_ts=self.train_start_ts,
                train_end_ts=self.train_end_ts,
            )

    def __len__(self) -> int:
        return len(self.windows)

    @property
    def pool_id(self) -> str:
        """Identifies the pool an environment was drawn from.

        Recorded alongside the environment: a pool rebuilt under different
        settings is a different experiment, and a reproduction that silently used
        today's pool would be reproducing something else.
        """
        return sha256_hex(
            canonical_json(
                {
                    "bar_ms": self.bar_ms,
                    "train": [self.train_start_ts, self.train_end_ts],
                    "windows": [[w.start_ts, w.end_ts] for w in self.windows],
                }
            )
        )[:16]

    def contains(self, window: TrainingWindow) -> bool:
        return window in self.windows


def build_window_pool(policy: SplitPolicy, settings: EnvironmentSettings) -> WindowPool:
    """Enumerate the valid training windows (Rome section 8).

    Windows of several lengths, starting on a stride grid, all inside the
    training segment. Several lengths rather than one because a strategy that is
    only ever shown five-year windows learns the statistics of five-year windows;
    the stride makes the pool granular enough that consecutive draws are rarely
    the same window without making it so granular that two draws are effectively
    the same data.

    The bounds are fractions of the training segment rather than absolute bar
    counts, so one configuration is meaningful for an eight-year hourly split and
    for a test fixture of nine hundred bars alike. ``min_window_bars`` is the
    floor underneath both: Rome section 8 forbids sampling windows "obviously too
    short to produce meaningful evaluation", and a floor stated in bars is the
    only form of that rule which does not shrink with the dataset.

    Raises:
        ConfigError: the training segment cannot accommodate even the shortest
            permitted window. Fail closed (Rome section 37): a silently shortened
            window would be an unannounced change to what "training" means.
    """
    bar_ms = policy.bar_ms
    train = policy.segment("train")
    if train.end_ts is None:  # pragma: no cover - the train segment is always closed
        raise ConfigError("the training segment has no end, so no window can be built")
    total = (train.end_ts - train.start_ts) // bar_ms + 1

    shortest = max(settings.min_window_bars, int(total * settings.window_min_fraction))
    longest = max(shortest, int(total * settings.window_max_fraction))
    if shortest > total:
        raise ConfigError(
            "the training segment is shorter than the shortest permitted window",
            train_bars=total,
            min_window_bars=shortest,
        )

    step = max(1, int(total * settings.length_step_fraction))
    lengths = sorted({min(length, total) for length in range(shortest, longest + 1, step)})
    if longest not in lengths:
        lengths.append(longest)

    stride = max(1, int(total * settings.stride_fraction))
    windows: set[TrainingWindow] = set()
    for length in lengths:
        span = (length - 1) * bar_ms
        last_start = train.end_ts - span
        start = train.start_ts
        while start <= last_start:
            windows.add(TrainingWindow(start_ts=start, end_ts=start + span))
            start += stride * bar_ms
        # The window that ends exactly at the segment's end is always included:
        # a stride that does not divide the segment would otherwise make the most
        # recent history reachable only by the longest window.
        windows.add(TrainingWindow(start_ts=last_start, end_ts=train.end_ts))

    return WindowPool(
        windows=tuple(sorted(windows)),
        train_start_ts=train.start_ts,
        train_end_ts=train.end_ts,
        bar_ms=bar_ms,
    )


# ---------------------------------------------------------------------------
# the environment, and the expansion that produces it
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TrainingEnvironment:
    """One generation's hidden evaluation environment (Rome sections 4, 18).

    Every field here is audit material. None of it reaches a strategy or an LLM:
    the strategy is handed the bars the window resolves to and the ordinary
    backtest configuration those settings produce, and the LLM is handed
    performance metrics. Neither is handed this object, and
    ``tests/unit/test_environment_isolation.py`` is what keeps that true.

    ``seed_hex`` is the secret. Everything else in this object is derived from
    it, so possession of the seed is possession of the environment — which is
    exactly why it is stored in the audit table and passed to nothing.
    """

    environment_id: str
    generation_index: int
    window: TrainingWindow
    starting_capital: float
    slippage_bps: float
    asset_universe: tuple[str, ...]
    seed_hex: str
    pool_id: str
    dataset_version: str
    execution_model_version: str
    derivation_version: str = ENVIRONMENT_MODEL_VERSION

    def audit_record(self) -> dict[str, object]:
        """The full record, for the audit table and for nothing else.

        Named so that a reader of a call site can see what is being handed over.
        A method called ``to_dict`` would be reached for by accident.
        """
        return {
            "environment_id": self.environment_id,
            "generation_index": self.generation_index,
            "window_start_ts": self.window.start_ts,
            "window_end_ts": self.window.end_ts,
            "window_id": self.window.window_id,
            "starting_capital": self.starting_capital,
            "slippage_bps": self.slippage_bps,
            "asset_universe": list(self.asset_universe),
            "seed_hex": self.seed_hex,
            "pool_id": self.pool_id,
            "dataset_version": self.dataset_version,
            "execution_model_version": self.execution_model_version,
            "derivation_version": self.derivation_version,
        }


def _prf(seed: bytes, label: str, counter: int) -> bytes:
    """HMAC-SHA256 as a pseudo-random function keyed by the seed.

    Used instead of seeding a general-purpose PRNG because the seed is secret and
    the derivation must not leak it: HMAC's output tells an observer nothing
    about the key, whereas the internal state of a Mersenne twister is
    recoverable from its output. It also makes the expansion stable across Python
    versions, which a ``random.Random`` stream is not guaranteed to be, and an
    audit reproduction that changed with the interpreter would be no
    reproduction at all.
    """
    return hmac.new(seed, f"{label}|{counter}".encode(), hashlib.sha256).digest()


def _uniform_below(seed: bytes, label: str, bound: int) -> int:
    """A uniform integer in ``[0, bound)``, without modulo bias.

    Rejection sampling rather than ``% bound``: the pool sizes here are arbitrary
    and a modulo would over-weight the first ``2**256 % bound`` windows. The bias
    would be tiny and would still be a systematic preference for one end of
    history, which is the exact failure this module exists to prevent.
    """
    if bound <= 0:  # pragma: no cover - callers pass non-empty populations
        raise ConfigError("cannot draw from an empty range", bound=bound)
    limit = (1 << 256) - ((1 << 256) % bound)
    for counter in range(1024):
        value = int.from_bytes(_prf(seed, label, counter), "big")
        if value < limit:
            return value % bound
    raise ConfigError(  # pragma: no cover - probability below 2**-1024
        "rejection sampling failed to terminate", label=label, bound=bound
    )


def _uniform_between(seed: bytes, label: str, low: float, high: float, step: float) -> float:
    """A uniform multiple of ``step`` in ``[low, high]``.

    Quantised because these are money and basis points: an unrounded draw
    produces values no venue quotes and no ledger stores.
    """
    if high <= low:
        return round(low, 10)
    steps = round((high - low) / step)
    return round(low + _uniform_below(seed, label, steps + 1) * step, 10)


def _subset(seed: bytes, label: str, universe: Sequence[str], minimum: int) -> tuple[str, ...]:
    """A subset of ``universe`` of at least ``minimum`` members, in universe order.

    Order is the universe's, not the draw's, so two environments that selected
    the same assets are recognisably equal. Rome section 15 requires that the
    assets be real and available in the period; that responsibility sits with
    whoever configures the universe for a split, and the limitation is stated in
    ``docs/ENVIRONMENT.md`` rather than papered over here.
    """
    names = tuple(universe)
    floor = max(1, min(minimum, len(names)))
    if floor >= len(names):
        return names
    size = floor + _uniform_below(seed, f"{label}|size", len(names) - floor + 1)
    chosen: list[str] = []
    remaining = list(names)
    for index in range(size):
        pick = _uniform_below(seed, f"{label}|pick|{index}", len(remaining))
        chosen.append(remaining.pop(pick))
    return tuple(name for name in names if name in set(chosen))


def _derive(
    seed: bytes,
    *,
    generation_index: int,
    pool: WindowPool,
    settings: EnvironmentSettings,
    universe: Sequence[str],
    dataset_version: str,
    execution_model_version: str,
) -> TrainingEnvironment:
    """Expand a seed into a complete environment. Pure, and the audit's inverse."""
    window = pool.windows[_uniform_below(seed, "window", len(pool.windows))]
    return TrainingEnvironment(
        # Opaque by construction (Rome section 34): a one-way function of the
        # secret seed, so the id can travel in logs and run rows without carrying
        # the window, the capital or anything else back out with it.
        environment_id=hashlib.sha256(b"environment|" + seed).hexdigest()[:16],
        generation_index=generation_index,
        window=window,
        starting_capital=_uniform_between(
            seed,
            "capital",
            settings.min_starting_capital,
            settings.max_starting_capital,
            _CAPITAL_STEP,
        ),
        slippage_bps=_uniform_between(
            seed, "slippage", settings.min_slippage_bps, settings.max_slippage_bps, _SLIPPAGE_STEP
        ),
        asset_universe=_subset(seed, "assets", universe, settings.min_assets),
        seed_hex=seed.hex(),
        pool_id=pool.pool_id,
        dataset_version=dataset_version,
        execution_model_version=execution_model_version,
    )


class EnvironmentSelector:
    """Rome's trusted randomisation service (Rome sections 5, 17, 19).

    Constructed once per evolution run and asked for an environment per
    generation. It takes **no** input that could carry an outside preference: the
    only argument to :meth:`select` is the generation index, and that is used as
    a label on the record rather than as an ingredient of the draw. There is
    deliberately no parameter for a window, a seed, a capital or a slippage —
    not a validated one, not an optional one. An API that accepted a window and
    ignored it would be one refactor away from honouring it.

    The seed for each generation is fresh operating-system entropy. Two
    consecutive calls are independent in the strongest sense available: neither
    the object nor the process carries state that links them, so restarting Rome
    does not resume a sequence, and two workers drawing at the same moment cannot
    collide except with probability ``2**-256``.
    """

    __slots__ = (
        "_dataset_version",
        "_execution_model_version",
        "_pool",
        "_settings",
        "_universe",
    )

    def __init__(
        self,
        pool: WindowPool,
        settings: EnvironmentSettings,
        *,
        universe: Sequence[str],
        dataset_version: str,
        execution_model_version: str,
    ) -> None:
        self._pool = pool
        self._settings = settings
        self._universe = tuple(universe)
        self._dataset_version = dataset_version
        self._execution_model_version = execution_model_version
        if not self._universe:
            raise ConfigError("an environment needs at least one tradable asset")

    @property
    def pool(self) -> WindowPool:
        return self._pool

    def select(self, generation_index: int) -> TrainingEnvironment:
        """Draw a fresh environment for one generation.

        Args:
            generation_index: recorded on the environment. It is **not** an input
                to the draw: deriving anything from it would make the sequence a
                function of the generation counter, which is precisely the
                predictability Rome section 5 forbids.
        """
        return _derive(
            secrets.token_bytes(SEED_BYTES),
            generation_index=generation_index,
            pool=self._pool,
            settings=self._settings,
            universe=self._universe,
            dataset_version=self._dataset_version,
            execution_model_version=self._execution_model_version,
        )

    def reproduce(self, record: Mapping[str, object]) -> TrainingEnvironment:
        """Rebuild a recorded environment from its audit row. Privileged."""
        return reproduce(
            record,
            pool=self._pool,
            settings=self._settings,
            universe=self._universe,
        )


def reproduce(
    record: Mapping[str, object],
    *,
    pool: WindowPool,
    settings: EnvironmentSettings,
    universe: Sequence[str],
) -> TrainingEnvironment:
    """Rebuild an environment from its recorded seed (Rome sections 19, 24).

    The privileged audit path, and the reason the expansion is a PRF rather than
    a sequence of independent draws: everything except the seed is redundant, so
    a reproduction that disagrees with the stored row means the *derivation*
    changed, and the caller can see exactly which field moved.

    Raises:
        ConfigError: the record has no seed, or was produced by a different
            derivation version. Refused rather than reproduced approximately —
            an audit that quietly rebuilt something else would be worse than one
            that failed.
    """
    seed_hex = str(record.get("seed_hex", ""))
    if not seed_hex:
        raise ConfigError("the audit record carries no seed, so it cannot be reproduced")
    version = str(record.get("derivation_version", ENVIRONMENT_MODEL_VERSION))
    if version != ENVIRONMENT_MODEL_VERSION:
        raise ConfigError(
            "this record was produced by a different environment derivation; "
            "reproducing it under the current one would rebuild a different environment",
            recorded=version,
            current=ENVIRONMENT_MODEL_VERSION,
        )
    return _derive(
        bytes.fromhex(seed_hex),
        generation_index=int(str(record.get("generation_index", 0) or 0)),
        pool=pool,
        settings=settings,
        universe=universe,
        dataset_version=str(record.get("dataset_version", "")),
        execution_model_version=str(record.get("execution_model_version", "")),
    )


# ---------------------------------------------------------------------------
# exposure tracking (Rome sections 16, 18, 43)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SamplingReport:
    """How evenly the sampled environments have covered training history.

    Randomisation alone does not guarantee coverage: independent uniform draws
    still leave gaps and clumps, and evolutionary pressure over a long campaign
    can sit inside one of the clumps without anybody noticing. This is the
    instrument that notices.

    Two decisions make it measure the right thing.

    **Observations, not windows.** Windows overlap, so counting *windows* answers
    "which ranges were drawn" while counting the observations inside them answers
    Rome section 43's actual question — "how often has evolutionary pressure seen
    this bar?" Two disjoint windows drawn once each and one window drawn twice are
    very different exposures and identical window counts.

    **Compared against the pool, not against a flat line.** A window is a
    contiguous run of bars, so drawing windows uniformly still covers the middle
    of history far more often than its edges — a window can overlap the middle
    from either side and the first bucket only from one. Measured on this
    repository's own default settings, four hundred genuinely uniform draws reach
    1.6x the flat share, so a flat baseline would report a healthy campaign as
    concentrated and its warning would be worthless. The baseline here is the
    exposure the **pool itself** produces if every window is drawn equally often,
    which isolates what the campaign did from the pool's geometry, which the
    campaign did not choose.
    """

    n_environments: int
    #: Draws per distinct window id.
    window_counts: Mapping[str, int]
    #: Times each history bucket was covered by a selected window.
    bucket_counts: Mapping[int, int]
    #: Each bucket's share of realised exposure.
    observed_shares: Mapping[int, float]
    #: Each bucket's share if every pool window were drawn equally often.
    expected_shares: Mapping[int, float]
    #: Bucket width, in bars.
    bucket_bars: int
    #: Buckets whose realised share exceeds ``overuse_factor`` times expected.
    overused_buckets: tuple[int, ...]
    #: Buckets the pool can reach that no environment has ever covered.
    unsampled_buckets: tuple[int, ...]

    @property
    def concentration(self) -> float:
        """Worst bucket's realised exposure relative to expected.

        1.0 is exactly what uniform sampling of this pool produces; higher means
        evolutionary pressure has concentrated somewhere the pool did not put it.
        """
        ratios = [
            self.observed_shares[index] / expected
            for index, expected in self.expected_shares.items()
            if expected > 0
        ]
        return max(ratios) if ratios else 0.0

    @property
    def flagged(self) -> bool:
        """Whether sampling has become concentrated enough to report.

        Either a bucket is disproportionately over-used relative to what the pool
        would produce, or part of the reachable history has never been seen at
        all. Both are failures of the same purpose, and both are *reported* rather
        than corrected: Rome section 43 forbids "solving" the problem by excluding
        history, and re-weighting the pool would make the draw no longer uniform
        over valid windows.
        """
        return bool(self.overused_buckets) or bool(self.unsampled_buckets)

    def as_dict(self) -> dict[str, object]:
        return {
            "n_environments": self.n_environments,
            "bucket_bars": self.bucket_bars,
            "concentration": self.concentration,
            "overused_buckets": list(self.overused_buckets),
            "unsampled_buckets": list(self.unsampled_buckets),
            "flagged": self.flagged,
            "n_distinct_windows": len(self.window_counts),
        }


def _coverage(
    windows: Sequence[TrainingWindow], pool: WindowPool, buckets: int, bucket_ms: int
) -> dict[int, int]:
    """How many of ``windows`` cover each history bucket."""
    counts: dict[int, int] = dict.fromkeys(range(buckets), 0)
    for window in windows:
        first = max(0, (window.start_ts - pool.train_start_ts) // bucket_ms)
        last = min(buckets - 1, (window.end_ts - pool.train_start_ts) // bucket_ms)
        for index in range(first, last + 1):
            counts[index] += 1
    return counts


def _shares(counts: Mapping[int, int]) -> dict[int, float]:
    total = sum(counts.values())
    return {index: (count / total if total else 0.0) for index, count in counts.items()}


def sampling_report(
    windows: Sequence[TrainingWindow],
    pool: WindowPool,
    settings: EnvironmentSettings,
) -> SamplingReport:
    """Summarise which parts of training history have actually been sampled.

    Args:
        windows: the window of every environment selected so far, in any order.
        pool: the pool they were drawn from. It fixes both the span being covered
            and the baseline the realised exposure is judged against.
        settings: ``environment``; ``overuse_factor`` and ``n_history_buckets``.

    Returns:
        A :class:`SamplingReport`. With no environments yet it reports every
        reachable bucket unsampled, which is accurate rather than vacuous:
        nothing has been seen.
    """
    buckets = max(1, settings.n_history_buckets)
    span = pool.train_end_ts - pool.train_start_ts + pool.bar_ms
    bucket_ms = max(pool.bar_ms, span // buckets)

    window_counts: dict[str, int] = {}
    for window in windows:
        window_counts[window.window_id] = window_counts.get(window.window_id, 0) + 1

    bucket_counts = _coverage(windows, pool, buckets, bucket_ms)
    observed = _shares(bucket_counts)
    expected = _shares(_coverage(pool.windows, pool, buckets, bucket_ms))

    return SamplingReport(
        n_environments=len(windows),
        window_counts=window_counts,
        bucket_counts=bucket_counts,
        observed_shares=observed,
        expected_shares=expected,
        bucket_bars=max(1, bucket_ms // pool.bar_ms),
        overused_buckets=tuple(
            index
            for index in sorted(observed)
            if expected[index] > 0 and observed[index] > expected[index] * settings.overuse_factor
        ),
        unsampled_buckets=tuple(
            index
            for index in sorted(bucket_counts)
            if expected[index] > 0 and bucket_counts[index] == 0
        ),
    )


@dataclass(frozen=True, slots=True)
class GenerationEnvironment:
    """One generation's evaluation environment, as this loop sees it.

    Three fields, and the omissions are the design. The loop is handed the bars
    the environment resolves to, the backtest settings it produces, and an
    **opaque** id to record — never the window bounds, the seed, the starting
    capital or the slippage as named quantities. Rome section 6 requires the
    selection be hidden from strategy-facing paths, and the cheapest way to keep
    a value out of a prompt or a strategy input is for the code that builds them
    never to have held it.

    ``environment_id`` also enters the *segment* name, which is what keeps the run
    cache honest: a run's identity (section 11.2) is built from the strategy, the
    parameters, the split, the segment and the configuration hash, and **not**
    from the bar range. Two generations that drew different windows but the same
    capital and slippage would otherwise share a run id, and the second would
    silently be served the first one's results. Suffixing the segment makes the
    environment part of the identity without inventing a second identity rule.
    """

    bars: BarFrame
    config: BacktestConfig
    environment_id: str

    def segment_for(self, base: str) -> str:
        """``base`` qualified by this environment, e.g. ``train:9f2c...``.

        Safe to store and to display: the id is a one-way function of the seed
        (Rome section 34), so it distinguishes environments without carrying the
        window back out with it.
        """
        return f"{base}:{self.environment_id}" if self.environment_id else base
