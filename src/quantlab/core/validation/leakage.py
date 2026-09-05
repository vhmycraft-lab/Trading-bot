"""The truncation probe: proof, per strategy, that the past does not depend on the future.

Master spec section 14.2. INV-3 says a strategy can never observe bar ``t+1`` when
deciding at bar ``t``. ``BarWindow`` enforces that structurally for ``bar_loop``
strategies — it raises rather than returning a future bar. A **vectorised**
strategy gets no such protection: it is handed the whole segment as a DataFrame
and asked for a series, and pandas will happily shift, centre or aggregate across
the whole of it. Section 9.1 therefore requires every vectorised strategy to clear
this probe before any backtest, and the loader runs it automatically.

The idea is empirical rather than structural, which is why it catches what static
analysis cannot. Run the strategy on the whole segment, then run it again on a
prefix. A causal strategy cannot tell the difference: its decision at bar *i* was
never a function of anything after *i*, so truncating the tail must leave the head
byte-for-byte identical. Then do it once more with the tail *replaced* by a
reversed copy — a strategy that peeks will change its mind about the past when the
future changes underneath it, and one that does not, will not.

**The one bar of slack.** Section 9.1 shifts a vectorised strategy's output by one
bar before consuming it, so a signal recorded at bar *i* was computed from data up
to bar *i-1*. That shift is deliberate insurance, and it means the probe must
compare *one bar further* for a vectorised strategy than for a ``bar_loop`` one —
otherwise a strategy reading exactly one bar ahead would hide inside the slack the
engine grants it. :data:`SIGNAL_LAG` is that difference, and getting it wrong in
either direction is the difference between a probe that misses real leakage and
one that rejects honest code.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np

from quantlab.core.errors import LeakageDetected, ValidationError_
from quantlab.core.types import BacktestConfig, BacktestResult, BarFrame

__all__ = [
    "DEFAULT_CUT_FRACTIONS",
    "MIN_PROBE_BARS",
    "REVERSED_TAIL_FRACTION",
    "SIGNAL_LAG",
    "TAIL_MODES",
    "TAIL_SCALE",
    "Divergence",
    "Evaluate",
    "ProbeResult",
    "TailMode",
    "default_cut_points",
    "perturbed_tail",
    "probe_evaluations",
    "require_causal",
    "reversed_tail",
    "truncation_probe",
]

#: Five deterministic points through the segment (spec section 14.2). Fixed, not
#: sampled: a probe whose cut points moved between runs could not be reproduced,
#: and a leak that shows up only on some runs is worse than no probe at all.
DEFAULT_CUT_FRACTIONS: Final[tuple[float, ...]] = (0.20, 0.35, 0.50, 0.65, 0.80)

#: Share of the segment replaced by a reversed copy in the second test.
REVERSED_TAIL_FRACTION: Final[float] = 0.10

#: Bars of guaranteed lag between the data a signal may use and the bar it is
#: recorded at, per style (spec section 9.1).
SIGNAL_LAG: Final[dict[str, int]] = {"bar_loop": 0, "vectorized": 1}

#: Below this the cut points collapse into each other and the tail replacement has
#: nothing to reverse. Refusing is the honest answer: a probe that cannot separate
#: the head from the tail has not checked anything.
MIN_PROBE_BARS: Final[int] = 20

#: Runs one backtest over the bars it is given. Injected rather than assumed, so
#: the same probe serves a trusted strategy run in-process and an untrusted one run
#: through the sandbox — the comparison is identical, only the executor differs.
Evaluate = Callable[[BarFrame], BacktestResult]

_ProbeName = Literal["truncation", "reversed_tail"]


@dataclass(frozen=True, slots=True)
class Divergence:
    """One bar where two runs disagreed about the past."""

    probe: _ProbeName
    #: The cut point, or the first replaced bar index.
    at: int
    series: Literal["signal", "position_frac"]
    bar_index: int
    reference: str
    observed: str
    #: Which tail replacement produced the divergence, for the tail probe.
    mode: str = ""
    #: True when the bar lies inside the declared warm-up, which additionally
    #: means ``warmup_bars`` is under-declared (spec section 14.2).
    in_warmup: bool = False

    def __str__(self) -> str:
        where = f"{self.probe}[{self.mode}]@{self.at}" if self.mode else f"{self.probe}@{self.at}"
        warm = " (inside declared warm-up)" if self.in_warmup else ""
        return (
            f"{where}: {self.series} at bar {self.bar_index} changed "
            f"{self.reference!r} -> {self.observed!r}{warm}"
        )


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """What the probe found. ``passed`` is the whole verdict; the rest is evidence."""

    passed: bool
    n_bars: int
    style: str
    warmup_bars: int
    cut_points: tuple[int, ...]
    tail_start: int
    divergences: tuple[Divergence, ...] = ()
    n_evaluations: int = 0

    @property
    def warmup_under_declared(self) -> bool:
        """A divergence inside the declared warm-up means the declaration is short."""
        return any(divergence.in_warmup for divergence in self.divergences)

    def describe(self, limit: int = 10) -> str:
        if self.passed:
            return (
                f"causal: {len(self.cut_points)} truncations and one reversed tail over "
                f"{self.n_bars} bars agreed exactly"
            )
        lines = [f"LEAKAGE_DETECTED: {len(self.divergences)} divergence(s) over {self.n_bars} bars"]
        lines.extend(f"  {divergence}" for divergence in self.divergences[:limit])
        if len(self.divergences) > limit:
            lines.append(f"  ... and {len(self.divergences) - limit} more")
        if self.warmup_under_declared:
            lines.append(
                f"  warmup_bars={self.warmup_bars} is under-declared: the strategy's "
                "decisions inside it are not yet stable"
            )
        return "\n".join(lines)


def default_cut_points(
    n_bars: int, fractions: Sequence[float] = DEFAULT_CUT_FRACTIONS
) -> tuple[int, ...]:
    """Bar indices to truncate at, per spec section 14.2.

    Deduplicated and sorted, so two runs over the same segment probe exactly the
    same places.
    """
    if n_bars < MIN_PROBE_BARS:
        raise ValidationError_(
            "segment is too short to probe for leakage",
            n_bars=n_bars,
            minimum=MIN_PROBE_BARS,
        )
    points = {int(n_bars * fraction) for fraction in fractions}
    usable = sorted(point for point in points if 0 < point < n_bars)
    if not usable:
        raise ValidationError_("no usable cut points", n_bars=n_bars, fractions=list(fractions))
    return tuple(usable)


#: How the tail is replaced. ``reversed`` is the one spec section 14.2 mandates;
#: the two scalings exist because reversal alone leaves detection to chance.
TailMode = Literal["reversed", "scaled_up", "scaled_down"]
TAIL_MODES: Final[tuple[TailMode, ...]] = ("reversed", "scaled_up", "scaled_down")

#: Factor applied to the replaced prices in the scaled modes. Far outside any
#: plausible continuation, on purpose: see :func:`perturbed_tail`.
TAIL_SCALE: Final[float] = 8.0


def perturbed_tail(
    bars: BarFrame, fraction: float = REVERSED_TAIL_FRACTION, mode: TailMode = "reversed"
) -> tuple[BarFrame, int]:
    """Give the segment a different future, and return where the new one starts.

    Timestamps stay ascending and only prices and volumes change: the result is a
    different future, not a broken frame. A causal strategy cannot notice, because
    it never read those bars when deciding about the ones before them.

    Three modes, because one is not enough. ``reversed`` is what spec section 14.2
    asks for, and it is the right shape — but a strategy that peeks exactly one bar
    ahead is caught at one bar only, the boundary, and whether a *boolean* signal
    flips there is a coin toss. The two scalings close that gap: the replaced
    prices are moved far outside any plausible continuation, once upward and once
    downward, so a comparison that depends on the first replaced bar must come out
    differently in at least one of them. An honest strategy is unaffected by all
    three by construction, since its decisions before the boundary never read those
    bars at all.

    Returns the new frame and the index of the first replaced bar.
    """
    n = bars.n_bars
    if n < MIN_PROBE_BARS:
        raise ValidationError_(
            "segment is too short to perturb a tail", n_bars=n, minimum=MIN_PROBE_BARS
        )
    tail_length = max(2, int(n * fraction))
    start = n - tail_length
    if start <= 0:
        raise ValidationError_("perturbed tail would consume the whole segment", n_bars=n)

    frame = bars.to_pandas()
    price_columns = ("open", "high", "low", "close")
    for name in (name for name in frame.columns if name != "ts_open"):
        values = frame[name].to_numpy()
        head, tail = values[:start], values[start:]
        if mode == "reversed":
            replaced = tail[::-1]
        elif name in price_columns:
            factor = TAIL_SCALE if mode == "scaled_up" else 1.0 / TAIL_SCALE
            replaced = tail * factor
        else:
            replaced = tail
        frame[name] = np.concatenate([head, replaced])
    return (
        BarFrame(
            frame,
            symbol=bars.symbol,
            timeframe=bars.timeframe,
            dataset_id=bars.dataset_id,
        ),
        start,
    )


def reversed_tail(bars: BarFrame, fraction: float = REVERSED_TAIL_FRACTION) -> tuple[BarFrame, int]:
    """The reversed replacement of spec section 14.2."""
    return perturbed_tail(bars, fraction, "reversed")


def _signal_strings(result: BacktestResult) -> list[str]:
    return [str(value) for value in result.signals]


def _position_values(result: BacktestResult) -> np.ndarray:
    return np.asarray(result.position_frac.to_numpy(), dtype="float64")


def _compare(
    *,
    probe: _ProbeName,
    at: int,
    reference: BacktestResult,
    observed: BacktestResult,
    signal_horizon: int,
    position_horizon: int,
    warmup_bars: int,
    mode: str = "",
) -> list[Divergence]:
    """Every bar on which the two runs disagree, within their comparable horizons."""
    found: list[Divergence] = []

    left, right = _signal_strings(reference), _signal_strings(observed)
    for index in range(min(signal_horizon, len(left), len(right))):
        if left[index] != right[index]:
            found.append(
                Divergence(
                    probe=probe,
                    at=at,
                    series="signal",
                    bar_index=index,
                    reference=left[index],
                    observed=right[index],
                    in_warmup=index < warmup_bars,
                    mode=mode,
                )
            )

    left_pos, right_pos = _position_values(reference), _position_values(observed)
    for index in range(min(position_horizon, len(left_pos), len(right_pos))):
        if not _same(left_pos[index], right_pos[index]):
            found.append(
                Divergence(
                    probe=probe,
                    at=at,
                    series="position_frac",
                    bar_index=index,
                    reference=f"{left_pos[index]:.12g}",
                    observed=f"{right_pos[index]:.12g}",
                    in_warmup=index < warmup_bars,
                    mode=mode,
                )
            )
    return found


def _same(left: float, right: float) -> bool:
    """Exact, except that two NaNs are the same answer.

    Deliberately not a tolerance: two runs of a deterministic engine over the same
    bars produce the same floats bit for bit (INV-7), so any difference at all is
    a difference in what the strategy decided, not arithmetic noise.
    """
    if np.isnan(left) and np.isnan(right):
        return True
    return bool(left == right)


def probe_evaluations(
    evaluate: Evaluate,
    bars: BarFrame,
    *,
    style: str = "vectorized",
    warmup_bars: int = 0,
    cut_points: Sequence[int] | None = None,
    tail_fraction: float = REVERSED_TAIL_FRACTION,
) -> ProbeResult:
    """Run the probe using ``evaluate`` to execute each backtest.

    The comparison horizons are the whole of the method, so they are stated here
    rather than buried:

    *Truncation at k.* The truncated run's last bar is its end of data, where the
    engine liquidates whatever is open — a mechanical difference, not a decision —
    so positions are compared over ``[0, k)`` and signals, which are emitted before
    that close, over ``[0, k]``.

    *Reversed tail from p.* Positions at ``p`` already reflect bar ``p``'s own close
    through mark-to-market, so positions are compared over ``[0, p)``. Signals are
    compared over ``[0, p + lag]``, where ``lag`` is the bar of slack section 9.1
    grants the style: a vectorised signal recorded at ``p`` was computed from data
    up to ``p-1`` and must therefore be unchanged, and it is precisely there that a
    one-bar look-ahead would otherwise hide.

    Raises:
        ValidationError_: the segment is too short, or the style is unknown. The
            probe refuses rather than reporting a pass it did not earn.
    """
    if style not in SIGNAL_LAG:
        raise ValidationError_("unknown strategy style", style=style, known=sorted(SIGNAL_LAG))
    points = tuple(cut_points) if cut_points is not None else default_cut_points(bars.n_bars)
    if not points:
        raise ValidationError_("no cut points to probe", n_bars=bars.n_bars)
    for point in points:
        if not 0 < point < bars.n_bars:
            raise ValidationError_(
                "cut point outside the segment", cut_point=point, n_bars=bars.n_bars
            )

    lag = SIGNAL_LAG[style]
    full = evaluate(bars)
    evaluations = 1
    divergences: list[Divergence] = []

    for point in sorted(points):
        truncated = evaluate(bars.head(point + 1))
        evaluations += 1
        divergences.extend(
            _compare(
                probe="truncation",
                at=point,
                reference=full,
                observed=truncated,
                signal_horizon=point + 1,
                position_horizon=point,
                warmup_bars=warmup_bars,
            )
        )

    tail_start = 0
    for mode in TAIL_MODES:
        tail_bars, tail_start = perturbed_tail(bars, tail_fraction, mode)
        replaced = evaluate(tail_bars)
        evaluations += 1
        divergences.extend(
            _compare(
                probe="reversed_tail",
                at=tail_start,
                reference=full,
                observed=replaced,
                # Sound horizon: a signal recorded at bar i was computed from data
                # up to i - 1 + lag, so it is unchanged exactly while
                # i - 1 + lag < tail_start.
                signal_horizon=tail_start + lag,
                position_horizon=tail_start,
                warmup_bars=warmup_bars,
                mode=mode,
            )
        )

    return ProbeResult(
        passed=not divergences,
        n_bars=bars.n_bars,
        style=style,
        warmup_bars=warmup_bars,
        cut_points=tuple(sorted(points)),
        tail_start=tail_start,
        divergences=tuple(divergences),
        n_evaluations=evaluations,
    )


def truncation_probe(
    strategy: Any,
    bars: BarFrame,
    params: Mapping[str, Any],
    *,
    cut_points: Sequence[int] | None = None,
    engine: Any,
    config: BacktestConfig | None = None,
) -> ProbeResult:
    """The signature of spec section 14.2, for a strategy that may run in-process.

    Untrusted code must not (INV-4): the loader builds an ``evaluate`` that goes
    through the sandbox and calls :func:`probe_evaluations` directly. This is the
    convenience form for the platform's own strategies and for tests.
    """
    settings = config or BacktestConfig()

    def evaluate(segment: BarFrame) -> BacktestResult:
        return engine.run(strategy, segment, params, settings)  # type: ignore[no-any-return]

    return probe_evaluations(
        evaluate,
        bars,
        style=str(getattr(strategy, "style", "bar_loop")),
        warmup_bars=int(getattr(strategy, "warmup_bars", 0)),
        cut_points=cut_points,
    )


def require_causal(result: ProbeResult, *, strategy_id: str = "") -> ProbeResult:
    """Return ``result`` if it passed, otherwise refuse.

    Raises:
        LeakageDetected: the probe found a divergence. A hard reject (spec section
            18.1): a strategy that changed its mind about the past has no
            backtest worth reading, so there is nothing to fall back to.
    """
    if not result.passed:
        raise LeakageDetected(
            f"the truncation probe proved this strategy is not causal:\n{result.describe()}",
            probe=result,
            strategy_id=strategy_id,
            n_divergences=len(result.divergences),
        )
    return result
