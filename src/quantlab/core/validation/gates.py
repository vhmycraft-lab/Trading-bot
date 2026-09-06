"""Hard gates (master spec section 14.3).

Seven absolute rules. Failing any one of them is a ``REJECT``, and no soft check,
no score and no amount of good performance elsewhere can compensate — that is
what makes them *gates* rather than another weighted term.

The gates sit at the boundary between search and judgement, and the direction of
information matters as much as the rules. Everything here **consumes** validation
results; nothing here is ever consulted by the evolutionary search, which by
construction cannot even produce the numbers these gates read (INV-9 confines it
to the train segment). ``tests/unit/test_architecture.py`` holds that separation
as an import rule, so a future convenience cannot quietly wire a gate's verdict
back into fitness.

Every gate reports its observation and its threshold alongside its verdict, so a
rejection can be read and argued with rather than merely obeyed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from quantlab.core.config import ValidationSettings
from quantlab.core.metrics import MetricSet
from quantlab.core.types import Trade

__all__ = [
    "GATE_IDS",
    "GateInputs",
    "GateReport",
    "GateResult",
    "evaluate_gates",
    "gate_thresholds",
]

#: Every gate, in the order section 14.3 tabulates them.
GATE_IDS: Final[tuple[str, ...]] = (
    "G_LEAK",
    "G_MIN_TRADES",
    "G_SANITY",
    "G_COST",
    "G_PERM",
    "G_DEGRADE",
    "G_BENCH",
)

_EPS: Final[float] = 1e-12


@dataclass(frozen=True, slots=True)
class GateResult:
    """One gate's verdict, and the numbers behind it."""

    gate_id: str
    passed: bool
    observed: Any = None
    threshold: Any = None
    #: Why, in a sentence, for the report of section 14.1 step 9.
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "observed": self.observed,
            "threshold": self.threshold,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class GateReport:
    """Every gate's verdict (``hard_gates_json``, spec section 6)."""

    results: tuple[GateResult, ...] = ()

    @property
    def passed(self) -> bool:
        """True only when every gate passed. Section 14.4: any fail is REJECT."""
        return all(result.passed for result in self.results)

    @property
    def failed(self) -> tuple[str, ...]:
        return tuple(result.gate_id for result in self.results if not result.passed)

    def __getitem__(self, gate_id: str) -> GateResult:
        for result in self.results:
            if result.gate_id == gate_id:
                return result
        raise KeyError(gate_id)

    def as_dict(self) -> dict[str, dict[str, Any]]:
        """Section 14.3's stated output shape."""
        return {result.gate_id: result.as_dict() for result in self.results}


@dataclass(frozen=True, slots=True)
class GateInputs:
    """Everything the seven gates read (spec section 14.1, steps 1-5b).

    Assembled by the pipeline, never by a gate: a gate that fetched its own
    inputs could reach for a segment it has no business reading, and this way the
    one place that touches the validation segment is the one place section 14.1
    says does.
    """

    #: Section 14.2's verdict. ``None`` means the probe was never run, which is
    #: not the same as passing and is treated as a failure.
    probe_passed: bool | None = None
    metrics_train: MetricSet = field(default_factory=MetricSet)
    metrics_val: MetricSet = field(default_factory=MetricSet)
    #: Validation metrics at ``cost_survival_multiplier`` times the base costs.
    metrics_val_stressed: MetricSet | None = None
    trades_train: Sequence[Trade] = ()
    trades_val: Sequence[Trade] = ()
    #: Realised exposure per bar on the validation segment.
    position_frac_val: Sequence[float] = ()
    #: Market-permutation p-value (section 14.4, check 3).
    permutation_p: float | None = None
    #: ``buy_and_hold`` on the validation segment (section 14.1, step 5).
    metrics_buyhold_val: MetricSet | None = None
    #: The cap the engine was configured with, for ``G_SANITY``.
    max_position_fraction: float = 1.0


def evaluate_gates(inputs: GateInputs, settings: ValidationSettings) -> GateReport:
    """Run every gate of section 14.3, in the order that table lists them.

    All seven always run, even after one has failed. A rejection that named only
    the first problem would send an author round the loop once per gate, and the
    cost of computing the rest is nothing next to the runs that produced the
    inputs.
    """
    return GateReport(
        results=(
            _leak(inputs),
            _min_trades(inputs, settings),
            _sanity(inputs, settings),
            _cost(inputs, settings),
            _permutation(inputs, settings),
            _degrade(inputs, settings),
            _benchmark(inputs),
        )
    )


# ---------------------------------------------------------------------------
# the gates
# ---------------------------------------------------------------------------
def _leak(inputs: GateInputs) -> GateResult:
    """``G_LEAK``: the truncation probe of section 14.2 passed.

    ``None`` — the probe was never run — fails. "Not checked" and "checked and
    clean" are different states, and only one of them is evidence.
    """
    passed = inputs.probe_passed is True
    return GateResult(
        gate_id="G_LEAK",
        passed=passed,
        observed=inputs.probe_passed,
        threshold=True,
        reason=(
            "the leakage probe passed"
            if passed
            else "the strategy is not causal, or the probe was never run"
        ),
    )


def _min_trades(inputs: GateInputs, settings: ValidationSettings) -> GateResult:
    """``G_MIN_TRADES``: enough trades on both segments to mean anything.

    Both, not either. A strategy with two hundred training trades and four
    validation trades has not been validated; it has been observed four times.
    """
    train, val = int(inputs.metrics_train.n_trades), int(inputs.metrics_val.n_trades)
    passed = val >= settings.min_trades_val and train >= settings.min_trades_train
    return GateResult(
        gate_id="G_MIN_TRADES",
        passed=passed,
        observed={"train": train, "val": val},
        threshold={"train": settings.min_trades_train, "val": settings.min_trades_val},
        reason=(
            "both segments carry enough trades"
            if passed
            else "too few trades for the metrics to mean anything"
        ),
    )


def _sanity(inputs: GateInputs, settings: ValidationSettings) -> GateResult:
    """``G_SANITY``: no impossible trade, and no impossible position.

    Two symptoms of the same disease — a backtest that is not simulating what it
    claims. A single trade returning 40 % is usually a data error or a fill at a
    price that never traded; a position larger than the configured cap is the
    engine sizing against equity it does not have.

    The position check carries ``sanity_drift_allowance`` because the cap bounds
    the *target* fraction at the deciding bar, and the realised fraction drifts
    with the market until the next rebalance. The exact invariant is the engine's
    and is checked by its property tests; this one catches the gross case.
    """
    worst_trade = max(
        (abs(trade.pnl_pct) for trade in (*inputs.trades_train, *inputs.trades_val)), default=0.0
    )
    fractions = np.abs(np.asarray(inputs.position_frac_val, dtype="float64"))
    worst_position = float(fractions.max()) if fractions.size else 0.0
    ceiling = inputs.max_position_fraction * (1.0 + settings.sanity_drift_allowance)

    trade_ok = worst_trade <= settings.max_single_trade_pct
    position_ok = worst_position <= ceiling + _EPS
    reasons = []
    if not trade_ok:
        reasons.append(f"a single trade returned {worst_trade:.1%}")
    if not position_ok:
        reasons.append(f"exposure reached {worst_position:.2f} against a ceiling of {ceiling:.2f}")

    return GateResult(
        gate_id="G_SANITY",
        passed=trade_ok and position_ok,
        observed={"max_abs_trade_pct": worst_trade, "max_abs_position_frac": worst_position},
        threshold={
            "max_single_trade_pct": settings.max_single_trade_pct,
            "max_position_frac": ceiling,
        },
        reason="; ".join(reasons) if reasons else "no impossible trade or position",
    )


def _cost(inputs: GateInputs, settings: ValidationSettings) -> GateResult:
    """``G_COST``: still profitable at ``cost_survival_multiplier`` times the costs.

    An edge that disappears when costs double was never an edge; it was an
    estimate of the spread. The multiplier is deliberately blunt — the point is
    not to model a worse market but to ask whether the result has any margin at
    all.
    """
    stressed = inputs.metrics_val_stressed
    observed = None if stressed is None else stressed.net_return
    passed = observed is not None and observed > 0.0
    return GateResult(
        gate_id="G_COST",
        passed=passed,
        observed=observed,
        threshold=0.0,
        reason=(
            f"profitable at {settings.cost_survival_multiplier:g}x costs"
            if passed
            else "the edge does not survive the cost stress, or was never measured under it"
        ),
    )


def _permutation(inputs: GateInputs, settings: ValidationSettings) -> GateResult:
    """``G_PERM``: the result is distinguishable from the same strategy on noise.

    A p-value that was never computed fails, for the same reason ``G_LEAK`` does:
    the gate asks for evidence, and an absent measurement is not evidence.
    """
    p_value = inputs.permutation_p
    alpha = settings.permutation.alpha
    passed = p_value is not None and p_value <= alpha
    return GateResult(
        gate_id="G_PERM",
        passed=passed,
        observed=p_value,
        threshold=alpha,
        reason=(
            "the result beats its own permutations"
            if passed
            else "indistinguishable from the same strategy run on permuted returns"
        ),
    )


def _degrade(inputs: GateInputs, settings: ValidationSettings) -> GateResult:
    """``G_DEGRADE``: validation performance is positive and not a collapse.

    Two conditions, and the first is the one that catches the common case:
    ``sharpe_val > 0``. Some degradation from train to validation is *expected*
    — it is what out-of-sample means — so the ratio permits it while the sign
    test refuses a strategy that only worked in the past.

    A non-positive train Sharpe makes the ratio meaningless: dividing by it would
    let a strategy that lost money on both segments pass by losing slightly less
    on one. Only the sign test then applies.
    """
    sharpe_val, sharpe_train = inputs.metrics_val.sharpe, inputs.metrics_train.sharpe
    positive = sharpe_val is not None and sharpe_val > 0.0
    if not positive:
        ratio_ok, floor = False, None
    elif sharpe_train is None or sharpe_train <= _EPS:
        ratio_ok, floor = True, None
    else:
        floor = settings.degradation_min_ratio * sharpe_train
        ratio_ok = float(sharpe_val or 0.0) >= floor

    return GateResult(
        gate_id="G_DEGRADE",
        passed=positive and ratio_ok,
        observed={"sharpe_val": sharpe_val, "sharpe_train": sharpe_train},
        threshold={"sharpe_val": 0.0, "sharpe_val_floor": floor},
        reason=(
            "validation performance held up"
            if positive and ratio_ok
            else "validation performance is non-positive or collapsed against train"
        ),
    )


def _benchmark(inputs: GateInputs) -> GateResult:
    """``G_BENCH``: better than buying and holding, on return *or* on risk.

    Either a higher Sortino, or a materially smaller drawdown while still making
    money. The disjunction is the point: a strategy that earns less than
    buy-and-hold but halves the drawdown is a different and legitimate product,
    and a gate that demanded both would reject it for succeeding at one.
    """
    baseline = inputs.metrics_buyhold_val
    if baseline is None:
        return GateResult(
            gate_id="G_BENCH",
            passed=False,
            observed=None,
            threshold=None,
            reason="no buy-and-hold baseline was measured on the validation segment",
        )

    sortino, sortino_bh = inputs.metrics_val.sortino, baseline.sortino
    beats_return = sortino is not None and sortino_bh is not None and sortino > sortino_bh

    drawdown, drawdown_bh = inputs.metrics_val.max_drawdown, baseline.max_drawdown
    net = inputs.metrics_val.net_return
    beats_risk = (
        drawdown is not None
        and drawdown_bh is not None
        and drawdown < 0.5 * drawdown_bh
        and net is not None
        and net > 0.0
    )

    return GateResult(
        gate_id="G_BENCH",
        passed=beats_return or beats_risk,
        observed={
            "sortino_val": sortino,
            "sortino_buyhold": sortino_bh,
            "max_drawdown_val": drawdown,
            "max_drawdown_buyhold": drawdown_bh,
            "net_return_val": net,
        },
        threshold={
            "sortino_buyhold": sortino_bh,
            "half_buyhold_drawdown": None if drawdown_bh is None else 0.5 * drawdown_bh,
        },
        reason=(
            "beats buy-and-hold on risk-adjusted return"
            if beats_return
            else "halves the drawdown while still making money"
            if beats_risk
            else "does not beat buying and holding on either return or risk"
        ),
    )


def gate_thresholds(settings: ValidationSettings) -> Mapping[str, Any]:
    """The thresholds a verdict was reached under (``thresholds_json``, §14.4).

    Snapshotted with the verdict because thresholds change: a verdict read a year
    later must be interpretable against the rules that produced it, not against
    today's configuration.
    """
    return {
        "min_trades_train": settings.min_trades_train,
        "min_trades_val": settings.min_trades_val,
        "max_single_trade_pct": settings.max_single_trade_pct,
        "sanity_drift_allowance": settings.sanity_drift_allowance,
        "cost_survival_multiplier": settings.cost_survival_multiplier,
        "permutation_alpha": settings.permutation.alpha,
        "degradation_min_ratio": settings.degradation_min_ratio,
    }
