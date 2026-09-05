"""Multi-objective fitness for a candidate (master spec section 13.3).

Fitness is **gated, weighted, and penalised**, in that order, and the order is
the design rather than an implementation detail.

* **Gates** run first and are absolute. A candidate that fails one scores
  :data:`FITNESS_REJECTED` and nothing else is computed. This is what makes
  section 13's required property true as *control flow* rather than as a hopeful
  weighting: win rate can never override negative expectancy or an excessive
  drawdown, because win rate is only ever an input to a score the rejected
  candidate never reaches.
* **Weights** combine ten components, each mapped into ``[0, 1]`` so no term can
  dominate through its unit. Net profit is one of the ten and carries the
  second-smallest weight: a strategy cannot climb the ranking by making more
  money in a less trustworthy way.
* **Penalties** are multiplicative, not subtractive. Two independent robustness
  problems compound, and a candidate that is fragile in several ways at once
  should fall far rather than twice as little as it fell once.

The result carries every component and every penalty, not just the number. A
ranking nobody can take apart is a ranking nobody can argue with, and section
13.3's whole point is that the argument is where the information is.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

from quantlab.core.config import FitnessSettings
from quantlab.core.errors import ConfigError
from quantlab.core.metrics import MetricSet
from quantlab.core.types import Trade
from quantlab.core.validation.concentration import TradeRemovalReport, trade_removal_report

__all__ = [
    "FITNESS_REJECTED",
    "GATE_IDS",
    "FitnessResult",
    "InnerFold",
    "InnerFoldReport",
    "SensitivityReport",
    "compute_fitness",
]

#: The score of a candidate that failed a hard gate (spec section 13.3).
#:
#: Far below any attainable fitness — which lives in ``[0, 1]`` — so a rejected
#: candidate sorts last under a plain descending sort, with no special-casing at
#: every comparison site.
FITNESS_REJECTED: Final[float] = -1e9

#: Every gate, in the order section 13.3 tabulates them.
GATE_IDS: Final[tuple[str, ...]] = (
    "F_EXPECTANCY",
    "F_TRADES",
    "F_DRAWDOWN",
    "F_CONCENTRATION",
)

_EPS: Final[float] = 1e-12


def _clip(value: float) -> float:
    """``min(1, max(0, value))`` — section 13.3's ``clip``."""
    return min(1.0, max(0.0, value))


def _ratio(numerator: float | None, target: float) -> float:
    """A component as a fraction of the value at which it reaches 1.0.

    A missing metric scores 0. That is the conservative reading: a component that
    could not be measured has not been demonstrated, and scoring it as absent is
    the difference between "no evidence" and "no problem".
    """
    if numerator is None or target <= _EPS:
        return 0.0
    return _clip(float(numerator) / target)


# ---------------------------------------------------------------------------
# inputs produced elsewhere
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class InnerFold:
    """One in-sample/out-of-sample fold from inside train (spec section 13.6)."""

    is_sortino: float | None = None
    oos_sortino: float | None = None
    oos_return: float | None = None


@dataclass(frozen=True, slots=True)
class InnerFoldReport:
    """Generalisation measured inside train, never against validation.

    Section 13.6: the train segment is split into folds with their own embargo so
    that fitness can reward generalisation without spending the validation
    segment once per candidate per generation. This carries the per-fold numbers;
    the three aggregates section 13.3 asks for are derived here, in one place.

    An empty report is the honest state before the inner walk-forward has run,
    and every aggregate is then ``None`` rather than a flattering default.
    """

    folds: tuple[InnerFold, ...] = ()

    @staticmethod
    def _mean(values: Sequence[float | None]) -> float | None:
        defined = [float(value) for value in values if value is not None]
        return sum(defined) / len(defined) if defined else None

    @property
    def is_sortino(self) -> float | None:
        """Mean in-sample Sortino across folds."""
        return self._mean([fold.is_sortino for fold in self.folds])

    @property
    def oos_sortino(self) -> float | None:
        """Mean out-of-sample Sortino across folds."""
        return self._mean([fold.oos_sortino for fold in self.folds])

    @property
    def return_cv(self) -> float | None:
        """Coefficient of variation of per-fold OOS return (spec section 13.3).

        ``None`` with fewer than two defined folds — one number has no dispersion
        — and ``None`` when the mean is ~0, where the ratio is unbounded and would
        report an arbitrary penalty rather than an instability.
        """
        returns = [float(fold.oos_return) for fold in self.folds if fold.oos_return is not None]
        if len(returns) < 2:
            return None
        mean = sum(returns) / len(returns)
        if abs(mean) <= _EPS:
            return None
        variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
        return math.sqrt(variance) / abs(mean)


@dataclass(frozen=True, slots=True)
class SensitivityReport:
    """How far the objective falls around the chosen parameters (section 13.8)."""

    #: Median relative drop of the neighbourhood objective, as a fraction.
    median_drop: float | None = None
    n_neighbours: int = 0


# ---------------------------------------------------------------------------
# the result
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FitnessResult:
    """A candidate's fitness and everything that produced it."""

    fitness: float
    base_score: float = 0.0
    components: Mapping[str, float] = field(default_factory=dict)
    penalties: Mapping[str, float] = field(default_factory=dict)
    #: The gate that rejected the candidate, or ``None``. Stored in
    #: ``candidate.gate_failure`` (spec section 6).
    gate_failure: str | None = None
    removal: TradeRemovalReport | None = None

    @property
    def rejected(self) -> bool:
        return self.gate_failure is not None

    @property
    def penalty_product(self) -> float:
        product = 1.0
        for value in self.penalties.values():
            product *= value
        return product


# ---------------------------------------------------------------------------
# stage 1 — hard gates
# ---------------------------------------------------------------------------
def _failed_gate(
    metrics: MetricSet, removal: TradeRemovalReport, settings: FitnessSettings
) -> str | None:
    """The first gate this candidate fails, or ``None`` (spec section 13.3).

    Order follows the specification's table. Reporting the *first* failure rather
    than all of them is deliberate here — unlike the AST checker, which reports
    every violation because a strategy author is going to fix them all. A rejected
    candidate is discarded, so one reason is one reason more than is needed to
    discard it, and the cheapest reason is the most useful one to record.
    """
    gates = settings.gates
    expectancy = metrics.expectancy_pct
    if expectancy is None or expectancy <= gates.min_expectancy_pct:
        return "F_EXPECTANCY"
    if metrics.n_trades < gates.min_trades:
        return "F_TRADES"
    drawdown = metrics.max_drawdown
    if drawdown is not None and drawdown > gates.max_drawdown:
        return "F_DRAWDOWN"
    retention_1 = removal.retention_at(1)
    if retention_1 is not None and retention_1 <= gates.min_retention_top1:
        return "F_CONCENTRATION"
    return None


# ---------------------------------------------------------------------------
# stage 2 — weighted base score
# ---------------------------------------------------------------------------
def _components(
    metrics: MetricSet, inner: InnerFoldReport, settings: FitnessSettings
) -> dict[str, float]:
    """Every component of section 13.3's table, each already in ``[0, 1]``."""
    targets = settings.targets
    profit_factor = metrics.profit_factor
    return {
        "expectancy": _ratio(metrics.expectancy_pct, targets.expectancy_pct),
        "risk_adjusted": _ratio(metrics.sortino, targets.sortino),
        "drawdown": _drawdown_component(metrics.max_drawdown, targets.drawdown_ceiling),
        # `None` means there were no losing trades at all, which section 13.3
        # scores as 1.0 while forcing the low-trade warning: the ratio is
        # undefined rather than infinite, and the warning is where that is said.
        "profit_factor": (
            1.0
            if profit_factor is None
            else _clip((profit_factor - 1.0) / (targets.profit_factor - 1.0))
        ),
        "consistency": _clip(metrics.consistency or 0.0),
        "inner_oos": _inner_oos_component(inner),
        "trades": _clip(math.log1p(metrics.n_trades) / math.log1p(targets.trades)),
        "win_rate": _win_rate_component(metrics.win_rate, settings),
        "net_return": _ratio(metrics.cagr, targets.cagr),
        "concentration": _clip(1.0 - (metrics.top5_profit_share or 0.0)),
    }


def _drawdown_component(max_drawdown: float | None, ceiling: float) -> float:
    """``(ceiling - drawdown) / ceiling``, so a flat curve scores 1 and the
    ceiling scores 0. An unmeasured drawdown scores 0, as everywhere else."""
    if max_drawdown is None:
        return 0.0
    return _clip((ceiling - float(max_drawdown)) / ceiling)


def _inner_oos_component(inner: InnerFoldReport) -> float:
    """``oos_sortino / max(is_sortino, eps)`` (spec section 13.3).

    Zero when the inner walk-forward has not run: generalisation that was never
    measured has not been demonstrated. Zero as well when in-sample Sortino is not
    positive, where the ratio would reward a candidate for being equally bad on
    both sides of the fold.
    """
    is_sortino, oos_sortino = inner.is_sortino, inner.oos_sortino
    if is_sortino is None or oos_sortino is None or is_sortino <= _EPS:
        return 0.0
    return _clip(oos_sortino / is_sortino)


def _win_rate_component(win_rate: float | None, settings: FitnessSettings) -> float:
    targets = settings.targets
    if win_rate is None:
        return 0.0
    span = targets.win_rate - targets.win_rate_floor
    return _clip((float(win_rate) - targets.win_rate_floor) / span)


# ---------------------------------------------------------------------------
# stage 3 — multiplicative penalties
# ---------------------------------------------------------------------------
def _penalties(
    metrics: MetricSet,
    inner: InnerFoldReport,
    sensitivity: SensitivityReport | None,
    removal: TradeRemovalReport,
    settings: FitnessSettings,
    *,
    n_free_params: int,
    logic_lines: int,
) -> dict[str, float]:
    """Every penalty of section 13.3's table, each in ``[0, 1]``.

    A penalty that does not apply is ``1.0`` and is still reported, so the product
    can be read off the result and a missing penalty is visibly absent rather than
    silently omitted.
    """
    penalties = settings.penalties
    return {
        "p_drawdown": _linear_decay(
            metrics.max_drawdown, soft=penalties.drawdown_soft, hard=settings.gates.max_drawdown
        ),
        "p_trades": (
            1.0
            if metrics.n_trades >= penalties.trades_soft
            else _clip(metrics.n_trades / penalties.trades_soft)
            if penalties.trades_soft > 0
            else 1.0
        ),
        "p_instability": _decay_above(inner.return_cv, penalties.instability_max_cv),
        "p_sensitivity": _decay_above(
            None if sensitivity is None else sensitivity.median_drop,
            penalties.sensitivity_max_drop,
        ),
        "p_removal": _removal_penalty(removal, settings),
        "p_divergence": _divergence_penalty(inner, penalties.divergence_min_ratio),
        "p_complexity": _complexity_penalty(
            n_free_params=n_free_params,
            logic_lines=logic_lines,
            max_params=penalties.complexity_free_params,
            max_lines=penalties.complexity_logic_lines,
        ),
    }


def _linear_decay(value: float | None, *, soft: float, hard: float) -> float:
    """1.0 at or below ``soft``, falling linearly to 0.0 at ``hard``.

    ``hard`` is the gate, so a value that reaches it has already been rejected;
    the ramp exists to make the approach to it expensive rather than free.
    """
    if value is None or value <= soft:
        return 1.0
    if hard <= soft:  # pragma: no cover - config validation forbids this
        return 0.0
    return _clip((hard - float(value)) / (hard - soft))


def _decay_above(value: float | None, threshold: float) -> float:
    """1.0 at or below ``threshold``, decaying linearly to 0.0 at twice it.

    Section 13.3 says "linear decay" without naming an endpoint. Twice the
    threshold is chosen because it is the one endpoint that needs no second
    configuration knob and cannot be set inconsistently with the threshold it
    decays from.
    """
    if value is None or value <= threshold:
        return 1.0
    if threshold <= _EPS:  # pragma: no cover - config validation forbids this
        return 0.0
    return _clip((2.0 * threshold - float(value)) / threshold)


def _removal_penalty(removal: TradeRemovalReport, settings: FitnessSettings) -> float:
    """``min_k clip((ret_k - floor_k) / (target_k - floor_k))`` (section 13.3).

    The minimum across depths, not the mean: a strategy that survives the loss of
    its best trade but collapses on the loss of its best five is concentrated, and
    averaging that away is exactly the mistake this test exists to prevent.
    """
    penalties = settings.penalties
    worst = 1.0
    for k, floor, target in zip(
        penalties.removal_k, penalties.removal_floor, penalties.removal_target, strict=True
    ):
        retention = removal.retention_at(k)
        if retention is None:
            continue
        worst = min(worst, _clip((retention - floor) / (target - floor)))
    return worst


def _divergence_penalty(inner: InnerFoldReport, min_ratio: float) -> float:
    """Decays as the inner OOS/IS Sortino ratio falls below ``min_ratio``.

    ``1.0`` when the inner walk-forward has not run: this penalty punishes a
    *measured* divergence, and the absence of the measurement is already priced
    into the ``inner_oos`` component, which scores 0 there. Charging for it twice
    would make an unmeasured candidate look worse than a demonstrably divergent
    one.
    """
    is_sortino, oos_sortino = inner.is_sortino, inner.oos_sortino
    if is_sortino is None or oos_sortino is None or is_sortino <= _EPS:
        return 1.0
    ratio = oos_sortino / is_sortino
    if ratio >= min_ratio:
        return 1.0
    if min_ratio <= _EPS:  # pragma: no cover - config validation forbids this
        return 1.0
    return _clip(max(0.0, ratio) / min_ratio)


def _complexity_penalty(
    *, n_free_params: int, logic_lines: int, max_params: int, max_lines: int
) -> float:
    """``0.95 ** excess`` (spec section 13.3).

    Excess counts parameters above the limit plus, once, a body above the line
    limit. The base is soft on purpose: complexity is a smell, not a defect, and
    the hard limits live in section 9.2 where a strategy is refused outright.
    """
    excess = max(0, int(n_free_params) - int(max_params))
    if int(logic_lines) > int(max_lines):
        excess += 1
    return 0.95**excess


def _require_matching_keys(weights: Mapping[str, float], components: Mapping[str, float]) -> None:
    """Every weight scores a component and every component carries a weight.

    The weights sum to 1 (validated at config load), so a weight with no
    component would silently shrink the maximum attainable score and a component
    with no weight would silently count for nothing. Neither would raise on its
    own — the sum simply comes out smaller — which is exactly the kind of quiet
    drift that makes two runs incomparable.
    """
    missing = sorted(set(weights) - set(components))
    extra = sorted(set(components) - set(weights))
    if missing or extra:  # pragma: no cover - a constant, asserted by a test
        raise ConfigError(
            "fitness weights and components must name the same set",
            weights_without_component=missing,
            components_without_weight=extra,
        )


# ---------------------------------------------------------------------------
# the public entry point
# ---------------------------------------------------------------------------
def compute_fitness(
    metrics: MetricSet,
    trades: Sequence[Trade],
    inner: InnerFoldReport,
    sensitivity: SensitivityReport | None,
    settings: FitnessSettings,
    *,
    n_free_params: int = 0,
    logic_lines: int = 0,
) -> FitnessResult:
    """Score one candidate (spec section 13.3).

    Args:
        metrics: The candidate's metrics on the segment it was evaluated on.
        trades: Its trade ledger, for the removal test of section 14.4.
        inner: Per-fold generalisation from inside train (section 13.6). Pass an
            empty :class:`InnerFoldReport` before the inner walk-forward exists;
            the ``inner_oos`` component then scores 0 rather than a default that
            would flatter an unmeasured candidate.
        sensitivity: The neighbourhood report of section 13.8, or ``None``.
        settings: ``evolution.fitness``. Its own validators guarantee the weights
            sum to 1 and the soft thresholds sit inside the hard ones, so nothing
            here re-checks them.
        n_free_params: Declared parameter count, for ``p_complexity``.
        logic_lines: Body length from the AST check, for ``p_complexity``.

    Returns:
        A :class:`FitnessResult` in ``[0, 1]``, or one carrying
        :data:`FITNESS_REJECTED` and the gate that rejected it.
    """
    removal = trade_removal_report(trades, settings.penalties.removal_k)

    gate_failure = _failed_gate(metrics, removal, settings)
    if gate_failure is not None:
        return FitnessResult(fitness=FITNESS_REJECTED, gate_failure=gate_failure, removal=removal)

    components = _components(metrics, inner, settings)
    weights = settings.weights.as_dict()
    _require_matching_keys(weights, components)
    base_score = sum(weights[name] * value for name, value in components.items())

    penalties = _penalties(
        metrics,
        inner,
        sensitivity,
        removal,
        settings,
        n_free_params=n_free_params,
        logic_lines=logic_lines,
    )
    fitness = base_score
    for value in penalties.values():
        fitness *= value

    return FitnessResult(
        fitness=fitness,
        base_score=base_score,
        components=components,
        penalties=penalties,
        removal=removal,
    )
