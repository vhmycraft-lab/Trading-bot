"""What a parameter search maximises (master spec sections 13.1, 13.8).

Four objectives, and one rule about the ones that are missing: **return-only
objectives are forbidden**. Maximising net return or CAGR over a parameter grid
finds the parameters that happened to catch the largest moves in the training
window, which is curve fitting with extra steps. Every objective here prices risk
as well as reward, so a search cannot buy a higher score with a larger drawdown.

The ban is enforced at configuration load — ``objective: net_return`` raises
``ConfigError`` before anything runs — and again here, because this module is also
what section 13.3's ``p_sensitivity`` uses to score a neighbourhood, and a
neighbourhood scored on return would penalise the wrong thing.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Final

from quantlab.core.config import FORBIDDEN_OBJECTIVES, VALID_OBJECTIVES, OptimizeSettings
from quantlab.core.errors import ConfigError
from quantlab.core.metrics import MetricSet

__all__ = [
    "OBJECTIVE_FUNCTIONS",
    "REJECTED",
    "objective_value",
    "require_allowed_objective",
]

#: The score of a parameter set that failed a guard.
#:
#: Far below any attainable objective, so a rejected trial sorts last under a
#: plain descending sort and a sampler learns to avoid that region rather than
#: being handed a ``None`` it has to special-case.
REJECTED: Final[float] = -1e9


def require_allowed_objective(name: str) -> str:
    """Return ``name``, or refuse it.

    Raises:
        ConfigError: the objective is return-only, or is not one of the four.
            Both are configuration mistakes, and the second is usually a typo for
            the first.
    """
    if name in FORBIDDEN_OBJECTIVES:
        raise ConfigError(
            "return-only objectives are forbidden: maximising return over a parameter "
            "grid finds the parameters that caught the largest moves in the training "
            "window, which is curve fitting",
            objective=name,
            allowed=list(VALID_OBJECTIVES),
        )
    if name not in VALID_OBJECTIVES:
        raise ConfigError(
            "unknown optimisation objective", objective=name, allowed=list(VALID_OBJECTIVES)
        )
    return name


def _sortino_dd(metrics: MetricSet, settings: OptimizeSettings) -> float | None:
    """``sortino - dd_lambda * max_drawdown`` — the default (section 13.8).

    Risk enters twice on purpose: the Sortino ratio already divides by downside
    deviation, and the drawdown term prices the *path*, which a ratio of averages
    cannot see. A run that reached the same Sortino through one 40 % drawdown is
    not the same run.
    """
    if metrics.sortino is None or metrics.max_drawdown is None:
        return None
    return float(metrics.sortino) - settings.dd_lambda * float(metrics.max_drawdown)


#: Each objective, by the name configuration uses. ``None`` means undefined for
#: this run, which :func:`objective_value` turns into :data:`REJECTED`.
OBJECTIVE_FUNCTIONS: Final[Mapping[str, Callable[[MetricSet, OptimizeSettings], float | None]]] = {
    "sortino_dd": _sortino_dd,
    "sharpe": lambda metrics, _settings: metrics.sharpe,
    "calmar": lambda metrics, _settings: metrics.calmar,
    "expectancy": lambda metrics, _settings: metrics.expectancy_pct,
}


def objective_value(metrics: MetricSet, settings: OptimizeSettings) -> float:
    """Score one evaluation, or :data:`REJECTED` if it does not qualify.

    Two guards, both rejections rather than penalties:

    * **Too few trades.** Below ``optimize.min_trades`` the metrics are estimates
      from a handful of observations. A search allowed to maximise them would
      find the parameters that produced three lucky trades, which is the purest
      form of the thing this module exists to prevent.
    * **Undefined.** A metric that could not be computed has not been
      demonstrated. Scoring it as zero would rank it above a strategy that was
      measured and found slightly negative.
    """
    require_allowed_objective(settings.objective)
    if metrics.n_trades < settings.min_trades:
        return REJECTED
    value = OBJECTIVE_FUNCTIONS[settings.objective](metrics, settings)
    return REJECTED if value is None else float(value)
