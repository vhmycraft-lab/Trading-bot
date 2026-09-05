"""What a parameter search may maximise (master spec sections 13.1, 13.8, T51).

Section 22's acceptance criterion is short — ``objective: net_return`` raises
``ConfigError`` — and the reason is the whole point of the module: maximising
return over a parameter grid finds the parameters that caught the largest moves in
the training window. Every objective that remains prices risk as well as reward.
"""

from __future__ import annotations

import pytest

from quantlab.core.config import FORBIDDEN_OBJECTIVES, VALID_OBJECTIVES, OptimizeSettings
from quantlab.core.errors import ConfigError
from quantlab.core.metrics import MetricSet
from quantlab.optimize.objectives import (
    OBJECTIVE_FUNCTIONS,
    REJECTED,
    objective_value,
    require_allowed_objective,
)

SETTINGS = OptimizeSettings()


def metrics(**overrides: object) -> MetricSet:
    base: dict[str, object] = {
        "n_trades": 100.0,
        "sortino": 1.5,
        "sharpe": 1.0,
        "calmar": 0.8,
        "expectancy_pct": 0.002,
        "max_drawdown": 0.20,
    }
    base.update(overrides)
    return MetricSet(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# the ban
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", FORBIDDEN_OBJECTIVES)
def test_a_return_only_objective_is_refused(name: str) -> None:
    """Section 22's stated criterion, for every banned name — not just the one
    the specification happens to spell out."""
    with pytest.raises(ConfigError, match="return-only objectives are forbidden"):
        require_allowed_objective(name)
    with pytest.raises(ConfigError, match="return-only objectives are forbidden"):
        OptimizeSettings(objective=name)


def test_net_return_is_among_the_banned() -> None:
    assert "net_return" in FORBIDDEN_OBJECTIVES
    assert "cagr" in FORBIDDEN_OBJECTIVES


def test_an_unknown_objective_is_refused_with_the_allowed_list() -> None:
    """Usually a typo for a banned one, so the message names what is allowed."""
    with pytest.raises(ConfigError, match="unknown optimisation objective"):
        require_allowed_objective("profit")


def test_every_valid_objective_has_a_function() -> None:
    """A configured objective with no implementation would fail at the first
    trial, hours into a search."""
    assert set(OBJECTIVE_FUNCTIONS) == set(VALID_OBJECTIVES)


def test_the_ban_is_enforced_at_scoring_time_as_well_as_at_config_load() -> None:
    """This module also scores the neighbourhood of section 13.3's
    ``p_sensitivity``, which does not come through config validation."""
    settings = OptimizeSettings()
    object.__setattr__(settings, "objective", "net_return")
    with pytest.raises(ConfigError, match="return-only"):
        objective_value(metrics(), settings)


# ---------------------------------------------------------------------------
# the objectives themselves
# ---------------------------------------------------------------------------
def test_sortino_dd_prices_the_path_as_well_as_the_ratio() -> None:
    """``sortino - dd_lambda * max_drawdown``: two runs with the same Sortino and
    different drawdowns are not the same run."""
    shallow = objective_value(metrics(max_drawdown=0.10), SETTINGS)
    deep = objective_value(metrics(max_drawdown=0.40), SETTINGS)
    assert shallow == pytest.approx(1.5 - 0.5 * 0.10)
    assert deep == pytest.approx(1.5 - 0.5 * 0.40)
    assert shallow > deep


@pytest.mark.parametrize(
    ("objective", "field", "value"),
    [("sharpe", "sharpe", 1.25), ("calmar", "calmar", 0.6), ("expectancy", "expectancy_pct", 0.01)],
)
def test_each_simple_objective_reads_its_own_metric(
    objective: str, field: str, value: float
) -> None:
    settings = OptimizeSettings(objective=objective)
    assert objective_value(metrics(**{field: value}), settings) == pytest.approx(value)


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------
def test_too_few_trades_is_rejected_rather_than_scored() -> None:
    """Below ``min_trades`` the metrics are estimates from a handful of
    observations, and a search allowed to maximise them would find the three
    lucky trades."""
    assert objective_value(metrics(n_trades=5.0), SETTINGS) == REJECTED
    assert objective_value(metrics(n_trades=float(SETTINGS.min_trades)), SETTINGS) > REJECTED


def test_an_undefined_metric_is_rejected_not_scored_as_zero() -> None:
    """Zero would rank an unmeasured strategy above one measured and found
    slightly negative."""
    assert objective_value(metrics(sortino=None), SETTINGS) == REJECTED
    assert objective_value(metrics(max_drawdown=None), SETTINGS) == REJECTED
    assert objective_value(metrics(sharpe=None), OptimizeSettings(objective="sharpe")) == REJECTED


def test_a_rejected_trial_sorts_below_every_real_one() -> None:
    """So a sampler learns to avoid the region instead of special-casing a
    ``None``."""
    real = objective_value(metrics(sortino=-5.0), SETTINGS)
    assert real > REJECTED
    assert sorted([real, REJECTED])[0] == REJECTED


def test_the_guard_runs_before_the_objective_is_computed() -> None:
    """A run with too few trades is rejected even if its metrics are undefined,
    so the reason reported is the real one."""
    assert objective_value(metrics(n_trades=1.0, sortino=None), SETTINGS) == REJECTED


def test_a_lower_min_trades_admits_a_shorter_run() -> None:
    lenient = OptimizeSettings(min_trades=0)
    assert objective_value(metrics(n_trades=1.0), lenient) > REJECTED
