"""Property tests for fitness and concentration (master spec sections 13.3, 14.4).

Section 20 names two of these directly: fitness lies in ``[0, 1]`` or equals
``FITNESS_REJECTED`` for *any* metric set, and ``retention_k`` is monotonically
non-increasing in ``k``. Both are the kind of claim a table of examples cannot
establish — the interesting inputs are the ones nobody thought to write down.
"""

from __future__ import annotations

import itertools

from hypothesis import given, settings
from hypothesis import strategies as st

from quantlab.core.config import FitnessSettings
from quantlab.core.fitness import (
    FITNESS_REJECTED,
    InnerFold,
    InnerFoldReport,
    SensitivityReport,
    compute_fitness,
)
from quantlab.core.metrics import MetricSet
from quantlab.core.types import Trade
from quantlab.core.validation.concentration import (
    remaining_after_removal,
    trade_removal_report,
)

SETTINGS = FitnessSettings()

_optional_floats = st.one_of(
    st.none(), st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
)
_fractions = st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0))

_metric_sets = st.builds(
    MetricSet,
    expectancy_pct=_optional_floats,
    n_trades=st.floats(min_value=0.0, max_value=5_000.0),
    max_drawdown=_fractions,
    sortino=_optional_floats,
    profit_factor=_optional_floats,
    consistency=_fractions,
    win_rate=_fractions,
    cagr=_optional_floats,
    top5_profit_share=_fractions,
)

_folds = st.lists(
    st.builds(
        InnerFold,
        is_sortino=_optional_floats,
        oos_sortino=_optional_floats,
        oos_return=_optional_floats,
    ),
    max_size=6,
)

_sensitivity = st.one_of(
    st.none(),
    st.builds(
        SensitivityReport,
        median_drop=_optional_floats,
        n_neighbours=st.integers(min_value=0, max_value=50),
    ),
)


def _trade(no: int, pnl: float) -> Trade:
    return Trade(
        trade_no=no,
        side="long",
        entry_ts=1_600_000_000_000 + no * 3_600_000,
        entry_px=100.0,
        exit_ts=1_600_000_000_000 + (no + 1) * 3_600_000,
        exit_px=100.0 + pnl,
        qty=1.0,
        fees=0.0,
        slippage_cost=0.0,
        pnl=pnl,
        pnl_pct=pnl / 100.0,
        bars_held=1,
        exit_reason="signal",
    )


_pnls = st.lists(
    st.floats(min_value=-500.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    max_size=40,
)


@st.composite
def _ledgers(draw: st.DrawFn) -> list[Trade]:
    return [_trade(i, pnl) for i, pnl in enumerate(draw(_pnls), start=1)]


@given(metrics=_metric_sets, folds=_folds, ledger=_ledgers(), sensitivity=_sensitivity)
@settings(max_examples=250, deadline=None)
def test_fitness_is_in_the_unit_interval_or_rejected(
    metrics: MetricSet,
    folds: list[InnerFold],
    ledger: list[Trade],
    sensitivity: SensitivityReport | None,
) -> None:
    """Section 20's stated property. A score outside ``[0, 1]`` would make two
    candidates incomparable, which is the one thing a ranking must not be."""
    result = compute_fitness(
        metrics, ledger, InnerFoldReport(folds=tuple(folds)), sensitivity, SETTINGS
    )
    if result.rejected:
        assert result.fitness == FITNESS_REJECTED
        assert result.gate_failure is not None
    else:
        assert 0.0 <= result.fitness <= 1.0
        assert 0.0 <= result.base_score <= 1.0
        assert all(0.0 <= value <= 1.0 for value in result.components.values())
        assert all(0.0 <= value <= 1.0 for value in result.penalties.values())


@given(metrics=_metric_sets, folds=_folds, ledger=_ledgers())
@settings(max_examples=150, deadline=None)
def test_a_penalty_can_only_lower_a_score(
    metrics: MetricSet, folds: list[InnerFold], ledger: list[Trade]
) -> None:
    """Penalties multiply values in ``[0, 1]``, so fitness never exceeds the
    base score — the property that makes "penalised" mean what it says."""
    result = compute_fitness(metrics, ledger, InnerFoldReport(folds=tuple(folds)), None, SETTINGS)
    if not result.rejected:
        assert result.fitness <= result.base_score + 1e-12


@given(ledger=_ledgers(), depth=st.integers(min_value=1, max_value=12))
@settings(max_examples=200, deadline=None)
def test_retention_is_non_increasing_in_k(ledger: list[Trade], depth: int) -> None:
    """Section 20's other stated property. Removing more winners can only remove
    more profit, so a retention curve that rose would mean the removal picked the
    wrong trades."""
    report = trade_removal_report(ledger, tuple(range(1, depth + 1)))
    values = [report.retention[k] for k in range(1, depth + 1)]
    defined = [value for value in values if value is not None]
    assert len(defined) in (0, len(values))
    for earlier, later in itertools.pairwise(defined):
        assert later <= earlier + 1e-12


@given(ledger=_ledgers())
@settings(max_examples=200, deadline=None)
def test_removing_more_than_there_are_winners_removes_every_winner(
    ledger: list[Trade],
) -> None:
    """Losses are never removed, so the curve flattens rather than continuing to
    fall once the winners run out."""
    losers = [t for t in ledger if t.pnl <= 0]
    assert remaining_after_removal(ledger, len(ledger) + 5) == losers
