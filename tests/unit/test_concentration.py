"""The trade-removal test (master spec section 14.4, task T47).

Section 14.4's claim is that concentration can be measured exactly, from the
trade ledger, without re-simulating anything. These tests hold it to that: every
expected number below is worked out on paper from a ledger written down in the
test, so a disagreement is a disagreement about arithmetic rather than about a
fixture.
"""

from __future__ import annotations

import itertools

import pytest

from quantlab.core.metrics import (
    retention_after_removing_top_winners,
    trade_expectancy_pct,
    trade_profit_factor,
)
from quantlab.core.types import Trade
from quantlab.core.validation.concentration import (
    DEFAULT_K_VALUES,
    remaining_after_removal,
    trade_removal_report,
)


def trade(no: int, pnl: float, pnl_pct: float | None = None) -> Trade:
    """A round trip carrying only what this test measures."""
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
        pnl_pct=pnl / 100.0 if pnl_pct is None else pnl_pct,
        bars_held=1,
        exit_reason="signal",
    )


#: Net pnl 100: one outlier of 60, ten winners of 5 (110 gross), ten losers of -1.
CONCENTRATED = [
    trade(1, 60.0),
    *[trade(i, 5.0) for i in range(2, 12)],
    *[trade(i, -1.0) for i in range(12, 22)],
]

#: Net pnl 100 as well, and the same gross: twenty-two equal winners, ten losers.
SPREAD = [
    *[trade(i, 5.0) for i in range(1, 23)],
    *[trade(i, -1.0) for i in range(23, 33)],
]


def test_a_concentrated_ledger_loses_most_of_its_profit_to_one_trade() -> None:
    report = trade_removal_report(CONCENTRATED)
    assert report.n_trades == 21
    assert report.k_values == DEFAULT_K_VALUES
    # 100 - 60 = 40; 100 - (60+5+5) = 30; 100 - (60+5*4) = 20
    assert report.retention[1] == pytest.approx(0.40)
    assert report.retention[3] == pytest.approx(0.30)
    assert report.retention[5] == pytest.approx(0.20)


def test_a_spread_ledger_barely_notices() -> None:
    report = trade_removal_report(SPREAD)
    # 100 - 5 = 95; 100 - 15 = 85; 100 - 25 = 75
    assert report.retention[1] == pytest.approx(0.95)
    assert report.retention[3] == pytest.approx(0.85)
    assert report.retention[5] == pytest.approx(0.75)


def test_the_two_ledgers_earn_the_same_and_are_told_apart_anyway() -> None:
    """The point of section 14.4: equal profit, unequal evidence."""
    assert sum(t.pnl for t in CONCENTRATED) == pytest.approx(sum(t.pnl for t in SPREAD))
    concentrated = trade_removal_report(CONCENTRATED)
    spread = trade_removal_report(SPREAD)
    for k in DEFAULT_K_VALUES:
        assert concentrated.retention[k] < spread.retention[k]  # type: ignore[operator]


def test_retention_is_the_metric_module_s_answer_not_a_second_one() -> None:
    """``retention_1/3/5`` is a published metric; there must be one rule for it."""
    report = trade_removal_report(CONCENTRATED)
    assert report.retention == retention_after_removing_top_winners(CONCENTRATED, DEFAULT_K_VALUES)


def test_retention_agrees_with_the_ledger_it_reports_on() -> None:
    """Retention and the recomputed statistics must describe the same survivors."""
    total = sum(t.pnl for t in CONCENTRATED)
    for k in DEFAULT_K_VALUES:
        survivors = remaining_after_removal(CONCENTRATED, k)
        assert sum(t.pnl for t in survivors) / total == pytest.approx(
            trade_removal_report(CONCENTRATED).retention[k]
        )


def test_expectancy_and_profit_factor_are_recomputed_on_the_survivors() -> None:
    report = trade_removal_report(CONCENTRATED)
    for k in DEFAULT_K_VALUES:
        survivors = remaining_after_removal(CONCENTRATED, k)
        assert report.expectancy[k] == trade_expectancy_pct(survivors)
        assert report.profit_factor[k] == trade_profit_factor(survivors)


def test_only_winners_are_removed() -> None:
    """Removing "the k largest" from a ledger with fewer winners would start
    dropping the least-bad losses, which would improve what remains."""
    ledger = [trade(1, 5.0), trade(2, -1.0), trade(3, -8.0)]
    survivors = remaining_after_removal(ledger, 3)
    assert [t.trade_no for t in survivors] == [2, 3]


def test_a_tie_is_broken_deterministically() -> None:
    """Two identical pnls must be removed in a fixed order, or the report is not
    reproducible (INV-7)."""
    ledger = [trade(1, 4.0), trade(2, 4.0), trade(3, 4.0), trade(4, -1.0)]
    assert [t.trade_no for t in remaining_after_removal(ledger, 2)] == [3, 4]
    assert [t.trade_no for t in remaining_after_removal(ledger, 2)] == [3, 4]


def test_retention_is_undefined_for_a_losing_strategy() -> None:
    """Dividing by a negative total would flip the sign and report a plausible
    number for a meaningless quantity."""
    losing = [trade(1, 5.0), trade(2, -20.0)]
    report = trade_removal_report(losing)
    assert report.retention == dict.fromkeys(DEFAULT_K_VALUES)
    assert not report.is_defined
    assert report.retention_at(1) is None


def test_an_empty_ledger_reports_nothing_rather_than_zero() -> None:
    report = trade_removal_report([])
    assert report.n_trades == 0
    assert not report.is_defined
    assert report.expectancy[1] is None
    assert report.profit_factor[1] is None


def test_retention_is_non_increasing_in_k() -> None:
    """Removing more winners can only remove more profit; the property tests in
    ``tests/property/`` state this over random ledgers, and this pins the two
    fixtures the rest of the file reasons about."""
    for ledger in (CONCENTRATED, SPREAD):
        report = trade_removal_report(ledger, (1, 2, 3, 4, 5, 6))
        values = [report.retention[k] for k in (1, 2, 3, 4, 5, 6)]
        assert all(
            later <= earlier
            for earlier, later in itertools.pairwise(values)
            if earlier is not None and later is not None
        )


def test_custom_depths_are_honoured() -> None:
    report = trade_removal_report(SPREAD, (2, 10))
    assert report.k_values == (2, 10)
    assert set(report.retention) == {2, 10}
    assert report.retention[2] == pytest.approx(0.90)  # 100 - 10
    assert report.retention[10] == pytest.approx(0.50)  # 100 - 50


def test_a_depth_beyond_the_winner_count_removes_every_winner() -> None:
    """Retention below zero is a real answer, not an error: with every winner
    gone the ledger is a pure loss, and 110 gross profit against 100 net says so."""
    report = trade_removal_report(CONCENTRATED, (99,))
    assert report.retention[99] == pytest.approx(-0.10)
    assert remaining_after_removal(CONCENTRATED, 99) == [t for t in CONCENTRATED if t.pnl < 0]
