"""The trade-removal test (master spec section 14.4).

A strategy whose profit is carried by a handful of exceptional trades has not
demonstrated an edge; it has demonstrated that a few things happened. This module
removes the top ``k`` winning trades by ``pnl`` and recomputes what is left.

Everything is computed on the **trade pnl series**, exactly and without
re-simulation. That is what makes it cheap enough to run for every candidate in
every generation of section 13, and it is also what makes it well defined: there
is no question of which counterfactual bars the strategy would have traded, only
the arithmetic of removing rows from a ledger.

The results are used twice. During evolution they are the ``p_removal`` penalty
and the ``F_CONCENTRATION`` hard gate of section 13.3; during validation they are
soft check 10 of section 14.4. A strategy that becomes unprofitable after losing
its single best trade is rejected outright, not merely penalised.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from quantlab.core.metrics import (
    retention_after_removing_top_winners,
    trade_expectancy_pct,
    trade_profit_factor,
)
from quantlab.core.types import Trade

__all__ = [
    "DEFAULT_K_VALUES",
    "TradeRemovalReport",
    "remaining_after_removal",
    "trade_removal_report",
]

#: The removal depths section 14.4 names.
DEFAULT_K_VALUES: Final[tuple[int, ...]] = (1, 3, 5)


@dataclass(frozen=True, slots=True)
class TradeRemovalReport:
    """What survives the removal of the best trades (spec section 14.4)."""

    k_values: tuple[int, ...]
    #: ``Σ pnl(remaining) / Σ pnl(all)``; ``None`` when total pnl is not positive.
    retention: dict[int, float | None]
    expectancy: dict[int, float | None]
    profit_factor: dict[int, float | None]
    n_trades: int

    def retention_at(self, k: int) -> float | None:
        """Retention at depth ``k``, or ``None`` if it was not computed."""
        return self.retention.get(int(k))

    @property
    def is_defined(self) -> bool:
        """True when retention could be computed at all.

        False means total pnl was not positive, which cannot happen for a
        candidate that passed the ``F_EXPECTANCY`` gate — so a caller seeing this
        is looking at a losing strategy, not at a missing computation.
        """
        return any(value is not None for value in self.retention.values())


def remaining_after_removal(trades: Sequence[Trade], k: int) -> list[Trade]:
    """The ledger with the ``k`` largest *winning* trades removed.

    Only winners are removed. Dropping "the k largest" from a ledger with fewer
    than ``k`` winners would start removing the least-bad losses, which would
    *improve* the remaining statistics — the opposite of what this test is for.

    Ties are broken by ``trade_no`` so that two trades with identical pnl are
    removed in a fixed order and the report is reproducible (INV-7).
    """
    depth = max(0, int(k))
    winners = sorted(
        (trade for trade in trades if trade.pnl > 0),
        key=lambda trade: (-trade.pnl, trade.trade_no),
    )
    removed = {id(trade) for trade in winners[:depth]}
    return [trade for trade in trades if id(trade) not in removed]


def trade_removal_report(
    trades: Sequence[Trade], k_values: Sequence[int] = DEFAULT_K_VALUES
) -> TradeRemovalReport:
    """Recompute retention, expectancy and profit factor at each removal depth.

    Retention comes from :func:`~quantlab.core.metrics.retention_after_removing_top_winners`
    rather than from the remaining ledger built here: the ratio is a published
    metric (``retention_1/3/5`` on :class:`~quantlab.core.metrics.MetricSet`), and
    a second implementation of it would be a second answer to the same question.
    Expectancy and profit factor use the same functions ``compute_metrics`` does,
    applied to the ledger that survives.
    """
    depths = tuple(int(k) for k in k_values)
    retention = retention_after_removing_top_winners(trades, depths)
    expectancy: dict[int, float | None] = {}
    profit_factor: dict[int, float | None] = {}
    for depth in depths:
        survivors = remaining_after_removal(trades, depth)
        expectancy[depth] = trade_expectancy_pct(survivors)
        profit_factor[depth] = trade_profit_factor(survivors)
    return TradeRemovalReport(
        k_values=depths,
        retention={depth: retention.get(depth) for depth in depths},
        expectancy=expectancy,
        profit_factor=profit_factor,
        n_trades=len(trades),
    )
