"""Mark every recorded verdict with the ``M`` formula that produced it (§14.4).

Check 1 of §14.4 deflates a Sharpe ratio against ``M``, the number of trials the
search made: ``SR0``, the ratio the best of ``M`` trials reaches by luck alone, is
the benchmark the result has to beat. Until this revision the value actually
passed to the deflation was ``max(1, n_bars // 100)`` — a function of the
validation segment's *length*, not of the search — so a forty-thousand-evaluation
evolutionary campaign was deflated exactly as gently as a single backtest.
``M`` is now assembled from the store, as §14.4 always specified:
``evolution_run.n_evaluations + optuna trials + family.validation_touches``.

Every verdict written before that correction is therefore **overstated**: its
deflated Sharpe was measured against too low a bar, and check 1 is worth 25 of
the 100 overfit points. Nothing about the stored number reveals this — an
overstated verdict looks exactly like a sound one — so the formula is recorded
next to it.

**Nothing is deleted or rewritten.** §6 forbids rewriting a recorded verdict, and
what the platform once believed is part of the audit trail rather than a mistake
to be tidied away. The column's default does the marking: rows that already exist
acquire ``m1-bar-count`` without any of their recorded values being touched, and
code written after this revision stamps ``m2-search-size``. A verdict that is not
stamped with the current formula no longer authorises anything —
``require_candidate`` refuses to open the lockbox on one — so a superseded
verdict must be recomputed by re-running ``quantlab validate`` before it is
trusted again.

``ALTER TABLE ... ADD COLUMN`` with a non-null default is the one schema change
SQLite performs in place, so this revision does not rebuild the table and the
existing rows keep their identity.

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-07
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: Kept as a literal rather than imported from ``quantlab``: a migration records
#: what was true at its revision, and must not change meaning when the constant
#: it mirrors is next revised.
SUPERSEDED = "m1-bar-count"

UPGRADE_STATEMENTS: tuple[str, ...] = (
    "ALTER TABLE validation_verdict "
    f"ADD COLUMN m_formula_version TEXT NOT NULL DEFAULT '{SUPERSEDED}'",
)

DOWNGRADE_STATEMENTS: tuple[str, ...] = (
    "ALTER TABLE validation_verdict DROP COLUMN m_formula_version",
)


def upgrade() -> None:
    for statement in UPGRADE_STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    for statement in DOWNGRADE_STATEMENTS:
        op.execute(statement)
