"""The read-only dashboard (master spec section 22, T42).

Read-only is a hard property here, not an intention. ``tests/unit/test_dashboard.py``
greps this package for every mutating call on the store and asserts there are
none — the acceptance criterion section 22 states in exactly those terms.

Why so strict about a viewer: the store is append-only by design (section 6),
and its rows are the evidence every verdict rests on. A dashboard is the one
place where a person is looking at that evidence and has a mouse in their hand,
which makes it the likeliest place for a convenient "fix this row" button to
appear. There is no way to add one without failing a test.
"""

from __future__ import annotations

from quantlab.dashboard.view import (
    FamilyTree,
    build_family_tree,
    generation_summary,
)

__all__ = ["FamilyTree", "build_family_tree", "generation_summary"]
