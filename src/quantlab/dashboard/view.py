"""What the dashboard shows, computed apart from how it is drawn (spec T42, T56).

Streamlit is an optional extra and a poor place to put logic: a function that
only runs inside a running app is a function nobody tests. So the shaping lives
here, as pure functions over stored rows, and ``app.py`` is left with layout.

**The family tree is built from persisted lineage only.** Not from a re-run, not
from a recomputed fitness, not from anything this process derives. If a
candidate's fitness is not in the store, the tree shows it as unmeasured rather
than computing one — a number a dashboard invented would be indistinguishable on
screen from one a campaign earned, and it is the campaign's numbers people are
looking at the screen to see.

Nothing here writes. See :mod:`quantlab.dashboard` for why that is enforced
rather than intended.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "UNMEASURED",
    "FamilyTree",
    "TreeRow",
    "build_family_tree",
    "generation_summary",
]

#: What is shown where the store holds no fitness. A word rather than a dash or
#: a zero: zero is a score, and a candidate nobody scored did not score zero.
UNMEASURED: str = "unmeasured"


@dataclass(frozen=True)
class TreeRow:
    """One candidate, as a row in the rendered tree."""

    candidate_id: str
    gen_index: int
    depth: int
    origin: str
    survived: bool
    fitness: float | None

    @property
    def fitness_text(self) -> str:
        return UNMEASURED if self.fitness is None else f"{self.fitness:.4f}"

    @property
    def label(self) -> str:
        marker = "*" if self.survived else " "
        return f"{'  ' * self.depth}{marker} {self.candidate_id[:12]} ({self.origin})"


@dataclass(frozen=True)
class FamilyTree:
    """A whole run's lineage, flattened for display."""

    evolution_id: str
    rows: tuple[TreeRow, ...]
    generations: Mapping[int, Mapping[str, float]]

    @property
    def n_candidates(self) -> int:
        return len(self.rows)

    @property
    def n_survivors(self) -> int:
        return sum(1 for row in self.rows if row.survived)

    @property
    def n_unmeasured(self) -> int:
        """Candidates the store holds no fitness for.

        Surfaced rather than hidden: a run with many of these has not been
        measured as thoroughly as its candidate count suggests, and that is
        something a person reading the tree should see without counting rows.
        """
        return sum(1 for row in self.rows if row.fitness is None)


def build_family_tree(tree: Any) -> FamilyTree:
    """Flatten a :class:`~quantlab.evolution.lineage.LineageTree` for display.

    Depth-first from each root, children in the order the lineage records them,
    so the same run draws the same tree every time. A traversal that ordered by
    fitness would put the best candidate first and quietly rearrange the picture
    every time a generation was added.
    """
    rows: list[TreeRow] = []
    nodes = tree.nodes
    seen: set[str] = set()

    def walk(candidate_id: str, depth: int) -> None:
        if candidate_id in seen:
            # A cycle is a corrupt lineage, and `path_to_root` raises on one.
            # Drawing is not the place to discover it, but it is absolutely not
            # the place to loop forever either.
            return
        seen.add(candidate_id)
        node = nodes.get(candidate_id)
        if node is None:
            return
        rows.append(
            TreeRow(
                candidate_id=node.candidate_id,
                gen_index=int(node.gen_index),
                depth=depth,
                origin=str(node.origin),
                survived=bool(node.survived),
                fitness=node.fitness,
            )
        )
        for child in node.children:
            walk(child, depth + 1)

    for root in tree.roots:
        walk(root, 0)

    return FamilyTree(
        evolution_id=str(tree.evolution_id),
        rows=tuple(rows),
        generations=dict(tree.generations),
    )


def generation_summary(tree: FamilyTree) -> Sequence[Mapping[str, Any]]:
    """Per-generation figures, oldest first, as the stored run recorded them."""
    return [
        {
            "generation": index,
            "best": stats.get("best"),
            "median": stats.get("median"),
            "candidates": stats.get("n"),
        }
        for index, stats in sorted(tree.generations.items())
    ]
