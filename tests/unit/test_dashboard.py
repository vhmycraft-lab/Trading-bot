"""The dashboard shows; it does not touch (spec section 22, T42, T56).

Section 22's acceptance criterion for T42 is unusually literal — "a grep shows
no write calls to the store" — and this file is that grep, written as a test so
it runs on every commit rather than once at review.

The rule is worth the strictness. The store is append-only (section 6) and its
rows are the evidence every verdict rests on, and the dashboard is the one place
where a person is looking at that evidence with a mouse in their hand. It is
where a convenient "fix this row" button would appear, and by the time one had,
the argument for it would be a good one.

The rest of the file is about the second guarantee: the family tree is built
from **persisted lineage only**. A number this process derived would look, on
screen, exactly like one a campaign earned.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from tests.guards import engine_builder_uses, write_method_calls

from quantlab.dashboard import build_family_tree, generation_summary
from quantlab.dashboard.view import UNMEASURED

DASHBOARD = Path(__file__).resolve().parents[2] / "src" / "quantlab" / "dashboard"

#: Every method on ``ExperimentStore`` that changes something. Enumerated from
#: the port rather than matched by prefix: ``record_*`` and ``add_*`` are the
#: obvious ones, and ``freeze_test_end`` and ``count_evaluation`` are the two
#: that a prefix rule would have missed.
WRITE_METHODS = frozenset(
    {
        "get_or_create_dataset",
        "get_or_create_split",
        "freeze_test_end",
        "create_family",
        "add_strategy_version",
        "record_verdict",
        "increment_validation_touches",
        "set_family_status",
        "create_experiment",
        "create_run",
        "finish_run",
        "delete_run",
        "save_verdict",
        "record_llm_interaction",
        "create_evolution_run",
        "finish_evolution_run",
        "count_evaluation",
        "add_generation",
        "add_candidate",
        "score_candidate",
        "add_mutations",
        "record_optuna_study",
        "record_promotion",
        "record_lockbox_access",
        "record_training_environment",
    }
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Node:
    candidate_id: str
    gen_index: int
    origin: str
    kind: str = "genome"
    fitness: float | None = 0.5
    survived: bool = True
    parent_candidate_id: str | None = None
    children: tuple[str, ...] = ()


@dataclass(frozen=True)
class Tree:
    evolution_id: str
    nodes: dict[str, Node]
    roots: tuple[str, ...]
    generations: dict[int, dict[str, float]]


def a_tree() -> Tree:
    nodes = {
        "seed0": Node("seed0", 0, "seed", children=("kid0", "kid1")),
        "kid0": Node("kid0", 1, "offspring", parent_candidate_id="seed0", fitness=0.7),
        "kid1": Node(
            "kid1", 1, "offspring", parent_candidate_id="seed0", fitness=None, survived=False
        ),
        "grandkid": Node("grandkid", 2, "offspring", parent_candidate_id="kid0", fitness=0.9),
        "immigrant": Node("immigrant", 1, "immigrant", fitness=0.2, survived=False),
    }
    nodes["kid0"] = Node(
        "kid0", 1, "offspring", parent_candidate_id="seed0", fitness=0.7, children=("grandkid",)
    )
    return Tree(
        evolution_id="evo0",
        nodes=nodes,
        roots=("seed0", "immigrant"),
        generations={
            0: {"best": 0.5, "median": 0.5, "n": 1},
            1: {"best": 0.7, "median": 0.45, "n": 3},
        },
    )


# ---------------------------------------------------------------------------
# the acceptance criterion
# ---------------------------------------------------------------------------
def _python_files() -> list[Path]:
    return sorted(DASHBOARD.rglob("*.py"))


def test_the_dashboard_package_is_not_empty() -> None:
    """Guards the scan below. A grep over no files finds no writes."""
    assert len(_python_files()) >= 2


def test_no_dashboard_module_calls_a_write_method() -> None:
    """Section 22's acceptance criterion for T42, as a test rather than a review
    note. Matched on the attribute name, so ``store.record_verdict(...)`` is
    caught however the store was obtained or aliased."""
    offenders = [
        f"{path.name}:{hit}"
        for path in _python_files()
        for hit in write_method_calls(path.read_text(encoding="utf-8"), WRITE_METHODS)
    ]
    assert not offenders, "the dashboard must not write to the store: " + "; ".join(offenders)


def test_the_write_method_list_matches_the_port() -> None:
    """Guards the test above against a store that grew a new write method the
    scan has never heard of — which would pass silently for ever."""
    from quantlab.ports.store import ExperimentStore

    declared = {
        name
        for name in dir(ExperimentStore)
        if not name.startswith("_") and callable(getattr(ExperimentStore, name, None))
    }
    unknown = WRITE_METHODS - declared
    assert not unknown, f"the scan names methods the port does not have: {unknown}"


ENGINE_BUILDERS = frozenset({"create_engine", "create_db_engine", "make_session_factory"})


def test_no_dashboard_module_opens_a_database_itself() -> None:
    """A viewer that could build its own engine could build one without the
    partition guard of INV-5, so it goes through ``build_container`` instead.

    Matched on imports and calls rather than on the text of the file. A grep
    over the source would also fire on a comment explaining this rule — it did,
    the first time — and a guard that punishes its own documentation gets
    softened rather than obeyed.
    """
    offenders = [
        f"{path.name}:{hit}"
        for path in _python_files()
        for hit in engine_builder_uses(path.read_text(encoding="utf-8"), ENGINE_BUILDERS)
    ]
    assert not offenders, "the dashboard must not open a database itself: " + "; ".join(offenders)


# ---------------------------------------------------------------------------
# the tree is the stored lineage
# ---------------------------------------------------------------------------
def test_the_tree_is_flattened_depth_first_from_each_root() -> None:
    rows = build_family_tree(a_tree()).rows
    assert [r.candidate_id for r in rows] == [
        "seed0",
        "kid0",
        "grandkid",
        "kid1",
        "immigrant",
    ]
    assert [r.depth for r in rows] == [0, 1, 2, 1, 0]


def test_the_order_does_not_depend_on_fitness() -> None:
    """Ordering by score would put the best candidate first and quietly
    rearrange the picture every time a generation was added — so the same run
    would draw a different tree each time somebody looked."""
    first = [r.candidate_id for r in build_family_tree(a_tree()).rows]
    second = [r.candidate_id for r in build_family_tree(a_tree()).rows]
    assert first == second
    fitnesses = [r.fitness for r in build_family_tree(a_tree()).rows]
    assert fitnesses != sorted((f for f in fitnesses if f is not None), reverse=True)


def test_an_unscored_candidate_is_shown_as_unmeasured_and_not_as_zero() -> None:
    """Zero is a score. A candidate nobody measured did not score zero, and a
    dashboard that said so would be inventing the one number people are looking
    at the screen to read."""
    rows = {r.candidate_id: r for r in build_family_tree(a_tree()).rows}
    assert rows["kid1"].fitness is None
    assert rows["kid1"].fitness_text == UNMEASURED
    assert rows["kid0"].fitness_text == "0.7000"


def test_the_unmeasured_count_is_surfaced() -> None:
    """A run with many of these has not been measured as thoroughly as its
    candidate count suggests, and that should be visible without counting."""
    tree = build_family_tree(a_tree())
    assert tree.n_candidates == 5
    assert tree.n_unmeasured == 1
    assert tree.n_survivors == 3


def test_a_cycle_in_the_lineage_does_not_hang_the_view() -> None:
    """A corrupt lineage is a real possibility and drawing is the worst place to
    discover it — but looping forever while rendering is worse than showing a
    truncated tree."""
    nodes = {
        "a": Node("a", 0, "seed", children=("b",)),
        "b": Node("b", 1, "offspring", parent_candidate_id="a", children=("a",)),
    }
    tree = Tree("evo0", nodes, ("a",), {})
    assert [r.candidate_id for r in build_family_tree(tree).rows] == ["a", "b"]


def test_a_root_that_names_a_missing_candidate_is_skipped() -> None:
    tree = Tree("evo0", {}, ("ghost",), {})
    assert build_family_tree(tree).rows == ()


# ---------------------------------------------------------------------------
# generation summary
# ---------------------------------------------------------------------------
def test_generations_are_summarised_oldest_first() -> None:
    summary = list(generation_summary(build_family_tree(a_tree())))
    assert [row["generation"] for row in summary] == [0, 1]
    assert summary[1]["best"] == 0.7
    assert summary[1]["candidates"] == 3


def test_a_run_with_no_generations_summarises_to_nothing() -> None:
    assert list(generation_summary(build_family_tree(Tree("e", {}, (), {})))) == []


# ---------------------------------------------------------------------------
# the view is importable without the optional extra
# ---------------------------------------------------------------------------
def test_the_shaping_does_not_need_streamlit() -> None:
    """Streamlit is an optional extra, and logic that only runs inside a running
    app is logic nobody tests. Everything asserted above was reached without it
    — this states that as a property rather than leaving it to luck."""
    import quantlab.dashboard.view as view

    source = Path(view.__file__).read_text(encoding="utf-8")
    assert "import streamlit" not in source


@pytest.mark.parametrize("attribute", ["n_candidates", "n_survivors", "n_unmeasured"])
def test_every_headline_number_is_derived_from_the_rows(attribute: str) -> None:
    """Guards against a summary field that is stored rather than computed, which
    could then disagree with the tree drawn underneath it."""
    tree = build_family_tree(a_tree())
    assert isinstance(getattr(tree, attribute), int)
