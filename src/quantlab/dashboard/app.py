"""The Streamlit view (master spec section 22, T42).

Layout only. Everything that decides *what* to show lives in
:mod:`quantlab.dashboard.view`, as pure functions over stored rows, because a
function that only runs inside a running app is a function nobody tests — and
this app's correctness condition ("read-only", "persisted lineage only") is
exactly the kind that goes wrong quietly.

Run it with::

    uv sync --extra dashboard
    uv run streamlit run src/quantlab/dashboard/app.py

Non-photographic by construction: this draws tables and a text tree from the
database. There is no image pipeline here, nothing is generated, and the only
thing on screen is what a campaign recorded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from quantlab.dashboard.view import build_family_tree, generation_summary

__all__ = ["main", "render"]


def render(st: Any, store: Any, evolution_id: str) -> None:
    """Draw one run. ``st`` is injected so this is callable without Streamlit."""
    from quantlab.evolution.lineage import lineage_tree

    tree = build_family_tree(lineage_tree(store, evolution_id))

    st.subheader(f"Run {tree.evolution_id}")
    left, middle, right = st.columns(3)
    left.metric("candidates", tree.n_candidates)
    middle.metric("survivors", tree.n_survivors)
    # Shown next to the others rather than buried: a run with many unmeasured
    # candidates has not been measured as thoroughly as its candidate count
    # suggests, and the count is the first thing a reader takes from the page.
    right.metric("unmeasured", tree.n_unmeasured)

    st.caption(
        "Every number on this page is read from the database. Nothing here is "
        "recomputed, and a candidate the store holds no fitness for is shown as "
        "unmeasured rather than as zero."
    )

    st.markdown("#### Generations")
    st.dataframe(list(generation_summary(tree)), use_container_width=True)

    st.markdown("#### Family tree")
    st.code(
        "\n".join(f"{row.label}  {row.fitness_text}" for row in tree.rows) or "(no candidates)",
        language="text",
    )


def main() -> None:  # pragma: no cover - exercised by running the app
    import streamlit as st

    from quantlab.container import build_container
    from quantlab.core.config import load_config

    st.set_page_config(page_title="QuantLab", layout="wide")
    st.title("QuantLab — read-only")

    config_path = Path(st.sidebar.text_input("config", value="configs/default.yaml"))
    if not config_path.is_file():
        st.warning(f"no configuration at {config_path}")
        return

    # Through the container, on the research profile, rather than opening an
    # engine here. A viewer that built its own engine could build one without
    # the partition guard of INV-5 — and `tests/unit/test_dashboard.py` refuses
    # any module in this package that reaches for `create_db_engine`, which is
    # how that stayed true rather than becoming a comment about the past.
    container = build_container(load_config([config_path], []), profile="research")
    store = container.store
    runs = list(store.query_evolution_runs()) if hasattr(store, "query_evolution_runs") else []
    ids = [str(getattr(run, "evolution_id", "")) for run in runs]
    if not ids:
        st.info("this database has no evolution runs yet")
        return

    render(st, store, st.sidebar.selectbox("run", ids))


if __name__ == "__main__":  # pragma: no cover
    main()
