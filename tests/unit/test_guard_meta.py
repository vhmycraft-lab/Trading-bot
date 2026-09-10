"""Tests for the guards themselves — both directions, for each one. 🔒

Three structural guards protect properties no runtime test can reach: the
dashboard writes nothing to the store, it opens no database of its own, and the
paper CLI offers no way to go live. Each began as a text search, fired on its
own documentation, and was rewritten to inspect structure instead.

That rewrite is where a guard can quietly die. Narrowing a scanner until it
stops tripping on prose is one edit away from narrowing it until it trips on
nothing, and the suite looks identical either way — the guard still runs, still
passes, still reads like protection. The only difference is that it has stopped
being evidence, and nothing announces that.

So every guard is pinned from both sides here:

* **positive** — a real violation, and the scanner must find it;
* **negative** — the same words in a docstring, a comment, or a string, and the
  scanner must ignore them.

The negative cases are written from the exact text that broke each guard the
first time, so a regression reproduces the original failure rather than an
imagined one.
"""

from __future__ import annotations

import pytest
import typer
from tests.guards import declared_option_names, engine_builder_uses, write_method_calls

from quantlab.ports.store import ExperimentStore

WRITES = frozenset({"record_verdict", "create_run", "add_candidate", "freeze_test_end"})
ENGINES = frozenset({"create_engine", "create_db_engine", "make_session_factory"})


# ---------------------------------------------------------------------------
# guard 1: the dashboard writes nothing to the store
# ---------------------------------------------------------------------------
def test_the_write_scanner_catches_a_real_write() -> None:
    source = "def save(store, v):\n    store.record_verdict(verdict_id=v)\n"
    assert write_method_calls(source, WRITES) == ["2:record_verdict"]


def test_the_write_scanner_catches_a_write_through_an_alias() -> None:
    """The receiver's name is not the property. A module that renamed the store
    to `db` and called `db.create_run(...)` has written to it just the same."""
    source = "def go(db):\n    db.create_run(run_id='r')\n"
    assert write_method_calls(source, WRITES) == ["2:create_run"]


@pytest.mark.parametrize(
    "source",
    [
        pytest.param('"""This module must never call store.record_verdict."""\n', id="docstring"),
        pytest.param("# never call store.create_run() from here\nx = 1\n", id="comment"),
        pytest.param('MESSAGE = "do not use add_candidate() in a viewer"\n', id="string-literal"),
        pytest.param('"""See `freeze_test_end` in ports/store.py."""\n', id="cross-reference"),
    ],
)
def test_the_write_scanner_ignores_prose(source: str) -> None:
    """The failure that made this file necessary.

    The first version of the dashboard guard was a substring search, and a
    comment naming a write method failed it. A guard that punishes its own
    documentation gets softened rather than obeyed — and softening is how it
    stops catching anything.
    """
    assert write_method_calls(source, WRITES) == []


def test_a_local_function_of_the_same_name_is_not_a_store_write() -> None:
    """Matched as an attribute, not as a bare name. A module with its own helper
    called `create_run` has not touched the store, and flagging it would teach a
    reader that the guard cries wolf."""
    source = "def create_run():\n    return 1\n\n\ndef go():\n    return create_run()\n"
    assert write_method_calls(source, WRITES) == []


def test_the_write_scanner_finds_every_call_not_just_the_first() -> None:
    """A caller that fixed the reported line and re-ran should not discover the
    second write on the next commit."""
    source = "def go(s):\n    s.create_run(1)\n    s.record_verdict(2)\n"
    assert len(write_method_calls(source, WRITES)) == 2


def test_the_guarded_method_names_are_real_port_methods() -> None:
    """Guards the guard's vocabulary. A name misspelled in the write list would
    protect nothing, and the test above would still pass because it feeds the
    scanner the same misspelling."""
    declared = {name for name in dir(ExperimentStore) if not name.startswith("_")}
    assert declared >= WRITES, WRITES - declared


# ---------------------------------------------------------------------------
# guard 2: the dashboard opens no database of its own
# ---------------------------------------------------------------------------
def test_the_engine_scanner_catches_a_real_call() -> None:
    source = "def go(p):\n    return create_db_engine(p)\n"
    assert engine_builder_uses(source, ENGINES) == ["2:create_db_engine()"]


def test_the_engine_scanner_catches_a_bare_import() -> None:
    """An import is one line away from a call and looks clean in a diff."""
    source = "from quantlab.adapters.store.sqlite import create_db_engine\n"
    assert engine_builder_uses(source, ENGINES) == ["1:import create_db_engine"]


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            "# Through the container rather than create_db_engine: a viewer that\n"
            "# built its own engine could build one without INV-5's guard.\n"
            "x = 1\n",
            id="the-comment-that-broke-it",
        ),
        pytest.param('"""Never call create_engine here."""\n', id="docstring"),
        pytest.param('NOTE = "make_session_factory belongs to container.py"\n', id="string"),
    ],
)
def test_the_engine_scanner_ignores_prose(source: str) -> None:
    """The first parameter is the verbatim comment that failed the original
    text-based guard, so this reproduces the real regression."""
    assert engine_builder_uses(source, ENGINES) == []


def test_the_engine_scanner_sees_through_a_module_attribute_call() -> None:
    """`sqlalchemy.create_engine(...)` is the same act as `create_engine(...)`."""
    source = "import sqlalchemy\n\n\ndef go():\n    return sqlalchemy.create_engine('x')\n"
    assert engine_builder_uses(source, ENGINES) == ["5:create_engine()"]


# ---------------------------------------------------------------------------
# guard 3: the paper CLI offers no way to go live
# ---------------------------------------------------------------------------
def _app_with_live_flag() -> typer.Typer:
    app = typer.Typer()

    @app.command("start")
    def start(live: bool = typer.Option(False, "--live")) -> None:  # pragma: no cover
        """Start a session."""

    return app


def _app_that_only_mentions_live() -> typer.Typer:
    app = typer.Typer()

    @app.command("start")
    def start() -> None:  # pragma: no cover
        """Start a session. There is no --live flag and never will be."""

    return app


def test_the_option_scanner_catches_a_real_live_flag() -> None:
    assert "--live" in declared_option_names(_app_with_live_flag())


def test_the_option_scanner_ignores_a_docstring_that_mentions_it() -> None:
    """The failure this guard was rewritten for: the paper CLI's own docstring
    says there is no `--live` flag, and the text search flagged the sentence
    promising the flag does not exist."""
    assert "--live" not in declared_option_names(_app_that_only_mentions_live())


def test_the_option_scanner_reads_more_than_the_top_level_help() -> None:
    """A flag hidden on a subcommand is still a flag. Scanning only the root
    `--help` would miss every one of them."""
    options = declared_option_names(_app_with_live_flag())
    assert "--live" in options, "a flag on a subcommand was missed"


def test_the_real_paper_cli_offers_no_live_option() -> None:
    """The guard itself, now that both of its directions are pinned."""
    from quantlab.cli.paper import app

    options = declared_option_names(app)
    assert options, "no options were rendered; the scan proves nothing"
    for forbidden in ("--live", "--real", "--broker", "--api-key", "--secret"):
        assert forbidden not in options, forbidden
