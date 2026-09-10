"""The handbook's commands are real (spec section 22, T44).

A handbook is the one document that gets read by somebody who cannot yet tell
whether it is wrong. Every other file here is checked by a person who already
knows the system; this one is checked by a person who does not, and who will
conclude the *platform* is broken when a documented command turns out not to
exist.

So the commands are extracted from the Markdown and run against the real CLI.
Not a spot check — every shell block in the file, with the ones that would
touch a network or a database reduced to ``--help`` so this stays a unit test.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quantlab.cli import app

HANDBOOK = Path(__file__).resolve().parents[2] / "docs" / "HANDBOOK.md"
MAKEFILE = Path(__file__).resolve().parents[2] / "Makefile"

_BLOCK = re.compile(r"```bash\n(.*?)```", re.DOTALL)

#: Placeholders the handbook uses for values a reader supplies. Replaced with a
#: harmless token before the command is parsed, so ``<run_id>`` does not become
#: a shell redirect and an argument the CLI would reject for the wrong reason.
_PLACEHOLDER = re.compile(r"<[a-z_ ]+>")


def commands() -> list[str]:
    return [
        line.strip()
        for block in _BLOCK.findall(HANDBOOK.read_text(encoding="utf-8"))
        for line in block.splitlines()
        if line.strip()
    ]


def test_the_handbook_exists_and_has_commands() -> None:
    """Guards every test below: an empty parse passes them all."""
    assert HANDBOOK.is_file()
    assert len(commands()) >= 15


@pytest.mark.parametrize("command", commands())
def test_every_documented_command_is_one_the_platform_offers(command: str) -> None:
    """The claim this file exists to check."""
    cleaned = _PLACEHOLDER.sub("PLACEHOLDER", command)
    parts = cleaned.split()

    if parts[0] == "make":
        # `make sync`, `make check`, `make db-upgrade`. Checked against the
        # Makefile rather than executed: running the full gate from inside the
        # gate would not terminate.
        target = parts[1]
        assert re.search(rf"(?m)^{re.escape(target)}:", MAKEFILE.read_text(encoding="utf-8")), (
            f"the handbook documents `make {target}`, which the Makefile has no target for"
        )
        return

    if parts[:2] == ["uv", "run"]:
        parts = parts[2:]

    if parts[0] == "pre-commit":
        # A dev-extra tool. Checked as a declared dependency rather than run:
        # `pre-commit install` writes into `.git/hooks`, which a test must not.
        pyproject = (HANDBOOK.parents[1] / "pyproject.toml").read_text(encoding="utf-8")
        assert "pre-commit" in pyproject, (
            "the handbook documents pre-commit, which is not a dependency"
        )
        return

    if parts[0] == "quantlab":
        # Reduced to `--help`: the point is that the subcommand path resolves,
        # and actually running `data pull` in a unit test would reach a network.
        subcommands = [p for p in parts[1:] if not p.startswith("-") and p != "PLACEHOLDER"]
        result = CliRunner().invoke(app, [*subcommands, "--help"])
        assert result.exit_code == 0, (
            f"`{command}` does not resolve: {result.output}\n{result.exception}"
        )
        return

    if parts[0] == "quantlab-lockbox":
        # A separate console script (section 14.6). Invoked as a module so this
        # does not depend on the virtualenv's bin/ being on PATH.
        completed = subprocess.run(
            [sys.executable, "-m", "quantlab.cli.lockbox", "--help"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert completed.returncode == 0, completed.stderr
        return

    if parts[0] in {"uv", "streamlit"}:
        # `uv sync --extra dashboard`, `uv run streamlit run ...` — the extra
        # and the app path are checked, not executed.
        text = " ".join(parts)
        if "--extra" in parts:
            extra = parts[parts.index("--extra") + 1]
            pyproject = (HANDBOOK.parents[1] / "pyproject.toml").read_text(encoding="utf-8")
            assert f"{extra} = [" in pyproject, f"there is no `{extra}` extra"
            return
        if "streamlit" in text:
            script = parts[-1]
            assert (HANDBOOK.parents[1] / script).is_file(), f"no such file: {script}"
            return

    pytest.fail(f"unrecognised command in the handbook: {command!r}")


# ---------------------------------------------------------------------------
# claims that would age badly
# ---------------------------------------------------------------------------
def test_the_handbook_names_the_lockbox_binary_correctly() -> None:
    """``quantlab-lockbox``, not ``quantlab lockbox``. Section 14.6 makes them
    different binaries on purpose, and a reader who types the wrong one gets an
    unhelpful error at exactly the moment they are least sure of themselves."""
    blocks = commands()
    assert any(c.startswith("uv run quantlab-lockbox evaluate") for c in blocks), blocks
    # Asserted over the runnable blocks rather than the prose: the paragraph
    # above that command says "not `quantlab lockbox`", and a check that fired
    # on its own warning would be removed rather than heeded.
    assert not any("quantlab lockbox" in c for c in blocks)


def test_the_handbook_states_the_lockbox_budget_the_code_enforces() -> None:
    """A handbook that misstated this would have somebody spend a family's only
    look at the test partition on a misunderstanding."""
    from quantlab.core.config import LockboxSettings

    settings = LockboxSettings()
    text = HANDBOOK.read_text(encoding="utf-8")
    assert f"max_per_family = {settings.max_per_family}" in text
    assert "three looks a month" in text
    assert settings.max_per_month == 3


def test_the_handbook_names_the_superseded_formula_marker() -> None:
    """The one piece of operational advice in the file that a reader has to be
    able to match against a database row."""
    from quantlab.core.validation.deflated_sharpe import SUPERSEDED_M_FORMULA

    assert SUPERSEDED_M_FORMULA in HANDBOOK.read_text(encoding="utf-8")


def test_the_handbook_does_not_promise_a_live_mode() -> None:
    """INV-1, in the document most likely to be read by somebody hoping for
    one."""
    text = HANDBOOK.read_text(encoding="utf-8").lower()
    assert "never places" in text or "without ever placing" in text
    for phrase in ("live trading", "real money mode", "--live"):
        assert phrase not in text, phrase
