"""``quantlab paper`` (spec section 16.3), and the startup guard (16.2).

Most of this file is about the guard, because it is the one piece of the paper
CLI whose failure mode is not "an unhelpful error message".

The guard is often misread as "check we are in paper mode". It is not, and
could not be: nothing in this codebase can leave paper mode — there is no broker
but ``PaperBroker`` and no method on the port that transmits anything (INV-1).
What the guard actually asks is narrower and more useful: *if* an exchange key
happens to be configured on this machine, and *if* that key can trade spot, then
a mistake anywhere in the stack has somewhere to land. Refusing to start removes
the landing site. It is defence in depth against a bug this platform is
structured not to have.
"""

from __future__ import annotations

from typing import Any

import pytest

from quantlab.cli.paper import RUNNABLE_VERDICTS, notify, startup_guard
from quantlab.core.errors import ConfigError


class FakeExchange:
    def __init__(self, restrictions: dict[str, Any] | None = None, raises: bool = False) -> None:
        self._restrictions = restrictions or {}
        self._raises = raises
        self.calls = 0

    def sapi_get_account_apirestrictions(self) -> dict[str, Any]:
        self.calls += 1
        if self._raises:
            raise RuntimeError("network is down")
        return self._restrictions


# ---------------------------------------------------------------------------
# the startup guard
# ---------------------------------------------------------------------------
def test_no_key_is_not_an_error() -> None:
    """The public stream needs no key, and demanding one would push people
    towards creating a key they do not need — which is the hazard this guard
    exists to reduce, arrived at from the other direction."""
    startup_guard(None)


def test_a_trading_enabled_key_aborts_the_session() -> None:
    """Section 16.2's own words: abort if ``enableSpotAndMarginTrading`` is true."""
    with pytest.raises(ConfigError, match="spot and margin trading"):
        startup_guard(FakeExchange({"enableSpotAndMarginTrading": True}))


def test_a_read_only_key_is_allowed() -> None:
    """Guards the test above: a guard that refused every key would be removed
    the first time somebody wanted higher rate limits, which is the legitimate
    reason to have one at all."""
    startup_guard(FakeExchange({"enableSpotAndMarginTrading": False, "enableReading": True}))


def test_a_key_whose_permissions_cannot_be_read_aborts() -> None:
    """Fail closed. A key of unknown scope is not a key known to be safe, and
    "the permissions endpoint was down" is not evidence that trading is off."""
    with pytest.raises(ConfigError, match="unknown scope"):
        startup_guard(FakeExchange(raises=True))


def test_a_restrictions_response_missing_the_field_is_treated_as_safe() -> None:
    """An endpoint that answered and did not mention spot trading has said it is
    not enabled. This is the one place the guard reads an absence as a negative,
    and it is bounded: the call succeeded, so the account was reachable."""
    startup_guard(FakeExchange({"enableReading": True}))


def test_the_guard_actually_queries_the_key() -> None:
    """Guards every test above against a guard that returns early — which would
    pass all the permissive cases and none of the restrictive ones would ever
    run."""
    exchange = FakeExchange({"enableSpotAndMarginTrading": False})
    startup_guard(exchange)
    assert exchange.calls == 1


# ---------------------------------------------------------------------------
# notifications are best-effort
# ---------------------------------------------------------------------------
def test_a_notification_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A paper session that killed itself because a notification could not be
    delivered would be less useful than one that kept trading and logged."""
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/osascript")

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise OSError("no window server")

    monkeypatch.setattr("subprocess.run", explode)
    notify("drawdown exceeded")


def test_a_machine_without_osascript_is_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: None)
    notify("anything")


def test_a_notification_cannot_inject_a_second_applescript_statement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The message can contain a drawdown figure and a strategy name, and a
    strategy name is not something this platform chose. Quotes are neutralised
    and the payload is passed as an argument rather than through a shell, so
    there is no second statement to reach."""
    seen: list[list[str]] = []
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/osascript")
    monkeypatch.setattr(
        "subprocess.run", lambda cmd, **kwargs: seen.append(list(cmd)) or type("R", (), {})()
    )
    notify('bad" with title "x" \n do shell script "echo pwned')
    assert len(seen) == 1
    command = seen[0]
    assert command[0].endswith("osascript")
    assert '"' not in command[2].split("display notification ", 1)[1].split(" with title")[0][1:-1]


def test_a_very_long_message_is_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/osascript")
    monkeypatch.setattr(
        "subprocess.run", lambda cmd, **kwargs: seen.append(list(cmd)) or type("R", (), {})()
    )
    notify("x" * 10_000)
    assert len(seen[0][2]) < 400


# ---------------------------------------------------------------------------
# what may be run forward
# ---------------------------------------------------------------------------
def test_a_rejected_strategy_is_not_runnable() -> None:
    """A ``REJECT`` is a decision, and running one forward is a way of
    relitigating it — which is the loop the validation budget exists to stop,
    moved to a place with no budget."""
    assert "REJECT" not in RUNNABLE_VERDICTS
    assert "LOCKBOX_FAIL" not in RUNNABLE_VERDICTS


def test_a_weak_strategy_is_runnable() -> None:
    """Deliberately included. Watching a weak strategy fail in public costs
    nothing and is often more informative than the verdict was."""
    assert "WEAK" in RUNNABLE_VERDICTS
    assert "CANDIDATE" in RUNNABLE_VERDICTS


FORBIDDEN_OPTIONS = frozenset({"--live", "--real", "--broker", "--api-key", "--secret"})


def test_the_paper_command_offers_no_way_to_go_live() -> None:
    """INV-1 at the surface a person actually types at.

    Asserted against Click's declared parameters — not the module text, and not
    the rendered help either. Grepping the source fired on the docstring that
    explains this rule; scanning the rendered help fired on it too, because
    Typer renders a command's docstring *into* its help output. That second
    failure was found by ``tests/unit/test_guard_meta.py``, which feeds this
    scanner a command whose only mention of ``--live`` is in its docstring.
    Only the parameter list knows what the command will actually accept.
    """
    from tests.guards import declared_option_names

    from quantlab.cli.paper import app

    options = declared_option_names(app)
    assert options, "no options were found at all; this scan would prove nothing"
    for option in FORBIDDEN_OPTIONS:
        assert option not in options, f"{option} is offered by the paper CLI"
