"""The LLM port and its two adapters (spec sections 12.1-12.3).

``MockLLMProvider`` is what makes the whole search testable offline, so its
failure modes matter more than its success one: a mock that quietly wrapped
around, or that invented a price, would keep every downstream test green while
the thing under test stopped being exercised.

``GLMProvider`` is tested against a fake transport rather than a live endpoint.
No ``GLM_API_KEY`` exists in this repository and none is needed — what has to be
asserted is the adapter's *policy* (what it retries, what it refuses, what it
charges, what it never prints), and all of that is observable from the requests
it makes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, ValidationError

from quantlab.adapters.llm.glm import RETRYABLE_STATUS, DailyLedger, GLMProvider, load_pricing
from quantlab.adapters.llm.mock import RECORD_ENV_VAR, MockLLMProvider
from quantlab.core.errors import (
    BudgetExceeded,
    ConfigError,
    LLMMalformedResponse,
    LLMTransportError,
)
from quantlab.ports.llm import LLMProvider, LLMResponse, Message, is_message_sequence

PRICING = Path("configs/llm_pricing.yaml")
HELLO = [Message(role="user", content="propose a genome")]


class Tiny(BaseModel):
    value: int


def _reply(text: str, *, tokens_in: int = 10, tokens_out: int = 5) -> dict[str, Any]:
    return {
        "id": "chat-1",
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out},
    }


def _client(handler: Any) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://fake.invalid/v4",
        headers={"Authorization": "Bearer test-key-not-real"},
    )


def _provider(handler: Any, **kwargs: Any) -> GLMProvider:
    return GLMProvider(
        api_key="test-key-not-real",
        pricing_file=PRICING,
        client=_client(handler),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# the port
# ---------------------------------------------------------------------------
def test_both_adapters_satisfy_the_port() -> None:
    assert isinstance(MockLLMProvider("c", responses=["{}"]), LLMProvider)
    assert isinstance(_provider(lambda r: httpx.Response(200, json=_reply("{}"))), LLMProvider)


def test_a_bare_string_is_not_a_conversation() -> None:
    """Every provider API would accept a string as a single user turn — silently
    dropping the system message that carries the genome schema and the INV-11
    constraints, which is the half that does the work."""
    assert not is_message_sequence("propose a genome")
    assert not is_message_sequence([])
    assert is_message_sequence(HELLO)


def test_a_message_with_an_invented_role_is_refused() -> None:
    """A typo in a role is the kind of thing a provider accepts and then ignores."""
    with pytest.raises(ValidationError):
        Message(role="systen", content="x")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# the mock
# ---------------------------------------------------------------------------
def test_the_mock_replays_in_order() -> None:
    provider = MockLLMProvider("c", responses=['{"value": 1}', '{"value": 2}'])
    assert provider.complete(HELLO, response_model=Tiny).parsed == Tiny(value=1)
    assert provider.complete(HELLO, response_model=Tiny).parsed == Tiny(value=2)


def test_the_mock_refuses_to_wrap_around() -> None:
    """Cycling would answer the seventh question with the first answer, and the
    test would keep passing while measuring nothing."""
    provider = MockLLMProvider("c", responses=['{"value": 1}'])
    provider.complete(HELLO)
    with pytest.raises(LLMTransportError, match="no response"):
        provider.complete(HELLO)


def test_a_fixture_that_no_longer_validates_is_reported() -> None:
    """A schema change that outdates the fixtures should look like a schema
    change, not like a model that started replying badly."""
    provider = MockLLMProvider("c", responses=['{"wrong": 1}'])
    with pytest.raises(LLMMalformedResponse, match="does not validate"):
        provider.complete(HELLO, response_model=Tiny)


def test_the_mock_costs_nothing() -> None:
    """A replayed response cost nothing to obtain. A fabricated price would put
    invented euros into the daily ledger and into every report that reads it."""
    assert MockLLMProvider("c", responses=["{}"]).cost_eur(10_000, 10_000) == 0.0


def test_record_mode_is_not_something_a_test_run_can_switch_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It costs money and reaches a network, so it is human-triggered only —
    and a mock has nothing to record from in any case."""
    monkeypatch.setenv(RECORD_ENV_VAR, "1")
    with pytest.raises(LLMTransportError, match="nothing"):
        MockLLMProvider("c", responses=["{}"]).complete(HELLO)


def test_the_mock_remembers_what_it_was_shown() -> None:
    """The INV-6/INV-11 tests scan this. The guarantee is about the whole
    conversation, not the last turn."""
    provider = MockLLMProvider("c", responses=["{}"])
    provider.complete([Message(role="system", content="rules"), *HELLO])
    assert provider.prompts() == ["rules", "propose a genome"]


def test_the_mock_reads_a_recorded_file(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    root.mkdir()
    (root / "0.json").write_text(json.dumps({"text": '{"value": 7}'}), encoding="utf-8")
    provider = MockLLMProvider("campaign", root=tmp_path)
    assert provider.complete(HELLO, response_model=Tiny).parsed == Tiny(value=7)


# ---------------------------------------------------------------------------
# pricing, and the cap
# ---------------------------------------------------------------------------
def test_the_shipped_pricing_file_prices_the_default_model() -> None:
    """The configured default must be sendable out of the box; otherwise the
    first real call fails on a file nobody thought to look at."""
    assert "glm-4-plus" in load_pricing(PRICING)


def test_an_unpriced_model_is_refused_rather_than_charged_zero() -> None:
    """Zero makes ``daily_cost_cap_eur`` unreachable and the cap silently
    infinite — a spend limit that cannot be hit is not a limit."""
    with pytest.raises(ConfigError, match="no price is recorded"):
        _provider(lambda r: httpx.Response(200), model="glm-9-imaginary")


def test_a_missing_pricing_file_is_refused() -> None:
    with pytest.raises(ConfigError, match="pricing file"):
        load_pricing(Path("configs/does_not_exist.yaml"))


def test_the_cap_refuses_before_anything_is_sent() -> None:
    """The whole point. A cap enforced on the way out has already spent the
    money it was meant to prevent."""
    sent: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json=_reply("{}"))

    provider = _provider(handler, daily_cost_cap_eur=1e-9)
    with pytest.raises(BudgetExceeded, match="not sent"):
        provider.complete(HELLO)
    assert sent == []


def test_a_cap_of_zero_is_off_rather_than_impossible() -> None:
    """``0`` reads as "no cap" in this configuration; a literal zero-euro cap
    would make the platform unable to make any call at all, which is a way of
    disabling the LLM that nobody would choose by setting a *cost* key."""
    ledger = DailyLedger(None, cap_eur=0.0)
    ledger.check(999.0)


def test_the_ledger_reads_the_day_back_rather_than_counting_in_memory() -> None:
    """A process restart must not reset the day's spend to zero, which is the
    obvious way to make a cap not a cap."""

    class Store:
        def llm_cost_since(self, start_ms: int) -> float:
            return 9.99

    ledger = DailyLedger(Store(), cap_eur=10.0)
    with pytest.raises(BudgetExceeded):
        ledger.check(0.02)


def test_the_price_is_charged_per_million_tokens() -> None:
    provider = _provider(lambda r: httpx.Response(200, json=_reply("{}")), model="glm-4-air")
    assert provider.cost_eur(1_000_000, 0) == pytest.approx(0.13)


# ---------------------------------------------------------------------------
# transport policy
# ---------------------------------------------------------------------------
def test_a_successful_call_parses_and_reports_its_usage() -> None:
    provider = _provider(lambda r: httpx.Response(200, json=_reply('{"value": 3}')))
    response = provider.complete(HELLO, response_model=Tiny)
    assert isinstance(response, LLMResponse)
    assert response.parsed == Tiny(value=3)
    assert (response.tokens_in, response.tokens_out) == (10, 5)
    assert response.provider == "glm"


def test_a_json_response_format_and_the_schema_both_travel() -> None:
    """GLM's structured-output support varies by model, so the adapter must work
    with plain JSON mode — which means the schema has to be in the conversation
    as well as in the request field."""
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_reply('{"value": 1}'))

    _provider(handler).complete(HELLO, response_model=Tiny)
    body = seen[0]
    assert body["response_format"] == {"type": "json_object"}
    assert any("value" in m["content"] for m in body["messages"] if m["role"] == "system")


def test_a_malformed_reply_is_repaired_exactly_once() -> None:
    """Section 12.2: one retry with the error appended, then ``malformed``.
    Repairing indefinitely would let a model that cannot produce the schema burn
    the whole daily budget on one proposal."""
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_reply("not json at all"))

    response = _provider(handler).complete(HELLO, response_model=Tiny)
    assert len(bodies) == 2, "expected exactly one repair attempt"
    assert response.parsed is None
    assert response.finish_reason == "malformed"
    assert "did not validate" in bodies[1]["messages"][-1]["content"]


def test_a_repaired_reply_is_accepted() -> None:
    replies = iter(["nonsense", '{"value": 42}'])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_reply(next(replies)))

    assert _provider(handler).complete(HELLO, response_model=Tiny).parsed == Tiny(value=42)


def test_a_429_is_retried() -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) < 3:
            return httpx.Response(429)
        return httpx.Response(200, json=_reply("{}"))

    assert _provider(handler, max_retries=3).complete(HELLO).text == "{}"
    assert len(attempts) == 3


def test_a_400_is_not_retried() -> None:
    """A 4xx other than 429 is this platform being wrong about the request.
    Retrying repeats the mistake, more expensively and more slowly."""
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(400, json={"error": "bad request"})

    with pytest.raises(LLMTransportError, match="HTTP 400"):
        _provider(handler, max_retries=3).complete(HELLO)
    assert len(attempts) == 1


def test_401_is_not_in_the_retry_set() -> None:
    """Guards the test above: a bad key retried three times is three chances to
    get the account rate-limited for a problem no retry can fix."""
    assert 401 not in RETRYABLE_STATUS
    assert 403 not in RETRYABLE_STATUS
    assert 429 in RETRYABLE_STATUS


def test_an_empty_conversation_is_refused() -> None:
    with pytest.raises(LLMTransportError, match="empty conversation"):
        _provider(lambda r: httpx.Response(200)).complete([])


def test_a_reply_with_no_choices_is_a_transport_error_not_a_crash() -> None:
    provider = _provider(lambda r: httpx.Response(200, json={"id": "x", "choices": []}))
    with pytest.raises(LLMTransportError, match="no choices"):
        provider.complete(HELLO)


# ---------------------------------------------------------------------------
# the key
# ---------------------------------------------------------------------------
def test_the_api_key_is_never_on_the_instance() -> None:
    """Section 21.1: never logged, never in an artifact. A traceback prints
    ``vars()``, and a key on an attribute would ride out in every one of them."""
    provider = _provider(lambda r: httpx.Response(200, json=_reply("{}")))
    assert "test-key-not-real" not in repr(provider)
    assert not any("test-key-not-real" in str(v) for v in vars(provider).values())


def test_an_absent_key_is_refused_with_a_usable_message() -> None:
    """No ``GLM_API_KEY`` exists in this repository, and this is what that looks
    like from the inside: a refusal that names where to put one, not a call that
    fails somewhere in the transport with a 401."""
    with pytest.raises(ConfigError, match="Keychain"):
        GLMProvider(api_key="", pricing_file=PRICING)
