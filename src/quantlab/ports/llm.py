"""Language-model port (master spec section 12.1).

The model is a **proposer**, and this port is shaped to keep it one. It can be
asked to complete a conversation and it can price what that cost; there is no
method by which it could read a run, a metric, or a segment. Everything a model
is allowed to know arrives as text that the caller assembled, which is what
makes INV-6 and INV-11 checkable at one place — the prompt — rather than
everywhere a model might otherwise have reached.

Two other narrowings worth stating, because they are easy to widen by accident:

* ``complete`` is **synchronous and single-shot**. Retries, back-off and the
  one repair attempt of section 12.2 belong to the adapter; a caller that could
  see them would start making policy out of them.
* ``LLMResponse`` carries ``tokens_in``/``tokens_out`` rather than a cost. Price
  is a property of the *account*, not of the exchange, and section 12.2 keeps a
  daily ledger against it — so the number that gets ledgered is derived once,
  by :meth:`LLMProvider.cost_eur`, from figures the provider reported.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel

__all__ = [
    "FINISH_REASONS",
    "MESSAGE_ROLES",
    "STATUS_MALFORMED",
    "STATUS_OK",
    "LLMProvider",
    "LLMResponse",
    "Message",
]

#: The roles a message may take. ``tool`` is deliberately absent: the model
#: proposes JSON and never calls anything (section 12).
MESSAGE_ROLES: Final[tuple[str, ...]] = ("system", "user", "assistant")

#: ``finish_reason`` values this platform distinguishes. Anything else a
#: provider invents is passed through verbatim rather than mapped onto one of
#: these — a reason nobody recognised should look unrecognised.
FINISH_REASONS: Final[tuple[str, ...]] = ("stop", "length", "content_filter")

#: ``llm_interaction.status`` for a call that produced a validated object.
STATUS_OK: Final[str] = "ok"

#: ``llm_interaction.status`` for a call whose text never validated, repair
#: attempt included. The row is still written: what the model said when it
#: failed is evidence about the prompt.
STATUS_MALFORMED: Final[str] = "malformed"


class Message(BaseModel, frozen=True):
    """One turn of a conversation.

    A pydantic model rather than a dataclass so that ``role`` is *enforced*
    rather than merely annotated — these objects are assembled from templates
    and from provider replies, and a typo in a role is the kind of thing a
    provider accepts and then quietly ignores.
    """

    role: Literal["system", "user", "assistant"]
    content: str


class LLMResponse(BaseModel, frozen=True):
    """What a provider returned, and what it cost to get it.

    ``parsed`` is ``None`` when no ``response_model`` was requested *and* when
    one was requested but nothing validated. Those are different situations and
    the caller tells them apart by ``finish_reason``/``status`` at the adapter
    boundary rather than by probing this field, which is why there is no
    ``is_valid`` helper here to be read as an authorisation.
    """

    text: str
    parsed: BaseModel | None = None
    tokens_in: int
    tokens_out: int
    latency_ms: int
    model: str
    provider: str
    raw_id: str | None = None
    finish_reason: str

    model_config = {"arbitrary_types_allowed": True}


@runtime_checkable
class LLMProvider(Protocol):
    """A model that can be asked to complete a conversation."""

    name: str
    model: str

    def complete(
        self,
        messages: Sequence[Message],
        *,
        response_model: type[BaseModel] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 8000,
        seed: int | None = None,
    ) -> LLMResponse:
        """Complete ``messages``.

        When ``response_model`` is given the implementation MUST validate the
        reply against it and put the result in :attr:`LLMResponse.parsed`;
        section 12.2 permits exactly one repair attempt before the response is
        returned unparsed. Validation happens in the adapter and not in the
        caller so that a malformed proposal costs a round trip and never a
        backtest.

        ``seed`` is passed through where the provider honours it. It is not a
        determinism guarantee — no remote model offers one — and nothing in this
        platform's reproducibility rests on it; the recorded ``prompt.json`` and
        ``response.json`` of section 12.2 are what make a campaign replayable.

        Raises:
            LLMTransportError: the request could not be completed.
            BudgetExceeded: the request would cross the daily cost cap and was
                therefore **not sent**.
        """
        ...

    def cost_eur(self, tokens_in: int, tokens_out: int) -> float:
        """Price a completed exchange in euros, per ``configs/llm_pricing.yaml``."""
        ...


def is_message_sequence(value: Any) -> bool:
    """Whether ``value`` is a non-empty sequence of :class:`Message`.

    Used by adapters to refuse a bare string, which every provider API would
    otherwise accept as a single user turn — silently dropping the system
    message that carries the genome schema and the INV-11 constraints.
    """
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str | bytes)
        and len(value) > 0
        and all(isinstance(item, Message) for item in value)
    )
