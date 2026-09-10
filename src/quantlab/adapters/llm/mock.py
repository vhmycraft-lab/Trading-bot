"""A provider that replays recorded responses (master spec section 12.3).

Every evolution and proposer test uses this, which is what makes the whole
search testable offline and deterministically. It is not a stub that returns
plausible-looking text: it replays *real* responses captured from a real model,
in order, so a test exercises the same parsing, validation and repair paths a
live call would.

Two decisions worth stating:

* **Exhaustion is an error, not a wrap-around.** A campaign that asks for a
  seventh response when six were recorded has changed shape since the fixtures
  were made. Cycling would quietly answer the seventh question with the first
  answer and the test would keep passing while measuring nothing.
* **Record mode is human-triggered only** (``QUANTLAB_LLM_RECORD=1``). It costs
  money and reaches a network, so it may not be something a test run can switch
  on for itself.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ValidationError

from quantlab.core.errors import LLMMalformedResponse, LLMTransportError
from quantlab.ports.llm import LLMResponse, Message

__all__ = ["RECORD_ENV_VAR", "MockLLMProvider"]

#: Set to ``1`` by a person, at a shell, on purpose.
RECORD_ENV_VAR: Final[str] = "QUANTLAB_LLM_RECORD"

#: Where fixtures live, relative to the repository root.
FIXTURE_ROOT: Final[Path] = Path("tests/fixtures/llm")


class MockLLMProvider:
    """Replays ``<root>/<campaign>/<n>.json`` in order."""

    def __init__(
        self,
        campaign: str,
        *,
        root: Path | None = None,
        model: str = "mock-1",
        responses: Sequence[str] | None = None,
    ) -> None:
        self.name = "mock"
        self.model = model
        self.campaign = campaign
        self._root = (root or FIXTURE_ROOT) / campaign
        self._n = 0
        #: Calls this provider has been asked to make, in order. Tests assert
        #: against these rather than against a global log — what a prompt
        #: contained is exactly what INV-6 and INV-11 are about.
        self.calls: list[tuple[Message, ...]] = []
        self._inline: list[str] | None = list(responses) if responses is not None else None

    # -- replay -------------------------------------------------------------
    def _next_text(self) -> str:
        if self._inline is not None:
            if self._n >= len(self._inline):
                raise LLMTransportError(
                    f"mock provider for campaign {self.campaign!r} has no response "
                    f"{self._n}; {len(self._inline)} were supplied",
                    campaign=self.campaign,
                    index=self._n,
                )
            return self._inline[self._n]

        path = self._root / f"{self._n}.json"
        if not path.is_file():
            raise LLMTransportError(
                f"mock provider has no recorded response at {path}; the campaign has "
                "asked for more calls than were recorded, which means it no longer "
                "has the shape the fixtures were captured from",
                campaign=self.campaign,
                index=self._n,
                path=str(path),
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        return (
            str(payload["text"])
            if isinstance(payload, dict) and "text" in payload
            else path.read_text(encoding="utf-8")
        )

    def complete(
        self,
        messages: Sequence[Message],
        *,
        response_model: type[BaseModel] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 8000,
        seed: int | None = None,
    ) -> LLMResponse:
        del temperature, max_tokens, seed
        if os.environ.get(RECORD_ENV_VAR) == "1":
            raise LLMTransportError(
                f"{RECORD_ENV_VAR}=1 asks the mock to record, but a mock has nothing "
                "to record from; wrap a real provider instead",
            )

        self.calls.append(tuple(messages))
        started = time.monotonic()
        text = self._next_text()
        self._n += 1

        parsed: BaseModel | None = None
        if response_model is not None:
            try:
                parsed = response_model.model_validate_json(text)
            except ValidationError as exc:
                # The mock reproduces the *shape* of a malformed live reply
                # rather than hiding it: a fixture that no longer validates is
                # exactly the signal a schema change should produce.
                raise LLMMalformedResponse(
                    f"recorded response {self._n - 1} of campaign {self.campaign!r} does "
                    f"not validate against {response_model.__name__}",
                    campaign=self.campaign,
                    index=self._n - 1,
                ) from exc

        return LLMResponse(
            text=text,
            parsed=parsed,
            tokens_in=sum(len(m.content) for m in messages) // 4,
            tokens_out=len(text) // 4,
            latency_ms=max(0, int((time.monotonic() - started) * 1000)),
            model=self.model,
            provider=self.name,
            raw_id=f"mock-{self.campaign}-{self._n - 1}",
            finish_reason="stop",
        )

    def cost_eur(self, tokens_in: int, tokens_out: int) -> float:
        """Nothing. A replayed response cost nothing to obtain.

        Returning a fabricated price would put invented euros into the daily
        ledger and into every report that reads it.
        """
        del tokens_in, tokens_out
        return 0.0

    # -- introspection for tests -------------------------------------------
    @property
    def n_calls(self) -> int:
        return len(self.calls)

    def prompts(self) -> list[str]:
        """Every message body this provider has been sent, flattened.

        The INV-6/INV-11 tests scan this: the guarantee is about what the model
        was *shown*, which is the whole conversation and not just the last turn.
        """
        return [m.content for call in self.calls for m in call]

    def last_prompt(self) -> str:
        if not self.calls:
            raise LLMTransportError("the mock provider has not been called")
        return "\n\n".join(m.content for m in self.calls[-1])

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"MockLLMProvider(campaign={self.campaign!r}, n_calls={self.n_calls})"
