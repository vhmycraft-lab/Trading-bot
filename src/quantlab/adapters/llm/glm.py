"""The GLM provider (master spec section 12.2).

An OpenAI-shaped ``/chat/completions`` client with four things bolted on that
the shape does not give you, each of which exists because of a specific way this
platform can be harmed:

* **A daily cost ledger, checked before the request is sent.** Not after: a cap
  enforced on the way out has already spent the money it was meant to prevent.
  :class:`~quantlab.core.errors.BudgetExceeded` means *nothing was sent*.
* **A price that must be known.** A model absent from ``configs/llm_pricing.yaml``
  is refused rather than charged zero, because zero makes the cap unreachable.
* **Validation in the adapter, with exactly one repair attempt.** A malformed
  proposal costs a round trip and never a backtest.
* **The key is never logged, never in a repr, never in an artifact.** It is read
  once from ``Secrets`` into a header and is not stored on the instance in any
  form a traceback would print.

Retries are on 429 and 5xx and timeouts only. A 4xx other than 429 is the
platform being wrong about the request, and retrying it just repeats the mistake
more expensively.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import httpx
import yaml
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from quantlab.core.errors import (
    BudgetExceeded,
    ConfigError,
    LLMMalformedResponse,
    LLMTransportError,
)
from quantlab.ports.llm import STATUS_MALFORMED, STATUS_OK, LLMResponse, Message

__all__ = ["RETRYABLE_STATUS", "DailyLedger", "GLMProvider", "load_pricing"]

#: HTTP statuses worth trying again. Everything else in 4xx is a bad request.
RETRYABLE_STATUS: Final[frozenset[int]] = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

_MTOK: Final[float] = 1_000_000.0


class _Retryable(Exception):
    """Internal marker so tenacity retries transport faults and nothing else."""


def load_pricing(path: Path) -> dict[str, dict[str, float]]:
    """Read ``configs/llm_pricing.yaml``.

    Raises:
        ConfigError: the file is missing or malformed. Failing here is the point
            — an unpriced provider cannot enforce a cost cap.
    """
    if not path.is_file():
        raise ConfigError(
            f"LLM pricing file {path} is missing; without prices the daily cost cap "
            "cannot be enforced and every request would be charged zero",
            path=str(path),
        )
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"LLM pricing file {path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigError(f"LLM pricing file {path} must map model names to prices")

    table: dict[str, dict[str, float]] = {}
    for model, prices in raw.items():
        if not isinstance(prices, Mapping):
            raise ConfigError(f"pricing for {model!r} in {path} is not a mapping")
        missing = {"input_eur_per_mtok", "output_eur_per_mtok"} - set(prices)
        if missing:
            raise ConfigError(
                f"pricing for {model!r} in {path} is missing " + ", ".join(sorted(missing))
            )
        table[str(model)] = {k: float(prices[k]) for k in prices}
    return table


class DailyLedger:
    """What today's calls have cost, in euros.

    Kept in the store in production (``llm_interaction`` rows carry ``cost_eur``
    and ``created_at``), and re-read rather than accumulated in memory: a
    process restart must not reset the day's spend to zero, which is the obvious
    way to make a cap not a cap.
    """

    def __init__(self, store: Any = None, *, cap_eur: float) -> None:
        self._store = store
        self._cap = float(cap_eur)
        self._local = 0.0

    def spent_today(self) -> float:
        if self._store is None:
            return self._local
        start = int(
            datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000
        )
        reader = getattr(self._store, "llm_cost_since", None)
        if reader is None:
            return self._local
        return float(reader(start))

    def check(self, projected_eur: float) -> None:
        """Refuse a request that would cross the cap. Nothing is sent."""
        if self._cap <= 0:
            return
        spent = self.spent_today()
        if spent + projected_eur > self._cap:
            raise BudgetExceeded(
                f"this request would bring today's LLM spend to "
                f"EUR {spent + projected_eur:.4f}, over the cap of EUR {self._cap:.2f}; "
                "it was not sent",
                spent_eur=round(spent, 6),
                projected_eur=round(projected_eur, 6),
                cap_eur=self._cap,
            )

    def charge(self, eur: float) -> None:
        self._local += float(eur)


class GLMProvider:
    """``POST {base_url}/chat/completions``, with a budget and a repair attempt."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "glm-4-plus",
        base_url: str = "https://open.bigmodel.cn/api/paas/v4",
        pricing_file: Path = Path("configs/llm_pricing.yaml"),
        timeout_s: float = 60.0,
        max_retries: int = 3,
        daily_cost_cap_eur: float = 10.0,
        store: Any = None,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ConfigError(
                "GLM_API_KEY is not configured; set it in the macOS Keychain "
                "(service 'quantlab') or in .env, and never on the command line"
            )
        self.name = "glm"
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._pricing = load_pricing(pricing_file)
        if model not in self._pricing:
            raise ConfigError(
                f"no price is recorded for model {model!r} in {pricing_file}; refusing to "
                "send, because an unpriced model makes the daily cost cap unreachable",
                model=model,
                priced=sorted(self._pricing),
            )
        self._max_retries = max(1, int(max_retries))
        self._ledger = DailyLedger(store, cap_eur=daily_cost_cap_eur)
        # The key lives in the client's default headers and on no attribute of
        # this object, so no repr, traceback or `vars()` dump can print it.
        self._client = client or httpx.Client(
            base_url=self._base_url,
            timeout=timeout_s,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    # -- pricing ------------------------------------------------------------
    def cost_eur(self, tokens_in: int, tokens_out: int) -> float:
        prices = self._pricing[self.model]
        return (
            tokens_in * prices["input_eur_per_mtok"] + tokens_out * prices["output_eur_per_mtok"]
        ) / _MTOK

    # -- transport ----------------------------------------------------------
    def _post(self, body: Mapping[str, Any]) -> dict[str, Any]:
        @retry(
            reraise=True,
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            retry=retry_if_exception_type(_Retryable),
        )
        def _attempt() -> dict[str, Any]:
            try:
                response = self._client.post("/chat/completions", json=dict(body))
            except httpx.TimeoutException as exc:
                raise _Retryable(f"timeout talking to GLM: {exc}") from exc
            except httpx.HTTPError as exc:
                raise LLMTransportError(f"could not reach GLM: {exc}") from exc

            if response.status_code in RETRYABLE_STATUS:
                raise _Retryable(f"GLM returned {response.status_code}")
            if response.status_code >= 400:
                # Not retried: a 4xx is this platform being wrong about the
                # request, and repeating it repeats the mistake.
                raise LLMTransportError(
                    f"GLM refused the request with HTTP {response.status_code}",
                    status=response.status_code,
                )
            try:
                payload = response.json()
            except ValueError as exc:
                raise LLMTransportError("GLM returned a body that is not JSON") from exc
            if not isinstance(payload, dict):
                raise LLMTransportError("GLM returned a JSON body that is not an object")
            return payload

        try:
            return _attempt()
        except _Retryable as exc:
            raise LLMTransportError(
                f"GLM did not answer after {self._max_retries} attempts: {exc}"
            ) from exc

    # -- the call -----------------------------------------------------------
    def complete(
        self,
        messages: Sequence[Message],
        *,
        response_model: type[BaseModel] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 8000,
        seed: int | None = None,
    ) -> LLMResponse:
        if not messages:
            raise LLMTransportError("refusing to send an empty conversation")

        # Projected on the prompt alone plus the full output allowance: the
        # check has to happen before the tokens exist, so it assumes the reply
        # runs to `max_tokens`. Erring high is the safe direction for a cap.
        projected_in = sum(len(m.content) for m in messages) // 4
        self._ledger.check(self.cost_eur(projected_in, max_tokens))

        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            body["seed"] = seed
        if response_model is not None:
            # Plain JSON mode plus pydantic. GLM's structured-output support
            # varies by model, so the schema also travels in the conversation
            # and the reply is validated here regardless of what the mode did.
            body["response_format"] = {"type": "json_object"}
            body["messages"].append(
                {
                    "role": "system",
                    "content": (
                        "Reply with a single JSON object matching this schema, and "
                        "nothing else:\n" + json.dumps(response_model.model_json_schema(), indent=2)
                    ),
                }
            )

        started = time.monotonic()
        payload = self._post(body)
        latency_ms = max(0, int((time.monotonic() - started) * 1000))

        text, finish_reason, raw_id = _extract(payload)
        usage = payload.get("usage") or {}
        tokens_in = int(usage.get("prompt_tokens", projected_in))
        tokens_out = int(usage.get("completion_tokens", len(text) // 4))
        self._ledger.charge(self.cost_eur(tokens_in, tokens_out))

        parsed: BaseModel | None = None
        if response_model is not None:
            parsed, repair = _validate(text, response_model)
            if parsed is None and repair is not None:
                # One retry, with the validation error appended (section 12.2).
                body["messages"].append({"role": "assistant", "content": text})
                body["messages"].append({"role": "user", "content": repair})
                retry_payload = self._post(body)
                text, finish_reason, raw_id = _extract(retry_payload)
                usage = retry_payload.get("usage") or {}
                tokens_in += int(usage.get("prompt_tokens", 0))
                tokens_out += int(usage.get("completion_tokens", 0))
                self._ledger.charge(self.cost_eur(tokens_in, tokens_out))
                parsed, _ = _validate(text, response_model)

        return LLMResponse(
            text=text,
            parsed=parsed,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            model=self.model,
            provider=self.name,
            raw_id=raw_id,
            finish_reason=finish_reason
            if parsed is not None or response_model is None
            else STATUS_MALFORMED,
        )

    def status_of(self, response: LLMResponse, *, wanted_model: bool) -> str:
        """``ok`` or ``malformed``, for ``llm_interaction.status``."""
        if wanted_model and response.parsed is None:
            return STATUS_MALFORMED
        return STATUS_OK

    def close(self) -> None:
        self._client.close()

    def __repr__(self) -> str:
        # No key, no client, no headers. Named explicitly rather than left to
        # the default repr, which would print whatever gets added later.
        return f"GLMProvider(model={self.model!r}, base_url={self._base_url!r})"


def _extract(payload: Mapping[str, Any]) -> tuple[str, str, str | None]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMTransportError("GLM returned no choices")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise LLMTransportError("GLM returned a malformed choice")
    message = first.get("message")
    content = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(content, str):
        raise LLMTransportError("GLM returned a choice with no text content")
    raw_id = payload.get("id")
    return content, str(first.get("finish_reason", "stop")), str(raw_id) if raw_id else None


def _validate(text: str, model: type[BaseModel]) -> tuple[BaseModel | None, str | None]:
    """``(parsed, None)`` or ``(None, repair_instruction)``."""
    try:
        return model.model_validate_json(text), None
    except ValidationError as exc:
        return None, (
            "Your reply did not validate against the requested schema:\n"
            f"{exc}\n"
            "Return the corrected JSON object and nothing else."
        )


def raise_malformed(campaign: str, model_name: str) -> None:
    """Raise the error a caller should surface when a repair attempt also failed.

    Separate from :meth:`GLMProvider.complete` on purpose: the adapter's job is
    to report *what happened*, and whether a malformed proposal is fatal is the
    caller's policy. Section 12.2 says the adapter returns ``status='malformed'``;
    the proposer decides whether to draw a genome instead.
    """
    raise LLMMalformedResponse(
        f"campaign {campaign!r} received a reply from {model_name} that did not validate, "
        "and the one permitted repair attempt did not fix it",
        campaign=campaign,
        model=model_name,
    )
