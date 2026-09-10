"""LLM provider adapters (master spec section 12.2, 12.3).

``MockLLMProvider`` is imported eagerly because every evolution and proposer
test uses it. ``GLMProvider`` is not: it pulls in ``httpx`` and ``tenacity`` and
reads a pricing file, and a test suite that never talks to a network should not
pay for that at import time. Ask for it by name.
"""

from __future__ import annotations

from quantlab.adapters.llm.mock import MockLLMProvider

__all__ = ["MockLLMProvider"]
