"""The LLM proposer layer (master spec section 12).

Everything a model is allowed to see is assembled here, and nowhere else. That
is the whole point of the package boundary: INV-6 ("nothing derived from the
test partition is ever placed in an LLM prompt") and INV-11 (nor anything
derived from validation, while a run is active) are properties of *prompts*, and
prompts have exactly one source.

``research`` may import ``core`` and ``ports`` and nothing else (INV-8). It
therefore cannot reach a store directly and is handed one; it cannot reach the
lockbox at all.
"""

from __future__ import annotations

__all__: list[str] = []
