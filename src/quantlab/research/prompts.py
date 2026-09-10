"""Prompt templates and their rendering (master spec section 12.5).

Markdown with ``{{placeholders}}``, substituted by :meth:`str.replace` and
nothing more. Deliberately not Jinja: a template language can call, loop and
branch, which would make "what did the model see?" a question about program
execution rather than about a file plus a dictionary. The SHA-256 of the
template file is stored per interaction (``llm_interaction.prompt_template_sha256``),
and that hash only means something if rendering is a pure substitution.

Two properties this module is responsible for:

* **Every placeholder gets filled.** A ``{{market}}`` left in a sent prompt is a
  caller that forgot an argument, and the model would answer the literal braces.
  Rendering refuses rather than sending it.
* **No value smuggles a placeholder in.** Substituted text is never re-scanned,
  so a value containing ``{{indicators}}`` cannot cause a second substitution.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Final

__all__ = [
    "PROMPT_DIR",
    "TEMPLATE_NAMES",
    "known_templates",
    "render",
    "template_sha256",
    "template_text",
]

PROMPT_DIR: Final[Path] = Path(__file__).resolve().parent / "prompts"

#: The templates section 12.5 defines. ``review.md`` from spec 1.0 is gone along
#: with ``ReviewDecision``; a name not in this tuple is not a template.
TEMPLATE_NAMES: Final[tuple[str, ...]] = (
    "system",
    "propose_genome",
    "repair",
    "review_generation",
)

_PLACEHOLDER: Final[re.Pattern[str]] = re.compile(r"\{\{\s*([a-z_][a-z0-9_]*)\s*\}\}")


def known_templates() -> tuple[str, ...]:
    """The templates actually present on disk, sorted.

    Read from the filesystem rather than returned from :data:`TEMPLATE_NAMES`,
    so a test can assert the two agree — a template deleted in a refactor would
    otherwise still be listed as available right up until something rendered it.
    """
    return tuple(sorted(path.stem for path in PROMPT_DIR.glob("*.md")))


@lru_cache(maxsize=len(TEMPLATE_NAMES))
def template_text(name: str) -> str:
    """The raw text of template ``name``."""
    if name not in TEMPLATE_NAMES:
        raise KeyError(f"{name!r} is not a prompt template; expected one of {TEMPLATE_NAMES}")
    path = PROMPT_DIR / f"{name}.md"
    if not path.is_file():
        raise FileNotFoundError(f"prompt template {name!r} is missing from {PROMPT_DIR}")
    return path.read_text(encoding="utf-8")


def template_sha256(name: str) -> str:
    """The hash stored on every interaction that used this template."""
    return hashlib.sha256(template_text(name).encode("utf-8")).hexdigest()


def render(name: str, values: Mapping[str, str] | None = None) -> str:
    """Render template ``name``, substituting every placeholder exactly once.

    Raises:
        KeyError: a placeholder in the template has no value, or a value was
            supplied for a placeholder the template does not contain. Both are
            caller mistakes and both are silent in a substitution that only
            replaces what it finds.
    """
    text = template_text(name)
    supplied = dict(values or {})
    needed = {match.group(1) for match in _PLACEHOLDER.finditer(text)}

    missing = needed - supplied.keys()
    if missing:
        raise KeyError(
            f"prompt {name!r} has unfilled placeholder(s): " + ", ".join(sorted(missing))
        )
    extra = supplied.keys() - needed
    if extra:
        raise KeyError(f"prompt {name!r} has no placeholder for: " + ", ".join(sorted(extra)))

    # One pass over the template, substituting from `supplied`. A value is
    # emitted verbatim and never re-scanned, so a value that itself contains
    # `{{...}}` reaches the model as those characters rather than triggering a
    # second substitution.
    return _PLACEHOLDER.sub(lambda m: supplied[m.group(1)], text)
