"""What a model may be shown (master spec section 14.5; INV-6, INV-11).

Two invariants meet in this file, and they are different in kind:

* **INV-6** — nothing derived from the *test* partition ever reaches a prompt.
  Permanent, unconditional, and the reason the lockbox has any meaning: a model
  that had seen test numbers would launder them into every genome it proposed
  thereafter, and no later gate could tell.
* **INV-11** — while an evolution run is ``running``, nothing derived from
  *validation* reaches a prompt either. Narrower in time, wider in what it
  hides. Validation results reach the human, never the loop.

Both are enforced the same way: a report is **built from a whitelist** and then
**scanned**. Construction alone is not enough — a metric named in the whitelist
can still carry a validation-derived number if the caller passed the wrong run
— and a scan alone is not enough, because a scan cannot see a quantity it has
no literal to match. The two together are what section 14.5 asks for.

**Amendment 1.3(f): the scan also covers secrets and the environment.** The
original scan looked only for test-period dates and segment names, which reads
the threat as "the platform leaks its own answers". The other direction matters
as much and is easier to trip: a report assembled by string-formatting an
exception, a path, or a configuration dump can carry an API key or a machine's
environment straight into ``artifacts/llm/<id>/prompt.json``, which is written
to disk unencrypted and kept. Prompts are therefore scanned for the known secret
names of :mod:`quantlab.ports.secrets`, for values that look like credentials,
and for the process environment's own values before anything is sent.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Final

from pydantic import BaseModel, ConfigDict

from quantlab.core.errors import LockboxViolation
from quantlab.ports.secrets import OPTIONAL_SECRETS, REQUIRED_SECRETS

__all__ = [
    "ENVIRONMENT_ALLOWLIST",
    "MIN_SECRET_LENGTH",
    "REDACTED_FIELDS",
    "RedactedReport",
    "assert_prompt_is_clean",
    "build_generation_report",
    "build_redacted_report",
    "scan_for_leaks",
]

#: The only fields a redacted report may carry (section 14.5). A whitelist and
#: not a blacklist: a field added to the store later is absent from a prompt
#: until somebody puts it here deliberately.
REDACTED_FIELDS: Final[tuple[str, ...]] = (
    "strategy_id",
    "code",
    "lineage",
    "params",
    "metrics_train",
    "metrics_val",
    "metrics_wf_oos",
    "gates",
    "soft_checks",
    "overfit_score",
    "verdict",
    "baseline_metrics_train_val",
    "rejected_ideas_summary",
    "validation_touches",
)

#: Segment name prefixes that must never appear in a report shown to a model.
_TEST_SEGMENTS: Final[tuple[str, ...]] = ("test", "lockbox")
_VAL_SEGMENTS: Final[tuple[str, ...]] = ("val", "validation", "wf_oos")

#: A string shorter than this is not treated as a credential even if it sits
#: under a secret-shaped name. Short values are overwhelmingly placeholders
#: (``""``, ``"none"``, ``"changeme"``) and matching them would make the scan
#: fire on every honest report until somebody disabled it.
MIN_SECRET_LENGTH: Final[int] = 12

#: Environment variables whose values are safe to appear in a prompt. Everything
#: else in ``os.environ`` is treated as potentially sensitive, because on a
#: developer's machine it usually is.
ENVIRONMENT_ALLOWLIST: Final[frozenset[str]] = frozenset(
    {"LANG", "LC_ALL", "PATH", "PWD", "SHELL", "TERM", "TZ", "USER", "HOME", "TMPDIR"}
)

#: Words that make a partition reference *carry a number*. The pair is what the
#: scan matches, rather than the partition name alone, and the distinction is
#: load-bearing in both directions: the system prompt has to be able to say "you
#: will never see the test period" without tripping its own guard, while "the
#: test partition metrics" and "test_segment sharpe" must both fire. A separator
#: of up to three characters of whitespace or punctuation covers prose ("the
#: test partition"), field names (``test_segment``) and segment syntax
#: (``test:0``) with one pattern.
_QUANTITY_WORDS: Final[str] = (
    "segment|partition|sharpe|sortino|metric|metrics|return|returns|result|results"
    "|equity|drawdown|score|pnl|profit"
)

_PARTITION_QUANTITY: Final[dict[str, re.Pattern[str]]] = {
    # No ``\b`` after the name: ``_`` is a word character, so ``\btest\b`` never
    # matches ``test_segment`` — the single most likely spelling of the leak.
    # The lookbehind is what keeps ``latest`` from reading as ``test``.
    name: re.compile(
        rf"(?<![A-Za-z0-9_]){name}[\s_:.-]{{0,3}}(?:{_QUANTITY_WORDS})\b", re.IGNORECASE
    )
    for name in ("test", "lockbox", "val", "validation", "wf_oos")
}

#: Names that mark a value as a credential wherever they appear, in any case and
#: with any separator.
_SECRET_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(api[_-]?key|secret|password|passwd|token|bearer|credential|private[_-]?key)",
    re.IGNORECASE,
)

#: Shapes that are credentials regardless of what they are called.
_CREDENTIAL_SHAPES: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),  # OpenAI-style
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key id
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}"),  # GitHub
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\."),  # JWT
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


class RedactedReport(BaseModel):
    """Everything a model may be told about a strategy, and nothing else."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy_id: str
    code: str = ""
    lineage: tuple[str, ...] = ()
    params: Mapping[str, Any] = {}
    metrics_train: Mapping[str, float | None] = {}
    metrics_val: Mapping[str, float | None] = {}
    metrics_wf_oos: Mapping[str, float | None] = {}
    gates: Mapping[str, bool] = {}
    soft_checks: Mapping[str, Any] = {}
    overfit_score: float | None = None
    verdict: str | None = None
    baseline_metrics_train_val: Mapping[str, float | None] = {}
    rejected_ideas_summary: str = ""
    validation_touches: int = 0


# ---------------------------------------------------------------------------
# building
# ---------------------------------------------------------------------------
def _segment_is(segment: str, prefixes: Sequence[str]) -> bool:
    return any(segment == p or segment.startswith(p + ":") for p in prefixes)


def _runs(store: Any, strategy_id: str) -> list[Any]:
    return [r for r in store.query_runs(strategy_id=strategy_id) if r.status == "ok"]


def _metrics_of(
    store: Any, runs: Iterable[Any], prefixes: Sequence[str]
) -> dict[str, float | None]:
    """Metrics of the *last* matching run, or ``{}`` if there is none.

    The last rather than a merge across runs: two runs on the same segment are
    two measurements, and silently averaging them would report a number that was
    never observed.
    """
    matching = [r for r in runs if _segment_is(r.segment, prefixes)]
    if not matching:
        return {}
    return dict(store.metrics_for(matching[-1].run_id))


def build_redacted_report(store: Any, strategy_id: str) -> RedactedReport:
    """The report a model may see **outside** an active evolution run (INV-6).

    Test-segment runs are filtered out by segment name, and neither
    ``lockbox_access`` nor ``split_policy.test_*`` is read at all — there is no
    call to them in this function, which is a stronger statement than filtering
    their results would be.
    """
    runs = _runs(store, strategy_id)
    if any(_segment_is(r.segment, _TEST_SEGMENTS) for r in runs):
        runs = [r for r in runs if not _segment_is(r.segment, _TEST_SEGMENTS)]

    verdicts = list(store.verdicts_for(strategy_id))
    latest = verdicts[-1] if verdicts else None
    version = store.get_strategy_version(strategy_id)
    family = store.find_family(getattr(version, "family_id", "")) if version is not None else None

    return RedactedReport(
        strategy_id=strategy_id,
        lineage=tuple(str(v.strategy_id) for v in store.lineage(strategy_id)),
        metrics_train=_metrics_of(store, runs, ("train",)),
        metrics_val=_metrics_of(store, runs, ("val",)),
        metrics_wf_oos=_metrics_of(store, runs, ("wf_oos",)),
        overfit_score=getattr(latest, "overfit_score", None),
        verdict=getattr(latest, "verdict", None),
        validation_touches=int(getattr(family, "validation_touches", 0) or 0),
    )


def build_generation_report(store: Any, evolution_id: str, gen_index: int) -> RedactedReport:
    """The **only** report a model may see while an evolution run is active (INV-11).

    Stricter than :func:`build_redacted_report` in exactly one way that matters:
    validation is filtered out as well as test. A generation report carries
    training and inner-fold numbers, and the inner folds are cut from the
    training segment, so nothing here has ever touched validation.

    The strictness is unconditional rather than keyed off the run's status. A
    report built for a generation is a report about a search in progress; if the
    run has since stopped, the honest thing is still to answer the question that
    was asked, and a caller wanting the fuller picture asks for the other report
    by name.
    """
    run = store.find_evolution_run(evolution_id)
    if run is None:
        raise LockboxViolation(
            "no such evolution run; refusing to build a generation report from nothing",
            evolution_id=evolution_id,
        )

    lines: list[str] = []
    metrics_train: dict[str, float | None] = {}
    for candidate in store.candidates_for(evolution_id, gen_index):
        strategy_id = getattr(candidate, "strategy_id", None)
        if not strategy_id:
            continue
        runs = [
            r
            for r in _runs(store, str(strategy_id))
            if not _segment_is(r.segment, (*_TEST_SEGMENTS, *_VAL_SEGMENTS))
        ]
        if not runs:
            continue
        metrics_train = _metrics_of(store, runs, ("train",)) or metrics_train
        lines.append(str(getattr(candidate, "candidate_id", "")))

    return RedactedReport(
        strategy_id=f"{evolution_id}:gen{gen_index}",
        lineage=tuple(lines),
        metrics_train=metrics_train,
        rejected_ideas_summary=f"generation {gen_index} of {evolution_id}",
    )


# ---------------------------------------------------------------------------
# scanning
# ---------------------------------------------------------------------------
def _iso_dates(*timestamps: int | None) -> list[str]:
    dates = []
    for ts in timestamps:
        if ts is None:
            continue
        dates.append(datetime.fromtimestamp(ts / 1000, tz=UTC).date().isoformat())
    return dates


def _secret_values(secrets: Any) -> list[str]:
    values = []
    for name in (*REQUIRED_SECRETS, *OPTIONAL_SECRETS):
        try:
            value = secrets.try_get(name) if secrets is not None else None
        except Exception:  # a broken secret backend must not open the gate
            value = None
        if value and len(value) >= MIN_SECRET_LENGTH:
            values.append(value)
    return values


def scan_for_leaks(
    text: str,
    *,
    split: Any = None,
    forbid_validation: bool = False,
    secrets: Any = None,
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    """Every reason ``text`` must not be sent, in the order they were found.

    Returns a list rather than raising so a caller can report all of them at
    once; :func:`assert_prompt_is_clean` is the raising form. An empty list is
    the only clean result — there is no "warning" tier here, because a prompt is
    either sendable or it is not.
    """
    problems: list[str] = []
    haystack = text.lower()

    for name in _TEST_SEGMENTS:
        if _PARTITION_QUANTITY[name].search(haystack):
            problems.append(
                f"INV-6: the text pairs the {name} partition with a quantity, "
                "which no prompt may reference"
            )

    if split is not None:
        for date in _iso_dates(
            getattr(split, "test_start_ts", None), getattr(split, "test_end_ts", None)
        ):
            if date in text:
                problems.append(f"INV-6: the test-period date {date} appears in the text")
        if forbid_validation:
            for date in _iso_dates(
                getattr(split, "val_start_ts", None), getattr(split, "val_end_ts", None)
            ):
                if date in text:
                    problems.append(
                        f"INV-11: the validation-period date {date} appears in the text "
                        "while an evolution run is active"
                    )

    if forbid_validation:
        for name in _VAL_SEGMENTS:
            if _PARTITION_QUANTITY[name].search(haystack):
                problems.append(
                    f"INV-11: the text names a {name} quantity, which the loop may not see"
                )

    # --- amendment 1.3(f): secrets and the environment ----------------------
    for pattern in _CREDENTIAL_SHAPES:
        match = pattern.search(text)
        if match is not None:
            problems.append(
                "a value shaped like a credential appears in the text "
                f"(matched {pattern.pattern!r}); prompts are written to disk unencrypted"
            )

    for line in text.splitlines():
        if not _SECRET_NAME_PATTERN.search(line):
            continue
        _, _, tail = line.partition("=") if "=" in line else line.partition(":")
        candidate = tail.strip().strip("\"'")
        if len(candidate) >= MIN_SECRET_LENGTH:
            problems.append(
                "a secret-shaped name carries a value in the text: "
                f"{line.strip()[:40]!r}; redact it before sending"
            )

    for value in _secret_values(secrets):
        if value in text:
            problems.append("a configured secret's value appears verbatim in the text")

    env = os.environ if environ is None else environ
    for name, value in env.items():
        if name in ENVIRONMENT_ALLOWLIST or len(value) < MIN_SECRET_LENGTH:
            continue
        if value in text:
            problems.append(
                f"the value of environment variable {name} appears verbatim in the text"
            )

    return problems


def assert_prompt_is_clean(
    text: str,
    *,
    split: Any = None,
    forbid_validation: bool = False,
    secrets: Any = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Return ``text`` if it may be sent; raise otherwise.

    Raises :class:`~quantlab.core.errors.LockboxViolation` rather than an LLM
    error: what this catches is a partition boundary being crossed, and section
    14.6 says a ``LockboxViolation`` MUST NOT be caught anywhere but the CLI top
    level. An adapter that could catch it and retry would be able to turn a
    leak into a warning.
    """
    problems = scan_for_leaks(
        text,
        split=split,
        forbid_validation=forbid_validation,
        secrets=secrets,
        environ=environ,
    )
    if problems:
        raise LockboxViolation(
            "this prompt may not be sent: " + "; ".join(problems),
            n_problems=len(problems),
        )
    return text
