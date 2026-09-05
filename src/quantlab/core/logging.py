"""Structured logging with secret redaction (master spec sections 17 and 21).

JSON lines go to stderr and, when a log directory and date are supplied, to
``<artifacts>/logs/quantlab-YYYY-MM-DD.jsonl``.

This module deliberately never reads a clock: per spec section 0.3 no code in
``core/`` may call :func:`datetime.datetime.now`.  The caller (the CLI) passes
``log_date`` explicitly, which also makes the file naming testable.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import sys
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final, TextIO

import structlog

from quantlab.core.errors import ConfigError

__all__ = [
    "DEFAULT_REDACT_KEYS",
    "REDACTED",
    "SECRETLIKE_RE",
    "bind_context",
    "clear_context",
    "configure_logging",
    "get_logger",
    "key_is_sensitive",
    "log_context",
    "make_redactor",
    "redact_value",
]

#: Replacement written in place of anything that looks like a secret.
REDACTED: Final[str] = "***"

#: Keys redacted when ``logging.redact_keys`` is not configured.
DEFAULT_REDACT_KEYS: Final[tuple[str, ...]] = (
    "api_key",
    "secret",
    "token",
    "password",
    "authorization",
)

#: Values that look like credentials are redacted even under an innocent key.
SECRETLIKE_RE: Final[re.Pattern[str]] = re.compile(r"(?i)(sk|key|token)[-_]?[a-z0-9]{16,}")

_MAX_DEPTH: Final[int] = 12

_SEGMENT_SPLIT_RE: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")


def redact_value(value: Any) -> Any:
    """Redact secret-looking substrings inside a single value."""
    if isinstance(value, str):
        return SECRETLIKE_RE.sub(REDACTED, value)
    return value


def _segments(name: str) -> list[str]:
    """Split a key into lower-case alphanumeric segments: ``X-API-Key`` -> ``[x, api, key]``."""
    return [part for part in _SEGMENT_SPLIT_RE.split(name.lower()) if part]


def key_is_sensitive(key: str, redact_keys: Sequence[str]) -> bool:
    """True if ``key`` names a secret according to ``redact_keys``.

    Matching is on whole segments rather than raw substrings, so ``api_key``
    catches ``GLM_API_KEY`` and ``X-API-Key`` while ``token`` leaves the
    ``tokens_in``/``tokens_out`` counters that spec section 17 requires us to log.
    """
    parts = _segments(key)
    for needle in redact_keys:
        needle_parts = _segments(needle)
        if not needle_parts:
            continue
        span = len(needle_parts)
        if any(parts[i : i + span] == needle_parts for i in range(len(parts) - span + 1)):
            return True
    return False


def _redact(value: Any, redact_keys: Sequence[str], depth: int = 0) -> Any:
    if depth >= _MAX_DEPTH:
        return value
    if isinstance(value, Mapping):
        out: dict[Any, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and key_is_sensitive(key, redact_keys):
                out[key] = REDACTED
            else:
                out[key] = _redact(item, redact_keys, depth + 1)
        return out
    if isinstance(value, (list, tuple, set)):
        rendered = [_redact(item, redact_keys, depth + 1) for item in value]
        return type(value)(rendered) if isinstance(value, (list, tuple)) else set(rendered)
    return redact_value(value)


def make_redactor(
    redact_keys: Sequence[str] = DEFAULT_REDACT_KEYS,
) -> Any:
    """Build a structlog processor that strips secrets from every event.

    Keys are matched case-insensitively as substrings, so ``GLM_API_KEY`` and
    ``x-api-key`` are both caught by the configured key ``api_key``.
    """
    needles = tuple(key.lower() for key in redact_keys)

    def processor(
        _logger: Any,
        _method_name: str,
        event_dict: MutableMapping[str, Any],
    ) -> MutableMapping[str, Any]:
        redacted = _redact(dict(event_dict), needles)
        # _redact always returns a dict for a Mapping input.
        assert isinstance(redacted, dict)
        return redacted

    return processor


def _log_path(log_dir: Path, log_date: dt.date) -> Path:
    return log_dir / f"quantlab-{log_date.isoformat()}.jsonl"


def configure_logging(
    *,
    level: str = "INFO",
    json_output: bool = True,
    redact_keys: Sequence[str] = DEFAULT_REDACT_KEYS,
    log_dir: str | Path | None = None,
    log_date: dt.date | None = None,
    stream: TextIO | None = None,
) -> None:
    """Configure ``structlog`` and the stdlib root logger.

    Args:
        level: Minimum level name, e.g. ``"INFO"``.
        json_output: JSON lines when true, human-readable console output otherwise.
        redact_keys: Keys whose values are replaced by ``***``.
        log_dir: Directory for the dated JSONL log file; ``None`` disables it.
        log_date: Date used in the log file name.  Required with ``log_dir``.
        stream: Destination for the primary handler; defaults to ``sys.stderr``.

    Raises:
        ConfigError: if ``level`` is not a level name, or ``log_dir`` was given
            without ``log_date``.
    """
    numeric_level = logging.getLevelNamesMapping().get(level.upper())
    if numeric_level is None:
        raise ConfigError(f"unknown logging level {level!r}", level=level)

    handlers: list[logging.Handler] = [logging.StreamHandler(stream or sys.stderr)]

    if log_dir is not None:
        if log_date is None:
            raise ConfigError("log_date is required when log_dir is set")
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(_log_path(directory, log_date), encoding="utf-8"))

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
        existing.close()
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)
    root.setLevel(numeric_level)

    renderer: Any = (
        structlog.processors.JSONRenderer(sort_keys=True)
        if json_output
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            make_redactor(redact_keys),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str | None = None) -> Any:
    """Return a bound structlog logger."""
    return structlog.get_logger(name) if name else structlog.get_logger()


def bind_context(**values: Any) -> None:
    """Bind ``run_id``/``experiment_id``/``strategy_id``/``session_id`` etc. to all events."""
    structlog.contextvars.bind_contextvars(**values)


def clear_context() -> None:
    """Drop everything bound with :func:`bind_context`."""
    structlog.contextvars.clear_contextvars()


@contextmanager
def log_context(**values: Any) -> Iterator[None]:
    """Bind context for the duration of a block, then restore the previous state."""
    tokens = structlog.contextvars.bind_contextvars(**values)
    try:
        yield
    finally:
        structlog.contextvars.reset_contextvars(**tokens)
