"""Redaction and log configuration (master spec section 17).

No realistic credential appears as a literal here; probe values are built at
run time so that ``tests/unit/test_no_secrets.py`` stays honest.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import logging
from pathlib import Path

import pytest
import structlog

from quantlab.core.errors import ConfigError
from quantlab.core.logging import (
    DEFAULT_REDACT_KEYS,
    REDACTED,
    bind_context,
    clear_context,
    configure_logging,
    get_logger,
    log_context,
    make_redactor,
    redact_value,
)


@pytest.fixture(autouse=True)
def _reset_logging() -> None:
    clear_context()
    yield
    clear_context()
    structlog.reset_defaults()
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()


def _redact(event: dict[str, object], keys=DEFAULT_REDACT_KEYS) -> dict[str, object]:
    return dict(make_redactor(keys)(None, "info", event))


# --- key-based redaction ---------------------------------------------------
@pytest.mark.parametrize(
    "key",
    ["api_key", "API_KEY", "GLM_API_KEY", "x-api-key", "secret", "Authorization", "password"],
)
def test_redacts_configured_keys_case_insensitively(key: str) -> None:
    assert _redact({key: "whatever"})[key] == REDACTED


def test_redacts_nested_keys() -> None:
    event = {"request": {"headers": {"Authorization": "Bearer abc"}, "url": "https://x/y"}}
    result = _redact(event)
    assert result["request"]["headers"]["Authorization"] == REDACTED  # type: ignore[index]
    assert result["request"]["url"] == "https://x/y"  # type: ignore[index]


def test_redacts_inside_lists() -> None:
    event = {"items": [{"token": "abc"}, {"safe": 1}]}
    result = _redact(event)
    assert result["items"][0]["token"] == REDACTED  # type: ignore[index]
    assert result["items"][1]["safe"] == 1  # type: ignore[index]


def test_leaves_innocent_keys_untouched() -> None:
    event = {"run_id": "abc123", "n_trades": 42, "sortino": 1.5}
    assert _redact(event) == event


# --- value-based redaction -------------------------------------------------
def test_redacts_token_shaped_values_under_innocent_keys() -> None:
    probe = "sk-" + ("a1b2" * 8)
    result = _redact({"message": f"calling with {probe}"})
    assert probe not in str(result["message"])
    assert REDACTED in str(result["message"])


@pytest.mark.parametrize("prefix", ["sk", "SK", "key", "Token"])
def test_redacts_each_credential_prefix(prefix: str) -> None:
    probe = f"{prefix}_" + ("9f" * 12)
    assert redact_value(probe) == REDACTED


def test_does_not_redact_short_or_ordinary_values() -> None:
    assert redact_value("key-123") == "key-123"
    assert redact_value("sortino=1.234567890123456789") == "sortino=1.234567890123456789"
    assert redact_value(42) == 42


def test_custom_redact_keys_are_honoured() -> None:
    event = {"passphrase": "abc", "api_key": "def"}
    result = _redact(event, ["passphrase"])
    assert result["passphrase"] == REDACTED
    assert result["api_key"] == "def"


def test_deeply_nested_structures_terminate() -> None:
    event: dict[str, object] = {"secret": "x"}
    for _ in range(30):
        event = {"nested": event}
    _redact(event)  # must not recurse without bound


# --- configuration ---------------------------------------------------------
def test_configure_logging_emits_redacted_json_to_the_stream() -> None:
    stream = io.StringIO()
    configure_logging(level="INFO", json_output=True, stream=stream)
    get_logger("t").info("llm_call", api_key="anything", tokens_in=10)

    payload = json.loads(stream.getvalue().strip().splitlines()[-1])
    assert payload["event"] == "llm_call"
    assert payload["api_key"] == REDACTED
    assert payload["tokens_in"] == 10
    assert payload["level"] == "info"
    assert "timestamp" in payload


def test_configure_logging_writes_a_dated_file(tmp_path: Path) -> None:
    stream = io.StringIO()
    day = dt.date(2026, 9, 5)
    configure_logging(log_dir=tmp_path, log_date=day, stream=stream)
    get_logger().warning("data_gap", bars=4)

    log_file = tmp_path / "quantlab-2026-09-05.jsonl"
    assert log_file.is_file()
    assert json.loads(log_file.read_text(encoding="utf-8").strip())["event"] == "data_gap"


def test_configure_logging_requires_a_date_with_a_directory(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="log_date"):
        configure_logging(log_dir=tmp_path)


def test_configure_logging_rejects_an_unknown_level() -> None:
    with pytest.raises(ConfigError, match="unknown logging level"):
        configure_logging(level="LOUD")


def test_level_filters_events() -> None:
    stream = io.StringIO()
    configure_logging(level="WARNING", stream=stream)
    logger = get_logger()
    logger.info("suppressed")
    logger.error("kept")
    assert "suppressed" not in stream.getvalue()
    assert "kept" in stream.getvalue()


def test_console_renderer_is_used_when_json_is_off() -> None:
    stream = io.StringIO()
    configure_logging(json_output=False, stream=stream)
    get_logger().info("plain_event", run_id="abcd")
    output = stream.getvalue()
    assert "plain_event" in output
    assert not output.strip().startswith("{")


# --- context binding -------------------------------------------------------
def test_bound_context_appears_on_every_event() -> None:
    stream = io.StringIO()
    configure_logging(stream=stream)
    bind_context(run_id="r0", strategy_id="s0")
    get_logger().info("fill")

    payload = json.loads(stream.getvalue().strip().splitlines()[-1])
    assert payload["run_id"] == "r0"
    assert payload["strategy_id"] == "s0"


def test_log_context_restores_previous_state() -> None:
    stream = io.StringIO()
    configure_logging(stream=stream)
    bind_context(run_id="outer")
    with log_context(run_id="inner"):
        get_logger().info("inside")
    get_logger().info("outside")

    lines = [json.loads(line) for line in stream.getvalue().strip().splitlines()]
    assert lines[-2]["run_id"] == "inner"
    assert lines[-1]["run_id"] == "outer"
