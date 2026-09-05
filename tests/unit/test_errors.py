"""Exception hierarchy and exit codes (master spec section 18)."""

from __future__ import annotations

import pytest

from quantlab.core import errors
from quantlab.core.errors import (
    EXIT_CODES,
    ConfigError,
    DataError,
    DataValidationError,
    EngineError,
    LeakageDetected,
    LLMError,
    LockboxViolation,
    QuantLabError,
    SandboxError,
    StrategyError,
    StrategySafetyError,
    exit_code_for,
)


@pytest.mark.parametrize(
    ("family", "expected"),
    [
        (ConfigError, 2),
        (DataError, 3),
        (StrategyError, 4),
        (SandboxError, 5),
        (LLMError, 6),
        (LockboxViolation, 7),
        (EngineError, 8),
    ],
)
def test_exit_code_per_family(family: type[QuantLabError], expected: int) -> None:
    assert exit_code_for(family("boom")) == expected
    assert EXIT_CODES[family.__name__] == expected


def test_subclasses_inherit_their_family_exit_code() -> None:
    assert exit_code_for(DataValidationError("bad bars")) == 3
    assert exit_code_for(LeakageDetected("not causal")) == 4


def test_unknown_exception_maps_to_one() -> None:
    assert exit_code_for(RuntimeError("boom")) == 1
    assert EXIT_CODES["other"] == 1


def test_context_is_rendered_but_the_message_stays_first() -> None:
    exc = ConfigError("missing secret", secret="GLM_API_KEY")
    assert exc.message == "missing secret"
    assert exc.context == {"secret": "GLM_API_KEY"}
    assert str(exc).startswith("missing secret (")
    assert "GLM_API_KEY" in str(exc)


def test_message_without_context_is_unchanged() -> None:
    assert str(ConfigError("plain")) == "plain"


def test_safety_error_carries_violation_codes() -> None:
    exc = StrategySafetyError("rejected", violations=["FORBIDDEN_IMPORT", "BARE_EXEC"])
    assert exc.violations == ["FORBIDDEN_IMPORT", "BARE_EXEC"]
    assert StrategySafetyError("rejected").violations == []


def test_leakage_carries_the_probe_result() -> None:
    probe = {"cut": 42}
    assert LeakageDetected("leak", probe=probe).probe is probe


def test_every_public_error_derives_from_the_base() -> None:
    for name in errors.__all__:
        obj = getattr(errors, name)
        if isinstance(obj, type) and issubclass(obj, BaseException):
            assert issubclass(obj, QuantLabError), name


def test_lockbox_violation_is_not_a_data_or_strategy_error() -> None:
    """It must not be swallowed by a broad ``except DataError`` anywhere (INV-5)."""
    exc = LockboxViolation("test partition requested")
    assert not isinstance(exc, (DataError, StrategyError))
