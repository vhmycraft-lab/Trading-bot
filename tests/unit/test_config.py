"""Configuration loading, overrides and hashing (master spec section 5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError as PydanticValidationError

from quantlab.core.config import (
    FORBIDDEN_OBJECTIVES,
    AppConfig,
    LoggingSettings,
    OptimizeSettings,
    apply_overrides,
    deep_merge,
    env_overrides,
    load_config,
    parse_override,
)
from quantlab.core.errors import ConfigError


# --- defaults --------------------------------------------------------------
def test_defaults_load_from_the_repository(config: AppConfig) -> None:
    assert config.project.name == "quantlab"
    assert config.market.symbol == "BTC/USDT"
    assert config.market.timeframe == "1h"
    assert config.backtest.initial_equity == 10_000.0
    assert config.backtest.fee_bps == 10.0
    assert config.backtest.slippage.model == "fixed_bps"
    assert config.backtest.slippage.fixed_bps == 5.0
    assert config.backtest.allow_short is False
    assert config.optimize.objective == "sortino_dd"
    assert config.optimize.n_trials == 200
    assert config.walkforward.is_bars == 13_140
    assert config.validation.dsr_threshold == 0.95
    assert config.validation.score.candidate_max == 30
    assert config.lockbox.max_per_family == 1
    assert config.sandbox.memory_mb == 2048
    assert config.logging.redact_keys == (
        "api_key",
        "secret",
        "token",
        "password",
        "authorization",
    )


def test_default_yaml_declares_every_section(default_config_path: Path) -> None:
    raw = yaml.safe_load(default_config_path.read_text(encoding="utf-8"))
    assert set(raw) == set(AppConfig.model_fields)


def test_config_is_frozen(config: AppConfig) -> None:
    with pytest.raises(PydanticValidationError):
        config.market.symbol = "ETH/USDT"  # type: ignore[misc]


# --- extra=forbid ----------------------------------------------------------
def test_unknown_top_level_key_is_rejected(tmp_path: Path, default_config_path: Path) -> None:
    extra = tmp_path / "extra.yaml"
    extra.write_text("nonsense:\n  a: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid configuration"):
        load_config([extra], default_path=default_config_path)


def test_unknown_nested_key_is_rejected(tmp_path: Path, default_config_path: Path) -> None:
    extra = tmp_path / "extra.yaml"
    extra.write_text("market:\n  symbl: BTC/USDT\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="symbl"):
        load_config([extra], default_path=default_config_path)


def test_out_of_range_value_is_rejected(tmp_path: Path, default_config_path: Path) -> None:
    extra = tmp_path / "extra.yaml"
    extra.write_text("market:\n  lot_step: 0\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config([extra], default_path=default_config_path)


def test_missing_file_is_reported_clearly() -> None:
    with pytest.raises(ConfigError, match="config file not found"):
        load_config(default_path="configs/does-not-exist.yaml")


def test_non_mapping_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping at the top level"):
        load_config(default_path=path)


def test_invalid_yaml_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("a: [1, 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(default_path=path)


def test_empty_file_falls_back_to_model_defaults(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_config(default_path=path).market.symbol == "BTC/USDT"


# --- forbidden objectives --------------------------------------------------
@pytest.mark.parametrize("objective", FORBIDDEN_OBJECTIVES)
def test_return_only_objectives_are_forbidden(objective: str) -> None:
    with pytest.raises(ConfigError, match="return-only objectives are forbidden"):
        OptimizeSettings(objective=objective)


def test_unknown_objective_is_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown optimisation objective"):
        OptimizeSettings(objective="vibes")


@pytest.mark.parametrize("objective", ["sortino_dd", "sharpe", "calmar", "expectancy"])
def test_allowed_objectives_are_accepted(objective: str) -> None:
    assert OptimizeSettings(objective=objective).objective == objective


def test_forbidden_objective_via_yaml(tmp_path: Path, default_config_path: Path) -> None:
    extra = tmp_path / "extra.yaml"
    extra.write_text("optimize:\n  objective: net_return\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="return-only objectives are forbidden"):
        load_config([extra], default_path=default_config_path)


def test_logging_level_is_normalised_and_validated() -> None:
    assert LoggingSettings(level="debug").level == "DEBUG"
    with pytest.raises(ConfigError, match="unknown logging level"):
        LoggingSettings(level="chatty")


def test_score_bands_must_be_ordered(tmp_path: Path, default_config_path: Path) -> None:
    extra = tmp_path / "extra.yaml"
    extra.write_text("validation:\n  score:\n    candidate_max: 70\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="candidate_max"):
        load_config([extra], default_path=default_config_path)


def test_cost_stress_multipliers_must_be_positive(
    tmp_path: Path, default_config_path: Path
) -> None:
    extra = tmp_path / "extra.yaml"
    extra.write_text("backtest:\n  cost_stress_multipliers: [1.0, 0.0]\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="cost_stress_multipliers"):
        load_config([extra], default_path=default_config_path)


# --- merging ---------------------------------------------------------------
def test_deep_merge_is_recursive_and_non_mutating() -> None:
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    overlay = {"a": {"y": 20, "z": 30}}
    merged = deep_merge(base, overlay)
    assert merged == {"a": {"x": 1, "y": 20, "z": 30}, "b": 3}
    assert base == {"a": {"x": 1, "y": 2}, "b": 3}


def test_later_config_files_win(tmp_path: Path, default_config_path: Path) -> None:
    first = tmp_path / "a.yaml"
    first.write_text("optimize:\n  n_trials: 10\n", encoding="utf-8")
    second = tmp_path / "b.yaml"
    second.write_text("optimize:\n  n_trials: 20\n", encoding="utf-8")

    config = load_config([first, second], default_path=default_config_path)
    assert config.optimize.n_trials == 20
    assert config.optimize.seed == 42, "unrelated defaults must survive the merge"


# --- --set overrides -------------------------------------------------------
@pytest.mark.parametrize(
    ("item", "expected"),
    [
        ("optimize.n_trials=50", (["optimize", "n_trials"], 50)),
        ("backtest.allow_short=true", (["backtest", "allow_short"], True)),
        ("backtest.fee_bps=12.5", (["backtest", "fee_bps"], 12.5)),
        ("market.symbol=ETH/USDT", (["market", "symbol"], "ETH/USDT")),
        (
            "backtest.cost_stress_multipliers=[1, 2]",
            (["backtest", "cost_stress_multipliers"], [1, 2]),
        ),
    ],
)
def test_parse_override(item: str, expected: tuple[list[str], object]) -> None:
    assert parse_override(item) == expected


@pytest.mark.parametrize("item", ["no-equals-sign", "=value", ".=1"])
def test_parse_override_rejects_malformed_input(item: str) -> None:
    with pytest.raises(ConfigError):
        parse_override(item)


def test_apply_overrides_creates_missing_branches() -> None:
    assert apply_overrides({}, ["a.b.c=1"]) == {"a": {"b": {"c": 1}}}


def test_overrides_beat_config_files(tmp_path: Path, default_config_path: Path) -> None:
    extra = tmp_path / "a.yaml"
    extra.write_text("optimize:\n  n_trials: 10\n", encoding="utf-8")
    config = load_config([extra], ["optimize.n_trials=99"], default_path=default_config_path)
    assert config.optimize.n_trials == 99


def test_override_of_a_forbidden_objective_still_raises(default_config_path: Path) -> None:
    with pytest.raises(ConfigError, match="return-only"):
        load_config([], ["optimize.objective=net_return"], default_path=default_config_path)


# --- environment overrides -------------------------------------------------
def test_env_overrides_scalar_values(
    monkeypatch: pytest.MonkeyPatch, default_config_path: Path
) -> None:
    monkeypatch.setenv("QUANTLAB__LOGGING__LEVEL", "DEBUG")
    monkeypatch.setenv("QUANTLAB__OPTIMIZE__N_TRIALS", "7")
    config = load_config(default_path=default_config_path)
    assert config.logging.level == "DEBUG"
    assert config.optimize.n_trials == 7
    assert config.optimize.seed == 42


def test_env_beats_config_files_and_set_overrides(
    monkeypatch: pytest.MonkeyPatch, default_config_path: Path
) -> None:
    monkeypatch.setenv("QUANTLAB__OPTIMIZE__N_TRIALS", "5")
    config = load_config([], ["optimize.n_trials=99"], default_path=default_config_path)
    assert config.optimize.n_trials == 5


def test_env_overrides_are_listed_without_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUANTLAB__LLM__BASE_URL", "https://example.invalid")
    listed = env_overrides()
    assert "QUANTLAB__LLM__BASE_URL" in listed
    assert listed["QUANTLAB__LLM__BASE_URL"] == "***"


# --- config_hash -----------------------------------------------------------
def test_config_hash_is_stable_under_key_reordering(
    tmp_path: Path, default_config_path: Path
) -> None:
    raw = yaml.safe_load(default_config_path.read_text(encoding="utf-8"))
    shuffled = dict(reversed(list(raw.items())))
    reordered = tmp_path / "reordered.yaml"
    reordered.write_text(yaml.safe_dump(shuffled, sort_keys=False), encoding="utf-8")

    assert (
        load_config(default_path=reordered).config_hash
        == load_config(default_path=default_config_path).config_hash
    )


def test_config_hash_changes_with_any_value(config: AppConfig, default_config_path: Path) -> None:
    changed = load_config([], ["backtest.fee_bps=11.0"], default_path=default_config_path)
    assert changed.config_hash != config.config_hash


def test_config_hash_is_a_full_sha256(config: AppConfig) -> None:
    assert len(config.config_hash) == 64
    assert set(config.config_hash) <= set("0123456789abcdef")


def test_resolved_json_round_trips(config: AppConfig) -> None:
    assert json.loads(config.resolved_json()) == config.resolved_dict()
    assert set(json.loads(config.resolved_json())) == set(AppConfig.model_fields)
