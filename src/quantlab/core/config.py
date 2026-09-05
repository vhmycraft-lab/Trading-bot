"""Typed configuration (master spec section 5).

All tunables live in YAML under ``configs/`` and are loaded into pydantic v2
models with ``extra="forbid"`` so a typo fails loudly instead of silently
falling back to a default.

Precedence, lowest to highest:

1. ``configs/default.yaml``
2. every ``--config`` file, in the order given
3. every ``--set section.key=value`` override, in the order given
4. environment variables ``QUANTLAB__<SECTION>__<KEY>``

Secrets are never read from here; see :mod:`quantlab.ports.secrets`.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from quantlab.core.errors import ConfigError
from quantlab.core.hashing import canonical_json, sha256_hex

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "ENV_PREFIX",
    "FORBIDDEN_OBJECTIVES",
    "VALID_OBJECTIVES",
    "AppConfig",
    "BacktestSettings",
    "LLMSettings",
    "LockboxSettings",
    "LoggingSettings",
    "MarketSettings",
    "OptimizeSettings",
    "PaperSettings",
    "PboSettings",
    "PermutationSettings",
    "PlateauSettings",
    "ProjectSettings",
    "ResearchSettings",
    "SandboxSettings",
    "ScoreSettings",
    "SlippageSettings",
    "SplitsSettings",
    "ValidationSettings",
    "WalkForwardSettings",
    "apply_overrides",
    "deep_merge",
    "load_config",
    "parse_override",
]

DEFAULT_CONFIG_PATH: Final[Path] = Path("configs/default.yaml")
ENV_PREFIX: Final[str] = "QUANTLAB__"

#: Objectives that may be optimised (spec section 13.1).
VALID_OBJECTIVES: Final[tuple[str, ...]] = ("sortino_dd", "sharpe", "calmar", "expectancy")

#: Return-only objectives.  Optimising these invites curve fitting, so they are banned.
FORBIDDEN_OBJECTIVES: Final[tuple[str, ...]] = (
    "net_return",
    "cagr",
    "total_return",
    "return",
    "pnl",
)


class _Section(BaseModel):
    """Base for every config section: frozen and intolerant of unknown keys."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, validate_assignment=True, populate_by_name=True
    )


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------
class ProjectSettings(_Section):
    name: str = "quantlab"
    data_dir: Path = Path("./data")
    artifacts_dir: Path = Path("./artifacts")
    db_path: Path = Path("./quantlab.db")
    timezone: str = "UTC"  # informational; all internals are UTC milliseconds


class MarketSettings(_Section):
    exchange: str = "binance"
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    lot_step: float = Field(default=0.00001, gt=0)
    min_notional: float = Field(default=5.0, ge=0)


class SlippageSettings(_Section):
    model: Literal["fixed_bps", "volatility_scaled", "volume_impact"] = "fixed_bps"
    fixed_bps: float = Field(default=5.0, ge=0)
    vol_k: float = Field(default=0.10, ge=0)
    impact_a_bps: float = Field(default=2.0, ge=0)
    impact_b: float = Field(default=50.0, ge=0)


class BacktestSettings(_Section):
    initial_equity: float = Field(default=10_000.0, gt=0)
    fill_rule: Literal["next_open"] = "next_open"
    allow_short: bool = False
    short_borrow_bps_per_bar: float = Field(default=0.0, ge=0)
    fee_bps: float = Field(default=10.0, ge=0)
    slippage: SlippageSettings = SlippageSettings()
    cost_stress_multipliers: tuple[float, ...] = (1.0, 2.0, 3.0)
    max_position_fraction: float = Field(default=1.0, gt=0)

    @field_validator("cost_stress_multipliers")
    @classmethod
    def _positive_multipliers(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value:
            raise ValueError("cost_stress_multipliers must not be empty")
        if any(m <= 0 for m in value):
            raise ValueError("cost_stress_multipliers must all be > 0")
        return value


class SplitsSettings(_Section):
    policy_file: Path = Path("configs/splits/btcusdt_1h.yaml")


class PlateauSettings(_Section):
    top_k: int = Field(default=10, ge=1)
    perturbation_pcts: tuple[float, ...] = (0.10, 0.25)
    n_neighbors: int = Field(default=12, ge=1)


class OptimizeSettings(_Section):
    sampler: Literal["tpe", "random", "grid"] = "tpe"
    n_trials: int = Field(default=200, ge=1)
    timeout_s: int = Field(default=1800, ge=1)
    seed: int = 42
    objective: str = "sortino_dd"
    min_trades: int = Field(default=30, ge=0)
    dd_lambda: float = Field(default=0.5, ge=0)
    plateau: PlateauSettings = PlateauSettings()

    @field_validator("objective")
    @classmethod
    def _objective_allowed(cls, value: str) -> str:
        if value in FORBIDDEN_OBJECTIVES:
            raise ConfigError(
                "return-only objectives are forbidden",
                objective=value,
                allowed=list(VALID_OBJECTIVES),
            )
        if value not in VALID_OBJECTIVES:
            raise ConfigError(
                "unknown optimisation objective",
                objective=value,
                allowed=list(VALID_OBJECTIVES),
            )
        return value


class WalkForwardSettings(_Section):
    scheme: Literal["rolling", "anchored"] = "rolling"
    is_bars: int = Field(default=13_140, ge=1)
    oos_bars: int = Field(default=2_190, ge=1)
    step_bars: int = Field(default=2_190, ge=1)
    reoptimize_each_window: bool = True


class PermutationSettings(_Section):
    n_market_permutations: int = Field(default=200, ge=1)
    n_trade_shuffles: int = Field(default=1000, ge=1)
    block_len_bars: int = Field(default=168, ge=1)
    alpha: float = Field(default=0.05, gt=0, lt=1)


class PboSettings(_Section):
    n_blocks: int = Field(default=16, ge=2)
    max_combinations: int = Field(default=500, ge=1)


class ScoreSettings(_Section):
    candidate_max: int = Field(default=30, ge=0)
    weak_max: int = Field(default=60, ge=0)


class ValidationSettings(_Section):
    min_trades_val: int = Field(default=30, ge=0)
    min_trades_train: int = Field(default=100, ge=0)
    max_single_trade_pct: float = Field(default=0.25, gt=0)
    cost_survival_multiplier: float = Field(default=2.0, ge=1)
    permutation: PermutationSettings = PermutationSettings()
    degradation_min_ratio: float = Field(default=0.5, ge=0)
    dsr_threshold: float = Field(default=0.95, gt=0, lt=1)
    pbo: PboSettings = PboSettings()
    sensitivity_max_drop: float = Field(default=0.5, ge=0)
    concentration_top_n: int = Field(default=5, ge=1)
    concentration_max_share: float = Field(default=0.5, gt=0, le=1)
    max_free_params: int = Field(default=6, ge=1)
    max_logic_lines: int = Field(default=150, ge=1)
    random_entry_seeds: int = Field(default=100, ge=1)
    score: ScoreSettings = ScoreSettings()
    family_max_validation_touches: int = Field(default=20, ge=1)

    @field_validator("score")
    @classmethod
    def _ordered_score_bands(cls, value: ScoreSettings) -> ScoreSettings:
        if value.candidate_max > value.weak_max:
            raise ValueError("score.candidate_max must be <= score.weak_max")
        return value


class LockboxSettings(_Section):
    max_per_family: int = Field(default=1, ge=1)
    max_per_month: int = Field(default=3, ge=1)


class LLMSettings(_Section):
    provider: Literal["glm", "mock"] = "glm"
    model: str = "glm-4-plus"
    base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    temperature_propose: float = Field(default=0.7, ge=0, le=2)
    temperature_repair: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=8000, ge=1)
    timeout_s: int = Field(default=60, ge=1)
    max_retries: int = Field(default=3, ge=0)
    daily_cost_cap_eur: float = Field(default=10.0, ge=0)
    pricing_file: Path = Path("configs/llm_pricing.yaml")


class ResearchSettings(_Section):
    max_iterations: int = Field(default=25, ge=1)
    max_cost_eur: float = Field(default=20.0, ge=0)
    max_wall_clock_s: int = Field(default=14_400, ge=1)
    max_repair_attempts: int = Field(default=2, ge=0)
    smoke_bars: int = Field(default=200, ge=1)


class SandboxSettings(_Section):
    cpu_seconds: int = Field(default=120, ge=1)
    memory_mb: int = Field(default=2048, ge=1)
    wall_clock_s: int = Field(default=300, ge=1)
    allowed_imports: tuple[str, ...] = (
        "numpy",
        "pandas",
        "math",
        "statistics",
        "dataclasses",
        "typing",
        "quantlab.core.strategy",
        "quantlab.core.indicators",
        "quantlab.core.types",
    )


class PaperSettings(_Section):
    state_dir: Path = Path("./artifacts/paper")
    ws_url: str = "wss://stream.binance.com:9443/ws"
    backfill_limit: int = Field(default=1000, ge=1)
    alert_mdd_percentile: int = Field(default=95, ge=1, le=100)


class LoggingSettings(_Section):
    level: str = "INFO"
    #: YAML key is ``json``; the Python attribute is renamed because ``json``
    #: shadows a (deprecated) ``BaseModel`` method.
    json_output: bool = Field(default=True, alias="json")
    redact_keys: tuple[str, ...] = ("api_key", "secret", "token", "password", "authorization")

    @field_validator("level")
    @classmethod
    def _known_level(cls, value: str) -> str:
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
        upper = value.upper()
        if upper not in allowed:
            raise ConfigError("unknown logging level", level=value, allowed=sorted(allowed))
        return upper


# ---------------------------------------------------------------------------
# root
# ---------------------------------------------------------------------------
class AppConfig(BaseSettings):
    """The fully-resolved application configuration.

    Immutable, hashable via :attr:`config_hash`, and reproducible: the exact
    JSON that produced the hash is stored with every experiment.
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
        case_sensitive=False,
        populate_by_name=True,
    )

    project: ProjectSettings = ProjectSettings()
    market: MarketSettings = MarketSettings()
    backtest: BacktestSettings = BacktestSettings()
    splits: SplitsSettings = SplitsSettings()
    optimize: OptimizeSettings = OptimizeSettings()
    walkforward: WalkForwardSettings = WalkForwardSettings()
    validation: ValidationSettings = ValidationSettings()
    lockbox: LockboxSettings = LockboxSettings()
    llm: LLMSettings = LLMSettings()
    research: ResearchSettings = ResearchSettings()
    sandbox: SandboxSettings = SandboxSettings()
    paper: PaperSettings = PaperSettings()
    logging: LoggingSettings = LoggingSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],  # noqa: ARG003 - signature fixed by pydantic-settings
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,  # noqa: ARG003 - unused source
        file_secret_settings: PydanticBaseSettingsSource,  # noqa: ARG003 - unused source
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Environment variables win over the merged YAML passed as init kwargs."""
        return (env_settings, init_settings)

    # -- identity ----------------------------------------------------------
    def resolved_dict(self) -> dict[str, Any]:
        """JSON-compatible dump of the fully-resolved configuration."""
        return self.model_dump(mode="json", by_alias=True)

    def resolved_json(self) -> str:
        """Canonical JSON of :meth:`resolved_dict` — what the hash is taken over."""
        return canonical_json(self.resolved_dict())

    @property
    def config_hash(self) -> str:
        """SHA-256 (hex) of the canonical JSON of the resolved config."""
        return sha256_hex(self.resolved_json())


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` onto ``base`` without mutating either."""
    merged: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def parse_override(item: str) -> tuple[list[str], Any]:
    """Parse one ``--set section.key=value`` string into a key path and a value.

    The value is parsed as YAML, so ``true``, ``12``, ``0.5`` and ``[1, 2]`` all
    arrive with their natural type; anything else stays a string.
    """
    key, sep, raw = item.partition("=")
    if not sep or not key.strip():
        raise ConfigError("override must look like section.key=value", override=item)
    path = [part for part in key.strip().split(".") if part]
    if not path:
        raise ConfigError("override key is empty", override=item)
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as exc:  # pragma: no cover - yaml scalars rarely fail
        raise ConfigError("could not parse override value", override=item) from exc
    return path, value


def apply_overrides(data: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply ``--set`` overrides to a config dict, returning a new dict."""
    result: dict[str, Any] = dict(data)
    for item in overrides:
        path, value = parse_override(item)
        cursor = result
        for part in path[:-1]:
            existing = cursor.get(part)
            branch = dict(existing) if isinstance(existing, Mapping) else {}
            cursor[part] = branch
            cursor = branch
        cursor[path[-1]] = value
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError("config file not found", path=str(path))
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError("config file is not valid YAML", path=str(path)) from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError("config file must contain a mapping at the top level", path=str(path))
    return loaded


def env_overrides(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the ``QUANTLAB__*`` variables that will override the YAML, for logging."""
    source = os.environ if env is None else env
    return {key: "***" for key in sorted(source) if key.upper().startswith(ENV_PREFIX)}


def load_config(
    config_paths: Sequence[str | Path] | None = None,
    overrides: Sequence[str] | None = None,
    *,
    default_path: str | Path = DEFAULT_CONFIG_PATH,
) -> AppConfig:
    """Load, merge and validate the configuration.

    Args:
        config_paths: Extra YAML files merged over the defaults, in order.
        overrides: ``section.key=value`` strings applied after the files.
        default_path: The base configuration file.

    Raises:
        ConfigError: on a missing file, invalid YAML, an unknown key, a failed
            constraint, or a forbidden optimisation objective.
    """
    data = _read_yaml(Path(default_path))
    for extra in config_paths or ():
        data = deep_merge(data, _read_yaml(Path(extra)))
    if overrides:
        data = apply_overrides(data, overrides)

    try:
        return AppConfig(**data)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration:\n{exc}") from exc
