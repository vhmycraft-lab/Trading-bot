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

import itertools
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from quantlab.core.errors import ConfigError
from quantlab.core.hashing import canonical_json, sha256_hex

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "ENV_PREFIX",
    "FORBIDDEN_OBJECTIVES",
    "REMOVED_KEYS",
    "VALID_OBJECTIVES",
    "AppConfig",
    "BacktestSettings",
    "DiversitySettings",
    "EvolutionSettings",
    "FitnessGates",
    "FitnessPenalties",
    "FitnessSettings",
    "FitnessTargets",
    "FitnessWeights",
    "GenomeSettings",
    "InnerWalkForwardSettings",
    "LLMSettings",
    "LockboxSettings",
    "LoggingSettings",
    "MarketSettings",
    "MutationParameterSettings",
    "MutationSettings",
    "MutationStructuralSettings",
    "OptimizeSettings",
    "PaperSettings",
    "PboSettings",
    "PermutationSettings",
    "PlateauSettings",
    "ProjectSettings",
    "PromotionSettings",
    "ResearchSettings",
    "SandboxSettings",
    "ScoreSettings",
    "SlippageSettings",
    "SplitsSettings",
    "ValidationSettings",
    "WalkForwardSettings",
    "apply_overrides",
    "check_removed_keys",
    "deep_merge",
    "load_config",
    "parse_override",
]

DEFAULT_CONFIG_PATH: Final[Path] = Path("configs/default.yaml")

#: Keys removed by a specification change, with the migration to follow.
#: ``extra="forbid"`` would otherwise reject them with a generic "extra inputs
#: are not permitted", which tells a reader nothing about what to do instead.
REMOVED_KEYS: Final[Mapping[str, str]] = {
    "research.max_iterations": (
        "removed in spec 1.1: the sequential research loop is superseded by the "
        "evolutionary optimiser. Use evolution.max_generations."
    ),
}
ENV_PREFIX: Final[str] = "QUANTLAB__"

#: Objectives that may be optimised (spec section 13.1).
VALID_OBJECTIVES: Final[tuple[str, ...]] = ("sortino_dd", "sharpe", "calmar", "expectancy")

#: Slack allowed when checking that a set of weights sums to 1.0.
_WEIGHT_TOLERANCE: Final[float] = 1e-9

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
    #: Which search produces candidates.  ``evolution`` is the primary search
    #: (spec section 13); ``optuna`` refines a promoted candidate (spec 13.8).
    engine: Literal["evolution", "optuna"] = "evolution"
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
    #: Generations per in-sample window when ``optimize.engine == "evolution"``.
    evolution_generations: int = Field(default=8, ge=1)


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
    #: Slack on the position cap in ``G_SANITY`` (spec section 14.3).
    #:
    #: ``max_position_fraction`` bounds the *target* fraction at the deciding
    #: bar; the realised fraction then drifts with the market until the next
    #: rebalance, so a near-zero tolerance would fail correctly-behaved
    #: strategies.  The exact invariant — committed capital at a fill never
    #: exceeds ``max_position_fraction x equity`` at the deciding bar — is the
    #: engine's, and is checked by its property tests.
    sanity_drift_allowance: float = Field(default=0.5, ge=0)
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


class EnvironmentSettings(_Section):
    """Per-generation hidden training environments (Project Rome sections 4-19).

    Every band here is a *documented realistic range*, which Rome section 9
    requires and which is the difference between exposing a strategy to varied
    conditions and manufacturing difficulty. Two quantities are deliberately
    absent and must stay absent: **commission** (section 12) and **execution
    timing** (section 13) are structural properties of the venue, not sources of
    uncertainty, and randomising them would reject strategies for conditions no
    venue produces.

    Window bounds are fractions of the training segment so that one configuration
    is meaningful across splits of very different lengths; ``min_window_bars`` is
    the absolute floor underneath them.
    """

    enabled: bool = True
    #: Shortest window, as a fraction of the training segment.
    window_min_fraction: float = Field(default=0.35, gt=0, le=1)
    #: Longest window, as a fraction of the training segment.
    window_max_fraction: float = Field(default=0.75, gt=0, le=1)
    #: Spacing between the window lengths in the pool.
    length_step_fraction: float = Field(default=0.05, gt=0, le=1)
    #: Spacing between window start times.
    stride_fraction: float = Field(default=0.02, gt=0, le=1)
    #: Rome section 8: never sample a window too short to mean anything.
    min_window_bars: int = Field(default=200, ge=2)

    #: Realistic account sizes for this research programme, in the quote currency.
    min_starting_capital: float = Field(default=5_000.0, gt=0)
    max_starting_capital: float = Field(default=50_000.0, gt=0)

    #: Realistic slippage band for the venue (Rome section 11). Bounded, and
    #: bounded *low*: the point is to test dependence on one exact execution
    #: assumption, not to price in conditions the venue does not exhibit.
    min_slippage_bps: float = Field(default=2.0, ge=0)
    max_slippage_bps: float = Field(default=8.0, ge=0)

    #: Assets an environment may expose. Empty means "the split's own symbol",
    #: which is the honest default for a single-symbol split: fabricating a
    #: universe would violate Rome section 15's requirement that assets actually
    #: existed in the selected period.
    asset_universe: tuple[str, ...] = ()
    min_assets: int = Field(default=1, ge=1)

    #: Exposure tracking (Rome sections 18, 43).
    n_history_buckets: int = Field(default=20, ge=2)
    #: A bucket is reported over-used above this multiple of its uniform share.
    overuse_factor: float = Field(default=1.75, gt=1)

    @model_validator(mode="after")
    def _check_bands(self) -> EnvironmentSettings:
        if self.window_max_fraction < self.window_min_fraction:
            raise ValueError("window_max_fraction must be at least window_min_fraction")
        if self.max_starting_capital < self.min_starting_capital:
            raise ValueError("max_starting_capital must be at least min_starting_capital")
        if self.max_slippage_bps < self.min_slippage_bps:
            raise ValueError("max_slippage_bps must be at least min_slippage_bps")
        return self


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
    """What the LLM contributes to the evolutionary search (spec section 12)."""

    #: LLM-proposed genomes used to fill generation 0.
    max_seed_genomes: int = Field(default=8, ge=0)
    #: Share of offspring whose mutation operator the LLM may suggest.
    guided_mutation_share: float = Field(default=0.25, ge=0, le=1)
    max_cost_eur: float = Field(default=20.0, ge=0)
    max_wall_clock_s: int = Field(default=14_400, ge=1)
    max_repair_attempts: int = Field(default=2, ge=0)
    smoke_bars: int = Field(default=200, ge=1)


# ---------------------------------------------------------------------------
# evolutionary optimisation (spec section 13)
#
# Schema only.  Nothing reads these values yet; the optimiser lands in phase F'
# (T45-T54).  They are declared now so the shipped configuration matches
# specification v1.1, and — more usefully — so that a configuration that could
# not possibly work is rejected before any code depends on it.  Every validator
# below encodes a rule from the specification, not a preference.
# ---------------------------------------------------------------------------
class MutationParameterSettings(_Section):
    """How a numeric parameter moves under mutation (spec section 13.4)."""

    perturb_pct: float = Field(default=0.25, gt=0, le=1)
    jump_probability: float = Field(default=0.15, ge=0, le=1)
    #: Risk parameters move further than ordinary ones; they are coarser controls.
    risk_perturb_pct: float = Field(default=0.35, gt=0, le=1)


class MutationStructuralSettings(_Section):
    """Relative weights of the structural operators of spec section 13.4.

    One field per operator in that table.  The weights are relative and are
    normalised by :meth:`normalised`, so they need not sum to anything in
    particular — but they must not all be zero, which would leave the structural
    mutation path unable to choose an operator at all.
    """

    add_confirmation: float = Field(default=0.22, ge=0)
    remove_confirmation: float = Field(default=0.18, ge=0)
    modify_entry: float = Field(default=0.18, ge=0)
    modify_exit: float = Field(default=0.13, ge=0)
    add_filter: float = Field(default=0.09, ge=0)
    remove_filter: float = Field(default=0.09, ge=0)
    replace_indicator: float = Field(default=0.07, ge=0)
    change_tree_mode: float = Field(default=0.04, ge=0)

    @model_validator(mode="after")
    def _some_operator_is_reachable(self) -> MutationStructuralSettings:
        if self.total <= 0:
            raise ConfigError(
                "at least one structural mutation operator must have a positive weight",
                operators=sorted(type(self).model_fields),
            )
        return self

    @property
    def total(self) -> float:
        """Sum of the raw weights."""
        return float(sum(getattr(self, name) for name in type(self).model_fields))

    def normalised(self) -> dict[str, float]:
        """Return the operator weights as a probability distribution."""
        total = self.total
        return {name: getattr(self, name) / total for name in type(self).model_fields}


class MutationSettings(_Section):
    parameter_rate: float = Field(default=0.7, ge=0, le=1)
    structural_rate: float = Field(default=0.3, ge=0, le=1)
    max_mutations_per_child: int = Field(default=3, ge=1)
    max_repair_attempts: int = Field(default=3, ge=0)
    parameter: MutationParameterSettings = MutationParameterSettings()
    structural: MutationStructuralSettings = MutationStructuralSettings()

    @model_validator(mode="after")
    def _a_child_can_be_mutated(self) -> MutationSettings:
        if self.parameter_rate <= 0 and self.structural_rate <= 0:
            raise ConfigError(
                "parameter_rate and structural_rate are both zero, so no child could "
                "ever differ from its parent"
            )
        return self


class DiversitySettings(_Section):
    """Keeping the population from collapsing onto one strategy (spec section 13.5)."""

    max_pairwise_similarity: float = Field(default=0.90, gt=0, le=1)
    min_population_diversity: float = Field(default=0.35, ge=0, lt=1)
    structural_weight: float = Field(default=0.4, ge=0, le=1)
    behavioural_weight: float = Field(default=0.6, ge=0, le=1)
    immigrant_boost: int = Field(default=2, ge=0)
    max_immigrants: int = Field(default=4, ge=1)

    @model_validator(mode="after")
    def _similarity_weights_sum_to_one(self) -> DiversitySettings:
        total = self.structural_weight + self.behavioural_weight
        if abs(total - 1.0) > _WEIGHT_TOLERANCE:
            raise ConfigError(
                "structural_weight + behavioural_weight must sum to 1.0",
                structural_weight=self.structural_weight,
                behavioural_weight=self.behavioural_weight,
                total=total,
            )
        return self


class FitnessWeights(_Section):
    """Weights of the ten fitness components (spec section 13.3).

    ``net_return`` and ``win_rate`` are deliberately the smallest: profit is not
    the objective, and win rate is nearly meaningless on its own.  Neither can
    rescue a candidate anyway — expectancy and drawdown are hard gates that run
    before this score exists.
    """

    expectancy: float = Field(default=0.20, ge=0, le=1)
    profit_factor: float = Field(default=0.12, ge=0, le=1)
    risk_adjusted: float = Field(default=0.15, ge=0, le=1)
    drawdown: float = Field(default=0.15, ge=0, le=1)
    consistency: float = Field(default=0.12, ge=0, le=1)
    inner_oos: float = Field(default=0.12, ge=0, le=1)
    trades: float = Field(default=0.06, ge=0, le=1)
    win_rate: float = Field(default=0.04, ge=0, le=1)
    net_return: float = Field(default=0.02, ge=0, le=1)
    concentration: float = Field(default=0.02, ge=0, le=1)

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> FitnessWeights:
        if abs(self.total - 1.0) > _WEIGHT_TOLERANCE:
            raise ConfigError(
                "fitness weights must sum to 1.0 so that scores stay comparable "
                "across configurations",
                total=self.total,
                weights=self.as_dict(),
            )
        return self

    @property
    def total(self) -> float:
        return float(sum(getattr(self, name) for name in type(self).model_fields))

    def as_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in type(self).model_fields}


class FitnessTargets(_Section):
    """Values at which a fitness component reaches 1.0 (spec section 13.3).

    Several of these appear in denominators, so the bounds here are what keeps
    the fitness function total rather than merely usually-defined.
    """

    expectancy_pct: float = Field(default=0.002, gt=0)
    #: Must exceed 1: the component is ``(profit_factor - 1) / (target - 1)``.
    profit_factor: float = Field(default=1.5, gt=1)
    sortino: float = Field(default=1.5, gt=0)
    cagr: float = Field(default=0.20, gt=0)
    trades: int = Field(default=200, ge=1)
    win_rate: float = Field(default=0.55, gt=0, le=1)
    win_rate_floor: float = Field(default=0.35, ge=0, lt=1)
    drawdown_ceiling: float = Field(default=0.35, gt=0, le=1)
    consistency_period_bars: int = Field(default=720, ge=1)

    @model_validator(mode="after")
    def _win_rate_range_is_non_degenerate(self) -> FitnessTargets:
        if self.win_rate <= self.win_rate_floor:
            raise ConfigError(
                "targets.win_rate must exceed targets.win_rate_floor; the component "
                "divides by their difference",
                win_rate=self.win_rate,
                win_rate_floor=self.win_rate_floor,
            )
        return self


class FitnessGates(_Section):
    """Absolute rejections (spec section 13.3, stage 1).

    These run before the weighted score exists, which is how "win rate must never
    override negative expectancy or excessive drawdown" is expressed as control
    flow rather than as a hopeful weighting.
    """

    min_expectancy_pct: float = Field(default=0.0, ge=0)
    min_trades: int = Field(default=30, ge=0)
    max_drawdown: float = Field(default=0.50, gt=0, le=1)
    #: Retention after removing the single best trade; below this, reject.
    min_retention_top1: float = Field(default=0.0, le=1)


class FitnessPenalties(_Section):
    """Multiplicative robustness penalties (spec section 13.3, stage 3)."""

    drawdown_soft: float = Field(default=0.20, gt=0, le=1)
    trades_soft: int = Field(default=100, ge=0)
    instability_max_cv: float = Field(default=0.60, gt=0)
    sensitivity_max_drop: float = Field(default=0.50, gt=0, le=1)
    divergence_min_ratio: float = Field(default=0.50, ge=0, le=1)
    complexity_free_params: int = Field(default=6, ge=1)
    complexity_logic_lines: int = Field(default=150, ge=1)
    #: Trade-removal test (spec section 14.4): remove the top k winners.
    removal_k: tuple[int, ...] = (1, 3, 5)
    removal_floor: tuple[float, ...] = (0.40, 0.20, 0.10)
    removal_target: tuple[float, ...] = (0.80, 0.65, 0.55)

    @model_validator(mode="after")
    def _removal_curve_is_coherent(self) -> FitnessPenalties:
        if not self.removal_k:
            raise ConfigError("penalties.removal_k must not be empty")
        if not (len(self.removal_k) == len(self.removal_floor) == len(self.removal_target)):
            raise ConfigError(
                "penalties.removal_k, removal_floor and removal_target must be the same length",
                removal_k=list(self.removal_k),
                removal_floor=list(self.removal_floor),
                removal_target=list(self.removal_target),
            )
        if any(k < 1 for k in self.removal_k):
            raise ConfigError(
                "penalties.removal_k values must be >= 1", removal_k=list(self.removal_k)
            )
        if any(later <= earlier for earlier, later in itertools.pairwise(self.removal_k)):
            raise ConfigError(
                "penalties.removal_k must be strictly increasing", removal_k=list(self.removal_k)
            )
        for name, values in (
            ("removal_floor", self.removal_floor),
            ("removal_target", self.removal_target),
        ):
            if any(not 0.0 <= value <= 1.0 for value in values):
                raise ConfigError(
                    f"penalties.{name} values must lie in [0, 1]", values=list(values)
                )
            # Retention is non-increasing in k, so its thresholds must be too;
            # a floor that rose with k could never be satisfied.
            if any(later > earlier for earlier, later in itertools.pairwise(values)):
                raise ConfigError(
                    f"penalties.{name} must be non-increasing in k, because retention "
                    "after removing more trades can only fall",
                    values=list(values),
                )
        for k, floor, target in zip(
            self.removal_k, self.removal_floor, self.removal_target, strict=True
        ):
            if floor >= target:
                raise ConfigError(
                    "penalties.removal_floor must be below removal_target for every k",
                    k=k,
                    floor=floor,
                    target=target,
                )
        return self


class FitnessSettings(_Section):
    weights: FitnessWeights = FitnessWeights()
    targets: FitnessTargets = FitnessTargets()
    gates: FitnessGates = FitnessGates()
    penalties: FitnessPenalties = FitnessPenalties()

    @model_validator(mode="after")
    def _soft_thresholds_sit_inside_the_hard_ones(self) -> FitnessSettings:
        if self.penalties.drawdown_soft >= self.gates.max_drawdown:
            raise ConfigError(
                "penalties.drawdown_soft must be below gates.max_drawdown; the penalty "
                "ramps between them",
                drawdown_soft=self.penalties.drawdown_soft,
                max_drawdown=self.gates.max_drawdown,
            )
        if self.penalties.trades_soft < self.gates.min_trades:
            raise ConfigError(
                "penalties.trades_soft must be at or above gates.min_trades; below the "
                "gate a candidate is rejected outright and the penalty is unreachable",
                trades_soft=self.penalties.trades_soft,
                min_trades=self.gates.min_trades,
            )
        if self.targets.drawdown_ceiling > self.gates.max_drawdown:
            raise ConfigError(
                "targets.drawdown_ceiling must not exceed gates.max_drawdown; the "
                "drawdown component would still be positive for a rejected candidate",
                drawdown_ceiling=self.targets.drawdown_ceiling,
                max_drawdown=self.gates.max_drawdown,
            )
        return self


class InnerWalkForwardSettings(_Section):
    """Out-of-sample folds *inside* train (spec section 13.6).

    This is how fitness rewards generalisation without spending the validation
    segment once per candidate per generation.
    """

    n_folds: int = Field(default=4, ge=2)
    scheme: Literal["rolling", "anchored"] = "rolling"
    embargo_bars: int = Field(default=24, ge=0)


class PromotionSettings(_Section):
    """The only route from evolution to the validation segment (spec section 13.7)."""

    #: 0 means "only after the final generation".
    every_generations: int = Field(default=0, ge=0)
    n_promote: int = Field(default=3, ge=1)
    min_fitness: float = Field(default=0.35, ge=0, le=1)
    max_similarity_between_promoted: float = Field(default=0.80, gt=0, le=1)
    counts_as_validation_touch: bool = True


class GenomeSettings(_Section):
    """Structural limits on a candidate genome (spec section 9.6)."""

    max_conditions_entry: int = Field(default=4, ge=1)
    max_conditions_exit: int = Field(default=4, ge=1)
    max_filters: int = Field(default=3, ge=0)
    max_indicators: int = Field(default=6, ge=1)
    max_free_params: int = Field(default=6, ge=1)


class EvolutionSettings(_Section):
    """Population-based search over strategy candidates (spec section 13)."""

    enabled: bool = True
    population_size: int = Field(default=16, ge=2)
    n_survivors: int = Field(default=12, ge=1)
    n_offspring: int = Field(default=3, ge=0)
    #: At least one, always: spec section 13.5 reserves a slot per generation for a
    #: candidate that owes nothing to the current leader.  With the sum identity
    #: below, this also guarantees ``n_survivors < population_size``.
    n_immigrants: int = Field(default=1, ge=1)
    max_generations: int = Field(default=30, ge=1)
    seed: int = 42
    max_wall_clock_s: int = Field(default=21_600, ge=1)
    max_evaluations: int = Field(default=1_000, ge=1)
    stop_on_no_improvement_generations: int = Field(default=8, ge=1)
    min_improvement: float = Field(default=0.005, ge=0)
    mutation: MutationSettings = MutationSettings()
    diversity: DiversitySettings = DiversitySettings()
    fitness: FitnessSettings = FitnessSettings()
    inner_walkforward: InnerWalkForwardSettings = InnerWalkForwardSettings()
    promotion: PromotionSettings = PromotionSettings()
    genome: GenomeSettings = GenomeSettings()

    @property
    def open_slots(self) -> int:
        """Slots per generation not held by a survivor."""
        return self.n_offspring + self.n_immigrants

    @model_validator(mode="after")
    def _population_arithmetic(self) -> EvolutionSettings:
        total = self.n_survivors + self.n_offspring + self.n_immigrants
        if total != self.population_size:
            raise ConfigError(
                "n_survivors + n_offspring + n_immigrants must equal population_size; "
                "silently resizing the population would make every downstream trial "
                "count, and therefore every deflated Sharpe ratio, wrong",
                population_size=self.population_size,
                n_survivors=self.n_survivors,
                n_offspring=self.n_offspring,
                n_immigrants=self.n_immigrants,
                total=total,
            )
        if self.diversity.max_immigrants > self.open_slots:
            raise ConfigError(
                "diversity.max_immigrants must not exceed n_offspring + n_immigrants; "
                "an immigrant boost displaces offspring and never a survivor",
                max_immigrants=self.diversity.max_immigrants,
                open_slots=self.open_slots,
            )
        if self.diversity.max_immigrants < self.n_immigrants:
            raise ConfigError(
                "diversity.max_immigrants must be at least n_immigrants",
                max_immigrants=self.diversity.max_immigrants,
                n_immigrants=self.n_immigrants,
            )
        if self.max_evaluations < self.population_size:
            raise ConfigError(
                "max_evaluations must allow at least one full generation",
                max_evaluations=self.max_evaluations,
                population_size=self.population_size,
            )
        if self.promotion.n_promote > self.population_size:
            raise ConfigError(
                "promotion.n_promote cannot exceed the population size",
                n_promote=self.promotion.n_promote,
                population_size=self.population_size,
            )
        return self


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
    evolution: EvolutionSettings = EvolutionSettings()
    walkforward: WalkForwardSettings = WalkForwardSettings()
    validation: ValidationSettings = ValidationSettings()
    lockbox: LockboxSettings = LockboxSettings()
    environment: EnvironmentSettings = EnvironmentSettings()
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


def _lookup(data: Mapping[str, Any], dotted: str) -> bool:
    """True if ``dotted`` names a key present in ``data``."""
    cursor: Any = data
    for part in dotted.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return False
        cursor = cursor[part]
    return True


def check_removed_keys(data: Mapping[str, Any]) -> None:
    """Raise a migration-shaped error for any key a spec change has retired.

    Raises:
        ConfigError: naming the key and what replaced it.
    """
    found = [key for key in REMOVED_KEYS if _lookup(data, key)]
    if found:
        raise ConfigError(
            "configuration uses a key that no longer exists:\n"
            + "\n".join(f"  {key}: {REMOVED_KEYS[key]}" for key in sorted(found)),
            keys=sorted(found),
        )


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

    check_removed_keys(data)

    try:
        return AppConfig(**data)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration:\n{exc}") from exc
