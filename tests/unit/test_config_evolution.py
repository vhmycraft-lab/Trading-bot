"""Evolutionary optimiser configuration (master spec section 13, config in section 5).

Schema only: nothing reads these values yet.  The point of testing them now is
that every validator here encodes a rule from the specification, and a rule that
is not tested is a rule that will be quietly broken by the first person who finds
it inconvenient.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from quantlab.core.config import (
    REMOVED_KEYS,
    AppConfig,
    DiversitySettings,
    EvolutionSettings,
    FitnessGates,
    FitnessPenalties,
    FitnessSettings,
    FitnessTargets,
    FitnessWeights,
    GenomeSettings,
    InnerWalkForwardSettings,
    MutationSettings,
    MutationStructuralSettings,
    PromotionSettings,
    check_removed_keys,
    load_config,
)
from quantlab.core.errors import ConfigError


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "override.yaml"
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# defaults match the specification
# ---------------------------------------------------------------------------
def test_defaults_match_the_spec(config: AppConfig) -> None:
    e = config.evolution
    assert e.enabled is True
    # Project Rome section 1 fixes the active population at 32, superseding the
    # master spec's 16. The proportions are the same ones scaled.
    assert e.population_size == 32
    assert (e.n_survivors, e.n_offspring, e.n_immigrants) == (24, 6, 2)
    assert e.n_survivors + e.n_offspring + e.n_immigrants == e.population_size
    assert e.max_generations == 30
    assert e.seed == 42
    assert e.max_evaluations == 1_000
    assert e.stop_on_no_improvement_generations == 8
    assert e.min_improvement == 0.005


def test_mutation_defaults(config: AppConfig) -> None:
    m = config.evolution.mutation
    assert (m.parameter_rate, m.structural_rate) == (0.7, 0.3)
    assert m.max_mutations_per_child == 3
    assert m.max_repair_attempts == 3
    assert m.parameter.perturb_pct == 0.25
    assert m.parameter.jump_probability == 0.15
    assert m.parameter.risk_perturb_pct == 0.35


def test_structural_operator_weights_cover_every_operator(config: AppConfig) -> None:
    """One weight per operator in the spec section 13.4 table."""
    assert set(config.evolution.mutation.structural.normalised()) == {
        "add_confirmation",
        "remove_confirmation",
        "modify_entry",
        "modify_exit",
        "add_filter",
        "remove_filter",
        "replace_indicator",
        "change_tree_mode",
    }


def test_diversity_defaults(config: AppConfig) -> None:
    d = config.evolution.diversity
    assert d.max_pairwise_similarity == 0.90
    assert d.min_population_diversity == 0.35
    assert (d.structural_weight, d.behavioural_weight) == (0.4, 0.6)
    assert d.behavioural_weight > d.structural_weight, "behaviour matters more than structure"
    assert (d.immigrant_boost, d.max_immigrants) == (2, 8)


def test_fitness_defaults(config: AppConfig) -> None:
    f = config.evolution.fitness
    assert f.weights.total == pytest.approx(1.0)
    assert f.targets.expectancy_pct == 0.002
    assert f.targets.profit_factor == 1.5
    assert f.gates.min_trades == 30
    # ADR 0012: the drawdown gate is relative to buy-and-hold over the same
    # window. max_drawdown survives only as the fallback for a window with no
    # benchmark, which is why it is now 1.00 rather than a second, tighter rule.
    assert f.gates.max_drawdown_vs_benchmark == 1.00
    assert f.gates.max_drawdown == 1.00
    assert f.penalties.removal_k == (1, 3, 5)
    assert f.penalties.removal_floor == (0.40, 0.20, 0.10)
    assert f.penalties.removal_target == (0.80, 0.65, 0.55)


def test_profit_is_not_the_objective(config: AppConfig) -> None:
    """Spec section 13.3: net profit and win rate carry the smallest weights."""
    weights = config.evolution.fitness.weights.as_dict()
    assert weights["net_return"] == 0.02
    assert weights["win_rate"] == 0.04
    assert weights["net_return"] == min(weights.values())
    assert weights["expectancy"] == max(weights.values())
    assert weights["net_return"] + weights["win_rate"] < weights["expectancy"]


def test_remaining_section_defaults(config: AppConfig) -> None:
    e = config.evolution
    assert (e.inner_walkforward.n_folds, e.inner_walkforward.scheme) == (4, "rolling")
    assert e.inner_walkforward.embargo_bars == 24
    assert (e.promotion.every_generations, e.promotion.n_promote) == (0, 3)
    assert e.promotion.min_fitness == 0.35
    assert e.promotion.max_similarity_between_promoted == 0.80
    assert e.promotion.counts_as_validation_touch is True
    assert e.genome.max_conditions_entry == 4
    assert e.genome.max_free_params == 6


def test_related_sections_gained_their_v11_keys(config: AppConfig) -> None:
    assert config.optimize.engine == "evolution"
    assert config.walkforward.evolution_generations == 8
    assert config.research.max_seed_genomes == 8
    assert config.research.guided_mutation_share == 0.25


def test_yaml_and_model_agree(default_config_path: Path) -> None:
    """The shipped YAML declares exactly the keys the model expects."""
    raw = yaml.safe_load(default_config_path.read_text(encoding="utf-8"))
    assert set(raw["evolution"]) == set(EvolutionSettings.model_fields)
    assert set(raw["evolution"]["fitness"]) == set(FitnessSettings.model_fields)
    assert set(raw["evolution"]["fitness"]["weights"]) == set(FitnessWeights.model_fields)
    assert set(raw["evolution"]["mutation"]) == set(MutationSettings.model_fields)
    assert set(raw["evolution"]["diversity"]) == set(DiversitySettings.model_fields)
    assert set(raw["evolution"]["promotion"]) == set(PromotionSettings.model_fields)
    assert set(raw["evolution"]["genome"]) == set(GenomeSettings.model_fields)
    assert set(raw["evolution"]["inner_walkforward"]) == set(InnerWalkForwardSettings.model_fields)


# ---------------------------------------------------------------------------
# population arithmetic
# ---------------------------------------------------------------------------
def test_the_slots_must_add_up() -> None:
    """A silently resized population would make every downstream trial count wrong."""
    with pytest.raises(ConfigError, match="must equal population_size"):
        EvolutionSettings(population_size=16, n_survivors=12, n_offspring=2, n_immigrants=1)


@pytest.mark.parametrize(
    ("survivors", "offspring", "immigrants", "size"),
    [(8, 6, 2, 16), (12, 3, 1, 16), (5, 2, 1, 8), (1, 1, 1, 3)],
)
def test_valid_population_shapes_are_accepted(
    survivors: int, offspring: int, immigrants: int, size: int
) -> None:
    settings = EvolutionSettings(
        population_size=size,
        n_survivors=survivors,
        n_offspring=offspring,
        n_immigrants=immigrants,
        diversity=DiversitySettings(max_immigrants=offspring + immigrants),
        max_evaluations=max(1000, size),
        promotion=PromotionSettings(n_promote=min(3, size)),
    )
    assert settings.open_slots == offspring + immigrants


def test_at_least_one_immigrant_is_reserved(default_config_path: Path) -> None:
    """Spec section 13.5: every generation contains a candidate owing nothing to the leader."""
    with pytest.raises(ConfigError):
        load_config(
            [],
            ["evolution.n_offspring=4", "evolution.n_immigrants=0"],
            default_path=default_config_path,
        )


def test_survivors_can_never_fill_the_whole_population(config: AppConfig) -> None:
    """Implied, not separately checked: the sum identity plus ``n_immigrants >= 1``."""
    e = config.evolution
    assert e.n_survivors < e.population_size


def test_a_field_constraint_also_surfaces_as_a_config_error(default_config_path: Path) -> None:
    """Field bounds raise pydantic errors; ``load_config`` normalises them all."""
    with pytest.raises(ConfigError, match="invalid configuration"):
        load_config([], ["evolution.population_size=1"], default_path=default_config_path)


def test_an_immigrant_boost_may_not_displace_a_survivor() -> None:
    """An immigrant boost fills open slots; there are ``n_offspring +
    n_immigrants`` of those, and one more would evict an elite."""
    with pytest.raises(ConfigError, match="never a survivor"):
        EvolutionSettings(diversity=DiversitySettings(max_immigrants=9))


def test_max_immigrants_must_cover_the_baseline() -> None:
    with pytest.raises(ConfigError, match="at least n_immigrants"):
        EvolutionSettings(
            population_size=16,
            n_survivors=10,
            n_offspring=3,
            n_immigrants=3,
            diversity=DiversitySettings(max_immigrants=1),
        )


def test_the_budget_must_allow_one_generation() -> None:
    with pytest.raises(ConfigError, match="at least one full generation"):
        EvolutionSettings(max_evaluations=8)


def test_cannot_promote_more_than_the_population() -> None:
    with pytest.raises(ConfigError, match="cannot exceed the population size"):
        EvolutionSettings(promotion=PromotionSettings(n_promote=40))


# ---------------------------------------------------------------------------
# fitness weights and targets
# ---------------------------------------------------------------------------
def test_weights_must_sum_to_one() -> None:
    with pytest.raises(ConfigError, match=r"must sum to 1\.0"):
        FitnessWeights(expectancy=0.5)


def test_weights_may_be_redistributed_if_they_still_sum_to_one() -> None:
    weights = FitnessWeights(expectancy=0.30, net_return=0.0, win_rate=0.0, trades=0.02)
    assert weights.total == pytest.approx(1.0)
    assert weights.as_dict()["expectancy"] == 0.30


def test_a_negative_weight_is_rejected() -> None:
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError or ConfigError
        FitnessWeights(expectancy=-0.1, profit_factor=0.42)


def test_profit_factor_target_must_exceed_one() -> None:
    """The component divides by ``target - 1``."""
    with pytest.raises(Exception):  # noqa: B017
        FitnessTargets(profit_factor=1.0)


def test_win_rate_range_must_be_non_degenerate() -> None:
    with pytest.raises(ConfigError, match=r"must exceed targets\.win_rate_floor"):
        FitnessTargets(win_rate=0.35, win_rate_floor=0.35)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expectancy_pct": 0.0},
        {"sortino": 0.0},
        {"cagr": 0.0},
        {"trades": 0},
        {"drawdown_ceiling": 0.0},
        {"consistency_period_bars": 0},
    ],
)
def test_zero_targets_are_rejected_because_they_divide(kwargs: dict[str, float]) -> None:
    with pytest.raises(Exception):  # noqa: B017
        FitnessTargets(**kwargs)


# ---------------------------------------------------------------------------
# gates and penalties must be coherent with one another
# ---------------------------------------------------------------------------
def test_soft_drawdown_must_sit_below_the_hard_gate() -> None:
    with pytest.raises(ConfigError, match=r"drawdown_soft must be below"):
        FitnessSettings(
            gates=FitnessGates(max_drawdown=0.20),
            penalties=FitnessPenalties(drawdown_soft=0.30),
            targets=FitnessTargets(drawdown_ceiling=0.15),
        )


def test_soft_trade_count_must_sit_above_the_hard_gate() -> None:
    with pytest.raises(ConfigError, match=r"trades_soft must be at or above"):
        FitnessSettings(
            gates=FitnessGates(min_trades=200), penalties=FitnessPenalties(trades_soft=100)
        )


def test_drawdown_ceiling_must_not_exceed_the_gate() -> None:
    with pytest.raises(ConfigError, match=r"drawdown_ceiling must not exceed"):
        FitnessSettings(
            gates=FitnessGates(max_drawdown=0.30), targets=FitnessTargets(drawdown_ceiling=0.40)
        )


# ---------------------------------------------------------------------------
# the trade-removal curve
# ---------------------------------------------------------------------------
def test_removal_lists_must_be_the_same_length() -> None:
    with pytest.raises(ConfigError, match="same length"):
        FitnessPenalties(removal_k=(1, 3), removal_floor=(0.4, 0.2, 0.1))


def test_removal_k_must_be_strictly_increasing() -> None:
    with pytest.raises(ConfigError, match="strictly increasing"):
        FitnessPenalties(
            removal_k=(3, 1, 5), removal_floor=(0.4, 0.2, 0.1), removal_target=(0.8, 0.65, 0.55)
        )


def test_removal_k_must_be_at_least_one() -> None:
    with pytest.raises(ConfigError, match=">= 1"):
        FitnessPenalties(
            removal_k=(0, 3, 5), removal_floor=(0.4, 0.2, 0.1), removal_target=(0.8, 0.65, 0.55)
        )


def test_removal_k_must_not_be_empty() -> None:
    with pytest.raises(ConfigError, match="must not be empty"):
        FitnessPenalties(removal_k=(), removal_floor=(), removal_target=())


def test_removal_thresholds_must_not_rise_with_k() -> None:
    """Retention can only fall as more winners are removed, so its floors must too."""
    with pytest.raises(ConfigError, match="non-increasing in k"):
        FitnessPenalties(
            removal_k=(1, 3, 5), removal_floor=(0.1, 0.2, 0.4), removal_target=(0.8, 0.65, 0.55)
        )


def test_a_floor_must_sit_below_its_target() -> None:
    with pytest.raises(ConfigError, match="below removal_target"):
        FitnessPenalties(
            removal_k=(1, 3, 5), removal_floor=(0.9, 0.2, 0.1), removal_target=(0.8, 0.65, 0.55)
        )


def test_removal_thresholds_must_be_fractions() -> None:
    with pytest.raises(ConfigError, match=r"must lie in \[0, 1\]"):
        FitnessPenalties(
            removal_k=(1, 3, 5), removal_floor=(1.4, 0.2, 0.1), removal_target=(0.8, 0.65, 0.55)
        )


def test_a_single_removal_point_is_allowed() -> None:
    penalties = FitnessPenalties(removal_k=(1,), removal_floor=(0.4,), removal_target=(0.8,))
    assert penalties.removal_k == (1,)


# ---------------------------------------------------------------------------
# mutation and diversity
# ---------------------------------------------------------------------------
def test_structural_weights_are_normalised_not_required_to_sum() -> None:
    structural = MutationStructuralSettings(add_confirmation=2.0, remove_confirmation=2.0)
    normalised = structural.normalised()
    assert sum(normalised.values()) == pytest.approx(1.0)
    assert normalised["add_confirmation"] == pytest.approx(normalised["remove_confirmation"])


def test_all_zero_structural_weights_are_rejected() -> None:
    with pytest.raises(ConfigError, match="at least one structural mutation operator"):
        MutationStructuralSettings(
            add_confirmation=0.0,
            remove_confirmation=0.0,
            modify_entry=0.0,
            modify_exit=0.0,
            add_filter=0.0,
            remove_filter=0.0,
            replace_indicator=0.0,
            change_tree_mode=0.0,
        )


def test_a_child_must_be_able_to_differ_from_its_parent() -> None:
    with pytest.raises(ConfigError, match="no child could ever differ"):
        MutationSettings(parameter_rate=0.0, structural_rate=0.0)


def test_similarity_weights_must_sum_to_one() -> None:
    with pytest.raises(ConfigError, match=r"must sum to 1\.0"):
        DiversitySettings(structural_weight=0.4, behavioural_weight=0.4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_pairwise_similarity": 0.0},
        {"max_pairwise_similarity": 1.5},
        {"min_population_diversity": 1.0},
        {"max_immigrants": 0},
    ],
)
def test_out_of_range_diversity_values_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(Exception):  # noqa: B017
        DiversitySettings(**kwargs)


def test_inner_walkforward_needs_at_least_two_folds() -> None:
    with pytest.raises(Exception):  # noqa: B017
        InnerWalkForwardSettings(n_folds=1)


# ---------------------------------------------------------------------------
# through the loader
# ---------------------------------------------------------------------------
def test_a_bad_population_in_yaml_is_reported(tmp_path: Path, default_config_path: Path) -> None:
    extra = write(tmp_path, "evolution:\n  n_offspring: 9\n")
    with pytest.raises(ConfigError, match="must equal population_size"):
        load_config([extra], default_path=default_config_path)


def test_bad_weights_in_yaml_are_reported(tmp_path: Path, default_config_path: Path) -> None:
    extra = write(tmp_path, "evolution:\n  fitness:\n    weights:\n      net_return: 0.5\n")
    with pytest.raises(ConfigError, match=r"must sum to 1\.0"):
        load_config([extra], default_path=default_config_path)


def test_an_unknown_evolution_key_is_rejected(tmp_path: Path, default_config_path: Path) -> None:
    extra = write(tmp_path, "evolution:\n  populaton_size: 16\n")
    with pytest.raises(ConfigError, match="populaton_size"):
        load_config([extra], default_path=default_config_path)


def test_set_override_reaches_the_evolution_section(default_config_path: Path) -> None:
    config = load_config(
        [],
        ["evolution.n_survivors=24", "evolution.n_offspring=6", "evolution.n_immigrants=2"],
        default_path=default_config_path,
    )
    assert (config.evolution.n_survivors, config.evolution.n_offspring) == (24, 6)


def test_env_override_reaches_the_evolution_section(
    monkeypatch: pytest.MonkeyPatch, default_config_path: Path
) -> None:
    monkeypatch.setenv("QUANTLAB__EVOLUTION__MAX_GENERATIONS", "5")
    assert load_config(default_path=default_config_path).evolution.max_generations == 5


def test_the_evolution_section_changes_the_config_hash(
    config: AppConfig, default_config_path: Path
) -> None:
    changed = load_config([], ["evolution.seed=43"], default_path=default_config_path)
    assert changed.config_hash != config.config_hash


def test_an_unknown_optimize_engine_is_rejected(tmp_path: Path, default_config_path: Path) -> None:
    extra = write(tmp_path, "optimize:\n  engine: telepathy\n")
    with pytest.raises(ConfigError):
        load_config([extra], default_path=default_config_path)


# ---------------------------------------------------------------------------
# backward compatibility
# ---------------------------------------------------------------------------
def test_a_config_without_an_evolution_section_still_loads(
    tmp_path: Path, default_config_path: Path
) -> None:
    """An existing overlay that predates spec 1.1 keeps working."""
    extra = write(tmp_path, "optimize:\n  n_trials: 40\nbacktest:\n  fee_bps: 12.0\n")
    config = load_config([extra], default_path=default_config_path)
    assert config.optimize.n_trials == 40
    assert config.evolution.population_size == 32


def test_the_model_supplies_the_whole_section_when_yaml_omits_it(tmp_path: Path) -> None:
    minimal = tmp_path / "minimal.yaml"
    minimal.write_text("market:\n  symbol: BTC/USDT\n", encoding="utf-8")
    config = load_config(default_path=minimal)
    assert config.evolution.population_size == 32
    assert config.evolution.fitness.weights.total == pytest.approx(1.0)


def test_a_removed_key_gets_a_migration_message(tmp_path: Path, default_config_path: Path) -> None:
    """`extra="forbid"` alone would say only "extra inputs are not permitted"."""
    extra = write(tmp_path, "research:\n  max_iterations: 25\n")
    with pytest.raises(ConfigError, match=r"evolution\.max_generations"):
        load_config([extra], default_path=default_config_path)


def test_removed_keys_are_documented() -> None:
    assert "research.max_iterations" in REMOVED_KEYS
    assert "spec 1.1" in REMOVED_KEYS["research.max_iterations"]


def test_check_removed_keys_ignores_a_clean_config() -> None:
    check_removed_keys({"research": {"max_seed_genomes": 8}, "evolution": {}})


def test_check_removed_keys_tolerates_a_non_mapping_branch() -> None:
    check_removed_keys({"research": "not-a-mapping"})


# ---------------------------------------------------------------------------
# Project Rome section 1: exactly 32 active strategies
# ---------------------------------------------------------------------------
def test_the_active_population_is_exactly_thirty_two(config: AppConfig) -> None:
    """Rome section 46 asks for this verbatim: "Active population = exactly 32".

    It is a property of the *configuration*, not of the loop, because the three
    slot counts are validated to sum to the population size. That is what makes
    "no cycle may finish with 31 or 33" unreachable rather than merely unlikely:
    a generation is assembled from survivors, offspring and immigrants, and there
    is no configuration in which those add up to anything else.
    """
    e = config.evolution
    assert e.population_size == 32
    assert e.n_survivors + e.n_offspring + e.n_immigrants == 32


def test_a_population_whose_slots_do_not_add_up_is_refused() -> None:
    """The enforcement behind the test above. Without it, 32 would be a number in
    a file rather than a guarantee."""
    with pytest.raises(ConfigError, match="must equal population_size"):
        EvolutionSettings(population_size=32, n_survivors=24, n_offspring=6, n_immigrants=3)
    with pytest.raises(ConfigError, match="must equal population_size"):
        EvolutionSettings(population_size=32, n_survivors=24, n_offspring=6, n_immigrants=1)


def test_a_generation_always_reserves_a_slot_for_an_unrelated_candidate() -> None:
    """Rome section 26's anti-memorisation, in the population rules: at least one
    immigrant every generation, so the search always carries a candidate that
    owes nothing to the current leader."""
    assert EvolutionSettings().n_immigrants >= 1
    with pytest.raises(ValidationError):
        EvolutionSettings(population_size=32, n_survivors=26, n_offspring=6, n_immigrants=0)
