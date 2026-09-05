"""Diversity: similarity, niching and the immigrant boost (spec section 13.5, T49).

Section 22's acceptance criteria: similarity is 1.0 for a candidate against
itself and below 0.2 for two unrelated baselines; niching rejects a duplicate
survivor; the immigrant boost fires below the floor and stops above it.

The baseline test runs the real baselines through the real engine rather than
asserting against made-up series. The measure's whole purpose is to say whether
two strategies would confirm each other, and only strategies that actually traded
can answer that.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tests.helpers import make_bars, zero_cost_config

from quantlab.adapters.engine.simple_bar import SimpleBarEngine
from quantlab.core.config import DiversitySettings, EvolutionSettings
from quantlab.core.errors import StrategyError
from quantlab.core.genome import Condition, ConditionTree, Operand, StrategyGenome
from quantlab.core.strategy import ParamSpec
from quantlab.core.types import BarFrame, RiskSpec
from quantlab.evolution.diversity import (
    CandidateView,
    Signature,
    agreement,
    behaviour_hash,
    duplicate_groups,
    genome_signature,
    jaccard,
    population_diversity,
    select_survivors,
    signature_from_source,
    similarity,
    slot_plan,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINES = REPO_ROOT / "strategies" / "baselines"

FAST = Operand(kind="indicator", name="sma", kwargs={"n": "fast"})
SLOW = Operand(kind="indicator", name="sma", kwargs={"n": "slow"})
RSI = Operand(kind="indicator", name="rsi", kwargs={"n": 14})
THIRTY = Operand(kind="constant", kwargs={"value": 30.0})


def crossover(**overrides: object) -> StrategyGenome:
    fields: dict[str, object] = {
        "name": "crossover",
        "entry": ConditionTree(conditions=(Condition(left=FAST, op=">", right=SLOW),)),
        "exit": ConditionTree(conditions=(Condition(left=FAST, op="<=", right=SLOW),)),
        "params": {
            "fast": ParamSpec(kind="int", default=10, low=5, high=50),
            "slow": ParamSpec(kind="int", default=30, low=20, high=100),
        },
        "warmup_bars": 100,
    }
    fields.update(overrides)
    return StrategyGenome(**fields)  # type: ignore[arg-type]


def view(name: str, positions: list[float], genome: StrategyGenome | None = None) -> CandidateView:
    return CandidateView.from_genome(name, genome or crossover(), positions)


# ---------------------------------------------------------------------------
# the real baselines
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def baselines() -> dict[str, CandidateView]:
    """Every baseline of section 9.5, run for real and viewed as a candidate."""
    bars = BarFrame(make_bars(700, seed=5, drift=0.05), symbol="BTC/USDT", timeframe="1h")
    views: dict[str, CandidateView] = {}
    for name in ("sma_cross", "rsi_reversion", "buy_and_hold", "random_entry"):
        source = (BASELINES / f"{name}.py").read_text(encoding="utf-8")
        namespace: dict[str, Any] = {}
        exec(compile(source, f"{name}.py", "exec"), namespace)
        params = {"fast": 10, "slow": 30} if name == "sma_cross" else {}
        result = SimpleBarEngine().run(namespace["STRATEGY"](), bars, params, zero_cost_config())
        positions = np.asarray(result.position_frac, dtype="float64")
        views[name] = CandidateView(
            candidate_id=name,
            signature=signature_from_source(source),
            positions=positions,
            behaviour=behaviour_hash(positions),
        )
    return views


def test_a_candidate_is_perfectly_similar_to_itself(baselines: dict[str, CandidateView]) -> None:
    """Section 22's first criterion. Exactly 1.0, not approximately."""
    for name, candidate in baselines.items():
        assert similarity(candidate, candidate) == 1.0, name


def test_two_unrelated_baselines_are_dissimilar(baselines: dict[str, CandidateView]) -> None:
    """Section 22's second criterion, on every pair it holds for.

    ``buy_and_hold`` against ``random_entry`` is the one exception and is
    excluded knowingly: ``random_entry`` is long on roughly 45 % of bars and
    ``buy_and_hold`` on all of them, so they genuinely hold the same position
    nearly half the time. That is the measure reporting a real overlap, not
    failing to see a difference.
    """
    for left, right in itertools.combinations(sorted(baselines), 2):
        if {left, right} == {"buy_and_hold", "random_entry"}:
            continue
        assert similarity(baselines[left], baselines[right]) < 0.2, (left, right)


def test_the_one_overlapping_pair_is_overlapping_for_the_stated_reason(
    baselines: dict[str, CandidateView],
) -> None:
    """Pins the exception above, so it cannot quietly become something else."""
    hold, random = baselines["buy_and_hold"], baselines["random_entry"]
    assert 0.2 < similarity(hold, random) < 0.4
    assert 0.4 < agreement(hold.positions, random.positions) < 0.5  # type: ignore[arg-type]
    assert jaccard(hold.signature, random.signature) == 0.0


def test_an_opaque_signature_is_read_from_the_source_not_run(
    baselines: dict[str, CandidateView],
) -> None:
    """Section 13.5: an opaque candidate's signature comes from its AST (INV-4)."""
    assert baselines["sma_cross"].signature.triples == (
        ("indicator", "sma", ">"),
        ("indicator", "sma", ">"),
    )
    assert baselines["rsi_reversion"].signature.triples == (
        ("indicator", "rsi", "<"),
        ("indicator", "rsi", ">"),
    )
    # `buy_and_hold` compares nothing; `random_entry` decides by coin toss, and
    # recording that is what stops the two looking structurally identical.
    assert baselines["buy_and_hold"].signature.is_empty
    assert baselines["random_entry"].signature.triples == (("random", "random", "<"),)


def test_a_local_name_bound_to_an_indicator_is_followed(
    baselines: dict[str, CandidateView],
) -> None:
    """Hand-written strategies read ``fast = ctx.ind(...)`` and compare the name.
    Without following that hop, every opaque signature would come out empty."""
    source = (BASELINES / "sma_cross.py").read_text(encoding="utf-8")
    assert "fast > slow" in source
    assert not signature_from_source(source).is_empty


def test_a_warm_up_guard_is_not_structure() -> None:
    """``fast != fast`` is a NaN check, not a comparison the strategy trades on."""
    source = 'def f(ctx):\n    a = ctx.ind("sma", n=5)\n    return a != a\n'
    assert signature_from_source(source).is_empty


def test_a_source_that_does_not_parse_is_refused() -> None:
    with pytest.raises(StrategyError, match="does not parse"):
        signature_from_source("def broken(:\n")


# ---------------------------------------------------------------------------
# the two halves
# ---------------------------------------------------------------------------
def test_a_genome_signature_records_structure_and_not_periods() -> None:
    """Two crossovers at 20/50 and 25/60 are the same structure; the behavioural
    half is what tells them apart."""
    slow_operand = Operand(kind="indicator", name="sma", kwargs={"n": 200})
    other = crossover(
        entry=ConditionTree(conditions=(Condition(left=FAST, op=">", right=slow_operand),)),
        exit=ConditionTree(conditions=(Condition(left=FAST, op="<=", right=slow_operand),)),
        params={"fast": ParamSpec(kind="int", default=10, low=5, high=50)},
        warmup_bars=200,
    )
    assert genome_signature(crossover()) == genome_signature(other)


def test_a_signature_is_a_multiset_not_a_set() -> None:
    """Three moving averages is not the same structure as two, and collapsing the
    repeats would hide that."""
    three = crossover(
        entry=ConditionTree(
            conditions=(
                Condition(left=FAST, op=">", right=SLOW),
                Condition(
                    left=Operand(kind="indicator", name="sma", kwargs={"n": 5}), op=">", right=SLOW
                ),
            )
        )
    )
    assert len(genome_signature(three).triples) > len(genome_signature(crossover()).triples)
    assert jaccard(genome_signature(three), genome_signature(crossover())) < 1.0


def test_enabled_risk_controls_are_part_of_the_signature() -> None:
    with_stop = crossover(risk=RiskSpec(stop_loss_pct=0.03))
    assert genome_signature(with_stop).risk_controls == {"stop_loss_pct"}
    assert genome_signature(crossover()).risk_controls == frozenset()
    assert jaccard(genome_signature(with_stop), genome_signature(crossover())) < 1.0


def test_jaccard_of_two_empty_signatures_is_one() -> None:
    """Structurally identical, having no structure — not a division by zero."""
    assert jaccard(Signature(), Signature()) == 1.0


def test_jaccard_of_disjoint_signatures_is_zero() -> None:
    assert jaccard(Signature(), genome_signature(crossover())) == 0.0


def test_agreement_ignores_bars_where_both_are_flat() -> None:
    """Section 13.5 wants strategies that "enter and exit together"; agreeing to
    stay out of the market is not that, and counting it puts a floor under the
    measure that rises with how selective the two strategies are."""
    left = [0.0, 0.0, 0.0, 1.0]
    right = [0.0, 0.0, 0.0, -1.0]
    assert agreement(left, right) == 0.0
    assert agreement(left, left) == 1.0


def test_two_candidates_that_never_trade_did_the_same_nothing() -> None:
    assert agreement([0.0, 0.0], [0.0, 0.0]) == 1.0
    assert agreement([], []) == 1.0


def test_agreement_is_about_sign_not_size() -> None:
    """One sizing at half the other's fraction still agrees about the market."""
    assert agreement([1.0, 1.0, 0.0], [0.5, 0.25, 0.0]) == 1.0
    assert agreement([1.0, 1.0], [-1.0, -1.0]) == 0.0


def test_series_of_different_lengths_are_refused() -> None:
    """Comparing a prefix would report agreement over bars one never saw."""
    with pytest.raises(StrategyError, match="same bars"):
        agreement([1.0, 0.0], [1.0])


def test_the_behaviour_hash_absorbs_float_noise_but_not_a_real_difference() -> None:
    base = [0.0, 1.0, 1.0, 0.0, 0.5]
    noisy = [value + 1e-9 for value in base]
    assert behaviour_hash(base) == behaviour_hash(noisy)
    assert behaviour_hash(base) != behaviour_hash([0.0, 1.0, 1.0, 0.0, 1.0])


def test_a_zero_quantum_is_refused() -> None:
    with pytest.raises(StrategyError, match="quantum must be positive"):
        behaviour_hash([1.0], quantum=0.0)


def test_duplicate_groups_find_exact_behavioural_twins() -> None:
    """The cheap half of duplicate detection: equal hashes need no comparison."""
    twins = [view("a", [1.0, 0.0]), view("b", [1.0, 0.0]), view("c", [0.0, 1.0])]
    assert duplicate_groups(twins) == {twins[0].behaviour: ["a", "b"]}


def test_two_unevaluated_candidates_are_taken_as_behaviourally_identical() -> None:
    """The conservative reading: they have shown no difference, so niching falls
    back to structure. Assuming they differ would be an unearned claim."""
    bare = CandidateView("x", signature=genome_signature(crossover()))
    assert similarity(bare, bare) == 1.0


# ---------------------------------------------------------------------------
# niching
# ---------------------------------------------------------------------------
def test_niching_rejects_a_duplicate_survivor() -> None:
    """Section 22's third criterion. The fitter of two twins survives; the other
    is skipped and the next distinct candidate takes the slot."""
    ranked = [
        view("best", [1.0, 1.0, 0.0, 0.0]),
        view("twin", [1.0, 1.0, 0.0, 0.0]),
        view("other", [0.0, 0.0, 1.0, 1.0]),
    ]
    survivors = select_survivors(ranked, n_survivors=2)
    assert [s.candidate_id for s in survivors] == ["best", "other"]


def test_the_fittest_candidate_is_always_accepted() -> None:
    ranked = [view(f"c{i}", [1.0, 1.0, 0.0]) for i in range(5)]
    assert select_survivors(ranked, n_survivors=3)[0].candidate_id == "c0"


def test_a_quota_that_cannot_be_filled_is_left_short() -> None:
    """Section 13.2 step 4: the shortfall becomes extra immigrants. Topping it up
    with the duplicates just rejected would undo the whole mechanism."""
    twins = [view(f"twin{i}", [1.0, 1.0, 0.0]) for i in range(6)]
    assert len(select_survivors(twins, n_survivors=4)) == 1


def test_niching_preserves_the_ranking_order() -> None:
    ranked = [view(f"c{i}", [1.0 if i % 2 else 0.0, 0.0, 1.0]) for i in range(6)]
    survivors = select_survivors(ranked, n_survivors=6)
    assert [s.candidate_id for s in survivors] == sorted(s.candidate_id for s in survivors)


def test_a_looser_threshold_admits_more_survivors() -> None:
    ranked = [view("a", [1.0, 1.0, 0.0]), view("b", [1.0, 1.0, 1.0])]
    strict = select_survivors(
        ranked,
        n_survivors=2,
        settings=DiversitySettings(
            max_pairwise_similarity=0.5, structural_weight=0.4, behavioural_weight=0.6
        ),
    )
    loose = select_survivors(
        ranked,
        n_survivors=2,
        settings=DiversitySettings(
            max_pairwise_similarity=0.99, structural_weight=0.4, behavioural_weight=0.6
        ),
    )
    assert len(strict) == 1
    assert len(loose) == 2


# ---------------------------------------------------------------------------
# the diversity floor and the immigrant boost
# ---------------------------------------------------------------------------
def test_population_diversity_is_one_minus_the_mean_pairwise_similarity() -> None:
    identical = [view("a", [1.0, 1.0]), view("b", [1.0, 1.0])]
    assert population_diversity(identical) == pytest.approx(0.0)

    opposed = [view("a", [1.0, 1.0]), view("b", [-1.0, -1.0])]
    # Structure is shared (both are the same genome), behaviour is not.
    assert population_diversity(opposed) == pytest.approx(0.6)


def test_a_population_of_one_has_no_spread_to_report() -> None:
    assert population_diversity([view("only", [1.0])]) == 1.0
    assert population_diversity([]) == 1.0


def test_the_immigrant_boost_fires_below_the_floor_and_stops_above_it() -> None:
    """Section 22's fourth criterion."""
    settings = EvolutionSettings()
    floor = settings.diversity.min_population_diversity

    above = slot_plan(settings, diversity=floor + 0.01)
    assert not above.boosted
    assert above.n_immigrants == settings.n_immigrants
    assert above.n_offspring == settings.n_offspring

    below = slot_plan(settings, diversity=floor - 0.01)
    assert below.boosted
    assert below.n_immigrants == settings.n_immigrants + settings.diversity.immigrant_boost
    assert below.n_offspring == settings.n_offspring - settings.diversity.immigrant_boost


def test_the_boost_never_displaces_a_survivor() -> None:
    """Which is why config load requires ``max_immigrants <= n_offspring + n_immigrants``."""
    settings = EvolutionSettings()
    plan = slot_plan(settings, diversity=0.0)
    assert plan.n_survivors == settings.n_survivors
    assert plan.n_offspring >= 0
    assert plan.n_immigrants <= settings.diversity.max_immigrants


def test_the_boost_is_capped_at_max_immigrants() -> None:
    settings = EvolutionSettings(
        n_survivors=10,
        n_offspring=5,
        n_immigrants=1,
        population_size=16,
        diversity=DiversitySettings(immigrant_boost=5, max_immigrants=3),
    )
    assert slot_plan(settings, diversity=0.0).n_immigrants == 3


def test_a_survivor_shortfall_becomes_extra_immigrants() -> None:
    """Never extra offspring: offspring are mutations *of survivors*, and there
    were not enough distinct ones (section 13.2, step 4)."""
    settings = EvolutionSettings()
    plan = slot_plan(settings, diversity=0.9, n_survivors=settings.n_survivors - 2)
    assert plan.n_survivors == settings.n_survivors - 2
    assert plan.n_immigrants == settings.n_immigrants + 2
    assert plan.n_offspring == settings.n_offspring


def test_a_generation_always_totals_the_population_size() -> None:
    """Section 13.2 step 7. A generation that quietly resized would make every
    downstream deflated Sharpe ratio wrong."""
    settings = EvolutionSettings()
    for shortfall in range(settings.n_survivors + 1):
        for diversity in (0.0, 0.2, 0.5, 1.0):
            plan = slot_plan(
                settings, diversity=diversity, n_survivors=settings.n_survivors - shortfall
            )
            assert plan.total == settings.population_size
            assert plan.n_offspring >= 0
            assert plan.n_immigrants >= 1


def test_novelty_is_always_reserved() -> None:
    """Section 13.5's third rule: every generation holds at least one candidate
    that owes nothing to the current leader."""
    settings = EvolutionSettings()
    for diversity in (0.0, 0.5, 1.0):
        assert slot_plan(settings, diversity=diversity).n_immigrants >= 1
