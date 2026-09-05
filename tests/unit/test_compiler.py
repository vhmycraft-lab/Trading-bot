"""Genome compilation (master spec section 9.6, task T46).

Section 9.6 makes three claims about ``compile_genome`` and this file holds it to
all three: compilation is **deterministic** (the same genome always produces
byte-identical source, so ``strategy_id`` is stable and the run cache of section
11.2 works across generations), **total** (every genome that validated compiles),
and **executes nothing** (INV-4).

The behavioural tests run the generated module. That is not a sandbox bypass: the
genome under test is written here, the source is put through the section 9.2 AST
check first — which is one of the acceptance criteria anyway — and what is being
checked is whether the compiler's output *means* what the genome said. Untrusted
code still goes through the sandbox; nothing here is untrusted.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tests.helpers import flat_frame, make_bars, zero_cost_config

from quantlab.adapters.engine.simple_bar import SimpleBarEngine
from quantlab.core.genome import (
    Condition,
    ConditionTree,
    Operand,
    StrategyGenome,
    genome_id,
)
from quantlab.core.hashing import strategy_id
from quantlab.core.strategy import ParamSpec
from quantlab.core.types import BarFrame, RiskSpec, SizingSpec
from quantlab.evolution.compiler import compile_genome
from quantlab.sandbox.ast_check import require_safe_source

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPILER_SOURCE = REPO_ROOT / "src" / "quantlab" / "evolution" / "compiler.py"
BASELINES = REPO_ROOT / "strategies" / "baselines"

FAST = Operand(kind="indicator", name="sma", kwargs={"n": "fast"})
SLOW = Operand(kind="indicator", name="sma", kwargs={"n": "slow"})
CLOSE = Operand(kind="price", name="close")
ZERO = Operand(kind="constant", kwargs={"value": 0.0})


# ---------------------------------------------------------------------------
# genomes equivalent to the baselines of section 9.5
# ---------------------------------------------------------------------------
def sma_cross_genome() -> StrategyGenome:
    """``strategies/baselines/sma_cross.py``: long while fast is above slow.

    The baseline is memoryless — ``LONG if fast > slow else FLAT`` — so the exit
    is the exact negation of the entry, ties included. Anything looser would
    differ from it on a bar where the two averages are equal.
    """
    return StrategyGenome(
        name="sma_cross",
        entry=ConditionTree(conditions=(Condition(left=FAST, op=">", right=SLOW),)),
        exit=ConditionTree(conditions=(Condition(left=FAST, op="<=", right=SLOW),)),
        params={
            "fast": ParamSpec(kind="int", default=50, low=5, high=200),
            "slow": ParamSpec(kind="int", default=200, low=20, high=400),
        },
        warmup_bars=400,
    )


def rsi_reversion_genome() -> StrategyGenome:
    """``strategies/baselines/rsi_reversion.py``: long below one level, flat above another.

    The baseline holds between the thresholds, which is what an entry tree and a
    separate exit tree mean; the asymmetry is the point of the strategy.
    """
    rsi = Operand(kind="indicator", name="rsi", kwargs={"n": "n"})
    return StrategyGenome(
        name="rsi_reversion",
        entry=ConditionTree(
            conditions=(Condition(left=rsi, op="<", right=Operand(kind="param", name="oversold")),)
        ),
        exit=ConditionTree(
            conditions=(
                Condition(left=rsi, op=">", right=Operand(kind="param", name="exit_level")),
            )
        ),
        params={
            "n": ParamSpec(kind="int", default=14, low=2, high=50),
            "oversold": ParamSpec(kind="float", default=30.0, low=5.0, high=45.0),
            "exit_level": ParamSpec(kind="float", default=55.0, low=46.0, high=90.0),
        },
        warmup_bars=60,
    )


def buy_and_hold_genome() -> StrategyGenome:
    """``strategies/baselines/buy_and_hold.py``: long from the first bar.

    A genome always states a condition, so "always" is spelled as the one thing
    every bar satisfies: a positive close. Bar validation rejects a non-positive
    price, so this is true on every bar the engine can be handed.
    """
    return StrategyGenome(
        name="buy_and_hold",
        entry=ConditionTree(conditions=(Condition(left=CLOSE, op=">", right=ZERO),)),
        warmup_bars=0,
    )


BASELINE_GENOMES = {
    "sma_cross": sma_cross_genome,
    "rsi_reversion": rsi_reversion_genome,
    "buy_and_hold": buy_and_hold_genome,
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def build(source: str, *, name: str = "compiled") -> Any:
    """AST-check the generated source, then instantiate what it declares."""
    require_safe_source(source)
    namespace: dict[str, Any] = {}
    exec(compile(source, f"{name}.py", "exec"), namespace)
    return namespace["STRATEGY"]()


def load_baseline(name: str) -> Any:
    return build((BASELINES / f"{name}.py").read_text(encoding="utf-8"), name=name)


def fixture_bars(n: int = 700) -> BarFrame:
    """Deterministic bars chosen to exercise every baseline.

    This seed produces both moving-average crossings and RSI excursions below 30
    and above 55, so the equivalence tests below compare two strategies that
    actually trade rather than two that agree about staying flat.
    """
    frame = make_bars(n, seed=5, drift=0.05)
    return BarFrame(frame, symbol="BTC/USDT", timeframe="1h")


def positions(strategy: Any, bars: BarFrame, params: dict[str, Any] | None = None) -> np.ndarray:
    result = SimpleBarEngine().run(strategy, bars, params or {}, zero_cost_config())
    return np.asarray(result.position_frac, dtype="float64")


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------
def test_the_same_genome_compiles_byte_identically() -> None:
    """Section 9.6's central claim: ``strategy_id`` is stable across generations."""
    genome = sma_cross_genome()
    assert compile_genome(genome) == compile_genome(genome)
    assert compile_genome(genome) == compile_genome(sma_cross_genome())


def test_compilation_survives_a_round_trip_through_json() -> None:
    """A genome read back from ``candidate.genome_json`` must compile the same."""
    import json

    genome = sma_cross_genome()
    restored = StrategyGenome.model_validate(json.loads(genome.canonical()))
    assert compile_genome(restored) == compile_genome(genome)
    assert genome_id(restored) == genome_id(genome)


def test_compilation_is_stable_in_a_fresh_process() -> None:
    """Nothing about the source may depend on dict ordering, ids, or the clock."""
    script = (
        "import json;"
        "from quantlab.core.genome import StrategyGenome;"
        "from quantlab.evolution.compiler import compile_genome;"
        "import sys;"
        "print(compile_genome(StrategyGenome.model_validate(json.load(sys.stdin))), end='')"
    )
    done = subprocess.run(
        [sys.executable, "-c", script],
        input=sma_cross_genome().canonical(),
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert done.stdout == compile_genome(sma_cross_genome())


def test_a_different_structure_compiles_to_different_source() -> None:
    base = sma_cross_genome()
    variants = [
        base,
        base.model_copy(update={"warmup_bars": 401}),
        base.model_copy(update={"version": "2"}),
        base.model_copy(update={"risk": RiskSpec(stop_loss_pct=0.03)}),
        base.model_copy(update={"sizing": SizingSpec(fraction=0.25)}),
    ]
    sources = {compile_genome(variant) for variant in variants}
    assert len(sources) == len(variants)


def test_the_strategy_id_of_the_compiled_source_is_the_genome_s_fingerprint() -> None:
    """Two ids, on purpose: the structure's, and the bytes' (section 9.6)."""
    genome = sma_cross_genome()
    identifier = strategy_id(compile_genome(genome))
    assert identifier == strategy_id(compile_genome(sma_cross_genome()))
    assert identifier != genome_id(genome)


# ---------------------------------------------------------------------------
# totality and safety
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(BASELINE_GENOMES))
def test_compiled_output_passes_the_section_9_2_check(name: str) -> None:
    report = require_safe_source(compile_genome(BASELINE_GENOMES[name]()))
    assert report.style == "bar_loop"
    assert report.warmup_bars == BASELINE_GENOMES[name]().warmup_bars


def test_a_genome_using_every_operand_kind_and_operator_compiles_and_checks() -> None:
    """Totality: nothing the genome can express falls through the compiler."""
    rsi = Operand(kind="indicator", name="rsi", kwargs={"n": 14})
    high = Operand(kind="price", name="high")
    genome = StrategyGenome(
        name="everything",
        entry=ConditionTree(
            mode="any",
            conditions=(
                Condition(left=FAST, op="cross_above", right=SLOW),
                Condition(left=CLOSE, op=">=", right=Operand(kind="param", name="floor")),
            ),
        ),
        exit=ConditionTree(
            conditions=(
                Condition(left=FAST, op="cross_below", right=SLOW),
                Condition(left=rsi, op=">", right=Operand(kind="constant", kwargs={"value": 70})),
            )
        ),
        filters=(Condition(left=high, op=">", right=CLOSE),),
        risk=RiskSpec(stop_loss_pct=0.02, take_profit_pct=0.06, time_stop_bars=48),
        sizing=SizingSpec(mode="fixed_fraction", fraction=0.5),
        params={
            "fast": ParamSpec(kind="int", default=10, low=5, high=50),
            "slow": ParamSpec(kind="int", default=30, low=20, high=100),
            "floor": ParamSpec(kind="float", default=1.0, low=0.0, high=1e6),
        },
        warmup_bars=100,
    )
    source = compile_genome(genome)
    require_safe_source(source)
    strategy = build(source)
    assert strategy.risk.stop_loss_pct == 0.02
    assert strategy.sizing.fraction == 0.5
    assert positions(strategy, fixture_bars()).shape == (700,)


def test_the_compiler_itself_executes_nothing() -> None:
    """INV-4, read off the compiler's own syntax tree rather than promised."""
    tree = ast.parse(COMPILER_SOURCE.read_text(encoding="utf-8"))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not called & {"exec", "eval", "compile", "__import__", "open"}


def test_the_generated_tree_is_excluded_from_the_repository_formatter() -> None:
    """Generated source is stored under the hash of its own bytes, so a
    ``make format`` that rewrote it would change the ``strategy_id`` every run
    cites and split the cache of section 11.2.

    The guarantee is the exclude, not the layout: the compiler emits readable
    Python, but chasing a formatter's exact line-splitting for arbitrary boolean
    expressions would be a claim it could not keep.
    """
    config = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "strategies/generated" in config

    source = compile_genome(sma_cross_genome())
    assert "\t" not in source
    assert source.endswith("\n")
    assert not any(line.rstrip() != line for line in source.splitlines())


def test_the_class_name_is_derived_from_the_genome_name() -> None:
    source = compile_genome(sma_cross_genome())
    assert "class SmaCross:" in source
    assert "STRATEGY = SmaCross" in source
    assert f'GENOME_ID = "{genome_id(sma_cross_genome())}"' in source


# ---------------------------------------------------------------------------
# behaviour
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(BASELINE_GENOMES))
def test_a_compiled_genome_matches_its_baseline_bar_for_bar(name: str) -> None:
    """Section 22's acceptance criterion for T46: equal ``position_frac``."""
    bars = fixture_bars()
    compiled = positions(build(compile_genome(BASELINE_GENOMES[name]()), name=name), bars)
    baseline = positions(load_baseline(name), bars)
    np.testing.assert_allclose(compiled, baseline, rtol=0, atol=0)
    if name != "buy_and_hold":
        assert compiled.any(), "a test that never takes a position proves nothing"


def test_random_entry_has_no_genome_and_the_gap_is_deliberate() -> None:
    """``random_entry`` draws from ``ctx.rng``; the genome has no operand for
    randomness, so it stays an ``opaque`` candidate (section 9.6)."""
    source = (BASELINES / "random_entry.py").read_text(encoding="utf-8")
    assert "ctx.rng" in source
    assert "random" not in {name for name in dir(Operand) if not name.startswith("_")}


def test_an_empty_exit_tree_holds_until_a_risk_control_fires() -> None:
    genome = StrategyGenome(
        name="hold_forever",
        entry=ConditionTree(conditions=(Condition(left=CLOSE, op=">", right=ZERO),)),
        exit=ConditionTree(),
        warmup_bars=0,
    )
    held = positions(build(compile_genome(genome)), fixture_bars(120))
    # Bar 0 plans the order that bar 1 fills, and the engine closes the position
    # at the end of the data (section 8.4), so the held stretch is everything in
    # between — which, with no exit condition, is every bar there is.
    assert (held[1:-1] > 0).all()


def test_a_filter_gates_entry_without_forcing_an_exit() -> None:
    """A filter says when a position may be *opened*; the exit tree closes it."""
    up = Operand(kind="price", name="high")
    genome = StrategyGenome(
        name="filtered",
        entry=ConditionTree(conditions=(Condition(left=CLOSE, op=">", right=ZERO),)),
        exit=ConditionTree(conditions=(Condition(left=CLOSE, op="<", right=ZERO),)),
        filters=(Condition(left=up, op="<", right=CLOSE),),
        warmup_bars=0,
    )
    source = compile_genome(genome)
    assert "allowed = " in source
    assert "opening = entry and allowed" in source
    # The filter is never satisfiable — a bar's high is never below its close — so
    # nothing ever opens, which is the filter doing its job rather than the entry.
    assert not positions(build(source), fixture_bars(120)).any()


def crossing_genome() -> StrategyGenome:
    """Long on a 2-bar average crossing above a 3-bar one.

    The bounds are tight so the warm-up rule costs three bars and the hand-made
    series below can be read off on paper.
    """
    return StrategyGenome(
        name="crossing",
        entry=ConditionTree(conditions=(Condition(left=FAST, op="cross_above", right=SLOW),)),
        params={
            "fast": ParamSpec(kind="int", default=2, low=2, high=3),
            "slow": ParamSpec(kind="int", default=3, low=2, high=3),
        },
        warmup_bars=3,
    )


def signals_over(strategy: Any, closes: list[float], params: dict[str, Any]) -> list[str]:
    """Ask a compiled strategy for its signal at each bar, always from flat.

    Driven directly rather than through the engine: once the engine is long it
    stops asking whether to open, which is exactly the state that would hide a
    mistake in the crossing expression.
    """
    from quantlab.core.strategy import Context, IndicatorCache
    from quantlab.core.types import BarWindow, Position

    bars = flat_frame(closes, spread=0.0)
    cache = IndicatorCache(bars)
    strategy.prepare(params)
    out = []
    for i in range(bars.n_bars):
        if i < strategy.warmup_bars:
            continue
        signal = strategy.on_bar(
            Context(
                i=i,
                bars=BarWindow(bars, i),
                position=Position.flat(),
                equity=10_000.0,
                cash=10_000.0,
                params=params,
                cache=cache,
            )
        )
        out.append(signal.kind.value)
    return out


def test_a_crossing_fires_only_on_the_bar_that_crosses() -> None:
    """``sma(2)`` overtakes ``sma(3)`` at bar 4 of this series and stays above."""
    strategy = build(compile_genome(crossing_genome()))
    emitted = signals_over(strategy, [10, 10, 10, 10, 20, 20, 20], {"fast": 2, "slow": 3})
    #   bar:      3      4       5      6
    assert emitted == ["flat", "long", "flat", "flat"]


def test_a_crossing_is_never_invented_where_the_previous_bar_is_unknown() -> None:
    """The generated module remembers the last bar on the instance, and the first
    bar it is asked about has no last bar.

    This series crosses at bar 3 — the first bar past warm-up — so a compiler that
    assumed a previous value would report a crossing here. Refusing to is the
    conservative reading: whether it crossed is genuinely unknown.
    """
    strategy = build(compile_genome(crossing_genome()))
    emitted = signals_over(strategy, [10, 10, 10, 20, 20, 20, 20], {"fast": 2, "slow": 3})
    assert set(emitted) == {"flat"}


def test_the_generated_module_re_reads_state_deterministically() -> None:
    """Two runs of one compiled strategy over the same bars agree exactly — the
    per-bar memory a crossing needs must not leak between runs."""
    genome = StrategyGenome(
        name="crossing_twice",
        entry=ConditionTree(conditions=(Condition(left=FAST, op="cross_above", right=SLOW),)),
        exit=ConditionTree(conditions=(Condition(left=FAST, op="cross_below", right=SLOW),)),
        params={
            "fast": ParamSpec(kind="int", default=5, low=2, high=20),
            "slow": ParamSpec(kind="int", default=20, low=10, high=60),
        },
        warmup_bars=60,
    )
    source = compile_genome(genome)
    assert "previous is not None" in source
    bars = fixture_bars(300)
    strategy = build(source)
    first = positions(strategy, bars)
    second = positions(strategy, bars)
    np.testing.assert_allclose(first, second, rtol=0, atol=0)
    np.testing.assert_allclose(first, positions(build(source), bars), rtol=0, atol=0)
