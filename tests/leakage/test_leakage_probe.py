"""Every leaky fixture is detected; every honest one passes (spec section 14.2, T19).

This is the acceptance criterion for T19 stated as code. The fixtures are the
four leak shapes the specification names, plus honest strategies that use the
*same* pandas idioms causally — because a probe that rejects an expanding mean
along with a full-series aggregation would be abandoned within a week, and an
abandoned check protects nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quantlab.adapters.engine.simple_bar import SimpleBarEngine
from quantlab.core.errors import LeakageDetected, ValidationError_
from quantlab.core.types import BacktestConfig, BarFrame
from quantlab.core.validation.leakage import (
    DEFAULT_CUT_FRACTIONS,
    MIN_PROBE_BARS,
    REVERSED_TAIL_FRACTION,
    SIGNAL_LAG,
    TAIL_MODES,
    Divergence,
    ProbeResult,
    default_cut_points,
    perturbed_tail,
    probe_evaluations,
    require_causal,
    reversed_tail,
    truncation_probe,
)
from tests.leakage.conftest import (
    BASELINES,
    HONEST_DIR,
    LEAKY_DIR,
    default_params,
    load_strategy,
    make_bars,
)

#: The four shapes spec section 14.2 requires. Named individually so a fixture
#: that stops being detected fails by name rather than shrinking a count.
LEAKY = ("future_close.py", "forward_breakout.py", "centred_mean.py", "full_series_zscore.py")
HONEST = (
    "sma_cross_vectorized.py",
    "momentum_vectorized.py",
    "expanding_mean.py",
    "always_flat.py",
)


def _probe(path: Path, bars: BarFrame, **kwargs: object) -> ProbeResult:
    strategy = load_strategy(path)
    return truncation_probe(
        strategy,
        bars,
        default_params(strategy),
        engine=SimpleBarEngine(),
        **kwargs,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# the acceptance criterion
# ---------------------------------------------------------------------------
def test_the_four_required_leak_shapes_are_all_present() -> None:
    """Spec section 14.2 names them; the directory must hold exactly those."""
    assert sorted(p.name for p in LEAKY_DIR.glob("*.py")) == sorted(LEAKY)


@pytest.mark.parametrize("name", LEAKY)
def test_every_leaky_fixture_is_detected(name: str, bars: BarFrame) -> None:
    result = _probe(LEAKY_DIR / name, bars)
    assert not result.passed, f"{name} escaped the probe"
    assert result.divergences
    assert "LEAKAGE_DETECTED" in result.describe()


@pytest.mark.parametrize("name", HONEST)
def test_every_honest_fixture_passes(name: str, bars: BarFrame) -> None:
    result = _probe(HONEST_DIR / name, bars)
    assert result.passed, result.describe()
    assert result.divergences == ()
    assert "causal" in result.describe()


@pytest.mark.parametrize(
    "name", ["sma_cross.py", "buy_and_hold.py", "rsi_reversion.py", "random_entry.py"]
)
def test_the_shipped_baselines_are_causal(name: str, bars: BarFrame) -> None:
    """``bar_loop`` strategies are protected structurally by ``BarWindow``. The
    probe agreeing is the empirical confirmation of the structural claim."""
    assert _probe(BASELINES / name, bars).passed


@pytest.mark.parametrize("seed", [1, 2, 3, 11, 42, 99, 2024])
def test_the_verdicts_do_not_depend_on_which_bars_were_used(seed: int) -> None:
    """A probe that only worked on one series would be a coincidence, not a check.

    ``random_entry`` is excluded: it draws from ``ctx.rng``, which is seeded from
    the run seed and the bar index, so truncating the segment legitimately changes
    later draws. That is not leakage, and section 14.2's comparison does not claim
    it is — the bars it compares are identical in both runs.
    """
    bars = make_bars(seed=seed)
    for name in LEAKY:
        assert not _probe(LEAKY_DIR / name, bars).passed, f"{name} escaped on seed {seed}"
    for name in HONEST:
        assert _probe(HONEST_DIR / name, bars).passed, f"{name} was falsely rejected on {seed}"


# ---------------------------------------------------------------------------
# what each mechanism catches
# ---------------------------------------------------------------------------
def test_a_full_series_aggregation_is_caught_by_truncation(bars: BarFrame) -> None:
    """Its every value depends on every bar, so shortening the segment moves all
    of them. This is the shape truncation exists for."""
    result = _probe(LEAKY_DIR / "full_series_zscore.py", bars)
    assert any(divergence.probe == "truncation" for divergence in result.divergences)


def test_a_one_bar_peek_is_caught_only_at_the_replacement_boundary(bars: BarFrame) -> None:
    """``close.shift(-1)`` consumes exactly the bar of slack section 9.1 grants a
    vectorised strategy, so a prefix computes the same values and truncation is
    blind to it. Only replacing the future can catch it, and only at the boundary."""
    result = _probe(LEAKY_DIR / "future_close.py", bars)
    assert not result.passed
    assert {divergence.probe for divergence in result.divergences} == {"reversed_tail"}
    assert {divergence.bar_index for divergence in result.divergences} == {result.tail_start}


def test_the_scaled_tails_are_what_make_the_one_bar_peek_certain(bars: BarFrame) -> None:
    """Reversal alone leaves it to chance whether a boolean flips at that one bar.

    Scaling the replaced prices far up and far down means the peeked comparison
    must come out differently in at least one of them.
    """
    strategy = load_strategy(LEAKY_DIR / "future_close.py")
    params = default_params(strategy)
    engine = SimpleBarEngine()
    settings = BacktestConfig()

    def evaluate(segment: BarFrame) -> object:
        return engine.run(strategy, segment, params, settings)

    reversal_only = probe_evaluations(
        evaluate,  # type: ignore[arg-type]
        bars,
        style="vectorized",
        warmup_bars=strategy.warmup_bars,
        cut_points=(200,),
        tail_fraction=REVERSED_TAIL_FRACTION,
    )
    modes = {divergence.mode for divergence in reversal_only.divergences}
    assert modes & {"scaled_up", "scaled_down"}, (
        "the scaled replacements must be what catches a one-bar peek"
    )


def test_a_centred_window_is_caught_by_both_mechanisms(bars: BarFrame) -> None:
    result = _probe(LEAKY_DIR / "centred_mean.py", bars)
    assert {divergence.probe for divergence in result.divergences} == {
        "truncation",
        "reversed_tail",
    }


def test_a_divergence_inside_the_declared_warmup_reports_under_declaration() -> None:
    """Section 14.2: a divergence at a bar the strategy called warm-up additionally
    means ``warmup_bars`` is too short for what the strategy actually needs.

    Constructed rather than provoked, because ``SimpleBarEngine`` never calls the
    strategy before ``warmup_bars`` and records FLAT there (section 8.4) — so with
    *this* engine the warm-up region is identical between runs by construction and
    real divergences always land after it. The classification still has to be right:
    an engine that did consult the strategy during warm-up, or a strategy whose
    declared warm-up is shorter than its indicators need, is exactly what this flag
    is for, and it must not go quietly.
    """
    result = ProbeResult(
        passed=False,
        n_bars=400,
        style="vectorized",
        warmup_bars=50,
        cut_points=(200,),
        tail_start=360,
        divergences=(
            Divergence(
                probe="truncation",
                at=200,
                series="signal",
                bar_index=17,
                reference="long",
                observed="flat",
                in_warmup=True,
            ),
        ),
    )
    assert result.warmup_under_declared
    assert "under-declared" in result.describe()
    assert "inside declared warm-up" in str(result.divergences[0])


def test_real_divergences_are_classified_by_where_they_fall(bars: BarFrame) -> None:
    result = _probe(LEAKY_DIR / "full_series_zscore.py", bars)
    assert not result.passed
    assert all(
        divergence.in_warmup == (divergence.bar_index < result.warmup_bars)
        for divergence in result.divergences
    )


# ---------------------------------------------------------------------------
# the probe's own mechanics
# ---------------------------------------------------------------------------
def test_cut_points_are_the_five_the_spec_names(bars: BarFrame) -> None:
    assert DEFAULT_CUT_FRACTIONS == (0.20, 0.35, 0.50, 0.65, 0.80)
    assert default_cut_points(400) == (80, 140, 200, 260, 320)
    assert default_cut_points(400) == default_cut_points(400)


def test_cut_points_stay_inside_the_segment() -> None:
    for n in (MIN_PROBE_BARS, 37, 101, 1000):
        points = default_cut_points(n)
        assert points == tuple(sorted(set(points)))
        assert all(0 < point < n for point in points)


def test_a_segment_too_short_to_probe_is_refused() -> None:
    """Fail closed: a probe that cannot separate head from tail has checked
    nothing, and reporting a pass would be a lie."""
    with pytest.raises(ValidationError_, match="too short"):
        default_cut_points(MIN_PROBE_BARS - 1)
    with pytest.raises(ValidationError_, match="too short"):
        reversed_tail(make_bars(n=MIN_PROBE_BARS - 1))


def test_the_reversed_tail_keeps_the_frame_valid(bars: BarFrame) -> None:
    replaced, start = reversed_tail(bars)
    assert 0 < start < bars.n_bars
    assert replaced.n_bars == bars.n_bars
    # Timestamps are untouched: a different future, not a broken frame.
    assert list(replaced.ts_open) == list(bars.ts_open)
    assert list(replaced.close[:start]) == list(bars.close[:start])
    assert list(replaced.close[start:]) == list(bars.close[start:])[::-1]


@pytest.mark.parametrize("mode", TAIL_MODES)
def test_every_tail_mode_leaves_the_head_untouched(bars: BarFrame, mode: str) -> None:
    """The head is the part the probe compares; if a replacement changed it the
    probe would be measuring its own perturbation."""
    replaced, start = perturbed_tail(bars, mode=mode)  # type: ignore[arg-type]
    assert list(replaced.close[:start]) == list(bars.close[:start])
    assert list(replaced.open[:start]) == list(bars.open[:start])
    assert replaced.n_bars == bars.n_bars


def test_the_scaled_tails_move_the_prices_in_opposite_directions(bars: BarFrame) -> None:
    up, start = perturbed_tail(bars, mode="scaled_up")
    down, _ = perturbed_tail(bars, mode="scaled_down")
    assert float(up.close[start]) > float(bars.close[start])
    assert float(down.close[start]) < float(bars.close[start])


def test_the_signal_lag_matches_the_engines_shift() -> None:
    """Section 9.1 shifts a vectorised strategy by one bar and a ``bar_loop`` one
    by none. Getting this wrong hides a one-bar peek or rejects honest code."""
    assert SIGNAL_LAG == {"bar_loop": 0, "vectorized": 1}


def test_an_unknown_style_is_refused(bars: BarFrame) -> None:
    with pytest.raises(ValidationError_, match="unknown strategy style"):
        probe_evaluations(lambda _b: None, bars, style="quantum")  # type: ignore[arg-type,return-value]


def test_a_cut_point_outside_the_segment_is_refused(bars: BarFrame) -> None:
    for bad in (0, bars.n_bars, bars.n_bars + 1, -3):
        with pytest.raises(ValidationError_, match=r"cut point outside|no cut points"):
            probe_evaluations(lambda _b: None, bars, cut_points=(bad,))  # type: ignore[arg-type,return-value]


def test_the_probe_counts_its_own_evaluations(bars: BarFrame) -> None:
    """One full run, one per cut point, one per tail replacement. The count is what
    a caller budgets against when the executor is a sandboxed child process."""
    result = _probe(HONEST_DIR / "always_flat.py", bars)
    assert result.n_evaluations == 1 + len(result.cut_points) + len(TAIL_MODES)


def test_the_probe_never_looks_beyond_the_bars_it_was_given(bars: BarFrame) -> None:
    """INV-5 in miniature: the probe truncates and perturbs, it never extends.

    Every segment it evaluates is a prefix of, or the same length as, what it was
    handed, so a probe run on the training partition cannot reach the validation
    or test partitions.
    """
    seen: list[int] = []
    strategy = load_strategy(HONEST_DIR / "always_flat.py")
    engine = SimpleBarEngine()

    def evaluate(segment: BarFrame) -> object:
        seen.append(segment.n_bars)
        assert segment.n_bars <= bars.n_bars
        assert int(segment.ts_open[0]) == int(bars.ts_open[0])
        assert int(segment.ts_open[-1]) <= int(bars.ts_open[-1])
        return engine.run(strategy, segment, default_params(strategy), BacktestConfig())

    probe_evaluations(
        evaluate,  # type: ignore[arg-type]
        bars,
        style="vectorized",
        warmup_bars=strategy.warmup_bars,
    )
    assert max(seen) == bars.n_bars


def test_two_runs_of_the_probe_agree(bars: BarFrame) -> None:
    """INV-7: the probe is part of the load path, so its verdict has to be stable."""
    first = _probe(LEAKY_DIR / "centred_mean.py", bars)
    second = _probe(LEAKY_DIR / "centred_mean.py", bars)
    assert first == second


# ---------------------------------------------------------------------------
# require_causal
# ---------------------------------------------------------------------------
def test_require_causal_passes_a_causal_strategy(bars: BarFrame) -> None:
    result = _probe(HONEST_DIR / "momentum_vectorized.py", bars)
    assert require_causal(result) is result


def test_require_causal_raises_with_the_evidence(bars: BarFrame) -> None:
    result = _probe(LEAKY_DIR / "centred_mean.py", bars)
    with pytest.raises(LeakageDetected) as excinfo:
        require_causal(result, strategy_id="abc123")
    assert excinfo.value.probe is result
    assert excinfo.value.context["strategy_id"] == "abc123"
    assert "LEAKAGE_DETECTED" in str(excinfo.value)


def test_a_leakage_verdict_lists_its_divergences(bars: BarFrame) -> None:
    result = _probe(LEAKY_DIR / "full_series_zscore.py", bars)
    text = result.describe(limit=3)
    assert text.count("\n") <= 6
    assert "and" in text and "more" in text
    assert str(result.divergences[0]) in text
