"""Every configured gate, swept across its threshold from both sides.

Two questions per gate, and the second is the one tonight's findings came from:

1. **Placement** — does the verdict flip exactly where the configuration says,
   and nowhere else? Each gate is evaluated at its threshold, one increment
   below and one increment above, and the flip is asserted against the *config
   value* rather than against a literal, so a threshold change moves these tests
   with it instead of breaking them.
2. **Degenerate input** — what does the gate do with ``None``, NaN, ±inf and
   zero-length input? Most of the exploits found in this codebase were
   degenerate-input handling, not threshold placement: NaN failed closed only by
   the accident that every NaN comparison is False, while ``+inf`` compared
   greater than any threshold and actively passed.

The sweep is written so that a gate reading a *different* input than it should,
or flipping a bar early, shows up as an off-by-one here rather than as a
surprising verdict on a real candidate.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quantlab.core.config import load_config
from quantlab.core.fitness import InnerFoldReport, compute_fitness
from quantlab.core.metrics import MetricSet
from quantlab.core.types import Trade
from quantlab.core.validation.gates import GateInputs, evaluate_gates

NAN = float("nan")
INF = float("inf")
NON_FINITE = (NAN, INF, -INF)


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def vs(config):
    return config.validation


@pytest.fixture(scope="module")
def fs(config):
    return config.evolution.fitness


def a_trade(pnl: float, pnl_pct: float, i: int = 0) -> Trade:
    return Trade(
        trade_no=i,
        side="long",
        entry_ts=i * 3_600_000,
        entry_px=100.0,
        exit_ts=(i + 1) * 3_600_000,
        exit_px=100.0,
        qty=1.0,
        fees=0.0,
        slippage_cost=0.0,
        pnl=pnl,
        pnl_pct=pnl_pct,
        bars_held=1,
        exit_reason="signal",
    )


def healthy(vs) -> dict:
    """A candidate that passes all seven gates, for one field to be perturbed."""
    return {
        "probe_passed": True,
        "metrics_train": MetricSet(n_trades=float(vs.min_trades_train + 50), sharpe=1.0),
        "metrics_val": MetricSet(
            n_trades=float(vs.min_trades_val + 50),
            sharpe=0.8,
            sortino=1.2,
            max_drawdown=0.2,
            net_return=0.3,
        ),
        "metrics_val_stressed": MetricSet(net_return=0.1),
        "permutation_p": vs.permutation.alpha / 2.0,
        "metrics_buyhold_val": MetricSet(sortino=0.5, max_drawdown=0.6, net_return=0.2),
        "position_frac_val": np.array([0.5]),
        "max_position_fraction": 1.0,
    }


def verdict(vs, **overrides) -> dict[str, bool]:
    report = evaluate_gates(GateInputs(**{**healthy(vs), **overrides}), vs)
    return {result.gate_id: result.passed for result in report.results}


def test_the_healthy_baseline_passes_every_gate(vs) -> None:
    """Or every sweep below would be measuring the wrong thing."""
    assert all(verdict(vs).values())


# ---------------------------------------------------------------------------
# placement: the flip is at the configured value, from both sides
# ---------------------------------------------------------------------------
def test_g_min_trades_flips_at_both_configured_counts(vs) -> None:
    """Both segments, independently. The gate is an AND, not an OR."""
    for at, field, other in (
        (vs.min_trades_val, "metrics_val", "metrics_train"),
        (vs.min_trades_train, "metrics_train", "metrics_val"),
    ):
        below = MetricSet(
            n_trades=float(at - 1), sharpe=0.8, sortino=1.2, max_drawdown=0.2, net_return=0.3
        )
        exact = MetricSet(
            n_trades=float(at), sharpe=0.8, sortino=1.2, max_drawdown=0.2, net_return=0.3
        )
        assert not verdict(vs, **{field: below})["G_MIN_TRADES"], f"{field} at {at - 1}"
        assert verdict(vs, **{field: exact})["G_MIN_TRADES"], f"{field} at {at} (threshold is >=)"
        assert other  # both are read; the healthy fixture supplies the other


def test_g_sanity_flips_at_the_configured_trade_size(vs) -> None:
    limit = vs.max_single_trade_pct
    step = limit * 1e-9
    assert verdict(vs, trades_val=(a_trade(1.0, limit - step),))["G_SANITY"]
    assert verdict(vs, trades_val=(a_trade(1.0, limit),))["G_SANITY"], "threshold is <="
    assert not verdict(vs, trades_val=(a_trade(1.0, limit + step),))["G_SANITY"]
    # sign is irrelevant: a -30% trade is as impossible as a +30% one
    assert not verdict(vs, trades_val=(a_trade(-1.0, -(limit + step)),))["G_SANITY"]


def test_g_sanity_flips_at_the_cap_plus_its_drift_allowance(vs) -> None:
    ceiling = 1.0 * (1.0 + vs.sanity_drift_allowance)
    assert verdict(vs, position_frac_val=np.array([ceiling]))["G_SANITY"], "threshold is <="
    assert not verdict(vs, position_frac_val=np.array([ceiling * 1.0001]))["G_SANITY"]
    # and the drift allowance is genuinely applied, not ignored
    assert verdict(vs, position_frac_val=np.array([1.0 + vs.sanity_drift_allowance / 2]))[
        "G_SANITY"
    ]


def test_g_cost_flips_at_zero_not_at_a_tolerance(vs) -> None:
    """Strictly positive: breaking even under stress is not surviving it."""
    assert not verdict(vs, metrics_val_stressed=MetricSet(net_return=-1e-12))["G_COST"]
    assert not verdict(vs, metrics_val_stressed=MetricSet(net_return=0.0))["G_COST"]
    assert verdict(vs, metrics_val_stressed=MetricSet(net_return=1e-12))["G_COST"]


def test_g_perm_flips_at_alpha(vs) -> None:
    alpha = vs.permutation.alpha
    assert verdict(vs, permutation_p=alpha - 1e-9)["G_PERM"]
    assert verdict(vs, permutation_p=alpha)["G_PERM"], "threshold is <="
    assert not verdict(vs, permutation_p=alpha + 1e-9)["G_PERM"]
    assert verdict(vs, permutation_p=0.0)["G_PERM"]
    assert not verdict(vs, permutation_p=1.0)["G_PERM"]


def test_g_degrade_flips_at_the_configured_share_of_train_sharpe(vs) -> None:
    train = 1.0
    floor = vs.degradation_min_ratio * train
    base = {"metrics_train": MetricSet(n_trades=float(vs.min_trades_train + 50), sharpe=train)}

    def at(sharpe_val: float) -> bool:
        metrics = MetricSet(
            n_trades=float(vs.min_trades_val + 50),
            sharpe=sharpe_val,
            sortino=1.2,
            max_drawdown=0.2,
            net_return=0.3,
        )
        return verdict(vs, **base, metrics_val=metrics)["G_DEGRADE"]

    assert not at(floor - 1e-9)
    assert at(floor), "threshold is >="
    assert at(floor + 1e-9)
    # the sign test is independent of the ratio and binds first
    assert not at(0.0)
    assert not at(-1e-9)


def test_g_bench_flips_at_the_buy_hold_sortino_and_at_half_its_drawdown(vs) -> None:
    baseline = MetricSet(sortino=0.5, max_drawdown=0.6, net_return=0.2)

    def val(**kw) -> MetricSet:
        return MetricSet(
            n_trades=float(vs.min_trades_val + 50),
            sharpe=0.8,
            **{"sortino": 0.0, "max_drawdown": 0.6, "net_return": 0.3, **kw},
        )

    # return limb: strictly greater
    assert not verdict(vs, metrics_buyhold_val=baseline, metrics_val=val(sortino=0.5))["G_BENCH"]
    assert verdict(vs, metrics_buyhold_val=baseline, metrics_val=val(sortino=0.5 + 1e-9))["G_BENCH"]
    # risk limb: strictly below half, and only while still profitable
    half = 0.5 * 0.6
    assert not verdict(vs, metrics_buyhold_val=baseline, metrics_val=val(max_drawdown=half))[
        "G_BENCH"
    ]
    assert verdict(vs, metrics_buyhold_val=baseline, metrics_val=val(max_drawdown=half - 1e-9))[
        "G_BENCH"
    ]
    assert not verdict(
        vs, metrics_buyhold_val=baseline, metrics_val=val(max_drawdown=half - 1e-9, net_return=0.0)
    )["G_BENCH"], "halving the drawdown while losing money is not beating buy-and-hold"


# ---------------------------------------------------------------------------
# placement: the fitness gates of section 13.3
# ---------------------------------------------------------------------------
def fitness_of(fs, metrics: MetricSet, trades) -> str | None:
    return compute_fitness(metrics, trades, InnerFoldReport(), None, fs).gate_failure


def a_scoreable_ledger(fs, n: int):
    """Enough winners for F_CONCENTRATION and p_removal to be satisfied."""
    return [a_trade(10.0, 0.001, i) for i in range(n)]


def test_f_trades_flips_at_the_configured_minimum(fs) -> None:
    at = fs.gates.min_trades
    for n, expected in ((at - 1, "F_TRADES"), (at, None)):
        trades = a_scoreable_ledger(fs, n)
        metrics = MetricSet(
            n_trades=float(n),
            expectancy_pct=0.001,
            max_drawdown=0.1,
            top5_profit_share=0.2,
            win_rate=0.6,
        )
        assert fitness_of(fs, metrics, trades) == expected, f"{n} trades"


def test_f_expectancy_flips_at_the_configured_minimum(fs) -> None:
    at = fs.gates.min_expectancy_pct
    trades = a_scoreable_ledger(fs, fs.gates.min_trades)

    def gate(expectancy: float | None) -> str | None:
        return fitness_of(
            fs,
            MetricSet(
                n_trades=float(fs.gates.min_trades),
                expectancy_pct=expectancy,
                max_drawdown=0.1,
                top5_profit_share=0.2,
                win_rate=0.6,
            ),
            trades,
        )

    assert gate(at) == "F_EXPECTANCY", "threshold is <=, so exactly at it fails"
    assert gate(at - 1e-9) == "F_EXPECTANCY"
    assert gate(at + 1e-9) is None
    assert gate(None) == "F_EXPECTANCY", "unmeasured expectancy is not a passing expectancy"


def test_f_drawdown_flips_at_the_configured_ceiling(fs) -> None:
    at = fs.gates.max_drawdown
    trades = a_scoreable_ledger(fs, fs.gates.min_trades)

    def gate(drawdown: float) -> str | None:
        return fitness_of(
            fs,
            MetricSet(
                n_trades=float(fs.gates.min_trades),
                expectancy_pct=0.001,
                max_drawdown=drawdown,
                top5_profit_share=0.2,
                win_rate=0.6,
            ),
            trades,
        )

    assert gate(at) is None, "threshold is >, so exactly at the ceiling passes"
    assert gate(at - 1e-9) is None
    assert gate(at + 1e-9) == "F_DRAWDOWN"


def test_f_concentration_flips_at_the_configured_retention(fs) -> None:
    at = fs.gates.min_retention_top1
    n = fs.gates.min_trades

    def gate(trades) -> str | None:
        return fitness_of(
            fs,
            MetricSet(
                n_trades=float(len(trades)),
                expectancy_pct=0.001,
                max_drawdown=0.1,
                top5_profit_share=0.2,
                win_rate=0.6,
            ),
            trades,
        )

    # one winner carrying everything: retention_1 == 0.0 == the threshold
    single = [a_trade(100.0, 0.01, 0)] + [a_trade(0.0, 0.0, i) for i in range(1, n)]
    assert gate(single) == "F_CONCENTRATION", "threshold is <=, so exactly at it fails"
    # two equal winners: retention_1 == 0.5, comfortably above
    assert at == 0.0
    spread = [a_trade(50.0, 0.01, 0), a_trade(50.0, 0.01, 1)] + [
        a_trade(0.0, 0.0, i) for i in range(2, n)
    ]
    assert gate(spread) != "F_CONCENTRATION"


# ---------------------------------------------------------------------------
# degenerate input: None, NaN, ±inf, zero-length
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "gate_id,field,value",
    [
        ("G_LEAK", "probe_passed", None),
        ("G_COST", "metrics_val_stressed", None),
        ("G_PERM", "permutation_p", None),
        ("G_BENCH", "metrics_buyhold_val", None),
    ],
)
def test_an_absent_measurement_fails_its_gate(vs, gate_id, field, value) -> None:
    """ "Not checked" and "checked and clean" are different states."""
    assert not verdict(vs, **{field: value})[gate_id]


@pytest.mark.parametrize("bad", NON_FINITE)
def test_a_non_finite_validation_sharpe_fails_g_degrade(vs, bad) -> None:
    """+inf used to pass: it compares greater than any floor."""
    metrics = MetricSet(
        n_trades=float(vs.min_trades_val + 50),
        sharpe=bad,
        sortino=1.2,
        max_drawdown=0.2,
        net_return=0.3,
    )
    assert not verdict(vs, metrics_val=metrics)["G_DEGRADE"]


@pytest.mark.parametrize("bad", NON_FINITE)
def test_a_non_finite_stressed_return_fails_g_cost(vs, bad) -> None:
    assert not verdict(vs, metrics_val_stressed=MetricSet(net_return=bad))["G_COST"]


@pytest.mark.parametrize("bad", NON_FINITE)
def test_a_non_finite_p_value_fails_g_perm(vs, bad) -> None:
    assert not verdict(vs, permutation_p=bad)["G_PERM"]


@pytest.mark.parametrize("bad", NON_FINITE)
def test_a_non_finite_sortino_cannot_beat_buy_and_hold_on_return(vs, bad) -> None:
    """The risk limb is disabled here so the return limb is what is tested."""
    metrics = MetricSet(
        n_trades=float(vs.min_trades_val + 50),
        sharpe=0.8,
        sortino=bad,
        max_drawdown=0.6,
        net_return=0.3,
    )
    assert not verdict(vs, metrics_val=metrics)["G_BENCH"]


@pytest.mark.parametrize("bad", NON_FINITE)
def test_a_non_finite_position_or_trade_fails_g_sanity(vs, bad) -> None:
    assert not verdict(vs, position_frac_val=np.array([bad]))["G_SANITY"]
    assert not verdict(vs, trades_val=(a_trade(1.0, bad),))["G_SANITY"]


def test_zero_length_input_never_passes_the_suite_as_a_whole(vs) -> None:
    """G_SANITY passes vacuously on nothing, which is why it is not the only gate.

    Recorded rather than changed: "no impossible trade" is true of no trades, and
    G_MIN_TRADES is what refuses an empty candidate. The assertion is that the
    *suite* still rejects, so the vacuous pass can never be load-bearing.
    """
    empty = evaluate_gates(GateInputs(), vs)
    assert empty["G_SANITY"].passed
    assert not empty.passed
    assert set(empty.failed) >= {"G_LEAK", "G_MIN_TRADES", "G_COST", "G_PERM", "G_BENCH"}


def test_every_gate_reports_the_threshold_it_used(vs) -> None:
    """A rejection that cannot be argued with is a rejection nobody can act on."""
    for result in evaluate_gates(GateInputs(**healthy(vs)), vs).results:
        assert result.reason, result.gate_id
        assert result.observed is not None or result.gate_id == "G_LEAK"


def test_no_gate_silently_accepts_a_nan_anywhere(vs) -> None:
    """A sweep, so a gate added later that compares a float is covered too."""
    fields = {
        "permutation_p": NAN,
        "metrics_val_stressed": MetricSet(net_return=NAN),
        "metrics_val": MetricSet(
            n_trades=50.0, sharpe=NAN, sortino=NAN, max_drawdown=NAN, net_return=NAN
        ),
    }
    for field, value in fields.items():
        results = verdict(vs, **{field: value})
        assert not all(results.values()), f"{field}=NaN passed every gate"


def test_gate_ids_and_reported_results_stay_in_step(vs) -> None:
    """A gate added to the table but not evaluated would be silently absent."""
    from quantlab.core.validation.gates import GATE_IDS

    reported = tuple(result.gate_id for result in evaluate_gates(GateInputs(), vs).results)
    assert reported == GATE_IDS


def test_a_finite_zero_is_not_treated_as_missing(vs) -> None:
    """0.0 is a measurement. Funnelling non-finite values into None must not
    also swallow legitimate zeros, which `or`-style defaults would."""
    assert verdict(vs, permutation_p=0.0)["G_PERM"]
    metrics = MetricSet(
        n_trades=float(vs.min_trades_val + 50),
        sharpe=0.8,
        sortino=1.2,
        max_drawdown=0.0,
        net_return=0.3,
    )
    assert verdict(vs, metrics_val=metrics)["G_BENCH"]
    assert math.isfinite(0.0)
