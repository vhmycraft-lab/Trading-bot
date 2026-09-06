"""The validation pipeline, end to end (spec sections 14.1-14.4, task T31). 🔒

Section 22 names three cases and this file is built around them:

* an over-parameterised curve-fit fixture must be ``REJECT``;
* ``buy_and_hold`` must not be ``CANDIDATE``;
* an honest strategy on a synthetic trend must pass the gates.

Plus the two bookkeeping rules that make the budget real: a family's validation
touches are incremented per validation, and the family freezes at twenty.

The evidence is assembled here rather than produced by a live search, and that is
the point of the split: everything expensive belongs to the caller, and the
verdict logic can then be stated against a written-down picture. The three cases
below *are* those pictures, each describing a strategy whose problem is named in
its own docstring.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine

from quantlab.adapters.store.sqlite import SqliteExperimentStore, make_session_factory
from quantlab.core.config import ValidationSettings
from quantlab.core.errors import StrategyError
from quantlab.core.metrics import MetricSet
from quantlab.core.splits import SplitPolicy
from quantlab.core.types import Trade
from quantlab.core.validation.pipeline import Evidence, judge, require_open_family, validate
from quantlab.core.validation.sensitivity import SensitivityReport

SETTINGS = ValidationSettings()
HOUR = 3_600_000
EPOCH = 1_600_000_000_000


def trade(
    no: int, pnl: float, *, pnl_pct: float | None = None, exit_ts: int | None = None
) -> Trade:
    stamp = exit_ts if exit_ts is not None else EPOCH + no * 24 * HOUR
    return Trade(
        trade_no=no,
        side="long",
        entry_ts=stamp - HOUR,
        entry_px=100.0,
        exit_ts=stamp,
        exit_px=100.0 + pnl,
        qty=1.0,
        fees=0.0,
        slippage_cost=0.0,
        pnl=pnl,
        pnl_pct=pnl / 100.0 if pnl_pct is None else pnl_pct,
        bars_held=24,
        exit_reason="signal",
    )


def spread_ledger(n: int = 60, pnl: float = 5.0) -> list[Trade]:
    """Trades spread across a year, so no single quarter carries the profit."""
    return [
        trade(index, pnl if index % 3 else -pnl / 2, exit_ts=EPOCH + index * 6 * 24 * HOUR)
        for index in range(1, n + 1)
    ]


# ---------------------------------------------------------------------------
# the three cases section 22 names
# ---------------------------------------------------------------------------
def honest_trend_strategy() -> Evidence:
    """A strategy that found something and can show its work.

    Clears every gate: causal, enough trades on both segments, no impossible
    trade or position, profitable at doubled costs, distinguishable from its own
    permutations, holding up out of sample, and beating buy-and-hold.
    """
    ledger = spread_ledger()
    return Evidence(
        probe_passed=True,
        metrics_train=MetricSet(n_trades=220.0, sharpe=1.3, sortino=1.6),
        metrics_val=MetricSet(
            n_trades=60.0,
            sharpe=0.95,
            sortino=1.25,
            net_return=0.22,
            max_drawdown=0.11,
            top5_profit_share=0.18,
        ),
        metrics_val_stressed=MetricSet(net_return=0.08),
        metrics_buyhold_val=MetricSet(sortino=0.45, max_drawdown=0.38, net_return=0.14),
        trades_train=[trade(i, 3.0) for i in range(1, 11)],
        trades_val=ledger,
        position_frac_val=[0.0, 0.4, 0.9, 0.6],
        max_position_fraction=1.0,
        deflated_sharpe=0.98,
        pbo=0.12,
        permutation_p=0.004,
        sensitivity=SensitivityReport(median_drop=0.08, n_neighbours=24),
        wfe=0.85,
        profitable_oos_share=0.75,
        param_cv={"fast": 0.12, "slow": 0.09},
        random_entry_sortinos=[0.02 * index for index in range(50)],
        n_free_params=2,
        logic_lines=35,
        evolution_evaluations=180,
        inner_oos_component=0.72,
        n_trials_accounted=180,
    )


def curve_fit_strategy() -> Evidence:
    """The over-parameterised fixture: excellent in-sample, nothing out of it.

    Fourteen free parameters, a deflated Sharpe the size of the search cannot
    support, an in-sample winner that lands below median out of sample, a
    permutation test it cannot beat, a neighbourhood that collapses, profit
    carried by five trades and confined to one quarter, and a walk-forward
    efficiency near zero. It fails gates *and* scores off the top of the scale.
    """
    concentrated = [
        trade(1, 400.0, exit_ts=EPOCH + 10 * 24 * HOUR),
        *[trade(index, 2.0, exit_ts=EPOCH + 11 * 24 * HOUR) for index in range(2, 32)],
        *[trade(index, -3.0, exit_ts=EPOCH + 200 * 24 * HOUR) for index in range(32, 62)],
    ]
    return Evidence(
        probe_passed=True,
        metrics_train=MetricSet(n_trades=400.0, sharpe=3.2, sortino=4.0),
        metrics_val=MetricSet(
            n_trades=61.0,
            sharpe=0.12,
            sortino=0.08,
            net_return=0.03,
            max_drawdown=0.42,
            top5_profit_share=0.95,
        ),
        metrics_val_stressed=MetricSet(net_return=-0.04),
        metrics_buyhold_val=MetricSet(sortino=0.9, max_drawdown=0.30, net_return=0.25),
        trades_train=[trade(i, 3.0) for i in range(1, 11)],
        trades_val=concentrated,
        position_frac_val=[0.0, 0.8, 1.0],
        max_position_fraction=1.0,
        deflated_sharpe=0.20,
        pbo=0.78,
        permutation_p=0.44,
        sensitivity=SensitivityReport(median_drop=0.85, n_neighbours=24),
        wfe=0.12,
        profitable_oos_share=0.25,
        param_cv={"a": 1.9, "b": 2.4},
        random_entry_sortinos=[0.5 + 0.02 * index for index in range(50)],
        n_free_params=14,
        logic_lines=280,
        evolution_evaluations=4_200,
        inner_oos_component=0.11,
        n_trials_accounted=4_200,
    )


def buy_and_hold_strategy() -> Evidence:
    """The benchmark cannot beat itself.

    ``G_BENCH`` compares against buy-and-hold on the validation segment, so
    buy-and-hold ties on Sortino and does not halve its own drawdown. It also
    trades once, which ``G_MIN_TRADES`` refuses.
    """
    baseline = MetricSet(n_trades=1.0, sharpe=0.6, sortino=0.7, net_return=0.30, max_drawdown=0.35)
    return Evidence(
        probe_passed=True,
        metrics_train=MetricSet(n_trades=1.0, sharpe=0.8, sortino=0.9),
        metrics_val=baseline,
        metrics_val_stressed=MetricSet(net_return=0.29),
        metrics_buyhold_val=baseline,
        trades_val=[trade(1, 3_000.0)],
        position_frac_val=[1.0, 1.0, 1.0],
        max_position_fraction=1.0,
        deflated_sharpe=0.9,
        permutation_p=0.2,
        n_free_params=0,
        logic_lines=12,
    )


def test_an_honest_strategy_passes_every_gate() -> None:
    """Section 22's third criterion."""
    report = judge(honest_trend_strategy(), SETTINGS)
    assert report.gates.passed, report.gates.failed
    assert report.verdict == "CANDIDATE"
    assert report.overfit_score <= SETTINGS.score.candidate_max
    assert report.unmeasured == ()


def test_the_curve_fit_fixture_is_rejected() -> None:
    """Section 22's first criterion — and it fails on both halves at once, which
    is what an over-parameterised strategy looks like."""
    report = judge(curve_fit_strategy(), SETTINGS)
    assert report.verdict == "REJECT"
    assert not report.gates.passed
    assert {"G_COST", "G_DEGRADE", "G_PERM", "G_BENCH"} <= set(report.gates.failed)
    assert report.overfit_score > SETTINGS.score.weak_max


def test_the_curve_fit_fixture_would_be_rejected_on_its_score_alone() -> None:
    """Guards the case above against passing only because a gate happened to
    fail: the soft checks alone must also refuse it."""
    from quantlab.core.validation.gates import GateReport, GateResult

    passing = GateReport(results=(GateResult(gate_id="G_LEAK", passed=True),))
    from quantlab.core.validation.score import ScoreInputs, score_strategy

    evidence = curve_fit_strategy()
    report = score_strategy(
        ScoreInputs(
            deflated_sharpe=evidence.deflated_sharpe,
            pbo=evidence.pbo,
            permutation_p=evidence.permutation_p,
            sensitivity=evidence.sensitivity,
            metrics_val=evidence.metrics_val,
            wfe=evidence.wfe,
            profitable_oos_share=evidence.profitable_oos_share,
            param_cv=evidence.param_cv,
            n_free_params=evidence.n_free_params,
            logic_lines=evidence.logic_lines,
            trades_val=evidence.trades_val,
            random_entry_sortinos=evidence.random_entry_sortinos,
            removal=evidence.removal(),
            evolution_evaluations=evidence.evolution_evaluations,
            inner_oos_component=evidence.inner_oos_component,
        ),
        passing,
        SETTINGS,
    )
    assert report.verdict == "REJECT"
    assert report.overfit_score > SETTINGS.score.weak_max


def test_buy_and_hold_is_not_a_candidate() -> None:
    """Section 22's second criterion. The benchmark cannot beat itself."""
    report = judge(buy_and_hold_strategy(), SETTINGS)
    assert report.verdict != "CANDIDATE"
    assert "G_BENCH" in report.gates.failed
    assert "G_MIN_TRADES" in report.gates.failed


# ---------------------------------------------------------------------------
# the budget
# ---------------------------------------------------------------------------
@pytest.fixture
def store(db_engine: Engine) -> SqliteExperimentStore:
    return SqliteExperimentStore(make_session_factory(db_engine), now_ms=lambda: 1_700_000_000_000)


@pytest.fixture
def registered(store: SqliteExperimentStore) -> tuple[SqliteExperimentStore, str, str, str]:
    """A family, a strategy version and a split, ready to validate against."""
    store.get_or_create_dataset(
        dataset_id="d0",
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1h",
        start_ts=EPOCH,
        end_ts=EPOCH + 999 * HOUR,
        n_bars=1_000,
        manifest_json="{}",
    )
    split = store.get_or_create_split(
        SplitPolicy(
            symbol="BTC/USDT",
            timeframe="1h",
            train_start_ts=EPOCH,
            train_end_ts=EPOCH + 500 * HOUR,
            embargo_bars=0,
            val_start_ts=EPOCH + 501 * HOUR,
            val_end_ts=EPOCH + 800 * HOUR,
            test_start_ts=EPOCH + 801 * HOUR,
            test_end_ts=None,
            source_json="{}",
            dataset_id="d0",
        )
    )
    family = store.create_family(name="trend", origin="human")
    store.add_strategy_version(
        strategy_id="s0",
        family_id=family.family_id,
        code_path="strategies/generated/s0.py",
        code_sha256="a" * 64,
        class_name="Trend",
        param_schema_json="{}",
        style="bar_loop",
        author="human",
        logic_lines=35,
    )
    return store, "s0", family.family_id, split.split_id


def run_validation(
    registered: tuple[SqliteExperimentStore, str, str, str],
    evidence: Evidence,
    counter: Any,
    settings: ValidationSettings = SETTINGS,
) -> Any:
    store, strategy_id, family_id, split_id = registered
    return validate(
        store,
        strategy_id=strategy_id,
        family_id=family_id,
        split_id=split_id,
        params_json='{"fast":10}',
        evidence=evidence,
        settings=settings,
        verdict_id=lambda: f"v{next(counter):04d}",
    )


def test_a_validation_charges_the_family_one_touch(
    registered: tuple[SqliteExperimentStore, str, str, str],
) -> None:
    """Section 14.1 step 8. Every look at held-out data costs some of its power
    to say anything, and a budget that could be ignored is not a budget."""
    store, _strategy_id, family_id, _split_id = registered
    counter = itertools.count()

    assert store.find_family(family_id).validation_touches == 0
    outcome = run_validation(registered, honest_trend_strategy(), counter)
    assert outcome.validation_touches == 1
    assert store.find_family(family_id).validation_touches == 1

    run_validation(registered, honest_trend_strategy(), counter)
    run_validation(registered, honest_trend_strategy(), counter)
    assert store.find_family(family_id).validation_touches == 3


def test_a_family_freezes_at_its_limit(
    registered: tuple[SqliteExperimentStore, str, str, str],
) -> None:
    """Section 14.1 step 8's freeze, and step 0's refusal after it. The
    twenty-first validation cannot happen rather than merely being noticed."""
    store, _strategy_id, family_id, _split_id = registered
    counter = itertools.count()
    limit = SETTINGS.family_max_validation_touches

    for index in range(limit):
        outcome = run_validation(registered, honest_trend_strategy(), counter)
        assert outcome.family_frozen is (index == limit - 1)

    assert store.find_family(family_id).status == "frozen"
    assert store.find_family(family_id).validation_touches == limit

    with pytest.raises(StrategyError, match="not open for validation"):
        run_validation(registered, honest_trend_strategy(), counter)
    assert store.find_family(family_id).validation_touches == limit


def test_the_freeze_limit_follows_the_configuration(
    registered: tuple[SqliteExperimentStore, str, str, str],
) -> None:
    store, _strategy_id, family_id, _split_id = registered
    counter = itertools.count()
    settings = ValidationSettings(family_max_validation_touches=2)

    run_validation(registered, honest_trend_strategy(), counter, settings)
    assert store.find_family(family_id).status == "open"
    run_validation(registered, honest_trend_strategy(), counter, settings)
    assert store.find_family(family_id).status == "frozen"


def test_a_closed_family_is_refused_before_anything_is_charged(
    registered: tuple[SqliteExperimentStore, str, str, str],
) -> None:
    """A closed family failed the lockbox (section 14.6). Refusing here means a
    lockbox failure cannot be walked back by re-validating."""
    store, _strategy_id, family_id, _split_id = registered
    store.set_family_status(family_id, "closed")

    with pytest.raises(StrategyError, match="not open for validation"):
        run_validation(registered, honest_trend_strategy(), itertools.count())
    assert store.find_family(family_id).validation_touches == 0


def test_an_unknown_family_is_refused() -> None:
    class Empty:
        def find_family(self, family_id: str) -> None:
            return None

    with pytest.raises(StrategyError, match="no such strategy family"):
        validate(
            Empty(),
            strategy_id="s0",
            family_id="nobody",
            split_id="sp0",
            params_json="{}",
            evidence=Evidence(),
            settings=SETTINGS,
            verdict_id=lambda: "v0",
        )


def test_the_open_family_check_is_reachable_on_its_own() -> None:
    assert require_open_family("open") == "open"
    for status in ("frozen", "closed"):
        with pytest.raises(StrategyError, match="not open for validation"):
            require_open_family(status)


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------
def test_the_verdict_is_recorded_with_everything_behind_it(
    registered: tuple[SqliteExperimentStore, str, str, str],
) -> None:
    """Section 14.1 step 9. A verdict read a year later must be interpretable
    against the rules that produced it."""
    store, strategy_id, _family_id, split_id = registered
    outcome = run_validation(registered, honest_trend_strategy(), itertools.count())

    rows = store.verdicts_for(strategy_id)
    assert len(rows) == 1
    row = rows[0]
    assert row.verdict == outcome.verdict == "CANDIDATE"
    assert row.split_id == split_id
    assert row.overfit_score == outcome.overfit_score
    assert row.n_trials_accounted == 180

    gates = json.loads(row.hard_gates_json)
    checks = json.loads(row.soft_checks_json)
    thresholds = json.loads(row.thresholds_json)
    assert len(gates) == 7
    assert len(checks) == 11
    assert thresholds["dsr_threshold"] == SETTINGS.dsr_threshold
    assert thresholds["min_trades_val"] == SETTINGS.min_trades_val


def test_re_validating_writes_a_new_verdict_rather_than_revising_one(
    registered: tuple[SqliteExperimentStore, str, str, str],
) -> None:
    """A verdict is a statement about what was known when it was reached; the
    two together are the history of how the platform's opinion changed."""
    store, strategy_id, _family_id, _split_id = registered
    counter = itertools.count()

    run_validation(registered, honest_trend_strategy(), counter)
    run_validation(registered, curve_fit_strategy(), counter)

    rows = store.verdicts_for(strategy_id)
    assert [row.verdict for row in rows] == ["CANDIDATE", "REJECT"]
    assert len({row.verdict_id for row in rows}) == 2


def test_a_verdict_never_leaks_back_into_the_search() -> None:
    """Phase G judges; phase F searches. The import rule is enforced by
    ``tests/unit/test_architecture.py``; this states the intent where a reader
    of the pipeline will see it."""
    root = Path(__file__).resolve().parents[2]
    source = (root / "src" / "quantlab" / "core" / "validation" / "pipeline.py").read_text(
        encoding="utf-8"
    )
    assert "quantlab.evolution" not in source
    assert "quantlab.optimize" not in source
