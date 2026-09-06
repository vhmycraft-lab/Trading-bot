"""Rome's hidden training-environment randomisation (Project Rome sections 4-19). 🔒

The specification names twelve properties this system must be shown to have.
Nine of them are about the *selector* and are tested here; the three about
reachability — that neither strategy code nor an LLM can touch any of it — are in
``tests/unit/test_environment_isolation.py``, because they are statements about
the whole import graph rather than about this module.

Two tests here deserve a word about method. "Different generations can receive
different windows" and "consecutive generations are not forced to move
chronologically" are properties of a *random process*, so they are tested over a
sample large enough that a correct implementation passes with overwhelming
probability and the two failure modes the specification names — a constant window
and a rolling one — fail with certainty. The thresholds below are chosen against
those two alternatives, not tuned until the suite went green.
"""

from __future__ import annotations

import inspect
import itertools
from typing import Any

import pytest

from quantlab.core.config import EnvironmentSettings
from quantlab.core.environment import (
    ENVIRONMENT_MODEL_VERSION,
    SEED_BYTES,
    EnvironmentSelector,
    TrainingWindow,
    build_window_pool,
    reproduce,
    sampling_report,
)
from quantlab.core.errors import ConfigError
from quantlab.core.splits import SplitPolicy

HOUR = 3_600_000
EPOCH = 1_600_000_000_000

#: 4000 train bars, then validation, then an open-ended test partition.
POLICY = SplitPolicy(
    symbol="BTC/USDT",
    timeframe="1h",
    train_start_ts=EPOCH,
    train_end_ts=EPOCH + 3_999 * HOUR,
    val_start_ts=EPOCH + 4_024 * HOUR,
    val_end_ts=EPOCH + 4_999 * HOUR,
    test_start_ts=EPOCH + 5_024 * HOUR,
    test_end_ts=None,
    embargo_bars=24,
    dataset_id="ds-test",
)
SETTINGS = EnvironmentSettings()


def selector(settings: EnvironmentSettings = SETTINGS, **kwargs: Any) -> EnvironmentSelector:
    pool = build_window_pool(POLICY, settings)
    return EnvironmentSelector(
        pool,
        settings,
        universe=kwargs.pop("universe", ["BTC/USDT"]),
        dataset_version=kwargs.pop("dataset_version", "ds-test@1"),
        execution_model_version=kwargs.pop("execution_model_version", "exec/1"),
    )


# ---------------------------------------------------------------------------
# 9. selected windows are always valid
# ---------------------------------------------------------------------------
def test_the_pool_is_not_empty_and_every_window_is_inside_training() -> None:
    pool = build_window_pool(POLICY, SETTINGS)
    assert len(pool) > 1
    for window in pool.windows:
        assert window.start_ts >= POLICY.train_start_ts
        assert window.end_ts <= POLICY.train_end_ts
        assert window.start_ts < window.end_ts


def test_every_window_is_long_enough_to_mean_something() -> None:
    """Rome section 8: "Do not sample windows that are obviously too short"."""
    pool = build_window_pool(POLICY, SETTINGS)
    train_bars = (POLICY.train_end_ts - POLICY.train_start_ts) // HOUR + 1
    floor = max(SETTINGS.min_window_bars, int(train_bars * SETTINGS.window_min_fraction))
    assert min(window.n_bars(HOUR) for window in pool.windows) >= floor


def test_every_selected_window_comes_from_the_pool() -> None:
    """The draw is over an enumerated pool, never over arbitrary dates."""
    rome = selector()
    for index in range(200):
        assert rome.pool.contains(rome.select(index).window)


def test_a_training_segment_too_short_for_any_window_is_refused() -> None:
    """Fail closed (Rome section 37). Silently shortening the window would be an
    unannounced change to what "training" means."""
    tiny = SplitPolicy(
        symbol="BTC/USDT",
        timeframe="1h",
        train_start_ts=EPOCH,
        train_end_ts=EPOCH + 49 * HOUR,
        val_start_ts=EPOCH + 74 * HOUR,
        val_end_ts=EPOCH + 99 * HOUR,
        test_start_ts=EPOCH + 124 * HOUR,
        test_end_ts=None,
        embargo_bars=24,
    )
    with pytest.raises(ConfigError, match="shorter than the shortest permitted window"):
        build_window_pool(tiny, SETTINGS)


# ---------------------------------------------------------------------------
# 7 and 8. validation and final-holdout periods can never be selected
# ---------------------------------------------------------------------------
def test_no_pool_window_reaches_the_validation_segment() -> None:
    """Structural, not checked-after-the-fact: the pool is enumerated from the
    training segment, so there is no window to reject."""
    pool = build_window_pool(POLICY, SETTINGS)
    assert all(window.end_ts < POLICY.val_start_ts for window in pool.windows)


def test_no_pool_window_reaches_the_test_partition() -> None:
    pool = build_window_pool(POLICY, SETTINGS)
    assert all(window.end_ts < POLICY.test_start_ts for window in pool.windows)


def test_no_drawn_environment_ever_leaves_the_training_segment() -> None:
    """The enumeration argument again, but exercised through the draw, so a future
    selector that stopped using the pool would fail here rather than silently."""
    rome = selector()
    for index in range(500):
        window = rome.select(index).window
        assert POLICY.train_start_ts <= window.start_ts
        assert window.end_ts <= POLICY.train_end_ts
        assert window.end_ts < POLICY.val_start_ts < POLICY.test_start_ts


def test_the_pool_stays_inside_training_even_with_the_widest_settings() -> None:
    """The boundary case: a window allowed to be the whole training segment must
    end at its last bar and not one bar later."""
    settings = EnvironmentSettings(
        window_min_fraction=1.0, window_max_fraction=1.0, min_window_bars=2
    )
    pool = build_window_pool(POLICY, settings)
    assert {(w.start_ts, w.end_ts) for w in pool.windows} == {
        (POLICY.train_start_ts, POLICY.train_end_ts)
    }


# ---------------------------------------------------------------------------
# 1 and 2. independent, not chronological
# ---------------------------------------------------------------------------
def test_different_generations_receive_different_windows() -> None:
    """Rome section 17. A constant environment — the failure this exists to
    prevent — yields exactly one distinct window over any number of draws."""
    rome = selector()
    windows = {rome.select(index).window for index in range(100)}
    assert len(windows) > 20


def test_consecutive_generations_are_not_forced_to_move_chronologically() -> None:
    """Rome's third critical rule. A rolling window never steps backwards; over a
    hundred draws from a pool of hundreds, an independent selector steps backwards
    close to half the time. Anything above a handful separates the two decisively,
    and the assertion is set well below what independence produces so it is
    testing the *shape* rather than a tuned constant."""
    rome = selector()
    starts = [rome.select(index).window.start_ts for index in range(100)]
    backwards = sum(1 for a, b in itertools.pairwise(starts) if b < a)
    forwards = sum(1 for a, b in itertools.pairwise(starts) if b > a)
    assert backwards > 10, starts
    assert forwards > 10, starts


def test_the_sequence_of_windows_is_not_a_rolling_one() -> None:
    """Stated directly as well, because "some steps go backwards" would still pass
    for a mostly-rolling schedule with occasional jitter."""
    rome = selector()
    starts = [rome.select(index).window.start_ts for index in range(60)]
    assert starts != sorted(starts)
    assert starts != sorted(starts, reverse=True)


def test_the_same_generation_index_does_not_produce_the_same_environment() -> None:
    """The selection for generation *n* is not a function of *n*. If it were, the
    evolutionary algorithm could learn the schedule, and a resumed run would
    silently re-use the environment rather than record a fresh one."""
    rome = selector()
    first, second = rome.select(7), rome.select(7)
    assert first.seed_hex != second.seed_hex
    assert first.environment_id != second.environment_id


def test_two_selectors_do_not_share_a_sequence() -> None:
    """Rome section 35: restarting the application must not reproduce a sequence,
    and parallel workers must not collide. Neither selector carries state that
    could make them agree."""
    left = [env.seed_hex for env in (selector().select(i) for i in range(20))]
    right = [env.seed_hex for env in (selector().select(i) for i in range(20))]
    assert set(left).isdisjoint(right)


def test_the_draw_is_spread_across_the_pool_rather_than_clustered() -> None:
    """A selector that drew from a tenth of the pool would pass every test above.
    Five hundred draws from a pool of this size should touch a large fraction of
    it; a tenth of the pool cannot."""
    rome = selector()
    seen = {rome.select(index).window for index in range(500)}
    assert len(seen) > len(rome.pool) * 0.4, (len(seen), len(rome.pool))


# ---------------------------------------------------------------------------
# 3 and 4. nothing outside Rome can choose the window or the seed
# ---------------------------------------------------------------------------
def test_select_accepts_nothing_but_a_generation_index() -> None:
    """The API boundary is the enforcement (Rome section 48). A ``select`` that
    accepted a window, a seed or a capital — even optionally, even validated —
    would be one refactor away from honouring it, and no prompt instruction
    substitutes for the parameter not existing."""
    parameters = list(inspect.signature(EnvironmentSelector.select).parameters)
    assert parameters == ["self", "generation_index"]


def test_the_generation_index_is_recorded_but_does_not_steer_the_draw() -> None:
    """It labels the record. Two environments differing only in the index they
    were asked for must differ in everything, which is what shows the index is not
    an ingredient."""
    rome = selector()
    assert rome.select(0).generation_index == 0
    assert rome.select(999).generation_index == 999
    windows = {rome.select(0).window for _ in range(50)}
    assert len(windows) > 5


def test_the_seed_is_not_derived_from_anything_a_caller_supplies() -> None:
    """Rome section 5's list of forbidden seed sources, tested as one property:
    with every caller-visible input held fixed, the seed still changes."""
    rome = selector()
    seeds = {rome.select(3).seed_hex for _ in range(50)}
    assert len(seeds) == 50


# ---------------------------------------------------------------------------
# 10. the randomness is cryptographically secure
# ---------------------------------------------------------------------------
def test_the_seed_comes_from_the_operating_systems_csprng() -> None:
    """Observed at the call, not inferred from the import. ``secrets.token_bytes``
    is a thin wrapper over ``os.urandom``."""
    import quantlab.core.environment as module

    calls: list[int] = []
    real = module.secrets.token_bytes

    def spy(n: int) -> bytes:
        calls.append(n)
        return real(n)

    original = module.secrets.token_bytes
    module.secrets.token_bytes = spy  # type: ignore[assignment]
    try:
        selector().select(0)
    finally:
        module.secrets.token_bytes = original  # type: ignore[assignment]
    assert calls == [SEED_BYTES]
    assert SEED_BYTES >= 32


def test_the_module_does_not_reach_for_a_predictable_generator() -> None:
    """Rome section 5 forbids ``random()`` with a predictable seed here. A module
    that imported ``random`` at all would make the next contributor's shortcut
    available, so the import itself is what is asserted against."""
    import quantlab.core.environment as module

    source = inspect.getsource(module)
    assert "import random" not in source
    assert "numpy.random" not in source


def test_seeds_are_full_width_and_do_not_repeat() -> None:
    rome = selector()
    seeds = [rome.select(i).seed_hex for i in range(200)]
    assert len({*seeds}) == 200
    assert all(len(seed) == SEED_BYTES * 2 for seed in seeds)


# ---------------------------------------------------------------------------
# 11. a privileged audit can reproduce the environment
# ---------------------------------------------------------------------------
def test_an_environment_reproduces_exactly_from_its_recorded_seed() -> None:
    """Rome sections 19 and 24. Everything but the seed is redundant, so a
    reproduction that disagrees says the *derivation* changed and names the field."""
    rome = selector()
    original = rome.select(42)
    rebuilt = reproduce(
        original.audit_record(),
        pool=rome.pool,
        settings=SETTINGS,
        universe=["BTC/USDT"],
    )
    assert rebuilt == original
    assert rebuilt.audit_record() == original.audit_record()


def test_the_selector_reproduces_from_its_own_pool() -> None:
    rome = selector()
    original = rome.select(1)
    assert rome.reproduce(original.audit_record()) == original


def test_a_record_without_a_seed_cannot_be_reproduced() -> None:
    rome = selector()
    record = dict(rome.select(0).audit_record())
    record.pop("seed_hex")
    with pytest.raises(ConfigError, match="carries no seed"):
        reproduce(record, pool=rome.pool, settings=SETTINGS, universe=["BTC/USDT"])


def test_a_record_from_a_different_derivation_is_refused_not_approximated() -> None:
    """An audit that quietly rebuilt something else would be worse than one that
    failed."""
    rome = selector()
    record = dict(rome.select(0).audit_record())
    record["derivation_version"] = "env/0"
    with pytest.raises(ConfigError, match="different environment derivation"):
        reproduce(record, pool=rome.pool, settings=SETTINGS, universe=["BTC/USDT"])
    assert ENVIRONMENT_MODEL_VERSION == "env/1"


def test_the_environment_id_is_opaque() -> None:
    """Rome section 34: no ``BacktestEnvironment_2019_04_01_2020_08_14``. The id
    is a one-way function of the seed, so it can travel in logs and run rows
    without carrying the window back out with it."""
    environment = selector().select(0)
    assert environment.environment_id.isalnum()
    assert str(environment.window.start_ts) not in environment.environment_id
    assert str(environment.window.end_ts) not in environment.environment_id
    assert environment.seed_hex not in environment.environment_id
    assert len(environment.environment_id) == 16


# ---------------------------------------------------------------------------
# the realistic bands (Rome sections 9-13)
# ---------------------------------------------------------------------------
def test_capital_and_slippage_stay_inside_their_documented_bands() -> None:
    rome = selector()
    for index in range(300):
        environment = rome.select(index)
        assert SETTINGS.min_starting_capital <= environment.starting_capital
        assert environment.starting_capital <= SETTINGS.max_starting_capital
        assert SETTINGS.min_slippage_bps <= environment.slippage_bps
        assert environment.slippage_bps <= SETTINGS.max_slippage_bps


def test_capital_and_slippage_actually_vary() -> None:
    """Guards the band test above, which a constant would also pass."""
    rome = selector()
    environments = [rome.select(i) for i in range(100)]
    assert len({e.starting_capital for e in environments}) > 20
    assert len({e.slippage_bps for e in environments}) > 5


def test_no_field_of_the_environment_is_a_commission_or_an_execution_delay() -> None:
    """Rome sections 12 and 13 forbid randomising either. The strongest form of
    that rule is that the environment has nowhere to put one."""
    fields = set(selector().select(0).audit_record())
    assert not {name for name in fields if "fee" in name or "commission" in name}
    assert not {name for name in fields if "delay" in name or "latency" in name}


def test_an_environment_may_not_be_built_without_a_tradable_asset() -> None:
    with pytest.raises(ConfigError, match="at least one tradable asset"):
        selector(universe=[])


def test_the_asset_subset_is_drawn_from_the_configured_universe() -> None:
    """Rome section 15: real assets only. Nothing is invented; the subset is
    always a subset, and never empty."""
    universe = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
    settings = EnvironmentSettings(min_assets=2)
    rome = selector(settings, universe=universe)
    seen = set()
    for index in range(200):
        chosen = rome.select(index).asset_universe
        assert set(chosen) <= set(universe)
        assert len(chosen) >= 2
        assert list(chosen) == [name for name in universe if name in set(chosen)]
        seen.add(chosen)
    assert len(seen) > 1, "the asset subset never varied"


# ---------------------------------------------------------------------------
# 12. sampling statistics identify excessive reuse
# ---------------------------------------------------------------------------
def _pool() -> Any:
    return build_window_pool(POLICY, SETTINGS)


def test_repeated_use_of_one_period_is_reported() -> None:
    """Rome section 43's purpose, stated as a test: a campaign that hammered one
    window must be detectable from the record alone."""
    pool = _pool()
    hammered = pool.windows[0]
    report = sampling_report([hammered] * 50, pool, SETTINGS)
    assert report.flagged
    assert report.overused_buckets
    assert report.unsampled_buckets
    assert report.concentration > SETTINGS.overuse_factor


def test_an_even_spread_over_history_is_not_flagged() -> None:
    """Guards the test above: a detector that flagged everything would be useless.

    This is also what forced the baseline to be the pool rather than a flat share.
    Windows are contiguous, so uniform draws still cover the middle of history far
    more than its edges — against a flat line these four hundred honest draws
    measure about 1.6x and would be reported as concentrated.
    """
    pool = _pool()
    rome = selector()
    report = sampling_report([rome.select(i).window for i in range(400)], pool, SETTINGS)
    assert not report.overused_buckets, report.as_dict()
    assert report.concentration < SETTINGS.overuse_factor


def test_the_baseline_is_the_pool_and_not_a_flat_share() -> None:
    """Stated directly, because the test above would also pass for a flat baseline
    with a loose enough threshold. Under this pool's geometry the middle of
    history is genuinely expected to be seen more often than its edges, and the
    report must say so rather than call the difference over-use."""
    pool = _pool()
    report = sampling_report(list(pool.windows), pool, SETTINGS)
    expected = report.expected_shares
    flat = 1.0 / len(expected)
    assert max(expected.values()) > flat * 1.2, "this pool is not geometrically flat"
    # Drawing every window exactly once *is* the baseline, so nothing is over-used.
    assert report.overused_buckets == ()
    assert report.concentration == pytest.approx(1.0)


def test_exposure_is_counted_in_observations_not_in_windows() -> None:
    """Two disjoint windows drawn once each and one window drawn twice are very
    different exposures and identical *window* counts. Rome section 43 asks about
    the observations, so the report must distinguish them."""
    pool = _pool()
    first, last = pool.windows[0], pool.windows[-1]
    spread = sampling_report([first, last], pool, SETTINGS)
    repeated = sampling_report([first, first], pool, SETTINGS)
    assert len(spread.window_counts) == 2
    assert len(repeated.window_counts) == 1
    assert max(spread.observed_shares.values()) <= max(repeated.observed_shares.values())
    assert sum(spread.bucket_counts.values()) > 0


def test_a_report_over_nothing_says_nothing_was_sampled() -> None:
    """Accurate rather than vacuous: with no environments yet, every bucket is
    genuinely unsampled and the report says so."""
    pool = _pool()
    report = sampling_report([], pool, SETTINGS)
    assert report.n_environments == 0
    assert report.concentration == 0.0
    assert len(report.unsampled_buckets) == SETTINGS.n_history_buckets
    assert report.flagged


def test_the_report_is_json_serialisable_for_the_audit_record() -> None:
    import json

    document = sampling_report([_pool().windows[0]], _pool(), SETTINGS).as_dict()
    assert json.loads(json.dumps(document))["flagged"] is True


def test_window_ids_are_stable_and_distinguish_windows() -> None:
    left = TrainingWindow(start_ts=EPOCH, end_ts=EPOCH + 100 * HOUR)
    assert left.window_id == TrainingWindow(start_ts=EPOCH, end_ts=EPOCH + 100 * HOUR).window_id
    assert left.window_id != TrainingWindow(start_ts=EPOCH, end_ts=EPOCH + 101 * HOUR).window_id
