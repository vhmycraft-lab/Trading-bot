"""Rome's per-generation environment service, as the evolution loop consumes it.

Project Rome sections 4-19. :mod:`quantlab.core.environment` decides *what* an
environment is and draws one; this module is the part that has to happen in the
right order and be written down:

1. a generation already in the store is **replayed** under the environment it was
   recorded with, never re-drawn;
2. a new generation gets a fresh draw from the CSPRNG;
3. the draw is **recorded before the generation runs**;
4. only then are the bars sliced and the backtest settings derived.

Step 1 is what preserves the resume invariant. A resumed run must be
candidate-for-candidate identical to the run it is resuming, and candidates
depend on the fitness they were scored with, which depends on the environment
they were evaluated in. Re-drawing would produce a different search wearing the
same run id.

Step 3 is the same rule the lockbox follows and for the same reason: a record
written after the work is missing exactly the runs that crashed, which are the
ones an auditor most wants to reconstruct.

Nothing here reaches a strategy or an LLM. The provider hands the loop bars, a
:class:`~quantlab.core.types.BacktestConfig` and an opaque id; the window bounds,
the seed, the capital and the slippage go to the audit table and nowhere else.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from quantlab.core.environment import (
    EnvironmentSelector,
    GenerationEnvironment,
    TrainingEnvironment,
    TrainingWindow,
)
from quantlab.core.errors import ConfigError
from quantlab.core.types import BacktestConfig, BarFrame, SlippageConfig
from quantlab.ports.store import ExperimentStore

__all__ = ["GenerationEnvironmentProvider", "apply_environment", "record_to_mapping"]


def apply_environment(base: BacktestConfig, environment: TrainingEnvironment) -> BacktestConfig:
    """The backtest settings this environment produces.

    Exactly two fields move: ``initial_equity`` and the slippage model's
    ``fixed_bps``. Rome section 12 forbids randomising commission and section 13
    forbids randomising execution timing, so ``fee_bps``, ``fill_rule`` and every
    sizing rule are carried over from the configured base untouched.

    The slippage model itself is not switched either. Varying *how* slippage is
    computed between generations would be varying the execution model, which
    Rome section 39 says must be versioned rather than sampled; what varies is
    the level inside the venue's documented band.
    """
    return base.model_copy(
        update={
            "initial_equity": environment.starting_capital,
            "slippage": SlippageConfig.model_validate(
                {**base.slippage.model_dump(), "fixed_bps": environment.slippage_bps}
            ),
        }
    )


def record_to_mapping(row: Any) -> dict[str, object]:
    """An audit row in the shape :func:`~quantlab.core.environment.reproduce` reads.

    A translation rather than a shared type: ``core`` must not know what a
    database row looks like (INV-8), and the reproduction function should be
    usable against a JSON export as readily as against SQLite.
    """
    return {
        "environment_id": row.environment_id,
        "generation_index": row.gen_index,
        "seed_hex": row.seed_hex,
        "dataset_version": row.dataset_version,
        "execution_model_version": row.execution_model_version,
        "derivation_version": row.derivation_version,
    }


class GenerationEnvironmentProvider:
    """Selects, records and resolves one environment per generation.

    Constructed by the CLI, which is the only layer allowed to hold both a store
    adapter and market data. The loop sees only ``__call__``.
    """

    __slots__ = ("_bars", "_base", "_evolution_id", "_selector", "_store", "_used")

    def __init__(
        self,
        *,
        selector: EnvironmentSelector,
        store: ExperimentStore,
        evolution_id: str,
        bars_train: BarFrame,
        base_config: BacktestConfig,
    ) -> None:
        self._selector = selector
        self._store = store
        self._evolution_id = evolution_id
        self._bars = bars_train
        self._base = base_config
        self._used: list[TrainingWindow] = []

    def reproduce(self, row: Any) -> TrainingEnvironment:
        """Rebuild a recorded environment from its audit row. Privileged.

        Public because auditing is a first-class use of this object, not an
        internal detail: Rome section 24 requires that a future researcher be able
        to reconstruct how a result was produced.
        """
        return self._reproduce(row)

    @property
    def pool(self) -> Any:
        """The pool every draw came from, for the run's sampling report."""
        return self._selector.pool

    @property
    def windows_used(self) -> tuple[TrainingWindow, ...]:
        """Every window this run has been evaluated on, in the order drawn.

        What :func:`~quantlab.core.environment.sampling_report` is given at the
        end of a run, so a campaign that concentrated on one part of history says
        so in its own report rather than only in a later audit.
        """
        return tuple(self._used)

    def __call__(self, gen_index: int) -> GenerationEnvironment:
        environment = self._resolve(gen_index)
        self._used.append(environment.window)
        window = self._bars.slice(environment.window.start_ts, environment.window.end_ts)
        if window.n_bars == 0:
            raise ConfigError(
                "the selected training window resolved to no bars; the dataset and "
                "the split policy disagree about what the training segment holds",
                environment_id=environment.environment_id,
            )
        return GenerationEnvironment(
            bars=window,
            config=apply_environment(self._base, environment),
            environment_id=environment.environment_id,
        )

    def _resolve(self, gen_index: int) -> TrainingEnvironment:
        """The recorded environment for this generation, or a freshly drawn one.

        Reading first is what makes a resume a replay. The record is written
        before the caller does anything with the result, so a generation that
        crashes still leaves the environment it crashed under behind it.
        """
        existing = self._store.find_training_environment(self._evolution_id, gen_index)
        if existing is not None:
            return self._reproduce(existing)

        environment = self._selector.select(gen_index)
        self._store.record_training_environment(
            evolution_id=self._evolution_id,
            environment=environment.audit_record(),
        )
        return environment

    def _reproduce(self, row: Any) -> TrainingEnvironment:
        """Rebuild a recorded environment and check it against what was stored.

        The check is not ceremony. The reproduction is derived from the seed
        alone, so if it disagrees with the row the derivation has changed since
        the run was recorded, and continuing would evaluate the resumed
        generations on different history while claiming to replay them.
        """
        rebuilt = self._selector.reproduce(record_to_mapping(row))
        _assert_matches(rebuilt, row)
        return rebuilt


def _assert_matches(rebuilt: TrainingEnvironment, row: Any) -> None:
    stored: Mapping[str, object] = {
        "window_start_ts": int(row.window_start_ts),
        "window_end_ts": int(row.window_end_ts),
        "starting_capital": float(row.starting_capital),
        "slippage_bps": float(row.slippage_bps),
    }
    actual: Mapping[str, object] = {
        "window_start_ts": rebuilt.window.start_ts,
        "window_end_ts": rebuilt.window.end_ts,
        "starting_capital": rebuilt.starting_capital,
        "slippage_bps": rebuilt.slippage_bps,
    }
    differing = sorted(name for name, value in stored.items() if actual[name] != value)
    if differing:
        raise ConfigError(
            "a recorded environment did not reproduce from its seed; replaying this "
            "run would evaluate it on different history than it was recorded with",
            environment_id=str(row.environment_id),
            fields=differing,
        )
