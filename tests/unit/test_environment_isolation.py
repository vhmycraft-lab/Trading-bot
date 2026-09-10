"""Nothing outside Rome can reach the hidden environment (Project Rome 6, 20-22, 36). 🔒

Rome section 48 states the design principle these tests exist to hold: the system
must not depend on an LLM or a strategy *choosing* to behave. The properties
below are therefore about what is structurally reachable, not about what any
particular caller happens to do.

Three groups:

* **the strategy** cannot obtain the seed, the window as metadata, the capital or
  the slippage — because the wire format that carries work into the sandbox has
  no field for them and forbids extra ones;
* **the LLM** cannot specify any of them — because no prompt-facing module can
  even import the selector, and the selector's API has no parameter to carry a
  preference;
* **the attack attempts** of Rome section 36 all fail.

Rome section 7 is honest about the limit, and so is this file: a strategy that
receives timestamped candles can read the dates on them. What is hidden is the
*selection* — which environment Rome chose and why — and none of these tests
pretends otherwise.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest
from tests.unit.test_environment import POLICY, SETTINGS

from quantlab.core.config import EnvironmentSettings
from quantlab.core.environment import EnvironmentSelector, build_window_pool
from quantlab.core.strategy import Context
from quantlab.core.types import BacktestConfig
from quantlab.sandbox.protocol import SandboxRequest

SRC = Path(__file__).resolve().parents[2] / "src" / "quantlab"

#: Every name that would reveal the hidden selection if a strategy could read it.
SECRET_NAMES = (
    "seed_hex",
    "environment_id",
    "training_window",
    "window_start_ts",
    "window_end_ts",
    "starting_capital",
    "slippage_bps",
    "scenario_parameters",
    "asset_universe",
    "pool_id",
)


def _python_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def _module_name(path: Path) -> str:
    relative = path.relative_to(SRC.parent).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _imported(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module)
    return names


def selector() -> EnvironmentSelector:
    return EnvironmentSelector(
        build_window_pool(POLICY, SETTINGS),
        SETTINGS,
        universe=["BTC/USDT"],
        dataset_version="ds",
        execution_model_version="exec/1",
    )


# ---------------------------------------------------------------------------
# 6. strategy code cannot access the hidden environment configuration
# ---------------------------------------------------------------------------
def test_the_sandbox_wire_format_has_no_field_for_the_environment() -> None:
    """The strongest available form of "the strategy is not told": the request
    that carries work into the child process has nowhere to put it.

    ``extra="forbid"`` is what makes this a guarantee rather than an observation —
    a future caller cannot smuggle one in as an unrecognised key, because the
    model would reject the request.
    """
    assert SandboxRequest.model_config["extra"] == "forbid"
    fields = set(SandboxRequest.model_fields)
    assert not (fields & set(SECRET_NAMES))


def test_the_backtest_settings_carry_no_environment_metadata() -> None:
    """``BacktestConfig`` is the one configuration object that *does* reach the
    child, because the engine needs it. It carries the resulting equity and
    slippage — which are ordinary execution settings a backtest always has — and
    no label saying they were sampled, no seed and no window."""
    assert BacktestConfig.model_config["extra"] == "forbid"
    fields = set(BacktestConfig.model_fields)
    assert not (fields & set(SECRET_NAMES))
    assert "initial_equity" in fields and "slippage" in fields


def test_a_strategy_context_exposes_nothing_about_the_environment() -> None:
    """What a strategy actually holds at each bar. Rome section 10 permits it to
    see its own balance — that is normal trading information — and nothing here
    says how that balance was chosen."""
    visible = set(Context.__slots__)
    assert not (visible & set(SECRET_NAMES))
    assert {"equity", "cash", "bars", "params", "position", "i"} <= visible


def test_no_sandbox_module_can_import_the_environment_selector() -> None:
    """The sandbox layer is where untrusted code runs. A module there that could
    import the selector would put the seed one attribute lookup from the
    strategy, whatever the wire format said."""
    offenders = [
        _module_name(path)
        for path in _python_files()
        if _module_name(path).startswith("quantlab.sandbox")
        and "quantlab.core.environment" in _imported(path)
    ]
    assert offenders == []


def test_the_selector_is_reachable_from_only_the_layers_that_must_hold_it() -> None:
    """Rome section 44's separation, as an import rule.

    The orchestrator (`cli/`) constructs it, one service (`experiments/`) drives
    it, and the loop consumes only the value type. Everything else — strategies,
    the sandbox, reporting, research — has no path to it at all.
    """
    holders = sorted(
        _module_name(path)
        for path in _python_files()
        if "quantlab.core.environment" in _imported(path)
    )
    assert holders == [
        "quantlab.cli.evolve",
        "quantlab.evolution.loop",
        "quantlab.experiments.environment",
    ], holders


def test_the_loop_receives_the_value_type_and_not_the_selector() -> None:
    """``evolution/loop.py`` is the one place in the search that touches an
    environment, and it is handed bars, settings and an opaque id. It has no
    selector, no seed and no window, so there is nothing there for a guided
    mutation or a prompt builder to pick up."""
    from quantlab.core.environment import GenerationEnvironment

    fields = set(GenerationEnvironment.__dataclass_fields__)
    assert fields == {"bars", "config", "environment_id"}
    source = inspect.getsource(__import__("quantlab.evolution.loop", fromlist=["x"]))
    assert "seed_hex" not in source
    assert "EnvironmentSelector" not in source


# ---------------------------------------------------------------------------
# 3 and 4. the LLM cannot specify a window or a seed
# ---------------------------------------------------------------------------
def test_the_selector_has_no_parameter_that_could_carry_a_preference() -> None:
    """Rome section 48: enforce through code and API boundaries, not through a
    prompt asking the LLM not to. Every public entry point is checked, so adding
    one that took a window would fail here."""
    for name in ("select", "reproduce"):
        parameters = set(inspect.signature(getattr(EnvironmentSelector, name)).parameters)
        assert not (parameters & {"window", "seed", "seed_hex", "capital", "slippage"})
    assert set(inspect.signature(EnvironmentSelector.select).parameters) == {
        "self",
        "generation_index",
    }


def test_environment_settings_are_configuration_and_not_a_per_run_request() -> None:
    """The bands are set once, in configuration a person controls, and the draw is
    uniform inside them. There is no path by which a *request* — from an LLM, a
    tool call, or a strategy — narrows a band for one run and so chooses a value.
    """
    rome = selector()
    forbidden = {"window", "seed", "environment", "capital", "slippage_value"}
    assert not (set(inspect.signature(rome.select).parameters) & forbidden)
    tight = EnvironmentSettings(min_slippage_bps=5.0, max_slippage_bps=5.0)
    assert tight.min_slippage_bps == tight.max_slippage_bps  # configuration may pin a band
    with pytest.raises(ValueError, match="max_slippage_bps"):
        EnvironmentSettings(min_slippage_bps=9.0, max_slippage_bps=1.0)


def test_no_prompt_or_research_module_can_reach_the_environment() -> None:
    """INV-6 and INV-11 in Rome's terms: nothing that builds an LLM prompt may
    import the module that holds the seed.

    This was vacuous until ``research/`` existed, and the note saying so is
    replaced here by something that cannot go quietly vacuous again: the scan
    asserts it actually *looked* at both packages first. A guard over an empty
    set passes for the same reason a guard over a clean one does, and only one
    of those is evidence.
    """
    scanned = [
        _module_name(path)
        for path in _python_files()
        if _module_name(path).startswith(("quantlab.research", "quantlab.reporting"))
    ]
    assert any(name.startswith("quantlab.research") for name in scanned), scanned
    assert any(name.startswith("quantlab.reporting") for name in scanned), scanned

    offenders = [
        _module_name(path)
        for path in _python_files()
        if _module_name(path).startswith(("quantlab.research", "quantlab.reporting"))
        and "quantlab.core.environment" in _imported(path)
    ]
    assert offenders == []


# ---------------------------------------------------------------------------
# Rome section 36: attempt to break it
# ---------------------------------------------------------------------------
def test_a_caller_cannot_reach_the_seed_through_the_generation_environment() -> None:
    """The object the loop holds is frozen and slotted, so there is no attribute
    to add one to either."""
    from quantlab.core.environment import GenerationEnvironment

    rome = selector()
    environment = rome.select(0)
    handed = GenerationEnvironment(
        bars=None,  # type: ignore[arg-type]
        config=BacktestConfig(),
        environment_id=environment.environment_id,
    )
    assert not hasattr(handed, "seed_hex")
    with pytest.raises((AttributeError, TypeError)):
        handed.seed_hex = environment.seed_hex  # type: ignore[attr-defined]


def test_the_environment_id_does_not_leak_the_seed_or_the_window() -> None:
    """Rome section 34: opaque identifiers. Recovering the seed from the id means
    inverting SHA-256, and the id is the only part of the environment that
    travels outward."""
    rome = selector()
    for index in range(50):
        environment = rome.select(index)
        identifier = environment.environment_id
        assert environment.seed_hex[:8] not in identifier
        assert str(environment.window.start_ts) not in identifier
        assert str(int(environment.starting_capital)) not in identifier


def test_repeating_a_request_does_not_reveal_the_previous_answer() -> None:
    """A selector that returned a cached environment for a repeated index would
    let a caller learn the environment by asking twice and comparing."""
    rome = selector()
    assert rome.select(4).seed_hex != rome.select(4).seed_hex


def test_a_frozen_environment_cannot_be_edited_after_the_fact() -> None:
    """Rome section 25: the audit record is not rewritable in memory either, so
    nothing can hand the store a doctored copy of what it drew."""
    environment = selector().select(0)
    with pytest.raises((AttributeError, TypeError)):
        environment.starting_capital = 1.0  # type: ignore[misc]


def test_the_audit_record_is_the_only_thing_that_carries_the_seed() -> None:
    """It is named so a reader of a call site can see what is being handed over.
    A method called ``to_dict`` would be reached for by accident."""
    environment = selector().select(0)
    record = environment.audit_record()
    assert record["seed_hex"] == environment.seed_hex
    assert not hasattr(environment, "to_dict")
    assert not hasattr(environment, "as_dict")


def test_no_module_outside_the_audit_path_mentions_the_seed_field() -> None:
    """A grep-shaped guard, deliberately. ``seed_hex`` reaching a report, a
    prompt, a log formatter or an artifact writer is exactly the indirect leak
    Rome section 34 warns about, and the cheapest reliable detector is that the
    string does not appear where it should not."""
    allowed = {
        "quantlab.core.environment",
        "quantlab.experiments.environment",
        "quantlab.adapters.store.models",
        "quantlab.adapters.store.sqlite",
    }
    offenders = [
        _module_name(path)
        for path in _python_files()
        if "seed_hex" in path.read_text(encoding="utf-8") and _module_name(path) not in allowed
    ]
    assert offenders == [], offenders


def test_every_allowed_module_actually_needs_its_exemption() -> None:
    """Guards the test above from both directions.

    A typo in the allow-list would silently widen it, and an entry for a module
    that no longer mentions the seed would leave a permission nobody is using —
    the way an allow-list rots into permitting more than anyone intended.
    """
    holders = {
        _module_name(path)
        for path in _python_files()
        if "seed_hex" in path.read_text(encoding="utf-8")
    }
    assert holders == {
        "quantlab.core.environment",
        "quantlab.experiments.environment",
        "quantlab.adapters.store.models",
        "quantlab.adapters.store.sqlite",
    }, holders


def test_a_strategys_own_rng_seed_is_not_the_environment_seed(tmp_path: Any) -> None:
    """``Context.rng`` exists so a strategy can be stochastic reproducibly, and it
    is seeded from ``BacktestConfig.seed`` — the run's determinism seed, which is
    configuration, not the environment's secret. Confusing the two would hand the
    secret to every strategy that asked for a random number."""
    del tmp_path
    environment = selector().select(0)
    config = BacktestConfig(seed=7)
    assert config.seed == 7
    assert str(config.seed) != environment.seed_hex
    assert "seed_hex" not in set(BacktestConfig.model_fields)
