"""The sandbox child process (spec §21.3, INV-4, task T18).

Every test here spawns a real child. That is the point: a sandbox verified with
mocks is a sandbox nobody has run. The cost is a second or so per test, which is
the right trade for the one control standing between a search process that writes
code and the machine it writes it on.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantlab.adapters.engine.simple_bar import SimpleBarEngine
from quantlab.core.errors import (
    SandboxError,
    SandboxProtocolError,
    SandboxResourceLimit,
    SandboxTimeout,
)
from quantlab.core.types import BacktestConfig, BarFrame
from quantlab.sandbox.guards import STDLIB_ALLOWED, import_is_allowed
from quantlab.sandbox.protocol import (
    ENGINE_PACKAGE,
    PROTOCOL_VERSION,
    REQUEST_JSON,
    SandboxRequest,
    read_request,
    write_request,
)
from quantlab.sandbox.runner import CHILD_MODULE, SandboxLimits, SandboxRunner

ENGINE = {
    "engine_module": "quantlab.adapters.engine.simple_bar",
    "engine_class": "SimpleBarEngine",
}


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def make_bars(n: int = 200) -> BarFrame:
    """A deterministic, mildly cyclical series — enough to produce real trades."""
    index = np.arange(n)
    close = 100.0 + np.cumsum(np.sin(index / 9.0))
    frame = pd.DataFrame(
        {
            "ts_open": (index * 3_600_000).astype("int64"),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 10.0),
            "quote_volume": np.full(n, 1000.0),
            "trades": np.full(n, 5, dtype="int64"),
            "is_gap_filled": np.zeros(n, dtype=bool),
        }
    )
    return BarFrame(frame, symbol="BTCUSDT", timeframe="1h", dataset_id="fixture")


def strategy_source(body: str, *, imports: str = "", warmup: int = 5) -> str:
    """A minimal valid strategy whose ``on_bar`` body is ``body``."""
    return (
        "from quantlab.core.strategy import Context, ParamSpec, Signal, SignalKind\n"
        f"{imports}\n\n"
        "class Probe:\n"
        '    name = "probe"\n'
        '    version = "1"\n'
        '    style = "bar_loop"\n'
        '    params = {"n": ParamSpec(kind="int", default=5, low=2, high=50)}\n'
        f"    warmup_bars = {warmup}\n\n"
        "    def prepare(self, params):\n"
        "        self.p = dict(params)\n\n"
        "    def on_bar(self, ctx: Context) -> Signal:\n"
        f"{textwrap.indent(textwrap.dedent(body).strip(), '        ')}\n\n\n"
        "STRATEGY = Probe\n"
    )


CROSSOVER = strategy_source(
    """
    fast = ctx.ind("sma", n=self.p["n"])
    slow = ctx.ind("sma", n=self.p["n"] * 3)
    return Signal(SignalKind.LONG) if fast > slow else Signal(SignalKind.FLAT)
    """,
    warmup=20,
)

ALWAYS_LONG = strategy_source("return Signal(SignalKind.LONG)", warmup=0)


@pytest.fixture(scope="module")
def bars() -> BarFrame:
    return make_bars()


@pytest.fixture(scope="module")
def runner() -> SandboxRunner:
    return SandboxRunner()


# ---------------------------------------------------------------------------
# round trip: the sandbox must not change the answer
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def crossover_outcome(runner: SandboxRunner, bars: BarFrame):  # type: ignore[no-untyped-def]
    return runner.run(CROSSOVER, bars, **ENGINE)


def test_a_strategy_runs_and_comes_back(crossover_outcome) -> None:  # type: ignore[no-untyped-def]
    assert crossover_outcome.ok, crossover_outcome.stderr[-2000:]
    result = crossover_outcome.require()
    assert result.n_bars == 200
    assert len(result.trades) > 0
    assert result.engine_name == "simple_bar"
    assert result.engine_version == SimpleBarEngine.version


def test_the_sandbox_result_equals_an_in_process_run(bars: BarFrame, crossover_outcome) -> None:  # type: ignore[no-untyped-def]
    """A sandbox that changed the numbers would be worse than no sandbox: every
    stored run would depend on which side of the process boundary it was made."""
    namespace: dict[str, object] = {}
    exec(compile(CROSSOVER, "strategy.py", "exec"), namespace)
    strategy = namespace["STRATEGY"]()  # type: ignore[operator]
    direct = SimpleBarEngine().run(strategy, bars, {}, BacktestConfig())
    boxed = crossover_outcome.require()

    pd.testing.assert_series_equal(direct.equity, boxed.equity)
    pd.testing.assert_series_equal(direct.position_frac, boxed.position_frac)
    pd.testing.assert_series_equal(direct.signals, boxed.signals)
    assert direct.trades == boxed.trades
    assert direct.fills == boxed.fills
    assert direct.cost_summary == boxed.cost_summary
    assert direct.log == boxed.log
    assert direct.warmup_bars == boxed.warmup_bars
    assert direct.ruined == boxed.ruined


def test_two_sandbox_runs_are_identical(runner: SandboxRunner, bars: BarFrame) -> None:
    """INV-7 across the process boundary."""
    first = runner.run(CROSSOVER, bars, **ENGINE).require()
    second = runner.run(CROSSOVER, bars, **ENGINE).require()
    pd.testing.assert_series_equal(first.equity, second.equity)
    assert first.trades == second.trades


def test_params_and_config_cross_the_boundary(runner: SandboxRunner, bars: BarFrame) -> None:
    cheap = runner.run(ALWAYS_LONG, bars, config=BacktestConfig(fee_bps=0.0), **ENGINE).require()
    dear = runner.run(ALWAYS_LONG, bars, config=BacktestConfig(fee_bps=200.0), **ENGINE).require()
    assert dear.cost_summary["total_fees"] > cheap.cost_summary["total_fees"]

    fast = runner.run(CROSSOVER, bars, params={"n": 3}, **ENGINE).require()
    slow = runner.run(CROSSOVER, bars, params={"n": 12}, **ENGINE).require()
    assert fast.trades != slow.trades
    assert fast.final_equity != slow.final_equity


# ---------------------------------------------------------------------------
# resource limits
# ---------------------------------------------------------------------------
SPIN = strategy_source(
    """
    total = 0
    while True:
        total += 1
    return Signal(SignalKind.FLAT)
    """,
    warmup=0,
)

HOG = "import numpy as np\n" + strategy_source(
    """
        self.keep.append(np.ones(200_000_000, dtype="float64"))
        return Signal(SignalKind.FLAT)
        """,
    warmup=0,
).replace("self.p = dict(params)", "self.p = dict(params)\n        self.keep = []")


@pytest.mark.slow
def test_wall_clock_timeout_kills_the_child(bars: BarFrame) -> None:
    runner = SandboxRunner(limits=SandboxLimits(wall_clock_s=3, cpu_seconds=60))
    outcome = runner.run(SPIN, bars, **ENGINE)
    assert outcome.timed_out
    assert outcome.status == "timeout"
    assert not outcome.ok
    with pytest.raises(SandboxTimeout):
        outcome.require()


@pytest.mark.slow
def test_cpu_limit_kills_the_child(bars: BarFrame) -> None:
    """The wall clock is generous here, so only RLIMIT_CPU can end this run."""
    runner = SandboxRunner(limits=SandboxLimits(cpu_seconds=2, wall_clock_s=60))
    outcome = runner.run(SPIN, bars, **ENGINE)
    assert outcome.status == "resource_limit"
    assert not outcome.timed_out
    assert "CPU" in outcome.response.error_message
    with pytest.raises(SandboxResourceLimit):
        outcome.require()


@pytest.mark.slow
def test_memory_limit_refuses_the_allocation(bars: BarFrame) -> None:
    runner = SandboxRunner(limits=SandboxLimits(memory_mb=1024, wall_clock_s=60))
    outcome = runner.run(HOG, bars, **ENGINE)
    assert outcome.status == "resource_limit"
    with pytest.raises(SandboxResourceLimit):
        outcome.require()


def test_limits_must_be_positive() -> None:
    for field in ("cpu_seconds", "memory_mb", "wall_clock_s"):
        with pytest.raises(ValueError, match=field):
            SandboxLimits(**{field: 0})


# ---------------------------------------------------------------------------
# the run-time guards
# ---------------------------------------------------------------------------
def _guard_probe(body: str, *, imports: str = "") -> str:
    return strategy_source(body, imports=imports, warmup=0)


@pytest.mark.parametrize(
    "module",
    ["os", "sys", "socket", "subprocess", "pathlib", "quantlab.adapters.store.sqlite"],
)
def test_forbidden_import_at_run_time(runner: SandboxRunner, bars: BarFrame, module: str) -> None:
    """With the AST check switched off, the second layer must still hold.

    These sources would never pass §9.2. Running them anyway is the only way to
    show that the run-time guard is a control in its own right and not a comment.
    """
    source = _guard_probe(f"import {module}\nreturn Signal(SignalKind.FLAT)")
    outcome = runner.run(source, bars, ast_check=False, **ENGINE)
    assert outcome.status == "forbidden_import", outcome.stderr[-1500:]
    assert module in outcome.response.error_message
    with pytest.raises(SandboxError):
        outcome.require()


def test_the_ast_check_runs_inside_the_child_too(runner: SandboxRunner, bars: BarFrame) -> None:
    source = _guard_probe("import os\nreturn Signal(SignalKind.FLAT)")
    outcome = runner.run(source, bars, **ENGINE)
    assert outcome.status == "strategy_load"
    assert "E_FORBIDDEN_IMPORT" in outcome.response.error_codes


def test_allowed_imports_still_work(runner: SandboxRunner, bars: BarFrame) -> None:
    """The guard must not be so tight that an honest strategy cannot compute."""
    source = _guard_probe(
        """
        value = float(np.nanmean(np.asarray([1.0, 2.0, 3.0])))
        good = value > 0 and math.isfinite(value)
        return Signal(SignalKind.LONG) if good else Signal(SignalKind.FLAT)
        """,
        imports="import math\nimport numpy as np",
    )
    assert runner.run(source, bars, **ENGINE).ok


def test_sockets_raise_even_when_reached_through_a_live_reference() -> None:
    """A reference taken *before* the guard went up must raise too.

    Rebinding ``socket.socket`` alone would leave a library that had already
    captured the class holding a working one, so the class itself is patched.
    """
    program = (
        "import socket\n"
        "held_class = socket.socket\n"
        "from quantlab.sandbox.guards import NetworkBlocked, block_network\n"
        "block_network()\n"
        "calls = (held_class, socket.socket, socket.create_connection, socket.getaddrinfo)\n"
        "for call in calls:\n"
        "    try:\n"
        "        call()\n"
        "    except NetworkBlocked:\n"
        "        continue\n"
        "    raise SystemExit('network was reachable')\n"
        "print('blocked')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "blocked" in completed.stdout


@pytest.mark.parametrize(
    ("name", "allowed"),
    [
        ("numpy", True),
        ("numpy.linalg", True),
        ("math", True),
        ("__future__", True),
        ("quantlab.core.strategy", True),
        ("quantlab", False),
        ("quantlab.adapters", False),
        ("os", False),
        ("scipy", False),
    ],
)
def test_import_allow_list_semantics(name: str, allowed: bool) -> None:
    permitted = frozenset({"numpy", "quantlab.core.strategy"})
    assert import_is_allowed(name, permitted) is allowed


def test_the_stdlib_subset_excludes_everything_that_touches_the_world() -> None:
    dangerous = {
        "os",
        "sys",
        "socket",
        "subprocess",
        "pathlib",
        "shutil",
        "importlib",
        "pickle",
        "marshal",
        "ctypes",
        "threading",
        "multiprocessing",
        "asyncio",
        "inspect",
        "builtins",
        "time",
        "datetime",
        "random",
        "urllib",
        "http",
    }
    assert not (STDLIB_ALLOWED & dangerous)


# ---------------------------------------------------------------------------
# failure reporting
# ---------------------------------------------------------------------------
def test_a_strategy_that_raises_is_an_outcome_not_an_exception(
    runner: SandboxRunner, bars: BarFrame
) -> None:
    source = _guard_probe('raise ValueError("deliberate")')
    outcome = runner.run(source, bars, **ENGINE)
    assert outcome.status == "strategy_runtime"
    assert not outcome.ok
    assert outcome.result is None
    assert "deliberate" in outcome.response.error_message
    with pytest.raises(SandboxError, match="deliberate"):
        outcome.require()


def test_source_that_does_not_parse_fails_to_load(runner: SandboxRunner, bars: BarFrame) -> None:
    outcome = runner.run("class Broken:\n    params = {\n", bars, **ENGINE)
    assert outcome.status == "strategy_load"


def test_a_module_without_strategy_fails_to_load(runner: SandboxRunner, bars: BarFrame) -> None:
    outcome = runner.run("X = 1\n", bars, ast_check=False, **ENGINE)
    assert outcome.status == "strategy_load"
    assert "STRATEGY" in outcome.response.error_message


def test_the_childs_traceback_reaches_the_parent(runner: SandboxRunner, bars: BarFrame) -> None:
    """Without it, debugging a failed candidate means re-running it by hand."""
    outcome = runner.run(_guard_probe('raise ValueError("deliberate")'), bars, **ENGINE)
    assert "deliberate" in outcome.stderr
    assert "Traceback" in outcome.stderr


# ---------------------------------------------------------------------------
# isolation of the child process
# ---------------------------------------------------------------------------
def test_the_child_environment_carries_nothing_but_path_and_determinism(
    runner: SandboxRunner,
) -> None:
    env = SandboxRunner()._child_env()
    assert set(env) == {
        "PATH",
        "PYTHONHASHSEED",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONNOUSERSITE",
        "MPLBACKEND",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    }
    assert env["PYTHONHASHSEED"] == "0"
    for name in ("GLM_API_KEY", "BINANCE_API_KEY", "BINANCE_API_SECRET", "HOME", "PYTHONPATH"):
        assert name not in env


def test_the_child_runs_in_isolated_mode() -> None:
    """``-I`` is what makes PYTHONPATH, user site-packages and sitecustomize
    irrelevant to the interpreter that executes untrusted code."""
    source = Path("src/quantlab/sandbox/runner.py").read_text(encoding="utf-8")
    assert '"-I"' in source
    assert CHILD_MODULE == "quantlab.sandbox.child_main"


def test_the_exchange_directory_is_removed_after_a_run(
    runner: SandboxRunner, bars: BarFrame
) -> None:
    """A strategy must not be able to leave anything behind for the next one."""
    before = set(Path("/tmp").glob("quantlab-sandbox-*"))
    runner.run(ALWAYS_LONG, bars, **ENGINE)
    assert set(Path("/tmp").glob("quantlab-sandbox-*")) == before


def test_only_an_engine_adapter_may_be_named(bars: BarFrame) -> None:
    """Otherwise the request is an "import this module in the sandbox" instruction,
    and the store and the LLM client are one string away."""
    with pytest.raises(ValueError, match=ENGINE_PACKAGE):
        SandboxRunner().run(
            ALWAYS_LONG,
            bars,
            engine_module="quantlab.adapters.store.sqlite",
            engine_class="SqliteStore",
        )


# ---------------------------------------------------------------------------
# the wire format
# ---------------------------------------------------------------------------
def test_request_round_trips_through_the_directory(tmp_path: Path, bars: BarFrame) -> None:
    request = SandboxRequest(
        symbol=bars.symbol,
        timeframe=bars.timeframe,
        dataset_id=bars.dataset_id,
        params={"n": 7},
        config=BacktestConfig(fee_bps=3.0),
        allowed_imports=("numpy",),
        **ENGINE,
    )
    write_request(tmp_path, request, bars, ALWAYS_LONG)
    back, back_bars, source = read_request(tmp_path)

    assert back == request
    assert source == ALWAYS_LONG
    pd.testing.assert_frame_equal(back_bars.to_pandas(), bars.to_pandas())
    assert back_bars.symbol == bars.symbol
    assert back_bars.dataset_id == bars.dataset_id


def test_a_protocol_version_mismatch_fails_closed(tmp_path: Path, bars: BarFrame) -> None:
    request = SandboxRequest(symbol="BTCUSDT", timeframe="1h", **ENGINE)
    write_request(tmp_path, request, bars, ALWAYS_LONG)
    payload = (tmp_path / REQUEST_JSON).read_text(encoding="utf-8")
    (tmp_path / REQUEST_JSON).write_text(
        payload.replace(f'"protocol_version": {PROTOCOL_VERSION}', '"protocol_version": 99'),
        encoding="utf-8",
    )
    with pytest.raises(SandboxProtocolError, match="protocol version"):
        read_request(tmp_path)


def test_an_unreadable_request_fails_closed(tmp_path: Path) -> None:
    (tmp_path / REQUEST_JSON).write_text("{not json", encoding="utf-8")
    with pytest.raises(SandboxProtocolError):
        read_request(tmp_path)


def test_the_wire_format_carries_no_pickle() -> None:
    """Data crosses, never a program: a pickle stream authored by the untrusted
    side would execute on load.

    ``test_architecture.py`` exempts this package from the INV-4 module ban so the
    child can use ``importlib`` and ``compile``. That exemption covers ``pickle``
    too, which is why the ban is re-stated here where it still has to hold.
    """
    banned = {"pickle", "marshal", "shelve", "dill", "cloudpickle"}
    for path in sorted(Path("src/quantlab/sandbox").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not banned & {a.name.split(".")[0] for a in node.names}, path
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in banned, path
