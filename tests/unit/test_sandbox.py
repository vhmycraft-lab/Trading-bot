"""The sandbox child process (spec §21.3, INV-4, task T18).

Every test here spawns a real child. That is the point: a sandbox verified with
mocks is a sandbox nobody has run. The cost is a second or so per test, which is
the right trade for the one control standing between a search process that writes
code and the machine it writes it on.
"""

from __future__ import annotations

import ast
import builtins
import shutil
import subprocess
import sys
import tempfile
import textwrap
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from quantlab.adapters.engine.simple_bar import SimpleBarEngine
from quantlab.core.errors import (
    SandboxError,
    SandboxProtocolError,
    SandboxResourceLimit,
    SandboxTimeout,
    StrategyLoadError,
    StrategyRuntimeError,
    StrategySafetyError,
)
from quantlab.core.types import BacktestConfig, BarFrame
from quantlab.sandbox import child_main
from quantlab.sandbox.ast_check import FORBIDDEN_MODULES, FORBIDDEN_NAMES
from quantlab.sandbox.guards import (
    STDLIB_ALLOWED,
    ForbiddenImport,
    NetworkBlocked,
    apply_network_block,
    block_network,
    import_is_allowed,
    make_guarded_import,
    network_raiser,
)
from quantlab.sandbox.protocol import (
    BARS_PARQUET,
    ENGINE_PACKAGE,
    EXIT_BAD_REQUEST,
    EXIT_FORBIDDEN_IMPORT,
    EXIT_INTERNAL,
    EXIT_NETWORK,
    EXIT_RESOURCE_LIMIT,
    EXIT_STRATEGY_LOAD,
    EXIT_STRATEGY_RUNTIME,
    PROTOCOL_VERSION,
    REQUEST_JSON,
    RESPONSE_JSON,
    STRATEGY_PY,
    TRADES_PARQUET,
    SandboxRequest,
    SandboxResponse,
    read_request,
    read_response,
    read_result,
    write_request,
    write_response,
    write_result,
)
from quantlab.sandbox.runner import (
    CHILD_MODULE,
    SandboxLimits,
    SandboxRunner,
    _killed_response,
)

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


# ---------------------------------------------------------------------------
# The child's own helpers, exercised in this process.
#
# Everything below runs here rather than in a child, which is only safe because
# none of it installs a guard: `install_import_guard` and `block_network` are
# irreversible by design, so a test process that called them could not go on to
# test anything else. `_run` is therefore the one function covered by the
# subprocess tests alone.
# ---------------------------------------------------------------------------
def test_chain_walks_cause_and_context() -> None:
    root = ValueError("root")
    middle = KeyError("middle")
    middle.__cause__ = root
    top = RuntimeError("top")
    top.__context__ = middle
    assert list(child_main.chain(top)) == [top, middle, root]


def test_chain_terminates_on_a_cycle() -> None:
    """A self-referential ``__context__`` must not hang the child's error path."""
    first = ValueError("first")
    second = ValueError("second")
    first.__context__ = second
    second.__context__ = first
    assert list(child_main.chain(first)) == [first, second]


def test_caused_by_sees_through_a_wrapper() -> None:
    inner = ForbiddenImport("no")
    outer = StrategyRuntimeError("strategy raised while deciding")
    outer.__cause__ = inner
    assert child_main.caused_by(outer, ForbiddenImport)
    assert not child_main.caused_by(outer, MemoryError)


def _wrapped(inner: BaseException) -> BaseException:
    """What the engine does to anything a strategy raises."""
    outer = StrategyRuntimeError("strategy raised while deciding (bar=7)")
    outer.__cause__ = inner
    return outer


@pytest.mark.parametrize(
    ("exc", "status", "code"),
    [
        (MemoryError("out of memory"), "resource_limit", EXIT_RESOURCE_LIMIT),
        (ForbiddenImport("no"), "forbidden_import", EXIT_FORBIDDEN_IMPORT),
        (NetworkBlocked("no"), "network", EXIT_NETWORK),
        (SandboxProtocolError("bad"), "bad_request", EXIT_BAD_REQUEST),
        (StrategySafetyError("unsafe"), "strategy_load", EXIT_STRATEGY_LOAD),
        (StrategyLoadError("no STRATEGY"), "strategy_load", EXIT_STRATEGY_LOAD),
        (SyntaxError("bad syntax"), "strategy_load", EXIT_STRATEGY_LOAD),
        (ValueError("deliberate"), "strategy_runtime", EXIT_STRATEGY_RUNTIME),
        (KeyboardInterrupt(), "internal", EXIT_INTERNAL),
    ],
)
def test_classify_failure(exc: BaseException, status: str, code: int) -> None:
    assert child_main.classify_failure(exc) == (status, code)


@pytest.mark.parametrize(
    ("inner", "status"),
    [
        (MemoryError("out of memory"), "resource_limit"),
        (ForbiddenImport("no"), "forbidden_import"),
        (NetworkBlocked("no"), "network"),
    ],
)
def test_a_wrapped_failure_keeps_its_own_classification(inner: BaseException, status: str) -> None:
    """The engine wraps strategy exceptions to say which bar failed. A guard that
    tripped inside is still a guard tripping, not ordinary misbehaviour."""
    assert child_main.classify_failure(_wrapped(inner))[0] == status


def test_classification_precedence_puts_the_limit_first() -> None:
    """Memory exhaustion inside an import is the limit, not the import rule: the
    run needs more room, and the candidate is not the problem."""
    exc = ForbiddenImport("no")
    exc.__cause__ = MemoryError("out of memory")
    assert child_main.classify_failure(exc)[0] == "resource_limit"


def test_failure_response_carries_the_message_not_the_traceback() -> None:
    response = child_main.failure_response(_wrapped(ValueError("deliberate")))
    assert response.status == "strategy_runtime"
    assert response.error_type == "StrategyRuntimeError"
    assert "bar=7" in response.error_message
    assert "Traceback" not in response.error_message
    assert not response.ok


def test_failure_response_truncates_a_hostile_message() -> None:
    """A strategy controls this text. It must not be able to make the response
    file arbitrarily large."""
    response = child_main.failure_response(ValueError("x" * 100_000))
    assert len(response.error_message) == 4000


def test_failure_response_lifts_violation_codes_from_the_chain() -> None:
    inner = StrategySafetyError(
        "rejected", violations=["E_FORBIDDEN_IMPORT: os"], codes=["E_FORBIDDEN_IMPORT"]
    )
    response = child_main.failure_response(_wrapped(inner))
    assert response.error_codes == ("E_FORBIDDEN_IMPORT",)


def test_failure_response_has_no_codes_when_nothing_carries_them() -> None:
    assert child_main.failure_response(ValueError("plain")).error_codes == ()


def test_ok_response_describes_the_run(bars: BarFrame) -> None:
    namespace: dict[str, object] = {}
    exec(compile(ALWAYS_LONG, "strategy.py", "exec"), namespace)
    result = SimpleBarEngine().run(namespace["STRATEGY"](), bars, {}, BacktestConfig())  # type: ignore[operator]
    response = child_main.ok_response(result, cpu=1.5, rss=64.0)

    assert response.ok
    assert response.status == "ok"
    assert response.engine_name == "simple_bar"
    assert response.engine_version == SimpleBarEngine.version
    assert response.n_bars == result.n_bars
    assert response.warmup_bars == result.warmup_bars
    assert response.bars_per_year == result.bars_per_year
    assert response.cost_summary == result.cost_summary
    assert response.log == result.log
    assert response.ruined is result.ruined
    assert response.cpu_seconds == 1.5
    assert response.peak_rss_mb == 64.0


def test_rusage_reports_something_plausible() -> None:
    cpu, rss = child_main._rusage()
    assert cpu >= 0.0
    assert rss > 0.0


def test_safe_builtins_removes_the_forbidden_names_but_keeps_import() -> None:
    namespace = child_main.safe_builtins()
    for name in FORBIDDEN_NAMES - {"__import__"}:
        assert name not in namespace, name
    # Without __import__ the import statement itself stops working, which would
    # abolish a strategy's imports rather than restrict them. The guard is the
    # control; this dict is the second lock on the same door.
    assert "__import__" in namespace
    for name in ("len", "range", "isinstance", "float", "min", "max", "sum"):
        assert name in namespace


def test_safe_builtins_does_not_mutate_the_real_builtins() -> None:
    child_main.safe_builtins()
    assert callable(builtins.open)
    assert callable(builtins.eval)


def test_load_strategy_returns_the_exported_class_instance() -> None:
    strategy = child_main.load_strategy(ALWAYS_LONG)
    assert type(strategy).__name__ == "Probe"
    assert strategy.warmup_bars == 0


def test_load_strategy_requires_the_export() -> None:
    with pytest.raises(StrategyLoadError, match="STRATEGY"):
        child_main.load_strategy("X = 1\n")


def test_load_strategy_propagates_a_syntax_error() -> None:
    with pytest.raises(SyntaxError):
        child_main.load_strategy("class Broken:\n    params = {\n")


def test_the_strategy_module_is_not_importable_afterwards() -> None:
    """It is executed into a namespace of its own, never registered, so nothing
    can import the strategy back by name."""
    child_main.load_strategy(ALWAYS_LONG)
    assert child_main._MODULE_NAME not in sys.modules


def test_report_failure_writes_a_response_and_returns_the_code(tmp_path: Path) -> None:
    code = child_main._report_failure(tmp_path, ValueError("deliberate"))
    assert code == EXIT_STRATEGY_RUNTIME
    assert read_response(tmp_path).error_message == "deliberate"


def test_report_failure_survives_a_directory_that_is_gone(tmp_path: Path) -> None:
    """A child killed mid-write leaves no response; the exit code still carries
    the verdict, so this must not raise on top of the failure it is reporting."""
    assert child_main._report_failure(tmp_path / "gone", ValueError("x")) == EXIT_STRATEGY_RUNTIME


def test_main_rejects_a_wrong_argument_count() -> None:
    assert child_main.main([]) == EXIT_BAD_REQUEST
    assert child_main.main(["a", "b"]) == EXIT_BAD_REQUEST


def test_main_reports_an_unreadable_request_rather_than_raising(tmp_path: Path) -> None:
    assert child_main.main([str(tmp_path / "nothing-here")]) == EXIT_BAD_REQUEST


# ---------------------------------------------------------------------------
# The import guard, built but not installed.
# ---------------------------------------------------------------------------
SANDBOXED = "quantlab_sandboxed_strategy"


def _guard(allowed: tuple[str, ...] = ("numpy", "math")) -> Any:
    calls: list[tuple[str, int]] = []

    def fake_import(name: str, *args: Any, **kwargs: Any) -> str:
        calls.append((name, args[3] if len(args) > 3 else 0))
        return f"module:{name}"

    guarded = make_guarded_import(fake_import, allowed, (SANDBOXED,))
    return guarded, calls


def test_the_guard_permits_an_allowed_import_from_the_strategy() -> None:
    guarded, calls = _guard()
    assert guarded("numpy", {"__name__": SANDBOXED}) == "module:numpy"
    assert calls == [("numpy", 0)]


def test_the_guard_refuses_a_disallowed_import_from_the_strategy() -> None:
    guarded, calls = _guard()
    with pytest.raises(ForbiddenImport, match="'os'"):
        guarded("os", {"__name__": SANDBOXED})
    assert calls == []


def test_the_guard_refuses_a_relative_import_from_the_strategy() -> None:
    guarded, _ = _guard()
    with pytest.raises(ForbiddenImport, match="relative"):
        guarded("sibling", {"__name__": SANDBOXED}, None, (), 1)


def test_the_guard_leaves_trusted_modules_alone() -> None:
    """NumPy importing its own internals, or pydantic reaching for pydantic_core
    mid-validation, is not an escape attempt. A blanket rule breaks the libraries
    the strategy is allowed to use, and a sandbox switched off to get work done
    protects nothing."""
    guarded, _ = _guard()
    assert guarded("os", {"__name__": "pandas.io.parquet"}) == "module:os"
    assert guarded("sibling", {"__name__": "pydantic.main"}, None, (), 1) == "module:sibling"


def test_the_guard_treats_an_unattributable_import_as_trusted() -> None:
    """C-level machinery can import with no globals. The strategy cannot reach
    ``__import__`` by name at all — the AST check rejects it — so the only callers
    that arrive this way are library internals."""
    guarded, _ = _guard()
    assert guarded("os") == "module:os"
    assert guarded("os", {}) == "module:os"


def test_the_guard_names_the_allow_list_in_its_message() -> None:
    """A rejection the author cannot act on costs a whole generation."""
    guarded, _ = _guard(("numpy", "pandas"))
    with pytest.raises(ForbiddenImport, match=r"\['numpy', 'pandas'\]"):
        guarded("scipy", {"__name__": SANDBOXED})


# ---------------------------------------------------------------------------
# The network block, applied to a stand-in module.
# ---------------------------------------------------------------------------
class _FakeSocket:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.opened = True

    def connect(self, *args: object) -> str:
        return "connected"

    def bind(self, *args: object) -> str:
        return "bound"


def _fake_socket_module() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        socket=_FakeSocket,
        socketpair=lambda *a, **k: "pair",
        create_connection=lambda *a, **k: "connection",
        getaddrinfo=lambda *a, **k: [("addr",)],
    )


def test_apply_network_block_replaces_the_module_attributes() -> None:
    module = _fake_socket_module()
    apply_network_block(module)
    for attribute in ("socket", "socketpair", "create_connection", "getaddrinfo"):
        with pytest.raises(NetworkBlocked):
            getattr(module, attribute)()


def test_apply_network_block_reaches_a_reference_taken_earlier() -> None:
    """Rebinding the module attribute alone would leave a library that had already
    captured the class holding a working one."""
    module = _fake_socket_module()
    held = module.socket
    apply_network_block(module)
    with pytest.raises(NetworkBlocked):
        held()
    with pytest.raises(NetworkBlocked):
        _FakeSocket.connect(object())  # type: ignore[arg-type]


def test_apply_network_block_ignores_a_module_without_sockets() -> None:
    module = types.SimpleNamespace(pi=3.14)
    apply_network_block(module)
    assert module.pi == 3.14


def test_block_network_skips_a_module_that_was_never_imported() -> None:
    block_network(("a_module_nobody_imported",))


def test_network_raiser_refuses_whatever_it_is_handed() -> None:
    with pytest.raises(NetworkBlocked):
        network_raiser("host", 443, timeout=1)


# ---------------------------------------------------------------------------
# Security: the engine_module mechanism
#
# The engine is the one adapter the child imports, and its name travels in the
# request because the sandbox layer may not depend on an adapter (INV-8). That
# makes the name the single most valuable thing for an attacker to control, so it
# is constrained twice: once when the request is built or parsed, and again at
# the moment of use.
# ---------------------------------------------------------------------------
FORBIDDEN_ENGINE_MODULES = [
    "quantlab.adapters.store.sqlite",
    "quantlab.adapters.store.artifacts",
    "quantlab.adapters.llm.glm",
    "quantlab.adapters.broker.paper",
    "quantlab.adapters.data.binance_archive",
    "quantlab.adapters.secrets.keychain",
    "quantlab.container",
    "quantlab.core.types",
    "os",
    "subprocess",
    "builtins",
    # A package whose name merely starts with the approved one.
    "quantlab.adapters.engineering.evil",
    # Shapes that are not plain module paths at all.
    "quantlab.adapters.engine.",
    "quantlab.adapters.engine..simple_bar",
    "quantlab.adapters.engine._private",
    "quantlab.adapters.engine.a/b",
    "quantlab.adapters.engine.a-b",
    # The package itself exports no engine, and naming it would widen the target.
    "quantlab.adapters.engine",
]


@pytest.mark.parametrize("module", FORBIDDEN_ENGINE_MODULES)
def test_a_request_cannot_name_a_module_outside_the_engine_package(module: str) -> None:
    with pytest.raises(ValidationError):
        SandboxRequest(
            symbol="BTCUSDT", timeframe="1h", engine_module=module, engine_class="SimpleBarEngine"
        )


@pytest.mark.parametrize("module", FORBIDDEN_ENGINE_MODULES)
def test_a_tampered_request_file_cannot_name_one_either(
    tmp_path: Path, bars: BarFrame, module: str
) -> None:
    """The child parses the request from disk. Validation has to happen there
    too, or the constraint only binds the process that already meant well."""
    write_request(tmp_path, SandboxRequest(symbol="B", timeframe="1h", **ENGINE), bars, ALWAYS_LONG)
    payload = (tmp_path / REQUEST_JSON).read_text(encoding="utf-8")
    (tmp_path / REQUEST_JSON).write_text(
        payload.replace(ENGINE["engine_module"], module), encoding="utf-8"
    )
    with pytest.raises(SandboxProtocolError):
        read_request(tmp_path)


@pytest.mark.parametrize("name", ["_Private", "os.system", "", "1abc", "a b", "a.b"])
def test_engine_class_must_be_a_public_identifier(name: str) -> None:
    with pytest.raises(ValidationError):
        SandboxRequest(
            symbol="B", timeframe="1h", engine_class=name, engine_module=ENGINE["engine_module"]
        )


def test_the_child_rechecks_the_engine_name_at_the_point_of_use() -> None:
    """``model_construct`` skips validators. Nothing in this repository calls it
    on a request — and the child must not depend on that staying true."""
    for module in ("quantlab.adapters.store.sqlite", "os", "quantlab.adapters.engineering.x"):
        request = SandboxRequest.model_construct(
            symbol="B", timeframe="1h", engine_module=module, engine_class="SimpleBarEngine"
        )
        with pytest.raises(SandboxProtocolError, match="engine package"):
            child_main._load_engine(request)


def test_the_child_rechecks_the_engine_class_at_the_point_of_use() -> None:
    request = SandboxRequest.model_construct(
        symbol="B",
        timeframe="1h",
        engine_module=ENGINE["engine_module"],
        engine_class="_State",
    )
    with pytest.raises(SandboxProtocolError, match="public name"):
        child_main._load_engine(request)


def test_the_child_refuses_a_name_that_is_not_a_class() -> None:
    """An engine module legitimately holds ``np``, ``pd`` and ``math``. None of
    them is an engine, and none may be instantiated on the strength of a string."""
    for name in ("np", "pd", "math", "build_slippage_model", "NotThere"):
        request = SandboxRequest.model_construct(
            symbol="B",
            timeframe="1h",
            engine_module=ENGINE["engine_module"],
            engine_class=name,
        )
        with pytest.raises(SandboxProtocolError):
            child_main._load_engine(request)


def test_the_child_refuses_a_class_that_is_not_an_engine() -> None:
    """Checked by shape, not by name: ``quantlab.ports`` is off-limits to this
    layer (INV-8), and an object that cannot run a backtest must not reach the
    point where untrusted code is already loaded."""
    request = SandboxRequest.model_construct(
        symbol="B",
        timeframe="1h",
        engine_module=ENGINE["engine_module"],
        engine_class="RiskSpec",
    )
    with pytest.raises(SandboxProtocolError, match="not a backtest engine"):
        child_main._load_engine(request)


def test_the_child_loads_the_approved_engine() -> None:
    request = SandboxRequest(symbol="B", timeframe="1h", **ENGINE)
    engine = child_main._load_engine(request)
    assert isinstance(engine, SimpleBarEngine)


def test_the_engine_package_holds_nothing_but_engines() -> None:
    """The approved namespace is the whole of the trust decision, so it is worth
    knowing exactly how large it is."""
    package = Path("src/quantlab/adapters/engine")
    modules = sorted(p.stem for p in package.glob("*.py"))
    assert modules == ["__init__", "simple_bar"]


def test_the_sandbox_package_imports_no_adapter() -> None:
    """INV-8 restated where it matters most: the engine reaches the child through
    the request, never through an import in this layer. ``test_architecture.py``
    enforces the same rule tree-wide; a regression here is a security regression,
    not a layering nit."""
    for path in sorted(Path("src/quantlab/sandbox").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            imported = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            for name in imported:
                assert not name.startswith("quantlab.adapters"), f"{path}: {name}"
                assert not name.startswith("quantlab.ports"), f"{path}: {name}"


def test_the_child_installs_its_guards_before_it_loads_the_strategy() -> None:
    """The whole design is this order. Reversing the last two statements by one
    line would hand a strategy an unguarded interpreter, and no test that runs a
    *well-behaved* strategy would notice."""
    source = Path("src/quantlab/sandbox/child_main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    run = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_run"
    )
    calls = [
        node
        for node in ast.walk(run)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    calls.sort(key=lambda node: (node.lineno, node.col_offset))
    called = [node.func.id for node in calls if isinstance(node.func, ast.Name)]
    order = [
        name
        for name in called
        if name
        in {
            "_load_engine",
            "require_safe_source",
            "block_network",
            "install_import_guard",
            "load_strategy",
        }
    ]
    assert order == [
        "_load_engine",
        "require_safe_source",
        "block_network",
        "install_import_guard",
        "load_strategy",
    ]


def test_the_engine_is_resolved_before_the_guards_go_up() -> None:
    """Trusted infrastructure has to load while imports are still open; the guard
    that follows applies to the strategy's module, not to this one."""
    source = Path("src/quantlab/sandbox/child_main.py").read_text(encoding="utf-8")
    assert source.index("engine = _load_engine(request)") < source.index("block_network()")
    assert "install_import_guard(request.allowed_imports, (_MODULE_NAME,))" in source


# ---------------------------------------------------------------------------
# The wire format's failure paths
# ---------------------------------------------------------------------------
def test_missing_bars_fail_closed(tmp_path: Path, bars: BarFrame) -> None:
    write_request(tmp_path, SandboxRequest(symbol="B", timeframe="1h", **ENGINE), bars, ALWAYS_LONG)
    (tmp_path / BARS_PARQUET).unlink()
    with pytest.raises(SandboxProtocolError, match="inputs"):
        read_request(tmp_path)


def test_result_tables_round_trip(tmp_path: Path, bars: BarFrame) -> None:
    """The tables are the whole result; a lossy hop would make a stored run
    depend on which side of the process boundary produced it."""
    namespace: dict[str, object] = {}
    exec(compile(CROSSOVER, "strategy.py", "exec"), namespace)
    result = SimpleBarEngine().run(namespace["STRATEGY"](), bars, {}, BacktestConfig())  # type: ignore[operator]
    assert result.trades and result.fills

    response = child_main.ok_response(result, cpu=0.0, rss=0.0)
    write_result(tmp_path, result)
    back = read_result(tmp_path, response)

    pd.testing.assert_series_equal(back.equity, result.equity)
    pd.testing.assert_series_equal(back.position_frac, result.position_frac)
    pd.testing.assert_series_equal(back.signals, result.signals)
    assert back.trades == result.trades
    assert back.fills == result.fills
    assert back.cost_summary == result.cost_summary


def test_a_missing_result_table_fails_closed(tmp_path: Path, bars: BarFrame) -> None:
    namespace: dict[str, object] = {}
    exec(compile(ALWAYS_LONG, "strategy.py", "exec"), namespace)
    result = SimpleBarEngine().run(namespace["STRATEGY"](), bars, {}, BacktestConfig())  # type: ignore[operator]
    write_result(tmp_path, result)
    (tmp_path / TRADES_PARQUET).unlink()
    with pytest.raises(SandboxProtocolError, match="result"):
        read_result(tmp_path, child_main.ok_response(result, 0.0, 0.0))


def test_an_unknown_exit_reason_is_refused(tmp_path: Path, bars: BarFrame) -> None:
    """``exit_reason`` reaches the parent as a bare string. It becomes a typed
    field, so a value outside the three the engine can emit is a protocol error,
    not something to pass through and discover later."""
    namespace: dict[str, object] = {}
    exec(compile(CROSSOVER, "strategy.py", "exec"), namespace)
    result = SimpleBarEngine().run(namespace["STRATEGY"](), bars, {}, BacktestConfig())  # type: ignore[operator]
    write_result(tmp_path, result)

    trades = pd.read_parquet(tmp_path / TRADES_PARQUET)
    trades.loc[0, "exit_reason"] = "made_up"
    trades.to_parquet(tmp_path / TRADES_PARQUET, engine="pyarrow", index=False)
    with pytest.raises(SandboxProtocolError, match="exit_reason"):
        read_result(tmp_path, child_main.ok_response(result, 0.0, 0.0))


def test_a_missing_response_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(SandboxProtocolError, match="no response"):
        read_response(tmp_path)


def test_an_unreadable_response_fails_closed(tmp_path: Path) -> None:
    (tmp_path / RESPONSE_JSON).write_text("{not json", encoding="utf-8")
    with pytest.raises(SandboxProtocolError, match="unreadable"):
        read_response(tmp_path)


# ---------------------------------------------------------------------------
# The parent's diagnosis of a child that died without speaking
# ---------------------------------------------------------------------------
def test_a_child_killed_by_sigxcpu_is_diagnosed_as_the_cpu_limit() -> None:
    limits = SandboxLimits(cpu_seconds=7)
    response = _killed_response(-24, "", limits)
    assert response.status == "resource_limit"
    assert "CPU limit of 7s" in response.error_message


def test_a_child_killed_by_sigkill_is_diagnosed_as_the_memory_limit() -> None:
    response = _killed_response(-9, "", SandboxLimits(memory_mb=64))
    assert response.status == "resource_limit"
    assert "64 MB" in response.error_message


def test_a_memory_error_on_stderr_is_diagnosed_as_the_memory_limit() -> None:
    response = _killed_response(1, "MemoryError: cannot allocate", SandboxLimits())
    assert response.status == "resource_limit"


def test_any_other_silent_death_keeps_the_exit_code_meaning() -> None:
    response = _killed_response(EXIT_FORBIDDEN_IMPORT, "", SandboxLimits())
    assert response.status == "forbidden_import"
    assert "without writing a response" in response.error_message

    unknown = _killed_response(3, "", SandboxLimits())
    assert unknown.status == "internal"


def test_a_child_that_claims_success_but_exits_nonzero_is_not_believed(tmp_path: Path) -> None:
    """It was killed after writing. The exit code is the harder evidence."""
    write_response(tmp_path, SandboxResponse(status="ok", n_bars=10))
    response = SandboxRunner()._response_for(tmp_path, EXIT_STRATEGY_RUNTIME, "")
    assert response.status == "strategy_runtime"
    assert response.error_type == "ExitCodeMismatch"
    assert response.n_bars == 10


def test_a_child_that_wrote_nothing_is_diagnosed_from_how_it_died(tmp_path: Path) -> None:
    response = SandboxRunner()._response_for(tmp_path, -24, "")
    assert response.status == "resource_limit"


def test_a_failed_run_can_keep_its_directory_for_inspection(bars: BarFrame) -> None:
    runner = SandboxRunner(keep_failed_dirs=True)
    outcome = runner.run(_guard_probe('raise ValueError("deliberate")'), bars, **ENGINE)
    assert not outcome.ok
    kept = sorted(Path(tempfile.gettempdir()).glob("quantlab-sandbox-*"))
    assert kept, "the failed run's directory was removed"
    for directory in kept:
        assert (directory / STRATEGY_PY).exists()
        shutil.rmtree(directory, ignore_errors=True)


# ---------------------------------------------------------------------------
# Security: the allow-list has a floor
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("module", sorted(FORBIDDEN_MODULES))
def test_a_forbidden_module_cannot_be_allow_listed_back_in(module: str) -> None:
    """The allow-list widens what is permitted; it never re-opens what is banned.

    ``allowed_imports`` travels in the request, and the child reads it from a file
    in a directory the strategy's own process can see. A request that said
    ``["os"]`` must still not hand over the filesystem — and the AST check applies
    the same floor at parse time, so the two layers cannot disagree."""
    assert not import_is_allowed(module, frozenset({module}))
    assert not import_is_allowed(f"{module}.sub", frozenset({module}))


def test_the_guard_refuses_a_forbidden_module_even_when_allow_listed() -> None:
    guarded, calls = _guard(("os", "socket", "numpy"))
    for module in ("os", "socket"):
        with pytest.raises(ForbiddenImport):
            guarded(module, {"__name__": SANDBOXED})
    assert guarded("numpy", {"__name__": SANDBOXED}) == "module:numpy"
    assert calls == [("numpy", 0)]


def test_a_widened_allow_list_does_not_reach_the_filesystem(
    runner: SandboxRunner, bars: BarFrame
) -> None:
    """End to end: even with the limits themselves widened, the floor holds."""
    widened = SandboxRunner(
        limits=SandboxLimits(allowed_imports=(*SandboxLimits().allowed_imports, "os", "socket"))
    )
    outcome = widened.run(
        _guard_probe("import os\nreturn Signal(SignalKind.FLAT)"), bars, ast_check=False, **ENGINE
    )
    assert outcome.status == "forbidden_import"
