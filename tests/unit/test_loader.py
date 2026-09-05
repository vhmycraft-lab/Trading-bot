"""The strategy loader (master spec sections 9, 11, task T20).

The loader is the gate between a file of Python and a `strategy_version` row that
runs can cite. What it must get right is narrow and unforgiving: the id is the
hash of the bytes, the stored copy is those same bytes, and a later run resolves
both from the store rather than from whatever the working tree says by then.

Nothing here executes a strategy. That is the property under test as much as any
assertion: reading a declaration must never become running it (INV-4).
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Final

import pytest
from sqlalchemy import Engine

from quantlab.adapters.store.artifacts import FileSourceStore
from quantlab.adapters.store.sqlite import SqliteExperimentStore, make_session_factory
from quantlab.core.errors import StoreError, StrategyLoadError, StrategySafetyError
from quantlab.core.hashing import sha256_hex, short_id
from quantlab.core.hashing import strategy_id as canonical_strategy_id
from quantlab.ports.store import SourceStore
from quantlab.strategies_io import SUPPORTED_STYLES, LoadedStrategy, StrategyLoader

REPO: Final[Path] = Path(__file__).resolve().parents[2]
FIXTURES: Final[Path] = REPO / "tests" / "fixtures" / "strategies"
BASELINES: Final[Path] = REPO / "strategies" / "baselines"

SMA_CROSS: Final[str] = (BASELINES / "sma_cross.py").read_text(encoding="utf-8")
BUY_AND_HOLD: Final[str] = (BASELINES / "buy_and_hold.py").read_text(encoding="utf-8")
VECTORIZED: Final[str] = (FIXTURES / "valid" / "vectorized_valid.py").read_text(encoding="utf-8")


@pytest.fixture
def sources(tmp_path: Path) -> FileSourceStore:
    return FileSourceStore(tmp_path / "strategies" / "generated")


@pytest.fixture
def store(db_engine: Engine) -> SqliteExperimentStore:
    return SqliteExperimentStore(make_session_factory(db_engine), now_ms=lambda: 1_700_000_000_000)


@pytest.fixture
def loader(store: SqliteExperimentStore, sources: FileSourceStore) -> StrategyLoader:
    return StrategyLoader(store, sources)


# ---------------------------------------------------------------------------
# loading a valid bar_loop strategy
# ---------------------------------------------------------------------------
def test_a_bar_loop_strategy_loads(loader: StrategyLoader) -> None:
    loaded = loader.load_source(SMA_CROSS, family="sma_cross")
    assert isinstance(loaded, LoadedStrategy)
    assert loaded.class_name == "SmaCross"
    assert loaded.style == "bar_loop"
    assert loaded.warmup_bars == 400
    assert loaded.logic_lines > 0
    assert loaded.source == SMA_CROSS
    assert loaded.author == "human"
    assert loaded.parent_strategy_id is None


@pytest.mark.parametrize(
    "name", ["sma_cross.py", "buy_and_hold.py", "rsi_reversion.py", "random_entry.py"]
)
def test_every_shipped_baseline_loads(loader: StrategyLoader, name: str) -> None:
    """The baselines are run on every segment (section 9.5). If the loader cannot
    admit them, nothing else in the platform has a reference point."""
    loaded = loader.load_path(BASELINES / name, family=name.removesuffix(".py"))
    assert loaded.style == "bar_loop"
    assert loaded.class_name


def test_the_template_loads(loader: StrategyLoader) -> None:
    loaded = loader.load_path(REPO / "strategies" / "TEMPLATE.py", family="template")
    assert loaded.class_name == "MyStrategy"


def test_load_path_and_load_source_agree(loader: StrategyLoader, tmp_path: Path) -> None:
    path = tmp_path / "copy.py"
    path.write_text(SMA_CROSS, encoding="utf-8")
    from_disk = loader.load_path(path, family="sma_cross")
    from_text = loader.load_source(SMA_CROSS, family="sma_cross")
    assert from_disk.strategy_id == from_text.strategy_id
    assert from_disk.code_sha256 == from_text.code_sha256


def test_an_unreadable_path_is_reported(loader: StrategyLoader, tmp_path: Path) -> None:
    with pytest.raises(StrategyLoadError, match="cannot read strategy source"):
        loader.load_path(tmp_path / "absent.py", family="f")


# ---------------------------------------------------------------------------
# identity: strategy_id and code_sha256
# ---------------------------------------------------------------------------
def test_the_strategy_id_is_the_canonical_hash_of_the_source(loader: StrategyLoader) -> None:
    """Section 6: ``sha256(code)[:16]``. The rule lives in ``core.hashing`` and is
    called, never re-implemented — the run id of section 11.2 is built from it, and
    two spellings of one rule would silently split the cache."""
    loaded = loader.load_source(SMA_CROSS, family="sma_cross")
    assert loaded.strategy_id == canonical_strategy_id(SMA_CROSS)
    assert loaded.strategy_id == short_id(SMA_CROSS)
    assert loaded.strategy_id == sha256_hex(SMA_CROSS)[:16]
    assert len(loaded.strategy_id) == 16


def test_the_recorded_checksum_is_the_full_digest(loader: StrategyLoader) -> None:
    loaded = loader.load_source(SMA_CROSS, family="sma_cross")
    assert loaded.code_sha256 == sha256_hex(SMA_CROSS)
    assert len(loaded.code_sha256) == 64
    assert loaded.code_sha256.startswith(loaded.strategy_id)


def test_one_changed_byte_is_a_different_version(loader: StrategyLoader) -> None:
    first = loader.load_source(SMA_CROSS, family="sma_cross")
    second = loader.load_source(SMA_CROSS + "\n", family="sma_cross")
    assert first.strategy_id != second.strategy_id
    assert first.code_sha256 != second.code_sha256


def test_the_loader_does_not_reimplement_the_id_rule() -> None:
    """Rule 10 of the task, checked rather than asserted in prose: the loader must
    call ``core.hashing``, not hash anything itself."""
    tree = ast.parse(
        (REPO / "src" / "quantlab" / "strategies_io" / "loader.py").read_text(encoding="utf-8")
    )
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "hashlib" not in imported

    from_hashing = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "quantlab.core.hashing"
        for alias in node.names
    }
    assert "compute_strategy_id" in from_hashing, "the id rule must come from core.hashing"

    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "compute_strategy_id" in called


# ---------------------------------------------------------------------------
# the immutable copy
# ---------------------------------------------------------------------------
def test_the_source_is_copied_byte_for_byte(
    loader: StrategyLoader, sources: FileSourceStore
) -> None:
    loaded = loader.load_source(SMA_CROSS, family="sma_cross")
    stored = sources.path_for(loaded.strategy_id)
    assert stored.is_file()
    assert stored.read_text(encoding="utf-8") == SMA_CROSS
    assert sources.read_source(loaded.strategy_id) == SMA_CROSS


def test_the_stored_copy_survives_an_edit_to_the_working_file(
    loader: StrategyLoader, sources: FileSourceStore, tmp_path: Path
) -> None:
    """The point of the copy: a run cites bytes, and the working tree moves on."""
    working = tmp_path / "work.py"
    working.write_text(SMA_CROSS, encoding="utf-8")
    loaded = loader.load_path(working, family="sma_cross")

    working.write_text(SMA_CROSS.replace("default=50", "default=51"), encoding="utf-8")
    assert loader.verify(loaded.strategy_id) == SMA_CROSS


def test_the_code_path_is_a_store_key_not_a_machine_path(
    loader: StrategyLoader, sources: FileSourceStore
) -> None:
    """A database of absolute paths stops resolving the moment the repo moves."""
    loaded = loader.load_source(SMA_CROSS, family="sma_cross")
    assert loaded.code_path == f"{loaded.strategy_id}.py"
    assert not Path(loaded.code_path).is_absolute()
    assert sources.path_for(loaded.strategy_id).name == loaded.code_path


def test_the_source_store_refuses_to_overwrite_with_different_bytes(
    sources: FileSourceStore,
) -> None:
    """An overwrite would rewrite the source of every run already citing the id."""
    sources.write_source("abc123", "x = 1\n")
    assert sources.write_source("abc123", "x = 1\n") == "abc123.py"
    with pytest.raises(StoreError, match="different source is already stored"):
        sources.write_source("abc123", "x = 2\n")


def test_reading_a_source_that_was_never_stored_is_an_error(sources: FileSourceStore) -> None:
    assert not sources.exists("nope")
    with pytest.raises(StoreError, match="no stored source"):
        sources.read_source("nope")


@pytest.mark.parametrize("bad", ["..", ".", "", "a/b", "../escape", "/absolute"])
def test_a_strategy_id_that_could_escape_the_tree_is_refused(
    sources: FileSourceStore, bad: str
) -> None:
    with pytest.raises(StoreError, match=r"path component|directory reference"):
        sources.path_for(bad)


# ---------------------------------------------------------------------------
# registration, family and lineage
# ---------------------------------------------------------------------------
def test_the_version_is_registered_with_everything_reproduction_needs(
    loader: StrategyLoader, store: SqliteExperimentStore
) -> None:
    loaded = loader.load_source(SMA_CROSS, family="sma_cross")
    row = store.lineage(loaded.strategy_id)[-1]

    assert row.strategy_id == loaded.strategy_id
    assert row.family_id == loaded.family_id
    assert row.code_path == loaded.code_path
    assert row.code_sha256 == loaded.code_sha256
    assert row.class_name == "SmaCross"
    assert row.style == "bar_loop"
    assert row.author == "human"
    assert row.logic_lines == loaded.logic_lines
    assert row.parent_strategy_id is None


def test_the_stored_param_schema_is_read_from_the_declaration(
    loader: StrategyLoader, store: SqliteExperimentStore
) -> None:
    import json

    loaded = loader.load_source(SMA_CROSS, family="sma_cross")
    assert loaded.param_schema == {
        "fast": {"kind": "int", "default": 50, "low": 5, "high": 200},
        "slow": {"kind": "int", "default": 200, "low": 20, "high": 400},
    }
    assert loaded.n_params == 2
    row = store.lineage(loaded.strategy_id)[-1]
    assert json.loads(row.param_schema_json) == loaded.param_schema


def test_a_strategy_with_no_params_stores_an_empty_schema(loader: StrategyLoader) -> None:
    loaded = loader.load_source(BUY_AND_HOLD, family="buy_and_hold")
    assert loaded.param_schema == {}
    assert loaded.n_params == 0


def test_the_family_is_created_once_and_reused(
    loader: StrategyLoader, store: SqliteExperimentStore
) -> None:
    first = loader.load_source(SMA_CROSS, family="sma_cross")
    second = loader.load_source(BUY_AND_HOLD, family="sma_cross")
    assert first.family_id == second.family_id
    assert store.set_family_status(first.family_id, "open").name == "sma_cross"


def test_two_families_are_distinct(loader: StrategyLoader) -> None:
    a = loader.load_source(SMA_CROSS, family="alpha")
    b = loader.load_source(BUY_AND_HOLD, family="beta")
    assert a.family_id != b.family_id


def test_the_origin_of_a_family_is_recorded(loader: StrategyLoader) -> None:
    """Section 6 distinguishes human from model-authored lines of enquiry; section
    14.4 counts them separately when deflating for how much was tried."""
    loaded = loader.load_source(SMA_CROSS, family="proposed", origin="llm", author="llm:mock:m1")
    assert loaded.author == "llm:mock:m1"


def test_lineage_is_recorded_and_read_back_oldest_first(loader: StrategyLoader) -> None:
    parent = loader.load_source(SMA_CROSS, family="sma_cross")
    child = loader.load_source(
        SMA_CROSS.replace("default=50", "default=40"),
        family="sma_cross",
        parent_strategy_id=parent.strategy_id,
    )
    grandchild = loader.load_source(
        SMA_CROSS.replace("default=50", "default=30"),
        family="sma_cross",
        parent_strategy_id=child.strategy_id,
    )
    assert child.parent_strategy_id == parent.strategy_id

    chain = [row.strategy_id for row in loader.lineage(grandchild.strategy_id)]
    assert chain == [parent.strategy_id, child.strategy_id, grandchild.strategy_id]
    assert [row.strategy_id for row in loader.lineage(parent.strategy_id)] == [parent.strategy_id]


def test_an_llm_interaction_can_be_cited(
    loader: StrategyLoader, store: SqliteExperimentStore
) -> None:
    interaction = store.record_llm_interaction(
        campaign="c1",
        provider="mock",
        model="m1",
        purpose="propose",
        prompt_sha256="b" * 64,
        prompt_path="artifacts/llm/p.json",
        response_path="artifacts/llm/r.json",
        prompt_template_sha256="c" * 64,
        tokens_in=1,
        tokens_out=1,
        cost_eur=0.0,
        temperature=0.0,
        latency_ms=1,
        status="ok",
    )
    loaded = loader.load_source(
        SMA_CROSS,
        family="proposed",
        origin="llm",
        author="llm:mock:m1",
        llm_interaction_id=interaction.interaction_id,
    )
    assert loaded.strategy_id


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------
def test_loading_the_same_source_twice_is_the_same_version(
    loader: StrategyLoader, store: SqliteExperimentStore, sources: FileSourceStore
) -> None:
    first = loader.load_source(SMA_CROSS, family="sma_cross")
    second = loader.load_source(SMA_CROSS, family="sma_cross")

    assert first == second
    assert len(store.lineage(first.strategy_id)) == 1
    assert sources.read_source(first.strategy_id) == SMA_CROSS


def test_loading_is_independent_of_the_store_it_lands_in(tmp_path: Path, db_engine: Engine) -> None:
    """Two fresh stores, same source: the same id and the same bytes. Identity is a
    property of the source, not of where it was filed."""
    ids = set()
    for index in range(2):
        store = SqliteExperimentStore(
            make_session_factory(db_engine), now_ms=lambda index=index: index
        )
        loader = StrategyLoader(store, FileSourceStore(tmp_path / f"gen{index}"))
        ids.add(loader.load_source(SMA_CROSS, family="sma_cross").strategy_id)
    assert len(ids) == 1


def test_the_loader_holds_no_state_between_calls(loader: StrategyLoader) -> None:
    a = loader.load_source(SMA_CROSS, family="f")
    b = loader.load_source(BUY_AND_HOLD, family="f")
    c = loader.load_source(SMA_CROSS, family="f")
    assert a == c != b


# ---------------------------------------------------------------------------
# tamper detection
# ---------------------------------------------------------------------------
def test_verify_returns_the_source_when_nothing_has_changed(loader: StrategyLoader) -> None:
    loaded = loader.load_source(SMA_CROSS, family="sma_cross")
    assert loader.verify(loaded.strategy_id) == SMA_CROSS


def test_an_edited_stored_file_is_detected(
    loader: StrategyLoader, sources: FileSourceStore
) -> None:
    """The recorded checksum catches a file edited after registration."""
    loaded = loader.load_source(SMA_CROSS, family="sma_cross")
    sources.path_for(loaded.strategy_id).write_text(
        SMA_CROSS.replace("default=50", "default=999"), encoding="utf-8"
    )
    with pytest.raises(StrategyLoadError, match="does not match its recorded checksum"):
        loader.verify(loaded.strategy_id)


def test_a_truncated_stored_file_is_detected(
    loader: StrategyLoader, sources: FileSourceStore
) -> None:
    loaded = loader.load_source(SMA_CROSS, family="sma_cross")
    sources.path_for(loaded.strategy_id).write_text("", encoding="utf-8")
    with pytest.raises(StrategyLoadError, match="checksum"):
        loader.verify(loaded.strategy_id)


def test_editing_the_file_and_the_checksum_together_is_still_detected(
    loader: StrategyLoader, sources: FileSourceStore, db_engine: Engine
) -> None:
    """An attacker who changes the file is caught by the checksum. One who changes
    both is caught by the id, which is the primary key every run cites and cannot
    be edited without orphaning the runs."""
    from sqlalchemy import text

    loaded = loader.load_source(SMA_CROSS, family="sma_cross")
    tampered = SMA_CROSS.replace("default=50", "default=999")
    sources.path_for(loaded.strategy_id).write_text(tampered, encoding="utf-8")
    with db_engine.begin() as connection:
        connection.execute(
            text("UPDATE strategy_version SET code_sha256 = :sha WHERE strategy_id = :sid"),
            {"sha": sha256_hex(tampered), "sid": loaded.strategy_id},
        )
    with pytest.raises(StrategyLoadError, match="does not hash to its own id"):
        loader.verify(loaded.strategy_id)


def test_a_missing_stored_file_is_reported(
    loader: StrategyLoader, sources: FileSourceStore
) -> None:
    loaded = loader.load_source(SMA_CROSS, family="sma_cross")
    sources.path_for(loaded.strategy_id).unlink()
    with pytest.raises(StoreError, match="no stored source"):
        loader.verify(loaded.strategy_id)


def test_verifying_an_unregistered_version_is_reported(loader: StrategyLoader) -> None:
    with pytest.raises(StrategyLoadError, match="no such strategy version"):
        loader.verify("0" * 16)


def test_load_registered_reads_everything_from_the_store(
    loader: StrategyLoader, sources: FileSourceStore
) -> None:
    """INV-7: a later run reproduces what it executed, not what the file says now."""
    original = loader.load_source(SMA_CROSS, family="sma_cross")
    back = loader.load_registered(original.strategy_id)

    assert back.strategy_id == original.strategy_id
    assert back.source == original.source
    assert back.code_sha256 == original.code_sha256
    assert back.code_path == original.code_path
    assert back.class_name == original.class_name
    assert back.param_schema == original.param_schema
    assert back.warmup_bars == original.warmup_bars


def test_load_registered_refuses_a_tampered_version(
    loader: StrategyLoader, sources: FileSourceStore
) -> None:
    loaded = loader.load_source(SMA_CROSS, family="sma_cross")
    sources.path_for(loaded.strategy_id).write_text(SMA_CROSS + "# x\n", encoding="utf-8")
    with pytest.raises(StrategyLoadError, match="checksum"):
        loader.load_registered(loaded.strategy_id)


def test_reading_back_does_not_write(loader: StrategyLoader, sources: FileSourceStore) -> None:
    """A read path that wrote would defeat the tamper check it just performed."""
    loaded = loader.load_source(SMA_CROSS, family="sma_cross")
    path = sources.path_for(loaded.strategy_id)
    before = path.stat().st_mtime_ns
    loader.load_registered(loaded.strategy_id)
    assert path.stat().st_mtime_ns == before


# ---------------------------------------------------------------------------
# unsafe source is refused, and nothing is persisted
# ---------------------------------------------------------------------------
REJECTED = sorted(p.name for p in (FIXTURES / "rejected").glob("*.py"))


@pytest.mark.parametrize("name", REJECTED)
def test_every_unsafe_fixture_is_refused(loader: StrategyLoader, name: str) -> None:
    source = (FIXTURES / "rejected" / name).read_text(encoding="utf-8")
    with pytest.raises((StrategySafetyError, StrategyLoadError)):
        loader.load_source(source, family="hostile")


def test_a_refused_strategy_leaves_no_trace(
    loader: StrategyLoader, store: SqliteExperimentStore, sources: FileSourceStore
) -> None:
    """The check runs before the copy and the row, so a rejected file is never
    registered, never copied, and never given an id anything could cite."""
    source = (FIXTURES / "rejected" / "import_os.py").read_text(encoding="utf-8")
    with pytest.raises(StrategySafetyError):
        loader.load_source(source, family="hostile")

    assert not list(sources.root.glob("*.py")) if sources.root.exists() else True
    with pytest.raises(StrategyLoadError, match="no such strategy version"):
        loader.verify(canonical_strategy_id(source))


def test_the_violation_codes_reach_the_caller(loader: StrategyLoader) -> None:
    source = (FIXTURES / "rejected" / "import_os.py").read_text(encoding="utf-8")
    with pytest.raises(StrategySafetyError) as excinfo:
        loader.load_source(source, family="hostile")
    assert "E_FORBIDDEN_IMPORT" in excinfo.value.codes


def test_the_allow_list_is_the_loaders_to_set(
    loader: StrategyLoader, store: SqliteExperimentStore, sources: FileSourceStore
) -> None:
    source = (FIXTURES / "rejected" / "import_not_allowed.py").read_text(encoding="utf-8")
    with pytest.raises(StrategySafetyError):
        loader.load_source(source, family="f")

    from quantlab.sandbox.ast_check import DEFAULT_ALLOWED_IMPORTS

    widened = StrategyLoader(store, sources, allowed_imports=(*DEFAULT_ALLOWED_IMPORTS, "scipy"))
    assert widened.load_source(source, family="f").class_name == "Broken"


# ---------------------------------------------------------------------------
# vectorized strategies: refused until T19 exists
# ---------------------------------------------------------------------------
def test_a_vectorized_strategy_is_refused_because_the_probe_does_not_exist(
    loader: StrategyLoader,
) -> None:
    """Section 9.1 requires the truncation probe of section 14.2 to run here before
    any backtest. It is T19 and is not implemented, so loading one would record a
    guarantee nothing checked."""
    with pytest.raises(StrategyLoadError) as excinfo:
        loader.load_source(VECTORIZED, family="vectorized")

    message = str(excinfo.value)
    assert "vectorized" in message
    assert "T19" in message
    assert "truncation probe" in message


def test_the_refusal_is_deterministic(loader: StrategyLoader) -> None:
    messages = set()
    for _ in range(3):
        with pytest.raises(StrategyLoadError) as excinfo:
            loader.load_source(VECTORIZED, family="vectorized")
        messages.add(str(excinfo.value))
    assert len(messages) == 1


def test_a_refused_vectorized_strategy_is_not_registered_or_copied(
    loader: StrategyLoader, sources: FileSourceStore
) -> None:
    """Fail closed: the refusal happens before anything is written, so no row and
    no file exist for a strategy this build cannot vouch for."""
    with pytest.raises(StrategyLoadError):
        loader.load_source(VECTORIZED, family="vectorized")

    identifier = canonical_strategy_id(VECTORIZED)
    assert not sources.exists(identifier)
    with pytest.raises(StrategyLoadError, match="no such strategy version"):
        loader.verify(identifier)


def test_the_refusal_is_not_a_silent_pass_or_a_stub_probe() -> None:
    """The refusal must not be quietly convertible into a success.

    A stub that returned "probe passed" would satisfy every other test in this
    file and leave the platform asserting causality it never checked.
    """
    source = (REPO / "src" / "quantlab" / "strategies_io" / "loader.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    probe = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_probe_vectorized"
    )
    raises = [node for node in ast.walk(probe) if isinstance(node, ast.Raise)]
    returns = [node for node in ast.walk(probe) if isinstance(node, ast.Return)]
    assert raises, "the vectorized probe placeholder must raise"
    assert not returns, "the placeholder must not return, which would read as a pass"
    assert "bar_loop" in SUPPORTED_STYLES
    assert "vectorized" not in SUPPORTED_STYLES


def test_bar_loop_still_loads_alongside_the_refusal(loader: StrategyLoader) -> None:
    """The refusal must be narrow: it costs nothing to the styles that are fine."""
    with pytest.raises(StrategyLoadError):
        loader.load_source(VECTORIZED, family="vectorized")
    assert loader.load_source(SMA_CROSS, family="sma_cross").style == "bar_loop"


def test_an_unknown_style_is_refused(loader: StrategyLoader) -> None:
    source = SMA_CROSS.replace('style = "bar_loop"', 'style = "quantum"')
    with pytest.raises(StrategyLoadError, match="unknown strategy style"):
        loader.load_source(source, family="f")


# ---------------------------------------------------------------------------
# invariants
# ---------------------------------------------------------------------------
def test_inv4_the_loader_never_executes_the_strategy() -> None:
    """INV-4: untrusted code never runs in this process.

    The loader reads a declaration out of a parse. ``ast.literal_eval`` evaluates
    literals and nothing else; ``exec``, ``eval``, ``compile``, ``__import__`` and
    ``importlib`` are absent, and ``test_architecture.py`` enforces the same rule
    across the tree.
    """
    path = REPO / "src" / "quantlab" / "strategies_io" / "loader.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {"exec", "eval", "compile", "__import__"}
        if isinstance(node, ast.Import):
            assert not {a.name.split(".")[0] for a in node.names} & {
                "importlib",
                "pickle",
                "marshal",
                "subprocess",
                "os",
            }


def test_inv4_a_strategy_with_a_side_effect_is_not_run(
    loader: StrategyLoader, tmp_path: Path
) -> None:
    """A module-level side effect would fire on import. This one never does,
    because the loader does not import — and the AST checker refuses the shape
    anyway, which is the belt to this brace."""
    canary = tmp_path / "canary.txt"
    hostile = SMA_CROSS.replace(
        "STRATEGY = SmaCross",
        f'open({str(canary)!r}, "w").write("ran")\n\nSTRATEGY = SmaCross',
    )
    with pytest.raises((StrategySafetyError, StrategyLoadError)):
        loader.load_source(hostile, family="hostile")
    assert not canary.exists()


def test_inv8_the_loader_stays_inside_its_layer() -> None:
    """INV-8: ``strategies_io`` may reach ``core``, ``ports`` and ``sandbox``, and
    no adapter. It is handed its stores through the ports, which is also what makes
    it testable against anything satisfying them."""
    allowed = ("quantlab.core", "quantlab.ports", "quantlab.sandbox", "quantlab.strategies_io")
    for path in sorted((REPO / "src" / "quantlab" / "strategies_io").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            for name in names:
                if name.startswith("quantlab."):
                    assert name.startswith(allowed), f"{path}: {name}"


def test_inv8_the_loader_accepts_any_store_satisfying_the_ports(
    store: SqliteExperimentStore, tmp_path: Path
) -> None:
    """A hand-written source store, no adapter in sight."""

    class MemorySourceStore:
        def __init__(self) -> None:
            self.files: dict[str, str] = {}

        def write_source(self, strategy_id: str, source: str) -> str:
            self.files[strategy_id] = source
            return f"{strategy_id}.py"

        def read_source(self, strategy_id: str) -> str:
            return self.files[strategy_id]

        def path_for(self, strategy_id: str) -> Path:
            return Path(f"{strategy_id}.py")

        def exists(self, strategy_id: str) -> bool:
            return strategy_id in self.files

    memory = MemorySourceStore()
    assert isinstance(memory, SourceStore)
    loaded = StrategyLoader(store, memory).load_source(SMA_CROSS, family="sma_cross")
    assert memory.files[loaded.strategy_id] == SMA_CROSS


def test_inv2_no_credential_reaches_the_stored_record(
    loader: StrategyLoader, store: SqliteExperimentStore
) -> None:
    """INV-2: what the loader persists is source, a hash and a path. Nothing it
    writes could carry a secret that was not already in the source it was given."""
    loaded = loader.load_source(SMA_CROSS, family="sma_cross")
    row = store.lineage(loaded.strategy_id)[-1]
    for value in (row.code_path, row.code_sha256, row.class_name, row.author):
        assert "key" not in value.lower()
        assert "secret" not in value.lower()


def test_inv7_the_record_is_enough_to_reproduce_the_load(
    loader: StrategyLoader, store: SqliteExperimentStore, sources: FileSourceStore
) -> None:
    """INV-7: everything a later run needs comes from the row and the stored bytes.

    Rebuild the loader from scratch — new object, same ports — and the version
    reads back identically. Nothing was cached in the loader that a fresh process
    would lack.
    """
    original = loader.load_source(SMA_CROSS, family="sma_cross")
    fresh = StrategyLoader(store, sources)
    back = fresh.load_registered(original.strategy_id)
    assert back == LoadedStrategy(
        strategy_id=original.strategy_id,
        family_id=original.family_id,
        class_name=original.class_name,
        style=original.style,
        code_path=original.code_path,
        code_sha256=original.code_sha256,
        logic_lines=original.logic_lines,
        warmup_bars=original.warmup_bars,
        param_schema=original.param_schema,
        source=original.source,
        parent_strategy_id=None,
    )


# ---------------------------------------------------------------------------
# reading the declaration
# ---------------------------------------------------------------------------
def test_a_non_literal_parameter_bound_is_refused(loader: StrategyLoader) -> None:
    """A bound only knowable by running the module is not a bound. The schema is
    stored and later used to mutate parameters, so it has to be readable as data."""
    source = SMA_CROSS.replace("high=200", "high=MAX_FAST")
    with pytest.raises((StrategyLoadError, StrategySafetyError)):
        loader.load_source(source, family="f")


def test_the_schema_survives_a_round_trip_as_json(loader: StrategyLoader) -> None:
    import json

    from quantlab.core.hashing import canonical_json

    loaded = loader.load_source(SMA_CROSS, family="sma_cross")
    assert json.loads(canonical_json(loaded.param_schema)) == loaded.param_schema


def test_the_report_and_the_schema_agree_on_the_parameter_names(
    loader: StrategyLoader,
) -> None:
    from quantlab.sandbox.ast_check import check_source

    report = check_source(SMA_CROSS)
    loaded = loader.load_source(SMA_CROSS, family="sma_cross")
    assert set(report.param_names) == set(loaded.param_schema)
    assert report.n_params == loaded.n_params
    assert report.style == loaded.style
    assert report.warmup_bars == loaded.warmup_bars


def test_the_loader_reprs_usefully(loader: StrategyLoader) -> None:
    assert "StrategyLoader" in repr(loader)
    assert "FileSourceStore" in repr(loader)


def test_loaded_strategy_is_immutable(loader: StrategyLoader) -> None:
    loaded = loader.load_source(SMA_CROSS, family="sma_cross")
    with pytest.raises((AttributeError, TypeError)):
        loaded.strategy_id = "tampered"  # type: ignore[misc]


def test_the_supported_styles_are_declared_not_implied() -> None:
    assert frozenset({"bar_loop"}) == SUPPORTED_STYLES


# ---------------------------------------------------------------------------
# declaration shapes the AST checker permits but the schema cannot represent
# ---------------------------------------------------------------------------
def _with_params(body: str) -> str:
    """A minimal valid strategy whose ``params`` is ``body``."""
    return (
        "from quantlab.core.strategy import Context, ParamSpec, Signal, SignalKind\n\n\n"
        "class Probe:\n"
        '    name = "probe"\n'
        '    version = "1"\n'
        '    style = "bar_loop"\n'
        f"    params = {body}\n"
        "    warmup_bars = 5\n\n"
        "    def prepare(self, params):\n"
        "        self.p = dict(params)\n\n"
        "    def on_bar(self, ctx: Context) -> Signal:\n"
        "        return Signal(SignalKind.FLAT)\n\n\n"
        "STRATEGY = Probe\n"
    )


def test_a_non_string_parameter_name_is_refused(loader: StrategyLoader) -> None:
    """§9.2 lets this through — it only checks that each value is a ParamSpec. The
    schema is a JSON object, so a non-string key could not survive storage."""
    source = _with_params('{1: ParamSpec(kind="bool", default=True)}')
    with pytest.raises(StrategyLoadError, match="string literals"):
        loader.load_source(source, family="f")


def test_parameters_splatted_from_a_dict_are_refused(loader: StrategyLoader) -> None:
    """``ParamSpec(**{...})`` hides which arguments were supplied. The mutation
    operators of §13.4 act on named bounds; a splat has none to act on."""
    source = _with_params('{"n": ParamSpec(**{"kind": "bool", "default": True})}')
    with pytest.raises(StrategyLoadError, match="does not accept"):
        loader.load_source(source, family="f")


def test_an_annotated_params_declaration_is_read(loader: StrategyLoader) -> None:
    """``params: dict = {...}`` declares the same thing and must read the same."""
    source = _with_params('{"n": ParamSpec(kind="int", default=5, low=1, high=9)}').replace(
        "    params = {", "    params: dict = {"
    )
    loaded = loader.load_source(source, family="f")
    assert loaded.param_schema == {"n": {"kind": "int", "default": 5, "low": 1, "high": 9}}


def test_a_parameter_that_is_not_a_paramspec_call_is_refused() -> None:
    """Reached directly: §9.2 rejects this first, so the loader's own guard would
    otherwise never be exercised — and a guard nobody runs is a guess."""
    from quantlab.sandbox.ast_check import check_source
    from quantlab.strategies_io.loader import _param_schema

    source = _with_params('{"n": 20}')
    with pytest.raises(StrategyLoadError, match="not a ParamSpec"):
        _param_schema(source, check_source(source))


def test_a_class_without_params_yields_an_empty_schema() -> None:
    from quantlab.sandbox.ast_check import check_source
    from quantlab.strategies_io.loader import _find_params_dict, _param_schema

    source = (
        "class Probe:\n"
        '    """A class body holds more than assignments."""\n\n'
        "    warmup_bars = 1\n\n"
        "    def on_bar(self, ctx):\n"
        "        return None\n"
    )
    assert _find_params_dict(ast.parse(source), "Probe") is None
    assert _find_params_dict(ast.parse(source), "Absent") is None
    assert _param_schema(source, check_source(source)) == {}


def test_a_store_that_answers_with_the_wrong_lineage_is_not_believed(
    sources: FileSourceStore,
) -> None:
    """The loader asked for one version and must be handed that version.

    A store returning a chain the id is not in has answered a different question;
    treating the last row as the answer would attribute one strategy's checksum to
    another's source.
    """

    class Other:
        strategy_id = "f" * 16

    class WrongLineageStore:
        def lineage(self, strategy_id: str) -> list[Any]:
            return [Other()]

        def __getattr__(self, name: str) -> Any:  # pragma: no cover - not reached
            raise AssertionError(f"unexpected call: {name}")

    loader = StrategyLoader(WrongLineageStore(), sources)  # type: ignore[arg-type]
    with pytest.raises(StrategyLoadError, match="no such strategy version"):
        loader.verify("0" * 16)
