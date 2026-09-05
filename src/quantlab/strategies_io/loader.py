"""Load, check, hash and register a strategy version (master spec §§9, 11, task T20).

This is the gate between a file of Python and a `strategy_version` row that runs
can cite. Four things happen, in this order, and the order is the design:

1. **Check** the source statically (§9.2). A rejected file never reaches step 2,
   so nothing unsafe is ever registered, copied or given an id.
2. **Identify** it: ``strategy_id = sha256(code)[:16]`` (§6). The rule lives in
   :func:`quantlab.core.hashing.strategy_id` and is called, never re-implemented —
   the run id of §11.2 is built from it, and two spellings of the same rule would
   silently split the cache.
3. **Copy** the source into the :class:`~quantlab.ports.store.SourceStore` under
   that id, so the bytes a run was executed from survive edits to the working
   file.
4. **Register** the version with the :class:`~quantlab.ports.store.ExperimentStore`.

The loader never executes the strategy. It reads the declaration out of the parse
the checker already did, because running a module to find out whether it is safe
to run is the thing INV-4 exists to prevent. Execution happens later, in the
sandbox child (§21.3).

**Vectorised strategies are refused.** §9.1 requires the truncation probe of §14.2
to run automatically here before any backtest, and that probe (T19) is not
implemented. Loading one anyway would mean asserting a guarantee nothing checks,
so the loader fails closed and says exactly what is missing. :func:`_probe_vectorized`
is the single place the real probe replaces.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from quantlab.core.errors import StoreError, StrategyLoadError
from quantlab.core.hashing import canonical_json, sha256_hex
from quantlab.core.hashing import strategy_id as compute_strategy_id
from quantlab.ports.store import ExperimentStore, SourceStore, StrategyVersionRecord
from quantlab.sandbox.ast_check import (
    DEFAULT_ALLOWED_IMPORTS,
    DEFAULT_MAX_LOGIC_LINES,
    DEFAULT_MAX_PARAMS,
    MAX_SOURCE_LINES,
    AstReport,
    require_safe_source,
)

__all__ = [
    "SUPPORTED_STYLES",
    "LoadedStrategy",
    "StrategyLoader",
]

#: Styles this build can load. ``vectorized`` is declared by §9.1 but cannot be
#: admitted until the truncation probe of §14.2 exists (T19).
SUPPORTED_STYLES: Final[frozenset[str]] = frozenset({"bar_loop"})

_PROBE_REQUIRED_STYLES: Final[frozenset[str]] = frozenset({"vectorized"})


@dataclass(frozen=True, slots=True)
class LoadedStrategy:
    """A registered strategy version and the source it was registered from.

    Carries the source text rather than an instance: nothing here has executed the
    module, and the sandbox child is what turns these bytes into an object (§21.3).
    """

    strategy_id: str
    family_id: str
    class_name: str
    style: str
    code_path: str
    code_sha256: str
    logic_lines: int
    warmup_bars: int
    param_schema: dict[str, dict[str, Any]]
    source: str
    parent_strategy_id: str | None = None
    author: str = "human"

    @property
    def n_params(self) -> int:
        return len(self.param_schema)


def _probe_vectorized(style: str) -> None:
    """Run the truncation probe a vectorized strategy must pass before any backtest.

    §9.1: *"Vectorized strategies MUST pass the truncation probe (§14.2) before any
    backtest; the probe is executed automatically by the loader."*

    The probe is T19 and does not exist yet, so this refuses instead. That is the
    only honest option: a vectorized strategy computes its signals over the whole
    segment and the engine shifts them by one bar, which is precisely the shape
    that hides look-ahead, and the probe is what would catch it. Loading one now
    would record a guarantee nothing checked.

    When T19 lands, the body of this function becomes the call to
    ``truncation_probe`` and the refusal disappears; nothing else in the loader
    changes.

    Raises:
        StrategyLoadError: always, while T19 is unimplemented.
    """
    raise StrategyLoadError(
        "vectorized strategies cannot be loaded: the truncation probe required by "
        "spec section 9.1 before any backtest is not implemented yet (task T19). "
        "Rewrite the strategy as style='bar_loop', or implement the probe.",
        style=style,
        required_by="spec 9.1 / 14.2",
        blocked_on="T19",
    )


class StrategyLoader:
    """Turns a file of Python into a registered, reproducible strategy version.

    Holds no state between calls: the same source loaded twice produces the same
    id, the same stored bytes and the same row, whichever order the calls arrive in.
    """

    __slots__ = (
        "allowed_imports",
        "max_lines",
        "max_logic_lines",
        "max_params",
        "sources",
        "store",
    )

    def __init__(
        self,
        store: ExperimentStore,
        sources: SourceStore,
        *,
        allowed_imports: tuple[str, ...] = DEFAULT_ALLOWED_IMPORTS,
        max_params: int = DEFAULT_MAX_PARAMS,
        max_lines: int = MAX_SOURCE_LINES,
        max_logic_lines: int = DEFAULT_MAX_LOGIC_LINES,
    ) -> None:
        self.store = store
        self.sources = sources
        self.allowed_imports = allowed_imports
        self.max_params = max_params
        self.max_lines = max_lines
        self.max_logic_lines = max_logic_lines

    def __repr__(self) -> str:
        return f"StrategyLoader(store={self.store!r}, sources={self.sources!r})"

    # -- loading ------------------------------------------------------------
    def load_path(
        self,
        path: str | Path,
        *,
        family: str,
        origin: str = "human",
        author: str = "human",
        parent_strategy_id: str | None = None,
        llm_interaction_id: str | None = None,
    ) -> LoadedStrategy:
        """Load a strategy from a file on disk.

        The file is read as UTF-8 bytes and hashed exactly as read: the id is of
        the source, not of a normalised rendering of it.
        """
        source_path = Path(path)
        try:
            source = source_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise StrategyLoadError(f"cannot read strategy source: {exc}", path=str(path)) from exc
        return self.load_source(
            source,
            family=family,
            origin=origin,
            author=author,
            parent_strategy_id=parent_strategy_id,
            llm_interaction_id=llm_interaction_id,
        )

    def load_source(
        self,
        source: str,
        *,
        family: str,
        origin: str = "human",
        author: str = "human",
        parent_strategy_id: str | None = None,
        llm_interaction_id: str | None = None,
    ) -> LoadedStrategy:
        """Check, identify, copy and register ``source``.

        Idempotent: the same source is the same version, so a second load returns
        the same id, leaves the stored copy untouched and reuses the existing row.

        Raises:
            StrategySafetyError: the source broke a rule of §9.2. Nothing is
                registered or written.
            StrategyLoadError: the declaration cannot be read, or the style needs
                a check this build cannot perform.
        """
        report = require_safe_source(
            source,
            allowed_imports=self.allowed_imports,
            max_params=self.max_params,
            max_lines=self.max_lines,
            max_logic_lines=self.max_logic_lines,
        )
        style = self._require_supported_style(report)
        schema = _param_schema(source, report)

        identifier = compute_strategy_id(source)
        code_sha256 = sha256_hex(source)

        # Order matters on failure as much as on success: the copy lands before
        # the row, so a row can never point at source that was never written.
        code_path = self.sources.write_source(identifier, source)
        family_row = self.store.create_family(name=family, origin=origin)
        self.store.add_strategy_version(
            strategy_id=identifier,
            family_id=family_row.family_id,
            code_path=code_path,
            code_sha256=code_sha256,
            class_name=report.class_name,
            param_schema_json=_schema_json(schema),
            style=style,
            author=author,
            logic_lines=report.logic_lines,
            parent_strategy_id=parent_strategy_id,
            llm_interaction_id=llm_interaction_id,
        )
        return LoadedStrategy(
            strategy_id=identifier,
            family_id=family_row.family_id,
            class_name=report.class_name,
            style=style,
            code_path=code_path,
            code_sha256=code_sha256,
            logic_lines=report.logic_lines,
            warmup_bars=report.warmup_bars,
            param_schema=schema,
            source=source,
            parent_strategy_id=parent_strategy_id,
            author=author,
        )

    # -- reading back -------------------------------------------------------
    def verify(self, strategy_id: str) -> str:
        """Return the stored source, having proved it is the source that was registered.

        Two independent checks, because they fail differently. The recorded
        ``code_sha256`` catches a file edited after registration. Recomputing the
        *id* from the bytes catches a row whose hash column was edited to match a
        tampered file — an attacker who changes one and not the other is caught by
        the first; one who changes both is caught by the second, because the id is
        also the primary key every run cites.

        Raises:
            StrategyLoadError: the stored source is not what was registered.
            StoreError: there is no such version, or no stored source.
        """
        row = self._require_version(strategy_id)
        source = self.sources.read_source(strategy_id)

        actual_sha = sha256_hex(source)
        if actual_sha != row.code_sha256:
            raise StrategyLoadError(
                "stored strategy source does not match its recorded checksum",
                strategy_id=strategy_id,
                recorded=row.code_sha256,
                actual=actual_sha,
            )
        recomputed_id = compute_strategy_id(source)
        if recomputed_id != strategy_id:
            raise StrategyLoadError(
                "stored strategy source does not hash to its own id",
                strategy_id=strategy_id,
                recomputed=recomputed_id,
            )
        return source

    def load_registered(self, strategy_id: str) -> LoadedStrategy:
        """Rebuild a :class:`LoadedStrategy` from the store, verifying it first.

        This is the path a later run takes: everything it needs comes from the
        recorded row and the stored bytes, never from the working tree, so a run
        reproduces what it executed rather than whatever the file says today
        (INV-7).
        """
        source = self.verify(strategy_id)
        row = self._require_version(strategy_id)
        report = require_safe_source(
            source,
            allowed_imports=self.allowed_imports,
            max_params=self.max_params,
            max_lines=self.max_lines,
            max_logic_lines=self.max_logic_lines,
        )
        style = self._require_supported_style(report)
        return LoadedStrategy(
            strategy_id=strategy_id,
            family_id=row.family_id,
            class_name=report.class_name,
            style=style,
            # The recorded path, not a fresh write: reading back a version must
            # not touch the store it is reading from.
            code_path=row.code_path,
            code_sha256=row.code_sha256,
            logic_lines=row.logic_lines,
            warmup_bars=report.warmup_bars,
            param_schema=_param_schema(source, report),
            source=source,
            parent_strategy_id=row.parent_strategy_id,
        )

    def lineage(self, strategy_id: str) -> list[StrategyVersionRecord]:
        """Every ancestor of a version, oldest first (spec section 11.1)."""
        return self.store.lineage(strategy_id)

    # -- internals ----------------------------------------------------------
    def _require_supported_style(self, report: AstReport) -> str:
        style = report.style
        if style in _PROBE_REQUIRED_STYLES:
            _probe_vectorized(style)
        if style not in SUPPORTED_STYLES:
            raise StrategyLoadError(
                "unknown strategy style",
                style=style,
                supported=sorted(SUPPORTED_STYLES),
            )
        return style

    def _require_version(self, strategy_id: str) -> StrategyVersionRecord:
        """The registered row for ``strategy_id``.

        Read through ``lineage`` because that is what the port offers, and the
        chain ends with the version itself. A store that has never heard of the id
        is the loader's problem to report, not the store's: the caller asked to
        load a strategy, so the answer is a ``StrategyLoadError``.
        """
        try:
            chain = self.store.lineage(strategy_id)
        except StoreError as exc:
            raise StrategyLoadError("no such strategy version", strategy_id=strategy_id) from exc
        for row in chain:
            if row.strategy_id == strategy_id:
                return row
        raise StrategyLoadError("no such strategy version", strategy_id=strategy_id)


# ---------------------------------------------------------------------------
# reading the declaration without running it
# ---------------------------------------------------------------------------
def _schema_json(schema: dict[str, dict[str, Any]]) -> str:
    """The stored form of the schema: canonical, so it hashes and diffs stably."""
    return canonical_json(schema)


def _param_schema(source: str, report: AstReport) -> dict[str, dict[str, Any]]:
    """Read ``params`` out of the source as data.

    The AST checker has already established that ``params`` is a dict literal of
    ``ParamSpec(...)`` calls with bounds (§9.2), so every argument here is a
    literal and ``ast.literal_eval`` can read them without running anything.

    Raises:
        StrategyLoadError: an argument is not a literal. The checker permits an
            expression there; the schema this produces is stored and later used to
            mutate parameters, and a bound that is only knowable by running the
            module is not a bound.
    """
    tree = ast.parse(source)
    params = _find_params_dict(tree, report.class_name)
    if params is None:
        return {}

    schema: dict[str, dict[str, Any]] = {}
    for key, value in zip(params.keys, params.values, strict=True):
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            raise StrategyLoadError("parameter names must be string literals")
        if not isinstance(value, ast.Call):
            raise StrategyLoadError("parameter is not a ParamSpec(...) call", param=key.value)
        entry: dict[str, Any] = {}
        for keyword in value.keywords:
            if keyword.arg is None:
                raise StrategyLoadError(
                    "ParamSpec does not accept **kwargs in a declaration", param=key.value
                )
            try:
                entry[keyword.arg] = ast.literal_eval(keyword.value)
            except (ValueError, SyntaxError, TypeError) as exc:
                raise StrategyLoadError(
                    "ParamSpec arguments must be literals so the schema can be stored",
                    param=key.value,
                    argument=keyword.arg,
                ) from exc
        schema[key.value] = entry
    return schema


def _find_params_dict(tree: ast.Module, class_name: str) -> ast.Dict | None:
    """The ``params`` dict literal of the declared class, or ``None``."""
    for node in tree.body:
        if not (isinstance(node, ast.ClassDef) and node.name == class_name):
            continue
        for statement in node.body:
            if isinstance(statement, ast.Assign):
                targets: list[ast.expr] = list(statement.targets)
            elif isinstance(statement, ast.AnnAssign):
                targets = [statement.target]
            else:
                continue
            for target in targets:
                if isinstance(target, ast.Name) and target.id == "params":
                    value = statement.value
                    return value if isinstance(value, ast.Dict) else None
    return None
