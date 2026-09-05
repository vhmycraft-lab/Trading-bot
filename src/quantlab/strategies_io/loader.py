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

**Vectorised strategies are probed here.** §9.1 requires the truncation probe of
§14.2 to run automatically at load, before any backtest, because a vectorised
strategy is handed the whole segment and nothing structural stops it reading
forward. The probe evaluates the strategy nine times — through the sandbox, never
in this process — and a strategy that changed its mind about the past is refused
outright (§18.1: a hard reject, there is nothing to fall back to). ``bar_loop``
strategies are not probed at load: ``BarWindow`` refuses a future bar by
construction (INV-3), so there is nothing an empirical check could add.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from quantlab.core.errors import StrategyLoadError
from quantlab.core.hashing import canonical_json, sha256_hex
from quantlab.core.hashing import strategy_id as compute_strategy_id
from quantlab.core.types import BarFrame
from quantlab.core.validation.leakage import ProbeResult, probe_evaluations, require_causal
from quantlab.ports.store import ExperimentStore, SourceStore, StrategyVersionRecord
from quantlab.sandbox.ast_check import (
    DEFAULT_ALLOWED_IMPORTS,
    DEFAULT_MAX_LOGIC_LINES,
    DEFAULT_MAX_PARAMS,
    MAX_SOURCE_LINES,
    AstReport,
    require_safe_source,
)
from quantlab.sandbox.runner import SandboxRunner
from quantlab.strategies_io.probe import sandbox_evaluator

__all__ = [
    "PROBE_REQUIRED_STYLES",
    "SUPPORTED_STYLES",
    "LoadedStrategy",
    "StrategyLoader",
]

#: Styles the engine knows how to consume (spec section 9.1).
SUPPORTED_STYLES: Final[frozenset[str]] = frozenset({"bar_loop", "vectorized"})

#: Styles that must clear the truncation probe before any backtest (section 9.1).
PROBE_REQUIRED_STYLES: Final[frozenset[str]] = frozenset({"vectorized"})


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
    #: The section 14.2 verdict, for a style that requires one. ``None`` for
    #: ``bar_loop``, which ``BarWindow`` protects structurally and never probes.
    probe: ProbeResult | None = None

    @property
    def n_params(self) -> int:
        return len(self.param_schema)


class StrategyLoader:
    """Turns a file of Python into a registered, reproducible strategy version.

    Holds no state between calls: the same source loaded twice produces the same
    id, the same stored bytes and the same row, whichever order the calls arrive in.
    """

    __slots__ = (
        "allowed_imports",
        "engine_class",
        "engine_module",
        "max_lines",
        "max_logic_lines",
        "max_params",
        "probe_bars",
        "probe_cut_points",
        "sandbox",
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
        sandbox: SandboxRunner | None = None,
        engine_module: str = "",
        engine_class: str = "",
        probe_bars: BarFrame | None = None,
        probe_cut_points: Sequence[int] | None = None,
    ) -> None:
        self.store = store
        self.sources = sources
        self.allowed_imports = allowed_imports
        self.max_params = max_params
        self.max_lines = max_lines
        self.max_logic_lines = max_logic_lines
        # Everything the section 14.2 probe needs. Absent, a vectorised strategy
        # cannot be admitted -- see `_probe_if_required`.
        self.sandbox = sandbox
        self.engine_module = engine_module
        self.engine_class = engine_class
        self.probe_bars = probe_bars
        self.probe_cut_points = None if probe_cut_points is None else tuple(probe_cut_points)

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

        # Before the copy and the row: a strategy that fails the probe must leave
        # nothing behind that a run could later cite (sections 9.1, 18.2).
        probe = self._probe_if_required(source, style, report.warmup_bars, identifier)

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
            probe=probe,
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
        # Not re-probed: the probe ran when the version was registered, over bars
        # this call has no reason to hold, and its verdict is a property of the
        # source -- which `verify` has just proved is unchanged.
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
        if style not in SUPPORTED_STYLES:
            raise StrategyLoadError(
                "unknown strategy style",
                style=style,
                supported=sorted(SUPPORTED_STYLES),
            )
        return style

    def _probe_if_required(
        self, source: str, style: str, warmup_bars: int, strategy_id: str
    ) -> ProbeResult | None:
        """Run the section 14.2 probe when the style needs one; refuse if it fails.

        Configured, not optional: a loader asked to admit a vectorised strategy
        without the bars, the sandbox and the engine the probe needs cannot check
        it, and an unchecked vectorised strategy is exactly what section 9.1
        forbids. The refusal names what is missing rather than quietly passing.

        The evaluations run in the sandbox child, never here: at this point the
        source has passed the AST check and nothing else, which is not enough to
        run it in the process that decides what to trust (INV-4).

        Raises:
            StrategyLoadError: the probe is required but not configured, or an
                evaluation failed.
            LeakageDetected: the strategy is not causal.
        """
        if style not in PROBE_REQUIRED_STYLES:
            return None
        missing = [
            name
            for name, value in (
                ("sandbox", self.sandbox),
                ("engine_module", self.engine_module),
                ("engine_class", self.engine_class),
                ("probe_bars", self.probe_bars),
            )
            if not value
        ]
        if missing or self.sandbox is None or self.probe_bars is None:
            raise StrategyLoadError(
                "a vectorized strategy must clear the truncation probe before any "
                "backtest (spec 9.1), and this loader is not configured to run it",
                style=style,
                missing=missing,
            )
        evaluate = sandbox_evaluator(
            self.sandbox,
            source,
            engine_module=self.engine_module,
            engine_class=self.engine_class,
        )
        result = probe_evaluations(
            evaluate,
            self.probe_bars,
            style=style,
            warmup_bars=warmup_bars,
            cut_points=self.probe_cut_points,
        )
        return require_causal(result, strategy_id=strategy_id)

    def _require_version(self, strategy_id: str) -> StrategyVersionRecord:
        """The registered row for ``strategy_id``, or a refusal naming it."""
        row = self.store.get_strategy_version(strategy_id)
        if row is None:
            raise StrategyLoadError("no such strategy version", strategy_id=strategy_id)
        return row


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
