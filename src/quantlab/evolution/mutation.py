"""Mutation operators (master spec section 13.4).

A child receives between one and ``max_mutations_per_child`` edits, drawn by
category from ``parameter_rate`` / ``structural_rate``. Every edit is recorded as
an :class:`AppliedMutation` carrying the operator, the genome path it touched, the
before and after values, and the RNG seed that produced it — which is what makes
INV-10 replayable: re-applying a child's recorded mutations to its parent's genome
must reproduce the child byte for byte.

That requirement shapes the record. Each mutation stores the **resulting value at
its path**, not the recipe that produced it, so replay is a pure application of
edits and needs no random draws of its own. The seed is kept because section 13.4
requires it and because it is what lets an audit ask *why* a value was chosen, but
replay never needs it.

**Preventing invalid strategies**, section 13.4's four layers, in order:

1. *Validity by construction* — an operator emits a genome edit, and a genome
   that breaks a section 9.6 rule cannot be constructed at all, so an invalid
   edit raises here rather than escaping into the population.
2. *Redraw* — an operator that produced an invalid genome is retried with a fresh
   draw, up to ``max_repair_attempts``. Failures are counted per operator for
   ``generation.stats_json``.
3. *Fall back* — if every attempt fails the slot is left empty and
   :attr:`MutationResult.fell_back` is set, so the caller fills it with an
   immigrant. A child is never repaired by relaxing a rule.
4. *Defence in depth* — the compiled source still faces the AST check, the
   sandbox and the leakage probe before it can be scored.

``opaque`` candidates admit parameter mutations only: their structure is
free-form Python that cannot be edited safely, so a structural operator drawn for
one is re-drawn as a parameter operator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Final, Literal

import numpy as np
from pydantic import ValidationError

from quantlab.core.config import MutationSettings
from quantlab.core.errors import StrategyError
from quantlab.core.genome import Condition, ConditionTree, Operand, StrategyGenome
from quantlab.core.hashing import canonical_json
from quantlab.core.strategy import ParamSpec
from quantlab.core.types import RiskSpec, SizingSpec
from quantlab.evolution.library import RISK_CONTROL_RANGES, OperatorLibrary, required_warmup

__all__ = [
    "PARAMETER_OPERATORS",
    "STRUCTURAL_OPERATORS",
    "AppliedMutation",
    "MutationResult",
    "Phenotype",
    "mutate",
]

#: Section 13.4's parameter operators.
PARAMETER_OPERATORS: Final[tuple[str, ...]] = (
    "perturb_numeric",
    "jump_numeric",
    "toggle_bool",
    "resample_categorical",
    "perturb_risk",
    "perturb_sizing",
    "toggle_risk_control",
)

#: Section 13.4's structural operators, in the order that table lists them.
STRUCTURAL_OPERATORS: Final[tuple[str, ...]] = (
    "add_confirmation",
    "remove_confirmation",
    "modify_entry",
    "modify_exit",
    "add_filter",
    "remove_filter",
    "replace_indicator",
    "change_tree_mode",
)

#: Numeric ``SizingSpec`` fields a mutation may move.
_SIZING_FIELDS: Final[tuple[str, ...]] = ("fraction", "target_vol_annual", "atr_risk_pct")

_EPS: Final[float] = 1e-12


@dataclass(frozen=True, slots=True)
class Phenotype:
    """A candidate as a mutation sees it: a structure, values, and their bounds.

    This is the store's ``candidate`` row without its identity columns —
    ``genome_json`` (absent for ``opaque``), ``params_json``, and ``kind``. Kept
    separate from the row so that mutation can be tested, replayed and reasoned
    about without a database.
    """

    kind: Literal["genome", "opaque"] = "genome"
    genome: StrategyGenome | None = None
    #: Concrete parameter values the candidate is evaluated with.
    params: Mapping[str, Any] = field(default_factory=dict)
    #: The bounds those values move inside. Taken from the genome when there is
    #: one, and from the strategy version's ``param_schema`` for an ``opaque``
    #: candidate.
    schema: Mapping[str, ParamSpec] = field(default_factory=dict)
    risk: RiskSpec = field(default_factory=RiskSpec)
    sizing: SizingSpec = field(default_factory=SizingSpec)

    def __post_init__(self) -> None:
        if self.kind == "genome" and self.genome is None:
            raise StrategyError("a genome candidate must carry a genome")
        if self.kind == "opaque" and self.genome is not None:
            raise StrategyError("an opaque candidate has no genome to carry")

    @classmethod
    def from_genome(
        cls, genome: StrategyGenome, params: Mapping[str, Any] | None = None
    ) -> Phenotype:
        """Build a phenotype from a genome, taking bounds and controls from it."""
        values = (
            dict(params)
            if params is not None
            else {name: spec.default for name, spec in genome.params.items()}
        )
        return cls(
            kind="genome",
            genome=genome,
            params=values,
            schema=dict(genome.params),
            risk=genome.risk,
            sizing=genome.sizing,
        )

    def rebuilt(self) -> Phenotype:
        """This phenotype with its genome brought back into agreement with it.

        The genome owns ``params``, ``risk`` and ``sizing`` (section 9.6), so an
        edit to any of those has to land there too, or the compiled source and
        the recorded values would describe different strategies. Warm-up is
        recomputed at the same time: a structural edit can lengthen the longest
        lookback, and rule 6 is a lower bound.
        """
        if self.kind != "genome" or self.genome is None:
            return self
        conditions = list(self.genome.conditions())
        updated = self.genome.model_copy(
            update={
                "params": dict(self.schema),
                "risk": self.risk,
                "sizing": self.sizing,
                # Recomputed, never ratcheted. This was `max(existing, required)`,
                # which could only ever rise: a lineage that once held a
                # long-lookback indicator kept its warm-up after mutating that
                # indicator away, and was thereafter scored on a shorter, later
                # window than its competitors. Warm-up follows the structure that
                # is actually there.
                "warmup_bars": required_warmup(conditions, self.schema),
            }
        )
        return replace(self, genome=StrategyGenome.model_validate(updated.model_dump()))


@dataclass(frozen=True, slots=True)
class AppliedMutation:
    """One recorded edit — the ``mutation`` row of section 6."""

    category: Literal["parameter", "structural"]
    operator: str
    #: Genome path, e.g. ``entry.conditions[1].right``.
    target: str
    before_json: str
    after_json: str
    rng_seed: int

    def describe(self) -> str:
        return f"{self.operator} at {self.target}"


@dataclass(frozen=True, slots=True)
class MutationResult:
    """A mutated child, or the reason there isn't one."""

    child: Phenotype | None
    mutations: tuple[AppliedMutation, ...] = ()
    attempts: int = 0
    #: Redraws per operator, for ``generation.stats_json`` (section 13.4).
    failures: Mapping[str, int] = field(default_factory=dict)

    @property
    def fell_back(self) -> bool:
        """True when every attempt failed and the slot needs an immigrant."""
        return self.child is None

    @property
    def genome(self) -> StrategyGenome | None:
        return None if self.child is None else self.child.genome


# ---------------------------------------------------------------------------
# the public entry point
# ---------------------------------------------------------------------------
def mutate(
    parent: Phenotype,
    settings: MutationSettings,
    rng: np.random.Generator,
    library: OperatorLibrary,
) -> MutationResult:
    """Produce one mutated child of ``parent`` (spec section 13.4).

    Deterministic in ``rng``: two calls with generators in the same state produce
    identical children and identical mutation records, which is what lets a
    generation be reproduced from its seed (INV-7).

    Returns:
        A :class:`MutationResult` whose ``child`` is ``None`` when every attempt
        failed. The caller fills that slot with an immigrant (section 13.4, layer
        3) rather than relaxing a rule.
    """
    n_mutations = int(rng.integers(1, settings.max_mutations_per_child + 1))
    current = parent
    applied: list[AppliedMutation] = []
    failures: dict[str, int] = {}
    attempts = 0

    for _ in range(n_mutations):
        outcome = _one_mutation(current, settings, rng, library, failures)
        attempts += outcome.attempts
        if outcome.mutation is None:
            # Every redraw for this edit failed. Section 13.4 layer 3: the slot
            # falls back rather than being patched into validity.
            return MutationResult(
                child=None, mutations=tuple(applied), attempts=attempts, failures=dict(failures)
            )
        assert outcome.child is not None
        current = outcome.child
        applied.append(outcome.mutation)

    if not applied:  # pragma: no cover - n_mutations is at least 1
        return MutationResult(child=None, attempts=attempts, failures=dict(failures))
    return MutationResult(
        child=current, mutations=tuple(applied), attempts=attempts, failures=dict(failures)
    )


@dataclass(frozen=True, slots=True)
class _Outcome:
    child: Phenotype | None
    mutation: AppliedMutation | None
    attempts: int


def _one_mutation(
    parent: Phenotype,
    settings: MutationSettings,
    rng: np.random.Generator,
    library: OperatorLibrary,
    failures: dict[str, int],
) -> _Outcome:
    """Apply one operator, redrawing up to ``max_repair_attempts`` times."""
    attempts = 0
    for _ in range(settings.max_repair_attempts + 1):
        attempts += 1
        seed = int(rng.integers(0, 2**31 - 1))
        draw = np.random.default_rng(seed)
        category = _draw_category(parent, settings, draw)
        operator = _draw_operator(category, parent, settings, draw)
        try:
            child, mutation = _apply(parent, category, operator, draw, library, settings, seed)
        except (StrategyError, ValidationError, ValueError):
            failures[operator] = failures.get(operator, 0) + 1
            continue
        return _Outcome(child=child, mutation=mutation, attempts=attempts)
    return _Outcome(child=None, mutation=None, attempts=attempts)


def _draw_category(
    parent: Phenotype, settings: MutationSettings, rng: np.random.Generator
) -> Literal["parameter", "structural"]:
    """Parameter or structural, per section 13.4's rates.

    An ``opaque`` candidate always gets a parameter mutation: its structure is
    free-form Python, and section 13.4 says a structural operator drawn for one is
    re-drawn as a parameter operator rather than attempted and failed.
    """
    if parent.kind == "opaque":
        return "parameter"
    total = settings.parameter_rate + settings.structural_rate
    if total <= _EPS:  # pragma: no cover - config validation forbids this
        return "parameter"
    return "parameter" if rng.uniform(0.0, total) < settings.parameter_rate else "structural"


def _applicable_parameter_operators(parent: Phenotype) -> tuple[str, ...]:
    """Parameter operators that have something to act on in this phenotype.

    A genome's parameters are numeric by construction (section 9.6), so
    ``toggle_bool`` and ``resample_categorical`` can never apply to one. Drawing
    them anyway would spend the repair budget of section 13.4's layer 2 on
    operators that are not merely unlucky but impossible, and push otherwise
    healthy children into the immigrant fallback.

    This narrows *which operator is drawn*, never what an operator may do. An
    operator that is applicable and still fails — a comparator that produced a
    duplicate condition, a perturbation that landed back on its own value — is
    redrawn exactly as before.
    """
    kinds = {spec.kind for spec in parent.schema.values()}
    numeric = bool(kinds & {"int", "float"})
    categorical = any(
        spec.kind == "categorical" and len(spec.choices or ()) > 1
        for spec in parent.schema.values()
    )
    applicable = {
        "perturb_numeric": numeric,
        "jump_numeric": numeric,
        "toggle_bool": "bool" in kinds,
        "resample_categorical": categorical,
        "perturb_risk": parent.risk.model_dump(exclude_none=True) != {},
        "perturb_sizing": True,
        "toggle_risk_control": True,
    }
    return tuple(name for name in PARAMETER_OPERATORS if applicable[name])


def _draw_operator(
    category: str, parent: Phenotype, settings: MutationSettings, rng: np.random.Generator
) -> str:
    if category == "parameter":
        operators = _applicable_parameter_operators(parent)
        if not operators:  # pragma: no cover - sizing is always applicable
            raise StrategyError("no parameter operator can act on this candidate")
        return str(rng.choice(operators))
    weights = np.array(
        [getattr(settings.structural, name) for name in STRUCTURAL_OPERATORS], dtype="float64"
    )
    return str(rng.choice(STRUCTURAL_OPERATORS, p=weights / weights.sum()))


def _apply(
    parent: Phenotype,
    category: Literal["parameter", "structural"],
    operator: str,
    rng: np.random.Generator,
    library: OperatorLibrary,
    settings: MutationSettings,
    seed: int,
) -> tuple[Phenotype, AppliedMutation]:
    """Run one operator, or raise so the caller redraws."""
    if category == "parameter":
        target, before, after, child = _apply_parameter(parent, operator, rng, library, settings)
    else:
        target, before, after, child = _apply_structural(parent, operator, rng, library)
    return child.rebuilt(), AppliedMutation(
        category=category,
        operator=operator,
        target=target,
        before_json=canonical_json(before),
        after_json=canonical_json(after),
        rng_seed=seed,
    )


# ---------------------------------------------------------------------------
# parameter operators
# ---------------------------------------------------------------------------
def _apply_parameter(
    parent: Phenotype,
    operator: str,
    rng: np.random.Generator,
    library: OperatorLibrary,
    settings: MutationSettings,
) -> tuple[str, Any, Any, Phenotype]:
    if operator in ("perturb_risk", "toggle_risk_control"):
        return _mutate_risk(parent, operator, rng, library, settings)
    if operator == "perturb_sizing":
        return _mutate_sizing(parent, rng, settings)
    return _mutate_param_value(parent, operator, rng, settings)


def _mutate_param_value(
    parent: Phenotype,
    operator: str,
    rng: np.random.Generator,
    settings: MutationSettings,
) -> tuple[str, Any, Any, Phenotype]:
    """Move one declared parameter, snapped to its own bounds and step."""
    wanted = {
        "perturb_numeric": ("int", "float"),
        "jump_numeric": ("int", "float"),
        "toggle_bool": ("bool",),
        "resample_categorical": ("categorical",),
    }[operator]
    names = sorted(name for name, spec in parent.schema.items() if spec.kind in wanted)
    if not names:
        raise StrategyError("no parameter of the kind this operator needs", operator=operator)

    name = str(rng.choice(names))
    spec = parent.schema[name]
    before = parent.params.get(name, spec.default)
    after = _new_value(operator, spec, before, rng, settings)
    if after == before:
        raise StrategyError("mutation did not change anything", operator=operator, param=name)

    params = {**parent.params, name: after}
    return f"params.{name}", before, after, replace(parent, params=params)


def _new_value(
    operator: str,
    spec: ParamSpec,
    before: Any,
    rng: np.random.Generator,
    settings: MutationSettings,
) -> Any:
    parameters = settings.parameter
    if operator == "toggle_bool":
        return not bool(before)
    if operator == "resample_categorical":
        choices = [choice for choice in (spec.choices or ()) if choice != before]
        if not choices:
            raise StrategyError("a categorical with one choice cannot be resampled")
        return str(rng.choice(choices))
    if operator == "jump_numeric":
        return spec.clamp(rng.uniform(float(spec.low or 0.0), float(spec.high or 1.0)))
    # perturb_numeric: x *= 1 +/- U(0, perturb_pct), snapped to the spec.
    factor = 1.0 + float(rng.uniform(-parameters.perturb_pct, parameters.perturb_pct))
    return spec.clamp(float(before) * factor)


def _mutate_risk(
    parent: Phenotype,
    operator: str,
    rng: np.random.Generator,
    library: OperatorLibrary,
    settings: MutationSettings,
) -> tuple[str, Any, Any, Phenotype]:
    """Move or switch one of section 8.6's engine-applied exits."""
    control = str(rng.choice(sorted(RISK_CONTROL_RANGES)))
    before = getattr(parent.risk, control)

    if operator == "toggle_risk_control":
        after = None if before is not None else library.draw_risk_value(rng, control)
    else:
        if before is None:
            raise StrategyError("cannot perturb a control that is switched off", control=control)
        span = RISK_CONTROL_RANGES[control]
        factor = 1.0 + float(
            rng.uniform(-settings.parameter.risk_perturb_pct, settings.parameter.risk_perturb_pct)
        )
        moved = min(max(float(before) * factor, span[0]), span[1])
        after = round(moved) if control == "time_stop_bars" else moved
    if after == before:
        raise StrategyError("risk mutation did not change anything", control=control)

    risk = RiskSpec.model_validate({**parent.risk.model_dump(), control: after})
    return f"risk.{control}", before, after, replace(parent, risk=risk)


def _mutate_sizing(
    parent: Phenotype, rng: np.random.Generator, settings: MutationSettings
) -> tuple[str, Any, Any, Phenotype]:
    """Move one numeric field of the sizing spec that is actually in use."""
    live = [name for name in _SIZING_FIELDS if getattr(parent.sizing, name) is not None]
    if not live:  # pragma: no cover - `fraction` always has a value
        raise StrategyError("no sizing field is set")
    name = str(rng.choice(live))
    before = float(getattr(parent.sizing, name))
    factor = 1.0 + float(
        rng.uniform(-settings.parameter.risk_perturb_pct, settings.parameter.risk_perturb_pct)
    )
    after = max(_EPS, before * factor)
    if name == "fraction":
        after = min(after, 1.0)
    if after == before:
        raise StrategyError("sizing mutation did not change anything", field=name)

    sizing = SizingSpec.model_validate({**parent.sizing.model_dump(), name: after})
    return f"sizing.{name}", before, after, replace(parent, sizing=sizing)


# ---------------------------------------------------------------------------
# structural operators
# ---------------------------------------------------------------------------
def _apply_structural(
    parent: Phenotype, operator: str, rng: np.random.Generator, library: OperatorLibrary
) -> tuple[str, Any, Any, Phenotype]:
    genome = parent.genome
    if genome is None:  # pragma: no cover - `_draw_category` never gets here
        raise StrategyError("an opaque candidate has no structure to mutate")
    handler = {
        "add_confirmation": _add_confirmation,
        "remove_confirmation": _remove_confirmation,
        "modify_entry": _modify_entry,
        "modify_exit": _modify_exit,
        "add_filter": _add_filter,
        "remove_filter": _remove_filter,
        "replace_indicator": _replace_indicator,
        "change_tree_mode": _change_tree_mode,
    }[operator]
    return handler(parent, genome, rng, library)


def _tree_name(rng: np.random.Generator) -> str:
    return str(rng.choice(["entry", "exit"]))


def _with_tree(genome: StrategyGenome, name: str, tree: ConditionTree) -> StrategyGenome:
    return StrategyGenome.model_validate(genome.model_copy(update={name: tree}).model_dump())


def _add_confirmation(
    parent: Phenotype, genome: StrategyGenome, rng: np.random.Generator, library: OperatorLibrary
) -> tuple[str, Any, Any, Phenotype]:
    name = _tree_name(rng)
    tree: ConditionTree = getattr(genome, name)
    condition = library.draw_condition(rng, params=genome.params, avoid=tree.conditions)
    grown = ConditionTree(mode=tree.mode, conditions=(*tree.conditions, condition))
    target = f"{name}.conditions[{len(tree.conditions)}]"
    return (
        target,
        None,
        condition.model_dump(mode="json"),
        replace(parent, genome=_with_tree(genome, name, grown)),
    )


def _remove_confirmation(
    parent: Phenotype,
    genome: StrategyGenome,
    rng: np.random.Generator,
    library: OperatorLibrary,  # noqa: ARG001 - uniform signature for the dispatch table
) -> tuple[str, Any, Any, Phenotype]:
    """Drop a condition — never the last one in ``entry`` (section 13.4)."""
    options = [name for name in ("entry", "exit") if len(getattr(genome, name).conditions) > 0]
    if "entry" in options and len(genome.entry.conditions) <= 1:
        options.remove("entry")
    if not options:
        raise StrategyError("nothing may be removed without emptying the entry tree")

    name = str(rng.choice(sorted(options)))
    tree: ConditionTree = getattr(genome, name)
    index = int(rng.integers(0, len(tree.conditions)))
    kept = tuple(c for i, c in enumerate(tree.conditions) if i != index)
    shrunk = ConditionTree(mode=tree.mode, conditions=kept)
    return (
        f"{name}.conditions[{index}]",
        tree.conditions[index].model_dump(mode="json"),
        None,
        replace(parent, genome=_with_tree(genome, name, shrunk)),
    )


def _modify_tree(
    parent: Phenotype,
    genome: StrategyGenome,
    rng: np.random.Generator,
    library: OperatorLibrary,
    name: str,
) -> tuple[str, Any, Any, Phenotype]:
    """Change one condition's comparator, or replace one of its operands."""
    tree: ConditionTree = getattr(genome, name)
    if not tree.conditions:
        raise StrategyError("that tree has no condition to modify", tree=name)
    index = int(rng.integers(0, len(tree.conditions)))
    before = tree.conditions[index]

    # The record names the *field* that changed, not the whole condition: replay
    # sets exactly what was set, and an audit reads one before/after pair rather
    # than diffing two condition documents to find the edit.
    if rng.uniform() < 0.5:
        field_name = "op"
        old_value: Any = before.op
        new_value: Any = str(rng.choice(sorted(set(_ALL_OPS) - {before.op})))
    else:
        field_name = str(rng.choice(["left", "right"]))
        keeper: Operand = getattr(before, "right" if field_name == "left" else "left")
        domain = keeper.domain
        drawn = (
            library.draw_indicator_operand(rng, domain=domain)
            if domain is not None
            else library.draw_indicator_operand(rng)
        )
        old_value = getattr(before, field_name).model_dump(mode="json")
        new_value = drawn.model_dump(mode="json")

    after = Condition.model_validate({**before.model_dump(), field_name: new_value})
    replaced = tuple(after if i == index else c for i, c in enumerate(tree.conditions))
    updated = ConditionTree(mode=tree.mode, conditions=replaced)
    return (
        f"{name}.conditions[{index}].{field_name}",
        old_value,
        new_value,
        replace(parent, genome=_with_tree(genome, name, updated)),
    )


_ALL_OPS: Final[tuple[str, ...]] = (
    "<",
    "<=",
    ">",
    ">=",
    "cross_above",
    "cross_below",
)


def _modify_entry(
    parent: Phenotype, genome: StrategyGenome, rng: np.random.Generator, library: OperatorLibrary
) -> tuple[str, Any, Any, Phenotype]:
    return _modify_tree(parent, genome, rng, library, "entry")


def _modify_exit(
    parent: Phenotype, genome: StrategyGenome, rng: np.random.Generator, library: OperatorLibrary
) -> tuple[str, Any, Any, Phenotype]:
    return _modify_tree(parent, genome, rng, library, "exit")


def _add_filter(
    parent: Phenotype, genome: StrategyGenome, rng: np.random.Generator, library: OperatorLibrary
) -> tuple[str, Any, Any, Phenotype]:
    condition = library.draw_condition(rng, params=genome.params, avoid=genome.filters)
    grown = (*genome.filters, condition)
    updated = StrategyGenome.model_validate(
        genome.model_copy(update={"filters": grown}).model_dump()
    )
    return (
        f"filters[{len(genome.filters)}]",
        None,
        condition.model_dump(mode="json"),
        replace(parent, genome=updated),
    )


def _remove_filter(
    parent: Phenotype,
    genome: StrategyGenome,
    rng: np.random.Generator,
    library: OperatorLibrary,  # noqa: ARG001 - uniform signature for the dispatch table
) -> tuple[str, Any, Any, Phenotype]:
    if not genome.filters:
        raise StrategyError("there is no filter to remove")
    index = int(rng.integers(0, len(genome.filters)))
    kept = tuple(c for i, c in enumerate(genome.filters) if i != index)
    updated = StrategyGenome.model_validate(
        genome.model_copy(update={"filters": kept}).model_dump()
    )
    return (
        f"filters[{index}]",
        genome.filters[index].model_dump(mode="json"),
        None,
        replace(parent, genome=updated),
    )


def _replace_indicator(
    parent: Phenotype, genome: StrategyGenome, rng: np.random.Generator, library: OperatorLibrary
) -> tuple[str, Any, Any, Phenotype]:
    """Swap an indicator operand for another measured in the same units."""
    sites = [
        (name, index, side)
        for name in ("entry", "exit")
        for index, condition in enumerate(getattr(genome, name).conditions)
        for side in ("left", "right")
        if getattr(condition, side).kind == "indicator"
    ]
    if not sites:
        raise StrategyError("no indicator operand to replace")

    name, index, side = sites[int(rng.integers(0, len(sites)))]
    tree: ConditionTree = getattr(genome, name)
    condition = tree.conditions[index]
    before: Operand = getattr(condition, side)
    options = library.compatible_indicators(before)
    if not options:
        raise StrategyError("no compatible indicator to swap in", indicator=before.name)

    after_operand = before.model_copy(update={"name": str(rng.choice(sorted(options)))})
    updated_condition = condition.model_copy(update={side: after_operand})
    replaced = tuple(updated_condition if i == index else c for i, c in enumerate(tree.conditions))
    return (
        f"{name}.conditions[{index}].{side}",
        before.model_dump(mode="json"),
        after_operand.model_dump(mode="json"),
        replace(
            parent,
            genome=_with_tree(genome, name, ConditionTree(mode=tree.mode, conditions=replaced)),
        ),
    )


def _change_tree_mode(
    parent: Phenotype,
    genome: StrategyGenome,
    rng: np.random.Generator,
    library: OperatorLibrary,  # noqa: ARG001 - uniform signature for the dispatch table
) -> tuple[str, Any, Any, Phenotype]:
    name = _tree_name(rng)
    tree: ConditionTree = getattr(genome, name)
    if len(tree.conditions) < 2:
        raise StrategyError("all and any agree on a single condition", tree=name)
    after = "any" if tree.mode == "all" else "all"
    flipped = ConditionTree(mode=after, conditions=tree.conditions)  # type: ignore[arg-type]
    return (
        f"{name}.mode",
        tree.mode,
        after,
        replace(parent, genome=_with_tree(genome, name, flipped)),
    )


def replay(parent: Phenotype, mutations: Sequence[AppliedMutation]) -> Phenotype:
    """Re-apply recorded mutations to a parent — INV-10's checkable half.

    Each record carries the value at its path *after* the edit, so replay is a
    pure application and draws nothing. That is why INV-10 holds without the
    replaying process having to reproduce the original RNG stream.

    Raises:
        StrategyError: a path names something the parent does not have. That is a
            corrupt lineage rather than a mutation that failed, so it raises
            rather than being skipped.
    """
    import json

    current = parent
    for mutation in mutations:
        after = json.loads(mutation.after_json)
        current = _set_path(current, mutation.target, after)
    return current.rebuilt()


def _set_path(phenotype: Phenotype, path: str, value: Any) -> Phenotype:
    """Apply one recorded edit at ``path`` (see the grammar in :func:`mutate`)."""
    if path.startswith("params."):
        return replace(phenotype, params={**phenotype.params, path.removeprefix("params."): value})
    if path.startswith("risk."):
        field_name = path.removeprefix("risk.")
        return replace(
            phenotype,
            risk=RiskSpec.model_validate({**phenotype.risk.model_dump(), field_name: value}),
        )
    if path.startswith("sizing."):
        field_name = path.removeprefix("sizing.")
        return replace(
            phenotype,
            sizing=SizingSpec.model_validate({**phenotype.sizing.model_dump(), field_name: value}),
        )
    genome = phenotype.genome
    if genome is None:
        raise StrategyError("this path needs a genome", path=path)
    return replace(phenotype, genome=_set_genome_path(genome, path, value))


def _set_genome_path(genome: StrategyGenome, path: str, value: Any) -> StrategyGenome:
    head, _, rest = path.partition(".")
    if head.startswith("filters["):
        return _set_condition_list(genome, "filters", _index_of(head), rest, value)
    if head in ("entry", "exit"):
        sub, _, tail = rest.partition(".")
        if sub == "mode":
            tree: ConditionTree = getattr(genome, head)
            return _with_tree(genome, head, ConditionTree(mode=value, conditions=tree.conditions))
        if sub.startswith("conditions["):
            return _set_condition_list(genome, head, _index_of(sub), tail, value)
    raise StrategyError("unknown genome path", path=path)


def _index_of(token: str) -> int:
    try:
        return int(token[token.index("[") + 1 : token.index("]")])
    except (ValueError, IndexError) as exc:
        raise StrategyError("malformed path index", token=token) from exc


def _set_condition_list(
    genome: StrategyGenome, where: str, index: int, tail: str, value: Any
) -> StrategyGenome:
    """Set, append or remove one condition, and optionally one field inside it."""
    conditions = list(genome.filters if where == "filters" else getattr(genome, where).conditions)
    if tail:
        if not 0 <= index < len(conditions):
            raise StrategyError("no condition at that index", where=where, index=index)
        conditions[index] = Condition.model_validate(
            {**conditions[index].model_dump(), tail: value}
        )
    elif value is None:
        if not 0 <= index < len(conditions):
            raise StrategyError("no condition at that index", where=where, index=index)
        conditions.pop(index)
    elif index == len(conditions):
        conditions.append(Condition.model_validate(value))
    elif 0 <= index < len(conditions):
        conditions[index] = Condition.model_validate(value)
    else:
        raise StrategyError("condition index is past the end", where=where, index=index)

    if where == "filters":
        return StrategyGenome.model_validate(
            genome.model_copy(update={"filters": tuple(conditions)}).model_dump()
        )
    tree: ConditionTree = getattr(genome, where)
    return _with_tree(genome, where, ConditionTree(mode=tree.mode, conditions=tuple(conditions)))
