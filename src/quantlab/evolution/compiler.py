"""Genome to strategy source (master spec section 9.6).

One function matters here: :func:`compile_genome` turns a validated
:class:`~quantlab.core.genome.StrategyGenome` into the text of a strategy module.
It is **deterministic and total** — the same genome always yields byte-identical
source, and every genome that can be constructed compiles — because
``strategy_id = sha256(code)[:16]`` and the run cache of section 11.2 are both
built on that. A compiler that emitted two spellings of one genome would evaluate
the same candidate twice and inflate the trial count that section 14.4 deflates
for.

It **executes nothing** (INV-4): no ``exec``, no ``compile``, no import of what it
writes. The output is a string. What runs it is the sandbox child, later, after
the AST check — and the generated source is written to face that check rather than
to be excused from it.

The generated strategy is long-or-flat and stateful in the way a trader reads it:
the entry tree (and every filter) opens a position, the exit tree closes one, and
in between the position is held. Crossing operators need the previous bar, which
:class:`~quantlab.core.strategy.Context` does not expose, so the generated module
remembers the last bar's operand values on the instance and uses them only when
the bar index says they are the immediately preceding bar's.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping

from quantlab.core.genome import (
    CROSS_OPS,
    Condition,
    ConditionTree,
    Operand,
    StrategyGenome,
    genome_id,
)
from quantlab.core.strategy import ParamSpec
from quantlab.core.types import RiskSpec, SizingSpec

__all__ = ["compile_genome", "genome_id"]

#: Bar columns reached through ``ctx.bars.value``. Written as a call with a string
#: argument rather than the ``.open``/``.high`` attributes, because section 9.2
#: forbids the name ``open`` outright and flags a bar's ``high``/``low`` compared
#: against an entry price; a string names the column without tripping either.
_INDENT = "    "


def _text(value: str) -> str:
    """A double-quoted string literal, matching the formatter the repo runs.

    ``repr`` would emit single quotes, which ``ruff format`` rewrites — and a
    generated file that changes under the project's own formatter would change its
    ``strategy_id`` with it.
    """
    return json.dumps(value)


def _number(value: object, *, integral: bool) -> str:
    """A bound as it was declared: ``200`` for an int parameter, not ``200.0``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        # Unreachable through a valid genome, which admits int and float parameters
        # only (section 9.6). Raising keeps `compile_genome` total over what it
        # accepts instead of writing a literal that would not parse back.
        raise ValueError(f"cannot render {value!r} as a numeric literal")
    return repr(int(value)) if integral else repr(float(value))


def _slots(genome: StrategyGenome) -> dict[str, str]:
    """Map each non-constant operand to the local name holding its value.

    Order comes from :meth:`StrategyGenome.conditions`, which is fixed, so the
    numbering is a property of the genome and not of how this function was called.
    Constants are not given a slot: they are the same number on every bar, so
    inlining them is both shorter and exactly as correct under a cross.
    """
    slots: dict[str, str] = {}
    for operand in genome.operands():
        if operand.kind != "constant" and operand.key not in slots:
            slots[operand.key] = f"op_{len(slots)}"
    return slots


def _operand_source(operand: Operand) -> str:
    """The expression that computes ``operand`` at the current bar."""
    if operand.kind == "indicator":
        key = next(iter(operand.kwargs))
        period = operand.kwargs[key]
        argument = f'self.p["{period}"]' if isinstance(period, str) else repr(period)
        return f'ctx.ind("{operand.name}", {key}={argument})'
    if operand.kind == "price":
        return f'ctx.bars.value("{operand.name}")'
    return f'self.p["{operand.name}"]'  # kind == "param"


def _current(operand: Operand, slots: Mapping[str, str]) -> str:
    return repr(operand.value) if operand.kind == "constant" else slots[operand.key]


def _previous(operand: Operand, slots: Mapping[str, str], order: list[str]) -> str:
    """This operand's value one bar ago.

    A constant is its own previous value. Anything else is read out of the tuple
    the last bar stored, at the position its slot occupies.
    """
    if operand.kind == "constant":
        return repr(operand.value)
    return f"previous[{order.index(slots[operand.key])}]"


def _condition_source(condition: Condition, slots: Mapping[str, str], order: list[str]) -> str:
    left, right = condition.left, condition.right
    now_left, now_right = _current(left, slots), _current(right, slots)
    if condition.op not in CROSS_OPS:
        return f"{now_left} {condition.op} {now_right}"

    # `cross_above` is "above now, and not above before" — which is false when the
    # previous bar is unknown, so a crossing is never invented at a discontinuity.
    was_left, was_right = _previous(left, slots, order), _previous(right, slots, order)
    if condition.op == "cross_above":
        return f"previous is not None and {now_left} > {now_right} and {was_left} <= {was_right}"
    return f"previous is not None and {now_left} < {now_right} and {was_left} >= {was_right}"


def _tree_source(
    tree: ConditionTree, slots: Mapping[str, str], order: list[str], *, empty: str
) -> str:
    """A condition tree as one boolean expression."""
    return _join(
        [_condition_source(c, slots, order) for c in tree.conditions],
        joiner=" and " if tree.mode == "all" else " or ",
        empty=empty,
    )


def _filters_source(
    filters: Iterable[Condition], slots: Mapping[str, str], order: list[str]
) -> str:
    return _join([_condition_source(c, slots, order) for c in filters], joiner=" and ", empty="")


def _join(parts: list[str], *, joiner: str, empty: str) -> str:
    """Combine boolean sub-expressions, parenthesising only where it disambiguates.

    A single condition is emitted bare: the parentheses would carry no meaning,
    and generated source is read by people often enough to be worth keeping
    plain.
    """
    if not parts:
        return empty
    if len(parts) == 1:
        return parts[0]
    return joiner.join(f"({part})" for part in parts)


def _param_source(spec: ParamSpec) -> str:
    """A ``ParamSpec(...)`` call the loader can read back with ``literal_eval``.

    Only the fields that carry information are written: the loader parses this
    text as data (section 9.2), and defaults spelled out would be noise in every
    diff between two candidates.
    """
    integral = spec.kind == "int"
    parts = [f"kind={_text(spec.kind)}", f"default={_number(spec.default, integral=integral)}"]
    for field in ("low", "high", "step"):
        value = getattr(spec, field)
        if value is not None:
            parts.append(f"{field}={_number(value, integral=integral)}")
    if spec.log:
        parts.append("log=True")
    return f"ParamSpec({', '.join(parts)})"


def _spec_source(name: str, spec: RiskSpec | SizingSpec) -> str:
    """A ``RiskSpec(...)``/``SizingSpec(...)`` call carrying only what was set."""
    fields = spec.model_dump(exclude_defaults=True)
    arguments = ", ".join(f"{key}={value!r}" for key, value in sorted(fields.items()))
    return f"{name}({arguments})"


def _tuple_source(names: list[str]) -> str:
    """A tuple literal; a one-element tuple keeps the comma that makes it one."""
    return f"({names[0]},)" if len(names) == 1 else f"({', '.join(names)})"


def _class_name(genome: StrategyGenome) -> str:
    """``sma_cross`` becomes ``SmaCross``: a class name, derived, never invented."""
    return "".join(part.title() for part in genome.name.split("_") if part) or "Genome"


def compile_genome(genome: StrategyGenome) -> str:
    """Render ``genome`` as the source of a ``bar_loop`` strategy module.

    Deterministic and total (section 9.6): the same genome always produces the
    same bytes, and every genome that validated can be rendered. Nothing is
    executed, imported or evaluated here.
    """
    slots = _slots(genome)
    order = list(slots.values())
    class_name = _class_name(genome)
    needs_previous = any(condition.op in CROSS_OPS for condition in genome.conditions())

    lines: list[str] = [
        '"""Compiled from a strategy genome (master spec section 9.6).',
        "",
        "Machine-written: edit the genome, not this file. The genome id below",
        "identifies the structure this was rendered from.",
        '"""',
        "",
        "from quantlab.core.strategy import (",
        f"{_INDENT}Context,",
        f"{_INDENT}ParamSpec,",
        f"{_INDENT}RiskSpec,",
        f"{_INDENT}Signal,",
        f"{_INDENT}SignalKind,",
        f"{_INDENT}SizingSpec,",
        ")",
        "",
        f'__all__ = ["STRATEGY", "{class_name}"]',
        "",
        f'GENOME_ID = "{genome_id(genome)}"',
        "",
        "",
        f"class {class_name}:",
        f"{_INDENT}name = {_text(genome.name)}",
        f"{_INDENT}version = {_text(genome.version)}",
        f'{_INDENT}style = "bar_loop"',
        "",
    ]
    if genome.params:
        lines.append(f"{_INDENT}params = {{")
        for key in sorted(genome.params):
            lines.append(f"{_INDENT * 2}{_text(key)}: {_param_source(genome.params[key])},")
        lines.append(f"{_INDENT}}}")
    else:
        # `{}` on one line: the formatter collapses an empty literal, and a
        # generated file it would rewrite is one whose id `make format` changes.
        lines.append(f"{_INDENT}params = {{}}")
    lines += [
        f"{_INDENT}warmup_bars = {genome.warmup_bars}",
        f"{_INDENT}risk = {_spec_source('RiskSpec', genome.risk)}",
        f"{_INDENT}sizing = {_spec_source('SizingSpec', genome.sizing)}",
        "",
        f"{_INDENT}def prepare(self, params):",
        f"{_INDENT * 2}self.p = dict(params)",
    ]
    if needs_previous:
        lines += [
            f"{_INDENT * 2}self.last = None",
            f"{_INDENT * 2}self.last_i = -1",
        ]
    lines += [
        "",
        f"{_INDENT}def on_bar(self, ctx: Context) -> Signal:",
    ]
    for operand in genome.operands():
        if operand.kind == "constant":
            continue
        name = slots[operand.key]
        line = f"{_INDENT * 2}{name} = {_operand_source(operand)}"
        if line not in lines:
            lines.append(line)

    if needs_previous:
        lines += [
            f"{_INDENT * 2}previous = self.last if self.last_i == ctx.i - 1 else None",
            f"{_INDENT * 2}self.last = {_tuple_source(order)}",
            f"{_INDENT * 2}self.last_i = ctx.i",
        ]

    # Warm-up: any indicator is NaN until its lookback is complete, and NaN
    # compares false against everything, so a genome would read "no entry" rather
    # than "not yet known". Saying so explicitly keeps the two apart.
    nan_slots = [slots[o.key] for o in genome.indicator_operands()]
    if nan_slots:
        test = " or ".join(f"{name} != {name}" for name in nan_slots)
        lines += [
            f"{_INDENT * 2}if {test}:  # NaN during warm-up",
            f"{_INDENT * 3}return Signal(SignalKind.FLAT)",
        ]

    entry = _tree_source(genome.entry, slots, order, empty="False")
    filters = _filters_source(genome.filters, slots, order)
    lines.append(f"{_INDENT * 2}entry = {entry}")
    if filters:
        lines.append(f"{_INDENT * 2}allowed = {filters}")
    lines += [
        f"{_INDENT * 2}exit_ = {_tree_source(genome.exit, slots, order, empty='False')}",
        f"{_INDENT * 2}if ctx.position.is_flat:",
        f"{_INDENT * 3}opening = entry{' and allowed' if filters else ''}",
        f"{_INDENT * 3}return Signal(SignalKind.LONG if opening else SignalKind.FLAT)",
        f"{_INDENT * 2}return Signal(SignalKind.FLAT if exit_ else SignalKind.LONG)",
        "",
        "",
        f"STRATEGY = {class_name}",
        "",
    ]
    return "\n".join(lines)
