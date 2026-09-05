"""The AST checker (spec section 9.2, task T17).

The fixtures are the point of this file. Each rejected fixture is a plausible
strategy that breaks exactly one rule, so the expected code set below doubles as
the checker's documented vocabulary: if a rule stops firing, or starts firing on
something it should not, one named case fails rather than a vague count.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quantlab.core.errors import StrategySafetyError
from quantlab.sandbox.ast_check import (
    DEFAULT_ALLOWED_IMPORTS,
    DEFAULT_MAX_LOGIC_LINES,
    DEFAULT_MAX_PARAMS,
    FORBIDDEN_MODULES,
    FORBIDDEN_NAMES,
    MAX_SOURCE_LINES,
    AstReport,
    check_source,
    count_logic_lines,
    require_safe_source,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "strategies"
REJECTED_DIR = FIXTURES / "rejected"
VALID_DIR = FIXTURES / "valid"
REPO_STRATEGIES = Path(__file__).resolve().parents[2] / "strategies"

#: fixture file -> the exact set of violation codes it must produce.
#: Exact, not "contains": a rule that fires twice, or drags a second rule along
#: with it, tells the author to fix things that are not wrong.
REJECTED: dict[str, frozenset[str]] = {
    "import_os.py": frozenset({"E_FORBIDDEN_IMPORT"}),
    "import_from_subprocess.py": frozenset({"E_FORBIDDEN_IMPORT"}),
    "import_not_allowed.py": frozenset({"E_IMPORT_NOT_ALLOWED"}),
    "star_import.py": frozenset({"E_STAR_IMPORT"}),
    "relative_import.py": frozenset({"E_RELATIVE_IMPORT"}),
    "builtin_open.py": frozenset({"E_FORBIDDEN_NAME"}),
    "builtin_eval.py": frozenset({"E_FORBIDDEN_NAME"}),
    "dunder_import.py": frozenset({"E_FORBIDDEN_NAME"}),
    "getattr_probe.py": frozenset({"E_FORBIDDEN_NAME"}),
    "bare_module_name.py": frozenset({"E_FORBIDDEN_NAME"}),
    "dunder_attribute.py": frozenset({"E_DUNDER_ACCESS"}),
    "two_classes.py": frozenset({"E_MULTIPLE_CLASSES"}),
    "no_class.py": frozenset({"E_NO_CLASS", "E_NO_STRATEGY_EXPORT"}),
    "module_function.py": frozenset({"E_MODULE_CODE"}),
    "module_subscript_assign.py": frozenset({"E_MODULE_CODE"}),
    "mutable_module_state.py": frozenset({"E_MUTABLE_MODULE_STATE"}),
    "lowercase_module_state.py": frozenset({"E_MODULE_STATE"}),
    "no_params.py": frozenset({"E_NO_PARAMS"}),
    "params_not_literal.py": frozenset({"E_PARAMS_NOT_LITERAL"}),
    "param_not_spec.py": frozenset({"E_PARAM_NOT_SPEC"}),
    "unbounded_param.py": frozenset({"E_UNBOUNDED_PARAM"}),
    "unbounded_categorical.py": frozenset({"E_UNBOUNDED_PARAM"}),
    "too_many_params.py": frozenset({"E_TOO_MANY_PARAMS"}),
    "no_warmup.py": frozenset({"E_NO_WARMUP"}),
    "warmup_not_int.py": frozenset({"E_WARMUP_NOT_INT"}),
    "warmup_negative.py": frozenset({"E_WARMUP_NEGATIVE"}),
    "no_strategy_export.py": frozenset({"E_NO_STRATEGY_EXPORT"}),
    "strategy_mismatch.py": frozenset({"E_STRATEGY_MISMATCH"}),
    "hand_rolled_stop.py": frozenset({"E_INTRABAR_STOP"}),
}

VALID = (
    "sma_cross_valid.py",
    "vectorized_valid.py",
    "breakout_valid.py",
    "categorical_params.py",
    "compiled_genome.py",
)

#: Every code the checker can emit. A new rule must be added here *and* be
#: exercised by a fixture, so the vocabulary cannot drift away from the tests.
ALL_CODES: frozenset[str] = frozenset().union(*REJECTED.values()) | {
    "E_FILE_TOO_LONG",
    "E_SYNTAX",
}


def _read(directory: Path, name: str) -> str:
    return (directory / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The acceptance criterion: 29 rejected, 5 accepted.
# ---------------------------------------------------------------------------


def test_rejected_fixture_inventory_is_complete() -> None:
    """Every file on disk is claimed by the table, and vice versa."""
    on_disk = {path.name for path in REJECTED_DIR.glob("*.py")}
    assert on_disk == set(REJECTED)
    assert len(REJECTED) >= 20


def test_valid_fixture_inventory_is_complete() -> None:
    on_disk = {path.name for path in VALID_DIR.glob("*.py")}
    assert on_disk == set(VALID)
    assert len(VALID) == 5


@pytest.mark.parametrize("name", sorted(REJECTED))
def test_rejected_fixture_produces_exactly_its_codes(name: str) -> None:
    report = check_source(_read(REJECTED_DIR, name))
    assert not report.ok, f"{name} was accepted"
    assert set(report.codes) == REJECTED[name], report.describe()


@pytest.mark.parametrize("name", sorted(REJECTED))
def test_every_rejection_names_a_line(name: str) -> None:
    """A violation the author cannot locate is a violation they cannot fix."""
    report = check_source(_read(REJECTED_DIR, name))
    located = [v for v in report.violations if v.code not in ("E_NO_CLASS", "E_NO_STRATEGY_EXPORT")]
    assert all(v.line > 0 for v in located), report.describe()
    assert all(v.message for v in report.violations)


@pytest.mark.parametrize("name", sorted(VALID))
def test_valid_fixture_is_accepted(name: str) -> None:
    report = check_source(_read(VALID_DIR, name))
    assert report.ok, report.describe()
    assert report.codes == ()
    assert report.warnings == ()
    assert report.class_name
    assert 0 < report.logic_lines <= DEFAULT_MAX_LOGIC_LINES
    assert report.n_lines > 0


@pytest.mark.parametrize(
    "name",
    [
        "TEMPLATE.py",
        *(
            f"baselines/{p}"
            for p in ("buy_and_hold.py", "sma_cross.py", "rsi_reversion.py", "random_entry.py")
        ),
    ],
)
def test_shipped_strategies_pass_their_own_checker(name: str) -> None:
    """The template and the four baselines are real strategies, not just fixtures."""
    report = check_source((REPO_STRATEGIES / name).read_text(encoding="utf-8"))
    assert report.ok, report.describe()


def test_the_codes_the_checker_emits_are_all_exercised() -> None:
    """No rule may exist without a fixture that fires it."""
    fired = set(REJECTED_DIR.glob("*.py"))
    emitted: set[str] = set()
    for path in fired:
        emitted |= set(check_source(path.read_text(encoding="utf-8")).codes)
    emitted |= {"E_FILE_TOO_LONG", "E_SYNTAX"}  # covered by generated sources below
    assert emitted == ALL_CODES


# ---------------------------------------------------------------------------
# Rules that need generated rather than stored source.
# ---------------------------------------------------------------------------


def test_file_too_long_is_rejected_without_parsing() -> None:
    source = "\n".join(["# padding"] * (MAX_SOURCE_LINES + 1))
    report = check_source(source)
    assert report.codes == ("E_FILE_TOO_LONG",)
    assert report.n_lines == MAX_SOURCE_LINES + 1


def test_file_at_the_limit_is_not_rejected_for_length() -> None:
    source = "\n".join(["# padding"] * MAX_SOURCE_LINES)
    assert "E_FILE_TOO_LONG" not in check_source(source).codes


def test_unparseable_source_reports_a_syntax_error() -> None:
    report = check_source("class Broken:\n    params = {\n")
    assert report.codes == ("E_SYNTAX",)
    assert report.violations[0].line > 0


def test_syntax_error_never_escapes_as_an_exception() -> None:
    """A malformed file is a report, not a crash: the caller shows it to the author."""
    assert not check_source("def (").ok


# ---------------------------------------------------------------------------
# Import rules.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module", sorted(FORBIDDEN_MODULES))
def test_every_forbidden_module_is_rejected(module: str) -> None:
    assert "E_FORBIDDEN_IMPORT" in check_source(f"import {module}\n").codes
    assert "E_FORBIDDEN_IMPORT" in check_source(f"from {module} import thing\n").codes


@pytest.mark.parametrize("module", sorted(FORBIDDEN_MODULES))
def test_a_forbidden_module_stays_forbidden_as_a_submodule(module: str) -> None:
    assert "E_FORBIDDEN_IMPORT" in check_source(f"import {module}.sub\n").codes


@pytest.mark.parametrize("name", sorted(FORBIDDEN_NAMES))
def test_every_forbidden_name_is_rejected(name: str) -> None:
    assert "E_FORBIDDEN_NAME" in check_source(f"X = {name}\n").codes


def test_future_import_is_always_allowed() -> None:
    assert "E_IMPORT_NOT_ALLOWED" not in check_source("from __future__ import annotations\n").codes


def test_allow_list_matches_full_paths_not_prefixes() -> None:
    """`quantlab.core.strategy` is allowed; `quantlab.adapters` is not."""
    codes = check_source("import quantlab.adapters.store.sqlite\n").codes
    assert "E_IMPORT_NOT_ALLOWED" in codes


def test_submodule_of_an_allowed_module_is_allowed() -> None:
    assert "E_IMPORT_NOT_ALLOWED" not in check_source("import numpy.linalg\n").codes


def test_allow_list_is_configurable() -> None:
    source = "import scipy\n"
    assert "E_IMPORT_NOT_ALLOWED" in check_source(source).codes
    widened = (*DEFAULT_ALLOWED_IMPORTS, "scipy")
    assert "E_IMPORT_NOT_ALLOWED" not in check_source(source, allowed_imports=widened).codes


def test_a_forbidden_module_cannot_be_allow_listed_back_in() -> None:
    """The allow-list widens what is permitted; it never re-opens what is banned."""
    codes = check_source("import os\n", allowed_imports=("os",)).codes
    assert "E_FORBIDDEN_IMPORT" in codes


# ---------------------------------------------------------------------------
# The intrabar-stop rule, and its false-positive boundary.
# ---------------------------------------------------------------------------

_STOP_BODY = """
from quantlab.core.strategy import ParamSpec


class S:
    params = {{"n": ParamSpec(kind="int", default=2, low=1, high=3)}}
    warmup_bars = 1

    def on_bar(self, ctx):
        return {expr}


STRATEGY = S
"""


@pytest.mark.parametrize(
    "expr",
    [
        "ctx.bars.low[-1] < ctx.position.entry_price",
        "ctx.position.entry_price * 0.98 > ctx.bars.low[-1]",
        "ctx.bars.high[-1] >= ctx.position.entry_price * 1.05",
        "ctx.bars.low[ctx.i] <= self.entry_price",
    ],
)
def test_hand_rolled_intrabar_stops_are_rejected(expr: str) -> None:
    assert "E_INTRABAR_STOP" in check_source(_STOP_BODY.format(expr=expr)).codes


@pytest.mark.parametrize(
    "expr",
    [
        "ctx.bars.high[-1] > ctx.ind('highest', n=20)",
        "ctx.bars.low[-1] < ctx.bars.close[-2]",
        "ctx.bars.close[-1] > ctx.position.entry_price",
        "ctx.bars.high[-1] - ctx.bars.low[-1] > 0",
    ],
)
def test_legitimate_reads_of_high_and_low_are_accepted(expr: str) -> None:
    """Only high/low *against an entry price* is a stop. Everything else is data."""
    assert "E_INTRABAR_STOP" not in check_source(_STOP_BODY.format(expr=expr)).codes


# ---------------------------------------------------------------------------
# Parameter and warm-up rules.
# ---------------------------------------------------------------------------


def test_param_ceiling_matches_the_specified_slack() -> None:
    """Spec section 9.2: the hard limit is `validation.max_free_params + 4`."""
    assert DEFAULT_MAX_PARAMS == 10


def test_param_ceiling_is_configurable() -> None:
    source = _read(REJECTED_DIR, "too_many_params.py")
    assert "E_TOO_MANY_PARAMS" in check_source(source).codes
    assert "E_TOO_MANY_PARAMS" not in check_source(source, max_params=11).codes


def test_a_param_count_at_the_limit_is_accepted() -> None:
    entries = ", ".join(
        f'"p{i}": ParamSpec(kind="int", default=1, low=0, high=2)'
        for i in range(DEFAULT_MAX_PARAMS)
    )
    source = _STOP_BODY.format(expr="True").replace(
        '{"n": ParamSpec(kind="int", default=2, low=1, high=3)}', "{" + entries + "}"
    )
    report = check_source(source)
    assert "E_TOO_MANY_PARAMS" not in report.codes
    assert report.n_params == DEFAULT_MAX_PARAMS


def test_bool_params_need_no_bounds() -> None:
    source = _STOP_BODY.format(expr="True").replace(
        'ParamSpec(kind="int", default=2, low=1, high=3)', 'ParamSpec(kind="bool", default=True)'
    )
    assert "E_UNBOUNDED_PARAM" not in check_source(source).codes


def test_a_bool_warmup_is_not_an_int() -> None:
    source = _STOP_BODY.format(expr="True").replace("warmup_bars = 1", "warmup_bars = True")
    assert "E_WARMUP_NOT_INT" in check_source(source).codes


def test_zero_warmup_is_allowed() -> None:
    """buy_and_hold legitimately needs no warm-up."""
    source = _STOP_BODY.format(expr="True").replace("warmup_bars = 1", "warmup_bars = 0")
    codes = check_source(source).codes
    assert "E_WARMUP_NEGATIVE" not in codes
    assert "E_WARMUP_NOT_INT" not in codes


def test_annotated_class_attributes_are_recognised() -> None:
    """`warmup_bars: int = 100` declares the attribute just as plainly."""
    source = _STOP_BODY.format(expr="True").replace("warmup_bars = 1", "warmup_bars: int = 100")
    codes = check_source(source).codes
    assert "E_NO_WARMUP" not in codes


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------


def test_all_violations_are_reported_at_once() -> None:
    """One rejection per attempt would make fixing a strategy a guessing game."""
    source = "import os\nimport socket\nCACHE = {}\n"
    report = check_source(source)
    assert {"E_FORBIDDEN_IMPORT", "E_MUTABLE_MODULE_STATE", "E_NO_CLASS"} <= set(report.codes)


def test_violations_are_ordered_by_line() -> None:
    report = check_source("import socket\n\n\nimport os\nCACHE = {}\n")
    lines = [v.line for v in report.violations if v.line]
    assert lines == sorted(lines)


def test_describe_mentions_every_violation_and_warning() -> None:
    report = check_source(_read(REJECTED_DIR, "import_os.py"))
    text = report.describe()
    assert "E_FORBIDDEN_IMPORT" in text
    assert "line" in text


def test_violation_str_includes_the_line_when_known() -> None:
    report = check_source("import os\n")
    rendered = {v.code: str(v) for v in report.violations}
    assert rendered["E_FORBIDDEN_IMPORT"].endswith("(line 1)")
    # A whole-file finding has no line to point at, and must not invent one.
    assert rendered["E_NO_CLASS"] == "E_NO_CLASS: no strategy class found"


def test_a_long_but_legal_strategy_warns_rather_than_rejects() -> None:
    filler = "\n".join(f"        x{i} = {i}" for i in range(DEFAULT_MAX_LOGIC_LINES + 10))
    source = _STOP_BODY.format(expr="True").replace(
        "        return True", filler + "\n        return True"
    )
    report = check_source(source)
    assert report.ok, report.describe()
    assert report.logic_lines > DEFAULT_MAX_LOGIC_LINES
    assert [w.code for w in report.warnings] == ["W_LOGIC_LINES"]


def test_logic_lines_exclude_blanks_comments_and_imports() -> None:
    source = "import numpy\n\n# a comment\nX = 1\n"
    assert count_logic_lines(source) == 1


def test_report_carries_the_class_name_and_shape() -> None:
    report = check_source(_read(VALID_DIR, "compiled_genome.py"))
    assert report.class_name == "CompiledGenome"
    assert report.n_params == 3
    assert report.n_lines == len(_read(VALID_DIR, "compiled_genome.py").splitlines())


def test_empty_report_is_ok() -> None:
    assert AstReport().ok


# ---------------------------------------------------------------------------
# require_safe_source.
# ---------------------------------------------------------------------------


def test_require_safe_source_returns_the_report_for_safe_source() -> None:
    report = require_safe_source(_read(VALID_DIR, "sma_cross_valid.py"))
    assert report.ok


def test_require_safe_source_raises_with_every_code() -> None:
    with pytest.raises(StrategySafetyError) as excinfo:
        require_safe_source(_read(REJECTED_DIR, "no_class.py"))
    assert set(excinfo.value.codes) == REJECTED["no_class.py"]
    assert excinfo.value.violations


def test_require_safe_source_forwards_its_options() -> None:
    source = _read(REJECTED_DIR, "import_not_allowed.py")
    with pytest.raises(StrategySafetyError):
        require_safe_source(source)
    widened = (*DEFAULT_ALLOWED_IMPORTS, "scipy")
    assert require_safe_source(source, allowed_imports=widened).ok


def test_a_warmup_referring_to_a_constant_is_not_a_literal() -> None:
    """`warmup_bars = LOOKBACK` cannot be checked without running the module."""
    source = _STOP_BODY.format(expr="True").replace("warmup_bars = 1", "warmup_bars = LOOKBACK")
    assert "E_WARMUP_NOT_INT" in check_source(source).codes
