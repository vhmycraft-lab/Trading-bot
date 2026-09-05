"""A run as Markdown (master spec section 11, task T23).

The report is built from the *stored* record and artifacts, never from a live
object, so it says what was recorded rather than what happens to be in memory.
That is the point: a report you can regenerate a year later from the database
alone is evidence; one that needed the original process is a screenshot.

Two things are stated rather than implied, because both change how a number
should be read: a run made on a dirty working tree is flagged (section 11.3), and
so is every metric the engine could not define — a profit factor with no losing
trades is `None`, and printing it as `0` would be a lie about a good run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["RunReport", "format_metric", "render_run_report"]

#: Metrics shown first, in this order. The rest follow alphabetically; a report
#: that buried drawdown under an alphabetical list would be read wrongly.
HEADLINE_METRICS: tuple[str, ...] = (
    "net_return",
    "cagr",
    "max_drawdown",
    "sharpe",
    "sortino",
    "calmar",
    "profit_factor",
    "expectancy",
    "win_rate",
    "n_trades",
)


def format_metric(value: float | bool | None) -> str:
    """Render one metric, keeping "undefined" distinct from zero.

    A profit factor with no losing trades is undefined, not zero; a Sharpe over a
    flat equity curve is undefined, not bad. Collapsing either into a number would
    make a report that reads plausibly and says something false.
    """
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if float(value).is_integer() and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.6g}"


@dataclass(frozen=True, slots=True)
class RunReport:
    """Everything the Markdown is rendered from, gathered in one place."""

    run_id: str
    strategy_id: str
    segment: str
    status: str
    engine: str
    dataset_id: str
    split_id: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float | bool | None] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    n_trades: int = 0
    artifact_dir: str = ""
    tearsheet: str = ""
    #: Extra caveats from the caller. The two the specification names —
    #: a dirty tree and a non-``ok`` status — are derived by the renderer, so a
    #: caller cannot forget them.
    warnings: tuple[str, ...] = ()

    def caveats(self) -> tuple[str, ...]:
        """Everything that changes how these numbers should be read.

        Derived here rather than supplied, because section 11.3's "flagged in
        reports" is a property of the report. A flag that depended on the caller
        remembering would eventually be missed on the one report that needed it.
        """
        found: list[str] = []
        if self.environment.get("git_dirty"):
            found.append(
                "produced on a working tree with uncommitted changes: the recorded "
                "commit does not fully describe the code that ran (spec 11.3)"
            )
        if self.status != "ok":
            found.append(f"this run is recorded as `{self.status}`")
        found.extend(self.warnings)
        return tuple(found)


def _table(rows: list[tuple[str, str]], headers: tuple[str, str]) -> list[str]:
    lines = [f"| {headers[0]} | {headers[1]} |", "|---|---|"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    return lines


def render_run_report(report: RunReport) -> str:
    """Render ``report`` as Markdown.

    Deterministic: the same record produces the same bytes, so a report can be
    diffed against an earlier one and the difference means something.
    """
    lines: list[str] = [f"# Run `{report.run_id}`", ""]

    caveats = report.caveats()
    if caveats:
        lines.append("> **Read with care**")
        lines.extend(f"> - {caveat}" for caveat in caveats)
        lines.append("")

    lines.extend(
        _table(
            [
                ("strategy", f"`{report.strategy_id}`"),
                ("segment", report.segment),
                ("status", report.status),
                ("engine", report.engine),
                ("dataset", f"`{report.dataset_id}`"),
                ("split", f"`{report.split_id}`"),
                ("trades", str(report.n_trades)),
                ("artifacts", f"`{report.artifact_dir}`" if report.artifact_dir else "—"),
            ],
            ("field", "value"),
        )
    )
    lines.append("")

    ordered = [name for name in HEADLINE_METRICS if name in report.metrics]
    ordered += sorted(name for name in report.metrics if name not in HEADLINE_METRICS)
    if ordered:
        lines.extend(["## Metrics", ""])
        lines.extend(
            _table(
                [(name, format_metric(report.metrics[name])) for name in ordered],
                ("metric", "value"),
            )
        )
        lines.append("")

    if report.params:
        lines.extend(["## Parameters", ""])
        lines.extend(
            _table(
                [
                    (
                        name,
                        format_metric(value)
                        if isinstance(value, (int, float, bool))
                        else str(value),
                    )
                    for name, value in sorted(report.params.items())
                ],
                ("parameter", "value"),
            )
        )
        lines.append("")

    if report.environment:
        lines.extend(["## Environment", ""])
        rows = [
            (name, str(report.environment[name]))
            for name in sorted(report.environment)
            if name != "packages"
        ]
        packages = report.environment.get("packages") or {}
        rows.extend(
            (f"packages.{name}", str(version)) for name, version in sorted(packages.items())
        )
        lines.extend(_table(rows, ("key", "value")))
        lines.append("")

    if report.tearsheet:
        lines.extend(["## Equity", "", f"![equity]({report.tearsheet})", ""])

    return "\n".join(lines).rstrip() + "\n"
