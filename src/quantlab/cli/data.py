"""``quantlab data`` — ingest, validate and inspect market data (spec section 7.5).

quantlab data pull --symbol BTC/USDT --tf 1h --from 2017-08 [--to YYYY-MM]
quantlab data update
quantlab data validate
quantlab data info
quantlab data splits
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from quantlab.adapters.data.binance_archive import (
    BinanceArchiveIngestor,
    ParquetBarStore,
    httpx_fetcher,
    months_between,
)
from quantlab.adapters.data.ccxt_rest import CcxtRestSource
from quantlab.core.errors import ConfigError, DataError
from quantlab.core.splits import load_split_policy, walk_forward_windows
from quantlab.core.types import format_ts, from_ms, to_ms

__all__ = ["app"]

app = typer.Typer(help="Ingest, validate and inspect historical market data.", no_args_is_help=True)
console = Console()


SymbolOption = Annotated[str | None, typer.Option("--symbol", help="Market symbol, e.g. BTC/USDT.")]
TimeframeOption = Annotated[str | None, typer.Option("--tf", help="Timeframe, e.g. 1h.")]


def _market(ctx: typer.Context, symbol: str | None, timeframe: str | None) -> tuple[str, str]:
    """Resolve the symbol and timeframe, defaulting to the configured market."""
    config = ctx.obj.config
    return symbol or config.market.symbol, timeframe or config.market.timeframe


def _store(ctx: typer.Context) -> ParquetBarStore:
    config = ctx.obj.config
    return ParquetBarStore(config.project.data_dir, exchange=config.market.exchange)


def _current_month() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m")


@app.command("pull")
def pull(
    ctx: typer.Context,
    symbol: SymbolOption = None,
    timeframe: TimeframeOption = None,
    from_month: Annotated[
        str, typer.Option("--from", help="First month to fetch, as YYYY-MM.")
    ] = "2017-08",
    to_month: Annotated[
        str | None, typer.Option("--to", help="Last month to fetch (default: last full month).")
    ] = None,
    allow_gaps: Annotated[
        bool, typer.Option("--allow-gaps", help="Accept gaps longer than 3 bars.")
    ] = False,
    drop_off_grid: Annotated[
        bool,
        typer.Option(
            "--drop-off-grid",
            help="Discard bars whose open time is off the timeframe grid, leaving a gap.",
        ),
    ] = False,
    no_verify: Annotated[
        bool, typer.Option("--no-verify", help="Skip archive checksum verification.")
    ] = False,
) -> None:
    """Download Binance monthly archives and rebuild the processed dataset."""
    market, tf = _market(ctx, symbol, timeframe)
    store = _store(ctx)
    ingestor = BinanceArchiveIngestor(
        store, fetch=httpx_fetcher(), exchange=ctx.obj.config.market.exchange
    )

    last = to_month or _current_month()
    months = months_between(from_month, last)
    console.print(f"fetching {len(months)} month(s) of {market} {tf} from the Binance archive")

    result = ingestor.pull(
        market,
        tf,
        months=months,
        allow_gaps=allow_gaps,
        off_grid="drop" if drop_off_grid else "error",
        verify=not no_verify,
    )
    console.print(result.summary())
    console.print(result.report.summary())


@app.command("update")
def update(
    ctx: typer.Context,
    symbol: SymbolOption = None,
    timeframe: TimeframeOption = None,
    allow_gaps: Annotated[
        bool, typer.Option("--allow-gaps", help="Accept gaps longer than 3 bars.")
    ] = False,
    drop_off_grid: Annotated[
        bool,
        typer.Option(
            "--drop-off-grid",
            help="Discard bars whose open time is off the timeframe grid, leaving a gap.",
        ),
    ] = False,
) -> None:
    """Extend the stored dataset to the latest closed bar using the ccxt REST API."""
    market, tf = _market(ctx, symbol, timeframe)
    store = _store(ctx)
    manifest = store.require_manifest(market, tf)

    import ccxt

    source = CcxtRestSource(ccxt.binance({"enableRateLimit": True}))
    now_ms = int(dt.datetime.now(dt.UTC).timestamp() * 1000)
    tail = source.fetch_tail(market, tf, after_ts=manifest.end_ts, now_ms=now_ms)
    console.print(f"fetched {len(tail)} new closed bar(s) after {format_ts(manifest.end_ts)}")

    ingestor = BinanceArchiveIngestor(store, exchange=ctx.obj.config.market.exchange)
    result = ingestor.rebuild(
        market,
        tf,
        extra_frames=[tail],
        allow_gaps=allow_gaps,
        off_grid="drop" if drop_off_grid else "error",
    )
    console.print(result.summary())


@app.command("validate")
def validate(
    ctx: typer.Context,
    symbol: SymbolOption = None,
    timeframe: TimeframeOption = None,
) -> None:
    """Re-validate the stored dataset: hashes, ordering, duplicates and gaps."""
    market, tf = _market(ctx, symbol, timeframe)
    store = _store(ctx)
    report = store.validate(market, tf)
    console.print(report.summary())
    if report.gaps_reported:
        console.print(
            f"[yellow]{len(report.gaps_reported)} gap(s) longer than the auto-fill limit[/]"
        )
    else:
        console.print("[green]dataset is consistent[/]")


@app.command("info")
def info(
    ctx: typer.Context,
    symbol: SymbolOption = None,
    timeframe: TimeframeOption = None,
) -> None:
    """Print the dataset manifest and its dataset_id."""
    market, tf = _market(ctx, symbol, timeframe)
    store = _store(ctx)
    manifest = store.require_manifest(market, tf)

    console.print(f"exchange   : {manifest.exchange}")
    console.print(f"symbol     : {manifest.symbol}   timeframe: {manifest.timeframe}")
    console.print(f"dataset_id : [bold]{manifest.dataset_id}[/]")
    console.print(f"bars       : {manifest.n_bars}")
    console.print(f"range      : {format_ts(manifest.start_ts)} .. {format_ts(manifest.end_ts)}")
    console.print(f"built_at   : {manifest.built_at}")

    table = Table(title="files", header_style="bold")
    table.add_column("path")
    table.add_column("bars", justify="right")
    table.add_column("first")
    table.add_column("last")
    table.add_column("sha256")
    for entry in sorted(manifest.files, key=lambda e: e.path):
        table.add_row(
            entry.path,
            str(entry.n_bars),
            format_ts(entry.first_ts),
            format_ts(entry.last_ts),
            entry.sha256[:16] + "...",
        )
    console.print(table)


@app.command("splits")
def splits(
    ctx: typer.Context,
    policy_file: Annotated[
        Path | None, typer.Option("--policy", help="Split policy YAML (default: from config).")
    ] = None,
    show_windows: Annotated[
        bool, typer.Option("--windows/--no-windows", help="List the walk-forward windows.")
    ] = False,
) -> None:
    """Print the split policy, its bar counts and its walk-forward windows."""
    config = ctx.obj.config
    store = _store(ctx)
    path = policy_file or config.splits.policy_file

    manifest = store.read_manifest(config.market.symbol, config.market.timeframe)
    policy = load_split_policy(path, dataset_id=manifest.dataset_id if manifest else "")
    console.print(policy.describe())

    windows = walk_forward_windows(
        policy,
        is_bars=config.walkforward.is_bars,
        oos_bars=config.walkforward.oos_bars,
        step_bars=config.walkforward.step_bars,
        scheme=config.walkforward.scheme,
    )
    console.print(f"walk-forward: {len(windows)} {config.walkforward.scheme} window(s)")
    if show_windows:
        for window in windows:
            console.print("  " + window.describe())


@app.command("range")
def show_range(
    ctx: typer.Context,
    symbol: SymbolOption = None,
    timeframe: TimeframeOption = None,
    start: Annotated[str | None, typer.Option("--from", help="ISO 8601 or YYYY-MM.")] = None,
    end: Annotated[str | None, typer.Option("--to", help="ISO 8601 or YYYY-MM.")] = None,
    guarded: Annotated[
        bool,
        typer.Option(
            "--guarded/--unguarded",
            help="Apply the research partition guard, which refuses test-partition bars.",
        ),
    ] = True,
) -> None:
    """Load a range and print what the backtester would receive."""
    config = ctx.obj.config
    market, tf = _market(ctx, symbol, timeframe)
    store = _store(ctx)

    available = store.available_range(market, tf)
    if available is None:
        raise DataError("no dataset found; run `quantlab data pull` first", symbol=market)
    start_ts = to_ms(start) if start else available[0]
    end_ts = to_ms(end) if end else available[1]

    source: object = store
    if guarded:
        from quantlab.adapters.data.guard import PartitionGuard

        manifest = store.require_manifest(market, tf)
        policy = load_split_policy(config.splits.policy_file, dataset_id=manifest.dataset_id)
        if policy.symbol != market or policy.timeframe != tf:
            raise ConfigError(
                "the split policy does not describe this market",
                policy=f"{policy.symbol} {policy.timeframe}",
                requested=f"{market} {tf}",
            )
        source = PartitionGuard(store, policy)

    bars = source.load(market, tf, start_ts, end_ts)  # type: ignore[attr-defined]
    console.print(repr(bars))
    if bars.n_bars:
        first = bars.bar(0)
        console.print(
            f"first bar  : {format_ts(first.ts_open)} "
            f"O={first.open} H={first.high} L={first.low} C={first.close}"
        )
        console.print(f"local time : {from_ms(first.ts_open).isoformat()} (always UTC)")
        console.print(f"gap-filled : {int(bars.is_gap_filled.sum())} bar(s)")
