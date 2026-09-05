"""The wire format between the parent process and the sandbox child (spec §21.3).

Two processes have to agree on something, and what they agree on decides what an
attacker gets to influence. The rule here is that **only data crosses**: Parquet
for tables, JSON for everything else, never pickle. A pickle stream is a program,
and a program written by the untrusted side is exactly what the sandbox exists to
prevent from running in this process.

The exchange happens inside one directory, which is also the child's working
directory: the child is given a path and nothing else, so a request cannot name a
file outside the sandbox for it to read or overwrite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final, Literal, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from quantlab.core.errors import SandboxProtocolError
from quantlab.core.types import (
    BacktestConfig,
    BacktestResult,
    BarFrame,
    Fill,
    Side,
    Trade,
)

__all__ = [
    "BARS_PARQUET",
    "ENGINE_PACKAGE",
    "EXIT_BAD_REQUEST",
    "EXIT_FORBIDDEN_IMPORT",
    "EXIT_INTERNAL",
    "EXIT_NETWORK",
    "EXIT_OK",
    "EXIT_RESOURCE_LIMIT",
    "EXIT_STATUS",
    "EXIT_STRATEGY_LOAD",
    "EXIT_STRATEGY_RUNTIME",
    "FILLS_PARQUET",
    "PROTOCOL_VERSION",
    "REQUEST_JSON",
    "RESPONSE_JSON",
    "SERIES_PARQUET",
    "STRATEGY_PY",
    "TRADES_PARQUET",
    "SandboxRequest",
    "SandboxResponse",
    "SandboxStatus",
    "read_request",
    "read_response",
    "read_result",
    "write_request",
    "write_response",
    "write_result",
]

PROTOCOL_VERSION: Final[int] = 1

#: The only package a sandbox request may name an engine from.
ENGINE_PACKAGE: Final[str] = "quantlab.adapters.engine"

#: Files inside the exchange directory. Fixed names, so nothing in a request can
#: point the child at a path of the requester's choosing.
REQUEST_JSON: Final[str] = "request.json"
BARS_PARQUET: Final[str] = "bars.parquet"
STRATEGY_PY: Final[str] = "strategy.py"
RESPONSE_JSON: Final[str] = "response.json"
SERIES_PARQUET: Final[str] = "series.parquet"
TRADES_PARQUET: Final[str] = "trades.parquet"
FILLS_PARQUET: Final[str] = "fills.parquet"

_PARQUET_KWARGS: Final[dict[str, Any]] = {
    "engine": "pyarrow",
    "compression": "zstd",
    "index": False,
}

SandboxStatus = Literal[
    "ok",
    "bad_request",
    "strategy_load",
    "strategy_runtime",
    "forbidden_import",
    "network",
    "internal",
    # Only the parent sets these two: a child that hit a hard limit was killed and
    # wrote nothing, so the diagnosis is made from how it died.
    "timeout",
    "resource_limit",
]

EXIT_OK: Final[int] = 0
EXIT_STRATEGY_LOAD: Final[int] = 10
EXIT_STRATEGY_RUNTIME: Final[int] = 11
EXIT_FORBIDDEN_IMPORT: Final[int] = 12
EXIT_NETWORK: Final[int] = 13
EXIT_BAD_REQUEST: Final[int] = 14
EXIT_INTERNAL: Final[int] = 15
EXIT_RESOURCE_LIMIT: Final[int] = 16

#: Exit code -> status, so the parent can classify a child that died before it
#: managed to write a response.
EXIT_STATUS: Final[dict[int, SandboxStatus]] = {
    EXIT_OK: "ok",
    EXIT_STRATEGY_LOAD: "strategy_load",
    EXIT_STRATEGY_RUNTIME: "strategy_runtime",
    EXIT_FORBIDDEN_IMPORT: "forbidden_import",
    EXIT_NETWORK: "network",
    EXIT_BAD_REQUEST: "bad_request",
    EXIT_INTERNAL: "internal",
    EXIT_RESOURCE_LIMIT: "resource_limit",
}


def _is_public_identifier(name: str) -> bool:
    return name.isidentifier() and not name.startswith("_")


class SandboxRequest(BaseModel):
    """What the parent asks the child to do.

    The engine is named rather than imported: the sandbox layer must not depend
    on an adapter (INV-8), and the caller that already holds an engine is the one
    that knows which. The child resolves the name *before* it installs the import
    guard, so trusted infrastructure loads and untrusted code does not.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: int = PROTOCOL_VERSION
    symbol: str
    timeframe: str
    dataset_id: str = ""
    strategy_name: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    config: BacktestConfig = BacktestConfig()
    engine_module: str
    engine_class: str
    allowed_imports: tuple[str, ...] = ()
    #: Re-run the AST check inside the child. Defence in depth: the loader checks
    #: too, and a source that reached here unchecked must still not run.
    ast_check: bool = True

    @field_validator("engine_module")
    @classmethod
    def _engine_must_be_an_engine(cls, value: str) -> str:
        """Only an engine adapter may be named.

        The engine is the one adapter the child loads. Without this the request
        would be a general "import this module in the sandbox process" instruction,
        and the store, the LLM client and the exchange adapters would all be one
        string away from a process that also runs untrusted code.

        The prefix is checked *with* its trailing dot, so a module named
        ``...engineering`` cannot pass as one under ``...engine``, and each
        remaining segment must be an ordinary public identifier — no ``..``, no
        empty segment, no private module.
        """
        prefix = f"{ENGINE_PACKAGE}."
        if not value.startswith(prefix):
            raise ValueError(f"engine_module must live under {ENGINE_PACKAGE}, got {value!r}")
        tail = value[len(prefix) :]
        if not tail or not all(_is_public_identifier(part) for part in tail.split(".")):
            raise ValueError(f"engine_module is not a plain module path: {value!r}")
        return value

    @field_validator("engine_class")
    @classmethod
    def _engine_class_must_be_a_public_name(cls, value: str) -> str:
        """A single public identifier, so the child fetches a class and not a
        module the engine happens to have imported, nor a private internal."""
        if not _is_public_identifier(value):
            raise ValueError(f"engine_class must be a public identifier, got {value!r}")
        return value


class SandboxResponse(BaseModel):
    """What the child reports back. Everything except the tables."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: int = PROTOCOL_VERSION
    status: SandboxStatus = "ok"
    engine_name: str = ""
    engine_version: str = ""
    warmup_bars: int = 0
    n_bars: int = 0
    bars_per_year: int = 8_760
    cost_summary: dict[str, float] = Field(default_factory=dict)
    log: tuple[str, ...] = ()
    ruined: bool = False
    error_type: str = ""
    error_message: str = ""
    error_codes: tuple[str, ...] = ()
    cpu_seconds: float = 0.0
    peak_rss_mb: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "ok"


# ---------------------------------------------------------------------------
# request
# ---------------------------------------------------------------------------
def write_request(directory: Path, request: SandboxRequest, bars: BarFrame, source: str) -> None:
    """Lay out one request in ``directory``, which becomes the child's cwd."""
    directory.mkdir(parents=True, exist_ok=True)
    bars.to_pandas().to_parquet(directory / BARS_PARQUET, **_PARQUET_KWARGS)
    (directory / STRATEGY_PY).write_text(source, encoding="utf-8")
    (directory / REQUEST_JSON).write_text(
        json.dumps(request.model_dump(mode="json"), sort_keys=True), encoding="utf-8"
    )


def read_request(directory: Path) -> tuple[SandboxRequest, BarFrame, str]:
    """Read back what :func:`write_request` wrote.

    Raises:
        SandboxProtocolError: on anything malformed, including a protocol version
            this build does not speak. Fail closed (spec §21.7): a request that
            cannot be understood is never guessed at.
    """
    try:
        payload = json.loads((directory / REQUEST_JSON).read_text(encoding="utf-8"))
        request = SandboxRequest.model_validate(payload)
    except Exception as exc:
        raise SandboxProtocolError(f"unreadable sandbox request: {exc}") from exc
    if request.protocol_version != PROTOCOL_VERSION:
        raise SandboxProtocolError(
            "sandbox protocol version mismatch",
            expected=PROTOCOL_VERSION,
            got=request.protocol_version,
        )
    try:
        frame = pd.read_parquet(directory / BARS_PARQUET)
        bars = BarFrame(
            frame,
            symbol=request.symbol,
            timeframe=request.timeframe,
            dataset_id=request.dataset_id,
        )
        source = (directory / STRATEGY_PY).read_text(encoding="utf-8")
    except Exception as exc:
        raise SandboxProtocolError(f"unreadable sandbox inputs: {exc}") from exc
    return request, bars, source


# ---------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------
def _trades_frame(trades: tuple[Trade, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_no": pd.Series([t.trade_no for t in trades], dtype="int64"),
            "side": pd.Series([t.side.value for t in trades], dtype="object"),
            "entry_ts": pd.Series([t.entry_ts for t in trades], dtype="int64"),
            "entry_px": pd.Series([t.entry_px for t in trades], dtype="float64"),
            "exit_ts": pd.Series([t.exit_ts for t in trades], dtype="int64"),
            "exit_px": pd.Series([t.exit_px for t in trades], dtype="float64"),
            "qty": pd.Series([t.qty for t in trades], dtype="float64"),
            "fees": pd.Series([t.fees for t in trades], dtype="float64"),
            "slippage_cost": pd.Series([t.slippage_cost for t in trades], dtype="float64"),
            "pnl": pd.Series([t.pnl for t in trades], dtype="float64"),
            "pnl_pct": pd.Series([t.pnl_pct for t in trades], dtype="float64"),
            "bars_held": pd.Series([t.bars_held for t in trades], dtype="int64"),
            "exit_reason": pd.Series([t.exit_reason for t in trades], dtype="object"),
        }
    )


def _fills_frame(fills: tuple[Fill, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bar_index": pd.Series([f.bar_index for f in fills], dtype="int64"),
            "ts": pd.Series([f.ts for f in fills], dtype="int64"),
            "side": pd.Series([f.side.value for f in fills], dtype="object"),
            "qty": pd.Series([f.qty for f in fills], dtype="float64"),
            "ref_price": pd.Series([f.ref_price for f in fills], dtype="float64"),
            "fill_price": pd.Series([f.fill_price for f in fills], dtype="float64"),
            "fee": pd.Series([f.fee for f in fills], dtype="float64"),
            "slippage_cost": pd.Series([f.slippage_cost for f in fills], dtype="float64"),
        }
    )


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Rows as plain dicts, so reconstruction is explicit about every cast."""
    return [{str(k): v for k, v in row.items()} for row in frame.to_dict("records")]


def _exit_reason(value: Any) -> Literal["signal", "end_of_data", "stop"]:
    text = str(value)
    if text not in ("signal", "end_of_data", "stop"):
        raise SandboxProtocolError("unknown trade exit_reason from the sandbox", reason=text)
    return cast(Literal["signal", "end_of_data", "stop"], text)


def write_response(directory: Path, response: SandboxResponse) -> None:
    (directory / RESPONSE_JSON).write_text(
        json.dumps(response.model_dump(mode="json"), sort_keys=True), encoding="utf-8"
    )


def write_result(directory: Path, result: BacktestResult) -> None:
    """Write the three tables a :class:`BacktestResult` carries."""
    series = pd.DataFrame(
        {
            "ts_open": pd.Series(np.asarray(result.equity.index, dtype="int64"), dtype="int64"),
            "equity": pd.Series(result.equity.to_numpy(dtype="float64"), dtype="float64"),
            "position_frac": pd.Series(
                result.position_frac.to_numpy(dtype="float64"), dtype="float64"
            ),
            "signal": pd.Series(list(result.signals), dtype="object"),
        }
    )
    series.to_parquet(directory / SERIES_PARQUET, **_PARQUET_KWARGS)
    _trades_frame(result.trades).to_parquet(directory / TRADES_PARQUET, **_PARQUET_KWARGS)
    _fills_frame(result.fills).to_parquet(directory / FILLS_PARQUET, **_PARQUET_KWARGS)


def read_result(directory: Path, response: SandboxResponse) -> BacktestResult:
    """Rebuild the :class:`BacktestResult` the child produced.

    Raises:
        SandboxProtocolError: if any table is missing or malformed.
    """
    try:
        series = pd.read_parquet(directory / SERIES_PARQUET)
        trades = pd.read_parquet(directory / TRADES_PARQUET)
        fills = pd.read_parquet(directory / FILLS_PARQUET)
    except Exception as exc:
        raise SandboxProtocolError(f"unreadable sandbox result: {exc}") from exc

    index = pd.Index(series["ts_open"].to_numpy(dtype="int64"), name="ts_open")
    return BacktestResult(
        equity=pd.Series(
            series["equity"].to_numpy(dtype="float64"), index=index, name="equity", dtype="float64"
        ),
        position_frac=pd.Series(
            series["position_frac"].to_numpy(dtype="float64"),
            index=index,
            name="position_frac",
            dtype="float64",
        ),
        signals=pd.Series(list(series["signal"]), index=index, name="signal", dtype="object"),
        fills=tuple(
            Fill(
                bar_index=int(row["bar_index"]),
                ts=int(row["ts"]),
                side=Side(row["side"]),
                qty=float(row["qty"]),
                ref_price=float(row["ref_price"]),
                fill_price=float(row["fill_price"]),
                fee=float(row["fee"]),
                slippage_cost=float(row["slippage_cost"]),
            )
            for row in _records(fills)
        ),
        trades=tuple(
            Trade(
                trade_no=int(row["trade_no"]),
                side=Side(row["side"]),
                entry_ts=int(row["entry_ts"]),
                entry_px=float(row["entry_px"]),
                exit_ts=int(row["exit_ts"]),
                exit_px=float(row["exit_px"]),
                qty=float(row["qty"]),
                fees=float(row["fees"]),
                slippage_cost=float(row["slippage_cost"]),
                pnl=float(row["pnl"]),
                pnl_pct=float(row["pnl_pct"]),
                bars_held=int(row["bars_held"]),
                exit_reason=_exit_reason(row["exit_reason"]),
            )
            for row in _records(trades)
        ),
        warmup_bars=response.warmup_bars,
        n_bars=response.n_bars,
        bars_per_year=response.bars_per_year,
        engine_name=response.engine_name,
        engine_version=response.engine_version,
        cost_summary=dict(response.cost_summary),
        log=tuple(response.log),
        ruined=response.ruined,
    )


def read_response(directory: Path) -> SandboxResponse:
    """Read the child's response, or raise if it did not write a usable one."""
    path = directory / RESPONSE_JSON
    if not path.exists():
        raise SandboxProtocolError("the sandbox child wrote no response")
    try:
        return SandboxResponse.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise SandboxProtocolError(f"unreadable sandbox response: {exc}") from exc
