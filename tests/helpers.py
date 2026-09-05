"""Deterministic builders for market-data tests.

Everything here is seeded and offline.  No test in this suite touches the
network, the wall clock, or the developer's real ``data/`` directory.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd

from quantlab.adapters.data.binance_archive import kline_columns
from quantlab.core.types import coerce_bar_frame_df, empty_bar_frame_df, timeframe_ms, to_ms

if TYPE_CHECKING:
    from quantlab.core.types import BarFrame

#: A stable starting point used by most fixtures: 2023-01-01T00:00:00Z.
EPOCH_2023 = to_ms("2023-01-01T00:00:00Z")


def make_bars(
    n: int = 24,
    *,
    start_ts: int = EPOCH_2023,
    timeframe: str = "1h",
    start_price: float = 100.0,
    seed: int = 7,
    drift: float = 0.0,
) -> pd.DataFrame:
    """Build ``n`` canonical bars with a seeded random walk.

    The OHLC relationships are constructed to be valid by definition, so a test
    that breaks one is breaking it on purpose.

    The series is **prefix-stable**: ``make_bars(60)`` starts with exactly the
    bars of ``make_bars(30)``.  Each column draws from its own seeded stream,
    because a shared stream would shift when the length changes and would make
    the truncation tests in ``tests/unit/test_lookahead.py`` pass or fail for
    reasons that have nothing to do with look-ahead.
    """
    bar_ms = timeframe_ms(timeframe)
    if n <= 0:
        return empty_bar_frame_df()

    def stream(offset: int) -> np.random.Generator:
        return np.random.default_rng(seed * 1000 + offset)

    steps = stream(0).normal(loc=drift, scale=0.5, size=n)
    closes = start_price + np.cumsum(steps)
    closes = np.maximum(closes, 1.0)
    opens = np.concatenate([[start_price], closes[:-1]])
    highs = np.maximum(opens, closes) + np.abs(stream(1).normal(0.0, 0.2, size=n))
    lows = np.minimum(opens, closes) - np.abs(stream(2).normal(0.0, 0.2, size=n))
    lows = np.minimum(lows, np.minimum(opens, closes))
    lows = np.maximum(lows, 0.01)

    frame = pd.DataFrame(
        {
            "ts_open": np.arange(start_ts, start_ts + n * bar_ms, bar_ms, dtype="int64"),
            "open": opens.astype("float64"),
            "high": highs.astype("float64"),
            "low": lows.astype("float64"),
            "close": closes.astype("float64"),
            "volume": np.abs(stream(3).normal(10.0, 2.0, size=n)).astype("float64"),
            "quote_volume": np.abs(stream(4).normal(1000.0, 50.0, size=n)).astype("float64"),
            "trades": stream(5).integers(1, 100, size=n).astype("int64"),
            "is_gap_filled": np.zeros(n, dtype=bool),
        }
    )
    return coerce_bar_frame_df(frame)


def drop_bars(frame: pd.DataFrame, positions: Sequence[int]) -> pd.DataFrame:
    """Remove bars by position, creating a gap."""
    keep = [i for i in range(len(frame)) if i not in set(positions)]
    return frame.iloc[keep].reset_index(drop=True)


def kline_rows(frame: pd.DataFrame, *, timeframe: str = "1h", microseconds: bool = False) -> str:
    """Render canonical bars as a Binance kline CSV body."""
    bar_ms = timeframe_ms(timeframe)
    scale = 1000 if microseconds else 1
    lines = []
    for row in frame.itertuples(index=False):
        open_time = int(row.ts_open) * scale
        close_time = (int(row.ts_open) + bar_ms - 1) * scale
        lines.append(
            ",".join(
                [
                    str(open_time),
                    f"{row.open:.8f}",
                    f"{row.high:.8f}",
                    f"{row.low:.8f}",
                    f"{row.close:.8f}",
                    f"{row.volume:.8f}",
                    str(close_time),
                    f"{row.quote_volume:.8f}",
                    str(int(row.trades)),
                    "0.0",
                    "0.0",
                    "0",
                ]
            )
        )
    return "\n".join(lines) + "\n"


def make_kline_zip(
    frame: pd.DataFrame,
    *,
    name: str = "BTCUSDT-1h-2023-01.csv",
    timeframe: str = "1h",
    header: bool = False,
    microseconds: bool = False,
) -> bytes:
    """Build an in-memory Binance kline zip, with or without a header row."""
    body = kline_rows(frame, timeframe=timeframe, microseconds=microseconds)
    if header:
        body = ",".join(kline_columns()) + "\n" + body
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, body)
    return buffer.getvalue()


def ohlcv_rows(frame: pd.DataFrame) -> list[list[float]]:
    """Render canonical bars as ccxt ``fetch_ohlcv`` rows."""
    return [
        [
            float(row.ts_open),
            float(row.open),
            float(row.high),
            float(row.low),
            float(row.close),
            float(row.volume),
        ]
        for row in frame.itertuples(index=False)
    ]


# ---------------------------------------------------------------------------
# deterministic toy bars and strategies for engine tests
# ---------------------------------------------------------------------------
def toy_frame(
    rows: Sequence[Sequence[float]],
    *,
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    start_ts: int = EPOCH_2023,
    gap_filled: Sequence[bool] | None = None,
) -> BarFrame:
    """Build a :class:`BarFrame` from explicit ``(open, high, low, close)`` rows.

    Volume and trade count are fixed and non-zero so that no cost model divides
    by zero; timestamps are consecutive bars from ``start_ts``.  Every engine
    test that asserts an exact number uses this, so the number can be worked out
    on paper.
    """
    from quantlab.core.types import BarFrame, coerce_bar_frame_df, timeframe_ms

    bar_ms = timeframe_ms(timeframe)
    frame = pd.DataFrame(
        {
            "ts_open": [start_ts + i * bar_ms for i in range(len(rows))],
            "open": [float(r[0]) for r in rows],
            "high": [float(r[1]) for r in rows],
            "low": [float(r[2]) for r in rows],
            "close": [float(r[3]) for r in rows],
            "volume": [10.0] * len(rows),
            "quote_volume": [100_000.0] * len(rows),
            "trades": [10] * len(rows),
            "is_gap_filled": list(gap_filled) if gap_filled else [False] * len(rows),
        }
    )
    return BarFrame(coerce_bar_frame_df(frame), symbol=symbol, timeframe=timeframe)


def flat_frame(prices: Sequence[float], *, spread: float = 1.0, **kwargs) -> BarFrame:
    """Bars whose open and close are ``price`` and whose range is +/- ``spread``."""
    return toy_frame(
        [(p, p + spread, p - spread, p) for p in prices],
        **kwargs,
    )


def zero_cost_config(**overrides):
    """A config with no fees and no slippage, for isolating one mechanism at a time."""
    from quantlab.core.types import BacktestConfig, SlippageConfig

    base = {
        "initial_equity": 10_000.0,
        "fee_bps": 0.0,
        "slippage": SlippageConfig(model="fixed_bps", fixed_bps=0.0),
        "lot_step": 1e-12,
        "min_notional": 0.0,
        "bars_per_year": 8_760,
    }
    base.update(overrides)
    return BacktestConfig(**base)


class ScriptedStrategy:
    """Emits a fixed sequence of signals, one per bar.

    The clearest possible strategy: what it does is written down, so any surprise
    in a test is a surprise in the engine.
    """

    name = "scripted"
    version = "1"
    style = "bar_loop"
    params: ClassVar[dict] = {}
    warmup_bars = 0

    def __init__(self, script: Sequence[str], *, warmup: int = 0) -> None:
        from quantlab.core.types import RiskSpec, SizingSpec

        self.script = list(script)
        self.warmup_bars = warmup
        self.risk = RiskSpec()
        self.sizing = SizingSpec()
        self.seen: list[int] = []

    def prepare(self, params) -> None:
        self.seen = []

    def on_bar(self, ctx):
        from quantlab.core.types import Signal, SignalKind

        self.seen.append(ctx.i)
        kind = self.script[ctx.i] if ctx.i < len(self.script) else "flat"
        return Signal(SignalKind(kind))
