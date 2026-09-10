"""The bar-close backtest engine (master spec sections 8.4 and 8.6).

The whole engine is one pass over the bars, and every rule in it resolves
ambiguity **against the trader**.  That bias is the point.  A backtest exists to
decide whether an edge is real, and any place where the simulation guesses
generously is a place a search process will learn to exploit — reliably, and
invisibly, because the resulting equity curve looks like skill.

Per bar ``i``:

1. fill the order created at ``i-1``, at ``open[i]``, unless the bar is gap-filled
2. apply engine-side risk exits using bar ``i``'s own OHLC (§8.6)
3. accrue borrow, then mark to market at ``close[i]``
4. below the warm-up: record FLAT and stop here
5. ask the strategy for a signal, seeing bars ``[0..i]`` only
6. translate the signal into the order that bar ``i+1`` will fill

Step 2 sits before step 3 so that a position opened at ``open[i]`` can be stopped
out during bar ``i`` — which is what really happens, and is the pessimistic
reading. Spec §8.4 numbers the risk exits last; the ordering here is the same
rule stated unambiguously, and §8.6's own table ("for an open long position at
bar ``i``, using only bar ``i``'s own OHLC") is what it must mean.

Determinism: no clocks, no unseeded randomness, no dict-ordering dependence.
The same inputs produce the same result bit for bit, which is what INV-7 checks.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import pandas as pd

from quantlab.core.costs import SlippageModel, build_slippage_model
from quantlab.core.errors import EngineError, StrategyRuntimeError

# The fill arithmetic is NOT defined here. It lives in `core/execution.py`
# because `paper/broker.py` calls the same functions, and section 16.2 requires
# a paper fill to price identically to a backtest one — bit for bit, which two
# implementations cannot hold. If you are about to inline or "simplify" any of
# these, read the warning at the top of that module first: the expressions are
# grouped the way this engine has always evaluated them, and re-grouping them
# changes recorded trade prices in the last decimal place.
from quantlab.core.execution import (
    fee_of,
    needs_order,
    round_to_lot,
    size_from_notional,
    slipped_price,
)
from quantlab.core.strategy import Context, IndicatorCache, Strategy, resolve_params
from quantlab.core.types import (
    MAX_ENGINE_LOG_LINES,
    BacktestConfig,
    BacktestResult,
    BarFrame,
    Fill,
    Position,
    RiskSpec,
    Side,
    Signal,
    SignalKind,
    SizingSpec,
    Trade,
)

__all__ = ["ENGINE_VERSION", "SimpleBarEngine"]

#: Bumped whenever any fill or accounting rule changes (spec section 8.4).
#: Bumped to "2": the end-of-data close now lands on the last bar at which
#: trading was possible, rather than on a gap-filled one (ADR 0009). No golden
#: baseline moved — none of them ends on a synthetic bar — but the fill rule
#: changed, and spec 0.3 versions the execution model rather than sampling it.
ENGINE_VERSION: Final[str] = "2"

#: Tolerance of the independent equity reconstruction, relative to equity.
_ACCOUNTING_TOL: Final[float] = 1e-6

_EPS: Final[float] = 1e-12


@dataclass(slots=True)
class _PendingOrder:
    """A market order created at bar ``created_bar``, to fill at the next open."""

    created_bar: int
    created_ts: int
    target_fraction: float
    reference_equity: float


@dataclass(slots=True)
class _OpenPosition:
    """Accumulators for the trade currently open."""

    direction: int  # +1 long, -1 short
    qty: float = 0.0
    entry_notional: float = 0.0
    entry_qty: float = 0.0
    exit_notional: float = 0.0
    exit_qty: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    entry_bar: int = 0
    entry_ts: int = 0
    equity_at_entry: float = 0.0
    extreme_close: float = 0.0  # running max (long) / min (short) for the trailing stop

    @property
    def vw_entry(self) -> float:
        return self.entry_notional / self.entry_qty if self.entry_qty > _EPS else 0.0

    @property
    def vw_exit(self) -> float:
        return self.exit_notional / self.exit_qty if self.exit_qty > _EPS else 0.0


@dataclass(slots=True)
class _State:
    cash: float
    initial_equity: float
    position_qty: float = 0.0  # signed: positive long, negative short
    open_position: _OpenPosition | None = None
    pending: _PendingOrder | None = None
    realised_pnl: float = 0.0
    borrow_paid: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    turnover: float = 0.0
    ruined: bool = False
    trades: list[Trade] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)


class SimpleBarEngine:
    """Deterministic bar-close engine with next-open fills.

    Implements :class:`~quantlab.ports.engine.BacktestEngine`.  It knows nothing
    about optimisation, search or evolution: it takes a strategy and some bars
    and reports what would have happened.
    """

    name: str = "simple_bar"
    version: str = ENGINE_VERSION

    # -- public ------------------------------------------------------------
    def run(
        self,
        strategy: Strategy,
        bars: BarFrame,
        params: Mapping[str, Any],
        config: BacktestConfig,
    ) -> BacktestResult:
        """Simulate ``strategy`` over ``bars`` (spec sections 8.4, 8.6)."""
        n = bars.n_bars
        resolved = resolve_params(dict(getattr(strategy, "params", {})), params)
        strategy.prepare(resolved)

        risk = self._risk_for(strategy, config)
        sizing = self._sizing_for(strategy, config)
        slippage = build_slippage_model(config.slippage)
        cache = IndicatorCache(bars)
        warmup = max(0, int(getattr(strategy, "warmup_bars", 0)))

        state = _State(
            cash=float(config.initial_equity), initial_equity=float(config.initial_equity)
        )
        equity = np.full(n, float(config.initial_equity), dtype="float64")
        position_frac = np.zeros(n, dtype="float64")
        signals: list[str] = [SignalKind.FLAT.value] * n
        log: list[str] = []

        precomputed = self._precomputed_signals(strategy, bars, resolved, n)

        closes = bars.close
        timestamps = bars.ts_open
        gap_filled = bars.is_gap_filled

        for i in range(n):
            # 1. fill the order created at i-1
            if state.pending is not None:
                if bool(gap_filled[i]):
                    self._log(
                        log, f"bar={i} order_deferred reason=gap_filled ts={int(timestamps[i])}"
                    )
                else:
                    self._execute_pending(state, bars, i, config, slippage, log)

            # 2. engine-side risk exits, on this bar's own OHLC
            if state.open_position is not None and risk.is_active and not bool(gap_filled[i]):
                self._apply_risk_exit(state, bars, i, config, slippage, risk, log)

            # 3. borrow accrual, then mark to market
            if state.position_qty < -_EPS and config.short_borrow_bps_per_bar > 0:
                borrow = (
                    abs(state.position_qty)
                    * float(closes[i])
                    * config.short_borrow_bps_per_bar
                    * config.cost_multiplier
                    / 1e4
                )
                state.cash -= borrow
                state.borrow_paid += borrow

            marked = state.cash + state.position_qty * float(closes[i])
            self._check_accounting(state, marked, i, closes)

            # 3b. ruin. An account whose equity reaches zero is gone; a real one
            # is liquidated. Continuing to trade would let a strategy "recover"
            # through negative-equity arithmetic, and a search process would find
            # that and exploit it.
            if not state.ruined and marked <= 0.0:
                marked = self._liquidate(state, bars, i, config, slippage, log, marked)

            equity[i] = marked
            position_frac[i] = (
                state.position_qty * float(closes[i]) / marked if marked > _EPS else 0.0
            )

            if state.open_position is not None:
                self._update_trailing_extreme(state.open_position, float(closes[i]))

            # 4. warm-up, or a ruined account: no decision is taken at all
            if i < warmup or state.ruined:
                continue

            # 5. ask the strategy
            signal = self._signal_at(strategy, precomputed, i, bars, state, resolved, cache, config)
            signals[i] = signal.kind.value

            # 6. translate into the order bar i+1 will fill
            if i < n - 1:
                self._plan_order(
                    state, bars, i, signal, config, sizing, marked, cache, log, timestamps
                )

        if n:
            self._close_at_end_of_data(state, bars, config, slippage, log, equity, position_frac)

        return self._build_result(
            bars=bars,
            state=state,
            equity=equity,
            position_frac=position_frac,
            signals=signals,
            warmup=warmup,
            config=config,
            log=log,
        )

    # -- configuration -----------------------------------------------------
    @staticmethod
    def _risk_for(strategy: Strategy, config: BacktestConfig) -> RiskSpec:
        """Config wins over the strategy's class default (spec section 9.1).

        A candidate's mutated stop must override whatever the source declared,
        otherwise mutating risk parameters would silently do nothing.
        """
        if config.risk.is_active:
            return config.risk
        declared = getattr(strategy, "risk", None)
        return declared if isinstance(declared, RiskSpec) else RiskSpec()

    @staticmethod
    def _sizing_for(strategy: Strategy, config: BacktestConfig) -> SizingSpec:
        if config.sizing != SizingSpec():
            return config.sizing
        declared = getattr(strategy, "sizing", None)
        return declared if isinstance(declared, SizingSpec) else SizingSpec()

    # -- signals -----------------------------------------------------------
    def _precomputed_signals(
        self, strategy: Strategy, bars: BarFrame, params: Mapping[str, Any], n: int
    ) -> list[SignalKind] | None:
        """Run a vectorised strategy once, then shift by one bar (spec section 9.1).

        The shift gives vectorised strategies one more bar of lag than
        ``on_bar`` strategies.  That is deliberate: vectorised code is far easier
        to make accidentally non-causal, and the extra bar is the cheap insurance.
        """
        if getattr(strategy, "style", "bar_loop") != "vectorized":
            return None
        emitted = strategy.signals(bars.to_pandas(), params)  # type: ignore[attr-defined]
        if len(emitted) != n:
            raise StrategyRuntimeError(
                "vectorized strategy returned the wrong number of signals",
                expected=n,
                received=len(emitted),
            )
        shifted: list[SignalKind] = [SignalKind.FLAT]
        shifted.extend(SignalKind(str(value)) for value in list(emitted)[:-1])
        return shifted

    def _signal_at(
        self,
        strategy: Strategy,
        precomputed: list[SignalKind] | None,
        i: int,
        bars: BarFrame,
        state: _State,
        params: Mapping[str, Any],
        cache: IndicatorCache,
        config: BacktestConfig,
    ) -> Signal:
        if precomputed is not None:
            return Signal(precomputed[i])

        context = Context(
            i=i,
            bars=bars.window(i),
            position=self._position_value(state),
            equity=state.cash + state.position_qty * float(bars.close[i]),
            cash=state.cash,
            params=params,
            cache=cache,
            seed=config.seed,
        )
        try:
            signal = strategy.on_bar(context)
        except Exception as exc:
            raise StrategyRuntimeError(
                "strategy raised while deciding", bar=i, error=f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(signal, Signal):
            raise StrategyRuntimeError(
                "strategy returned something that is not a Signal", bar=i, returned=repr(signal)
            )
        return signal

    @staticmethod
    def _position_value(state: _State) -> Position:
        if state.open_position is None or abs(state.position_qty) <= _EPS:
            return Position.flat()
        held = state.open_position
        return Position(
            side=Side.LONG if held.direction > 0 else Side.SHORT,
            qty=abs(state.position_qty),
            entry_px=held.vw_entry,
            entry_bar=held.entry_bar,
        )

    # -- sizing ------------------------------------------------------------
    def _target_fraction(
        self,
        signal: Signal,
        config: BacktestConfig,
        sizing: SizingSpec,
        bars: BarFrame,
        i: int,
        cache: IndicatorCache,
        log: list[str],
    ) -> float:
        """Signed fraction of equity the strategy wants after bar ``i``."""
        if signal.kind is SignalKind.FLAT:
            return 0.0
        if signal.kind is SignalKind.SHORT and not config.allow_short:
            self._log(log, f"bar={i} short_ignored reason=allow_short_false")
            return 0.0

        magnitude = abs(float(signal.target_fraction)) * self._sizing_scale(
            sizing, bars, i, cache, config.bars_per_year
        )
        magnitude = min(magnitude, config.max_position_fraction)
        direction = 1.0 if signal.kind is SignalKind.LONG else -1.0
        return direction * magnitude

    @staticmethod
    def _sizing_scale(
        sizing: SizingSpec,
        bars: BarFrame,
        i: int,
        cache: IndicatorCache,
        bars_per_year: int,
    ) -> float:
        """Fraction-of-equity multiplier for the configured sizing mode.

        Every mode reads bars ``[0..i]`` only.  A mode whose input is still
        warming up falls back to ``fraction``, so sizing degrades to the fixed
        rule rather than to an arbitrary number.
        """
        base = float(sizing.fraction)
        if sizing.mode == "fixed_fraction":
            return base

        if sizing.mode == "volatility_target":
            series = cache.get("rolling_vol", n=sizing.vol_lookback)
            realised = float(series[i])
            if not math.isfinite(realised) or realised <= _EPS:
                return base
            annualised = realised * math.sqrt(float(bars_per_year))
            target = float(sizing.target_vol_annual or 0.0)
            return base * min(1.0, target / annualised) if annualised > _EPS else base

        # atr_risk: size so that an adverse move of one ATR costs atr_risk_pct.
        series = cache.get("atr", n=sizing.atr_period)
        current_atr = float(series[i])
        close = float(bars.close[i])
        if not math.isfinite(current_atr) or current_atr <= _EPS or close <= _EPS:
            return base
        risk_per_unit = current_atr / close
        return base * min(1.0, float(sizing.atr_risk_pct or 0.0) / risk_per_unit)

    def _plan_order(
        self,
        state: _State,
        bars: BarFrame,
        i: int,
        signal: Signal,
        config: BacktestConfig,
        sizing: SizingSpec,
        equity_now: float,
        cache: IndicatorCache,
        log: list[str],
        timestamps: np.ndarray,
    ) -> None:
        desired = self._target_fraction(signal, config, sizing, bars, i, cache, log)
        current = (
            state.position_qty * float(bars.close[i]) / equity_now
            if abs(equity_now) > _EPS
            else 0.0
        )
        # The deadband lives in `core/execution.py` so the paper runtime opens
        # and closes on the same bars this does (section 16.2).
        if not needs_order(desired, current):
            return
        state.pending = _PendingOrder(
            created_bar=i,
            created_ts=int(timestamps[i]),
            target_fraction=desired,
            reference_equity=equity_now,
        )
        self._log(log, f"bar={i} order_created target_fraction={desired:.6f}")

    # -- execution ---------------------------------------------------------
    def _slipped_price(
        self,
        bars: BarFrame,
        i: int,
        *,
        buying: bool,
        notional: float,
        config: BacktestConfig,
        slippage: SlippageModel,
        reference: float,
    ) -> float:
        """Apply slippage to ``reference``, always against the trader.

        The arithmetic is :func:`quantlab.core.execution.slipped_price`, which
        the paper broker also calls — section 16.2 requires a paper fill to
        price identically to a backtest one, and the only way to hold that is
        for there to be one implementation."""
        return slipped_price(
            reference,
            slippage.bps(bars, i, abs(notional)),
            buying=buying,
            cost_multiplier=config.cost_multiplier,
        )

    def _execute_pending(
        self,
        state: _State,
        bars: BarFrame,
        i: int,
        config: BacktestConfig,
        slippage: SlippageModel,
        log: list[str],
    ) -> None:
        order = state.pending
        assert order is not None
        state.pending = None

        reference = float(bars.open[i])
        if reference <= _EPS:
            self._log(log, f"bar={i} order_dropped reason=non_positive_open")
            return

        target_notional = order.target_fraction * order.reference_equity
        current_qty = state.position_qty

        # A reversal is two fills at the same open: close to flat, then open the
        # other way. Two fills means two fee charges, which is what really happens.
        reversing = (current_qty * target_notional < 0) or (
            abs(target_notional) <= _EPS and abs(current_qty) > _EPS
        )
        if reversing:
            price = self._slipped_price(
                bars,
                i,
                buying=current_qty < 0,
                notional=abs(current_qty) * reference,
                config=config,
                slippage=slippage,
                reference=reference,
            )
            self._fill(state, bars, i, -current_qty, price, reference, config, log, "signal")
            current_qty = 0.0

        # Size from the fill price, net of the fee that buying it will cost.
        # Dividing by (1 + fee_rate) is what keeps a 100 % target at exactly
        # 100 % of equity: without it the fee is financed, cash goes negative and
        # position_frac exceeds max_position_fraction (spec section 14.3's
        # G_SANITY gate would then fail on a correctly-behaved strategy).
        provisional = self._slipped_price(
            bars,
            i,
            buying=target_notional > current_qty * reference,
            notional=abs(target_notional),
            config=config,
            slippage=slippage,
            reference=reference,
        )
        if provisional <= _EPS:
            self._log(log, f"bar={i} order_dropped reason=non_positive_price")
            return
        target_qty = size_from_notional(target_notional, provisional, config)

        delta = self._round_to_lot(target_qty - current_qty, config.lot_step)
        if abs(delta) <= _EPS:
            return
        price = self._slipped_price(
            bars,
            i,
            buying=delta > 0,
            notional=abs(delta) * reference,
            config=config,
            slippage=slippage,
            reference=reference,
        )
        self._fill(state, bars, i, delta, price, reference, config, log, "signal")

    def _fill(
        self,
        state: _State,
        bars: BarFrame,
        i: int,
        delta_qty: float,
        price: float,
        reference: float,
        config: BacktestConfig,
        log: list[str],
        reason: str,
    ) -> None:
        """Execute ``delta_qty`` (signed) at ``price``, charging the fee.

        ``price`` already carries slippage; ``reference`` is the unslipped price
        the fill is measured against, so ``Fill.slippage_cost`` reports what the
        slippage actually cost rather than restating the fill price.
        """
        if abs(delta_qty) <= _EPS:
            return
        if price <= _EPS:
            self._log(log, f"bar={i} fill_dropped reason=non_positive_price")
            return

        buying = delta_qty > 0
        qty = abs(delta_qty)
        # The minimum-notional rule applies to opening and increasing fills only.
        # Enforcing it on the way out would trap a position that had shrunk below
        # the exchange minimum, which no exchange actually does.
        increasing = abs(state.position_qty + delta_qty) > abs(state.position_qty) + _EPS
        if increasing and qty * price < config.min_notional:
            self._log(
                log,
                f"bar={i} order_dropped reason=below_min_notional "
                f"notional={qty * price:.6f} min={config.min_notional}",
            )
            return

        fee = fee_of(qty, price, config)
        slippage_cost = abs(price - reference) * qty

        previous_qty = state.position_qty
        equity_before_fill = previous_qty * float(bars.close[i]) + state.cash
        state.cash += (-qty * price - fee) if buying else (qty * price - fee)
        state.position_qty = self._round_to_lot(previous_qty + delta_qty, config.lot_step)
        state.total_fees += fee
        state.total_slippage += slippage_cost
        state.turnover += qty * price

        state.fills.append(
            Fill(
                bar_index=i,
                ts=int(bars.ts_open[i]),
                side=Side.LONG if buying else Side.SHORT,
                qty=qty,
                ref_price=reference,
                fill_price=price,
                fee=fee,
                slippage_cost=slippage_cost,
            )
        )
        self._log(
            log,
            f"bar={i} fill side={'buy' if buying else 'sell'} qty={qty:.8f} "
            f"price={price:.8f} fee={fee:.8f} reason={reason}",
        )
        self._apply_to_trade(
            state, bars, i, delta_qty, price, fee, slippage_cost, reason, equity_before_fill
        )

    # -- trade ledger ------------------------------------------------------
    def _apply_to_trade(
        self,
        state: _State,
        bars: BarFrame,
        i: int,
        delta_qty: float,
        price: float,
        fee: float,
        slippage_cost: float,
        reason: str,
        equity_before_fill: float,
    ) -> None:
        qty = abs(delta_qty)
        if state.open_position is None:
            state.open_position = _OpenPosition(
                direction=1 if delta_qty > 0 else -1,
                entry_bar=i,
                entry_ts=int(bars.ts_open[i]),
                equity_at_entry=equity_before_fill,
                extreme_close=price,
            )
        held = state.open_position
        held.fees += fee
        held.slippage += slippage_cost

        opening = (delta_qty > 0 and held.direction > 0) or (delta_qty < 0 and held.direction < 0)
        if opening:
            held.entry_notional += qty * price
            held.entry_qty += qty
            held.qty += qty
        else:
            held.exit_notional += qty * price
            held.exit_qty += qty

        if abs(state.position_qty) <= _EPS:
            self._close_trade(state, bars, i, reason)

    def _close_trade(self, state: _State, bars: BarFrame, i: int, reason: str) -> None:
        held = state.open_position
        assert held is not None
        pnl = held.direction * held.exit_qty * (held.vw_exit - held.vw_entry) - held.fees
        equity_at_entry = held.equity_at_entry if abs(held.equity_at_entry) > _EPS else 1.0

        state.trades.append(
            Trade(
                trade_no=len(state.trades) + 1,
                side=Side.LONG if held.direction > 0 else Side.SHORT,
                entry_ts=held.entry_ts,
                entry_px=held.vw_entry,
                exit_ts=int(bars.ts_open[i]),
                exit_px=held.vw_exit,
                qty=held.entry_qty,
                fees=held.fees,
                slippage_cost=held.slippage,
                pnl=pnl,
                pnl_pct=pnl / equity_at_entry,
                bars_held=i - held.entry_bar,
                exit_reason=reason,  # type: ignore[arg-type]
            )
        )
        state.realised_pnl += pnl
        state.open_position = None

    # -- risk exits (spec section 8.6) -------------------------------------
    @staticmethod
    def _update_trailing_extreme(held: _OpenPosition, close: float) -> None:
        """Extend the trailing reference with a *closed* bar's close (INV-3)."""
        held.extreme_close = (
            max(held.extreme_close, close) if held.direction > 0 else min(held.extreme_close, close)
        )

    def _apply_risk_exit(
        self,
        state: _State,
        bars: BarFrame,
        i: int,
        config: BacktestConfig,
        slippage: SlippageModel,
        risk: RiskSpec,
        log: list[str],
    ) -> None:
        held = state.open_position
        assert held is not None
        entry = held.vw_entry
        if entry <= _EPS:
            return

        long = held.direction > 0
        high = float(bars.high[i])
        low = float(bars.low[i])
        open_px = float(bars.open[i])

        loss_exits: list[tuple[str, float]] = []
        if risk.stop_loss_pct is not None:
            stop_px = entry * (1 - risk.stop_loss_pct) if long else entry * (1 + risk.stop_loss_pct)
            if (low <= stop_px) if long else (high >= stop_px):
                loss_exits.append(
                    ("stop_loss", min(stop_px, open_px) if long else max(stop_px, open_px))
                )
        if risk.trailing_stop_pct is not None:
            trail_px = (
                held.extreme_close * (1 - risk.trailing_stop_pct)
                if long
                else held.extreme_close * (1 + risk.trailing_stop_pct)
            )
            if (low <= trail_px) if long else (high >= trail_px):
                loss_exits.append(
                    ("trailing_stop", min(trail_px, open_px) if long else max(trail_px, open_px))
                )

        if loss_exits:
            # Both a fixed and a trailing stop can trigger on one bar. Without an
            # intrabar path there is no way to know which came first, so take the
            # worse fill: the assumption that costs the strategy money.
            control, price = (
                min(loss_exits, key=lambda item: item[1])
                if long
                else max(loss_exits, key=lambda item: item[1])
            )
        elif risk.take_profit_pct is not None and self._take_profit_hit(
            risk, entry, long, high, low
        ):
            target = (
                entry * (1 + risk.take_profit_pct) if long else entry * (1 - risk.take_profit_pct)
            )
            control, price = "take_profit", (max(target, open_px) if long else min(target, open_px))
        elif risk.time_stop_bars is not None and (i - held.entry_bar) >= risk.time_stop_bars:
            control, price = "time_stop", float(bars.close[i])
        else:
            return

        self._log(log, f"bar={i} risk_exit control={control} price={price:.8f}")
        # Slippage applies to a risk exit exactly as to a signal exit (§8.6 rule 5).
        fill_price = self._slipped_price(
            bars,
            i,
            buying=state.position_qty < 0,
            notional=abs(state.position_qty) * price,
            config=config,
            slippage=slippage,
            reference=price,
        )
        self._fill(state, bars, i, -state.position_qty, fill_price, price, config, log, "stop")
        # A risk exit supersedes whatever the previous bar asked for.
        state.pending = None

    @staticmethod
    def _take_profit_hit(risk: RiskSpec, entry: float, long: bool, high: float, low: float) -> bool:
        target = entry * (1 + risk.take_profit_pct) if long else entry * (1 - risk.take_profit_pct)  # type: ignore[operator]
        return (high >= target) if long else (low <= target)

    def _liquidate(
        self,
        state: _State,
        bars: BarFrame,
        i: int,
        config: BacktestConfig,
        slippage: SlippageModel,
        log: list[str],
        marked: float,
    ) -> float:
        """Close everything at this bar's close and stop trading for good."""
        state.ruined = True
        state.pending = None
        if abs(state.position_qty) > _EPS:
            reference = float(bars.close[i])
            price = self._slipped_price(
                bars,
                i,
                buying=state.position_qty < 0,
                notional=abs(state.position_qty) * reference,
                config=config,
                slippage=slippage,
                reference=reference,
            )
            self._fill(state, bars, i, -state.position_qty, price, reference, config, log, "stop")
            marked = state.cash + state.position_qty * float(bars.close[i])
        self._log(log, f"bar={i} account_ruined equity={marked:.8f}")
        return marked

    # -- end of data -------------------------------------------------------
    def _close_at_end_of_data(
        self,
        state: _State,
        bars: BarFrame,
        config: BacktestConfig,
        slippage: SlippageModel,
        log: list[str],
        equity: np.ndarray,
        position_frac: np.ndarray,
    ) -> None:
        if abs(state.position_qty) <= _EPS or state.ruined:
            return

        # Close on the last bar at which trading was actually possible. The
        # engine defers orders and suppresses risk exits on gap-filled bars
        # because no trade could occur at a carried-forward price; the forced
        # close used to ignore that rule, so a dataset ending on a synthetic bar
        # liquidated every open position at a price that never traded.
        gap_filled = bars.is_gap_filled
        last = bars.n_bars - 1
        while last >= 0 and bool(gap_filled[last]):
            last -= 1
        if last < 0:
            # Every bar is synthetic: there is no real price to close at, so the
            # position stays open and says so, rather than being closed at fiction.
            self._log(log, "end_of_data_close skipped reason=no_real_bar")
            return

        reference = float(bars.close[last])
        price = self._slipped_price(
            bars,
            last,
            buying=state.position_qty < 0,
            notional=abs(state.position_qty) * reference,
            config=config,
            slippage=slippage,
            reference=reference,
        )
        self._fill(
            state, bars, last, -state.position_qty, price, reference, config, log, "end_of_data"
        )
        # The position is flat from the closing bar onwards, so every remaining
        # bar marks to cash. Without this the synthetic tail would still carry
        # the pre-close equity.
        closed_equity = state.cash + state.position_qty * float(bars.close[last])
        equity[last:] = closed_equity
        position_frac[last:] = 0.0
        self._log(log, f"bar={last} end_of_data_close equity={closed_equity:.8f}")

    # -- accounting --------------------------------------------------------
    def _check_accounting(self, state: _State, marked: float, i: int, closes: np.ndarray) -> None:
        """Reconstruct equity independently and refuse to continue if it differs.

        ``marked`` comes from cash and the position; this rebuilds the same number
        from the flows instead. A disagreement is an engine bug, never a strategy
        bug, so it propagates rather than being recorded as a failed run
        (spec section 18.2).
        """
        open_pnl = 0.0
        if state.open_position is not None:
            held = state.open_position
            # Scaling out realises part of the position before the trade closes,
            # so the open leg contributes both a realised and an unrealised term.
            realised_leg = held.direction * held.exit_qty * (held.vw_exit - held.vw_entry)
            unrealised_leg = (
                held.direction * abs(state.position_qty) * (float(closes[i]) - held.vw_entry)
            )
            open_pnl = realised_leg + unrealised_leg - held.fees
        expected = state.initial_equity + state.realised_pnl - state.borrow_paid + open_pnl
        if abs(expected - marked) > _ACCOUNTING_TOL * max(1.0, abs(marked)):
            raise EngineError(
                "equity does not reconcile with the recorded flows",
                bar=i,
                marked=marked,
                reconstructed=expected,
                difference=marked - expected,
            )

    # -- result ------------------------------------------------------------
    @staticmethod
    def _round_to_lot(qty: float, lot_step: float) -> float:
        return round_to_lot(qty, lot_step)

    @staticmethod
    def _log(log: list[str], message: str) -> None:
        if len(log) < MAX_ENGINE_LOG_LINES:
            log.append(message)

    def _build_result(
        self,
        *,
        bars: BarFrame,
        state: _State,
        equity: np.ndarray,
        position_frac: np.ndarray,
        signals: Sequence[str],
        warmup: int,
        config: BacktestConfig,
        log: list[str],
    ) -> BacktestResult:
        index = pd.Index(np.asarray(bars.ts_open, dtype="int64"), name="ts_open")
        return BacktestResult(
            equity=pd.Series(equity, index=index, name="equity", dtype="float64"),
            position_frac=pd.Series(
                position_frac, index=index, name="position_frac", dtype="float64"
            ),
            signals=pd.Series(list(signals), index=index, name="signal", dtype="object"),
            fills=tuple(state.fills),
            trades=tuple(state.trades),
            warmup_bars=warmup,
            n_bars=bars.n_bars,
            bars_per_year=config.bars_per_year,
            engine_name=self.name,
            engine_version=self.version,
            cost_summary={
                "total_fees": state.total_fees,
                "total_slippage": state.total_slippage,
                "total_borrow": state.borrow_paid,
                "turnover": state.turnover,
            },
            log=tuple(log),
            ruined=state.ruined,
        )
