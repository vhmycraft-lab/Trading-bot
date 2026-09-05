# Metrics

Normative source: `CLAUDE_CODE_MASTER_SPEC.md` §10. This document mirrors those
definitions and **must be updated in the same commit as any change to them**.

Implemented in `quantlab.core.metrics`; tested against a hand-computed fixture in
`tests/unit/test_metrics.py`.

---

## The rule that shapes everything

Every metric is computed from the **equity series and the trade list only**.
Nothing re-reads prices — `buy_hold_return` is the single exception, and it is
supplied by the caller rather than looked up.

That constraint is what makes metrics comparable across engines: two engines that
produce the same equity curve get the same numbers by construction, so a metric
can never disagree with the accounting that produced it.

**`None` means undefined, not zero.** A profit factor with no losing trades, a
Sharpe ratio over a flat curve, a CAGR across three days — each is genuinely
undefined, and reporting `0.0` invites a comparison that means nothing.

## Conventions

Returns are measured on bars **after the warm-up**. `T` is the number of such
bars and `Y = T / bars_per_year`. `bars_per_year` is 8 760 for 1h, 525 600 for
1m and 365 for 1d: crypto trades continuously, so there are no session counts.

`E_0` and `E_T` are the first and last equity values after the warm-up.

---

## Return and growth

| metric | definition |
|---|---|
| `net_return` | `E_T / E_0 − 1` |
| `cagr` | `(E_T / E_0)^(1/Y) − 1`; `None` when `Y < 1/365` |
| `buy_hold_return` | supplied by the caller: `close[−1] / open[warmup] − 1`, net of one round trip of costs |
| `excess_vs_buy_hold` | `net_return − buy_hold_return` |

## Risk

| metric | definition |
|---|---|
| `max_drawdown` | `max_t (1 − E_t / max_{s≤t} E_s)`, **clamped to `[0, 1]`** |
| `max_drawdown_bars` | longest peak-to-recovery span, in bars |
| `ann_volatility` | `std(r, ddof=1) · sqrt(bars_per_year)` |
| `calmar` | `cagr / max_drawdown`; `None` when either is undefined |

Per-bar returns are `r_t = E_t / E_{t−1} − 1`.

Two details worth stating:

* **The drawdown clamp.** A blown-up short can leave equity below zero, where the
  raw formula exceeds 1. Spec §10 defines drawdown as a fraction, so the value is
  capped: losing everything is 100 %, and the fact that the account also owes
  money is reported by `BacktestResult.ruined` and by the equity series, not by a
  drawdown of 200 %.
* **Peak-to-recovery, not peak-to-trough.** A dip that lasts one bar and recovers
  on the next spans two. A curve that only rises has spans of zero, not of one:
  there was no drawdown to wait out.

## Risk-adjusted return

| metric | definition |
|---|---|
| `sharpe` | `mean(r − rf_bar) / std(r, ddof=1) · sqrt(bars_per_year)`; `None` when `std == 0` |
| `sortino` | `mean(r − rf_bar) / sqrt(mean(min(r − rf_bar, 0)²)) · sqrt(bars_per_year)`; `None` when the downside deviation is 0 |
| `sharpe_ci_low` / `sharpe_ci_high` | Lo (2002): `SR ± 1.96 · sqrt((1 + SR²/2) / T)` |

The risk-free rate is 0 by spec §1.3, so `rf_bar = 0`.

The confidence interval is not decoration. A Sharpe of 1.5 over 200 bars has an
interval roughly `[0.9, 2.1]`; the same estimate over 20 000 bars is `[1.4, 1.6]`.
Reporting the point estimate alone makes those look like the same claim.

## Trade statistics

| metric | definition |
|---|---|
| `n_trades` | number of completed round trips |
| `win_rate` | `#(pnl > 0) / n_trades` |
| `profit_factor` | `Σ pnl⁺ / abs(Σ pnl⁻)`; `None` with no losers, which also forces `low_trade_warning` |
| `avg_trade_pct` | `mean(pnl_pct)` |
| `avg_trade_usdt` | `mean(pnl)` |
| `expectancy_pct` | `win_rate · mean(pnl_pct⁺) − (1 − win_rate) · abs(mean(pnl_pct⁻))` |
| `avg_bars_held` | `mean(bars_held)` |
| `longest_losing_streak` | longest run of consecutive losing trades |
| `low_trade_warning` | `n_trades < 30`, or profit factor undefined |

`pnl` is net of the entry and exit fees and of slippage, which is already
embedded in the fill prices. `pnl_pct = pnl / equity_at_entry_bar`.

## Concentration and consistency

| metric | definition |
|---|---|
| `top5_profit_share` | share of gross profit from the five largest winners, clamped to `[0, 1]` |
| `retention_1` / `_3` / `_5` | `Σ pnl(remaining) / Σ pnl(all)` after removing the top 1 / 3 / 5 **winning** trades |
| `consistency` | share of non-overlapping 720-bar windows that ended in profit |
| `exposure` | `#(position_frac ≠ 0) / T` |
| `turnover` | total traded notional, from the engine's cost summary |

**Retention** is the sharpest of these. It answers "how much of this profit
survives if the few best trades had not happened?", computed on the trade list
exactly and without re-simulation — cheap enough to run for every candidate in
every generation of the search (spec §14.4).

| observation | reading |
|---|---|
| `retention_1 ≈ 0.9` | profit is broadly distributed; the edge is plausible |
| `retention_1 ≈ 0.5` | half the profit came from one trade |
| `retention_1 ≤ 0` | the strategy loses money without its single best trade |

Only *winners* are removed: with two winners, `k = 5` must not start deleting the
least-bad losses, which would make the number improve. `None` when total pnl is
not positive, where the ratio is uninterpretable — dividing by a loss flips its
sign.

Retention is monotonically non-increasing in `k`, which is asserted as a
property test.

---

## Known bias

`top5_profit_share`, `retention_*` and `consistency` together encode a
preference for frequent, evenly distributed profits. A genuine trend-following
strategy legitimately earns most of its money in a few large moves and will score
poorly on all three.

That is a deliberate bias of the default configuration, not an oversight. A
trend-following campaign needs different thresholds, chosen consciously and
recorded in an ADR (see `docs/EVOLUTION.md` §3).
