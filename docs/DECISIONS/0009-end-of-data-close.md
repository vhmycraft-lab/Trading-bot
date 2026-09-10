# 9. The forced final close must land on a bar that traded

* **Status:** Accepted
* **Date:** 2026-09-10

## Context

The engine treats gap-filled bars as untradeable, in two places and for one
reason — no trade could occur at a price the ingester carried forward:

* a pending order is deferred (`order_deferred reason=gap_filled`);
* engine-side risk exits are suppressed.

`_close_at_end_of_data` had no such check. Constructed rather than reasoned
about: a series of six real bars followed by three synthetic ones, with a
strategy that never exits, produced

```
bar=8 fill side=sell qty=100.00000000 price=130.00000000 reason=end_of_data
```

— a fill on bar 8, which is `is_gap_filled=True`. The engine's own invariant,
that no fill is ever recorded on a synthetic bar, was violated by the one code
path that bypasses the order loop.

The carried-forward price equals the last real close by construction, so the
*price* was not wrong. What was wrong: the fill was attributed to a bar on which
nothing traded, `bars_held` counted 7 instead of 4 (charging the position for
synthetic holding time), `exit_ts` named a timestamp with no market, and equity
carried its pre-close value across the synthetic tail.

This is reachable with real data. The BTC/USDT 1h dataset carries 169 gap-filled
bars (ADR 0005), and a `data pull` whose final month is incomplete, or a REST
tail that stops short, leaves synthetic bars at the end.

## Decision

Walk back from the end to the last bar with `is_gap_filled=False` and close
there. Equity and `position_frac` are then flat from that bar onward, because
the position is.

If **every** bar is synthetic there is no real price to close at. The position
stays open and the engine logs `end_of_data_close skipped reason=no_real_bar`,
rather than inventing a liquidation.

`ENGINE_VERSION` is bumped `1` → `2`. No golden baseline moved — none of the
five ends on a synthetic bar — but the fill rule changed, and spec §0.3 requires
the execution model to be versioned rather than quietly adjusted. A golden that
did not move is not a reason to skip the bump; it is a statement about coverage.

## Consequences

* No fill is recorded on a gap-filled bar by any path, so the invariant the
  paper broker and the golden baselines rest on is now true without exception.
* `bars_held` and `exit_ts` for an end-of-data close describe real market time.
* Prior runs recorded under `engine_version = "1"` remain reproducible against
  that version; they are not silently re-interpreted.

## Checked at the same time, and sound

* **Volatility suppression through gaps.** Holding across a 75-bar gap-filled
  stretch was tested for flattering the risk metrics. It does not: Sharpe fell
  (5.32 → 5.12) and annualised volatility rose slightly, because zero-return
  bars reduce the mean at least as much as the dispersion. Drawdown and net
  return were unchanged. Hypothesis refuted.
* **Minimum notional.** Enforced on increasing fills only, so a position that
  shrank below the exchange minimum is not trapped; a sub-minimum order is
  dropped entirely rather than granted for free.
* **Costs.** Every fill is charged `fee_of`, including the end-of-data close.
  There is no path to exposure without cost, and `apply_environment` varies only
  `initial_equity` and slippage — never `fee_bps`.

## Residual, recorded not fixed

`is_gap_filled` is deliberately absent from `PRICE_COLUMNS` so that "a strategy
has no business trading on it". But a gap-filled bar has `volume == 0.0` and
`trades == 0` exactly, and `volume` **is** readable. The flag is hidden; the
fact is not.

No exploit follows from it — orders on those bars are deferred regardless, and
knowing a re-pricing is coming says nothing about its direction — so the
protection that matters is the deferral, not the concealment. It is recorded
because the concealment reads like the protection, and because volume-based
indicators genuinely see a stretch of zeros there, which is a data-quality
effect on indicators rather than an exploitable one.
