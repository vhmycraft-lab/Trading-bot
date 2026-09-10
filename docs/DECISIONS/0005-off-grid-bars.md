# 5. Off-grid bars are dropped, never snapped

* **Status:** Accepted
* **Date:** 2026-09-10

## Context

Ingesting BTC/USDT 1h from the Binance public archive failed:

```
error ts_open is not aligned to the timeframe grid
      (row=4189, timeframe='1h', ts='2018-02-09T09:28:14.789000Z')
```

The check was right and the data was not corrupt. Scanning all 65 cached monthly
archives (46,982 rows) found exactly one affected run, in `BTCUSDT-1h-2018-02`:

| what | when |
| --- | --- |
| last bar before the outage, truncated to 28m14.788s | `2018-02-08T00:00:00Z` |
| no data at all, 33 hours | `02-08T00:28Z` .. `02-09T09:28Z` |
| 43 bars, each exactly 60.000 min, phase-shifted +28m14.789s | `02-09T09:28:14.789Z` .. `02-11T03:28:14.789Z` |
| re-sync bar, truncated to 31m46s, back onto the hour | `02-11T03:28:14.789Z` |
| normal hourly bars resume | `02-11T04:00:00Z` |

The exchange went down mid-bar, and when it came back it restarted its kline
windows from the resume instant rather than realigning to the UTC hour. The
archive is an honest record of that. The 43 bars are real observations; they
simply cannot be placed on an hourly grid. The published re-sync bar even
overlaps the following one by a second (`04:00:00.999` vs `04:00:00.000`).

No other month in the training range is affected.

## Decision

Add an explicit `off_grid` policy to `normalise_bars`, defaulting to `"error"`.
`off_grid="drop"` (CLI: `--drop-off-grid`) discards off-grid rows, records each
contiguous run as an `OffGridWindow` in the `ValidationReport`, and leaves the
window they occupied as an ordinary gap.

The rejected alternatives matter more than the chosen one:

* **Relaxing the alignment check** would let a bar opening at `:28` be indexed as
  if it opened at `:00`. Every downstream join, resample and split boundary
  assumes bar *k* covers `[t0 + k·Δ, t0 + (k+1)·Δ)`. This is the one option that
  produces silently wrong backtests, which is exactly what §7.3 fails closed to
  prevent.
* **Snapping `ts_open` to the nearest hour** fabricates the observation: it
  asserts a bar covered 09:00–10:00 when it covered 09:28–10:28. It also
  collides — the re-sync bar would snap onto `04:00`, which a real bar owns.
* **Re-deriving hourly bars** from finer data cannot recover intra-bar structure
  we do not have, and the 1m archive has the same outage.
* **Excluding the affected months** would cost six months of training data to
  describe a three-day event.

Dropping is the only option that neither invents data nor mislabels it. The
cost is honest and stated: three days of February 2018 are not available for
backtesting.

`validate_bars` deliberately gets no policy knob. Dropping is an *ingestion*
decision; a stored dataset that still contains an off-grid bar is a defect in
whatever wrote it, and re-forgiving it on read would hide that.

`--drop-off-grid` does not imply `--allow-gaps`. The hole is a 75-bar gap and
still has to be accepted on its own, so admitting this data takes two separate,
deliberate acknowledgements.

## Consequences

* The window becomes gap-filled flat bars carrying `is_gap_filled=True`. The
  engine defers orders and suppresses risk exits on those bars, and
  `is_gap_filled` is excluded from `PRICE_COLUMNS`, so a strategy can neither
  trade the window nor read the flag and condition on it. Time that existed is
  preserved as time on which no decision may be made.
* `ValidationReport.ok` is false whenever a window was dropped, so a caller
  checking `ok` cannot miss it, and `summary()` prints the window.
* BTC/USDT 1h ingests as 47,108 bars over 2017-08-17 .. 2022-12-31, of which 169
  are synthetic (30 across 17 short gaps, 139 across 10 long ones). 2018 holds
  exactly 8,760 bars — a complete grid. `quantlab data validate`, which runs the
  strict validator with no policy, passes.
* A position held through the window marks to the stale pre-outage close and
  then re-prices on the first real bar. That is the pre-existing behaviour of
  every long gap accepted via `--allow-gaps`, not something this record changes.
* The 28-minute truncated bar at `2018-02-08T00:00:00Z` is kept: its `ts_open` is
  on the grid and its OHLC is a real observation. The schema has no way to say
  "this bar covers less wall-clock than nominal", and one such bar at an outage
  boundary is inherent to all vendor bar data.

## Deviation recorded by this commit

Spec §7.3 is amended to name the policy. No threshold, split boundary, metric or
cost default changed. `engine_version` is untouched: the fill and accounting
rules are unchanged.
