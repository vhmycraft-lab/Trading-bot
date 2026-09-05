# QuantLab

AI-assisted quantitative trading **research and paper-trading** platform.

QuantLab imports historical Binance spot data, backtests Python strategies with
realistic fees and slippage, optimises them on training data only, validates
them with walk-forward analysis and statistical over-fitting tests, and runs the
survivors in **paper trading**.

> **QuantLab never places a real order.** There is no live-broker code path, and
> `tests/unit/test_architecture.py` fails the build if one is ever added
> (invariant INV-1 of `CLAUDE_CODE_MASTER_SPEC.md`).

---

## Status

Phases 1 (foundation) and 2 (historical market data) are complete.

| Area | Module | State |
|---|---|---|
| Configuration | `quantlab.core.config` | complete (all keys of spec §5, including the `evolution:` schema) |
| Errors | `quantlab.core.errors` | complete (spec §18.1) |
| Hashing / canonical JSON | `quantlab.core.hashing` | complete (spec §1.3) |
| Logging + redaction | `quantlab.core.logging` | complete (spec §17) |
| Secrets port + adapters | `quantlab.ports.secrets`, `quantlab.adapters.secrets` | complete (spec §21.1) |
| Database schema + migrations | `quantlab.adapters.store`, `migrations/` | complete (spec §6) |
| Core types, `BarFrame`, `BarWindow` | `quantlab.core.types` | complete (spec §7.2, §8.1) |
| Bar validation and gap handling | `quantlab.core.data_validation` | complete (spec §7.3) |
| Splits, embargo, walk-forward windows | `quantlab.core.splits` | complete (spec §5, §15.1) |
| Archive ingestion + Parquet store | `quantlab.adapters.data.binance_archive` | complete (spec §7.1, §7.2) |
| REST tail updates | `quantlab.adapters.data.ccxt_rest` | complete (spec §7.1) |
| Partition guard (INV-5) | `quantlab.adapters.data.guard` | complete (spec §7.4) |
| Indicators | `quantlab.core.indicators` | complete (spec §9.3, 14 indicators, all causal) |
| Strategy contract | `quantlab.core.strategy` | complete (spec §9.1) |
| Cost models | `quantlab.core.costs` | complete (spec §8.5) |
| Backtest engine | `quantlab.adapters.engine.simple_bar` | complete (spec §8.4, §8.6, §8.7) |
| Metrics | `quantlab.core.metrics` | complete (spec §10) |
| Baseline strategies | `strategies/` | complete (spec §9.4, §9.5) |
| Golden fixtures | `tests/fixtures/`, `tests/golden/` | complete (spec T15; 1 416 real bars) |
| CLI | `quantlab.cli` (`version`, `doctor`, `config`, `db`, `data`) | complete |

Not implemented yet, by design: the sandbox, the evolutionary optimiser, the
validation suite, the LLM proposer and the paper-trading runtime.

### The backtester

Deterministic, and independent of everything above it — it takes a strategy and
some bars and reports what would have happened. It knows nothing about search.

A decision at the close of bar `t` fills at the **open of bar `t+1`**. Risk
controls (stop-loss, take-profit, trailing and time stops) are applied by the
engine rather than by the strategy, because a stop is an intrabar event and the
`BarWindow` refuses intrabar data. Every ambiguity is resolved **against the
trader**: fills take the worse of trigger price and bar open, a stop-loss beats a
take-profit on the same bar, and no exit fires on a gap-filled bar.

### How strategies are searched

The optimiser is a **population-based evolutionary strategy**, specified in
spec §13 and `docs/EVOLUTION.md` and landing in phase F′. Each generation
evaluates 16 candidates on training data only, scores them with a gated
multi-objective fitness (net profit carries a weight of 0.02; win rate cannot
override negative expectancy or excessive drawdown, because both are hard gates),
keeps the strongest 12 subject to a diversity constraint, and refills the
population by mutation plus at least one novel candidate.

Every candidate evaluation counts as a trial against the deflated Sharpe ratio,
so a wider search must clear a higher significance bar. Full lineage — parent,
generation, and every mutation with the seed that drew it — is persisted and
replayable.

The `evolution:` configuration block is already present and validated
(`quantlab config show`), so a setting that could not work is rejected today
rather than during phase F′. Nothing reads those values yet.

---

## Requirements

* **Python 3.12.x** (`>=3.12,<3.13`)
* [**uv**](https://docs.astral.sh/uv/) for the environment and lockfile
* macOS or Linux

---

## Setup

```bash
git clone <this-repo> quantlab && cd quantlab

# 1. Install uv if you do not have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create the virtualenv and install everything (runtime + dev + dashboard)
make sync

# 3. Install the git hooks (ruff, mypy, gitleaks)
uv run pre-commit install

# 4. Provide secrets (optional at this stage — nothing in Phase 1 needs them)
cp .env.example .env
$EDITOR .env

# 5. Create the database
make db-upgrade

# 6. Verify the installation
uv run quantlab --version      # -> 0.1.0
uv run quantlab doctor
make check
```

`make check` runs formatting, linting, type checking and the test suite with
coverage gates. It must be green before every commit.

---

## Command line

```bash
uv run quantlab --help
```

| Command | Purpose |
|---|---|
| `quantlab --version` | Print the package version and exit |
| `quantlab doctor` | Check the environment, config, secrets and database |
| `quantlab config show` | Print the fully-resolved configuration as JSON |
| `quantlab config hash` | Print the `config_hash` stored with every experiment |
| `quantlab db upgrade` | Apply Alembic migrations to `project.db_path` |
| `quantlab db info` | Show the database path, size and migration revision |
| `quantlab data pull` | Download Binance monthly archives and build the dataset |
| `quantlab data update` | Extend the dataset to the latest closed bar via ccxt |
| `quantlab data validate` | Re-validate the stored dataset: hashes, order, gaps |
| `quantlab data info` | Print the manifest and the `dataset_id` |
| `quantlab data splits` | Print the split policy and its walk-forward windows |
| `quantlab data range` | Load a range and show what the backtester receives |

Global options accepted by every command:

```bash
quantlab --config configs/experiment.yaml --set optimize.n_trials=50 config show
```

* `--config PATH` — a YAML file merged over `configs/default.yaml`; repeatable.
* `--set section.key=value` — a single scalar override; repeatable and logged.
* Environment variables `QUANTLAB__<SECTION>__<KEY>` override everything else,
  e.g. `QUANTLAB__LOGGING__LEVEL=DEBUG`.

---

## Market data

```bash
# 1. Fetch history (verifies every archive's published SHA-256)
uv run quantlab data pull --symbol BTC/USDT --tf 1h --from 2017-08

# 2. Top up to the latest closed bar
uv run quantlab data update

# 3. Check what you have
uv run quantlab data info
uv run quantlab data validate
uv run quantlab data splits --windows
```

**Raw and processed data are kept apart**, and only the processed side is ever
loaded for a backtest:

```
data/raw/binance/BTCUSDT/1h/*.zip          downloaded archives, verified, never edited
data/binance/BTCUSDT/1h/year=YYYY/*.parquet processed: canonical, validated, hashed
data/binance/BTCUSDT/1h/manifest.json       file hashes + the dataset_id
```

The canonical bar schema is `ts_open` (int64 ms UTC, bar **open** time,
left-labelled), `open/high/low/close/volume/quote_volume` (float64), `trades`
(int64) and `is_gap_filled` (bool). Ingestion sorts, de-duplicates and validates
every source identically; runs of up to three missing bars are filled flat and
flagged, longer gaps are refused unless you pass `--allow-gaps`. Every Parquet
file's hash is checked against the manifest before it is read, so an edited byte
fails the run instead of quietly changing a backtest.

### Two guarantees against using future data

**Within a run — no look-ahead (INV-3).** A strategy deciding at bar `i` is
handed a `BarWindow` over `bars[0..i]`. Every accessor stops at `i`; reading
further raises `LookaheadError` rather than returning a number, so the failure is
loud instead of showing up as a suspiciously good equity curve.

**Across runs — no peeking at the held-out data (INV-5).** `build_container`
wraps the data source in a `PartitionGuard` for every profile except `lockbox`.
Ask for a range that reaches the test partition and you get a `LockboxViolation`,
not a shorter result — silently clipping would turn a bug into a wrong backtest.
Even the *extent* of the held-out data stays hidden: `available_range` is clipped
below the test start.

---

## Configuration

`configs/default.yaml` is authoritative and documented in spec §5. Every value
is validated by pydantic with `extra="forbid"`, so a misspelled key is an error
rather than a silently ignored line.

The resolved configuration is hashed (SHA-256 over canonical JSON) into
`AppConfig.config_hash`; the hash and the resolved JSON are stored with every
experiment so a run can be reproduced from its inputs alone.

Thresholds, split boundaries and cost defaults **must not** be changed to make a
strategy look better. Changing one requires an ADR in `docs/DECISIONS/`.

---

## Secrets

Secrets never live in YAML, never appear on the command line, and are never
logged. `Secrets.get(name)` resolves in this order:

1. macOS Keychain — service `quantlab`, account = the secret name
2. `.env` in the repository root (git-ignored)
3. `ConfigError("missing secret <name>; run quantlab doctor")`

```bash
# macOS keychain (preferred)
uv run quantlab doctor            # tells you exactly which names are missing
security add-generic-password -s quantlab -a GLM_API_KEY -w

# or, for local development
cp .env.example .env
```

See `docs/SECURITY.md`. `gitleaks` (configured by `.gitleaks.toml`) runs in
pre-commit and `tests/unit/test_no_secrets.py` scans the tree on every
`make check` (INV-2).

---

## Database

SQLite with `STRICT` tables, foreign keys enforced and WAL journaling. The
schema is spec §6; migrations are managed by Alembic in `migrations/`.

```bash
make db-upgrade                    # or: uv run quantlab db upgrade
uv run quantlab db info
```

The database path comes from `project.db_path` (default `./quantlab.db`) and is
git-ignored.

---

## Repository layout

```
configs/        YAML configuration (authoritative defaults + split policies)
data/           git-ignored: downloaded market data
artifacts/      git-ignored: run artifacts, logs, reports
docs/           architecture, security, metrics, validation, ADRs
migrations/     Alembic migration scripts
scripts/        developer utilities
src/quantlab/   the package (see docs/ARCHITECTURE.md for the layering rules)
strategies/     strategy sources; strategies/generated/ is git-ignored
tests/          unit, property, golden, leakage, synthetic and integration tests
```

Layering is enforced mechanically (INV-8): `core` imports only `core`; `ports`
imports `core`; adapters import `core` and `ports`; only `container.py`, `cli/`
and `tests/` may import `adapters`.

---

## Test fixtures

`tests/fixtures/data/btcusdt_1h_2023-01_02.parquet` holds **1 416 real BTC/USDT
hourly bars** from January and February 2023, downloaded from the Binance
archive and verified against its published SHA-256. It is committed (72 KB) so
the golden tests run offline.

```bash
uv run python scripts/make_fixtures.py data     # re-download (needs network)
uv run python scripts/make_fixtures.py golden   # re-record goldens (offline)
```

The goldens exist to **fail**. Any change to a fill rule, a cost, a risk exit or
a metric definition changes them, and that failure forces the change to be
deliberate: regenerating them without bumping `engine_version` is caught, and
bumping `engine_version` without regenerating them is caught too.

## Development

```bash
make fmt          # format
make lint         # ruff
make type         # mypy (strict on core/, ports/, sandbox/)
make test         # pytest + coverage gates
make check        # all of the above — the commit gate
make audit        # pip-audit
```

Tests are organised as `tests/unit`, `tests/property`, `tests/golden`,
`tests/leakage`, `tests/synthetic` and `tests/integration`. Everything runs
offline; the single networked smoke test is marked `@pytest.mark.live` and is
excluded from `make check`.

Coverage gates (enforced by `scripts/check_coverage.py`): `core/`, `sandbox/`,
`research/redaction.py` and `cli/lockbox.py` at 90 %, overall at 75 %.

---

## Documentation

* `CLAUDE_CODE_MASTER_SPEC.md` — the normative specification (v1.1)
* `docs/ARCHITECTURE.md` — layers, ports and the composition root
* `docs/METRICS.md` — metric definitions, mirroring spec §10
* `docs/EVOLUTION.md` — evolutionary optimiser design notes (non-normative)
* `docs/SECURITY.md` — secrets, sandboxing, supply chain, repo hygiene
* `docs/DECISIONS/` — architecture decision records

## Licence

Proprietary. Not investment advice; for research and paper trading only.
