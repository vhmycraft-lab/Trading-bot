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

Phase 1 (foundation) is complete. The repository currently provides:

| Area | Module | State |
|---|---|---|
| Configuration | `quantlab.core.config` | complete (all keys of spec §5) |
| Errors | `quantlab.core.errors` | complete (spec §18.1) |
| Hashing / canonical JSON | `quantlab.core.hashing` | complete (spec §1.3) |
| Logging + redaction | `quantlab.core.logging` | complete (spec §17) |
| Secrets port + adapters | `quantlab.ports.secrets`, `quantlab.adapters.secrets` | complete (spec §21.1) |
| Database schema + migrations | `quantlab.adapters.store`, `migrations/` | complete (spec §6) |
| CLI skeleton | `quantlab.cli` (`version`, `doctor`, `config`, `db`) | complete |

Not implemented yet, by design: the backtest engine, indicators, metrics,
sandbox, optimiser, validation suite, LLM researcher and paper-trading runtime.

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

Global options accepted by every command:

```bash
quantlab --config configs/experiment.yaml --set optimize.n_trials=50 config show
```

* `--config PATH` — a YAML file merged over `configs/default.yaml`; repeatable.
* `--set section.key=value` — a single scalar override; repeatable and logged.
* Environment variables `QUANTLAB__<SECTION>__<KEY>` override everything else,
  e.g. `QUANTLAB__LOGGING__LEVEL=DEBUG`.

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

* `CLAUDE_CODE_MASTER_SPEC.md` — the normative specification
* `docs/ARCHITECTURE.md` — layers, ports and the composition root
* `docs/SECURITY.md` — secrets, sandboxing, supply chain, repo hygiene
* `docs/DECISIONS/` — architecture decision records

## Licence

Proprietary. Not investment advice; for research and paper trading only.
