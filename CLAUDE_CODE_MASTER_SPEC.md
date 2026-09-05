# CLAUDE_CODE_MASTER_SPEC.md

**Project:** QuantLab — AI-assisted quantitative trading *research and paper-trading* platform
**Spec version:** 1.1.9 (2026-09-05) · **Companion documents:** `QuantLab_Implementation_Plan.md` (rationale), `docs/EVOLUTION.md` (evolutionary optimiser rationale, non-normative). This file is normative; where any two disagree, this file wins.

**Change log**

| Version | Change |
|---|---|
| 1.0 | Initial specification. |
| 1.1.9 | Three points settled while implementing the runner. (a) §11.1 gains `StrategyEvaluator` in `ports/engine.py`: `experiments/` may import only `core` and `ports` (INV-8) and must not hold a strategy object at all (INV-4), so the runner is handed something already bound to one strategy that only needs a segment. Every evaluation goes through the sandbox, the platform's own baselines included — a second, faster in-process path would be a second set of numbers. (b) §11.2's `force` **re-executes and returns the fresh result without rewriting the stored run**: §6 permits no deletion or rewrite of a successful run, whose numbers may already have been cited, so "re-run" means verification, which is also what §11.4's `reproduce` does. A *failed* run is deleted and re-executed, as §11.2 says. (c) §11.3 requires `packages` from `importlib.metadata`, which INV-4's blanket ban on `importlib` outside `sandbox/` would forbid. The ban is narrowed by exactly one name — `importlib.metadata` reads distribution metadata and executes nothing — and `tests/unit/test_architecture.py` asserts both that the exemption stays that narrow and that only `experiments/env.py` uses it. |
| 1.1.8 | §11.1 gains two members found necessary while wiring the loader. **`SourceStore`**: the immutable copy of a strategy's *source* is keyed by `strategy_id` and outlives every run that cites it, so it cannot live in the run-keyed `ArtifactStore` of §11.3 without filing inputs among outputs; `FileSourceStore` roots it at `strategies/generated/` per §2 and §21.7, and `strategy_version.code_path` is a key into it rather than a machine path. **`ExperimentStore.get_strategy_version`**: reading one version by walking `lineage` answered a different question and cost a traversal to answer it. Also: §9.1's automatic probe for vectorised strategies runs in the **sandbox**, not in the loading process — at load the source has passed the AST check and nothing else, which is not enough to run it where trust is decided (INV-4) — and a loader without the bars, sandbox and engine the probe needs refuses a vectorised strategy rather than admitting it unchecked. |
| 1.1.7 | §14.2 made precise while implementing the probe. (a) The reversed tail is joined by two **scaled** tail replacements. A strategy reading exactly one bar ahead consumes the bar of slack §9.1 grants a vectorised strategy, so truncation cannot see it and reversal catches it at one bar only — where whether a boolean flips is a coin toss. Moving the replaced prices far up and far down makes that comparison come out differently in at least one replacement, which turns a probabilistic catch into a certain one; an honest strategy is unaffected by all three by construction. (b) The comparison horizons are stated: truncation at `k` compares signals over `[0, k]` and positions over `[0, k)` (the truncated run's last bar is its end-of-data close); a tail replaced from `p` compares signals over `[0, p + lag)` and positions over `[0, p)`, where `lag` is 1 for `vectorized` and 0 for `bar_loop`. (c) The "same-bar `high` breakout filled at same-bar open" fixture is expressed as a forward-looking channel: this engine fills at the next open (§8.4) and shifts a vectorised signal a further bar, so the literal form is causal here and there would be nothing to detect. |
| 1.1.6 | §6 clarified while implementing the store: `split_policy.wf_json` holds the **canonical policy document** the split was built from. The column had no stated contents, and the row otherwise cannot reproduce the policy it came from — the schema has no `symbol`/`timeframe`/source columns, and `split_id` is a hash of that document, so a policy rebuilt without it would not hash to its own primary key. Walk-forward windows are a function of the document plus configuration (§7.3) and are still derived, not stored. |
| 1.1.5 | §21.3 made precise while implementing the sandbox: the child runs the **engine** as well as the strategy — a `bar_loop` strategy is handed live engine state on every bar, so the two cannot be split across processes — and the engine's name therefore travels in the request, since the sandbox layer may not import an adapter (INV-8). The name is constrained to `quantlab.adapters.engine.*` and re-checked at the point of use. §19's "child never imports `quantlab.adapters`" is amended accordingly: no adapter *other than the named engine*. See ADR `docs/DECISIONS/0004`. |
| 1.1.4 | §9.2 made precise while implementing the AST checker: every rule now has a stable machine-readable **violation code**, the checker reports *all* violations at once rather than the first, and the hand-rolled-stop rule of §9.1 is stated as a rejection (`E_INTRABAR_STOP`) with its false-positive boundary defined — only a `high`/`low` comparison that also mentions an entry price is refused. |
| 1.1.3 | Addition found while implementing the engine: new §8.7 **Ruin**. Equity reaching zero now liquidates the account and stops trading, instead of continuing to trade a negative balance. |
| 1.1.2 | Correction found while implementing the engine: `G_SANITY` (§14.3) compared the *realised* `position_frac` against `max_position_fraction + 1e-9`, which no correct run can satisfy — a position sized at the deciding bar's close is marked one bar later, after the market has moved. The gate now allows a documented drift allowance, and the exact no-drift invariant moved to where it holds. |
| 1.1.1 | Corrections to 1.1, found while implementing the config schema: `mutation.structural` now carries one weight per operator in the §13.4 table (`replace_indicator` and `change_tree_mode` were missing); `diversity.max_immigrants` default lowered 6 → 4 and constrained to `n_offspring + n_immigrants`, since an immigrant boost displaces offspring and never a survivor. |
| 1.1 | §13 replaced: the sequential *propose → backtest → modify* research loop is superseded by an **evolutionary optimiser** over a population of strategy candidates. Adds the strategy **genome** and first-class risk controls (§8.4, §9), multi-objective **fitness** (§13.3), **mutation** operators (§13.4), **diversity** management (§13.5), **lineage** persistence (§6), the **trade-removal** robustness test (§14.4), and invariants INV-9…INV-11. The LLM's role changes from sequential author to genome proposer (§12). See ADR `docs/DECISIONS/0003`. |
**Audience:** Claude Code. Every instruction below is addressed to you, the implementing agent.

The key words MUST, MUST NOT, SHOULD, MAY are to be read as in RFC 2119.

---

## 0. Working protocol — read before touching any file

### 0.1 Incremental execution is mandatory

You MUST NOT implement the whole project, or a whole phase, in one operation. Work strictly one **step** at a time (steps are enumerated in §22). For every step:

```
1. Read this spec's section(s) referenced by the step. Re-read §0 if the step touches a 🔒 invariant.
2. Implement ONLY the component named in the step. No speculative work on later steps.
3. Run: make check          (ruff format --check, ruff check, mypy, pytest)
4. If anything fails: read the full failure output. Do not guess. Identify the root cause.
5. Fix the root cause. Never weaken a test, skip a test, or loosen a threshold to make it pass.
6. Run make check again. Repeat 4–6 until green.
7. Verify the step's acceptance criteria (§22) explicitly, one by one, and print the evidence.
8. Commit: "T<NN>: <step title>" on branch task/<NN>-<slug>. Then stop and report.
```

You MUST stop after each step and produce a short report: what was built, test counts before/after, acceptance criteria with evidence, open questions. Do not begin the next step in the same operation unless the human explicitly says "continue".

### 0.2 Non-negotiable invariants (🔒)

These MUST hold at every commit from the moment they are introduced. Each has an automated test; you MUST NOT delete or weaken those tests.

| ID | Invariant | Enforced by |
|---|---|---|
| INV-1 | No code path can sign, send, or simulate sending a real-money order to any exchange. No class named `LiveBroker`, `RealBroker`, `ExchangeBroker`; no reference to `create_order`, `createOrder`, `create_market_order`, `private_post_order` anywhere in `src/`. | `tests/unit/test_architecture.py` |
| INV-2 | No API key, secret, token or password literal in the repository. | `gitleaks` pre-commit + `tests/unit/test_no_secrets.py` (regex scan of `src/`, `configs/`, `tests/`) |
| INV-3 | A strategy can never observe bar `t+1` when deciding at bar `t`. | `tests/leakage/` + `Context` design (§9) |
| INV-4 | Untrusted (LLM-generated) code never executes in the main process. | `tests/unit/test_sandbox.py`, `tests/unit/test_architecture.py` (no `exec`/`eval`/`importlib` outside `sandbox/`) |
| INV-5 | The final test partition is never loaded by any code path other than `quantlab lockbox`. | `tests/unit/test_lockbox.py` |
| INV-6 | Nothing derived from the test partition is ever placed in an LLM prompt. | `tests/unit/test_redaction.py` |
| INV-7 | Every stored run is reproducible from its stored inputs to 1e-9 on all metrics. An evolution run replayed from its stored seed and configuration produces the identical sequence of `candidate_id`s. | `tests/integration/test_reproduce.py`, `tests/integration/test_evolution_replay.py` |
| INV-8 | Import boundaries: `core` imports nothing from the package except `core`; `ports` imports `core`; `adapters` import `core`+`ports`; `research/optimize/evolution/walkforward/paper/reporting` import `core`+`ports`; only `container.py`, `cli/`, and `tests/` import `adapters`. | `tests/unit/test_architecture.py` |
| INV-9 | The evolutionary optimiser evaluates candidates on **train only**. The validation segment is reachable only through the recorded promotion path (§13.7), and every promotion is persisted before the run executes. The test partition is never reachable at all. | `tests/unit/test_evolution_segments.py`, `PartitionGuard` (§7.4) |
| INV-10 | Lineage is complete and replayable: every candidate except a seed names a parent that exists, the parent chain reaches a seed without cycles, and re-applying a child's recorded mutations to its parent's genome reproduces the child's genome byte for byte. | `tests/unit/test_lineage.py` |
| INV-11 | While an evolution run is active, no quantity derived from the validation or test segment may enter an LLM prompt. Guided mutation sees training and inner-out-of-sample numbers only. | `tests/unit/test_redaction.py`, `tests/integration/test_information_barrier.py` |

### 0.3 Things you MUST NOT do

- MUST NOT add a dependency not listed in §4 without writing an ADR in `docs/DECISIONS/` explaining why.
- MUST NOT use `exec`, `eval`, `compile`, `importlib`, `pickle`, `marshal`, `shelve` outside `src/quantlab/sandbox/`.
- MUST NOT use `datetime.now()`, `time.time()`, `random` (unseeded), or `np.random` global state inside `core/`, `adapters/engine/`, or any strategy. Clocks and RNGs are injected.
- MUST NOT write to any location outside the repo, `data/`, `artifacts/`, `~/.quantlab/` (keychain fallback dir), or `/tmp`.
- MUST NOT change validation thresholds, split boundaries, or cost defaults to make a strategy look better. These live in config and changes to them require an ADR.
- MUST NOT "temporarily" bypass the sandbox, the leakage probe, or the lockbox for convenience.
- MUST NOT print or log secret values.
- MUST NOT optimise anything on the validation or test segments. In particular, no evolutionary fitness component, mutation choice, survival decision or stopping rule may read a validation- or test-segment result (INV-9).
- MUST NOT let a candidate reach the population without passing genome validation, the AST check and the leakage probe. A mutation that cannot produce a valid candidate is discarded, never repaired by hand.

### 0.4 Research philosophy that shapes code decisions

The platform's purpose is *statistically defensible research*, not impressive numbers. When two implementations differ, prefer the one that is (in order): more correct, more reproducible, more conservative (pessimistic fills, pessimistic costs), more auditable, simpler. Speed is last. A backtest that is 10× slower but provably free of lookahead is the right choice.

---

## 1. Project objective

### 1.1 Objective

Build a modular, reproducible Python platform that:

1. Imports and validates historical BTC/USDT OHLCV data.
2. Backtests Python strategies with realistic fees and configurable slippage.
3. Computes net return, CAGR, max drawdown, profit factor, Sharpe, Sortino, win rate, number of trades, average trade, expectancy, exposure (plus supporting extras).
4. Uses the GLM API (behind a swappable interface) as an automated researcher that proposes, receives feedback on, and iterates strategies.
5. Optimises parameters on training data only, validates with walk-forward analysis on validation data, and holds a locked final test partition.
6. Detects and rejects overfit or leaky strategies using statistical tests.
7. Stores every experiment, run, trade, LLM interaction and verdict.
8. Runs accepted strategies in paper-trading mode on live market data.

### 1.2 Non-goals (do not build)

Real-money execution of any kind; multi-asset portfolios (design for, do not build); margin, futures, funding, liquidation; order-book/tick simulation; GUI beyond a read-only Streamlit dashboard in the last phase; cloud deployment; Docker.

### 1.3 Fixed decisions

| Decision | Value | Rationale |
|---|---|---|
| Exchange / market | Binance spot, symbol `BTC/USDT` | Longest liquid crypto history; public bulk archive |
| Research timeframe | `1h` | Enough bars (~79k) for statistics; low enough cost sensitivity to be realistic |
| Secondary timeframe | `1m` (ingested in Phase I only, for paper-fill realism) | |
| Quote currency & capital | USDT, initial equity `10_000.0` | |
| Direction | Long/flat by default; `allow_short: false` | Spot cannot short; synthetic shorts must be opt-in and labelled |
| Fill rule | Decision at close of bar `t` → market fill at open of bar `t+1` | Removes same-bar lookahead |
| Fee | 10 bps per side (taker) | Binance spot default without discounts |
| Slippage default | `fixed_bps: 5` per side, adverse | |
| Lot step / min notional | `0.00001 BTC` / `5.0 USDT` | Binance spot filters |
| Annualisation | bars per year = `8760` for 1h, `525600` for 1m, `365` for 1d | Crypto trades 24/7 |
| Risk-free rate | `0.0` | |
| Numeric type | `float64` everywhere; quantities rounded down to lot step | |
| Timestamps | `int64` milliseconds UTC, bar **open** time, left-labelled | |
| Hashing | SHA-256, hex; ids are first 16 hex chars | |
| Canonical JSON | `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)` | Stable hashes |

---

## 2. Architecture

### 2.1 Layers

```
cli/ (Typer)  ──►  container.py (composition root)
                      │ builds
                      ▼
 orchestration: research/  optimize/  evolution/  walkforward/  paper/  reporting/
                      │ depend only on
                      ▼
 ports/  : BacktestEngine · LLMProvider · MarketDataSource · MarketDataFeed · Broker · ExperimentStore · Clock · Secrets
                      │ implemented by
                      ▼
 adapters/: engine/simple_bar · llm/{glm,mock} · data/{binance_archive,ccxt_rest,binance_ws} · broker/paper · store/{sqlite,artifacts} · secrets/{keychain,dotenv}
                      │ all of the above use
                      ▼
 core/   : types · config · strategy · genome · indicators · costs · metrics · splits ·
            fitness · validation/ · hashing · errors
 sandbox/: ast_check · runner (child-process execution of strategies)
```

### 2.2 Composition root

`src/quantlab/container.py` exposes exactly one public function:

```python
def build_container(config: AppConfig, *, profile: Literal["research", "lockbox", "paper", "test"]) -> Container
```

`Container` is a frozen dataclass holding one instance per port. The `profile` decides which adapters are wired (e.g. `test` → `MockLLMProvider`, in-memory SQLite; `research` → `GLMProvider`, but a `MarketDataSource` that refuses the test partition; `lockbox` → the only profile whose data source may load the test partition). Nothing else in the codebase instantiates adapters.

### 2.3 Replaceability contract

Each port is a `typing.Protocol` (structural) defined in `ports/`. Adapters MUST NOT be imported by name anywhere except `container.py`, `cli/`, and tests. To replace a component, add an adapter and one branch in `build_container`. A test (`test_architecture.py`) fails on any other import of `quantlab.adapters`.

---

## 3. Exact directory structure

Create exactly this tree. Files marked `(P<letter>)` are created in that phase; do not create them earlier.

```
quantlab/
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── Makefile
├── README.md
├── pyproject.toml
├── uv.lock
├── alembic.ini                                   (PE)
├── configs/
│   ├── default.yaml
│   ├── llm_pricing.yaml                          (PH)
│   ├── splits/
│   │   └── btcusdt_1h.yaml                       (PB)
│   └── research/
│       └── example_campaign.yaml                 (PH)
├── data/                       # git-ignored
│   └── .gitkeep
├── artifacts/                  # git-ignored
│   └── .gitkeep
├── docs/
│   ├── ARCHITECTURE.md
│   ├── METRICS.md                                (PC)
│   ├── VALIDATION.md                             (PG)
│   ├── SECURITY.md
│   └── DECISIONS/
│       └── 0001-record-architecture-decisions.md
├── scripts/
│   └── make_fixtures.py                          (PC)
├── src/quantlab/
│   ├── __init__.py             # __version__
│   ├── container.py
│   ├── cli/
│   │   ├── __init__.py         # Typer app root: `quantlab`
│   │   ├── doctor.py
│   │   ├── data.py                               (PB)
│   │   ├── backtest.py                           (PE)
│   │   ├── optimize.py                           (PF)
│   │   ├── evolve.py                             (PF)
│   │   ├── walkforward.py                        (PF)
│   │   ├── validate.py                           (PG)
│   │   ├── lockbox.py                            (PG)
│   │   ├── research.py                           (PH)
│   │   ├── paper.py                              (PI)
│   │   └── report.py                             (PE)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── errors.py
│   │   ├── hashing.py
│   │   ├── config.py
│   │   ├── logging.py
│   │   ├── types.py                              (PB)
│   │   ├── data_validation.py                    (PB)
│   │   ├── splits.py                             (PB)
│   │   ├── costs.py                              (PC)
│   │   ├── metrics.py                            (PC)
│   │   ├── strategy.py                           (PC)   # + RiskSpec, SizingSpec (§9.1)
│   │   ├── genome.py                             (PF)   # declarative strategy genome (§9.6)
│   │   ├── indicators.py                         (PC)
│   │   ├── fitness.py                            (PF)   # multi-objective fitness (§13.3)
│   │   └── validation/
│   │       ├── __init__.py
│   │       ├── leakage.py                        (PD)
│   │       ├── gates.py                          (PG)
│   │       ├── deflated_sharpe.py                (PG)
│   │       ├── pbo.py                            (PG)
│   │       ├── permutation.py                    (PG)
│   │       ├── sensitivity.py                    (PG)
│   │       ├── concentration.py                  (PG)   # trade-removal test (§14.4)
│   │       └── score.py                          (PG)
│   ├── ports/
│   │   ├── __init__.py
│   │   ├── clock.py                              (PB)
│   │   ├── data.py                               (PB)
│   │   ├── store.py                              (PB)
│   │   ├── engine.py                             (PC)
│   │   ├── secrets.py
│   │   ├── llm.py                                (PH)
│   │   └── broker.py                             (PI)
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── secrets/{__init__,keychain,dotenv}.py
│   │   ├── data/{__init__,binance_archive,ccxt_rest}.py   (PB)   binance_ws.py (PI)
│   │   ├── engine/{__init__,simple_bar}.py       (PC)
│   │   ├── store/{__init__,sqlite,artifacts,models}.py    (PE)
│   │   ├── llm/{__init__,glm,mock}.py            (PH)
│   │   └── broker/{__init__,paper}.py            (PI)
│   ├── sandbox/
│   │   ├── __init__.py                           (PD)
│   │   ├── ast_check.py                          (PD)
│   │   ├── runner.py                             (PD)
│   │   └── child_main.py                         (PD)   # entry point executed in the child process
│   ├── strategies_io/
│   │   ├── __init__.py                           (PD)
│   │   └── loader.py                             (PD)
│   ├── experiments/
│   │   ├── __init__.py                           (PE)
│   │   └── runner.py                             (PE)
│   ├── optimize/                # parameter refinement of a promoted candidate only
│   │   ├── __init__.py                           (PF)
│   │   ├── objectives.py                         (PF)
│   │   ├── study.py                              (PF)
│   │   └── plateau.py                            (PF)
│   ├── evolution/              # the primary search (§13)
│   │   ├── __init__.py                           (PF)
│   │   ├── compiler.py                           (PF)   # genome -> strategy module source
│   │   ├── mutation.py                           (PF)   # typed mutation operators (§13.4)
│   │   ├── diversity.py                          (PF)   # similarity + niching (§13.5)
│   │   ├── population.py                         (PF)   # ranking, survival, next generation
│   │   ├── lineage.py                            (PF)   # parent -> child reconstruction (§13.6)
│   │   └── loop.py                               (PF)   # the generation loop (§13.2)
│   ├── walkforward/
│   │   ├── __init__.py                           (PF)
│   │   └── runner.py                             (PF)
│   ├── research/
│   │   ├── __init__.py                           (PH)
│   │   ├── schemas.py                            (PH)
│   │   ├── redaction.py                          (PH)
│   │   ├── loop.py                               (PH)   # seeds + guided mutation, not a sequential loop
│   │   └── prompts/{system,propose_genome,repair,review_generation}.md   (PH)
│   ├── paper/
│   │   ├── __init__.py                           (PI)
│   │   ├── runtime.py                            (PI)
│   │   ├── state.py                              (PI)
│   │   └── launchd/com.quantlab.paper.plist.template     (PI)
│   ├── reporting/
│   │   ├── __init__.py                           (PE)
│   │   ├── markdown.py                           (PE)
│   │   └── tearsheet.py                          (PE)
│   └── dashboard/
│       └── app.py                                (PJ)
├── strategies/
│   ├── TEMPLATE.py                               (PC)
│   ├── baselines/{buy_and_hold,sma_cross,random_entry,rsi_reversion}.py   (PC)
│   └── generated/            # immutable LLM output; git-ignored except .gitkeep
├── migrations/                                   (PE)  # alembic versions
│                                                 # 0001 base schema, 0002 evolution tables
└── tests/
    ├── conftest.py
    ├── fixtures/
    │   ├── data/btcusdt_1h_2023-01_02.parquet    (PC)   # ~1400 real bars, committed
    │   ├── data/synthetic/                       (PC)
    │   ├── golden/                               (PC)
    │   ├── strategies/{honest,leaky,malicious}/  (PD)
    │   └── llm/                                  (PH)   # recorded GLM responses
    ├── unit/
    ├── property/
    ├── golden/
    ├── leakage/
    ├── synthetic/
    └── integration/
```

---

## 4. Python version and dependencies

- **Python 3.12.x** (pin in `pyproject.toml` as `requires-python = ">=3.12,<3.13"`). Use `uv` for environment and lockfile. macOS is the target; code MUST also run on Linux CI.
- All runtime dependencies pinned to exact versions in `uv.lock`; `pyproject.toml` uses compatible-release specifiers.

```toml
[project]
name = "quantlab"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
  "numpy~=2.1", "pandas~=2.2", "pyarrow~=17.0", "scipy~=1.14",
  "pydantic~=2.9", "pydantic-settings~=2.5", "pyyaml~=6.0",
  "sqlalchemy~=2.0", "alembic~=1.13",
  "typer~=0.12", "rich~=13.9", "structlog~=24.4",
  "httpx~=0.27", "tenacity~=9.0",
  "optuna~=4.0",
  "ccxt~=4.4",
  "keyring~=25.4", "python-dotenv~=1.0",
  "matplotlib~=3.9",
  "websockets~=13.1",
]
[project.optional-dependencies]
dev = ["pytest~=8.3", "pytest-cov~=5.0", "hypothesis~=6.115", "mypy~=1.11", "ruff~=0.6",
       "pre-commit~=4.0", "pip-audit~=2.7", "freezegun~=1.5", "respx~=0.21", "pandas-stubs", "types-PyYAML"]
dashboard = ["streamlit~=1.39"]
[project.scripts]
quantlab = "quantlab.cli:app"
```

`Makefile` targets: `sync` (`uv sync --all-extras`), `fmt`, `lint` (ruff), `type` (mypy: strict on `core/`, `ports/`, `sandbox/`; normal elsewhere), `test` (`pytest -q`), `check` (= fmt-check + lint + type + test), `audit` (`pip-audit`).

---

## 5. Configuration approach

- All tunables live in YAML under `configs/`, loaded into **pydantic v2 models** (`core/config.py`) with `extra="forbid"` so typos fail loudly. Environment variables MAY override scalar values via `QUANTLAB__<SECTION>__<KEY>` (pydantic-settings, double underscore delimiter). Secrets are **never** in YAML (see §21).
- `AppConfig.config_hash` = sha256 of the canonical JSON of the fully-resolved config (after env overrides). Every experiment stores the hash and the resolved JSON.
- The CLI accepts `--config path.yaml` (merged over `default.yaml`) and `--set section.key=value` overrides; every override is logged.

`configs/default.yaml` (authoritative keys — implement exactly these):

```yaml
project:
  name: quantlab
  data_dir: ./data
  artifacts_dir: ./artifacts
  db_path: ./quantlab.db
  timezone: UTC                   # informational; all internals are UTC ms

market:
  exchange: binance
  symbol: BTC/USDT
  timeframe: 1h
  lot_step: 0.00001
  min_notional: 5.0

backtest:
  initial_equity: 10000.0
  fill_rule: next_open            # only value in v1; next_close reserved for stress tests
  allow_short: false
  short_borrow_bps_per_bar: 0.0
  fee_bps: 10.0
  slippage:
    model: fixed_bps              # fixed_bps | volatility_scaled | volume_impact
    fixed_bps: 5.0
    vol_k: 0.10                   # bps = vol_k * ATR14/close * 1e4
    impact_a_bps: 2.0
    impact_b: 50.0                # bps += impact_b * notional / bar_quote_volume
  cost_stress_multipliers: [1.0, 2.0, 3.0]
  max_position_fraction: 1.0

splits:
  policy_file: configs/splits/btcusdt_1h.yaml

optimize:                         # parameter refinement of a promoted candidate (§13.8)
  engine: evolution               # evolution | optuna   (which search produces candidates)
  sampler: tpe                    # tpe | random | grid
  n_trials: 200
  timeout_s: 1800
  seed: 42
  objective: sortino_dd           # sortino_dd | sharpe | calmar | expectancy  (net_return is FORBIDDEN)
  min_trades: 30
  dd_lambda: 0.5
  plateau:
    top_k: 10
    perturbation_pcts: [0.10, 0.25]
    n_neighbors: 12

evolution:                        # the primary search (§13)
  enabled: true
  population_size: 16
  n_survivors: 12                 # elite carried into the next generation unchanged
  n_offspring: 3                  # children produced from survivors by mutation
  n_immigrants: 1                 # novel candidates, unrelated to any survivor
  # INVARIANT: n_survivors + n_offspring + n_immigrants == population_size
  max_generations: 30
  seed: 42
  max_wall_clock_s: 21600
  max_evaluations: 1000           # counted into DSR's M (§14.4)
  stop_on_no_improvement_generations: 8
  min_improvement: 0.005          # fitness delta that counts as improvement

  mutation:
    parameter_rate: 0.7           # P(child receives at least one parameter mutation)
    structural_rate: 0.3          # P(child receives at least one structural mutation)
    max_mutations_per_child: 3
    max_repair_attempts: 3        # invalid child -> redraw; then fall back to an immigrant
    parameter:
      perturb_pct: 0.25           # x *= 1 +/- U(0, perturb_pct)
      jump_probability: 0.15      # otherwise resample uniformly from the ParamSpec range
      risk_perturb_pct: 0.35      # stop-loss / take-profit / trailing / sizing move further
    structural:                   # relative weights, normalised at load;
      add_confirmation: 0.22      # one key per operator in the §13.4 table
      remove_confirmation: 0.18
      modify_entry: 0.18
      modify_exit: 0.13
      add_filter: 0.09
      remove_filter: 0.09
      replace_indicator: 0.07
      change_tree_mode: 0.04

  diversity:
    max_pairwise_similarity: 0.90 # a survivor too similar to a fitter one is skipped
    min_population_diversity: 0.35
    structural_weight: 0.4
    behavioural_weight: 0.6
    immigrant_boost: 2            # extra immigrants while diversity is below the floor
    max_immigrants: 4             # INVARIANT: <= n_offspring + n_immigrants, because a
                                  # boost displaces offspring and never a survivor

  fitness:                        # §13.3; weights MUST sum to 1.0
    weights:
      expectancy: 0.20
      profit_factor: 0.12
      risk_adjusted: 0.15
      drawdown: 0.15
      consistency: 0.12
      inner_oos: 0.12
      trades: 0.06
      win_rate: 0.04
      net_return: 0.02
      concentration: 0.02
    targets:
      expectancy_pct: 0.002
      profit_factor: 1.5
      sortino: 1.5
      cagr: 0.20
      trades: 200
      win_rate: 0.55
      win_rate_floor: 0.35
      drawdown_ceiling: 0.35
      consistency_period_bars: 720
    gates:                        # failing any gate sets fitness to FITNESS_REJECTED
      min_expectancy_pct: 0.0
      min_trades: 30
      max_drawdown: 0.50
      min_retention_top1: 0.0     # must still be profitable without its single best trade
    penalties:
      drawdown_soft: 0.20
      trades_soft: 100
      instability_max_cv: 0.60
      sensitivity_max_drop: 0.50
      divergence_min_ratio: 0.50
      complexity_free_params: 6
      complexity_logic_lines: 150
      removal_k: [1, 3, 5]        # §14.4 trade-removal test
      removal_floor: [0.40, 0.20, 0.10]
      removal_target: [0.80, 0.65, 0.55]

  inner_walkforward:              # out-of-sample *inside train*; never touches val
    n_folds: 4
    scheme: rolling
    embargo_bars: 24

  promotion:                      # the only route from evolution to the validation segment
    every_generations: 0          # 0 = only after the final generation
    n_promote: 3
    min_fitness: 0.35
    max_similarity_between_promoted: 0.80
    counts_as_validation_touch: true

  genome:                         # structural limits (§9.6)
    max_conditions_entry: 4
    max_conditions_exit: 4
    max_filters: 3
    max_indicators: 6
    max_free_params: 6

walkforward:
  scheme: rolling                 # rolling | anchored
  is_bars: 13140                  # 18 months of 1h
  oos_bars: 2190                  # 3 months
  step_bars: 2190
  reoptimize_each_window: true
  evolution_generations: 8        # generations per IS window when engine == evolution (§15)

validation:
  min_trades_val: 30
  min_trades_train: 100
  max_single_trade_pct: 0.25
  cost_survival_multiplier: 2.0
  permutation:
    n_market_permutations: 200
    n_trade_shuffles: 1000
    block_len_bars: 168
    alpha: 0.05
  degradation_min_ratio: 0.5
  dsr_threshold: 0.95
  pbo:
    n_blocks: 16
    max_combinations: 500
  sensitivity_max_drop: 0.5
  concentration_top_n: 5
  concentration_max_share: 0.5
  max_free_params: 6
  max_logic_lines: 150
  random_entry_seeds: 100
  score:
    candidate_max: 30
    weak_max: 60
  family_max_validation_touches: 20

lockbox:
  max_per_family: 1
  max_per_month: 3

llm:
  provider: glm                   # glm | mock
  model: glm-4-plus
  base_url: https://open.bigmodel.cn/api/paas/v4
  temperature_propose: 0.7
  temperature_repair: 0.2
  max_tokens: 8000
  timeout_s: 60
  max_retries: 3
  daily_cost_cap_eur: 10.0
  pricing_file: configs/llm_pricing.yaml

research:                         # LLM contribution to evolution (§12)
  max_seed_genomes: 8             # LLM-proposed genomes used to fill generation 0
  guided_mutation_share: 0.25     # share of offspring whose operator the LLM may suggest
  max_cost_eur: 20.0
  max_wall_clock_s: 14400
  max_repair_attempts: 2
  smoke_bars: 200

sandbox:
  cpu_seconds: 120
  memory_mb: 2048
  wall_clock_s: 300
  allowed_imports: [numpy, pandas, math, statistics, dataclasses, typing, quantlab.core.strategy, quantlab.core.indicators, quantlab.core.types]

paper:
  state_dir: ./artifacts/paper
  ws_url: wss://stream.binance.com:9443/ws
  backfill_limit: 1000
  alert_mdd_percentile: 95

logging:
  level: INFO
  json: true
  redact_keys: [api_key, secret, token, password, authorization]
```

`configs/splits/btcusdt_1h.yaml`:

```yaml
split_id_note: "changing anything here creates a new split_id; old verdicts stay bound to the old id"
symbol: BTC/USDT
timeframe: 1h
train:      { start: "2017-08-17T00:00:00Z", end: "2022-12-31T23:00:00Z" }
embargo_bars: 720          # 30 days
validation: { start: "2023-01-31T00:00:00Z", end: "2024-12-31T23:00:00Z" }
test:       { start: "2025-01-31T00:00:00Z", end: null }   # null = latest available at lockbox time; frozen at first lockbox use
```

**Decision:** `test.end = null` is resolved and frozen (written back to the DB as `SplitPolicy.test_end_ts`) the first time the lockbox is used, so all lockbox evaluations for a split share the same period.

---

## 6. Database schema (SQLite, SQLAlchemy 2.0 models + Alembic)

Conventions: `TEXT` ids are hex hashes or UUID4; all timestamps `INTEGER` ms UTC; JSON columns are `TEXT` containing canonical JSON; `STRICT` tables; foreign keys ON; WAL mode.

```sql
CREATE TABLE dataset (
  dataset_id     TEXT PRIMARY KEY,           -- sha256(manifest entries)[:16]
  exchange       TEXT NOT NULL, symbol TEXT NOT NULL, timeframe TEXT NOT NULL,
  start_ts INTEGER NOT NULL, end_ts INTEGER NOT NULL, n_bars INTEGER NOT NULL,
  manifest_json  TEXT NOT NULL, created_at INTEGER NOT NULL
) STRICT;

CREATE TABLE split_policy (
  split_id       TEXT PRIMARY KEY,           -- sha256(canonical yaml + dataset_id)[:16]
  dataset_id     TEXT NOT NULL REFERENCES dataset,
  train_start_ts INTEGER NOT NULL, train_end_ts INTEGER NOT NULL,
  val_start_ts   INTEGER NOT NULL, val_end_ts INTEGER NOT NULL,
  test_start_ts  INTEGER NOT NULL, test_end_ts INTEGER,          -- NULL until frozen
  embargo_bars   INTEGER NOT NULL, wf_json TEXT NOT NULL, created_at INTEGER NOT NULL
                 -- wf_json: the canonical policy document this split was built
                 -- from; split_id is its hash, so the row can reproduce the
                 -- policy and still hash to its own key (clarified in 1.1.6)
) STRICT;

CREATE TABLE strategy_family (
  family_id   TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE,
  origin      TEXT NOT NULL CHECK (origin IN ('human','llm')),
  description TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'open'
              CHECK (status IN ('open','frozen','closed')),
  validation_touches INTEGER NOT NULL DEFAULT 0, created_at INTEGER NOT NULL
) STRICT;

CREATE TABLE strategy_version (
  strategy_id        TEXT PRIMARY KEY,       -- sha256(code bytes)[:16]
  family_id          TEXT NOT NULL REFERENCES strategy_family,
  parent_strategy_id TEXT REFERENCES strategy_version,
  code_path          TEXT NOT NULL, code_sha256 TEXT NOT NULL,
  class_name         TEXT NOT NULL, param_schema_json TEXT NOT NULL,
  style              TEXT NOT NULL CHECK (style IN ('bar_loop','vectorized')),
  author             TEXT NOT NULL,          -- 'human' | 'llm:<provider>:<model>'
  llm_interaction_id TEXT REFERENCES llm_interaction,
  logic_lines        INTEGER NOT NULL, created_at INTEGER NOT NULL
) STRICT;

CREATE TABLE experiment (
  experiment_id TEXT PRIMARY KEY, campaign TEXT NOT NULL,
  purpose TEXT NOT NULL CHECK (purpose IN ('smoke','train','optimize','validate','walkforward','lockbox','paper','baseline')),
  config_hash TEXT NOT NULL, config_json TEXT NOT NULL, seed INTEGER NOT NULL,
  git_commit TEXT NOT NULL, created_at INTEGER NOT NULL
) STRICT;

CREATE TABLE run (
  run_id        TEXT PRIMARY KEY,            -- see §11.2
  experiment_id TEXT NOT NULL REFERENCES experiment,
  strategy_id   TEXT NOT NULL REFERENCES strategy_version,
  dataset_id    TEXT NOT NULL REFERENCES dataset,
  split_id      TEXT NOT NULL REFERENCES split_policy,
  segment       TEXT NOT NULL,               -- 'train' | 'val' | 'test' | 'wf_is:<k>' | 'wf_oos:<k>' | 'custom:<start>-<end>'
  params_json   TEXT NOT NULL, engine_name TEXT NOT NULL, engine_version TEXT NOT NULL,
  cost_multiplier REAL NOT NULL DEFAULT 1.0,
  status        TEXT NOT NULL CHECK (status IN ('pending','running','ok','failed','rejected')),
  error_json    TEXT, artifact_dir TEXT NOT NULL,
  started_at INTEGER, finished_at INTEGER, created_at INTEGER NOT NULL
) STRICT;
CREATE INDEX ix_run_strategy ON run(strategy_id); CREATE INDEX ix_run_experiment ON run(experiment_id);

CREATE TABLE metric (
  run_id TEXT NOT NULL REFERENCES run, name TEXT NOT NULL, value REAL,      -- NULL for undefined (e.g. PF with no losers)
  PRIMARY KEY (run_id, name)
) STRICT WITHOUT ROWID;

CREATE TABLE trade (
  run_id TEXT NOT NULL REFERENCES run, trade_no INTEGER NOT NULL,
  side TEXT NOT NULL CHECK (side IN ('long','short')),
  entry_ts INTEGER NOT NULL, entry_px REAL NOT NULL, exit_ts INTEGER NOT NULL, exit_px REAL NOT NULL,
  qty REAL NOT NULL, fees REAL NOT NULL, slippage_cost REAL NOT NULL,
  pnl REAL NOT NULL, pnl_pct REAL NOT NULL, bars_held INTEGER NOT NULL,
  exit_reason TEXT NOT NULL CHECK (exit_reason IN ('signal','end_of_data','stop')),
  PRIMARY KEY (run_id, trade_no)
) STRICT WITHOUT ROWID;

CREATE TABLE optuna_study (
  study_id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL REFERENCES experiment,
  strategy_id TEXT NOT NULL REFERENCES strategy_version, segment TEXT NOT NULL,
  sampler TEXT NOT NULL, seed INTEGER NOT NULL, n_trials INTEGER NOT NULL,
  objective TEXT NOT NULL, best_trial_json TEXT NOT NULL, plateau_json TEXT NOT NULL,
  storage_path TEXT NOT NULL, created_at INTEGER NOT NULL
) STRICT;

CREATE TABLE validation_verdict (
  verdict_id TEXT PRIMARY KEY, strategy_id TEXT NOT NULL REFERENCES strategy_version,
  split_id TEXT NOT NULL REFERENCES split_policy, params_json TEXT NOT NULL,
  verdict TEXT NOT NULL CHECK (verdict IN ('REJECT','WEAK','CANDIDATE','LOCKBOX_PASS','LOCKBOX_FAIL')),
  overfit_score REAL NOT NULL, hard_gates_json TEXT NOT NULL, soft_checks_json TEXT NOT NULL,
  thresholds_json TEXT NOT NULL, n_trials_accounted INTEGER NOT NULL, created_at INTEGER NOT NULL
) STRICT;

CREATE TABLE llm_interaction (
  interaction_id TEXT PRIMARY KEY, campaign TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL,
  purpose TEXT NOT NULL CHECK (purpose IN ('propose','repair','review')),
  prompt_sha256 TEXT NOT NULL, prompt_path TEXT NOT NULL, response_path TEXT NOT NULL,
  prompt_template_sha256 TEXT NOT NULL, tokens_in INTEGER NOT NULL, tokens_out INTEGER NOT NULL,
  cost_eur REAL NOT NULL, temperature REAL NOT NULL, seed INTEGER, latency_ms INTEGER NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('ok','malformed','error')), created_at INTEGER NOT NULL
) STRICT;

CREATE TABLE lockbox_access (
  access_id TEXT PRIMARY KEY, strategy_id TEXT NOT NULL REFERENCES strategy_version,
  family_id TEXT NOT NULL REFERENCES strategy_family, run_id TEXT REFERENCES run,
  os_user TEXT NOT NULL, reason TEXT NOT NULL, created_at INTEGER NOT NULL
) STRICT;

CREATE TABLE evolution_run (
  evolution_id   TEXT PRIMARY KEY,           -- sha256(experiment_id|config|seed)[:16]
  experiment_id  TEXT NOT NULL REFERENCES experiment,
  campaign       TEXT NOT NULL,
  dataset_id     TEXT NOT NULL REFERENCES dataset,
  split_id       TEXT NOT NULL REFERENCES split_policy,
  population_size INTEGER NOT NULL, n_survivors INTEGER NOT NULL,
  n_offspring INTEGER NOT NULL, n_immigrants INTEGER NOT NULL,
  max_generations INTEGER NOT NULL, seed INTEGER NOT NULL,
  fitness_config_json TEXT NOT NULL, mutation_config_json TEXT NOT NULL,
  diversity_config_json TEXT NOT NULL,
  n_evaluations  INTEGER NOT NULL DEFAULT 0,  -- feeds DSR's M (§14.4)
  status         TEXT NOT NULL CHECK (status IN ('running','completed','stopped','failed')),
  stop_reason    TEXT,
  started_at INTEGER NOT NULL, finished_at INTEGER, created_at INTEGER NOT NULL
) STRICT;

CREATE TABLE generation (
  generation_id  TEXT PRIMARY KEY,
  evolution_id   TEXT NOT NULL REFERENCES evolution_run,
  gen_index      INTEGER NOT NULL,
  best_fitness REAL, median_fitness REAL, mean_fitness REAL,
  diversity      REAL NOT NULL,             -- 1 - mean pairwise similarity (§13.5)
  n_evaluated INTEGER NOT NULL, n_cache_hits INTEGER NOT NULL,
  n_rejected_by_gate INTEGER NOT NULL, n_immigrants_used INTEGER NOT NULL,
  stats_json     TEXT NOT NULL, created_at INTEGER NOT NULL,
  UNIQUE (evolution_id, gen_index)
) STRICT;

CREATE TABLE candidate (
  candidate_id   TEXT PRIMARY KEY,           -- sha256(evolution_id|gen_index|strategy_id|params)[:16]
  evolution_id   TEXT NOT NULL REFERENCES evolution_run,
  generation_id  TEXT NOT NULL REFERENCES generation,
  gen_index      INTEGER NOT NULL,
  strategy_id    TEXT NOT NULL REFERENCES strategy_version,
  params_json    TEXT NOT NULL,
  genome_json    TEXT,                       -- NULL for 'opaque' candidates
  kind           TEXT NOT NULL CHECK (kind IN ('genome','opaque')),
  origin         TEXT NOT NULL CHECK (origin IN ('seed','survivor','mutant','immigrant','refined')),
  parent_candidate_id TEXT REFERENCES candidate,
  run_id         TEXT REFERENCES run,        -- the train-segment evaluation
  fitness REAL, base_score REAL, penalty_product REAL,
  components_json TEXT NOT NULL DEFAULT '{}',  -- every fitness component, for audit
  penalties_json  TEXT NOT NULL DEFAULT '{}',
  gate_failure   TEXT,                       -- NULL, else the id of the gate that rejected it
  rank           INTEGER,
  survived       INTEGER NOT NULL DEFAULT 0 CHECK (survived IN (0,1)),
  behaviour_hash TEXT,                       -- digest of the position_frac series (§13.5)
  signature_json TEXT NOT NULL DEFAULT '{}', -- structural fingerprint (§13.5)
  created_at     INTEGER NOT NULL
) STRICT;
CREATE INDEX ix_candidate_evolution ON candidate(evolution_id, gen_index);
CREATE INDEX ix_candidate_parent ON candidate(parent_candidate_id);
CREATE INDEX ix_candidate_strategy ON candidate(strategy_id);

CREATE TABLE mutation (
  mutation_id    TEXT PRIMARY KEY,
  candidate_id   TEXT NOT NULL REFERENCES candidate,
  parent_candidate_id TEXT NOT NULL REFERENCES candidate,
  seq            INTEGER NOT NULL,           -- order of application within the child
  category       TEXT NOT NULL CHECK (category IN ('parameter','structural')),
  operator       TEXT NOT NULL,              -- e.g. 'perturb_numeric', 'add_confirmation'
  target         TEXT NOT NULL,              -- genome path, e.g. 'entry.conditions[1].right'
  before_json    TEXT NOT NULL, after_json TEXT NOT NULL,
  rng_seed       INTEGER NOT NULL,           -- makes the mutation replayable (INV-10)
  suggested_by   TEXT NOT NULL DEFAULT 'rng' CHECK (suggested_by IN ('rng','llm')),
  llm_interaction_id TEXT REFERENCES llm_interaction,
  created_at     INTEGER NOT NULL,
  UNIQUE (candidate_id, seq)
) STRICT;
CREATE INDEX ix_mutation_candidate ON mutation(candidate_id);

CREATE TABLE candidate_promotion (               -- the ONLY route to the validation segment
  promotion_id   TEXT PRIMARY KEY,
  candidate_id   TEXT NOT NULL REFERENCES candidate,
  evolution_id   TEXT NOT NULL REFERENCES evolution_run,
  gen_index      INTEGER NOT NULL,
  segment        TEXT NOT NULL,              -- 'val' | 'wf_oos:<k>'
  run_id         TEXT REFERENCES run,
  verdict_id     TEXT REFERENCES validation_verdict,
  reason         TEXT NOT NULL,
  created_at     INTEGER NOT NULL            -- written BEFORE the run executes
) STRICT;
CREATE INDEX ix_promotion_candidate ON candidate_promotion(candidate_id);

CREATE TABLE paper_session (
  session_id TEXT PRIMARY KEY, strategy_id TEXT NOT NULL REFERENCES strategy_version,
  params_json TEXT NOT NULL, source_run_id TEXT REFERENCES run,
  started_at INTEGER NOT NULL, stopped_at INTEGER, last_bar_ts INTEGER,
  state_path TEXT NOT NULL, status TEXT NOT NULL CHECK (status IN ('running','stopped','crashed'))
) STRICT;
```

Rules: rows are append-only except `run.status/error_json/started_at/finished_at`, `strategy_family.status/validation_touches`, `split_policy.test_end_ts`, `paper_session.*_at/status/last_bar_ts`, `evolution_run.status/stop_reason/n_evaluations/finished_at`, `candidate.run_id/fitness/base_score/penalty_product/components_json/penalties_json/gate_failure/rank/survived/behaviour_hash`, and `candidate_promotion.run_id/verdict_id`. A `mutation` row is **never** mutable: it is the record INV-10 replays. A `delete_run(run_id, confirm: Literal[True])` exists only for failed runs and logs at WARNING.

---

## 7. Data ingestion

### 7.1 Sources

1. **Bulk history:** `https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-YYYY-MM.zip` (+ `.CHECKSUM`). Verify the SHA-256 checksum file; abort on mismatch. Use `daily/` for the current partial month.
2. **Tail updates:** `ccxt.binance().fetch_ohlcv(symbol, timeframe, since, limit=1000)` paginated, public endpoint. Optional read-only API key only for rate limits.

### 7.2 Canonical bar schema (Parquet, pyarrow)

| column | arrow type | constraint |
|---|---|---|
| `ts_open` | int64 | strictly increasing, spacing == timeframe ms |
| `open` `high` `low` `close` | float64 | `low <= min(open,close)`, `max(open,close) <= high`, all > 0 |
| `volume` `quote_volume` | float64 | ≥ 0 |
| `trades` | int64 | ≥ 0 |
| `is_gap_filled` | bool | |

File layout: `data/binance/BTCUSDT/1h/year=YYYY/bars.parquet`. `data/binance/BTCUSDT/1h/manifest.json`:

```json
{"exchange":"binance","symbol":"BTC/USDT","timeframe":"1h","files":[{"path":"year=2017/bars.parquet","sha256":"…","n_bars":3288,"first_ts":…,"last_ts":…}],
 "dataset_id":"…","built_at":"2026-09-05T10:00:00Z","source_versions":{"archive_checksums":[…]}}
```

`dataset_id = sha256("|".join(f"{path}:{sha256}" for files sorted by path) + "|" + symbol + "|" + timeframe)[:16]`.

### 7.3 Validation (fail closed)

`core/data_validation.validate_bars(df, timeframe) -> ValidationReport` raising `DataValidationError` on: non-monotonic `ts_open`; duplicate timestamps; OHLC inconsistency; negative volume; NaN anywhere. Gaps: runs of ≤ 3 missing bars are filled with `open=high=low=close=previous close`, `volume=0`, `is_gap_filled=True`; longer gaps raise unless `--allow-gaps`, in which case they are filled the same way and listed in the report. Strategies receive `is_gap_filled` and the engine never fills orders on gap-filled bars (order is deferred to the next real bar).

### 7.4 Port

```python
class MarketDataSource(Protocol):
    def load(self, symbol: str, timeframe: str, start_ts: int, end_ts: int) -> BarFrame: ...
    def dataset_id(self, symbol: str, timeframe: str) -> str: ...
```

`BarFrame` is a thin immutable wrapper around a `pandas.DataFrame` with the schema above, plus `dataset_id`, `symbol`, `timeframe`, `bar_ms`. The research profile wraps the source in `PartitionGuard(source, split_policy)` which raises `LockboxViolation` if any requested range intersects `[test_start_ts, ∞)`. Only the `lockbox` profile omits the guard (INV-5).

### 7.5 CLI

`quantlab data pull --symbol BTC/USDT --tf 1h --from 2017-08 [--to YYYY-MM]` · `quantlab data update` · `quantlab data validate` · `quantlab data info` (prints manifest and `dataset_id`).

---

## 8. Backtesting API

### 8.1 Types (`core/types.py`, all `@dataclass(frozen=True, slots=True)`)

```python
class Side(StrEnum): LONG = "long"; SHORT = "short"
class SignalKind(StrEnum): LONG = "long"; FLAT = "flat"; SHORT = "short"

@dataclass(frozen=True, slots=True)
class Signal:
    kind: SignalKind
    target_fraction: float = 1.0        # of equity, in [0, max_position_fraction]; ignored for FLAT
    tag: str = ""                       # free text for analysis, ≤ 32 chars

@dataclass(frozen=True, slots=True)
class Order:      # market order to be filled at next real bar's open; there is no send()
    bar_index: int; side: Side; target_qty: float; created_ts: int

@dataclass(frozen=True, slots=True)
class Fill:
    bar_index: int; ts: int; side: Side; qty: float; ref_price: float; fill_price: float
    fee: float; slippage_cost: float

@dataclass(frozen=True, slots=True)
class Trade:      # round trip
    trade_no: int; side: Side; entry_ts: int; entry_px: float; exit_ts: int; exit_px: float
    qty: float; fees: float; slippage_cost: float; pnl: float; pnl_pct: float
    bars_held: int; exit_reason: Literal["signal", "end_of_data", "stop"]

@dataclass(frozen=True, slots=True)
class Position:
    side: Side | None; qty: float; entry_px: float; entry_bar: int
```

### 8.2 Config and result

```python
class BacktestConfig(BaseModel, frozen=True):
    initial_equity: float; fee_bps: float; slippage: SlippageConfig; cost_multiplier: float = 1.0
    fill_rule: Literal["next_open"] = "next_open"; allow_short: bool = False
    short_borrow_bps_per_bar: float = 0.0; lot_step: float; min_notional: float
    max_position_fraction: float = 1.0; bars_per_year: int; seed: int = 0
    risk: RiskSpec = RiskSpec()          # engine-applied exits (§8.6); mutable by evolution

class BacktestResult(BaseModel, frozen=True):
    equity: pd.Series          # index ts_open (int64), float64 equity after costs, one point per bar incl. warm-up
    position_frac: pd.Series   # signed fraction of equity in market per bar
    signals: pd.Series         # raw SignalKind per bar as emitted (for leakage audits)
    fills: list[Fill]; trades: list[Trade]
    warmup_bars: int; n_bars: int; bars_per_year: int
    engine_name: str; engine_version: str
    cost_summary: dict[str, float]    # total_fees, total_slippage, turnover
    log: list[str]             # human-readable engine events (bounded to 10k lines)
```

### 8.3 Port

```python
class BacktestEngine(Protocol):
    name: str; version: str
    def run(self, strategy: Strategy, bars: BarFrame, params: Mapping[str, Any], config: BacktestConfig) -> BacktestResult: ...
```

### 8.4 `SimpleBarEngine` normative behaviour

For `i` in `range(n_bars)`:

1. **Fill pending order** (created at `i-1`) at `open[i]` if `is_gap_filled[i]` is False; otherwise keep it pending. Fill price = `open[i] * (1 + slip)` for buys, `open[i] * (1 - slip)` for sells, `slip = slippage_bps(i) * cost_multiplier / 1e4`. Fee = `fill_price * qty * fee_bps * cost_multiplier / 1e4`, deducted from cash. Quantity is `floor(target_notional / fill_price / lot_step) * lot_step`; if `qty * fill_price < min_notional` the order is dropped and logged with reason `below_min_notional`.
2. **Mark to market** using `close[i]`: `equity[i] = cash + qty * close[i]` (short: `cash - qty * close[i]` minus accumulated borrow).
3. If `i < warmup_bars`: record `signals[i] = FLAT`, continue.
4. **Ask strategy:** `sig = strategy.on_bar(ctx)` where `ctx` exposes bars `[0..i]` inclusive only. Any exception → `StrategyRuntimeError` (run status `failed`).
5. **Translate signal to order:** compute desired signed target fraction `f` (`LONG → +target_fraction`, `SHORT → −target_fraction` if `allow_short` else FLAT with a logged warning, `FLAT → 0`), clipped to `±max_position_fraction`. If the sign changes or `|f_desired − f_current| > 0.01`, create one order sized to the difference, to be filled at step 1 of bar `i+1`. **Reversal = one sell-to-flat fill then one buy at the same open** (two fills, two fee charges).
6. **Apply engine-side risk exits** (§8.6) before step 3 of the next bar. A risk exit produces a fill on the *current* bar and sets `exit_reason="stop"`.
7. At the last bar, an open position is closed at `close[n-1]` with `exit_reason="end_of_data"` and full costs applied.

Trade extraction: a trade opens on a fill that moves the position from flat, and closes on the fill that returns it to flat. `pnl` is net of the entry and exit fees and slippage; `pnl_pct = pnl / equity_at_entry_bar`. Scaling in/out within one position is allowed by the engine but recorded as a single trade with volume-weighted entry/exit prices.

Determinism: no randomness in the engine. `engine_version` MUST be bumped whenever any fill/accounting rule changes; the golden tests will fail otherwise, which is intended.

### 8.5 Cost models (`core/costs.py`)

```python
class SlippageModel(Protocol):
    def bps(self, bars: BarFrame, i: int, notional: float) -> float
FixedBps(bps) · VolatilityScaled(k)  → k * ATR14[i-1]/close[i-1] * 1e4   (uses i-1: known at decision time)
VolumeImpact(a_bps, b) → a_bps + b * notional / quote_volume[i-1]
```
All models MUST return ≥ 0 (property test). The engine always applies slippage against the trader.

---

### 8.6 Engine-side risk exits (stop-loss, take-profit, trailing stop)

Risk controls are **engine behaviour**, not strategy behaviour. A strategy that
implemented its own stop would have to inspect intrabar prices, which the
`BarWindow` contract forbids; and evolution needs a fixed, typed surface to
mutate (§13.4). They are declared per candidate in `RiskSpec` (§9.1) and applied
by the engine.

For an open long position at bar `i`, using only bar `i`'s own OHLC:

| control | trigger | fill price |
|---|---|---|
| stop-loss | `low[i] <= stop_px` | `min(stop_px, open[i])` |
| take-profit | `high[i] >= target_px` | `max(target_px, open[i])` |
| trailing stop | `low[i] <= trail_px`, where `trail_px = running_max_close * (1 - trailing_stop_pct)` and the running max is over closes up to and including `i-1` | `min(trail_px, open[i])` |
| time stop | `i - entry_bar >= time_stop_bars` | `close[i]` |

Normative rules, all of which exist to keep the simulation pessimistic and free
of intrabar assumptions:

1. **No intrabar path is assumed.** If a stop-loss and a take-profit could both
   trigger on the same bar, the **stop-loss wins**. A backtest that resolved the
   ambiguity in the trader's favour would be systematically optimistic, and
   evolution would learn to exploit exactly that.
2. Fill prices are the **worse** of the trigger price and the bar open, so a bar
   that gaps through a stop fills at the gap, not at the stop.
3. The trailing reference uses closes up to `i-1` only, never `high[i]` (INV-3).
4. No risk exit fires on a bar with `is_gap_filled = True`; it is deferred to the
   next real bar, exactly as an order would be (§7.3).
5. Slippage and fees apply in full, at the same rates as a signal exit.
6. `exit_reason = "stop"` regardless of which control fired; the specific control
   is recorded in the engine log and in `Fill`.
7. Short positions mirror all of the above.

Changing any rule here is an `engine_version` bump and fails the golden tests,
which is intended.

### 8.7 Ruin

An account whose marked equity reaches zero is gone. The engine closes any open
position at that bar's close with full costs, sets `BacktestResult.ruined`, and
takes no further position for the rest of the run.

Without this rule the engine keeps trading a negative balance, and a search
process will eventually discover that negative-equity arithmetic can be made to
look profitable — a strategy that "recovers" from -5 000 to -1 000 shows a large
positive return over that stretch.

Liquidation happens at the **close**, never intrabar: identifying the moment
equity crossed zero inside a bar would need the intrabar path that §8.6 refuses
to assume. A bar that gaps far enough can therefore leave equity **below** zero,
which is realistic — a short squeeze can leave a real account owing money — and
is reported rather than hidden. `max_drawdown` is still a fraction in `[0, 1]`
(§10): a total loss is 100 %, and the debt is visible in the equity series and in
`ruined`.

Any validation gate MUST treat `ruined = True` as a rejection regardless of the
other metrics.

## 9. Strategy API

### 9.1 Contract (`core/strategy.py`)

```python
class ParamSpec(BaseModel, frozen=True):
    kind: Literal["int", "float", "categorical", "bool"]
    default: int | float | str | bool
    low: float | None = None; high: float | None = None; step: float | None = None
    choices: list[str] | None = None; log: bool = False
    # validators: int/float require low<high and default within; categorical requires choices; unbounded ⇒ ValueError

class Context:                               # constructed by the engine; NOT subclassable by strategies
    i: int                                   # current bar index (0-based, bar is CLOSED)
    bars: BarWindow                          # read-only view of bars[0..i]; BarWindow raises on any index > i
    position: Position; equity: float; cash: float
    params: Mapping[str, Any]
    def ind(self, name: str, **kwargs) -> float | np.ndarray   # causal indicator, cached; see §9.3
    rng: np.random.Generator                 # seeded from (run seed, i); the ONLY allowed randomness

class RiskSpec(BaseModel, frozen=True):      # applied by the ENGINE (§8.6), never by the strategy
    stop_loss_pct: float | None = None       # > 0, fraction of entry price
    take_profit_pct: float | None = None     # > 0
    trailing_stop_pct: float | None = None   # > 0
    time_stop_bars: int | None = None        # >= 1
    # validator: take_profit_pct > stop_loss_pct when both are set

class SizingSpec(BaseModel, frozen=True):
    mode: Literal["fixed_fraction", "volatility_target", "atr_risk"] = "fixed_fraction"
    fraction: float = 1.0                    # of equity, clipped to max_position_fraction
    target_vol_annual: float | None = None   # volatility_target mode
    atr_period: int = 14                     # atr_risk mode
    atr_risk_pct: float | None = None        # fraction of equity risked to the stop

class Strategy(Protocol):
    name: str; version: str; style: Literal["bar_loop", "vectorized"]
    params: ClassVar[dict[str, ParamSpec]]
    warmup_bars: int
    risk: ClassVar[RiskSpec]                 # default RiskSpec(); overridable per candidate
    sizing: ClassVar[SizingSpec]             # default SizingSpec()
    def prepare(self, params: Mapping[str, Any]) -> None: ...
    def on_bar(self, ctx: Context) -> Signal: ...                       # bar_loop style
    def signals(self, bars: pd.DataFrame, params: Mapping[str, Any]) -> pd.Series: ...   # vectorized style; returns SignalKind per bar
```

`risk` and `sizing` are class-level defaults that a *candidate* overrides: the
engine takes them from `BacktestConfig.risk` and the candidate's params, so a
mutation can change a stop without rewriting the strategy's code. A strategy MUST
NOT attempt to implement its own stop-loss from `ctx`; the AST checker rejects a
strategy that reads `ctx.bars.low` or `ctx.bars.high` at the current bar for that
purpose, because doing so requires intrabar assumptions the `BarWindow` cannot
honour.

`BarWindow` wraps a NumPy view; `__getitem__`, `.open/.high/.low/.close/.volume/.ts_open` return arrays or scalars for `[0..i]` only; `.df(n)` returns a copy of the last `n` rows as a DataFrame (bounded to `n ≤ 5000`). Any attempt to read beyond `i` raises `LookaheadError` (INV-3).

For vectorized strategies the engine calls `signals()` once on the *full segment*, then **shifts by +1 bar** and consumes the result bar by bar. Vectorized strategies MUST pass the truncation probe (§14.2) before any backtest; the probe is executed automatically by the loader.

### 9.2 Rules for strategy source (enforced by `sandbox/ast_check.py`)

- Exactly one class implementing `Strategy`; module-level code limited to imports, constants, the class, and `STRATEGY = ClassName`.
- Imports only from `sandbox.allowed_imports` (config). `from x import *` forbidden.
- Forbidden names anywhere: `open, exec, eval, compile, __import__, getattr, setattr, delattr, globals, locals, vars, breakpoint, input, exit, quit`, attribute access to any name starting with `__` (except `__init__` definition), `os, sys, subprocess, socket, http, urllib, requests, httpx, pathlib, shutil, pickle, marshal, ctypes, threading, multiprocessing, asyncio, importlib, inspect, builtins, time, datetime, random`.
- `params` MUST be declared and every entry bounded; the number of params MUST be ≤ `validation.max_free_params + 4` at load time (the soft penalty applies above `max_free_params`).
- `warmup_bars` MUST be an `int` ≥ the largest lookback used (the leakage probe also checks this empirically).
- No mutable module-level state; instance state only.
- File ≤ 400 lines; `logic_lines` (non-blank, non-comment, non-import) stored in DB.
- No hand-rolled intrabar stop: a comparison of the current bar's own `high`/`low` against an entry price is rejected (§9.1). Reading a closed bar's `high`/`low` for any other purpose — a breakout channel, a range — is legitimate and MUST be accepted.

The checker MUST report **every** violation of a file in one pass, never the first
only: a strategy author (human or model) that is told one problem per attempt
rewrites blind. Each violation carries a stable code, a message, and a line number
where one exists:

| Code | Rule |
|---|---|
| `E_SYNTAX` | the source does not parse |
| `E_FILE_TOO_LONG` | more than 400 lines |
| `E_FORBIDDEN_IMPORT` | imports a banned module, however spelled |
| `E_IMPORT_NOT_ALLOWED` | imports a module absent from `sandbox.allowed_imports` |
| `E_STAR_IMPORT` | `from x import *` |
| `E_RELATIVE_IMPORT` | relative import |
| `E_FORBIDDEN_NAME` | mentions a banned builtin or module name |
| `E_DUNDER_ACCESS` | attribute access to a name starting with `__` |
| `E_NO_CLASS` / `E_MULTIPLE_CLASSES` | not exactly one class |
| `E_MODULE_CODE` | module-level code beyond imports, constants, the class, `STRATEGY` |
| `E_MODULE_STATE` / `E_MUTABLE_MODULE_STATE` | non-constant or mutable module-level name |
| `E_NO_PARAMS` / `E_PARAMS_NOT_LITERAL` | `params` missing, or not a dict literal |
| `E_PARAM_NOT_SPEC` / `E_UNBOUNDED_PARAM` / `E_TOO_MANY_PARAMS` | parameter declaration faults |
| `E_NO_WARMUP` / `E_WARMUP_NOT_INT` / `E_WARMUP_NEGATIVE` | `warmup_bars` faults |
| `E_NO_STRATEGY_EXPORT` / `E_STRATEGY_MISMATCH` | `STRATEGY` missing or naming another class |
| `E_INTRABAR_STOP` | hand-rolled intrabar stop (§9.1) |
| `W_LOGIC_LINES` | *warning*: above `validation.max_logic_lines`, where the §14 complexity penalty applies |

### 9.3 Indicator library (`core/indicators.py`)

Implemented once, causal by construction, tested against pandas reference. All functions take `(window: BarWindow | np.ndarray, ...)` and compute using only `[0..i]`. Provide: `sma(n)`, `ema(n)`, `rsi(n)`, `atr(n)`, `bbands(n, k)`, `donchian(n)`, `returns(n)`, `log_returns(n)`, `rolling_vol(n)`, `zscore(n)`, `highest(n)`, `lowest(n)`, `roc(n)`, `macd(fast, slow, signal)`. `ctx.ind("rsi", n=14)` caches per `(name, kwargs)` and updates incrementally where the indicator permits; otherwise recomputes on the window (correctness over speed).

### 9.4 Template (`strategies/TEMPLATE.py`)

```python
from quantlab.core.strategy import ParamSpec, Context, Signal, SignalKind

class MyStrategy:
    name = "my_strategy"; version = "1"; style = "bar_loop"
    params = {"fast": ParamSpec(kind="int", default=20, low=5, high=100),
              "slow": ParamSpec(kind="int", default=100, low=20, high=400)}
    warmup_bars = 400
    def prepare(self, params): self.p = dict(params)
    def on_bar(self, ctx: Context) -> Signal:
        fast = ctx.ind("sma", n=self.p["fast"]); slow = ctx.ind("sma", n=self.p["slow"])
        return Signal(SignalKind.LONG) if fast > slow else Signal(SignalKind.FLAT)

STRATEGY = MyStrategy
```

### 9.5 Baselines (`strategies/baselines/`)

`buy_and_hold` (LONG from warm-up 0), `sma_cross` (params fast=50, slow=200), `rsi_reversion` (rsi(14) < 30 → LONG, > 55 → FLAT), `random_entry` (each bar: with prob `p_enter` LONG, hold `hold_bars`, uses `ctx.rng`). Baselines are `origin='human'` families and are run on every segment by `quantlab baseline run`.

---

### 9.6 The strategy genome (`core/genome.py`)

Structural mutation (§13.4) cannot operate safely on free-form Python: an edit to
a token stream produces invalid or subtly broken code far more often than it
produces a strategy. The genome is a **declarative, typed, validated description**
of a strategy from which the module source is generated. Mutations act on the
genome, so a mutated candidate is valid **by construction**, and the generated
code still passes the AST check, the sandbox and the leakage probe as defence in
depth.

```python
class Operand(BaseModel, frozen=True):
    kind: Literal["indicator", "price", "constant", "param"]
    name: str                                 # 'sma' | 'close' | '' | 'fast'
    kwargs: dict[str, int | float | str] = {}  # indicator arguments, values may name params

class Condition(BaseModel, frozen=True):
    left: Operand
    op: Literal["<", "<=", ">", ">=", "cross_above", "cross_below"]
    right: Operand

class ConditionTree(BaseModel, frozen=True):
    mode: Literal["all", "any"] = "all"
    conditions: tuple[Condition, ...]

class StrategyGenome(BaseModel, frozen=True):
    name: str; version: str
    entry: ConditionTree
    exit: ConditionTree
    filters: tuple[Condition, ...] = ()
    risk: RiskSpec = RiskSpec()
    sizing: SizingSpec = SizingSpec()
    params: dict[str, ParamSpec]
    warmup_bars: int
```

**Validity rules** (pydantic validators; a genome that breaks one cannot be
constructed, so it can never enter the population):

1. every `Operand.name` of kind `indicator` exists in `core/indicators.py`, and
   its `kwargs` match that indicator's signature;
2. every `param` operand names a key of `params`, and every declared param is
   referenced at least once — an unused parameter is free-parameter inflation
   with no effect, and would only reward the complexity penalty;
3. every `ParamSpec` is bounded (§9.1);
4. `entry.conditions` is non-empty; counts respect `evolution.genome.max_*`;
5. no duplicate condition within a tree, and no condition compares an operand to
   itself;
6. `warmup_bars >= ` the largest lookback implied by any indicator operand — and
   the leakage probe re-checks this empirically (§14.2);
7. `len(params) <= evolution.genome.max_free_params + 4`, matching §9.2.

`evolution/compiler.py` exposes:

```python
def compile_genome(genome: StrategyGenome) -> str      # -> module source per §9.2/§9.4
def genome_id(genome: StrategyGenome) -> str           # sha256(canonical genome)[:16]
```

Compilation is **deterministic and total**: the same genome always produces
byte-identical source, so `strategy_id = sha256(code)[:16]` is stable and the run
cache (§11.2) works across generations. `compile_genome` MUST NOT execute
anything it generates (INV-4).

Two candidate kinds exist. A `genome` candidate carries a genome and admits both
parameter and structural mutation. An `opaque` candidate is free-form Python
authored by a human or an LLM; it admits **parameter mutation only**, because its
structure cannot be edited safely. Both compete in the same population on the
same fitness.

## 10. Metrics (`core/metrics.py`)

```python
def compute_metrics(result: BacktestResult, *, rf_annual: float = 0.0) -> MetricSet
class MetricSet(BaseModel, frozen=True):   # every field float | None; None = undefined
    net_return, cagr, max_drawdown, max_drawdown_bars, profit_factor, sharpe, sortino, win_rate,
    n_trades, avg_trade_pct, avg_trade_usdt, expectancy_pct, exposure,
    calmar, ann_volatility, avg_bars_held, longest_losing_streak, top5_profit_share, turnover,
    buy_hold_return, excess_vs_buy_hold, sharpe_ci_low, sharpe_ci_high, low_trade_warning: bool
```

Definitions (returns are computed on bars after warm-up; `T` = number of such bars; `Y = T / bars_per_year`):

- `net_return = E_T / E_0 − 1`; `cagr = (E_T/E_0)^(1/Y) − 1` (None if `Y < 1/365`).
- `max_drawdown = max_t (1 − E_t / max_{s≤t} E_s)` ∈ [0, 1]; `max_drawdown_bars` = longest peak-to-recovery span.
- `profit_factor = Σ pnl⁺ / |Σ pnl⁻|`; None if no losing trades (and `low_trade_warning` forced True).
- Per-bar returns `r_t = E_t/E_{t−1} − 1`. `sharpe = mean(r − rf_bar)/std(r, ddof=1) · sqrt(bars_per_year)`; None if `std == 0`. `sortino` uses downside deviation `sqrt(mean(min(r − rf_bar, 0)²))`; None if downside deviation is 0.
- `sharpe_ci_*`: Lo (2002) asymptotic 95 % interval `SR ± 1.96 · sqrt((1 + SR²/2)/T)` (annualised consistently).
- `win_rate = #(pnl > 0)/N`; `n_trades = N`; `avg_trade_pct = mean(pnl_pct)`; `expectancy_pct = win_rate · mean(pnl_pct⁺) − (1 − win_rate) · |mean(pnl_pct⁻)|`.
- `exposure = #(position_frac ≠ 0) / T`.
- `low_trade_warning = N < 30`.
- `buy_hold_return = close[−1]/open[warmup] − 1` net of one round trip of costs; `excess_vs_buy_hold = net_return − buy_hold_return`.

All metrics MUST be computed from the equity series and trade list only (never re-reading prices except for buy-and-hold), so any engine adapter gets identical metric semantics. `docs/METRICS.md` MUST mirror these definitions and be updated in the same commit as any change.

---

## 11. Experiment tracking

### 11.1 Port (`ports/store.py`)

```python
class ExperimentStore(Protocol):
    def get_or_create_dataset(...)-> Dataset; def get_or_create_split(...)-> SplitPolicy
    def create_family(...)-> StrategyFamily; def add_strategy_version(...)-> StrategyVersion
    def create_experiment(...)-> Experiment
    def find_run(run_id) -> Run | None; def create_run(...)-> Run; def finish_run(run_id, status, metrics, trades, error=None)
    def save_verdict(...); def record_llm_interaction(...); def record_lockbox_access(...)
    def lineage(strategy_id) -> list[StrategyVersion]; def query_runs(**filters) -> list[Run]
    def increment_validation_touches(family_id) -> int; def set_family_status(family_id, status)
    # evolution (§13)
    def create_evolution_run(...)-> EvolutionRun; def finish_evolution_run(evolution_id, status, stop_reason)
    def add_generation(...)-> Generation; def add_candidate(...)-> Candidate
    def add_mutations(candidate_id, mutations) -> None       # append-only; never updated
    def record_promotion(...)-> Promotion                    # written BEFORE the run (INV-9)
    def candidates_for(evolution_id, gen_index=None) -> list[Candidate]
    def ancestry(candidate_id) -> list[Candidate]; def descendants(candidate_id) -> list[Candidate]
class ArtifactStore(Protocol):
    def dir_for(run_id) -> Path; def write_parquet(run_id, name, df); def write_json(run_id, name, obj); def read_json(run_id, name)
```

### 11.2 Identity and caching

```
run_id = sha256("|".join([strategy_id, canonical(params), dataset_id, split_id, segment,
                          config_hash_of(BacktestConfig), str(seed), engine_name, engine_version]))[:16]
```

`ExperimentRunner.run(...)` MUST first call `store.find_run(run_id)`; if a run exists with `status='ok'` it returns it without executing (log `cache_hit=True`) unless `force=True`. Failed runs are re-executed.

### 11.3 Artifacts per run (`artifacts/runs/<run_id>/`)

`params.json`, `backtest_config.json`, `equity.parquet`, `position.parquet`, `signals.parquet`, `trades.parquet`, `fills.parquet`, `metrics.json`, `engine_log.txt`, `env.json` (`python`, `platform`, `packages` from `importlib.metadata`, `git_commit`, `git_dirty: bool`, `engine_version`, `created_at`). A run with `git_dirty=true` is allowed but flagged in reports.

### 11.4 Reproduce

`quantlab reproduce <run_id>` reloads inputs from the store, re-runs in the sandbox, and compares every non-None metric with `math.isclose(rel_tol=1e-9, abs_tol=1e-12)`; on mismatch it exits non-zero and prints a diff of trades. This command is the enforcement of INV-7.

---

## 12. GLM integration interface

**Role change (spec 1.1).** The LLM no longer drives a sequential loop. It is a
*proposer* inside the evolutionary search (§13), in three bounded ways:

1. **Seed genomes** — up to `research.max_seed_genomes` genomes for generation 0.
2. **Immigrants** — novel genomes on request when diversity is low (§13.5).
3. **Guided mutation** — for at most `research.guided_mutation_share` of
   offspring, a suggested structural operator and target, which is then applied
   through the same typed operator as any other mutation and recorded with
   `mutation.suggested_by = 'llm'`.

The LLM proposes **genomes as JSON**, never free-form Python. This removes the
largest source of malformed output, makes every proposal validatable by pydantic
before it costs a backtest, and means an LLM proposal is structurally mutable
afterwards like any other candidate.

The LLM never sees a validation- or test-derived number while an evolution run is
active (INV-11), and it never decides survival, ranking or stopping — those are
fitness-driven and deterministic.

### 12.1 Port (`ports/llm.py`)

```python
@dataclass(frozen=True) class Message: role: Literal["system","user","assistant"]; content: str
@dataclass(frozen=True) class LLMResponse:
    text: str; parsed: BaseModel | None; tokens_in: int; tokens_out: int; latency_ms: int
    model: str; provider: str; raw_id: str | None; finish_reason: str

class LLMProvider(Protocol):
    name: str; model: str
    def complete(self, messages: Sequence[Message], *, response_model: type[BaseModel] | None,
                 temperature: float, max_tokens: int, seed: int | None) -> LLMResponse: ...
    def cost_eur(self, tokens_in: int, tokens_out: int) -> float: ...
```

### 12.2 `GLMProvider` (`adapters/llm/glm.py`)

- POST `{base_url}/chat/completions` with `Authorization: Bearer <key>` (key from `Secrets`, never logged); body: `model, messages, temperature, max_tokens, response_format={"type":"json_object"}` when `response_model` is given, plus the JSON schema embedded in the system message (GLM's structured-output support varies by model — the adapter MUST work with plain JSON mode and validate with pydantic; on `ValidationError` it retries once with the error appended, then returns `status="malformed"`).
- Retries: `tenacity` exponential back-off (1 s → 30 s, 3 attempts) on 429/5xx/timeouts; never on 4xx other than 429.
- **Cost accounting:** prices from `configs/llm_pricing.yaml` (`{model: {input_eur_per_mtok, output_eur_per_mtok}}`); the adapter keeps a daily ledger in the store and raises `BudgetExceeded` before sending a request that would cross `llm.daily_cost_cap_eur`.
- Every call writes `prompt.json`/`response.json` to `artifacts/llm/<interaction_id>/` and one `llm_interaction` row.

### 12.3 `MockLLMProvider`

Replays responses from `tests/fixtures/llm/<campaign>/<n>.json` in order; in `record` mode (env `QUANTLAB_LLM_RECORD=1`, human-triggered only) it wraps the real provider and saves responses. Every evolution and proposer test uses the mock, so the whole search is testable offline and deterministically.

### 12.4 Structured schemas (`research/schemas.py`)

```python
class StrategyProposal(BaseModel):
    hypothesis: str = Field(min_length=20, max_length=1000)     # why should this edge exist?
    code: str = Field(max_length=20000)                          # full module text per §9
    expected_trade_frequency: Literal["hours","days","weeks"]
    notes: str = ""
class GenomeProposal(BaseModel):                 # replaces StrategyProposal for genome candidates
    hypothesis: str = Field(min_length=20, max_length=1000)
    genome: StrategyGenome                       # validated by §9.6 before it costs anything
    expected_trade_frequency: Literal["hours","days","weeks"]
    notes: str = ""

class MutationSuggestion(BaseModel):             # guided mutation (§13.4)
    operator: str                                # must name a structural operator
    target: str                                  # genome path
    reasoning: str

class GenerationReview(BaseModel):
    observations: str
    suggested_directions: list[str] = []
    new_genomes: list[GenomeProposal] = []       # candidate immigrants
```

`ReviewDecision` with its `MODIFY / NEW_IDEA / ABANDON_FAMILY / STOP` actions is
**removed**: those decisions are now made by fitness, the diversity policy and
the stopping rule, not by the model.

### 12.5 Prompt files

Prompts are Markdown templates with `{{placeholders}}` (simple `str.replace`, no Jinja). The template file's SHA-256 is stored per interaction.

`system.md` MUST state: the **genome schema** (§9.6) and that proposals are JSON, not code; the allowed indicators and their arguments; that costs are 10 bps + slippage; that ≤ 6 parameters is preferred and every declared parameter must be used; that the model will never see the validation or test period; that lower out-of-sample than in-sample performance is expected; that fitness is multi-objective and net profit carries almost no weight; and the plain-language meaning of each fitness gate and rejection reason.

The three templates are:

| file | purpose | inputs |
|---|---|---|
| `propose_genome.md` | seed and immigrant genomes | market description, indicator library, existing population *signatures only* (never their fitness — that would let the model chase the leader) |
| `repair.md` | one retry for a malformed or invalid proposal | the pydantic validation error |
| `review_generation.md` | optional commentary and suggested directions | the INV-11 generation report: train and inner-fold numbers only |

`review.md` from spec 1.0 is removed along with `ReviewDecision`.

---

## 13. Evolutionary optimisation

Supersedes the sequential *propose → backtest → modify* loop of spec 1.0. The
search is a population-based evolutionary strategy: many candidates advance in
parallel, selection is on a multi-objective fitness rather than a single return
number, and the history of how every candidate came to exist is persisted.

Why population-based rather than sequential: a sequential loop keeps one line of
enquiry alive at a time, so a promising but currently-mediocre idea is discarded
before it is developed, and the researcher's attention (human or LLM) becomes a
single point of bias. A population explores several regions of strategy space at
once, and — critically for this project — it makes the size of the search
**countable**. Every candidate evaluation is a trial, and §14.4 charges all of
them to the deflated Sharpe ratio, so a wider search must clear a higher bar.

### 13.1 Vocabulary

| term | meaning |
|---|---|
| **candidate** | one `(strategy_version, params)` pair with its lineage and fitness; a row in `candidate` |
| **genome** | the declarative structure a candidate was compiled from (§9.6); `NULL` for `opaque` candidates |
| **generation** | one full evaluate → rank → select → reproduce cycle |
| **survivor** | a candidate carried into the next generation unchanged |
| **offspring** | a child produced from a survivor by mutation |
| **immigrant** | a novel candidate unrelated to any survivor |
| **promotion** | the recorded, budgeted act of evaluating a candidate on the validation segment |

### 13.2 The generation loop (`evolution/loop.py`)

```python
def evolve(*, seeds: Sequence[Candidate], bars_train: BarFrame, split: SplitPolicy,
           cfg: EvolutionConfig, runner: ExperimentRunner, store: ExperimentStore,
           llm: LLMProvider | None = None) -> EvolutionResult
```

For each generation `g` in `0 .. max_generations - 1`:

1. **Evaluate** every candidate in the population on the **train** segment
   (§13.7 forbids anything else). Each evaluation is an ordinary cached run
   (§11.2), so a survivor carried forward costs nothing and only genuinely new
   candidates consume budget. A candidate whose run fails is scored
   `FITNESS_REJECTED` and recorded with its error, never silently dropped.
2. **Score** each candidate with the multi-objective fitness of §13.3, storing
   every component and penalty in `candidate.components_json` / `penalties_json`
   so a ranking can be audited after the fact.
3. **Rank** by fitness, descending. Ties break by `(higher inner-OOS component,
   fewer free parameters, lower `candidate_id`)` — deterministic, and biased
   towards the simpler strategy.
4. **Select survivors**: walk the ranking and accept a candidate unless its
   similarity (§13.5) to an already-accepted survivor exceeds
   `diversity.max_pairwise_similarity`, in which case skip it and take the next.
   Stop at `n_survivors`. If diversity-aware selection cannot fill the quota,
   the shortfall becomes extra immigrants rather than near-duplicates.
5. **Reproduce**: produce `n_offspring` children by mutating survivors (§13.4).
   Parents are drawn by rank-proportional selection without replacement, so the
   best survivor is favoured but does not monopolise the next generation.
6. **Immigrate**: produce `n_immigrants` novel candidates (§13.5), raised by
   `diversity.immigrant_boost` (capped at `max_immigrants`) whenever population
   diversity is below `min_population_diversity`.
7. **Assemble** exactly `population_size` candidates. The identity
   `n_survivors + n_offspring + n_immigrants == population_size` is asserted at
   config load; a violation is a `ConfigError`, not a silent resize.
8. **Persist** the generation, its statistics and its diversity, then repeat.

**Stopping.** The loop stops at `max_generations`, or when `max_evaluations` or
`max_wall_clock_s` is reached, or when the best fitness has not improved by
`min_improvement` for `stop_on_no_improvement_generations` generations. The
reason is stored in `evolution_run.stop_reason`. Stopping early does not change
`n_evaluations`, which is what §14.4 charges.

**Determinism (INV-7).** Every stochastic choice draws from a
`numpy.random.Generator` seeded from `(evolution.seed, gen_index, slot_index,
attempt)`. Re-running an evolution from its stored configuration reproduces the
identical sequence of `candidate_id`s.

**Generation 0** is filled from, in order: explicitly supplied seed candidates
(baselines and any human strategies), LLM-proposed genomes up to
`research.max_seed_genomes` (§12), and random genomes drawn from the operator
library until the population is full.

### 13.3 Fitness (`core/fitness.py`)

```python
def compute_fitness(m: MetricSet, trades: Sequence[Trade], inner: InnerFoldReport,
                    sens: SensitivityReport | None, cfg: FitnessConfig) -> FitnessResult
```

Fitness is **gated, weighted, and penalised**, in that order. Net profit is one
of ten components and carries the smallest weight but one; a strategy cannot
climb the ranking by making more money in a less trustworthy way.

**Stage 1 — hard gates.** Failing any gate sets `fitness = FITNESS_REJECTED`
(`-1e9`) and records the gate id in `candidate.gate_failure`. Gates are absolute:
no other component can compensate for one.

| gate | rule |
|---|---|
| `F_EXPECTANCY` | `expectancy_pct > gates.min_expectancy_pct` |
| `F_TRADES` | `n_trades >= gates.min_trades` |
| `F_DRAWDOWN` | `max_drawdown <= gates.max_drawdown` |
| `F_CONCENTRATION` | `retention_1 > gates.min_retention_top1` (§14.4) — the strategy must still be profitable with its single best trade removed |

`F_EXPECTANCY` and `F_DRAWDOWN` are what make the required property true: **win
rate can never override negative expectancy or excessive drawdown**, because a
candidate that fails either is rejected before the weighted score is computed,
and win rate is only ever an input to that score.

**Stage 2 — weighted base score**, each component mapped into `[0, 1]` so no term
can dominate by unit choice. With `clip(x) = min(1, max(0, x))`:

| component | definition | default weight |
|---|---|---|
| `expectancy` | `clip(expectancy_pct / targets.expectancy_pct)` | 0.20 |
| `risk_adjusted` | `clip(sortino / targets.sortino)` | 0.15 |
| `drawdown` | `clip((ceiling - max_drawdown) / ceiling)`, `ceiling = targets.drawdown_ceiling` | 0.15 |
| `profit_factor` | `clip((profit_factor - 1) / (targets.profit_factor - 1))`; `None` (no losers) scores 1.0 and forces the low-trade warning | 0.12 |
| `consistency` | share of non-overlapping `targets.consistency_period_bars` windows with positive net return | 0.12 |
| `inner_oos` | `clip(inner_oos_sortino / max(inner_is_sortino, eps))` from §13.6 — generalisation measured **inside train** | 0.12 |
| `trades` | `clip(log1p(n_trades) / log1p(targets.trades))` | 0.06 |
| `win_rate` | `clip((win_rate - targets.win_rate_floor) / (targets.win_rate - targets.win_rate_floor))` | 0.04 |
| `net_return` | `clip(cagr / targets.cagr)` | 0.02 |
| `concentration` | `clip(1 - top5_profit_share)` | 0.02 |

`base_score = Σ weight_i · component_i ∈ [0, 1]`. Weights MUST sum to 1.0
(validated at config load) so the score stays comparable across configurations.

The `trades` and `consistency` components together encode the required
preference: a strategy earning steadily across many periods and many trades
outscores one earning the same total from a handful of outliers, which the
`concentration` component and the removal penalty then penalise directly.

**Stage 3 — multiplicative penalties**, each in `[0, 1]`:

| penalty | fires when | shape |
|---|---|---|
| `p_drawdown` | `max_drawdown > penalties.drawdown_soft` | linear to 0 at `gates.max_drawdown` |
| `p_trades` | `n_trades < penalties.trades_soft` | `clip(n_trades / trades_soft)` |
| `p_instability` | CV of per-fold inner-OOS return `> instability_max_cv` | linear decay |
| `p_sensitivity` | median neighbour objective drop `> sensitivity_max_drop` (§13.8 neighbourhood) | linear decay |
| `p_removal` | trade-removal retention below target (§14.4) | `min_k clip((ret_k - floor_k) / (target_k - floor_k))` |
| `p_divergence` | `inner_oos_sortino / inner_is_sortino < divergence_min_ratio` | linear decay |
| `p_complexity` | free params `> complexity_free_params` or logic lines `> complexity_logic_lines` | `0.95^excess` |

`fitness = base_score · Π penalties`, in `[0, 1]`, or `FITNESS_REJECTED`.

Penalties are multiplicative rather than subtractive on purpose: two independent
robustness problems compound, and a candidate that is fragile in several ways at
once should fall far, not twice as little as it fell once.

### 13.4 Mutation (`evolution/mutation.py`)

```python
def mutate(parent: Candidate, cfg: MutationConfig, rng: Generator,
           library: OperatorLibrary) -> MutationResult   # child genome + applied mutations
```

A child receives between 1 and `max_mutations_per_child` mutations, drawn by
category from `parameter_rate` / `structural_rate`. Each is recorded as a
`mutation` row carrying the operator, the genome path it touched, the before and
after values, and the RNG seed that produced it — which is what makes INV-10
replayable.

**Parameter mutations** operate on any bounded `ParamSpec`, including the risk
and sizing parameters that §9.1 makes first-class:

| operator | effect |
|---|---|
| `perturb_numeric` | `x *= 1 ± U(0, perturb_pct)`, snapped to the spec's bounds and step |
| `jump_numeric` | resample uniformly from the spec's range (probability `jump_probability`) |
| `toggle_bool` | flip |
| `resample_categorical` | draw a different choice |
| `perturb_risk` | as `perturb_numeric` but with `risk_perturb_pct`, applied to `stop_loss_pct`, `take_profit_pct`, `trailing_stop_pct`, `time_stop_bars` |
| `perturb_sizing` | as above, applied to `SizingSpec` |
| `toggle_risk_control` | enable or disable one risk control (`None` ↔ a drawn value) |

**Structural mutations** operate on the genome tree:

| operator | effect |
|---|---|
| `add_confirmation` | append a condition to `entry` (or `exit`), drawn from the operator library |
| `remove_confirmation` | drop a condition, never the last one in `entry` |
| `modify_entry` | change a condition's comparator, or replace one operand |
| `modify_exit` | as above, on `exit` |
| `add_filter` / `remove_filter` | add or drop a regime, volatility or session filter |
| `replace_indicator` | swap an indicator operand for another with a compatible signature |
| `change_tree_mode` | `all` ↔ `any` on `entry` or `exit` |

**Preventing invalid strategies.** Four layers, in order:

1. **Validity by construction** — operators emit genome edits, and a genome that
   breaks a §9.6 rule cannot be constructed at all.
2. **Redraw** — an operator that produces an invalid genome is retried with a new
   RNG draw up to `max_repair_attempts`; the failure is counted in
   `generation.stats_json`.
3. **Fall back** — if every attempt fails, the slot is filled by an immigrant.
   A child is never repaired by relaxing a rule.
4. **Defence in depth** — the compiled source still passes `ast_check` (§9.2),
   runs in the sandbox (§21.3) and clears the leakage probe (§14.2) before it can
   be scored. A candidate that fails any of these is recorded with
   `gate_failure` and scored `FITNESS_REJECTED`.

`opaque` candidates admit parameter mutations only; a structural operator drawn
for one is re-drawn as a parameter operator.

### 13.5 Diversity (`evolution/diversity.py`)

Without an explicit mechanism a population converges: the fittest candidate's
descendants fill every slot within a few generations, the search stops exploring,
and sixteen near-identical strategies produce sixteen near-identical — and
mutually uninformative — validation results.

**Similarity** between two candidates combines structure and behaviour:

```
similarity(a, b) = structural_weight · jaccard(signature(a), signature(b))
                 + behavioural_weight · agreement(position_frac(a), position_frac(b))
```

* `signature` is the multiset of `(component kind, indicator name, comparator)`
  triples from the genome, plus the set of enabled risk controls. For an `opaque`
  candidate it is derived from its AST.
* `agreement` is the fraction of train bars on which the two candidates hold the
  same sign of `position_frac`. This is the more important half, and it is
  weighted accordingly: two structurally different strategies that enter and exit
  together are **not** diverse, and their agreement is what would mislead a
  reviewer into thinking two independent methods had confirmed each other.
* `behaviour_hash` on `candidate` is a digest of the quantised `position_frac`
  series, so exact behavioural duplicates are detected without an O(n²) scan.

**Enforcement.**

1. *Niching in selection* — step 4 of §13.2 skips a candidate too similar to a
   fitter survivor already accepted.
2. *Diversity floor* — `generation.diversity = 1 − mean pairwise similarity`. While
   it is below `min_population_diversity`, the immigrant count rises by
   `immigrant_boost` (capped at `max_immigrants`) and offspring fall to match.
   Survivors are never displaced, so `max_immigrants` MUST NOT exceed
   `n_offspring + n_immigrants`; a configuration that does is a `ConfigError`.
3. *Reserved novelty* — `n_immigrants ≥ 1` always, so every generation contains at
   least one candidate that owes nothing to the current leader.

An immigrant is drawn as a fresh random genome, or requested from the LLM (§12)
when one is configured, and is rejected and redrawn if its similarity to any
current population member exceeds `max_pairwise_similarity`.

### 13.6 Lineage and the inner walk-forward (`evolution/lineage.py`)

**Lineage.** Every candidate except a seed names `parent_candidate_id`, and every
mutation that produced it is stored in order. This makes the whole run
reconstructable:

```python
def ancestry(store, candidate_id) -> list[Candidate]        # child -> ... -> seed
def descendants(store, candidate_id) -> list[Candidate]
def lineage_tree(store, evolution_id) -> LineageTree        # for the dashboard (§22 T42)
def replay(store, candidate_id) -> StrategyGenome           # re-applies the recorded mutations
```

INV-10 requires `replay(candidate_id) == stored genome` for every genome
candidate in the store, which is what turns "we can visualise the evolution" from
a reporting feature into a checked property.

**Inner walk-forward.** Generalisation must be measured without touching the
validation segment. The train segment is therefore split into
`inner_walkforward.n_folds` in-sample/out-of-sample folds with their own embargo,
entirely inside train. `InnerFoldReport` carries per-fold IS and OOS metrics; it
feeds the `inner_oos` fitness component, `p_instability` and `p_divergence`.

This is the design's answer to "fitness should consider out-of-sample
performance" without spending the validation set sixteen times per generation.

### 13.7 Segment discipline (INV-9)

| segment | who may read it | when |
|---|---|---|
| train (incl. inner folds) | the evolutionary optimiser | every generation |
| validation | the promotion path only (**Promotion**, below) | `promotion.n_promote` candidates, per `every_generations` |
| test | `quantlab lockbox` only | never during evolution |

Mechanically enforced:

* the container is built with `profile="research"`, so `PartitionGuard` (§7.4)
  raises `LockboxViolation` on any range reaching the test partition;
* `evolve()` asserts `segment.startswith(("train", "inner_is", "inner_oos"))`
  for every run it creates;
* a validation run may only be created by `promote()`, which writes the
  `candidate_promotion` row **before** the run executes, and increments
  `strategy_family.validation_touches`;
* no validation-derived number is fed back into fitness, selection, mutation or
  stopping. Promotion results inform the human and the final report only.

**Promotion.** After the final generation (or every `promotion.every_generations`
generations), the top `promotion.n_promote` candidates with
`fitness >= promotion.min_fitness` are promoted, subject to pairwise similarity
below `promotion.max_similarity_between_promoted` so that the validation budget
is not spent three times on the same strategy. Each promoted candidate then runs
the full validation pipeline of §14.1.

### 13.8 Parameter refinement and plateau selection (`optimize/`)

Retained from spec 1.0, demoted from primary search to a **post-evolution
refinement** of promoted candidates only.

* `optimize/objectives.py` is unchanged: return-only objectives raise
  `ConfigError`. It is now used for the neighbourhood objective in `p_sensitivity`
  and for refinement, not for the population ranking.
* `optimize/study.py` may run an Optuna study on a promoted candidate's parameters
  on the **train** segment (`assert segment.startswith(("train", "wf_is"))` still
  holds). Trials add to `n_trials_accounted`.
* `optimize/plateau.py` is **mandatory** before any validation run: the parameters
  submitted for validation are the plateau choice, never the point optimum. The
  report shows both.

Evolution finds *a* good region; plateau selection makes sure the point taken
from that region is not a spike that the next month of data will fall off.

### 13.9 CLI

```
quantlab evolve run     --campaign <name> [--generations N] [--population N] [--seed N]
quantlab evolve resume  <evolution_id>
quantlab evolve status  <evolution_id>
quantlab evolve lineage <candidate_id> [--tree]
quantlab evolve promote <evolution_id> [--n 3]      # runs the validation pipeline (§14.1)
```

`quantlab evolve run` is resumable: a crash mid-generation leaves the completed
candidates in the store, and `resume` re-enters at the first generation with no
`generation` row, re-using cached runs (§11.2) so no work is repeated.

## 14. Validation system

### 14.1 Pipeline (`quantlab validate <strategy_id> [--params run_id|json]`)

Entered either directly by a human, or by `quantlab evolve promote` for each
promoted candidate (§13.7). When entered by promotion, the `candidate_promotion`
row already exists and step 8 charges the touch to the candidate's family.

```
0. Load strategy (AST check already passed at load); refuse if family.status != 'open'
1. Leakage probe (§14.2)                                → hard gate
2. Train run (base costs)                               → metrics_train
3. Walk-forward over train+val windows (§15)            → metrics_wf_oos (stitched), WFE, param stability
4. Validation run with plateau params (base, 2×, 3× costs) → metrics_val, cost survival
5. Baselines on val (cached)                            → benchmark comparisons; random_entry distribution
5b. Trade-removal test (§14.4 check 10) on train and val → retention_1/3/5
6. Hard gates (§14.3)                                   → any fail ⇒ REJECT
7. Soft checks (§14.4) ⇒ overfit_score
8. Verdict; increment family.validation_touches; freeze family if ≥ family_max_validation_touches
9. Persist validation_verdict (+ thresholds_json snapshot) and a markdown report
```

### 14.2 Leakage probe (`core/validation/leakage.py`)

```python
def truncation_probe(strategy, bars: BarFrame, params, *, cut_points: Sequence[int], engine) -> ProbeResult
```
For each `k` in `cut_points` (default: 5 deterministic points at 20/35/50/65/80 % of the segment): run on `bars[0..k]` and on the full segment; the first `k+1` emitted signals MUST be identical **and** the first `k+1` position fractions identical. Also run once with the last 10 % of bars replaced by a reversed copy: signals before the replacement point MUST be unchanged. Any difference ⇒ `LEAKAGE_DETECTED`. Additionally, warm-up under-declaration is detected if signals at bars `< warmup_bars` differ between runs.

Fixtures in `tests/fixtures/strategies/leaky/` MUST include: `close.shift(-1)` use; same-bar `high` breakout entry filled at same-bar open; centred rolling mean; indicator computed on the full series then indexed. All MUST be detected.

### 14.3 Hard gates (`core/validation/gates.py`)

| Gate id | Rule (config keys in §5) |
|---|---|
| `G_LEAK` | probe passed |
| `G_MIN_TRADES` | `n_trades_val ≥ min_trades_val` and `n_trades_train ≥ min_trades_train` |
| `G_SANITY` | no trade with `abs(pnl_pct) > max_single_trade_pct`; `max(abs(position_frac)) ≤ max_position_fraction · (1 + sanity_drift_allowance)`, default allowance `0.5`. The cap bounds the *target* fraction at decision time; the realised fraction then drifts with the market between rebalances, so a `1e-9` tolerance would fail on correctly-behaved strategies. The exact invariant — committed capital at a fill never exceeds `max_position_fraction × equity` at the deciding bar — is checked by the engine's property tests. |
| `G_COST` | `net_return_val@(cost_survival_multiplier×) > 0` |
| `G_PERM` | market-permutation p-value ≤ `permutation.alpha` (§14.4-3) |
| `G_DEGRADE` | `sharpe_val > 0` and `sharpe_val ≥ degradation_min_ratio · sharpe_train` |
| `G_BENCH` | `sortino_val > sortino_buyhold_val` **or** (`max_drawdown_val < 0.5 · max_drawdown_buyhold_val` and `net_return_val > 0`) |

Output: `{gate_id: {"passed": bool, "observed": …, "threshold": …}}`.

### 14.4 Soft checks and score (`core/validation/score.py`)

**The trade-removal test (`core/validation/concentration.py`).** A strategy whose
profit is carried by a handful of exceptional trades has not demonstrated an
edge; it has demonstrated that a few things happened. The test removes the top
`k` winning trades by `pnl` for `k ∈ {1, 3, 5}` and recomputes:

```python
@dataclass(frozen=True, slots=True)
class TradeRemovalReport:
    k_values: tuple[int, ...]
    retention: dict[int, float]           # Σ pnl(remaining) / Σ pnl(all)
    expectancy: dict[int, float | None]   # recomputed on the remaining trades
    profit_factor: dict[int, float | None]
    n_trades: int

def trade_removal_report(trades: Sequence[Trade], k_values=(1, 3, 5)) -> TradeRemovalReport
```

Retention is computed on the **trade pnl series**, exactly and without
re-simulation, so it is well defined and cheap enough to run for every candidate
in every generation. `retention_k` is `None` when `Σ pnl(all) <= 0`, which cannot
occur for a candidate that passed the `F_EXPECTANCY` gate.

Its results are used in two places: as the `p_removal` fitness penalty during
evolution (§13.3), and as soft check 10 plus the `F_CONCENTRATION` hard gate
here. A strategy that becomes unprofitable after losing its single best trade is
rejected outright, not merely penalised.

| # | Check | Points |
|---|---|---|
| 1 | Deflated Sharpe (`deflated_sharpe.py`; Bailey & López de Prado 2014) with `M = n_trials_accounted = evolution_run.n_evaluations + optuna n_trials + family.validation_touches`, skew/kurtosis of per-bar returns, `T` bars. `DSR < dsr_threshold` | +25 |
| 2 | PBO via CSCV (`pbo.py`): `n_blocks` blocks over train; for up to `max_combinations` IS/OOS combinations, best-IS trial's OOS rank; `PBO > 0.5` / `> 0.3` | +25 / +10 |
| 3 | Market permutation (`permutation.py`): `n_market_permutations` stationary block-bootstrapped return series (block `block_len_bars`, seed fixed), same params; `p = (1 + #(sortino_perm ≥ sortino_real)) / (1 + n)`. Also trade-order shuffle `n_trade_shuffles` → `mdd_p95`. `p > alpha` is the hard gate; here `p > alpha/2` | +10 |
| 4 | Sensitivity (`sensitivity.py`): median neighbour objective drop `> sensitivity_max_drop` of optimum | +20 |
| 5 | Concentration: top-`concentration_top_n` trades share of gross profit `> concentration_max_share` | +10 |
| 6 | Walk-forward: `WFE < 0.5` / profitable OOS windows `< 50 %` / any param CV across windows `> 0.5` | +20 / +10 / +5 |
| 7 | Complexity: `+5` per param above `max_free_params`; `+5` if `logic_lines > max_logic_lines` | var |
| 8 | Regime: `> 90 %` of validation net profit inside one calendar quarter | +10 |
| 9 | Random-entry baseline: `sortino_val ≤` 95th percentile of `random_entry_seeds` seeded runs | +20 |
| 10 | **Trade removal** (`concentration.py`): remove the top 1, 3 and 5 winning trades by `pnl` and recompute. `retention_k = Σ pnl(remaining) / Σ pnl(all)`. `retention_1 < 0.40` / `retention_3 < 0.20` / `retention_5 < 0.10` | +25 / +15 / +10 |
| 11 | Evolution provenance: the candidate's `evolution_run.n_evaluations > 500` and its `inner_oos` fitness component `< 0.5` — a wide search that never generalised in-sample | +10 |

`overfit_score = min(100, Σ points)`. Verdict: all gates pass and `score ≤ candidate_max` ⇒ `CANDIDATE`; gates pass and `≤ weak_max` ⇒ `WEAK`; else `REJECT`. `thresholds_json` snapshots every threshold used.

### 14.5 Redaction (`research/redaction.py`, INV-6 and INV-11)

`RedactedReport` is a pydantic model whose fields are exclusively: strategy code + lineage, params, `metrics_train`, `metrics_val`, `metrics_wf_oos`, `gates`, `soft_checks`, `overfit_score`, `verdict`, `baseline_metrics_train_val`, `rejected_ideas_summary`, `validation_touches`. It is constructed by `build_redacted_report(store, strategy_id)` which filters `run.segment NOT LIKE 'test%'` and never reads `lockbox_access` or `split_policy.test_*`. Test: serialise the report and assert none of the test-period ISO dates or the strings `test`, `lockbox` (as segment names) appear.

**INV-11.** While an evolution run has `status='running'`, a stricter mode
applies: `build_generation_report(store, evolution_id, gen_index)` — the only
report an LLM may see during evolution — filters `run.segment NOT LIKE 'val%'`
as well, and carries train and inner-fold numbers only. Validation results reach
the human, never the loop. The test asserts that no promoted candidate's
validation metrics, and no validation-period ISO date, appear in any prompt file
written while an evolution run was active.

### 14.6 Lockbox (`cli/lockbox.py`, INV-5)

`quantlab lockbox evaluate <strategy_id> --params <run_id> --reason "<text≥20 chars>"`: requires verdict `CANDIDATE`; checks `lockbox.max_per_family` and `max_per_month`; freezes `test_end_ts` on first use; builds the container with `profile="lockbox"`; runs base + 2× costs on the test segment; applies gates `G_MIN_TRADES` (val thresholds), `G_COST`, `G_DEGRADE` (vs train), `G_BENCH`; writes `LOCKBOX_PASS/FAIL`; on FAIL sets `family.status='closed'`. Writes `lockbox_access` **before** running. The research CLI MUST NOT import `cli/lockbox.py`.

---

## 15. Walk-forward testing (`walkforward/runner.py`)

```python
def walk_forward(strategy, bars_train_val: BarFrame, split: SplitPolicy, cfg: WalkForwardConfig, *, runner, optimizer) -> WalkForwardResult
```

1. Windows from `core/splits.py`: rolling `(is_start, is_end, oos_start, oos_end)` with `is_bars`, `oos_bars`, `step_bars`, starting at `train_start`; last OOS window ends at `val_end`; windows crossing the embargo are shifted so that no window straddles it. `anchored` variant keeps `is_start = train_start`.
2. Per window: search on IS if `reoptimize_each_window`, else use the given params; select plateau params (§13.8); run OOS with `segment = f"wf_oos:{k}"`.
   * When `optimize.engine == "evolution"`, re-searching a window means running a
     short evolution of `walkforward.evolution_generations` generations with the
     same population configuration, seeded from `(evolution.seed, window k)`.
     This dominates the runtime of the whole pipeline: 23 windows × 8 generations
     × 16 candidates is ~2 900 evaluations. Budget for it, or set
     `reoptimize_each_window: false` and state plainly in the report that the
     walk-forward reused fixed parameters, which is **weaker evidence** because it
     does not test whether the search itself generalises.
   * Every evaluation inside a window counts towards `n_trials_accounted`
     (§14.4). A walk-forward that re-searches each window therefore raises the
     deflated-Sharpe bar substantially, which is correct: it is a much larger
     search.
3. Stitch OOS equity curves by chaining returns (`E_{k+1,0} = E_{k,end}`); compute `MetricSet` on the stitched curve (`metrics_wf_oos`).
4. `WFE = cagr_oos_stitched / mean_k(cagr_is_k)` (None if denominator ≤ 0); `profitable_oos_share`; per-param coefficient of variation across windows.
5. Result persisted as runs (`wf_is:k`, `wf_oos:k`) + `walkforward.json` artifact under the validation experiment.

**Ordering.** Walk-forward runs *after* evolution, on promoted candidates only.
Evolution's inner walk-forward (§13.6) lives entirely inside train and answers a
different question — "does this candidate generalise at all?" — cheaply enough to
run every generation. The walk-forward here spans train and validation, is far
more expensive, and answers "does the *search procedure* still work when it is
re-run through time?" Neither replaces the other.

---

## 16. Paper trading architecture

### 16.1 Ports

```python
class MarketDataFeed(Protocol):           # live, closed bars only
    async def bars(self, symbol: str, timeframe: str) -> AsyncIterator[Bar]: ...
    async def backfill(self, symbol, timeframe, since_ts: int, limit: int) -> list[Bar]: ...
class Broker(Protocol):                   # the ONLY implementation is PaperBroker (INV-1)
    def submit(self, order: Order) -> None; def on_bar_open(self, bar: Bar) -> list[Fill]
    def position(self) -> Position; def equity(self, mark_price: float) -> float
class Clock(Protocol): def now_ms(self) -> int; async def sleep(self, s: float) -> None
```

### 16.2 Runtime (`paper/runtime.py`)

- One `asyncio` loop per session. On start: load strategy + params from `source_run_id`; `backfill` ≥ `warmup_bars + 10` bars via REST; replay them through the strategy with a `PaperBroker` in "warm" mode (no fills) to rebuild indicator state; then consume the live stream.
- On each closed bar: `broker.on_bar_open(bar)` fills the pending order at `bar.open` (with the same fee/slippage model as backtests); mark to market; `strategy.on_bar(ctx)`; translate to order exactly as §8.4; persist `PaperState` (JSON: last_bar_ts, position, cash, pending order, indicator cache digest) atomically (write temp + rename).
- On restart: resume from state; backfill the missing bars; **replay** them (fills included) so the session is identical to an uninterrupted one.
- Divergence monitor: compare running drawdown with `mdd_p95` from the strategy's verdict artifacts; on breach log ERROR and send a macOS notification via `osascript -e 'display notification …'` (best-effort, never fatal).
- Startup guard: if any configured exchange key is present, call `ccxt.binance().fetch_status()/sapi_get_account_apirestrictions()` and **abort** if `enableSpotAndMarginTrading` is true. If no key is present, proceed (public stream).
- Replay integration test: feed a recorded stream through the runtime and assert trades equal `SimpleBarEngine` trades on the same bars (bit-for-bit on qty and prices).

### 16.3 CLI

`quantlab paper start --run <run_id> [--session-name x]` · `quantlab paper stop <session_id>` · `quantlab paper status` · `quantlab paper report <session_id>` · `quantlab paper install-launchd` (renders the plist template into `~/Library/LaunchAgents/`).

---

## 17. Logging

- `structlog` configured in `core/logging.py`; JSON lines to stderr and to `artifacts/logs/quantlab-YYYY-MM-DD.jsonl`; level from config.
- Every log event carries `run_id`/`experiment_id`/`strategy_id`/`session_id` where applicable (bound via `structlog.contextvars`).
- **Redaction processor:** any key in `logging.redact_keys` (case-insensitive substring match) is replaced by `"***"`; any string value matching `(?i)(sk|key|token)[-_]?[a-z0-9]{16,}` is replaced. Unit-tested.
- Engine and sandbox emit structured events: `order_created`, `order_dropped(reason)`, `fill`, `trade_closed`, `warmup_done`, `sandbox_start/stop(exit_code, cpu_s, rss_mb)`.
- Never log full prompts/responses at INFO (they go to artifacts); log their hashes and token counts.

## 18. Error handling

### 18.1 Exception hierarchy (`core/errors.py`)

```
QuantLabError
├── ConfigError
├── DataError ── DataValidationError, DataGapError, ManifestMismatchError
├── LockboxViolation            (INV-5)   – never caught except at CLI top level
├── StrategyError ── StrategyLoadError, StrategySafetyError(violations), StrategyRuntimeError, LookaheadError (INV-3)
├── LeakageDetected             (hard reject, carries ProbeResult)
├── SandboxError ── SandboxTimeout, SandboxResourceLimit, SandboxProtocolError
├── EngineError                  (accounting invariant broken – always a bug, never swallowed)
├── StoreError ── RunConflict, ImmutableRowError
├── LLMError ── LLMTransportError, LLMMalformedResponse, BudgetExceeded
└── ValidationError_ (renamed to avoid pydantic clash) ── GateFailure, VerdictError
```

### 18.2 Policy

- **Fail closed.** Any exception in data validation, hashing, sandbox, redaction, or lockbox checks aborts the operation. No "best-effort" fallbacks in those paths.
- Strategy exceptions inside a backtest mark the run `failed` with `error_json = {type, message, traceback_tail}`. During evolution the candidate is scored `FITNESS_REJECTED` with `candidate.gate_failure` set and stays in the record; it is never silently dropped, because a generation whose failures are invisible cannot be audited. For an `opaque` LLM-authored candidate the message may be fed back once as a repair request (max `research.max_repair_attempts`), after which the version is marked `rejected`.
- `EngineError` (e.g. equity ≠ cash + position value beyond 1e-6) MUST propagate and fail the whole command; it indicates a bug in the engine, not in the strategy.
- LLM transport errors are retried per §12.2; after retries the iteration is recorded with `status='error'` and the loop continues with the next iteration (counted against `max_iterations`).
- Every CLI command catches `QuantLabError` at the top, prints a one-line human message + `error_id`, logs the traceback, and exits with a distinct non-zero code per family (Config=2, Data=3, Strategy=4, Sandbox=5, LLM=6, Lockbox=7, Engine=8, other=1).
- Partial results are never persisted as `ok`. Writes to the store happen in a transaction per run.

## 19. Unit tests (catalogue — each MUST exist by the end of its phase)

| File | Covers |
|---|---|
| `tests/unit/test_architecture.py` | INV-1, INV-4 (no exec/eval outside sandbox), INV-8 import boundaries; no `LiveBroker`; no `create_order` |
| `tests/unit/test_no_secrets.py` | INV-2 regex scan; `.env` not tracked |
| `tests/unit/test_config.py` | defaults load; `extra=forbid`; env override; `config_hash` stable under key reordering; forbidden objective raises |
| `tests/unit/test_hashing.py` | canonical JSON; id lengths; dataset/split/run id determinism |
| `tests/unit/test_logging.py` | redaction of keys and token-like values |
| `tests/unit/test_data_validation.py` | each violation type; gap fill ≤3; gap > 3 raises; `--allow-gaps` |
| `tests/unit/test_binance_archive.py` | zip parsing on bundled sample; checksum mismatch aborts; idempotent rebuild |
| `tests/unit/test_ccxt_rest.py` | pagination; overlap de-dup (mocked ccxt) |
| `tests/unit/test_splits.py` | ranges resolve to bar indices; embargo respected; WF windows rolling/anchored |
| `tests/unit/test_costs.py` | fixed/vol/impact values; stress multipliers |
| `tests/unit/test_metrics.py` | hand-computed fixture (10-trade toy equity) for all 11 + extras; None cases; CI formula |
| `tests/unit/test_strategy_api.py` | ParamSpec validation; `BarWindow` raises `LookaheadError` beyond `i`; immutability |
| `tests/unit/test_indicators.py` | equality with pandas reference for each indicator |
| `tests/unit/test_engine.py` | next-open fill; slippage direction; fee math; min-notional drop; gap-bar deferral; reversal = two fills; end-of-data close; shorts disabled by default |
| `tests/unit/test_engine_edge_cases.py` | ruin (§8.7); degenerate frames; lot step larger than the position; extreme equities and prices; a 100 % fee; log bound |
| `tests/unit/test_engine_lookahead.py` | INV-3 for the engine: window width, truncation probe, reversed tail, costs and sizing reading only past bars |
| `tests/unit/test_ast_check.py` | 20 malicious/invalid snippets rejected with correct violation codes; 5 valid pass |
| `tests/unit/test_sandbox.py` | timeout; CPU limit; memory limit; forbidden import at runtime; socket blocked; result round-trip identical to an in-process run; the child imports no adapter other than the engine named in the request, which must live under `quantlab.adapters.engine` (amended in 1.1.5, ADR 0004); `engine_module`/`engine_class` validation on construction, on parse and at the point of use; guard-ordering invariant |
| `tests/unit/test_loader.py` | code hash; immutable copy; lineage; tamper detection |
| `tests/unit/test_store.py` | CRUD; immutability rules; migration from empty; transactions |
| `tests/unit/test_runner.py` | cache hit (engine spy not called); `force`; failed → re-run |
| `tests/unit/test_genome.py` | genome validity rules; unbounded param rejected; unused param rejected; warmup below max lookback rejected |
| `tests/unit/test_compiler.py` | deterministic byte-identical source; output passes `ast_check`; `genome_id` stability |
| `tests/unit/test_fitness.py` | each gate; each component at 0/mid/1; weights sum to 1; win rate cannot rescue negative expectancy or excessive drawdown; penalties multiply |
| `tests/unit/test_concentration.py` | trade-removal retention hand-computed on a toy trade list; `k > n_trades`; all-winners case |
| `tests/unit/test_mutation.py` | every operator; bounds respected; invalid child redrawn then falls back to an immigrant; `opaque` candidates get parameter mutations only; same seed ⇒ same child |
| `tests/unit/test_diversity.py` | similarity 1.0 for identical, ~0 for unrelated; niching skips a near-duplicate survivor; immigrant boost fires below the floor |
| `tests/unit/test_population.py` | ranking and tie-breaks; survivor count; `survivors + offspring + immigrants == population_size`; misconfiguration raises `ConfigError` |
| `tests/unit/test_lineage.py` | INV-10: parent chain reaches a seed, no cycles, `replay()` reproduces every stored genome |
| `tests/unit/test_evolution_segments.py` | INV-9: no run outside train/inner folds; a validation run without a prior `candidate_promotion` row raises |
| `tests/unit/test_evolution_loop.py` | generation bookkeeping; cache hits for survivors; stopping rules; failed candidate scored `FITNESS_REJECTED`, not dropped |
| `tests/unit/test_risk_exits.py` | stop-loss, take-profit, trailing and time stop fill prices; stop-loss wins a same-bar tie; no exit on a gap-filled bar; short mirror |
| `tests/unit/test_objectives.py` | guards; forbidden objectives |
| `tests/unit/test_study.py` | seed determinism; segment assertion; n_trials persisted |
| `tests/unit/test_plateau.py` | spike-vs-plateau synthetic objective |
| `tests/unit/test_walkforward.py` | window generation; stitching; WFE |
| `tests/unit/test_gates.py` | each gate pass/fail |
| `tests/unit/test_deflated_sharpe.py` | reference example; DSR=0.5 at SR=SR₀; monotone in M |
| `tests/unit/test_pbo.py` | noise ⇒ ≈0.5; strong edge ⇒ <0.1 |
| `tests/unit/test_permutation.py` | p-value calibration on random-entry; block bootstrap preserves marginal moments |
| `tests/unit/test_score.py` | scoring table; verdict thresholds; thresholds snapshot |
| `tests/unit/test_redaction.py` | INV-6 |
| `tests/unit/test_lockbox.py` | INV-5: research profile refuses test range; caps; family closure; access logged before run |
| `tests/unit/test_llm_glm.py` | request shape (respx); retries; malformed → retry once; budget cap |
| `tests/unit/test_llm_mock.py` | replay order; exhaustion error |
| `tests/unit/test_schemas.py` | proposal validation; code size limits |
| `tests/unit/test_paper_broker.py` | fills identical to engine for same bars |
| `tests/unit/test_paper_state.py` | atomic save; resume |
| `tests/unit/test_binance_ws.py` | recorded stream; unclosed bars filtered; reconnect backfill |

Property tests (`tests/property/`, hypothesis): indicators causal (append-invariance); metric invariants; engine conservation `abs(equity − (cash + qty·close)) < 1e-6` every bar; slippage ≥ 0; splits never overlap; canonical JSON idempotent; **fitness ∈ [0,1] ∪ {FITNESS_REJECTED} for any metric set**; **every mutation of a valid genome is valid or is rejected, never silently invalid**; **similarity is symmetric, in [0,1], and 1.0 only for behaviourally identical candidates**; **retention_k is monotonically non-increasing in k**.

Golden tests (`tests/golden/`): baselines on `btcusdt_1h_2023-01_02.parquet` ⇒ `trades.parquet` compared exactly value-by-value and `metrics.json` compared byte-for-byte (Parquet written with fixed `pyarrow` options, metrics JSON canonical with full float precision). The fixture's own SHA-256 is pinned in the test, because a changed byte there changes every golden.

Five cases are recorded — `buy_and_hold`, `sma_cross`, `rsi_reversion`, `random_entry` and `sma_cross_stopped` — and between them they cover all three `exit_reason` values, so a change to the §8.6 risk exits cannot pass unnoticed. Regenerate only with `scripts/make_fixtures.py golden`, and only alongside an `engine_version` bump: `test_every_golden_records_the_current_engine_version` fails on a bump without regeneration, and the value comparisons fail on a regeneration without a bump.

Leakage tests (`tests/leakage/`): all leaky fixtures detected; all honest fixtures pass.

Synthetic tests (`tests/synthetic/`): random walk ⇒ each baseline's mean net return over 50 seeds within `[−3·costs, +costs]`; trend series ⇒ `sma_cross` positive; OU series ⇒ `rsi_reversion` positive.

Coverage: `core/`, `sandbox/`, `research/redaction.py`, `cli/lockbox.py` ≥ 90 %; overall ≥ 75 %. `make check` fails below these.

## 20. Integration tests (`tests/integration/`)

| Test | Scenario |
|---|---|
| `test_data_pipeline.py` | sample zip → Parquet → manifest → validate → `dataset_id` stable |
| `test_backtest_cli.py` | `quantlab backtest` on fixture → run row, artifacts, report |
| `test_reproduce.py` | INV-7 on golden runs, and on a run executed in a fresh process |
| `test_optimize_walkforward.py` | `sma_cross` end-to-end with `n_trials=10`, 3 WF windows |
| `test_validate_pipeline.py` | over-parameterised curve-fit fixture ⇒ REJECT; `buy_and_hold` ⇒ not CANDIDATE; honest trend strategy on synthetic trend ⇒ gates pass |
| `test_evolution_end_to_end.py` | 5 generations × population 8 on the synthetic-trend fixture: population size held exactly, lineage complete, diversity above the floor, `n_evaluations` recorded |
| `test_evolution_replay.py` | INV-7: an evolution replayed from its stored seed and config reproduces the identical `candidate_id` sequence |
| `test_evolution_resume.py` | kill mid-generation, `evolve resume`, no duplicate runs and no lost candidates |
| `test_evolution_promotion.py` | INV-9: validation touched only via promotion; `candidate_promotion` written before the run; `validation_touches` incremented |
| `test_research_loop_mock.py` | `MockLLMProvider` seeds generation 0 and supplies immigrants and guided mutations; budget stop; malformed proposal rejected before it costs a backtest |
| `test_information_barrier.py` | INV-11: an evolution run against a store containing validation- and test-segment runs ⇒ no validation- or test-derived value in any prompt file written while the run was active |
| `test_paper_replay.py` | recorded stream through paper runtime ≡ engine trades |
| `test_paper_resume.py` | kill mid-session, restart, no missing bars, identical state |

Integration tests MUST be runnable offline (`MockLLMProvider`, recorded streams, bundled zip). A separate marker `@pytest.mark.live` exists for the one manual GLM smoke test and is excluded from `make check`.

## 21. Security requirements

1. **Secrets** (`ports/secrets.py`): `Secrets.get(name) -> str` resolves in order: macOS Keychain (`keyring`, service `quantlab`, username = name) → `.env` in repo root (git-ignored) → `ConfigError("missing secret <name>; run quantlab doctor")`. Secrets are read in `container.py` only and passed to adapters as constructor args; never as CLI flags, never in config files, never logged, never in artifacts. Names: `GLM_API_KEY`, optional `BINANCE_API_KEY`/`BINANCE_API_SECRET` (read-only).
2. **No trade permissions:** `quantlab doctor` and the paper runtime query Binance API restrictions when a key exists and refuse to proceed if trading is enabled. Documented in `SECURITY.md`.
3. **Sandbox** (`sandbox/runner.py`): child process via `subprocess.run([sys.executable, "-I", "-m", "quantlab.sandbox.child_main", …])` with: `resource.setrlimit` for CPU (`RLIMIT_CPU`), address space (`RLIMIT_AS`) and `RLIMIT_NPROC=0`(where supported); wall-clock timeout with kill; env scrubbed to `PATH` + `PYTHONHASHSEED=0`; cwd = fresh temp dir; inputs via a Parquet + JSON file, outputs via Parquet/JSON (never pickle); `child_main` installs an import hook that raises on any module outside `allowed_imports` + the stdlib subset, and replaces `socket.socket` with a raiser before loading the strategy. On macOS, if `sandbox-exec` is available, wrap with a deny-network profile (best effort; the import hook is the primary control).

   **Engine resolution** (added in 1.1.5; ADR `docs/DECISIONS/0004`). The child runs
   the *engine* as well as the strategy: a `bar_loop` strategy receives live engine
   state (`ctx.position`, `ctx.equity`) on every bar, so the two cannot be split
   across processes without an IPC round trip per bar. The sandbox layer may not
   import an adapter (INV-8), so the engine's name travels in `SandboxRequest` as
   `engine_module` / `engine_class`, supplied by the caller. This makes the name the
   most valuable thing for an attacker to control, and it MUST be constrained:

   - `engine_module` MUST start with `quantlab.adapters.engine.` — the prefix checked
     *with* its trailing dot, so `…engineering` cannot pass as `…engine` — and every
     remaining segment MUST be a public identifier;
   - `engine_class` MUST be a single public identifier;
   - both MUST be re-validated in the child at the point of use, so a request built
     by a route that skips validation still cannot redirect the import;
   - the resolved object MUST be a class, and the instance MUST satisfy the engine
     contract structurally (a callable `run`, a string `name`); `quantlab.ports` is
     off-limits to this layer, so the check is by shape, not by `isinstance`.

   The engine MUST be resolved **before** the guards are installed — it is trusted
   infrastructure, and the guards restrict the strategy — and the strategy MUST be
   loaded **after** them. The ordering is: read request → resolve engine → AST check
   → `block_network()` → `install_import_guard()` → execute strategy.

   **Import-guard scope.** The guard enforces `allowed_imports` for imports whose
   *caller* is the strategy's own module, identified by `__name__` in the calling
   frame's globals. A blanket rule breaks the libraries the strategy is permitted to
   use (pydantic reaching for `pydantic_core` mid-validation is not an escape
   attempt), and a sandbox that must be switched off to get work done protects
   nothing. A strategy cannot forge its way out: `__import__`, `exec`, `eval` and
   `globals` are rejected by §9.2 and removed from the namespace it executes in, and
   a function it defines carries its own module globals wherever it is called from.
   `FORBIDDEN_MODULES` (§9.2) is a floor under `allowed_imports` at run time as well
   as at parse time: the allow-list widens what is permitted, never re-opens what is
   banned.
4. **Untrusted text**: LLM output is only ever parsed by pydantic; strategy filenames are derived from hashes; prompts embed LLM text only inside clearly delimited blocks.
5. **Supply chain**: `uv.lock` committed; `pip-audit` in CI; no dependency added without ADR.
6. **Data integrity**: manifest hash verified before every load; mismatch ⇒ `ManifestMismatchError`.
7. **Repo hygiene**: `.gitignore` covers `data/`, `artifacts/`, `*.db`, `.env`, `strategies/generated/*` (except `.gitkeep`); pre-commit runs `gitleaks`, `ruff`, `mypy`.

---

## 22. Incremental phase plan with acceptance criteria

Each step = one Claude Code operation following §0.1. "AC" = acceptance criteria; all must be demonstrated with command output in the step report.

**Task identifiers are stable across spec versions.** Spec 1.1 keeps T01–T44 and
marks each as unchanged, **AMENDED** (same task, extra scope) or **SUPERSEDED**
(replaced by a task in phase F′). New work is T45–T56. The table below is the
authoritative map; the phase sections that follow carry the detail.

| task | status in 1.1 | what changes |
|---|---|---|
| T01 | unchanged | — |
| T02 | unchanged | — |
| T03 | AMENDED | `test_architecture.py` also enforces INV-9/10/11 boundaries once phase F′ lands |
| T04 | AMENDED — **done** | `configs/default.yaml` + `core/config.py` carry the `evolution:` section, `optimize.engine`, `walkforward.evolution_generations` and the reshaped `research:` section. Weights sum to 1; `n_survivors + n_offspring + n_immigrants == population_size`; `diversity.max_immigrants <= n_offspring + n_immigrants`. Schema only — no optimiser behaviour. |
| T05 | AMENDED — **done** | `core/types.py` carries `RiskSpec`, `SizingSpec`, `SlippageConfig`, `BacktestConfig`, `BacktestResult`; the control that fired is recorded in the engine log |
| T06–T09 | unchanged | — |
| T10 | **done** | `core/costs.py` |
| T11 | AMENDED — **done** | `MetricSet` carries `consistency` and `retention_1/3/5`; `docs/METRICS.md` written |
| T12 | AMENDED — **done** (AST part is T17) | `Strategy` carries `risk` and `sizing`; `ParamSpec`, `Context`, `IndicatorCache` implemented |
| T13 | **done** | `core/indicators.py`, all 14 indicators, causality proved per bar |
| T14 | AMENDED — **done** (goldens are T15) | `adapters/engine/simple_bar.py` at `engine_version = "1"`, including §8.6 risk exits and §8.7 ruin |
| T15 | AMENDED — **done** | `strategies/TEMPLATE.py`, four baselines with explicit `RiskSpec()`, `scripts/make_fixtures.py`, the real 1 416-bar 2023-01/02 Parquet fixture (checksum-verified from the Binance archive) and five golden cases including one that exercises the §8.6 risk exits |
| T16 | **done** | `tests/synthetic/` — no edge in noise, trend and reversion behave as expected |
| T17 | AMENDED — **done** | `sandbox/ast_check.py`; 29 rejected and 5 accepted fixtures, the accepted set including compiler output and a breakout that legitimately reads the bar's high; every violation code exercised |
| T18 | AMENDED — **done** | `sandbox/{protocol,guards,runner,child_main}.py`; the child runs the engine, named in the request and constrained to `quantlab.adapters.engine.*` (ADR 0004) |
| T19 | AMENDED — **done** | `core/validation/leakage.py`, four leaky and four honest fixtures, `tests/leakage/`; three tail replacements rather than one, and stated comparison horizons (1.1.7) |
| T20 | AMENDED — **done for source-loaded strategies** | `strategies_io/loader.py`, `strategies_io/probe.py`, `SourceStore` port + `FileSourceStore`, `ExperimentStore.get_strategy_version`. Vectorised strategies are probed at load through the sandbox (§9.1, T19). **Deferred:** `kind='genome'` and `genome_json` (no column before migration `0002`; needs T45/T46) |
| T21 | AMENDED — **done** | `adapters/store/{models,sqlite,artifacts}.py`, Alembic `0001` and `0002`, `ports/store.py` grown to §11.1 including the evolution methods; the five evolution tables, their append-only rules and lineage queries |
| T22 | AMENDED — **done** | `experiments/{runner,env}.py`, `cli/backtest.py`, `strategies_io/evaluators.py`, `ports/engine.StrategyEvaluator`; `force` verifies rather than overwrites (1.1.9) |
| T23 | unchanged | — |
| T24 | SUPERSEDED by T47/T51 | Optuna demoted to refinement (§13.8); objectives retained |
| T25 | AMENDED | plateau selection becomes **mandatory** before validation |
| T26 | AMENDED | walk-forward re-search uses evolution when `optimize.engine == evolution` |
| T27–T30 | unchanged | — |
| T31 | AMENDED | soft checks 10 and 11 added; DSR `M` includes `evolution_run.n_evaluations` |
| T32 | AMENDED | lockbox unchanged, but `test_lockbox.py` also asserts evolution cannot reach test |
| T33 | AMENDED | LLM returns `GenomeProposal`, not raw code |
| T34 | AMENDED | redaction adds the INV-11 generation-report mode |
| T35 | AMENDED | prompts become `propose_genome` / `review_generation` |
| T36 | SUPERSEDED by T52 | the sequential research loop is replaced by the evolution loop |
| T37 | AMENDED | first real campaign is an evolution campaign |
| T38–T41 | unchanged | — |
| T42 | AMENDED | dashboard gains the lineage tree and per-generation fitness view |
| T43 | AMENDED | docs include `docs/EVOLUTION.md` |
| T44 | unchanged | — |
| T45 | NEW — **done early** | risk controls landed with the engine rather than after it: §8.6 has no meaning without §8.4, and splitting them would have shipped an engine whose `Trade.exit_reason` could never be `"stop"` |
| T46–T56 | NEW | phase F′, below |

### Phase A — Skeleton

**T01 Bootstrap.** Create tree of §3 for phase A files, `pyproject.toml` (§4), `Makefile`, `.gitignore`, `.env.example`, `README.md`, `docs/ARCHITECTURE.md` (copy of §2), `docs/SECURITY.md` (copy of §21), ADR 0001, `quantlab --version`.
AC: `uv sync --all-extras` then `make check` green with ≥ 1 trivial test; `quantlab --version` prints `0.1.0`.

**T02 Tooling.** ruff/mypy config, pre-commit with gitleaks, `core/logging.py` with redaction, `core/errors.py`, `core/hashing.py`.
AC: `tests/unit/test_logging.py`, `test_hashing.py`, `test_no_secrets.py` pass; committing a scratch file containing `GLM_API_KEY=` followed by 40 random hex characters (generate them at test time; never write a real-looking key into any tracked file, including this spec) is blocked by pre-commit (show output, then delete the scratch file).

**T03 🔒 Architecture test.** `tests/unit/test_architecture.py` per INV-1/4/8.
AC: test passes; temporarily adding `from quantlab.adapters import x` to `core/__init__.py` fails it (show, then revert); adding a class `LiveBroker` anywhere fails it.

**T04 Config & secrets & doctor.** `core/config.py` (all keys of §5, `extra=forbid`), `ports/secrets.py`, `adapters/secrets/{keychain,dotenv}.py`, `container.py` skeleton, `cli/doctor.py`.
AC: `test_config.py` passes; `quantlab doctor` with no secrets prints a clear missing-key list and exit code 2; with `.env` present prints OK; `config_hash` identical for reordered YAML.

### Phase B — Data

**T05 Core types & ports.** `core/types.py`, `ports/{clock,data,store}.py`, `BarFrame`.
AC: mypy strict on `core/` + `ports/`; frozen dataclasses raise on mutation (test).

**T06 Binance archive ingestion.** `adapters/data/binance_archive.py`, `cli/data.py pull`, bundled sample zip in `tests/fixtures/data/`.
AC: `test_binance_archive.py`; `quantlab data pull --from 2023-01 --to 2023-02` (network) produces yearly Parquet + manifest; second run rebuilds identical hashes.

**T07 Validation & gaps & dataset_id.** `core/data_validation.py`, `quantlab data validate/info`.
AC: `test_data_validation.py`; a fixture with a 2-bar gap validates with `is_gap_filled` rows; a 10-bar gap raises; `dataset_id` printed and stable.

**T08 ccxt tail updates.** `adapters/data/ccxt_rest.py`, `quantlab data update`.
AC: `test_ccxt_rest.py` (mocked); manifest updated; overlaps de-duplicated.

**T09 Splits.** `core/splits.py`, `configs/splits/btcusdt_1h.yaml`, `SplitPolicy` model and `split_id`.
AC: `test_splits.py` + property test; printing the policy shows bar counts per segment and 20–24 rolling WF windows for the default config.

### Phase C — Engine & metrics

**T10 Costs.** `core/costs.py`. AC: `test_costs.py` + slippage ≥ 0 property.
**T11 Metrics.** `core/metrics.py`, `docs/METRICS.md`. AC: `test_metrics.py` toy fixture matches to 1e-12 for all fields; property invariants.
**T12 Strategy API.** `core/strategy.py` (`ParamSpec`, `Context`, `BarWindow`, `Signal`). AC: `test_strategy_api.py`; `BarWindow[i+1]` raises `LookaheadError`.
**T13 Indicators.** `core/indicators.py`. AC: `test_indicators.py` equality vs pandas; causal property test.
**T14 🔒 SimpleBarEngine.** `ports/engine.py`, `adapters/engine/simple_bar.py`. AC: `test_engine.py` all cases; conservation property; explicit test "signal at bar t fills at open of t+1".
**T15 Baselines & golden.** `strategies/TEMPLATE.py`, four baselines, `scripts/make_fixtures.py`, real 2-month fixture, golden files. AC: golden test passes twice in a row and after `git clean`; `engine_version` bump without golden update fails the test (show, revert).
**T16 Synthetic tests.** generators + `tests/synthetic/`. AC: all three synthetic assertions pass with documented tolerances.

### Phase D — Sandbox & leakage

**T17 🔒 AST checker.** `sandbox/ast_check.py` + malicious fixtures. AC: `test_ast_check.py` 20/20 rejected, 5/5 accepted, violation codes listed. — **done**: 29/29 rejected (each asserted against its *exact* code set), 5/5 accepted, plus the shipped template and four baselines; codes tabulated in §9.2.
**T18 🔒 Sandbox runner.** `sandbox/runner.py`, `child_main.py`. AC: `test_sandbox.py` (timeout, memory, socket, import hook, round-trip); `test_architecture.py` still green. — **done**: also `sandbox/protocol.py` (the Parquet/JSON wire format) and `sandbox/guards.py` (import guard, network block); the sandboxed result is equal bar for bar to an in-process run; engine resolution constrained per §21.3 and ADR 0004.
**T19 🔒 Leakage probe.** `core/validation/leakage.py` + leaky/honest fixtures. AC: `tests/leakage/` all detected / all pass; vectorized strategies auto-probed at load. — **done**: 4/4 leak shapes detected and 4/4 honest fixtures plus the four baselines pass, on seven independent bar series; the automatic probe at load is wired in the loader (T20).
**T20 Loader.** `strategies_io/loader.py`. AC: `test_loader.py`; tampering a stored file is detected. — **done**: check → probe → hash → copy → register, in that order, so a file that fails any of them is never registered; `strategy_id` comes from `core.hashing`, never re-derived; tamper detection compares both the recorded `code_sha256` and the id the bytes hash to. A `vectorized` strategy clears the §14.2 truncation probe, run in the sandbox, before it can be registered — and is refused outright if the loader is not configured to run it.

### Phase E — Store & CLI

**T21 SQLite store.** `adapters/store/{models,sqlite,artifacts}.py`, Alembic `0001`. AC: `test_store.py`; `alembic upgrade head` on empty DB; immutability enforced. — **done**, including the amendment: migration `0002` adds `evolution_run`, `generation`, `candidate`, `mutation` and `candidate_promotion`; `mutation` is deliberately absent from `MUTABLE_COLUMNS` because it is the record INV-10 replays; a promotion is written before the run it authorises (INV-9); `n_evaluations` is kept by the store because §14.4's `M` counts cached evaluations too.
**T22 Runner & caching.** `experiments/runner.py`, `cli/backtest.py`. AC: `test_runner.py`; running the same backtest twice shows `cache_hit=true` in logs; `--force` re-runs. — **done**: also `experiments/env.py` (§11.3's `env.json`), `strategies_io/evaluators.py` (the sandboxed `StrategyEvaluator`) and `ports/engine.StrategyEvaluator`. The run id is `core.hashing.run_id`, never re-derived; a failed run is re-executed, a successful one is verified rather than overwritten (1.1.9).
**T23 Reports & reproduce.** `reporting/`, `cli/report.py`, `quantlab reproduce`. AC: `test_reproduce.py` (INV-7) passes for golden runs; report Markdown + PNG created offline.

### Phase F′ — Evolutionary optimisation (replaces phase F)

Phase F′ depends on phases C (engine, metrics), D (sandbox, leakage) and E
(store). It must land **before** phase G's `cli/validate.py` is wired to
promotion, and before phase H.

**T45 Risk controls in the engine.** `RiskSpec`, `SizingSpec` in `core/strategy.py`; §8.6 exits in `adapters/engine/simple_bar.py`; `engine_version` bump.
AC: `test_risk_exits.py` — every control, the same-bar stop-loss tie, the gap-filled deferral, the short mirror; goldens regenerated in the same commit; conservation property still holds.

**T46 Genome and compiler.** `core/genome.py`, `evolution/compiler.py`.
AC: `test_genome.py`, `test_compiler.py`; compiling the same genome twice is byte-identical; compiled output passes `ast_check` and the leakage probe; every `strategies/baselines/*` has an equivalent genome that compiles to a behaviourally identical strategy (equal `position_frac` on the fixture).

**T47 Fitness.** `core/fitness.py`, `core/validation/concentration.py`; `MetricSet` gains `consistency` and `retention_*`.
AC: `test_fitness.py`, `test_concentration.py`; hand-computed toy example matches to 1e-12; a 90 %-win-rate strategy with negative expectancy scores `FITNESS_REJECTED`; a strategy with a 60 % drawdown scores `FITNESS_REJECTED` regardless of every other component; weights not summing to 1 raise `ConfigError`.

**T48 Mutation operators.** `evolution/mutation.py`.
AC: `test_mutation.py`; 20 seeded mutations of a valid genome are all valid; a deliberately impossible operator falls back to an immigrant after `max_repair_attempts`; identical seeds produce identical children; `opaque` candidates never receive a structural mutation.

**T49 Diversity.** `evolution/diversity.py`.
AC: `test_diversity.py`; similarity is 1.0 for a candidate against itself and below 0.2 for two unrelated baselines; niching rejects a duplicate survivor; the immigrant boost fires below the floor and stops above it.

**T50 Population and lineage persistence.** `evolution/population.py`, `evolution/lineage.py`, Alembic `0002`, store methods.
AC: `test_population.py`, `test_lineage.py`; `alembic upgrade head` on an 0001 database; INV-10 holds over a stored 5-generation run; `ancestry()` of a generation-4 candidate returns 5 rows ending at a seed.

**T51 Parameter refinement.** `optimize/{objectives,study,plateau}.py`, `cli/optimize.py` — the retained T24/T25 content, scoped to promoted candidates.
AC: `test_objectives.py`, `test_study.py`, `test_plateau.py`; `objective: net_return` raises `ConfigError`; the segment assertion still refuses `val`/`test`.

**T52 🔒 The evolution loop.** `evolution/loop.py`, `cli/evolve.py`.
AC: `test_evolution_loop.py`, `test_evolution_segments.py`, `test_evolution_end_to_end.py`, `test_evolution_replay.py`, `test_evolution_resume.py`; a 5-generation run on the synthetic-trend fixture holds population at exactly 16; INV-9 demonstrated to fail when the segment assertion is removed (show, then revert); `n_evaluations` matches the number of distinct runs created.

**T53 Promotion and validation wiring.** `evolve promote`, `candidate_promotion`, `validation_touches`.
AC: `test_evolution_promotion.py`; a promotion row exists before its run; a validation run created without one raises; promoting three candidates increments touches by three.

**T54 Walk-forward after evolution.** `walkforward/runner.py`, `cli/walkforward.py`.
AC: `test_walkforward.py`; `test_optimize_walkforward.py` on `sma_cross` completes with stitched metrics and WFE; with `optimize.engine == evolution` and `evolution_generations: 2`, a 3-window walk-forward completes and every window's evaluations are counted into `n_trials_accounted`.

### Phase G — Validation

**T27 Gates.** `core/validation/gates.py`. AC: `test_gates.py` each gate pass/fail.
**T28 Deflated Sharpe.** `deflated_sharpe.py`. AC: reference example reproduced; DSR monotone decreasing in `M`.
**T29 PBO.** `pbo.py`. AC: noise ≈ 0.5 ± 0.1 over 5 seeds; edge < 0.1.
**T30 Permutation.** `permutation.py`. AC: p-values for `random_entry` across 50 seeds pass a KS test against uniform at α=0.01; `mdd_p95` computed.
**T31 Score & verdict & CLI.** `sensitivity.py`, `score.py`, `cli/validate.py`, `docs/VALIDATION.md`. AC: `test_score.py`; `test_validate_pipeline.py` — curve-fit fixture ⇒ REJECT, `buy_and_hold` ⇒ not CANDIDATE, honest synthetic-trend strategy passes gates; family touches incremented; freeze at 20.
**T32 🔒 Lockbox.** `cli/lockbox.py`, `PartitionGuard`, container `lockbox` profile. AC: `test_lockbox.py` (INV-5) — research profile raises `LockboxViolation` on test range; second evaluation of a family refused; access row written before run; family closed on FAIL.

### Phase H — LLM researcher

**T33 LLM port + Mock + GLM.** `ports/llm.py`, `adapters/llm/{mock,glm}.py`, `configs/llm_pricing.yaml`. AC: `test_llm_glm.py` (respx), `test_llm_mock.py`; budget cap raises `BudgetExceeded` before the request.
**T34 🔒 Redaction.** `research/redaction.py`, `RedactedReport`. AC: `test_redaction.py` (INV-6) including string scan for test-period dates.
**T35 Prompts & schemas.** `research/prompts/*.md`, `research/schemas.py`. AC: `test_schemas.py`; rendered system prompt contains every rejection reason in plain language; template hashes stored.
**T55 (was T36) LLM contribution to evolution.** `research/loop.py` becomes a *proposer* module: seed genomes, immigrants on demand, guided mutation suggestions. `cli/research.py`, `configs/research/example_campaign.yaml`.
AC: `test_research_loop_mock.py` and `test_information_barrier.py` pass; a mock campaign seeds generation 0, supplies two immigrants and three guided mutations; a malformed proposal is rejected by pydantic before any backtest runs; INV-11 demonstrated — no validation-derived value appears in any prompt written during the run.

**T56 Lineage visualisation.** `evolution/lineage.py` export + `dashboard/app.py` view (may land with T42).
AC: a stored 5-generation run renders as a parent → child tree with per-generation best/median fitness and diversity; a grep shows no write calls to the store.
**T37 First real campaign (human-gated).** Only when the human says so: run one GLM-seeded **evolution** campaign with `max_cost_eur: 20`; write `docs/DECISIONS/000N-first-campaign-findings.md`. AC: campaign summary with fitness trajectory, diversity trajectory and verdict distribution for promoted candidates; `SELECT COUNT(*) FROM lockbox_access` = 0; total `cost_eur` ≤ 20; `n_evaluations` reported and reflected in the DSR of every verdict.

### Phase I — Paper trading

**T38 Broker port + PaperBroker.** `ports/broker.py`, `adapters/broker/paper.py`, `Clock`. AC: `test_paper_broker.py` fills ≡ engine fills.
**T39 Websocket feed.** `adapters/data/binance_ws.py`. AC: `test_binance_ws.py` on recorded stream; unclosed bars never emitted; reconnect backfills.
**T40 🔒 Runtime & persistence.** `paper/{runtime,state}.py`, `cli/paper.py`, launchd template, trade-permission startup guard. AC: `test_paper_replay.py`, `test_paper_resume.py`; fake key with trading enabled ⇒ startup aborts with exit code 7.
**T41 Paper campaign (human-gated).** Run ≥ 7 days on a CANDIDATE or `sma_cross`; `quantlab paper report`. AC: no crash; report compares realised drawdown vs `mdd_p95`.

### Phase J — Polish

**T42 Dashboard.** `dashboard/app.py` (read-only). AC: opens against real DB; a grep shows no write calls to the store.
**T43 Backup/restore & docs.** `quantlab backup/restore`; docs complete. AC: fresh clone + restore ⇒ `quantlab reproduce` of a golden run passes; README onboarding executed verbatim by a fresh session.
**T44 Nightly audit.** GitHub Actions (or `launchd`) job: reproduce 3 random runs, `pip-audit`, manifest re-hash. AC: job green; an altered Parquet byte makes it fail.

---

## 23. Global acceptance criteria (Definition of Done for v1)

1. `make check` green on a fresh clone with coverage thresholds met.
2. All eleven invariants of §0.2 have passing tests, and each has been demonstrated to fail when the invariant is broken (recorded in step reports).
3. `quantlab data pull` → `quantlab validate <baseline>` → `quantlab evolve run` (mock LLM) → `quantlab evolve promote` → `quantlab lockbox evaluate` (on a synthetic CANDIDATE) → `quantlab paper start` (replay) all execute from the README instructions alone.
4. The over-parameterised curve-fit fixture is `REJECT`ed and `buy_and_hold` is never `CANDIDATE`; a random-entry strategy never passes `G_PERM` in more than 5 % of seeds.
4b. An evolution run of 10 generations on a **pure random walk** promotes nothing that reaches `CANDIDATE`. If it does, the fitness function or the trial accounting is wrong, and that is the first thing to fix.
4c. Population diversity never falls below `min_population_diversity` for more than two consecutive generations in any recorded run.
5. No metric, threshold, or split has been changed from this spec without an ADR.
6. The repository contains no secret material and no code capable of placing an order on any exchange.
7. Every run in the store reproduces to 1e-9, and every stored evolution run replays to the identical candidate sequence.
8. Every stored genome candidate satisfies INV-10: its recorded mutations, re-applied to its parent, reproduce it exactly.

*End of specification.*
