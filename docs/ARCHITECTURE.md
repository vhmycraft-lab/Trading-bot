# Architecture

Normative source: `CLAUDE_CODE_MASTER_SPEC.md` §2. This document mirrors it and
must be updated in the same commit as any structural change.

## Layers

```
cli/ (Typer)  ──►  container.py (composition root)
                      │ builds
                      ▼
 orchestration: research/  optimize/  walkforward/  paper/  reporting/
                      │ depend only on
                      ▼
 ports/  : BacktestEngine · LLMProvider · MarketDataSource · MarketDataFeed ·
           Broker · ExperimentStore · Clock · Secrets
                      │ implemented by
                      ▼
 adapters/: engine/simple_bar · llm/{glm,mock} · data/{binance_archive,ccxt_rest,binance_ws}
            broker/paper · store/{sqlite,artifacts} · secrets/{keychain,dotenv}
                      │ all of the above use
                      ▼
 core/   : types · config · strategy · indicators · costs · metrics · splits ·
           validation/ · hashing · errors
 sandbox/: ast_check · runner (child-process execution of strategies)
```

## Composition root

`src/quantlab/container.py` exposes exactly one public function:

```python
def build_container(config: AppConfig, *, profile: Literal["research", "lockbox", "paper", "test"]) -> Container
```

`Container` is a frozen dataclass holding one instance per port. The `profile`
decides which adapters are wired:

| profile | wiring |
|---|---|
| `research` | real adapters, but a `MarketDataSource` wrapped in `PartitionGuard` that refuses the test partition |
| `lockbox` | the **only** profile whose data source may load the test partition (INV-5) |
| `paper` | websocket feed + `PaperBroker` |
| `test` | `MockLLMProvider`, in-memory SQLite |

Nothing else in the codebase instantiates adapters.

## Replaceability contract

Each port is a `typing.Protocol` (structural) defined in `ports/`. Adapters MUST
NOT be imported by name anywhere except `container.py`, `cli/` and tests. To
replace a component, add an adapter and one branch in `build_container`.
`tests/unit/test_architecture.py` fails on any other import of
`quantlab.adapters`.

## Import boundaries (INV-8)

| layer | may import |
|---|---|
| `core` | `core` only |
| `ports` | `core` |
| `adapters/*` | `core`, `ports` |
| `research`, `optimize`, `walkforward`, `paper`, `reporting`, `experiments`, `strategies_io` | `core`, `ports` |
| `container.py`, `cli/`, `tests/` | everything |

## Determinism rules

* No `datetime.now()`, `time.time()`, unseeded `random` or global `np.random`
  inside `core/`, `adapters/engine/` or any strategy — clocks and RNGs are
  injected. `core/logging.py` therefore takes the log date as a parameter.
* All ids are SHA-256 over canonical JSON (`core/hashing.py`); the first 16 hex
  characters form an id.
* A run's identity is the full set of its inputs (§11.2), so an identical
  request hits the cache and any changed input produces a new run.

## Current state (Phase 1)

Implemented: `core/{config,errors,hashing,logging}.py`, `ports/secrets.py`,
`adapters/secrets/*`, `adapters/store/{models,sqlite}.py`, `migrations/`,
`container.py`, `cli/`.

Every other directory in the tree is reserved for its phase and is intentionally
empty until then.
