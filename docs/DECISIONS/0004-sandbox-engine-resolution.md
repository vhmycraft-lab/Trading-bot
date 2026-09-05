# 4. The sandbox child loads the engine by name from the request

* **Status:** Accepted
* **Date:** 2026-09-05
* **Amends:** spec §19 (the `test_sandbox.py` row), §21.3
* **Task:** T18

## Context

INV-4 says untrusted code never executes in the main process. Spec §21.3 gives
the shape of the answer: a child process, resource-limited, talking back in
Parquet and JSON. Two constraints then collide.

The first is the strategy contract (§9.1). A `bar_loop` strategy is handed a live
`Context` on every bar — the current position, cash, equity, and an indicator
cache — and returns a `Signal` the engine acts on. The strategy and the engine are
therefore *interleaved*: there is no point at which the strategy's output can be
computed once and handed to an engine somewhere else. Running them in different
processes would mean an IPC round trip per bar, which for a 16-candidate
population over tens of thousands of bars is not an implementation detail but a
different project.

The second is INV-8. `quantlab.sandbox` sits beside `core` and may import only
`quantlab.core` and itself. `SimpleBarEngine` is an adapter. So the sandbox layer
cannot name the engine it has to run.

Spec §19's test table also asks that the child "never imports
`quantlab.adapters`", which was written on the assumption that the two constraints
did not collide.

## Decision

The child runs the engine, and the engine's name travels **in the request**.

`SandboxRequest` carries `engine_module` and `engine_class`. The caller — which
already holds an engine, and lives in a layer permitted to name one — supplies
them. `child_main` resolves the name with `importlib` **before** it installs any
guard, because the engine is trusted infrastructure and the guards exist to
restrict the strategy, not the platform.

The name is the one thing an attacker would most want to control, so it is
constrained twice and narrowly:

* `engine_module` must begin with `quantlab.adapters.engine.` — the prefix checked
  *with* its trailing dot, so `…engineering` cannot pass as `…engine` — and every
  remaining segment must be a public identifier. No empty segment, no `..`, no
  private module.
* `engine_class` must be a single public identifier.
* Both are re-checked in `child_main._load_engine` at the moment of use, so a
  request built by any route that skips validation (`model_construct`, a later
  refactor) still cannot turn the call into "import whatever I say".
* What the name resolves to must be a class, and the instance must look like an
  engine — a callable `run` and a string `name`. The check is structural because
  `quantlab.ports` is off-limits to this layer; an object that cannot run a
  backtest must not reach the point where untrusted code is already loaded.

The approved namespace is `src/quantlab/adapters/engine/`, which holds
`simple_bar.py` and nothing else. The store, the LLM client, the exchange
adapters, the secrets adapters and `container.py` are all outside it and
unreachable through this mechanism.

## Consequences

* Spec §19's "child never imports `quantlab.adapters`" is amended to "the child
  imports **no adapter other than the engine named in the request**, which must
  live under `quantlab.adapters.engine`". The intent — that the sandbox has no
  route to the database, the exchange, or the model — is preserved exactly, and is
  now enforced by validation rather than by the absence of an import statement.
* The sandbox package itself still imports no adapter and no port. `test_sandbox.py`
  re-states that rule inside the security suite, next to the tests that depend on
  it, in addition to `test_architecture.py` enforcing it tree-wide.
* A future non-adapter engine (one living in `core`) would need this rule widened.
  That is a deliberate cost: widening it should require an argument.

## Alternatives rejected

**Run only the strategy in the child and the engine in the parent.** Satisfies the
spec line verbatim and breaks the strategy contract: `ctx.position` and
`ctx.equity` are engine state, so either the strategy loses them or every bar
costs a round trip.

**Hard-code the engine in `child_main`.** Violates INV-8, and the layering test
would have to be given an exemption — a static import the architecture test is
told to ignore is a weaker guarantee than a validated string, because the
exemption is invisible at the call site.

**Resolve the engine through a registry in `core`.** Moves the adapter import into
`core`, which is worse: `core` may import nothing but itself.
