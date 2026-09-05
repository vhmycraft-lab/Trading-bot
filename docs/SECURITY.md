# Security

Normative source: `CLAUDE_CODE_MASTER_SPEC.md` §21 and the invariants in §0.2.

## 0. The invariant that matters most

**QuantLab cannot place a real order (INV-1).** There is no live-broker adapter,
no exchange order call, and no class named `LiveBroker`, `RealBroker` or
`ExchangeBroker` anywhere in `src/`. `tests/unit/test_architecture.py` scans the
source tree on every `make check` and fails the build if one appears.

## 1. Secrets

`Secrets.get(name) -> str` (`ports/secrets.py`) resolves in this order:

1. macOS Keychain — `keyring`, service `quantlab`, account = the secret name
2. `.env` in the repository root (git-ignored)
3. `ConfigError("missing secret <name>; run quantlab doctor")`

Rules:

* Secrets are read in `container.py` only and passed to adapters as constructor
  arguments.
* Never as CLI flags, never in `configs/`, never in artifacts, never logged.
* Names: `GLM_API_KEY` (required for the researcher), `BINANCE_API_KEY` /
  `BINANCE_API_SECRET` (optional, **read-only**, only to raise rate limits).

```bash
security add-generic-password -s quantlab -a GLM_API_KEY -w   # macOS
cp .env.example .env                                          # or local dev
```

INV-2 is enforced twice: `gitleaks` in pre-commit, and
`tests/unit/test_no_secrets.py`, which regex-scans `src/`, `configs/` and
`tests/` and asserts `.env` is not tracked.

## 2. No trade permissions

`quantlab doctor` and the paper-trading runtime call
`sapi_get_account_apirestrictions()` whenever a Binance key is configured and
**abort** if `enableSpotAndMarginTrading` is true (exit code 7). If no key is
present they proceed against the public stream.

## 3. Sandbox (INV-4)

LLM-generated strategy code never executes in the main process. It runs in a
child process started as
`subprocess.run([sys.executable, "-I", "-m", "quantlab.sandbox.child_main", ...])`
with:

* `RLIMIT_CPU`, `RLIMIT_AS` and (where supported) `RLIMIT_NPROC=0`
* a wall-clock timeout with a hard kill
* the environment scrubbed to `PATH` and `PYTHONHASHSEED=0`
* a fresh temporary working directory
* inputs and outputs as Parquet + JSON — **never** pickle
* an import hook in the child that raises on any module outside
  `sandbox.allowed_imports` plus a small stdlib subset, and a `socket.socket`
  replacement that raises, both installed before the strategy is loaded
* on macOS, a `sandbox-exec` deny-network profile where available (best effort;
  the import hook is the primary control)

`exec`, `eval`, `compile`, `importlib`, `pickle`, `marshal` and `shelve` are
forbidden outside `src/quantlab/sandbox/`, enforced by
`tests/unit/test_architecture.py`.

*Status: the sandbox lands in phase D (T17–T18). Until then no untrusted code is
loaded at all.*

## 4. Untrusted text

LLM output is only ever parsed by pydantic models. Strategy filenames are
derived from content hashes, never from model-supplied text. Prompts embed model
text only inside clearly delimited blocks.

## 5. Supply chain

`uv.lock` is committed and pinned. `pip-audit` runs in CI (`make audit`). No
dependency may be added without an ADR in `docs/DECISIONS/`.

## 6. Data integrity

The dataset manifest hash is verified before every load; a mismatch raises
`ManifestMismatchError` and aborts. Nothing "best-effort" happens in data
validation, hashing, the sandbox, redaction or the lockbox — those paths fail
closed.

## 7. Logging

`core/logging.py` installs a redaction processor:

* any event key containing (case-insensitively) one of
  `api_key`, `secret`, `token`, `password`, `authorization` is replaced by `***`
* any string value matching `(?i)(sk|key|token)[-_]?[a-z0-9]{16,}` is replaced

Full prompts and responses are never logged; only their hashes and token counts.

## 8. Repository hygiene

`.gitignore` covers `data/`, `artifacts/`, `*.db`, `.env` and
`strategies/generated/*` (except `.gitkeep`). Pre-commit runs `gitleaks`, `ruff`
and `mypy`.

`gitleaks` is configured by `.gitleaks.toml`, which extends the stock ruleset
with rules matching the credential shapes this project actually uses — the
default rules alone do not reliably flag `GLM_API_KEY=<hex>`. Those rules mirror
the regexes in `tests/unit/test_no_secrets.py` on purpose: the hook stops a
commit, the test stops a build, and the two agree on what counts as a secret.

`ruff` and `mypy` run as `local` hooks out of the project environment rather
than from pinned mirrors, so `make check` and the hooks can never disagree about
which version enforces which rules.

## Reporting a problem

This is a private research repository. Report anything that looks like a way to
place a real order, exfiltrate a secret, or read the locked test partition by
opening an issue titled `SECURITY` — do not include the secret itself.
