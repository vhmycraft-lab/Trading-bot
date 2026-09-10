# QuantLab — user handbook

Non-normative. `CLAUDE_CODE_MASTER_SPEC.md` is the specification; where the two
disagree, the spec wins and this file is wrong.

Every command below was run against this repository while writing it. If one
does not work, that is a defect in this file — please say so rather than
working around it.

---

## What this is, and what it refuses to be

QuantLab searches for trading strategies, measures them honestly, and runs the
survivors forward on live data **without ever placing an order**. That last part
is structural rather than a setting: there is no live broker class, no order
API in the dependency surface, and no method on the broker port that could
transmit anything (INV-1). There is no flag that changes it.

The harder problem this platform is built around is not finding a strategy that
looks good on historical data. Anything looks good on historical data if you
search long enough — that is what overfitting *is*, and a search of a few
thousand candidates will produce a beautiful equity curve out of pure noise. So
most of the machinery here exists to make that failure visible:

- the data is split into **train / validation / test**, and the test partition
  is behind a one-shot door (§14.6);
- the multiple-testing correction charges you for **every candidate the search
  looked at**, not just the one you kept (§14.4);
- the number of times a family may touch validation is **capped and counted**;
- a strategy that reads its own future is caught by a causality probe rather
  than by inspection (§14.2).

You will spend more time being told no than being told yes. That is the product
working.

---

## Setting up

```bash
make sync
```

That creates the virtualenv from `uv.lock` **with all extras** — which matters,
because `pre-commit` lives in the `dev` extra and a plain `uv sync` will not
install it.

```bash
uv run pre-commit install
```

```bash
make db-upgrade
```

Then confirm the whole thing is healthy:

```bash
make check
```

That runs formatting, linting, type-checking and the test suite with coverage
gates, and it should end with `make check: OK`. If it does not, stop here — a
red tree makes every number below untrustworthy.

### One environment note

Do not keep this repository inside iCloud Drive, Dropbox, or any other syncing
folder. The sync daemon rewrites files under `.venv/` while Python is reading
them, and the specific way it breaks is nasty: `site.addpackage` swallows the
resulting `OSError` silently, so the editable install stops working and you get
`ModuleNotFoundError: No module named 'quantlab'` with nothing in any log
explaining why. Keep it somewhere ordinary, like `~/Developer`.

---

## The shape of a session

```
data  ->  evolve  ->  promote  ->  validate  ->  (lockbox)  ->  paper
```

Left to right, each step is harder to get through than the last, and the
rightmost two are **irreversible in ways worth understanding before you use
them**.

### 1. Get some data

```bash
uv run quantlab data pull
```

```bash
uv run quantlab data update
```

`pull` downloads Binance's monthly archives; `update` extends the dataset to the
latest closed bar over the REST API, because the archives lag by up to a month.

Check what you got, and how it is split:

```bash
uv run quantlab data info
```

```bash
uv run quantlab data splits
```

`splits` is worth reading carefully the first time. It prints the train,
validation and test boundaries, the embargo between them, and the walk-forward
windows. Those boundaries are the entire basis for every claim this platform
will later make.

### 2. Search

```bash
uv run quantlab evolve run
```

This is the long one. A population of strategy candidates is scored on the
**train** segment only, mutated, and re-scored, generation after generation.
Watch it:

```bash
uv run quantlab evolve status <evolution_id>
```

If it dies — power cut, closed laptop, `Ctrl-C` — resume it:

```bash
uv run quantlab evolve resume <evolution_id>
```

Resume is exact. A resumed run produces the identical sequence of candidate
identifiers a run that was never interrupted would have (INV-7, INV-9), and
there is a test that breaks resume on purpose to prove that guarantee is not
vacuous. You will not get a subtly different search wearing the same run id.

To see where a candidate came from:

```bash
uv run quantlab evolve lineage <candidate_id>
```

### 3. Promote, then validate

```bash
uv run quantlab evolve promote <evolution_id>
```

Promotion is what authorises a validation run. Until a candidate is promoted,
nothing in the platform will measure it against the validation segment.

```bash
uv run quantlab validate run <strategy_id>
```

This is where you find out. It runs the hard gates (§14.3), computes the soft
checks that produce an overfitting score (§14.4), and writes a verdict:

| verdict | what it means |
|---|---|
| `REJECT` | It failed a hard gate. Something is wrong with it, not merely unimpressive. |
| `WEAK` | It passed the gates and accumulated too much overfitting evidence. |
| `CANDIDATE` | It survived. This is as good as an answer gets before the lockbox. |

**Every family's validation touches are counted**, and the count is capped. You
cannot re-validate until a favourable verdict falls out — that would be a search
over the validation segment, which is exactly what the budget exists to stop.

#### About the deflated Sharpe ratio

The single most important number in the verdict is the deflated Sharpe, and it
is deflated by **M** — the number of trials your search actually performed:
every evolutionary evaluation, every Optuna trial, every validation touch the
family has spent. A campaign of three generations at population 32 charges
`M = 96`, and on a real curve that is the difference between a Sharpe that
passes and the same Sharpe that does not.

If a verdict in your database predates this correction it carries
`m_formula_version = 'm1-bar-count'`, which means its `M` was computed from the
length of the validation segment rather than from the size of the search. Those
verdicts are **overstated**, they were not deleted (nothing here is), and the
lockbox will refuse to open on one. Re-run `quantlab validate` and use the new
verdict.

### 4. The lockbox — read this part twice

```bash
uv run quantlab-lockbox evaluate <strategy_id> --params <run_id> --reason "<at least 20 characters>"
```

Note the binary: `quantlab-lockbox`, not `quantlab lockbox`. It is a separate
executable so that no research code path can import it, and therefore no
research code path can reach the unguarded data source.

What this command spends, and does not give back:

- **A family gets one look at the test partition. Ever.** `max_per_family = 1`.
- **The platform gets three looks a month**, across all families, on a rolling
  thirty-day window.
- **A failure closes the family permanently.** `LOCKBOX_FAIL` sets
  `family.status = 'closed'`, and there is no command to reopen it.

The written reason is mandatory and is stored. Write it for the version of
yourself who reads it in six months and needs to know what you thought you were
testing.

Do not open the lockbox to see how it goes. That is what the validation segment
is for.

### 5. Paper trading

```bash
uv run quantlab paper status
```

```bash
uv run quantlab paper report <session_id>
```

A paper session fills at the next bar's open with the same fee and slippage
arithmetic the backtest used — literally the same functions, in
`core/execution.py`, which is what makes the replay test able to assert
bit-for-bit equality rather than "close enough".

If a session is interrupted, it backfills the bars it missed and **replays them
with their fills**, so a restarted session holds the position an uninterrupted
one would have held. It does not restore the position and skip ahead; that would
be faster and would silently produce a different account.

---

## Reading the results

```bash
uv run quantlab report show <run_id>
```

```bash
uv run quantlab report reproduce <run_id>
```

`reproduce` re-runs a stored run and compares every metric it recorded (INV-7).
It is the cheapest way to find out whether a number you are about to rely on is
real. Use it before you quote anything to anyone, including yourself.

For a visual view of a whole campaign:

```bash
uv run streamlit run src/quantlab/dashboard/app.py
```

(`uv sync --extra dashboard` first.) It is strictly read-only — there is a test
that parses every module in that package and fails if any of them calls a
write method on the store. A candidate the database holds no fitness for is
shown as **unmeasured**, never as zero: zero is a score, and on a screen an
invented number looks exactly like an earned one.

---

## The LLM, and what it is never shown

The model proposes **genomes as JSON** — never Python. It is a proposer inside
the search, not a driver of it: it does not decide survival, ranking, or when to
stop. Those are deterministic and fitness-driven.

Two invariants bound what it can see:

- **INV-6** — nothing derived from the test partition ever enters a prompt.
  Permanent, unconditional.
- **INV-11** — while an evolution run is active, nothing derived from
  *validation* enters a prompt either. The loop sees training and inner-fold
  numbers only. Validation results reach you, never the model.

Every prompt is built from a whitelist and then scanned before it is sent. The
scan also looks for **credentials and environment values**, because prompts are
written to `artifacts/llm/<interaction_id>/prompt.json` unencrypted and kept —
a report assembled by formatting an exception can carry an API key into a file
that outlives the run.

### Running without an API key

You do not need one. `MockLLMProvider` replays recorded responses, and the whole
evolutionary search runs offline and deterministically against it. There is no
`GLM_API_KEY` in this repository and nothing here will invent one.

If you do configure a real key, put it in the macOS Keychain (service
`quantlab`) or in a git-ignored `.env`. Never on a command line — it lands in
your shell history — and never in a config file.

---

## When something goes wrong

```bash
uv run quantlab doctor
```

Checks the interpreter, the configuration, the directories, the secrets chain
and the database, and tells you which of them is unhappy.

```bash
uv run quantlab db info
```

Shows the database path, its size, its migration revision, and any tables that
are missing.

**`ModuleNotFoundError: No module named 'quantlab'`** — the editable install is
broken. Almost always a syncing folder (see the setup section). `make clean`
then `make sync`.

**`check_coverage: FAILED — the coverage report is not usable as evidence`** —
`coverage.json` is stale, or was measured without branch coverage, or names
files that are not in the tree. Run `make test` again rather than reading the
percentages it printed; the gate is telling you the report is not describing the
code you have.

**A test fails after you changed the engine** — check the golden baselines
first. They record trade prices exactly, and they will catch a change of one bit
in the last decimal place. That is deliberate: it is how the paper broker and
the backtest engine are kept honest about agreeing with each other.

---

## Things that are true and worth remembering

- Nothing in here can place an order, and no configuration changes that.
- Nothing is deleted. Verdicts, runs and lineage are append-only, and a
  correction is written as a new row rather than an edit to an old one.
- A resumed run is the same run. A restarted paper session is the same session.
- The test partition is opened once per family, three times a month, and a
  failure closes the family for good.
- You will mostly be told no.
