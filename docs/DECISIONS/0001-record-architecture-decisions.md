# 1. Record architecture decisions

* **Status:** Accepted
* **Date:** 2026-09-05

## Context

`CLAUDE_CODE_MASTER_SPEC.md` fixes a large number of decisions — thresholds,
split boundaries, cost defaults, the dependency list, the metric definitions.
Section 0.3 forbids changing any of them silently: a change requires a written
justification, because a quietly relaxed threshold is indistinguishable from
curve fitting after the fact.

We need a lightweight, durable way to record such changes next to the code they
affect, readable by both a human reviewer and by future sessions of the agent
implementing this project.

## Decision

We keep Architecture Decision Records in `docs/DECISIONS/`, numbered
sequentially (`NNNN-title-in-kebab-case.md`), in the format described by Michael
Nygard.

An ADR is **required** for any of the following:

1. adding, removing or changing a runtime or development dependency (spec §0.3);
2. changing a validation threshold, split boundary, cost default, or any value
   in `configs/default.yaml` that affects a verdict (spec §0.3, §23.5);
3. changing a metric definition or the engine's fill/accounting rules
   (`engine_version` must be bumped in the same commit);
4. changing an invariant of spec §0.2 or the test that enforces it;
5. any deviation from the file tree, phase order or acceptance criteria of
   spec §3 and §22.

An ADR is **not** required for ordinary implementation work that follows the
spec.

Each record states Context, Decision and Consequences, and is committed together
with the change it justifies. Records are immutable once accepted: a later
decision supersedes an earlier one with a new record and a `Superseded by` line
on the old one.

## Consequences

* Every deviation from the specification is discoverable in one directory, with
  its reasoning, rather than buried in commit messages.
* Reviewers can check a diff that moves a threshold against a written
  justification, which is the point of the rule.
* There is a small overhead on legitimate changes; that overhead is the
  intended friction.

## Deviation recorded by this commit

Phase 1 as requested by the project owner covers spec tasks T01 (bootstrap) plus
the parts of T02 (logging, errors, hashing), T04 (config, secrets, doctor,
container skeleton) and T21 (SQLite schema and Alembic migration) that the
owner's phase brief named explicitly: "configuration management", "logging
infrastructure", "the initial SQLite database layer", "database
migrations/schema", "a basic CLI entry point" and "unit-test infrastructure".

Consequently `alembic.ini`, `migrations/`, `src/quantlab/core/{config,logging,
errors,hashing}.py`, `src/quantlab/ports/secrets.py`,
`src/quantlab/adapters/{secrets,store}/` and `src/quantlab/container.py` exist
earlier than the phase letters in spec §3 indicate. No file marked for phases
B–J beyond that list has been created, and no threshold, split boundary, metric
or cost default has been changed from the specification.
