# QuantLab developer entry points.  `make check` is the gate every commit must pass.
.DEFAULT_GOAL := help
SHELL := /bin/bash
UV ?= uv
RUN := $(UV) run

.PHONY: help sync fmt fmt-check lint type test cov check audit db-upgrade db-revision clean

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

sync:  ## Create/refresh the virtualenv from pyproject + uv.lock
	$(UV) sync --all-extras

fmt:  ## Format the codebase
	$(RUN) ruff format .
	$(RUN) ruff check --fix-only .

fmt-check:  ## Verify formatting without writing
	$(RUN) ruff format --check .

lint:  ## Lint
	$(RUN) ruff check .

type:  ## Type-check (strict on core/, ports/, sandbox/)
	$(RUN) mypy

test:  ## Run the test suite with coverage gates
	$(RUN) pytest -q -m "not live" --cov --cov-report=term-missing --cov-report=json
	$(RUN) python scripts/check_coverage.py

cov: test  ## Alias for `make test`

check: fmt-check lint type test  ## Full gate: format + lint + types + tests
	@echo "make check: OK"

audit:  ## Audit dependencies for known vulnerabilities
	$(RUN) pip-audit

nightly:  ## Full nightly audit: reproduce runs, re-hash manifests, pip-audit, coverage freshness
	$(RUN) python scripts/nightly_audit.py

db-upgrade:  ## Apply all database migrations
	$(RUN) alembic upgrade head

db-revision:  ## Create a new migration:  make db-revision M="add foo"
	$(RUN) alembic revision -m "$(M)"

clean:  ## Remove caches and coverage output
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis htmlcov \
	       .coverage coverage.json coverage.xml
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
