# FitCheck developer tooling
# Run `make help` for the full target list.

SHELL := /bin/bash
PYTHON ?= python3
PIP ?= pip3
VENV ?= .venv

.PHONY: help install dev-install test test-fast lint typecheck security audit doctor \
	bench demo clean clean-reports coverage-html pre-commit install-hooks

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install FitCheck in the current environment (editable)
	$(PIP) install -e .

dev-install: ## Install FitCheck with all development and optional dependencies
	$(PIP) install -e ".[dev,docs,jupyter,plotly,polars,duckdb,mlflow,dvc]"

test: ## Run the full test suite with coverage
	$(PYTHON) -m pytest

test-fast: ## Run the suite without coverage reporting
	$(PYTHON) -m pytest -q --no-cov

lint: ## Run the Ruff linter
	ruff check fitcheck tests benchmarks scripts

format: ## Format code with Ruff
	ruff format fitcheck tests benchmarks scripts
	ruff check --fix fitcheck tests benchmarks scripts

typecheck: ## Run strict mypy type checking
	mypy fitcheck

security: ## Run the Bandit security scanner
	bandit -r fitcheck

audit: ## Audit runtime dependencies for known vulnerabilities
	pip-audit -r requirements-audit.txt

doctor: ## Diagnose the FitCheck environment
	fitcheck doctor

benchmark: ## Run the reproducible benchmark suite and append results
	$(PYTHON) benchmarks/run.py

bench: ## Alias for the reproducible benchmark suite
	$(MAKE) benchmark

bench-only: ## Legacy bench target
	$(PYTHON) benchmarks/run.py

bench-smoke: ## Quick one-million-row contact smoke test
	$(PYTHON) benchmarks/contact_smoke_test.py

demo: ## Run the built-in demo (headless)
	fitcheck demo --no-browser

coverage-html: ## Generate an HTML coverage report
	$(PYTHON) -m pytest -q --cov=fitcheck --cov-report=html:htmlcov --no-cov 2>/dev/null || \
	$(PYTHON) -m pytest -q --cov=fitcheck --cov-report=html:htmlcov

pre-commit: ## Run all pre-commit hooks on the entire repository
	pre-commit run --all-files

install-hooks: ## Install the pre-commit hooks for this repository
	pre-commit install

clean: ## Remove build, cache, and coverage artifacts
	rm -rf build dist *.egg-info htmlcov .pytest_cache .coverage \
		.mypy_cache .ruff_cache .mutmut-cache fitcheck_reports *.html

clean-reports: ## Remove generated fitcheck reports only
	rm -rf fitcheck_reports *.html
