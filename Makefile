# FedCRG developer interface. The Makefile only forwards to real commands;
# it never carries scientific configuration values. Paths live in
# config/study.yaml, so no target accepts path variables.

PYTHON ?= python3
RUFF ?= ruff
PYRIGHT ?= pyright
PYTEST ?= pytest
NOX ?= nox

DATASET ?=
EXPERIMENT ?= primary_nbaiot

.PHONY: help install format lint typecheck test test-unit test-integration \
	test-contract audit validate preprocess plan run campaign \
	status monitor results verify-results quality

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

install: ## Install the package and dev tooling from the locked dependency graph
	uv sync --locked --extra dev

format: ## Apply ruff formatting
	$(RUFF) format src tests
	$(RUFF) check --fix src tests

lint: ## Ruff lint check
	$(RUFF) check src tests
	$(RUFF) format --check src tests

typecheck: ## Pyright strict typing on src
	$(PYRIGHT) src/fedcrg

test: ## Full pytest suite
	$(PYTEST)

test-unit: ## Unit tests only
	$(PYTEST) tests/unit

test-integration: ## Integration tests only
	$(PYTEST) tests/integration

test-contract: ## Contract tests only
	$(PYTEST) tests/contract

audit: ## Re-audit the repository against the goal matrix
	$(PYTHON) tools/audit_repository.py

validate: ## Validate one resolved experiment configuration
	$(PYTHON) -m fedcrg.cli validate $(EXPERIMENT)

preprocess: ## Preprocess DATASET (default: every raw dataset) into data/preprocessed/
	$(PYTHON) -m fedcrg.cli preprocess $(DATASET)

plan: ## Plan the primary experiment
	$(PYTHON) -m fedcrg.cli plan $(EXPERIMENT)

run: ## Execute one experiment from prepared data
	$(PYTHON) -m fedcrg.cli run $(EXPERIMENT)

campaign: ## Execute the full experiment campaign from prepared data
	$(PYTHON) -m fedcrg.cli campaign

status: ## Show persistent status of the campaign
	$(PYTHON) -m fedcrg.cli status

monitor: ## Stream resource telemetry (CPU/RAM/GPU)
	$(PYTHON) -m fedcrg.cli monitor

results: ## Build the publication bundle
	$(PYTHON) -m fedcrg.cli results build

verify-results: ## Verify the publication bundle
	$(PYTHON) -m fedcrg.cli results verify

quality: ## Complete quality gate (format, lint, typecheck, full tests)
	$(RUFF) format --check src tests
	$(RUFF) check src tests
	$(PYRIGHT) src/fedcrg
	$(PYTEST)
