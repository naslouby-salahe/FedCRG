# FedCRG developer interface. The Makefile only forwards to real commands;
# it never carries scientific configuration values.

PYTHON ?= python3
PIP ?= pip
RUFF ?= ruff
PYRIGHT ?= pyright
PYTEST ?= pytest
NOX ?= nox

CONFIG ?= config/study.yaml
DATASET ?= nbaiot
CAMPAIGN ?= default
DATA_ROOT ?= data/raw
PREPARED_ROOT ?= data/preprocessed

.PHONY: help install format lint typecheck test test-unit test-integration \
	test-contract test-regression audit validate preprocess plan run campaign \
	status monitor results verify-results quality

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

install: ## Install the package and dev tooling
	$(PIP) install -e ".[dev]"

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

test-regression: ## Regression tests only
	$(PYTEST) tests/regression

audit: ## Re-audit the repository against the goal matrix
	$(PYTHON) tools/audit_repository.py

validate: ## Validate one resolved experiment configuration
	$(PYTHON) -m fedcrg.cli validate primary_nbaiot --config $(CONFIG)

preprocess: ## Preprocess DATASET into data/preprocessed/
	$(PYTHON) -m fedcrg.cli preprocess $(DATASET) --data-root $(DATA_ROOT) --config $(CONFIG)

plan: ## Plan the primary experiment
	$(PYTHON) -m fedcrg.cli plan primary_nbaiot --config $(CONFIG)

run: ## Execute the CONFIG experiment grid from prepared data
	$(PYTHON) -m fedcrg.cli run primary_nbaiot --prepared-root $(PREPARED_ROOT) --config $(CONFIG)

campaign: ## Run campaign CAMPAIGN over the configured experiments
	$(PYTHON) -m fedcrg.cli campaign --campaign-id $(CAMPAIGN) --prepared-root $(PREPARED_ROOT) --config $(CONFIG)

status: ## Show persistent status of campaign CAMPAIGN
	$(PYTHON) -m fedcrg.cli status --campaign-id $(CAMPAIGN)

monitor: ## Stream resource telemetry (CPU/RAM/GPU)
	$(PYTHON) -m fedcrg.cli monitor

results: ## Build the publication bundle for campaign CAMPAIGN
	$(PYTHON) -m fedcrg.cli results build $(CAMPAIGN)

verify-results: ## Verify the publication bundle for campaign CAMPAIGN
	$(PYTHON) -m fedcrg.cli results verify $(CAMPAIGN)

quality: ## Complete quality gate (format, lint, typecheck, full tests)
	$(RUFF) format --check src tests
	$(RUFF) check src tests
	$(PYRIGHT) src/fedcrg
	$(PYTEST)
