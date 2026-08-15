# FedCRG developer interface. The Makefile only forwards to real commands;
# it never carries scientific configuration values. Paths live in
# config/study.yaml, so no target accepts path variables.
#
# Experiment commands are generated from the catalogue below: every entry
# defines a validate-<name>, plan-<name> and run-<name> target.

PYTHON ?= python3
RUFF ?= ruff
PYRIGHT ?= pyright
PYTEST ?= pytest
NOX ?= nox

DATASET ?=
EXPERIMENT ?= primary_nbaiot

# --- Experiment catalogue ------------------------------------------------------
# Names use dashes; the CLI expects underscores (a 1:1 substitution).

# Synthetic experiments
SYNTHETIC := \
  readiness-theorem \
  target-fpr-synthetic \
  temporal-dependence \
  calibration-shift \
  calibration-contamination \
  mismatch-power

# Computational benchmark
BENCHMARK := computational-benchmark

# Primary experiment on real data
PRIMARY := primary-nbaiot

# External validation on real data
EXTERNAL := external-diad

# Sensitivity analyses
SENSITIVITY := \
  readiness-sample-size \
  mismatch-sample-size \
  tolerance-sensitivity \
  target-fpr-real \
  assurance-sensitivity \
  multiplicity-sensitivity \
  diad-feature-sensitivity

# Robustness analyses
ROBUSTNESS := \
  source-order-test \
  real-contamination \
  second-detector \
  source-order-calibration

EXPERIMENTS := $(SYNTHETIC) $(BENCHMARK) $(PRIMARY) $(EXTERNAL) $(SENSITIVITY) $(ROBUSTNESS)

# --- Generated per-experiment targets -------------------------------------------

define EXPERIMENT_RULES
validate-$(1): ## Validate the $(1) experiment configuration
	$(PYTHON) -m fedcrg.cli validate $(subst -,_,$(1))
plan-$(1): ## Preview the $(1) experiment plan
	$(PYTHON) -m fedcrg.cli plan $(subst -,_,$(1))
run-$(1): ## Run the $(1) experiment
	$(PYTHON) -m fedcrg.cli run $(subst -,_,$(1))
endef

$(foreach exp,$(EXPERIMENTS),$(eval $(call EXPERIMENT_RULES,$(exp))))

# Generated targets are commands, not files.
EXP_TARGETS := $(foreach exp,$(EXPERIMENTS),validate-$(exp) plan-$(exp) run-$(exp))
.PHONY: $(EXP_TARGETS)

# --- Core targets -----------------------------------------------------------------

.PHONY: help install doctor format lint typecheck test test-unit test-integration \
	test-contract audit freeze validate preprocess plan run campaign \
	status monitor report results verify-results quality

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'
	@echo
	@echo "Experiment commands: make <validate|plan|run>-<experiment>"
	@for exp in $(EXPERIMENTS); do printf '  %-28s %-28s %s\n' "validate-$$exp" "plan-$$exp" "run-$$exp"; done

install: ## Install the package and dev tooling from the locked dependency graph
	uv sync --locked --extra dev

doctor: ## Show library versions and CUDA availability
	$(PYTHON) -m fedcrg.cli doctor

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

freeze: ## Snapshot pinned dependency versions into requirements.lock
	$(PYTHON) tools/freeze_environment.py

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

report: ## Build the repository hygiene report and publication manifest
	$(PYTHON) -m fedcrg.cli report

results: ## Build the campaign results bundle (per-experiment bundles are auto-built by run/campaign)
	$(PYTHON) -m fedcrg.cli results build

verify-results: ## Verify the campaign results bundle and every experiment bundle
	$(PYTHON) -m fedcrg.cli results verify

quality: ## Complete quality gate (format, lint, typecheck, full tests)
	$(RUFF) format --check src tests
	$(RUFF) check src tests
	$(PYRIGHT) src/fedcrg
	$(PYTEST)
