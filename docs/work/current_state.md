# Current State

Snapshot 2026-08-13, branch `main`, working tree clean.

## Repository facts

- 14,917 LOC in `src/fedcrg/`; 51 test files; `pytest` green (125 passed).
- `pyright src/fedcrg` green (0 errors, 0 warnings).
- Statistical core is solid: Gate A/B numeric ledger tests pass to 1e-10, reference
  rank 4500 -> 4456, Gate-B minimum 736, five-state decision machine, strict `>` rule.
- Data adapters: N-BaIoT (9 clients, 115 features, source-order) and DIAD (115 source
  identities, 86-feature contract, water-filling attack-dev allocation) present with
  contract tests.
- Experiment catalogue S1-S6 / R1-R14 fully defined with locked scales and a completion
  ledger.
- Score cache is streaming, immutable, hash-finalized Parquet; calibration assignments
  are views over the reservoir; artifact verification by hash exists.
- Claim gates G0-G8, policy contrasts, paired bootstrap, split stability exist.

## Structural deviations from the target architecture (prompt.md §4)

1. Package names: `config/` vs target `configuration/`; `data/` vs `datasets/`;
   `method/` + `thresholds/` vs `decision/`; `runtime.py` vs `runtime/logging.py`;
   `cli/main.py` + 10 files vs `cli/app.py` + 4 command modules.
2. `pipeline/` package (10 modules) is a second orchestration layer; the prompt requires
   `experiments/` to own experiment/campaign execution with one spine.
3. Preprocessed datasets are materialized at `outputs/cache/datasets/`; the prompt
   mandates `data/preprocessed/`.
4. Outputs layout is missing `outputs/logs/` and `outputs/monitoring/`; has
   `outputs/cache/datasets/` (forbidden) and `outputs/cache/precomputed/` (not in target).
5. No `results/` publication area, no campaign concept, no `fedcrg results build|verify`,
   no `fedcrg monitor`, no `data status`.
6. No `Makefile`, no `noxfile.py`.

## Configuration ownership

- YAML owns most scientific values, but several Python config models still carry
  scientific defaults (`ProtocolConfig`, `TrainingConfig`, detector models,
  `RandomnessConfig`, `DatasetConfig.minimum_clients`), which the prompt forbids.
- `domain/constants.py` holds seeds/counts duplicated with YAML; the prompt forbids
  solving hardcoding with a Python `constants.py`.
- No `configs/statistics/` (bootstrap replicates/seeds, margins) and no
  `configs/randomness/` (dedicated seed registry) exist; values are inline or Python
  defaults.
- `configs/` file names deviate from the target tree (`autoencoder_nbaiot.yaml` vs
  `nbaiot_autoencoder.yaml`, `deep_svdd_nbaiot.yaml` vs `second_detector.yaml`,
  `target_fpr_{real,synthetic}.yaml`, `synthetic/default.yaml`, ...).

## Vocabulary

- `canonical` remains in `artifacts/manifests.py`, `config/experiment_config.py`,
  `artifacts/paths.py` docstring, `artifacts/environment.py`.
- `PolicyId` enum values are opaque protocol shorthand (`REF-Q99-R`, `GATE-A-ONLY`,
  `DEV-F1-LG-SELECT`, `LARIDI-STYLE-SS`, `SUP-F1-1000`, `ORACLE-TEST`) which the prompt
  forbids as enum values; publication labels belong in reporting.

## Tooling

- CI runs ruff + mypy + pytest; prompt requires Pyright/Pylance-compatible typing.
  Pyright passes locally but is not wired into CI/pyproject.
- No psutil/rich in dependencies; monitoring and Rich console are not implemented.

See `current_violations.md` for the actionable list and `next_actions.md` for the
execution order.
