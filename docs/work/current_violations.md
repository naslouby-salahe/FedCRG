# Current Violations

Actionable findings against prompt.md, grouped by severity. Cross-refs are the audit
matrix IDs (`docs/FedCRG Audit Matrix.md`).

## A. Forbidden vocabulary and naming (prompt §8)

- A1. `canonical` in production source:
  - `src/fedcrg/artifacts/manifests.py:107` local variable `canonical`
  - `src/fedcrg/config/experiment_config.py:69` method `canonical_json()`
  - `src/fedcrg/artifacts/paths.py:1` docstring "Canonical immutable output layout"
  - `src/fedcrg/artifacts/environment.py:100` local `canonical_name`
  - Rename to `serialized_payload` / `stable_representation` / `resolved_*` per prompt.
  - Matrix: R04.

- A2. Opaque `PolicyId` enum values (prompt §7): `REF-Q99-R`, `GLOBAL-Q99-FULL`,
  `LOCAL-Q99-FULL`, `GATE-A-ONLY`, `GATE-B-ONLY`, `SHRINKAGE`, `FEDDETECT-3SIGMA`,
  `DEV-F1-LG-SELECT`, `LARIDI-STYLE-SS`, `SUP-F1-1000`, `ORACLE-TEST`, `FEDCRG`.
  Replace with descriptive values (`reference_quantile`, `global_quantile`,
  `local_quantile`, `readiness_only`, `mismatch_only`, `shrinkage`, `three_sigma`,
  `development_f1_selection`, `summary_statistic`, `supervised_f1`, `oracle_test`,
  `fedcrg`). Publication labels move to reporting. Matrix: E07.
  Note: configs, run IDs, manifests, tests, and comparators all reference these values.

- A3. Vague/legacy README: describes `core/`, `protocol/`, `policies/`, `metrics/`,
  `application/` that no longer exist; links `docs/protocol_implementation_ledger.md`
  which does not exist. Matrix: H01.

## B. Configuration ownership (prompt §5, §9)

- B1. Scientific defaults in Pydantic models:
  - `ProtocolConfig` (method_config.py): alpha=0.01, rho=0.50, readiness_assurance=0.95,
    mismatch_confidence=0.95.
  - `TrainingConfig` (training_config.py): rounds=30, batch_size=64, lr 1e-3/1e-5,
    adam_betas=(0.9,0.999), adam_epsilon=1e-8, weight_decay=0.0, client_fraction=1.0.
  - `AutoencoderConfig.xavier_tanh_gain=5/3`; `RandomnessConfig` defaults from
    domain/constants.py.
  - `DatasetConfig.minimum_clients=1`, `expected_benign_counts={}` default.
  - Make fields required; YAML must supply them. Matrix: K01-K06, K10.

- B2. `src/fedcrg/domain/constants.py` exists with scientific seeds/counts duplicated
  with YAML (seeds, parameter counts, byte counts). Delete; move to YAML or derive.
  Matrix: K09, K11.

- B3. No `configs/statistics/confirmatory.yaml`; `paired_bootstrap.py` and
  `policy_contrasts.py` carry `replicates=10000`, `seed=424242` defaults; utility margin
  0.03 hardcoded in `analysis/claim_gates.py` / `policy_contrasts.py`. Matrix: K06, M06,
  M07, C14.

- B4. Config file layout differs from target tree (see current_state.md item 6).
  Matrix: K01-K04, E06.

## C. Structure (prompt §4, §11, §13, §14)

- C1. `pipeline/` package = second orchestration layer; one execution spine must live in
  `experiments/`. Matrix: R02, E09.
- C2. Preprocessed data at `outputs/cache/datasets/`; must be `data/preprocessed/`.
  Matrix: R06, D05, D12.
- C3. Outputs layout missing `logs/` and `monitoring/`; has `cache/datasets` (forbidden)
  and `cache/precomputed` (not in target; should be `cache/analysis/`). Matrix: R08.
- C4. No `results/`, no campaign, no `fedcrg results build|verify`. Matrix: R09, L04, L05.
- C5. Package names deviate from target (`config/`, `data/`, `method/`+`thresholds/`,
  `runtime.py`, `cli/`). Matrix: R01.

## D. CLI / logging / monitoring (prompt §16-§18, §34)

- D1. CLI groups differ from target; missing `campaign`, `results`, `monitor`,
  `data status`, `experiment validate|run`. Matrix: L01-L06.
- D2. Logs go to stderr only; no persistence under `outputs/logs/`. Matrix: L07.
- D3. No Rich console progress for long runs. Matrix: L08.
- D4. No resource telemetry (RAM/CPU/GPU/VRAM/stage timing), no `outputs/monitoring/`,
  no `fedcrg monitor`. Matrix: L10, L06.
- D5. GPU: device honored, but no device/VRAM logging, no explicit CUDA-required guard,
  no inference_mode. Matrix: L11.

## E. Typing / tests / tooling (prompt §28-§30)

- E1. `dict[str, object]` / `object` at non-boundary locations: `artifacts/records.py`
  (cells, metadata, to_dict), `artifacts/environment.py:46` return type, integrity
  helpers. json_io boundary itself is acceptable; type the rest. Matrix: T02, D09.
- E2. CI uses mypy; prompt requires Pyright/Pylance-compatible typing. Pyright passes
  locally; wire into pyproject + CI. Matrix: T03, T07.
- E3. Missing contract tests: preprocessed root, no pipeline package, no constants.py,
  results build/verify, config YAML ownership, no scientific defaults. Matrix: T06.
- E4. No `Makefile`, no `noxfile.py`. Matrix: R10, R11.

## F. Dead code / redundancy to audit during hostile audit

- F1. `experiments/execution.py` `TResult` generic + `ExperimentRunner` Callable:
  check actual usage after pipeline fold.
- F2. `pipeline/` modules may duplicate `experiments/` logic (planning vs run_experiment).
- F3. `descriptive_statistics.py` / `communication_cost.py` / `computational_benchmark.py`
  live in `analysis/` but belong to evaluation/experiments per target tree.
- F4. `outputs/cache/precomputed/` vs `cache/analysis/` naming.
