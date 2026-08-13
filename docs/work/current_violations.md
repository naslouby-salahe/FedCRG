# Current Violations

Actionable findings against prompt.md, grouped by severity. Cross-refs are the audit
matrix IDs (`docs/FedCRG Audit Matrix.md`).

## A. Forbidden vocabulary and naming (prompt §8)

- A1. ~~`canonical` in production source~~ RESOLVED 2026-08-13: `serialized_payload()`,
  `serialized`, `normalized_name`, docstrings cleaned; `rg -i canonical src tests` clean
  (only a test function name). Matrix: R04.
- A2. ~~Opaque `PolicyId` enum values~~ RESOLVED 2026-08-13: values renamed to
  descriptive snake_case (`reference_quantile`, `global_quantile`, `local_quantile`,
  `readiness_only`, `mismatch_only`, `shrinkage`, `three_sigma`,
  `development_f1_selection`, `summary_statistic`, `supervised_f1`, `oracle_test`,
  `fedcrg`); YAML configs and reporting updated; roadmap shorthand remains only in the
  normative spec tables. Matrix: E07.
- A3. ~~Vague/legacy README~~ RESOLVED 2026-08-13: architecture section matches tree;
  broken ledger link replaced with the audit matrix. Matrix: H01.

## B. Configuration ownership (prompt §5, §9)

- B1. ~~Scientific defaults in Pydantic models~~ RESOLVED 2026-08-13: ProtocolConfig,
  TrainingConfig, RandomnessConfig, DatasetConfig, StatisticsConfig, detector configs all
  require scientific values from YAML; `tests/_fixtures.py` carries explicit regression
  values for unit tests. Matrix: K01-K06, K10.
- B2. ~~`src/fedcrg/domain/constants.py`~~ DELETED 2026-08-13; values moved to YAML
  (seeds, calibration seeds, attack-split seed) or derived (autoencoder parameter/byte
  counts from architecture). Matrix: K09, K11.
- B3. ~~No `configs/statistics/confirmatory.yaml`~~ RESOLVED 2026-08-13: StatisticsConfig
  + YAML; paired bootstrap/contrasts/claim gates/margin/familywise all config-resolved,
  no hidden defaults. Matrix: K06, M06, M07, C14.
- B4. ~~Config file layout differs from target tree~~ RESOLVED 2026-08-13:
  `configs/method|training|randomness|statistics|detectors` layout matches target;
  experiment YAMLs reference the new files; obsolete files deleted. Matrix: K01-K04, E06.

## C. Structure (prompt §4, §11, §13, §14)

- C1. ~~`pipeline/` package = second orchestration layer~~ RESOLVED 2026-08-13: folded
  into `experiments/` (runner, preflight, verification, experiment_runner, policy_cells,
  model_training, dataset_preparation, table_precompute) and capability homes
  (`scoring/`, `evaluation/`). Matrix: R02, E09.
- C2. ~~Preprocessed data at `outputs/cache/datasets/`~~ RESOLVED 2026-08-13:
  materialization moved to `config.preprocessed_root` (default `data/preprocessed/`);
  `outputs/cache/datasets/` removed from tree. Matrix: R06, D05, D12.
- C3. ~~Outputs layout~~ RESOLVED 2026-08-13: `outputs/logs/` + `outputs/monitoring/`
  added; `cache/precomputed/` renamed to `cache/analysis/`; README and .gitignore
  updated. Matrix: R08.
- C4. ~~No `results/`, no campaign, no `fedcrg results build|verify`~~ RESOLVED
  2026-08-13: `experiments/campaign.py` + `reporting/results.py` + CLI
  (`campaign run|status|list`, `results build|verify`); campaign completion invokes the
  same ResultsBuilder. Matrix: R09, L04, L05, P03-P05.
- C5. Package names: `config/`->`configuration/`, `data/`->`datasets/`,
  `method/`+`thresholds/`->`decision/`, `runtime.py`->`runtime/` all RESOLVED 2026-08-13;
  `cli/` consolidation remains for Batch 7. Matrix: R01.

## D. CLI / logging / monitoring (prompt §16-§18, §34)

- D1. ~~CLI groups differ from target; missing `campaign`, `results`,
  `data status`, `experiment validate|run`~~ RESOLVED 2026-08-13: 14 CLI files
  consolidated to `app.py` + `data_commands.py` + `experiment_commands.py` +
  `analysis_commands.py` + `report_commands.py`; `campaign run|status|list`,
  `results build|verify`, `data preprocess|status`, `experiment validate`
  added; entry point `cli.app:cli`. Matrix: L01-L06.
- D2. ~~Logs go to stderr only~~ RESOLVED 2026-08-13: file handler writes
  `outputs/logs/fedcrg.log`; `runtime.py` split into `runtime/{logging,gpu,monitoring}.py`.
  Matrix: L07.
- D3. No Rich console progress for long runs. Matrix: L08.
- D4. ~~No resource telemetry / fedcrg monitor~~ RESOLVED 2026-08-13: `ResourceMonitor`
  (psutil + torch CUDA) writes `outputs/monitoring/telemetry.jsonl`; `fedcrg monitor`
  streams live samples; campaign telemetry hook pending Batch 5. Matrix: L10, L06.
- D5. ~~GPU: no device logging / no CUDA-required guard / no inference_mode~~ RESOLVED
  2026-08-13: `resolve_compute_device` refuses silent CPU fallback; device name/VRAM/peak
  logged; `torch.inference_mode()` scoring; device required (no default). Matrix: L11.

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
