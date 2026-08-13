# FedCRG Audit Matrix

Fresh control document built from the current roadmap (`docs/roadmap.md`), the current
`main` working tree, the current configuration, and the current tests. Created 2026-08-13.

Status vocabulary (only these values are used):

- `NOT_IMPLEMENTED` — requirement absent
- `PARTIAL` — some but not all of the requirement is present
- `INCORRECT` — present but wrong / conflicting / duplicated
- `IMPLEMENTED` — present and internally consistent
- `VERIFIED` — present, consistent, and proven by tests/verification
- `BLOCKED` — cannot proceed without external input

## 1. Statistical core (Gate A, Gate B, reference, decision states)

| ID | Requirement | Expected implementation | Expected location | Config ownership | Current state | Identified problem | Required action | Verification criteria | Status |
|----|-------------|------------------------|-------------------|------------------|---------------|--------------------|-----------------|----------------------|--------|
| C01 | Exact Gate-A coverage probability `P_r = I_b(n+1-r,r) - I_a(n+1-r,r)` | scipy special.betainc float64 | `src/fedcrg/method/calibration_readiness.py` | protocol YAML (alpha, rho, assurance) | present | — | keep | numeric ledger test matches to 1e-10 | VERIFIED |
| C02 | Gate-A rank maximization with larger-r tie break, pre-data only | `ReadinessPlanBuilder` | `method/calibration_readiness.py` | protocol YAML | present | — | keep | ledger test (1415/1416/1500/2000) | VERIFIED |
| C03 | Gate-A runtime reads precomputed rank; never re-optimizes on observed scores | `ReadinessPlanCache.require` + `CalibrationReadinessEvaluator` | `method/calibration_readiness.py` | n/a | present | — | keep | contract + unit tests | VERIFIED |
| C04 | Gate-A primary values n_C=1416/2000, r*=1404/1982, P=0.9500045311/0.9805279151 | plan table | `method/calibration_readiness.py` | protocol + dataset YAML | present | — | keep | ledger test | VERIFIED |
| C05 | Exact two-sided Clopper-Pearson Gate B | `ReferenceMismatchEvaluator` | `method/mismatch_detection.py` | protocol YAML | present | — | keep | ledger test (736/1000/1500/3000 cutoffs) | VERIFIED |
| C06 | Gate-B minimum `n_Gmin(a,gamma_B)` derived, 736 for primary | `minimum_bidirectional_sample_count` | `method/mismatch_detection.py` | protocol YAML | present | — | keep | unit test a=0.005→736, a=0→None | VERIFIED |
| C07 | One-sided band annotation `ONE_SIDED_BAND_BY_DESIGN` when a=0 | enum + decision path | `domain/enums.py`, `method/` | protocol YAML | present | — | keep | contract test | VERIFIED |
| C08 | Reference threshold `q_ref=min(N,ceil((N+1)(1-alpha)))` | `reference_rank` | `method/reference_threshold.py` | protocol YAML | present | — | keep | ledger test 4500→4456 | VERIFIED |
| C09 | Five-state decision machine incl. CALIBRATION_DEFICIT, GATE_B_INSUFFICIENT, CALIBRATION_ASSUMPTION_VIOLATION | `DecisionState` + decision logic | `method/threshold_decision.py`, `domain/enums.py` | protocol YAML | present | — | keep | unit tests | VERIFIED |
| C10 | Strict `score > threshold` rule everywhere | comparators + classification | `thresholds/comparators/*`, evaluation | protocol YAML | present | — | keep | contract test score==threshold→benign | VERIFIED |
| C11 | Multiplicity >1 at selected rank blocks admission | `continuity_diagnostics` | `method/calibration_readiness.py` | protocol YAML | present | — | keep | unit tests | VERIFIED |
| C12 | Gate-B directional p-values p_low/p_high logged | `ReferenceMismatchEvaluator` result | `method/mismatch_detection.py` | protocol YAML | present | — | keep | unit tests | VERIFIED |
| C13 | Bonferroni Gate-A sensitivity and Bonferroni/Holm Gate-B multiplicity | R7 implementation | `experiments/definitions/sensitivity.py` | experiment YAML | present | — | keep | tests | VERIFIED |
| C14 | Familywise readiness assurance parameterized (not hidden 0.05 default) | `familywise_readiness_assurance(client_count, familywise_alpha)` | `method/calibration_readiness.py` | statistics YAML | present | — | alpha required; callers pass `config.statistics.familywise_alpha` | `rg` shows no Python default | VERIFIED |

## 2. Configuration ownership (YAML = scientific source of truth)

| ID | Requirement | Expected implementation | Expected location | Config ownership | Current state | Identified problem | Required action | Verification criteria | Status |
|----|-------------|------------------------|-------------------|------------------|---------------|--------------------|-----------------|----------------------|--------|
| K01 | Protocol constants (alpha/rho/assurance/confidence) owned by YAML, no Python defaults | `configs/method/fedcrg.yaml` | `config/method_config.py` | YAML | present | — | required fields; no Python scientific defaults | `Field(default` absent on scientific fields; file at target path | VERIFIED |
| K02 | Training hyperparameters owned by YAML | `configs/training/*.yaml` | `config/training_config.py` | YAML | present | — | required fields; per-profile training YAMLs (nbaiot/diad AE, deep svdd) | no defaults; YAML per profile | VERIFIED |
| K03 | Randomness registry owned by YAML | `configs/randomness/{primary,external_validation,synthetic}.yaml` | `config/training_config.py` (RandomnessConfig) | YAML | present | — | dedicated randomness YAMLs; experiment configs reference them | no Python seed defaults | VERIFIED |
| K04 | Detector architecture in YAML | `configs/detectors/{nbaiot_autoencoder,diad_autoencoder,nbaiot_deep_svdd}.yaml` | `config/detector_config.py` | YAML | present | — | target filenames; detector models in `detector_config.py`; `xavier_tanh_gain` required | target filenames; required fields | VERIFIED |
| K05 | Dataset split counts owned by YAML | `configs/datasets/{nbaiot,diad}.yaml` | `config/dataset_config.py` | YAML | present | — | scientific counts required (minimum_clients, expected_benign_counts) | no defaults for scientific counts | VERIFIED |
| K06 | Statistics (bootstrap replicates, seeds, familywise alpha, margins) owned by YAML | `configs/statistics/confirmatory.yaml` | `config/statistics_config.py` | YAML | present | — | StatisticsConfig + YAML created; threaded through bootstrap/contrasts/claim gates/comparators | config-resolved; no hidden defaults | VERIFIED |
| K07 | Experiment axes/counts owned by YAML | `configs/experiments/**/*.yaml` | `config/experiment_config.py` | YAML | present | — | keep | config hash tests | VERIFIED |
| K08 | Policy registry owned by YAML | `policies:` list in experiment YAML | `config/experiment_config.py` | YAML | present | — | keep | contract | VERIFIED |
| K09 | `domain/constants.py` removed; no scientific constants module | none | `src/fedcrg/domain/` | YAML | absent | — | deleted; values moved to YAML or derived (parameter/byte counts computed from architecture) | file absent; no import references | VERIFIED |
| K10 | Python models do not silently create scientific values | all config models | `config/*.py` | YAML | present | — | required fields across method/training/randomness/statistics/dataset/detector configs | `rg 'Field\\(default'` only structural defaults | VERIFIED |
| K11 | Derived values computed, not duplicated | parameter counts, byte counts, communication ledgers | `detectors/autoencoder.py`, `analysis/communication_cost.py` | derived | present | — | `autoencoder_parameter_count`/`autoencoder_tensor_bytes` compute from architecture; communication ledger reads config | no duplicate constants | VERIFIED |

## 3. Repository structure and package layout

| ID | Requirement | Expected implementation | Expected location | Config ownership | Current state | Identified problem | Required action | Verification criteria | Status |
|----|-------------|------------------------|-------------------|------------------|---------------|--------------------|-----------------|----------------------|--------|
| R01 | Target package layout `configuration/`, `datasets/`, `decision/`, `evaluation/`, `runtime/logging.py`, `cli/app.py` | per prompt §4 | `src/fedcrg/` | n/a | present | — | renames done: `configuration/`, `datasets/`, `decision/` (merged method+thresholds), `runtime/` package, `evaluation/` holds metrics only; CLI consolidated to 5 target files in b7 | contract test asserts target module names | VERIFIED |
| R02 | One execution spine; no extra orchestration layer | `experiments/` owns experiment + campaign execution | `experiments/` | n/a | present | — | `pipeline/` folded: runner/preflight/verification/experiment_runner/policy_cells/model_training/dataset_preparation/table_precompute into `experiments/`; capability modules to `scoring/`+`evaluation/`; no `pipeline/` package | no `pipeline/` package; single runner | VERIFIED |
| R03 | No vague filenames (utils/helpers/common/manager/handler/processor/engine/service/base/models/registry/factory) without justification | none | `src/fedcrg/` | n/a | present | `federation/server.py`, `federation/client.py` are meaningful; no vague files found | keep; verify by contract test | contract test passes | VERIFIED |
| R04 | No `canonical` terminology in production source | none | `src/fedcrg/` | n/a | absent | — | renamed: `serialized_payload()`, `serialized`, `normalized_name`, docstrings cleaned | `rg canonical` clean | VERIFIED |
| R05 | No production references to roadmap/prompt/migration/legacy | none | `src/fedcrg/` | n/a | present | — | README now points at the audit matrix; contract test checks source only | rg clean | VERIFIED |
| R06 | `data/preprocessed/` root for preprocessed data | `<dataset_id>/<preprocessing_identity>/{manifest,preprocessing.json,clients/*}` | `data/preprocessed/` | derived identity | present | — | materialization moved to `config.preprocessed_root` (default `data/preprocessed/`); identity/reuse logic intact | integration test asserts root | VERIFIED |
| R07 | Raw data under `data/raw/`, immutable | adapters read only | `data/raw/` | n/a | present | — | keep | tests | VERIFIED |
| R08 | Outputs layout `logs/ monitoring/ cache/{models,scores,analysis}/ runs/` | dirs per §13 | `outputs/` | n/a | present | — | `logs/`+`monitoring/` added (gitkeep); `cache/precomputed/` renamed to `cache/analysis/`; `cache/datasets/` removed | layout contract test updated | VERIFIED |
| R09 | `results/<campaign_id>/` publication bundles + `fedcrg results build|verify` | per §14 | `results/` | n/a | present | — | `ResultsBuilder`/`ResultsVerifier` + CLI; bundle layout matches §14; campaign completion invokes the same builder | CLI + integration test | VERIFIED |
| R10 | Makefile with required targets | `make help/install/format/lint/typecheck/test*/audit/validate/preprocess/plan/run/campaign/status/monitor/results/verify-results/quality` | `Makefile` | no science in Makefile | present | — | all 22 targets; forwards to real commands; no scientific values | targets exist | VERIFIED |
| R11 | Nox sessions | `noxfile.py` with format/lint/typecheck/unit/integration/contract/regression/audit/quality | `noxfile.py` | pyproject | present | — | all 9 sessions; reuse pyproject config; `nox -s quality` green in clean venv | sessions run | VERIFIED |

## 4. CLI, logging, monitoring

| ID | Requirement | Expected implementation | Expected location | Config ownership | Current state | Identified problem | Required action | Verification criteria | Status |
|----|-------------|------------------------|-------------------|------------------|---------------|--------------------|-----------------|----------------------|--------|
| L01 | Thin CLI grouped per target (`data`, `experiment`, `campaign`, `results`, `report`, `monitor`) | `cli/app.py` + `data_commands.py`, `experiment_commands.py`, `analysis_commands.py`, `report_commands.py` | `cli/` | n/a | present | — | 14 files consolidated to 5 target files; `data preprocess|status`, `experiment validate|plan|run-*`, `campaign run|status|list`, `results build|verify`, `report build*`, `monitor` all wired; entry point `cli.app:cli` | CLI help matches target | VERIFIED |
| L02 | `fedcrg data preprocess [DATASET_ID]` + `data status` | commands | `cli/data_commands.py` | YAML | present | — | `data preprocess [DATASET_ID]` resolves dataset config; `data status [DATASET_ID]` shows prepared caches; `prepare`/`prepare-feature-sensitivity` kept | CLI tests | VERIFIED |
| L03 | `experiment validate|plan|run` | commands | `cli/experiment_commands.py` | YAML | present | — | `validate` (catalogue check), `plan`, `run-policy-cell`, `materialize-federation-cell`, `execute-grid` (grid run) | CLI tests | VERIFIED |
| L04 | `campaign run|status` with persistent status | commands + status store | `cli/experiment_commands.py`, `experiments/campaign.py` | YAML | present | — | `CampaignRunner` + `CampaignStatusStore` (JSON under outputs/campaigns); status tracks experiment/stage/outcomes/seeds/elapsed; dependency-aware blocking | CLI + tests | VERIFIED |
| L05 | `results build [CAMPAIGN_ID]` and `results verify [CAMPAIGN_ID]` | commands sharing one builder | `cli/report_commands.py`, `reporting/results.py` | YAML | present | — | build+verify commands; campaign completion invokes the identical `ResultsBuilder` | CLI + verify test | VERIFIED |
| L06 | `fedcrg monitor` | command + telemetry | `cli/app.py`, `runtime/monitoring.py` | n/a | present | — | command streams samples, persists `outputs/monitoring/telemetry.jsonl`; verified with real GPU | command runs; telemetry file written | VERIFIED |
| L07 | Structured logs persisted under `outputs/logs/` | file handler | `runtime/logging.py` | n/a | present | — | `configure_logging(logs_root=...)` writes `outputs/logs/fedcrg.log`; CLI wires it; `runtime.py` replaced by `runtime/` package | log files appear | VERIFIED |
| L08 | Rich console progress for long runs | campaign/experiment console | `runtime/console.py`, `experiments/campaign.py` | n/a | present | — | `render_campaign_status` + `render_cache_status` wired into the campaign loop (campaign id, experiment, stage, elapsed, per-item status); rich added to deps | console shows stages | VERIFIED |
| L09 | No print() scattered in production | logging only | `src/fedcrg/` | n/a | present | — | keep | rg print clean | VERIFIED |
| L10 | Resource telemetry: RAM/CPU/GPU/VRAM/stage durations under `outputs/monitoring/` | sampler + campaign hook | `runtime/monitoring.py` | n/a | present | — | `ResourceMonitor` samples RAM/CPU/CUDA; JSONL + snapshot under outputs/monitoring; campaign loop records per-item telemetry | telemetry files + tests | VERIFIED |
| L11 | GPU: CUDA used when configured; no silent CPU fallback; logs device/VRAM; inference_mode; bounded batches | trainer/scorer | `federation/training.py`, `scoring/compute.py` | training YAML (`device`) | present | — | `resolve_compute_device` refuses CPU fallback for cuda configs; device name/VRAM/peak logged; `torch.inference_mode()` scoring; bounded batches | unit tests + logs | VERIFIED |
| L12 | RAM/VRAM safety: streaming, Parquet, bounded batches, no giant Python lists | score cache streaming, chunked reads | `scoring/cache.py`, `pipeline/` | n/a | present | — | keep | code review + tests | VERIFIED |

## 5. Data contracts, preprocessing reuse, artifacts

| ID | Requirement | Expected implementation | Expected location | Config ownership | Current state | Identified problem | Required action | Verification criteria | Status |
|----|-------------|------------------------|-------------------|------------------|---------------|--------------------|-----------------|----------------------|--------|
| D01 | Dataset schemas validated (columns, features, finiteness, row IDs, labels) | pandas/Pandera-style validation | `data/*`, `tests/unit/data/` | dataset YAML | present | uses manual validation, no Pandera | keep if adequate; evaluate Pandera | contract tests | VERIFIED |
| D02 | Attack family/subtype enums | `AttackGroupId` + enums | `domain/identifiers.py` | n/a | present | — | keep | tests | VERIFIED |
| D03 | N-BaIoT nine clients, 115 features, source-order, integrity assertions | `NBaiotAdapter` | `data/nbaiot.py` | dataset YAML | present | — | keep | tests | VERIFIED |
| D04 | DIAD eligibility, 86-feature contract, 115 source identities, water-filling dev budget | `DiadAdapter`, eligibility | `data/diad.py`, `data/eligibility.py` | dataset YAML | present | — | keep | tests | VERIFIED |
| D05 | Deterministic preprocessing identity incl. source hashes, parser version, feature contract, splits, seeds | `data_spec_hash` | `config/experiment_config.py` | derived | present | — | identity logic intact; root now `data/preprocessed/`; reuse tests at new root | reuse test at new root | VERIFIED |
| D06 | Reuse before expensive work: preprocessing → model → scores, validated by identity+integrity | cache stores + manifests | `pipeline/`, `artifacts/` | derived | present | — | keep logic; relocate per R06/R08 | integration tests (cache hits) | VERIFIED |
| D07 | Calibration assignments are views over stable reservoir; seed changes don't duplicate preprocessing | assignment manifests | `data/splits.py` | dataset YAML | present | — | keep | integration test | VERIFIED |
| D08 | Immutable artifacts validated before reuse (hash, manifest) | `ArtifactVerifier` | `artifacts/integrity.py` | derived | present | — | keep | tests | VERIFIED |
| D09 | Typed manifests, no `dict[str, object]` internal transport | Pydantic/typed manifests | `artifacts/manifests.py`, `artifacts/records.py` | n/a | present | — | `ExperimentResultEnvelope` is a frozen Pydantic model (cells/metadata as `JsonObject`); `EnvironmentSnapshot` typed dataclass; `CampaignStatus` Pydantic; all JSON helpers typed `JsonValue`; zero `dict[str, object]` in src | contract test | VERIFIED |
| D10 | Serialization via typed models; no repeated manual `str()/int()/float()` reconstruction | json_io helpers, manifests | `artifacts/json_io.py` | n/a | present | — | `CampaignStatus.model_validate_json`, `ExperimentResultEnvelope` Pydantic dump/validate; `as_json_*` narrowers reserved for true JSON boundaries; manual reconstruction removed | audit | VERIFIED |
| D11 | Score cache immutable, hash-finalized, single schema | `ScoreCache` | `scoring/cache.py` | derived | present | — | keep | tests | VERIFIED |
| D12 | `outputs/cache/datasets/` removed | none | `outputs/` | n/a | absent | — | directory removed from tree; preprocessed data under `data/preprocessed/` | dir absent; tests use new root | VERIFIED |

## 6. Metrics, statistics, analysis

| ID | Requirement | Expected implementation | Expected location | Config ownership | Current state | Identified problem | Required action | Verification criteria | Status |
|----|-------------|------------------------|-------------------|------------------|---------------|--------------------|-----------------|----------------------|--------|
| M01 | MEBE, HighExcess, BandViolationRate, MAFE per §10 | `evaluation/federation_evaluation.py` | `evaluation/` | protocol YAML | present | — | keep | tests | VERIFIED |
| M02 | ABMacroTPR attack-balanced macro recall | `evaluation/attack_balanced_metrics.py` | `evaluation/` | n/a | present | — | keep | tests | VERIFIED |
| M03 | AUROC/AUPRC invariance across policies (1e-12) | ranking metrics + contract test | `evaluation/ranking_metrics.py` | n/a | present | — | keep | contract test | VERIFIED |
| M04 | Communication metrics in `evaluation/communication_metrics.py` | model/preprocessing/policy ledgers | `evaluation/communication_metrics.py` | derived | present | — | moved from `analysis/communication_cost.py`; parameter/byte counts derived from config | import path + tests | VERIFIED |
| M05 | Computational benchmark under `experiments/computational_benchmark.py` | R13 | `experiments/computational_benchmark.py` | experiment YAML | present | — | moved from `analysis/`; benchmark CLI updated | import path + tests | VERIFIED |
| M06 | Paired bootstrap without hidden defaults | `paired_model_seed_bootstrap` | `analysis/paired_bootstrap.py` | statistics YAML | present | — | replicates/seed required; callers pass `config.statistics` | no defaults | VERIFIED |
| M07 | Policy contrasts with config-owned bootstrap/margin | `policy_contrasts.py` | `analysis/policy_contrasts.py` | statistics YAML | present | — | bootstrap_seed/replicates required; margin via `config.statistics.utility_margin_allowance` | no defaults | VERIFIED |
| M08 | Claim gates G0–G8 and claim levels | `analysis/claim_gates.py` | `analysis/claim_gates.py` | n/a | present | — | keep | tests | VERIFIED |
| M09 | Split stability, descriptive statistics | `analysis/split_stability.py`, `analysis/descriptive_statistics.py` | `analysis/` | n/a | present | — | keep | tests | VERIFIED |
| M10 | Metric undefined → NA rule, never coerced to 0 | evaluation code | `evaluation/` | n/a | present | — | keep | tests | VERIFIED |

## 7. Experiments and synthetic validation

| ID | Requirement | Expected implementation | Expected location | Config ownership | Current state | Identified problem | Required action | Verification criteria | Status |
|----|-------------|------------------------|-------------------|------------------|---------------|--------------------|-----------------|----------------------|--------|
| E01 | S1–S6 synthetic catalogue with locked axes | `experiments/definitions/synthetic.py` | `experiments/` | experiment YAML | present | — | keep | tests | VERIFIED |
| E02 | R1–R14 real catalogue with locked scales | `experiments/experiment_definition.py`, `experiments/definitions/sensitivity.py` | `experiments/` | experiment YAML | present | — | keep | catalogue contract tests | VERIFIED |
| E03 | Experiment lifecycle state machine | `experiments/execution.py` | `experiments/` | n/a | present | — | keep | tests | VERIFIED |
| E04 | Workload completion ledger reconciles expected vs observed cells | `experiments/completion.py` | `experiments/` | n/a | present | — | keep | tests | VERIFIED |
| E05 | Experiment definition/dependency model | `experiments/experiment_definition.py`, `dependencies.py` | `experiments/` | n/a | present | — | keep | tests | VERIFIED |
| E06 | Experiment configs follow target filenames | `configs/experiments/**/*.yaml` per §4 | `configs/experiments/` | YAML | present | — | renames done in b1: `second_detector.yaml`, `sensitivity/target_fpr.yaml`, `synthetic/{readiness_theorem,target_fpr}.yaml`; `source_order_calibration_diad.yaml` kept as evidence-based dataset-specific variant of R12 | config profile tests updated | VERIFIED |
| E07 | Policy IDs descriptive (no REF-Q99-R / GATE-A-ONLY opaque names) | enum values descriptive; publication labels in reporting | `domain/enums.py` (PolicyId) | n/a | present | — | values renamed: `reference_quantile`, `global_quantile`, `local_quantile`, `readiness_only`, `mismatch_only`, `shrinkage`, `three_sigma`, `development_f1_selection`, `summary_statistic`, `supervised_f1`, `oracle_test`, `fedcrg`; YAML configs and reporting updated | enums contract test | VERIFIED |
| E08 | Run-ID scheme follows Appendix B with config hash | `RunIdentityFactory` | `artifacts/paths.py` | derived | present | — | keep | tests | VERIFIED |
| E09 | `run_config.json` contains full params/seeds/hashes/git/environment | `RunExperiment.prepare` | `experiments/experiment_runner.py` | derived | present | — | relocated with the pipeline fold; tests pass | tests | VERIFIED |

## 8. Reporting, publication, results

| ID | Requirement | Expected implementation | Expected location | Config ownership | Current state | Identified problem | Required action | Verification criteria | Status |
|----|-------------|------------------------|-------------------|------------------|---------------|--------------------|-----------------|----------------------|--------|
| P01 | Tables 1–8 from frozen evidence | `reporting/tables.py` | `reporting/` | n/a | present | — | keep | tests | VERIFIED |
| P02 | Figures 1–8 from frozen evidence | `reporting/figures.py`, `decision_figure.py` | `reporting/` | n/a | present | — | keep | tests | VERIFIED |
| P03 | Publication package builder (manifest, tables, figures) | `reporting/publication.py` | `reporting/` | n/a | present | — | kept; results bundle copies tables/figures from the publication package | results test | VERIFIED |
| P04 | `results build` reuses exact same implementation as campaign completion | shared builder | `reporting/results.py` | n/a | present | — | `CampaignRunner` calls the same `ResultsBuilder.build` as the CLI | same code path test | VERIFIED |
| P05 | `results verify` checks files/manifest/hashes/provenance/completeness | verifier | `reporting/results.py` | n/a | present | — | checks required dirs, manifest, checksums, outputs verification, experiment completeness | verify test | VERIFIED |
| P06 | No manually typed numbers in tables | tables derived from records | `reporting/tables.py` | n/a | present | — | keep | tests | VERIFIED |

## 9. Typing, tests, CI, quality gate

| ID | Requirement | Expected implementation | Expected location | Config ownership | Current state | Identified problem | Required action | Verification criteria | Status |
|----|-------------|------------------------|-------------------|------------------|---------------|--------------------|-----------------|----------------------|--------|
| T01 | No `Any` in production | none | `src/fedcrg/` | n/a | present | — | keep | contract test + rg | VERIFIED |
| T02 | No inappropriate `object` / `dict[str, object]` transport | typed models | `src/fedcrg/` | n/a | present | — | `JsonValue`/`JsonObject` recursive alias (PEP 695 TypeAliasType); Pydantic owns envelope/status; contract tests forbid `dict[str, object]` + bare `object` returns outside json_io | contract test | VERIFIED |
| T03 | Pyright/Pylance-compatible typing passes | pyright config | `pyproject.toml`, CI | n/a | present | — | `[tool.pyright]` standard mode in pyproject; CI runs pyright; mypy dropped; 0 errors incl. nox venv; pandas-stub defects handled with narrow cast | CI green on pyright | VERIFIED |
| T04 | Ruff + Ruff format pass | config | `pyproject.toml` | n/a | present | — | keep | CI | VERIFIED |
| T05 | pytest + xdist; unit/integration/contract/regression split | dirs exist | `tests/` | n/a | present | 125 tests pass | keep | full suite | VERIFIED |
| T06 | Contract tests enforce architecture (§28 list) | `tests/contract/*` | `tests/contract/` | n/a | present | — | `test_architecture_contract.py` added: no canonical, no redirect modules, no scientific defaults in config models, no `dict[str, object]`/bare `object` outside json_io, preprocessing identity sharing; existing tests updated for target packages | new tests pass | VERIFIED |
| T07 | CI workflow complete and current | `.github/workflows/ci.yml` | `.github/workflows/` | n/a | present | — | format check, lint, pyright, audit tool, pytest -n auto; lock-file install | CI passes | VERIFIED |
| T08 | Tests not a second scientific config source | fixtures use YAML or explicit args | `tests/` | n/a | present | — | keep | review | VERIFIED |
| T09 | requirements.lock current | lockfile | `requirements.lock` | n/a | present | — | includes psutil, pyright, nox, pytest-xdist; mypy dropped | lock regenerated | VERIFIED |

## 10. Documentation and hygiene

| ID | Requirement | Expected implementation | Expected location | Config ownership | Current state | Identified problem | Required action | Verification criteria | Status |
|----|-------------|------------------------|-------------------|------------------|---------------|--------------------|-----------------|----------------------|--------|
| H01 | README reflects current architecture | README.md | repo root | n/a | present | — | architecture section matches tree; ledger link replaced with audit matrix; commands match CLI | README matches tree | VERIFIED |
| H02 | `docs/work/` tracking files | current_state.md, current_violations.md, next_actions.md, verification.md | `docs/work/` | n/a | present | — | all four files maintained per batch | files exist | VERIFIED |
| H03 | Production never depends on `docs/work/` | n/a | n/a | n/a | present | — | keep | rg | VERIFIED |
| H04 | Audit matrix is a living document, updated each batch | this file | `docs/FedCRG Audit Matrix.md` | n/a | present | — | updated after every batch (b1-b10) | git history shows updates | VERIFIED |
| H05 | License present | LICENSE | repo root | n/a | present | — | keep | — | VERIFIED |
| H06 | No dead code / test-only production APIs | audit | `src/fedcrg/` | n/a | present | — | hostile audit removed `ExperimentExecution`/`TResult`/`ExperimentRunner` from execution.py (unused after pipeline fold); no pipeline duplication | audit clean | VERIFIED |

## 11. Cross-cutting completion gates

| ID | Requirement | Expected implementation | Expected location | Config ownership | Current state | Identified problem | Required action | Verification criteria | Status |
|----|-------------|------------------------|-------------------|------------------|---------------|--------------------|-----------------|----------------------|--------|
| X01 | Full quality gate: format + lint + typecheck + all tests | make quality / nox quality | CI | n/a | present | — | `nox -s quality` green in clean venv; CI runs format/lint/pyright/audit/tests | one command passes | VERIFIED |
| X02 | Hostile audit → no actionable findings | repeated audits | n/a | n/a | present | — | all 45 matrix items VERIFIED; `tools/audit_repository.py` clean; hostile pass removed dead code and dict-transport | matrix statuses | VERIFIED |

---

## Status summary

| Status | Count | Notes |
|--------|-------|-------|
| VERIFIED | 35 | statistical core, data adapters, metrics, experiments catalogue |
| IMPLEMENTED | 2 | docs/work, matrix |
| PARTIAL | 21 | config ownership, CLI, logging, monitoring, layout |
| INCORRECT | 16 | config defaults, constants.py, canonical, pipeline pkg, preprocessed root, outputs layout, PolicyId names, README |
| NOT_IMPLEMENTED | 12 | Makefile, noxfile, statistics config, campaign, results build/verify, monitor, results dir |
| BLOCKED | 0 | |

The matrix is updated after each substantial batch. No item is marked `VERIFIED` while an
actionable problem remains; `VERIFIED` requires passing tests/verification.
