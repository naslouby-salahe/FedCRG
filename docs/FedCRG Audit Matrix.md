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
| C14 | Familywise readiness assurance parameterized (not hidden 0.05 default) | `familywise_readiness_assurance(client_count, familywise_alpha)` | `method/calibration_readiness.py` | **statistics config MISSING** | `PARTIAL` | `familywise_alpha=0.05` is a Python default; no statistics YAML exists | add `configs/statistics/confirmatory.yaml`; make R7 pass alpha from config | config-resolved value used; no Python scientific default | `PARTIAL` |

## 2. Configuration ownership (YAML = scientific source of truth)

| ID | Requirement | Expected implementation | Expected location | Config ownership | Current state | Identified problem | Required action | Verification criteria | Status |
|----|-------------|------------------------|-------------------|------------------|---------------|--------------------|-----------------|----------------------|--------|
| K01 | Protocol constants (alpha/rho/assurance/confidence) owned by YAML, no Python defaults | `configs/method/fedcrg.yaml` (target) / `configs/protocol/default.yaml` (current) | `config/method_config.py` | YAML | present but **INCORRECT** | `ProtocolConfig` fields carry `Field(default=0.01)` etc.; file is `configs/protocol/default.yaml` not target `configs/method/fedcrg.yaml` | make fields required; move to `configs/method/fedcrg.yaml` | `Field(default` absent in method config; no scientific default | `INCORRECT` |
| K02 | Training hyperparameters owned by YAML | `configs/training/*.yaml` | `config/training_config.py` | YAML | present but **INCORRECT** | `TrainingConfig` has Python defaults (rounds=30, batch=64, lr=1e-3/1e-5, betas, eps, wd, client_fraction); no `configs/training/` dir | make required; add per-profile training YAMLs | no defaults; YAML per profile | `INCORRECT` |
| K03 | Randomness registry owned by YAML | `configs/randomness/{primary,external_validation,synthetic}.yaml` | `config/training_config.py` (RandomnessConfig) | YAML | present but **INCORRECT** | `RandomnessConfig` defaults to `domain/constants.py` values; seeds duplicated in every experiment YAML | move to dedicated randomness YAMLs, remove constants defaults | no Python seed defaults | `INCORRECT` |
| K04 | Detector architecture in YAML | `configs/detectors/{nbaiot_autoencoder,diad_autoencoder,nbaiot_deep_svdd}.yaml` | `config/training_config.py` (detector models) | YAML | present but **INCORRECT** | filenames are `autoencoder_nbaiot.yaml`/`deep_svdd.yaml`; `xavier_tanh_gain` default 5/3 in Python; detector config classes live inside `training_config.py` instead of `detector_config.py` | rename files; move detector models to `config/detector_config.py`; require xavier gain | target filenames; required fields | `INCORRECT` |
| K05 | Dataset split counts owned by YAML | `configs/datasets/{nbaiot,diad}.yaml` | `config/dataset_config.py` | YAML | present | `minimum_clients=1` default in Python (science-ish), `expected_benign_counts` default empty | make minimum_clients required; keep others required | no defaults for scientific counts | `PARTIAL` |
| K06 | Statistics (bootstrap replicates, seeds, familywise alpha, margins) owned by YAML | `configs/statistics/confirmatory.yaml` | `config/statistics_config.py` | YAML | `NOT_IMPLEMENTED` | `paired_bootstrap.py` has `replicates=10000, seed=424242` defaults; `policy_contrasts.py` defaults; no statistics config model/file | create StatisticsConfig + YAML; thread through analysis | config-resolved; no hidden defaults | `NOT_IMPLEMENTED` |
| K07 | Experiment axes/counts owned by YAML | `configs/experiments/**/*.yaml` | `config/experiment_config.py` | YAML | present | — | keep | config hash tests | VERIFIED |
| K08 | Policy registry owned by YAML | `policies:` list in experiment YAML | `config/experiment_config.py` | YAML | present | — | keep | contract | VERIFIED |
| K09 | `domain/constants.py` removed; no scientific constants module | none | `src/fedcrg/domain/` | YAML | present but **forbidden** | prompt §5: "Do not solve hardcoding by moving values into a Python `constants.py`"; file holds seeds/counts duplicated with YAML | delete `domain/constants.py`; move remaining true constants to YAML or derive | file absent; no import references | `INCORRECT` |
| K10 | Python models do not silently create scientific values | all config models | `config/*.py` | YAML | `INCORRECT` | multiple `Field(default=...)` on scientific fields | eliminate per K01–K06 | `rg 'Field\\(default'` clean | `INCORRECT` |
| K11 | Derived values computed, not duplicated | parameter counts, byte counts, communication ledgers | `analysis/communication_cost.py`, tests | derived | present | `domain/constants.py` hardcodes NBAIOT_AUTOENCODER_PARAMETERS etc. | compute from architecture; move computation into `evaluation/communication_metrics.py` | no duplicate constants | `PARTIAL` |

## 3. Repository structure and package layout

| ID | Requirement | Expected implementation | Expected location | Config ownership | Current state | Identified problem | Required action | Verification criteria | Status |
|----|-------------|------------------------|-------------------|------------------|---------------|--------------------|-----------------|----------------------|--------|
| R01 | Target package layout `configuration/`, `datasets/`, `decision/`, `evaluation/`, `runtime/logging.py`, `cli/app.py` | per prompt §4 | `src/fedcrg/` | n/a | `PARTIAL` | current: `config/`, `data/`, `method/`+`thresholds/`, `runtime.py`, `cli/main.py`+10 files | rename/consolidate packages | contract test asserts target module names | `PARTIAL` |
| R02 | One execution spine; no extra orchestration layer | `experiments/` owns experiment + campaign execution | `experiments/` | n/a | `INCORRECT` | `pipeline/` package (10 modules) is a second orchestration layer; prompt §4/§21 require one spine owned by `experiments/` | fold `pipeline/` responsibilities into `experiments/` (runner, preflight, verification) and capability modules | no `pipeline/` package; single runner | `INCORRECT` |
| R03 | No vague filenames (utils/helpers/common/manager/handler/processor/engine/service/base/models/registry/factory) without justification | none | `src/fedcrg/` | n/a | present | `federation/server.py`, `federation/client.py` are meaningful; no vague files found | keep; verify by contract test | contract test passes | VERIFIED |
| R04 | No `canonical` terminology in production source | none | `src/fedcrg/` | n/a | `INCORRECT` | `artifacts/manifests.py:107` var `canonical`; `config/experiment_config.py:69` `canonical_json()`; `artifacts/paths.py:1` docstring; `artifacts/environment.py:100` `canonical_name` | rename to `stable_representation`/`serialized_payload`/`resolved_*` | `rg canonical` clean | `INCORRECT` |
| R05 | No production references to roadmap/prompt/migration/legacy | none | `src/fedcrg/` | n/a | present | README references `docs/protocol_implementation_ledger.md` which does not exist | fix README; contract test already checks source | rg clean | `PARTIAL` |
| R06 | `data/preprocessed/` root for preprocessed data | `<dataset_id>/<preprocessing_identity>/{manifest,preprocessing.json,clients/*}` | `data/preprocessed/` | derived identity | `INCORRECT` | prepared datasets written to `outputs/cache/datasets/`; prompt §11 mandates `data/preprocessed/` | move materialization root; keep identity/reuse logic | integration test asserts root | `INCORRECT` |
| R07 | Raw data under `data/raw/`, immutable | adapters read only | `data/raw/` | n/a | present | — | keep | tests | VERIFIED |
| R08 | Outputs layout `logs/ monitoring/ cache/{models,scores,analysis}/ runs/` | dirs per §13 | `outputs/` | n/a | `INCORRECT` | missing `logs/`, `monitoring/`; has `cache/datasets/` (forbidden), `cache/precomputed/` (not in target); no `cache/analysis/` | add logs/monitoring; move precomputed→analysis; drop cache/datasets | layout contract test updated | `INCORRECT` |
| R09 | `results/<campaign_id>/` publication bundles + `fedcrg results build|verify` | per §14 | `results/` | n/a | `NOT_IMPLEMENTED` | no `results/` dir, no campaign concept, no results CLI | implement campaign + results build/verify | CLI + integration test | `NOT_IMPLEMENTED` |
| R10 | Makefile with required targets | `make help/install/format/lint/typecheck/test*/audit/validate/preprocess/plan/run/campaign/status/monitor/results/verify-results/quality` | `Makefile` | no science in Makefile | `NOT_IMPLEMENTED` | no Makefile | add | targets exist | `NOT_IMPLEMENTED` |
| R11 | Nox sessions | `noxfile.py` with format/lint/typecheck/unit/integration/contract/regression/audit/quality | `noxfile.py` | pyproject | `NOT_IMPLEMENTED` | no noxfile | add; reuse pyproject config | sessions run | `NOT_IMPLEMENTED` |

## 4. CLI, logging, monitoring

| ID | Requirement | Expected implementation | Expected location | Config ownership | Current state | Identified problem | Required action | Verification criteria | Status |
|----|-------------|------------------------|-------------------|------------------|---------------|--------------------|-----------------|----------------------|--------|
| L01 | Thin CLI grouped per target (`data`, `experiment`, `campaign`, `results`, `report`, `monitor`) | `cli/app.py` + `data_commands.py`, `experiment_commands.py`, `analysis_commands.py`, `report_commands.py` | `cli/` | n/a | `PARTIAL` | current: main.py + benchmark/claims/data/environment/evaluation/experiments/reporting/scoring/training/verification; missing `campaign`, `results`, `monitor`, `data status`; extra `doctor/config/claims` groups | consolidate to target groups; add campaign/results/monitor/data status | CLI help matches target | `PARTIAL` |
| L02 | `fedcrg data preprocess [DATASET_ID]` + `data status` | commands | `cli/data_commands.py` | YAML | `PARTIAL` | `data prepare --config` exists; no `preprocess [ID]` form, no `status` | align command surface | CLI tests | `PARTIAL` |
| L03 | `experiment validate|plan|run` | commands | `cli/experiment_commands.py` | YAML | `PARTIAL` | plan exists; no validate/run top-level | align | CLI tests | `PARTIAL` |
| L04 | `campaign run|status` with persistent status | commands + status store | `cli/experiment_commands.py`, `experiments/` | YAML | `NOT_IMPLEMENTED` | no campaign concept | implement campaign runner + status | CLI + tests | `NOT_IMPLEMENTED` |
| L05 | `results build [CAMPAIGN_ID]` and `results verify [CAMPAIGN_ID]` | commands sharing one builder | `cli/report_commands.py`, `reporting/` | YAML | `NOT_IMPLEMENTED` | publication build exists (`report build-publication`) but not under results/, no verify | implement | CLI + verify test | `NOT_IMPLEMENTED` |
| L06 | `fedcrg monitor` | command + telemetry | `cli/`, `runtime/` | n/a | `NOT_IMPLEMENTED` | no monitoring | implement (psutil, torch CUDA) | command runs; telemetry file written | `NOT_IMPLEMENTED` |
| L07 | Structured logs persisted under `outputs/logs/` | file handler | `runtime/logging.py` | n/a | `INCORRECT` | logging only to stderr; no file logs; `runtime.py` is not `runtime/logging.py` | add file handler to outputs/logs; rename module | log files appear | `INCORRECT` |
| L08 | Rich console progress for long runs | campaign/experiment console | `cli/`, `experiments/` | n/a | `NOT_IMPLEMENTED` | plain logging only | add Rich progress panels | console shows stages | `NOT_IMPLEMENTED` |
| L09 | No print() scattered in production | logging only | `src/fedcrg/` | n/a | present | — | keep | rg print clean | VERIFIED |
| L10 | Resource telemetry: RAM/CPU/GPU/VRAM/stage durations under `outputs/monitoring/` | sampler + campaign hook | `runtime/monitoring.py` (new) | n/a | `NOT_IMPLEMENTED` | absent | implement; wire into campaign execution | telemetry files + tests | `NOT_IMPLEMENTED` |
| L11 | GPU: CUDA used when configured; no silent CPU fallback; logs device/VRAM; inference_mode; bounded batches | trainer/scorer | `federation/training.py`, `pipeline/compute_scores.py` | training YAML (`device`) | `PARTIAL` | device config exists and is used; no GPU logging of name/VRAM; no explicit no-fallback check; no monitoring | add device logging, explicit CUDA-required guard, inference_mode, VRAM logging | unit tests + logs | `PARTIAL` |
| L12 | RAM/VRAM safety: streaming, Parquet, bounded batches, no giant Python lists | score cache streaming, chunked reads | `scoring/cache.py`, `pipeline/` | n/a | present | — | keep | code review + tests | VERIFIED |

## 5. Data contracts, preprocessing reuse, artifacts

| ID | Requirement | Expected implementation | Expected location | Config ownership | Current state | Identified problem | Required action | Verification criteria | Status |
|----|-------------|------------------------|-------------------|------------------|---------------|--------------------|-----------------|----------------------|--------|
| D01 | Dataset schemas validated (columns, features, finiteness, row IDs, labels) | pandas/Pandera-style validation | `data/*`, `tests/unit/data/` | dataset YAML | present | uses manual validation, no Pandera | keep if adequate; evaluate Pandera | contract tests | VERIFIED |
| D02 | Attack family/subtype enums | `AttackGroupId` + enums | `domain/identifiers.py` | n/a | present | — | keep | tests | VERIFIED |
| D03 | N-BaIoT nine clients, 115 features, source-order, integrity assertions | `NBaiotAdapter` | `data/nbaiot.py` | dataset YAML | present | — | keep | tests | VERIFIED |
| D04 | DIAD eligibility, 86-feature contract, 115 source identities, water-filling dev budget | `DiadAdapter`, eligibility | `data/diad.py`, `data/eligibility.py` | dataset YAML | present | — | keep | tests | VERIFIED |
| D05 | Deterministic preprocessing identity incl. source hashes, parser version, feature contract, splits, seeds | `data_spec_hash` | `config/experiment_config.py` | derived | present | identity logic exists but root is `outputs/cache/datasets` | move root; keep identity | reuse test at new root | `PARTIAL` |
| D06 | Reuse before expensive work: preprocessing → model → scores, validated by identity+integrity | cache stores + manifests | `pipeline/`, `artifacts/` | derived | present | — | keep logic; relocate per R06/R08 | integration tests (cache hits) | VERIFIED |
| D07 | Calibration assignments are views over stable reservoir; seed changes don't duplicate preprocessing | assignment manifests | `data/splits.py` | dataset YAML | present | — | keep | integration test | VERIFIED |
| D08 | Immutable artifacts validated before reuse (hash, manifest) | `ArtifactVerifier` | `artifacts/integrity.py` | derived | present | — | keep | tests | VERIFIED |
| D09 | Typed manifests, no `dict[str, object]` internal transport | Pydantic/typed manifests | `artifacts/manifests.py`, `artifacts/records.py` | n/a | `PARTIAL` | `records.py` cells/metadata as `dict[str, object]`; environment capture returns `dict[str, object]` | type them (typed models or dedicated records) | contract test | `PARTIAL` |
| D10 | Serialization via typed models; no repeated manual `str()/int()/float()` reconstruction | json_io helpers, manifests | `artifacts/json_io.py` | n/a | `PARTIAL` | `as_json_dict`/`as_json_list` narrowers used broadly | keep only at true boundaries; prefer TypeAdapter | audit | `PARTIAL` |
| D11 | Score cache immutable, hash-finalized, single schema | `ScoreCache` | `scoring/cache.py` | derived | present | — | keep | tests | VERIFIED |
| D12 | `outputs/cache/datasets/` removed | none | `outputs/` | n/a | `INCORRECT` | exists with data; forbidden by §11 | migrate to `data/preprocessed/` | dir absent; tests use new root | `INCORRECT` |

## 6. Metrics, statistics, analysis

| ID | Requirement | Expected implementation | Expected location | Config ownership | Current state | Identified problem | Required action | Verification criteria | Status |
|----|-------------|------------------------|-------------------|------------------|---------------|--------------------|-----------------|----------------------|--------|
| M01 | MEBE, HighExcess, BandViolationRate, MAFE per §10 | `evaluation/federation_evaluation.py` | `evaluation/` | protocol YAML | present | — | keep | tests | VERIFIED |
| M02 | ABMacroTPR attack-balanced macro recall | `evaluation/attack_balanced_metrics.py` | `evaluation/` | n/a | present | — | keep | tests | VERIFIED |
| M03 | AUROC/AUPRC invariance across policies (1e-12) | ranking metrics + contract test | `evaluation/ranking_metrics.py` | n/a | present | — | keep | contract test | VERIFIED |
| M04 | Communication metrics in `evaluation/communication_metrics.py` | model/preprocessing/policy ledgers | `evaluation/communication_metrics.py` | derived | `INCORRECT` | module lives in `analysis/communication_cost.py` | move to `evaluation/`; compute parameter/byte counts from config, not constants | import path + tests | `INCORRECT` |
| M05 | Computational benchmark under `experiments/computational_benchmark.py` | R13 | `experiments/computational_benchmark.py` | experiment YAML | `INCORRECT` | lives in `analysis/computational_benchmark.py` | move to experiments | import path + tests | `INCORRECT` |
| M06 | Paired bootstrap without hidden defaults | `paired_model_seed_bootstrap` | `analysis/paired_bootstrap.py` | statistics YAML | `PARTIAL` | `replicates=10000, seed=424242` Python defaults | remove defaults; config supplies | no defaults | `PARTIAL` |
| M07 | Policy contrasts with config-owned bootstrap/margin | `policy_contrasts.py` | `analysis/policy_contrasts.py` | statistics YAML | `PARTIAL` | bootstrap_seed/replicates defaults in signature; margin 0.03 hardcoded | config-driven | no defaults | `PARTIAL` |
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
| E06 | Experiment configs follow target filenames | `configs/experiments/**/*.yaml` per §4 | `configs/experiments/` | YAML | `PARTIAL` | `robustness/deep_svdd_nbaiot.yaml` should be `second_detector.yaml`; `sensitivity/target_fpr_{real,synthetic}.yaml` vs target single `target_fpr.yaml`; `synthetic/default.yaml` vs `readiness_theorem.yaml`; extra `source_order_calibration_diad.yaml` | rename to target layout | config profile tests updated | `PARTIAL` |
| E07 | Policy IDs descriptive (no REF-Q99-R / GATE-A-ONLY opaque names) | enum values descriptive; publication labels in reporting | `domain/enums.py` (PolicyId) | n/a | `INCORRECT` | `PolicyId` values are `REF-Q99-R`, `GLOBAL-Q99-FULL`, `GATE-A-ONLY`, `DEV-F1-LG-SELECT`, `LARIDI-STYLE-SS`, `SUP-F1-1000`, `ORACLE-TEST` — prompt §7 forbids these | rename enum values to descriptive (`reference_quantile`, `global_quantile`, `readiness_only`, `development_f1_selection`, ...); map paper labels in reporting | enums contract test | `INCORRECT` |
| E08 | Run-ID scheme follows Appendix B with config hash | `RunIdentityFactory` | `artifacts/paths.py` | derived | present | — | keep | tests | VERIFIED |
| E09 | `run_config.json` contains full params/seeds/hashes/git/environment | `RunExperiment.prepare` | `pipeline/run_experiment.py` → `experiments/` | derived | present | lives in pipeline/ (R02) | relocate with pipeline fold | tests | `PARTIAL` |

## 8. Reporting, publication, results

| ID | Requirement | Expected implementation | Expected location | Config ownership | Current state | Identified problem | Required action | Verification criteria | Status |
|----|-------------|------------------------|-------------------|------------------|---------------|--------------------|-----------------|----------------------|--------|
| P01 | Tables 1–8 from frozen evidence | `reporting/tables.py` | `reporting/` | n/a | present | — | keep | tests | VERIFIED |
| P02 | Figures 1–8 from frozen evidence | `reporting/figures.py`, `decision_figure.py` | `reporting/` | n/a | present | — | keep | tests | VERIFIED |
| P03 | Publication package builder (manifest, tables, figures) | `reporting/publication.py` | `reporting/` | n/a | present | builds to `outputs/reports/publication/` not `results/<campaign>/` | wire into results builder per §14 | results test | `PARTIAL` |
| P04 | `results build` reuses exact same implementation as campaign completion | shared builder | `reporting/` | n/a | `NOT_IMPLEMENTED` | campaign doesn't exist; no results build | implement | same code path test | `NOT_IMPLEMENTED` |
| P05 | `results verify` checks files/manifest/hashes/provenance/completeness | verifier | `reporting/` or `artifacts/` | n/a | `NOT_IMPLEMENTED` | absent | implement | verify test | `NOT_IMPLEMENTED` |
| P06 | No manually typed numbers in tables | tables derived from records | `reporting/tables.py` | n/a | present | — | keep | tests | VERIFIED |

## 9. Typing, tests, CI, quality gate

| ID | Requirement | Expected implementation | Expected location | Config ownership | Current state | Identified problem | Required action | Verification criteria | Status |
|----|-------------|------------------------|-------------------|------------------|---------------|--------------------|-----------------|----------------------|--------|
| T01 | No `Any` in production | none | `src/fedcrg/` | n/a | present | — | keep | contract test + rg | VERIFIED |
| T02 | No inappropriate `object` / `dict[str, object]` transport | typed models | `src/fedcrg/` | n/a | `PARTIAL` | json_io boundary + records/environment/integrity helpers | keep json_io boundary only; type the rest | contract test | `PARTIAL` |
| T03 | Pyright/Pylance-compatible typing passes | pyright config | `pyproject.toml`, CI | n/a | `INCORRECT` | CI uses mypy; pyright passes locally (0 errors) but not configured | switch CI to pyright; add config; drop mypy or keep both | CI green on pyright | `PARTIAL` |
| T04 | Ruff + Ruff format pass | config | `pyproject.toml` | n/a | present | — | keep | CI | VERIFIED |
| T05 | pytest + xdist; unit/integration/contract/regression split | dirs exist | `tests/` | n/a | present | 125 tests pass | keep | full suite | VERIFIED |
| T06 | Contract tests enforce architecture (§28 list) | `tests/contract/*` | `tests/contract/` | n/a | `PARTIAL` | many present; missing: preprocessed root, no pipeline pkg, no canonical (regex only? add), results verify, config YAML ownership, no constants.py | add missing contract tests; update stale ones | new tests pass | `PARTIAL` |
| T07 | CI workflow complete and current | `.github/workflows/ci.yml` | `.github/workflows/` | n/a | `PARTIAL` | runs ruff+mypy+pytest; no pyright, no nox | update | CI passes | `PARTIAL` |
| T08 | Tests not a second scientific config source | fixtures use YAML or explicit args | `tests/` | n/a | present | — | keep | review | VERIFIED |
| T09 | requirements.lock current | lockfile | `requirements.lock` | n/a | `PARTIAL` | missing psutil/rich if adopted; fine otherwise | update when deps added | lock regenerated | `PARTIAL` |

## 10. Documentation and hygiene

| ID | Requirement | Expected implementation | Expected location | Config ownership | Current state | Identified problem | Required action | Verification criteria | Status |
|----|-------------|------------------------|-------------------|------------------|---------------|--------------------|-----------------|----------------------|--------|
| H01 | README reflects current architecture | README.md | repo root | n/a | `INCORRECT` | README describes `core/`, `protocol/`, `policies/`, `metrics/`, `application/` which no longer exist; links missing `docs/protocol_implementation_ledger.md` | rewrite architecture section | README matches tree | `INCORRECT` |
| H02 | `docs/work/` tracking files | current_state.md, current_violations.md, next_actions.md, verification.md | `docs/work/` | n/a | `NOT_IMPLEMENTED` | absent | created in this batch | files exist | `IMPLEMENTED` |
| H03 | Production never depends on `docs/work/` | n/a | n/a | n/a | present | — | keep | rg | VERIFIED |
| H04 | Audit matrix is a living document, updated each batch | this file | `docs/FedCRG Audit Matrix.md` | n/a | `IMPLEMENTED` | created fresh now | update after each batch | git history shows updates | `IMPLEMENTED` |
| H05 | License present | LICENSE | repo root | n/a | present | — | keep | — | VERIFIED |
| H06 | No dead code / test-only production APIs | audit | `src/fedcrg/` | n/a | `PARTIAL` | `experiments/execution.py` `TResult` generic + `ExperimentRunner` Callable used?; `pipeline/` duplication | hostile audit later | audit clean | `PARTIAL` |

## 11. Cross-cutting completion gates

| ID | Requirement | Expected implementation | Expected location | Config ownership | Current state | Identified problem | Required action | Verification criteria | Status |
|----|-------------|------------------------|-------------------|------------------|---------------|--------------------|-----------------|----------------------|--------|
| X01 | Full quality gate: format + lint + typecheck + all tests | make quality / nox quality | CI | n/a | `PARTIAL` | no Makefile/nox; CI partially covers | after R10/R11/T03 | one command passes | `PARTIAL` |
| X02 | Hostile audit → no actionable findings | repeated audits | n/a | n/a | `PARTIAL` | this matrix itself lists actionable items | work until matrix fully VERIFIED | matrix statuses | `PARTIAL` |

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
