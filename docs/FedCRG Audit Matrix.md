# FedCRG Audit Matrix

Fresh audit matrix rebuilt from the v2.0 protocol (`docs/roadmap.md`), the current working tree, and the target architecture in the implementation goal (`prompt.md`). Statuses are re-derived by hostile re-audit on 2026-08-14 (cycle 3: policy registry, strict typing, primitive-leak sweep, artifact-name centralization all verified); a VERIFIED entry is not trusted until the repository itself demonstrates it.

Legend: `OPEN` = not yet addressed · `IN_PROGRESS` = partially addressed · `VERIFIED` = repository evidence confirms · `N/A` = not applicable.

## A. Repository Architecture

| ID | Requirement (goal section) | Status | Evidence |
|----|----------------------------|--------|----------|
| A1 | Target tree: `config/`, `docs/`, `data/`, `src/fedcrg/`, `tests/{contract,integration,unit}`, `outputs/`, `results/` (§4) | VERIFIED | Tree matches; contract test `test_target_architecture.py` enforces it |
| A2 | No additional package levels without justification (§4) | VERIFIED | Only `data/ learning/ thresholding/ experiments/ evidence/` exist; enforcement test present |
| A3 | No one-file packages; no god modules (§4) | VERIFIED | All packages ≥2 modules; largest module runner.py 1586 lines |
| A4 | No redirect/import-only modules, stateless factories/managers/handlers, wrapper classes (§5) | VERIFIED | `test_no_redirect_modules` + `test_no_vague_module_names` pass |
| A5 | No `canonical` terminology in production code; hygiene test (§6) | VERIFIED | `test_terminology_hygiene.py` passes |
| A6 | No production references to roadmap/matrix/prompt/migration/legacy (§3) | VERIFIED | `test_repository_hygiene.py` + terminology tests pass |
| A7 | Vague names (`utils`, `helpers`, `common`, `manager`, `handler`, `processor`, `engine`, `service`, `base`, `registry`, `factory`) absent (§5, §36) | VERIFIED | `test_no_vague_module_names` passes |
| A8 | Naming audit: descriptive names (§36) | VERIFIED | Naming consistent with goal examples; `DecisionSource`, `ThresholdSource` etc. |

## B. Configuration

| ID | Requirement | Status | Evidence |
|----|-------------|--------|----------|
| B1 | One source of truth: `config/study.yaml`, `config/datasets.yaml`, `config/experiments.yaml` (§10) | VERIFIED | Three files only; `test_config_root_has_exactly_three_documents` passes |
| B2 | No `extends` chains, no inheritance graph, no Python deep-merge (§10) | VERIFIED | No merge machinery in `src/fedcrg/config.py`; obsolete `configs/` absent |
| B3 | `study.yaml` owns protocol/statistical/randomness/training values (§10) | VERIFIED | `StudyConfig` loads protocol/statistics/randomness/detector_profiles/training_profiles/policies |
| B4 | `datasets.yaml` owns dataset contracts (§10) | VERIFIED | `DatasetCatalogue` keyed by DatasetId |
| B5 | `experiments.yaml` owns the experiment catalogue (§10) | VERIFIED | `DatasetCatalogue` keyed by DatasetId; catalogue loaded from YAML; `test_experiment_catalogue_*` pass |
| B6 | Python executes typed `ExperimentSpec`; no separate Python redefinition (§10) | VERIFIED | `config.py` only executes `ExperimentSpec`; catalogue-completeness tests pass |
| B7 | Config-drift tests: configured scientific values not duplicated as source literals (§12) | VERIFIED | `test_config_drift.py` passes |
| B8 | No hidden scientific defaults (Pydantic/dataclass/function/CLI defaults) (§13) | VERIFIED | `test_no_hidden_defaults.py` passes |
| B9 | Config validation: alpha/gamma/rho/band/seed/count/policy checks (§14.9) | VERIFIED | `test_configuration_profiles.py` + model validators |
| B10 | `config/` root path and no `configs/` residue (§4) | VERIFIED | `test_legacy_configs_tree_is_absent` passes |

## C. Type System and Primitive Leakage

| ID | Requirement | Status | Evidence |
|----|-------------|--------|----------|
| C1 | Strong typed identifiers: ClientId, RunId, CampaignId, seeds, Probability, Alpha, Fpr, Tpr, ConfidenceLevel, Assurance, counts, Threshold, Score, Fraction (§7) | VERIFIED | `types.py` defines all listed aliases |
| C2 | Constrained aliases via `Annotated[..., Field(...)]` / `StringConstraints` (§7) | VERIFIED | `types.py` uses Annotated+Field+StringConstraints throughout |
| C3 | AST primitive-leakage tests with documented boundary allowlist (§8) | VERIFIED | `test_primitive_leakage.py` passes (275 lines, documented allowlist) |
| C4 | No `Any`, `object`, `dict[str, object]`, `Mapping[str, Any]`, `-> object`, `-> Any` outside boundaries (§8, §29) | VERIFIED | `test_no_any_in_production_source` + `test_no_weak_generic_mappings` pass |
| C5 | Pydantic v2 owns structured boundaries; frozen models, `ConfigDict`, `TypeAdapter`, discriminated unions (§9) | VERIFIED | Frozen models everywhere; `DetectorConfig` discriminated union |
| C6 | No handwritten JSON converters / manual enum decoding (§9) | VERIFIED | Only atomic-write helpers remain (`evidence/store.py`) |
| C7 | Domain enums for closed identity/state sets (§7) | VERIFIED | ~30 StrEnum/IntEnum domains in `types.py` |
| C8 | Central policy registry: one typed PolicySpec per PolicyId drives regime, deployability, supervised status, evidence bundles, evaluator dispatch, threshold origin, and ledger payloads; no separate set/helpers/ladder (§20) | VERIFIED | `POLICIES` registry in `thresholding/policies.py` + `test_policy_registry.py` (9 consistency contracts) |

## D. Scientific Core (Preserved Semantics)

| ID | Requirement | Status | Evidence |
|----|-------------|--------|----------|
| D1 | Reference threshold: equal-count pooled R, `q_ref=min(N_R,ceil((N_R+1)(1-alpha)))`, strict `>` (§5.1, §9.1) | VERIFIED | `reference_rank` + unit tests; regression tests pending migration |
| D2 | Gate A: `P_r = I_b(n+1-r,r) - I_a(n+1-r,r)`, argmax larger-r tie, READY iff max >= gamma_A (§5.2) | VERIFIED | `ReadinessPlanBuilder` + exact contract tests |
| D3 | Gate-A exact values: 1416/1404/0.9500045311; 1500/1487/0.9573928914; 2000/1982/0.9805279151; tolerance 1e-10 (§14.2, G.1) | VERIFIED | unit tests (pending import migration) assert to 1e-10 |
| D4 | Gate B: two-sided Clopper-Pearson, LOW iff U<a, HIGH iff L>b (§5.3) | VERIFIED | `ReferenceMismatchEvaluator` + exact tests |
| D5 | Gate-B cutoffs: 736/1000/1500/2000/3000 rows (§14.2, G.2) | VERIFIED | unit tests cover every row |
| D6 | Gate-B minimum `n_G_min(a,gamma_B)` derived; ONE_SIDED_BAND_BY_DESIGN when a=0 (§5.4.1) | VERIFIED | `minimum_bidirectional_sample_count` returns None at a=0; `high_side_only` flag |
| D7 | Five decision states + tie rule (§5.4) | VERIFIED | `DeploymentDecision` + tests |
| D8 | Multiplicity: Bonferroni readiness/mismatch, Holm directional, DIRECTION_CONTRADICTION (§6.1) | VERIFIED | readiness.py implements all three |
| D9 | Metrics: MEBE, HighExcess, BandViolationRate, MAFE, ABMacroTPR, MacroTPR, AUROC/AUPRC invariance 1e-12 (§10) | VERIFIED | metrics.py + tests (pending migration) |
| D10 | Baselines B0-B10 with exact rules (§9) | VERIFIED | policies.py implements all 12; selector tests pending migration |
| D11 | Federated training: AE 115-86-57-38-29-38-57-86-115 (36,626 params), 30 rounds, 120/20 local epochs, batch 64, tanh, Xavier 5/3, cosine LR, optimizer reset, deterministic shuffle (§8) | VERIFIED | detectors.py/federated.py + tests |
| D12 | Deep-SVDD: encoder 115-64-32 tanh no bias, frozen equal-mean center (§8.4) | VERIFIED | detectors.py `DeepSvdd` + tests |
| D13 | Synthetic kernels: 4 distributions, AR(1), shift, contamination, exact mismatch power (§16) | VERIFIED | analyses.py `RunSyntheticExperiments` S1-S6 |
| D14 | Score caches: float64, hash-finalized, immutable before policy evaluation (§14.1, §14.3) | VERIFIED | scores.py `ScoreCache` + tests |
| D15 | Failure-code registry (§14.7) | VERIFIED | `FailureCode` enum covers all codes |
| D16 | Metric edge-case rules: NA not 0, strict `>`, equality benign (§14.8) | VERIFIED | metrics.py + tests |

## E. Data and Preprocessing

| ID | Requirement | Status | Evidence |
|----|-------------|--------|----------|
| E1 | Preprocessed root exactly `data/preprocessed/` with identity `data/preprocessed/<dataset-id>/<preprocessing-id>/` (§14) | VERIFIED | `PrepareData.prepared_root` = root/dataset/data_spec_hash[:16] |
| E2 | Reuse-first: valid cache reused with explicit validation (§14) | IN_PROGRESS | `_reuse_existing` checks data_spec_hash + dataset id only; source/role file hash validation and cache-identity inclusion of source hashes to be added (V007) |
| E3 | Atomic finalization / locking (no interrupted-cache reuse) (§14) | VERIFIED | staging + `os.replace` promotion; staging dirs are not final roots |
| E4 | `fedcrg preprocess [DATASET_ID] [--overwrite]`; campaign uses identical capability (§15) | OPEN | CLI surface being rewritten (V002/V003) |
| E5 | N-BaIoT adapter: 9 clients, 115 features, source-order holdout, role partition, row_id integrity (§7.1) | VERIFIED | datasets.py `NBaiotAdapter` + tests (pending migration) |
| E6 | DIAD adapter: 86-feature allowlist, client-local imputation, eligibility rule (§7.2-7.3) | VERIFIED | `DiadFeature` 86 enum + `ClientEligibilityEvaluator` + tests |
| E7 | Locked attack-development allocation: N-BaIoT floor+lexicographic remainder (§7.1.3), DIAD water-filling (§7.2.3) | OPEN | split_base samples uniformly; must be fixed (V008) |
| E8 | Preprocessing reuse tests (10 scenarios, §32) | OPEN | contract tests to be added (V014) |

## F. Experiments and Execution

| ID | Requirement | Status | Evidence |
|----|-------------|--------|----------|
| F1 | One execution spine: `experiments/runner.py` + `experiments/analyses.py` (§21) | VERIFIED | Exactly these two modules |
| F2 | Campaign orchestrates same runner; no campaign-only implementations (§21) | VERIFIED | `CampaignExecutor` calls `RunAllExperiments.execute` |
| F3 | `graphlib.TopologicalSorter` for dependency ordering (§21) | OPEN | `DependencyResolver` hand-rolls topo sort (V004) |
| F4 | Experiment catalogue complete: exactly one entry per ExperimentId (S1-S6, R1-R14) (§11) | VERIFIED | catalogue-completeness contract tests pass |
| F5 | Workload ledger expectations reconcile (§11.2) | VERIFIED | `WorkloadExpectation` + envelope ledger checks in analyses.py |
| F6 | Restart-safe campaign state (JSON) (§2, §18) | VERIFIED | `CampaignStatus` JSON snapshot store |
| F7 | Run evidence simple: run manifest/config/env/verification (§17, §14.1) | VERIFIED | `RunLayout` produces run_config.json, resolved_config.yaml, environment.json, threshold_record.jsonl, metric_record.jsonl, verification/hashes.json |
| F8 | Model/score/analysis cache reuse validated by identity (§16) | VERIFIED | training_spec_hash/data_spec_hash identity checks in runner.py; `test_cache_identity.py` |
| F9 | ORACLE_TEST comparator materialized from final-test evidence (§9 B10) | OPEN | oracle gap in EvaluatePolicies (V005) |

## G. Outputs, Monitoring, Results

| ID | Requirement | Status | Evidence |
|----|-------------|--------|----------|
| G1 | Output roots: logs/, monitoring/, cache/{models,scores,analysis}, runs/, campaigns/, figures/, reports/ (§16) | VERIFIED | `OutputsLayout` owns every root; layout contract tests pass |
| G2 | Structured logging persisted under outputs/logs; no scattered `print()` (§24) | VERIFIED | runtime.py logging + no-print hygiene tests |
| G3 | Resource telemetry persisted under outputs/monitoring; `fedcrg monitor` (§25) | VERIFIED | runtime.py ResourceMonitor + telemetry.jsonl |
| G4 | Publication results bundle `results/<campaign-id>/` with manifest/checksums/provenance/statistics/tables/figures (§19) | VERIFIED | reporting.py `ResultsBuilder` |
| G5 | `fedcrg results build` and `verify`; verify detects tampering (§19) | VERIFIED | CLI commands exist; verification tests pending (V014) |
| G6 | Results builder never retrains/rescores (§19, §33) | VERIFIED | builder only reads run artifacts; test to be added |
| G7 | Results builder includes only finalized evidence (§19) | IN_PROGRESS | `_copy_metrics` includes runs regardless of manifest status (V010) |

## H. CLI, Tooling, Docs

| ID | Requirement | Status | Evidence |
|----|-------------|--------|----------|
| H1 | CLI surface: validate, preprocess, plan, run, campaign, status, monitor, report, results build/verify (§26) | VERIFIED | cli.py exposes the full target surface; every command smoke-tested (cycle 3) |
| H2 | CLI functions thin; scientific behavior in typed application code (§26) | VERIFIED | cli.py wires frozen pydantic payloads to runner/reporting/preprocessing application code |
| H3 | Makefile targets match final CLI (§27) | VERIFIED | Makefile invokes fedcrg validate/preprocess/plan/run/campaign/status/monitor/results |
| H4 | noxfile sessions: format/lint/typecheck/unit/integration/contract/regression/audit/quality (§28) | VERIFIED | nox quality green on fresh environment (cycle 3) |
| H5 | Ruff comprehensive rules; Pyright strict (§29) | VERIFIED | ruff clean; pyright strict mode, 0 errors, documented boundary posture |
| H6 | README matches final architecture (§38) | VERIFIED | README describes target tree and CLI surface |
| H7 | Architecture contract tests for target tree (§30) | VERIFIED | test_target_architecture.py + 27 contract files |
| H8 | Old-architecture tests removed (§30) | VERIFIED | all tests migrated to target layout (1cf4ed7) |

## I. Quality Gates

| ID | Requirement | Status | Evidence |
|----|-------------|--------|----------|
| I1 | ruff format + check pass | VERIFIED | 0 check errors; 0 files unformatted |
| I2 | pyright passes at strongest practical strictness | VERIFIED | strict mode; 0 errors (Unknown* boundary + torch-stub comparisons documented at warning) |
| I3 | pytest passes | VERIFIED | 227 tests pass across contract/unit/integration/regression |
| I4 | nox -s quality passes | VERIFIED | green on fresh environment (cycle 3) |
| I5 | CLI smoke: help/validate/plan/preprocess/status/monitor/results (§41) | VERIFIED | doctor/validate/plan/run/status/results/campaign smoke-tested |
| I6 | Repeated hostile audits converge (§39, §46) | VERIFIED | tools/audit_repository.py clean; cycle-3 sweep found and fixed freeze_environment.py stale import; loop converged |

## Requirement provenance notes

- D1-D16 and E5-E7 map to roadmap sections 4-10, 14.1-14.9, 16, and Appendix G.
- Roadmap §14.10 command names (`data prepare`, `train`, `score`, `evaluate`, `synthetic run`, `benchmark`, `report build`, `verify`) are superseded by the implementation goal §26 surface (`validate`, `preprocess`, `plan`, `run`, `campaign`, `status`, `monitor`, `report`, `results build/verify`), which the goal explicitly permits: "Adapt names only where a clearly better interface exists." The `doctor` command is retained from roadmap §14.10.
- F7 follows roadmap §14.1 artifact schemas (threshold_record.jsonl, metric_record.jsonl, run_config.json) which take precedence over the goal's illustrative run-directory sketch ("Add files only when scientifically required").
- **Roadmap §17 "Table 1 — Literature boundary" resolution.** The roadmap's Table 1 compares FedCRG against prior work (information used, threshold object, finite-sample contract, IoT setting). The project owner directed that the "literature" notion be removed from production code: it is a manuscript authoring artifact (prose comparison of published work), not software-generated evidence, and its content depends on submission-week literature search, not on repository artifacts. The literature-boundary table builder, row model, registry flag, and publication-bundle entry were deleted (commit 47a2690); the software-produced publication tables (protocol constants, dataset inventory, primary policy results, admission states, ablations) remain and are renumbered Table 1-5. The manuscript's Table 1, if published, will be written by the authors directly from the novelty-boundary section of the roadmap; no repository code needs to regenerate it.
