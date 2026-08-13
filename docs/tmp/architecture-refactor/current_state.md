# FedCRG Repository Inventory (Read-Only Research)

Source: `src/fedcrg/` (141 `.py` files). Tests: `tests/` (51 `.py` files). Target architecture: `/home/naslouby/Projects/FedCRG/prompt.md`. No `docs/tmp/architecture-refactor/` exists yet.

---

## 1. Source inventory, grouped by current top-level package

### `fedcrg/` (root)

| File | Responsibility | Public symbols | Internal imports |
|---|---|---|---|
| `__init__.py` | Package marker, no exports. | — | none |
| `__main__.py` | `python -m fedcrg` entry point. | — | `cli.main` |

### `analysis/` (target: split across `analysis/` + `reporting/`)

| File | Responsibility | Public symbols | Internal imports |
|---|---|---|---|
| `__init__.py` | Empty. | — | none |
| `benchmark.py` | Single-thread benchmark harness for protocol primitives. | `BenchmarkResult`, `configure_single_thread_execution`, `benchmark` | none |
| `claims.py` | Claim-strength gate (G0-G8) evidence/assessment. | `ClaimGateEvidence`, `ClaimAssessment`, `assess_claim_level` | `core.enums` |
| `communication.py` | Deterministic communication-cost accounting. | `ModelCommunicationLedger`, `PreprocessingCommunicationLedger`, `PolicyCommunicationLedger`, `model_communication`, `preprocessing_communication`, `threshold_policy_communication` | `config.models`, `core.constants`, `core.enums` |
| `decision_architecture.py` | Builds the manuscript decision-architecture figure (rendering). | `build_decision_architecture_figure` | none |
| `figures.py` | Locked figure builders (rendering, manuscript output). | `readiness_frontier`, `mismatch_power_map`, `per_client_operating_points`, `reliability_utility_frontier`, `calibration_phase_transition`, `assumption_stress`, `external_replication` | none |
| `primary.py` | Fixed-federation primary contrasts + split-sensitivity analysis over run evidence. | `FederationResultRecord`, `ContrastMetricResult`, `PolicyContrastResult`, `load_federation_results`, `confirmatory_contrasts`, `split_sensitivity` | `analysis.statistics`, `core.enums` |
| `publication.py` | Assembles publication package (tables + figures) — rendering/orchestration. | `PublicationArtifact`, `PublicationPackage`, `PublicationPackageBuilder` | `analysis.decision_architecture`, `analysis.figures`, `analysis.tables`, `config.models`, `core.enums` |
| `robustness.py` | Locked synthetic robustness generators (temporal/calibration stress). | `RobustnessCell`, `temporal_dependence_stress`, `calibration_shift_stress` | `core.enums`, `core.types`, `protocol.readiness` |
| `stability.py` | Split-sensitivity stability metrics for thresholds/decision states. | `ThresholdStability`, `StateFrequency`, `StateStability`, `summarize_threshold_stability`, `summarize_state_stability` | `core.enums` |
| `statistics.py` | Descriptive stats + paired bootstrap kernels. | `DescriptiveSummary`, `PairedBootstrapInterval`, `describe`, `paired_model_seed_bootstrap`, `split_sensitivity_summary` | none |
| `tables.py` | Deterministic manuscript-table builder — rendering. | `PublicationTableBuilder` | `analysis.primary`, `artifacts.layout`, `config.models`, `core.enums` |

**Flag:** `decision_architecture.py`, `figures.py`, `publication.py`, `tables.py` are pure rendering/reporting concerns embedded in `analysis/` — prompt.md explicitly assigns these to `reporting/` (`report.py`, `publication.py`, `tables.py`, `figures.py`, `decision_figure.py`) and says `analysis/` "must not own publication rendering." This is the single biggest package-boundary violation to fix.

### `application/` (target: **removed entirely** — folded into `pipeline/`)

| File | Responsibility | Public symbols | Internal imports |
|---|---|---|---|
| `__init__.py` | Empty. | — | none |
| `benchmark.py` | Orchestrates R13 benchmark run using analysis.benchmark + protocol primitives. | `RunBenchmark` | `analysis.benchmark`, `artifacts.environment`, `artifacts.experiment_results`, `config.models`, `core.enums`, `core.ids`, `protocol.decision`, `protocol.mismatch`, `protocol.readiness`, `protocol.reference` |
| `claims.py` | Derives claim-strength gates from frozen experiment evidence (orchestration). | `GateDiagnostic`, `ClaimGateReport`, `ClaimGateEvaluator` | `analysis.claims`, `analysis.primary`, `artifacts.layout`, `artifacts.manifest`, `artifacts.serialization`, `artifacts.verification`, `core.enums`, `experiments.completion`, `metrics.federation`; local import of `application.verify` |
| `evaluate.py` | Orchestrates policy evaluation from one score cache (large fan-in). | `EvaluatePolicies` | `artifacts.records`, `artifacts.serialization`, `config.models`, `core.enums`, `core.ids`, `metrics.attack_balanced`, `metrics.classification`, `metrics.admission`, `metrics.federation`, `metrics.operating_band`, `metrics.ranking`, `metrics.results`, `policies.base`, `policies.oracle`, `policies.registry`, `protocol.mismatch`, `protocol.results`, `protocol.service`, `scoring.cache`, `scoring.views` |
| `feature_sensitivity.py` | Builds/freezes R14 DIAD numeric-safe feature contract; also prepares R14 data. | `BuildDiadFeatureSensitivityContract`, `r14_config`, `PrepareDiadFeatureSensitivity` | `artifacts.dataset`, `artifacts.serialization`, `config.models`, `core.enums`, `data.datasets.diad`, `data.feature_sensitivity`; local imports of `application.prepare_data`, `data.datasets.diad` |
| `federation_cell.py` | Materializes all policy runs for one frozen federation/calibration cell. | `PolicyRunDirectory`, `FederationCellResult`, `FederationCellMaterializer` | `application.policy_cell`, `application.run_experiment`, `config.models`, `core.enums`, `core.ids` |
| `pipeline.py` | End-to-end frozen-workload execution (train→score→policy). | `FrozenModelEvidence`, `WorkloadExecution`, `ExecuteFrozenWorkload` | `application.federation_cell`, `application.policy_cell`, `application.score`, `application.train`, `config.models`, `core.enums`, `core.ids` |
| `policy_cell.py` | Materializes immutable policy evidence for one frozen federation evaluation. | `FrozenCacheInputs`, `PolicyCellMaterializer` | `application.evaluate`, `artifacts.dataset`, `artifacts.hashing`, `artifacts.layout`, `artifacts.references`, `artifacts.serialization`, `artifacts.training`, `config.models`, `core.enums`, `core.ids`, `metrics.results`, `scoring.cache` |
| `precompute.py` | Pre-data finite-sample protocol table generation. | `MismatchCutoffCell`, `ProtocolTablePrecomputer` | `artifacts.serialization`, `config.models`, `core.enums`, `core.types`, `experiments.registry`, `protocol.mismatch`, `protocol.readiness` |
| `preflight.py` | Pre-training research preflight audit. | `PreflightResult`, `ResearchPreflight` | `application.precompute`, `config.models`, `data.audit` |
| `prepare_data.py` | Freezes eligibility, base partitions, preprocessing, calibration assignments — big orchestration. | `PrepareData` | `artifacts.dataset`, `artifacts.hashing`, `artifacts.serialization`, `config.models`, `core.enums`, `core.exceptions`, `core.ids`, `core.logging`, `data.adapter`, `data.datasets.diad`, `data.datasets.nbaiot`, `data.eligibility`, `data.manifests`, `data.models`, `data.preprocessing`, `data.splitting` |
| `report.py` | Builds run/repository reports from immutable evidence (orchestration + rendering call-out). | `ReportBuilder` | `analysis.primary`, `application.verify`, `artifacts.layout`, `artifacts.verification`, `core.enums` |
| `research_pipeline.py` | High-level audited execution path (preflight + frozen workload) — third orchestration layer. | `ResearchExecution`, `ExecuteResearchPipeline` | `application.pipeline`, `application.preflight`, `config.models`, `core.enums` |
| `robustness.py` | Thin wrapper training a Deep-SVDD detector for robustness experiments. | `RunRobustness` | `application.train`, `config.models`, `core.enums` |
| `run_experiment.py` | Lifecycle orchestration for one immutable policy cell — yet another orchestration entry point. | `RunExperiment` | `artifacts.environment`, `artifacts.identity`, `artifacts.layout`, `artifacts.manifest`, `artifacts.serialization`, `artifacts.verification`, `config.models`, `core.enums`, `core.ids`, `experiments.lifecycle`, `experiments.models`, `experiments.planner` |
| `score.py` | Generates immutable score cache for one frozen model seed. | `ComputeScores` | `application.train`, `artifacts.dataset`, `artifacts.hashing`, `artifacts.training`, `config.models`, `core.enums`, `core.ids`, `core.logging`, `data.manifests`, `detectors.base`, `scoring.cache`, `scoring.computer`, `scoring.models` |
| `sensitivity.py` | Runs pre-registered real-score sensitivities R2-R9, R12. | `SensitivityCell`, `SensitivityEnvelope`, `MultiplicityCell`, `MultiplicityEnvelope`, `SourceOrderBlockCell`, `SourceOrderEnvelope`, `RunRealSensitivities` | `application.evaluate`, `artifacts.serialization`, `config.models`, `config.variants`, `core.enums`, `core.ids`, `experiments.models`, `experiments.real_sensitivity`, `experiments.registry`, `metrics.results`, `protocol.mismatch`, `protocol.readiness`, `protocol.results`, `scoring.cache`, `scoring.models`, `scoring.views` |
| `source_order.py` | R12 source-order calibration-role sensitivity over primary score cache. | `RunSourceOrderCalibration` | `application.evaluate`, `artifacts.serialization`, `config.models`, `core.enums`, `scoring.cache` |
| `synthetic.py` | Runs locked S1-S6 synthetic programme. | `SyntheticExperimentEnvelope`, `RunSyntheticExperiments` | `analysis.robustness`, `artifacts.serialization`, `config.models`, `core.enums`, `experiments.models`, `experiments.registry`, `experiments.synthetic` |
| `train.py` | Frozen detector training + immutable training-cache persistence; also defines `feature_columns`. | `feature_columns`, `TrainDetector` | `artifacts.dataset`, `artifacts.hashing`, `artifacts.training`, `config.models`, `core.constants`, `core.enums`, `core.ids`, `detectors.base`, `detectors.deep_svdd`, `detectors.factory`, `federated.trainer` |
| `verify.py` | Run-level and repository-wide reproducibility verification. | `PrecomputeVerification`, `RunVerification`, `RepositoryVerification`, `VerifyOutputs` | `artifacts.layout`, `artifacts.manifest`, `artifacts.verification`, `core.enums`, `experiments.completion`, `experiments.registry`, `protocol.mismatch`, `protocol.readiness` |

**Flag — three overlapping orchestration frameworks in `application/`:** `run_experiment.py` (`RunExperiment`), `pipeline.py` (`ExecuteFrozenWorkload`), and `research_pipeline.py` (`ExecuteResearchPipeline`) all orchestrate the same train→score→evaluate spine at different granularities, plus `federation_cell.py`/`policy_cell.py` add two more materializer layers. This is exactly the "application pipeline / research pipeline / workload runner / pipeline runner" anti-pattern the prompt calls out; must collapse into the single `pipeline/` spine. `precompute.py`, `preflight.py` also belong to `pipeline/`. `robustness.py` (25 lines) is a thin pass-through around `TrainDetector` — candidate for merge/removal.

### `artifacts/` (target: consolidate to 6 files: `paths.py, manifests.py, records.py, json_io.py, integrity.py, environment.py`)

| File | Responsibility | Public symbols | Internal imports |
|---|---|---|---|
| `__init__.py` | Empty. | — | none |
| `dataset.py` | Persistence/hashing for prepared-dataset evidence (3 manifest stores). | `PreparedDatasetManifestStore`, `EligibilityManifestStore`, `CalibrationAssignmentManifestStore` | `artifacts.serialization`, `core.enums`, `core.ids`, `data.manifests`, `data.models` |
| `environment.py` | Captures reproducibility environment/repo-state (git info). | `capture_environment` | `artifacts.hashing` |
| `environment_lock.py` | Freezes validated Python environment to a lock artifact. | `EnvironmentLock`, `EnvironmentLocker` | `artifacts.hashing`, `artifacts.serialization`, `core.ids` |
| `experiment_results.py` | Typed envelope for experiment-level evidence outside run dirs. | `ExperimentResultEnvelope` | `artifacts.serialization`, `core.enums`, `core.ids` |
| `hashing.py` | File hashing (`sha256_file`). | `sha256_file` | none |
| `identity.py` | Builds immutable scientific run identities (`RunId` construction). | `RunIdentityFactory` | `config.models`, `core.enums`, `core.ids` |
| `layout.py` | Canonical output layout for one policy/run cell. | `RunLayout` | `core.ids` |
| `manifest.py` | Typed run manifest + completed-run immutability contract. | `RunManifest`, `RunManifestStore` | `artifacts.serialization`, `core.enums`, `core.exceptions`, `core.ids` |
| `preprocessing.py` | Typed persistence for frozen preprocessing evidence. | `PreprocessingManifestStore` | `artifacts.serialization`, `core.enums`, `core.ids`, `data.preprocessing` |
| `records.py` | Normalized threshold/metric evidence records (JSONL writer). | `ThresholdRecord`, `MetricRecord`, `write_jsonl` | `artifacts.serialization`, `core.enums`, `core.ids` |
| `references.py` | Verified references from run evidence to reusable cache artifacts. | `CacheReference`, `CacheReferenceStore` | `artifacts.hashing`, `artifacts.serialization`, `core.ids` |
| `serialization.py` | Atomic write + JSON-value coercion helpers. | `JsonValue`(type), `to_json_value`, `atomic_write_text`, `atomic_write_json`, `as_json_list`, `as_json_dict`, `as_json_int`, `as_json_float` | `core.ids` |
| `training.py` | Typed persistence for frozen detector training evidence. | `ClientTrainingCount`, `TrainingManifest`, `TrainingManifestStore` | `artifacts.serialization`, `core.enums`, `core.ids`, `federated.models` |
| `verification.py` | Cryptographic + semantic verification of run evidence. | `FileHashRecord`, `VerificationResult`, `ArtifactVerifier` | `artifacts.hashing`, `artifacts.layout`, `artifacts.manifest`, `artifacts.references`, `artifacts.serialization`, `core.enums`, `experiments.models` |

**Flag:** 14 files vs. target 6. `dataset.py`, `manifest.py`, `preprocessing.py`, `training.py`, `references.py`, `records.py`, `experiment_results.py` are all "one manifest-store class per artifact type" — exactly the pattern prompt.md forbids ("Do not maintain one file for every individual artifact dataclass"). `identity.py` is named per the forbidden-vague-name list (`identity.py`) and its responsibility (`RunIdentityFactory`) maps to `paths.py`/`manifests.py` territory. `layout.py` → `paths.py`. `hashing.py` + `serialization.py` → `json_io.py`/`integrity.py`. `environment.py` + `environment_lock.py` → single `environment.py` (currently duplicated concept: capture vs. lock, could stay split or merge per actual need).

### `cli/`

| File | Responsibility | Public symbols | Internal imports |
|---|---|---|---|
| `__init__.py` | Empty. | — | none |
| `claims.py` | Claim-gate CLI commands. | `claims_group`, `claims_evaluate` | `application.claims` |
| `data.py` | Dataset prep CLI commands. | `data_group`, `prepare_data`, `prepare_feature_sensitivity` | `application.prepare_data`, `cli.shared`; local import `application.feature_sensitivity` |
| `environment.py` | Environment lock-file CLI commands. | `environment_group`, `freeze_environment` | `artifacts.environment_lock` |
| `evaluation.py` | Evaluation + report CLI commands. | `evaluate_command`, `report_group`, `report_build`, `report_build_repository`, `report_build_publication` | `application.evaluate`, `application.report`, `artifacts.serialization`, `cli.shared`; local import `analysis.publication` |
| `experiments.py` | Experiment planning/execution CLI commands. | `experiment_group`, `plan_experiment`, `run_policy_cell`, `materialize_federation_cell`, `execute_grid` | `cli.shared`, `core.enums`, `experiments.planner`; local imports `application.policy_cell`, `application.run_experiment`, `application.federation_cell`, `application.research_pipeline` |
| `main.py` | Top-level `click` group; registers all sub-groups + `doctor`/`config` commands. | `cli`, `doctor`, `config_group`, `validate_config` | `cli.claims`, `cli.data`, `cli.environment`, `cli.evaluation`, `cli.experiments`, `cli.research`, `cli.shared`, `cli.training`, `cli.verification`, `core.logging` |
| `research.py` | Statistical precompute, synthetic experiments, robustness, benchmarking CLI commands. | `tables_group`, `precompute_readiness`, `synthetic_group`, `synthetic_run`, `robustness_group`, `train_deep_svdd`, `benchmark_command`, `sensitivity_group`, `sensitivity_run` | `application.benchmark`, `application.precompute`, `application.robustness`, `application.synthetic`, `cli.shared`; local imports `application.sensitivity`, `application.source_order` |
| `shared.py` | `load_config` used by every CLI module. | `load_config` | `config.models`, `config.resolver`, `config.validation` |
| `training.py` | Train/score CLI commands. | `train_command`, `score_command` | `application.score`, `application.train`, `cli.shared` |
| `verification.py` | Repository verification CLI. | `verify_command` | `application.verify` |

**Flag:** name `research.py` is on prompt.md's explicit forbidden-vague-name list; target tree splits it into `cli/scoring.py`, `cli/benchmark.py`, and folds precompute/synthetic/robustness into domain-appropriate CLI modules. `shared.py` is also a forbidden name (target keeps CLI helpers inline per-command or in `runtime.py`). Heavy local (function-body) imports of `application.*` throughout `experiments.py`/`research.py`/`data.py`/`evaluation.py` — these appear to be avoiding circular imports/startup cost, a smell worth resolving once `application/` collapses into `pipeline/`.

### `config/`

| File | Responsibility | Public symbols | Internal imports |
|---|---|---|---|
| `__init__.py` | Empty. | — | none |
| `loader.py` | YAML loading utility. | `load_yaml` | `core.exceptions` |
| `models.py` | All Pydantic config models (frozen). | `FrozenModel`, `ProtocolConfig`, `SplitConfig`, `DatasetConfig`, `AutoencoderConfig`, `DeepSvddConfig`, `DetectorConfig`(type alias), `TrainingConfig`, `RandomnessConfig`, `ExperimentConfig` | `core.constants`, `core.enums`, `core.types` |
| `resolver.py` | Resolves composable YAML docs into one `ExperimentConfig`. | `ExperimentConfigResolver` | `config.loader`, `config.models`, `core.enums`, `core.exceptions` |
| `validation.py` | Cross-model validation not expressible in Pydantic. | `validate_experiment_config` | `config.models`, `core.constants`, `core.enums`, `core.exceptions`, `policies.registry` |
| `variants.py` | Constructs pre-registered experiment config variants. | `ExperimentVariantFactory` | `config.models`, `config.validation`, `core.enums` |

**Flag:** `models.py` is a monolithic file covering dataset/training/method/experiment config together — target splits into `dataset_config.py`, `training_config.py`, `method_config.py`, `experiment_config.py`. `models.py` is also on the forbidden-vague-name list. `variants.py` is explicitly forbidden by name in prompt.md. `config/validation.py` importing `policies.registry` is a downward dependency into a package config precedes conceptually — worth checking against target dependency order (config → data/detectors/... ; policies/thresholds sit below config, so config depending on policies is backwards and should be inverted or the validation logic relocated).

### `core/` (target: **removed**, folded into `domain/` + `runtime.py`)

| File | Responsibility | Public symbols | Internal imports |
|---|---|---|---|
| `__init__.py` | Empty. | — | none |
| `constants.py` | Locked protocol-level numeric constants. | `NBAIOT_CLIENT_IDS`, `NBAIOT_EXPECTED_FEATURES`, `NBAIOT_EXPECTED_SOURCE_CLIENTS`, `NBAIOT_AUTOENCODER_PARAMETERS`, `NBAIOT_AUTOENCODER_BYTES`, `NBAIOT_MODEL_COMMUNICATION_BYTES`, `DIAD_EXPECTED_FEATURES`, `DIAD_EXPECTED_SOURCE_CLIENTS`, `DIAD_MINIMUM_ELIGIBLE_CLIENTS`, `DIAD_AUTOENCODER_PARAMETERS`, `DIAD_AUTOENCODER_BYTES`, `ATTACK_DEVELOPMENT_SEED`, `SHRINKAGE_N0_CANDIDATES`, `SUPERVISED_THRESHOLD_CANDIDATES`, `PRIMARY_MODEL_SEEDS`, `DEEP_SVDD_MODEL_SEEDS`, `SYNTHETIC_MASTER_SEED`, `BOOTSTRAP_SEED`, `BOOTSTRAP_REPLICATES` | none |
| `enums.py` | All domain `StrEnum` definitions (30 enums). | `ProtocolId`, `DatasetId`, `DatasetFeatureContractId`, `DetectorId`, `DataRole`, `CalibrationAssignmentMode`, `CalibrationReadinessState`, `MismatchOutcome`, `DecisionState`, `ThresholdSource`, `DecisionReason`, `FailureCode`, `EligibilityStatus`, `ActivationId`, `ComputeDeviceId`, `DeepSvddCenterMode`, `ChronologyStatus`, `AggregationId`, `OptimizerId`, `PolicyId`, `PolicyEvaluationStatus`, `ExperimentId`, `ExperimentCode`, `ExperimentAxisId`, `SyntheticDistribution`, `ContaminationDirection`, `MultiplicityProcedure`, `ExperimentType`, `ExperimentStatus`, `ArtifactType`, `ClaimLevel` | none |
| `exceptions.py` | Domain exception hierarchy. | `FedCRGError`, `ConfigurationError`, `DataIntegrityError`, `ImmutableRunError` | none |
| `ids.py` | Strong identifier value objects. | `ClientId`, `RowId`, `AttackGroupId`, `Sha256`, `RunId`, (also `CalibrationSeed`/`ModelSeed` referenced elsewhere — verify defined here) | none |
| `logging.py` | Process-wide logging configuration + `log_stage` context manager. | `configure_logging`, `get_logger`, `log_stage` | none |
| `types.py` | Small shared value objects. | `OperatingBand`, `ConfidenceInterval` | none |

Direct 1:1 mapping to target: `constants.py`→`domain/constants.py`, `enums.py`→`domain/enums.py`, `ids.py`+`types.py`→`domain/identifiers.py`+`domain/values.py` (need split of ID vs. value-object types), `exceptions.py`→`domain/errors.py`, `logging.py`→`runtime.py` (it's process/runtime behavior, not domain). No flags beyond needing the package rename/relocation itself.

### `data/`

| File | Responsibility | Public symbols | Internal imports |
|---|---|---|---|
| `__init__.py` | Empty. | — | none |
| `adapter.py` | Abstract dataset-adapter boundary. | `DatasetAdapter`(ABC) | `core.enums`, `core.ids`, `data.models` |
| `audit.py` | Independent audit of prepared-dataset caches pre-training. | `PreparedDataAudit`, `PreparedDatasetAuditor` | `artifacts.serialization`, `config.models`, `core.constants`, `core.enums`, `core.ids` |
| `datasets/__init__.py` | Re-exports `DiadAdapter`, `NBaiotAdapter`. | — | `data.datasets.diad`, `data.datasets.nbaiot` |
| `datasets/diad.py` | CIC IoT-DIAD adapter + locked 86-feature contract. | `DIAD_FEATURES`, `DiadAdapter`, `DiadFeatureSensitivityAdapter` | `core.constants`, `core.enums`, `core.exceptions`, `core.ids`, `core.logging`, `data.adapter`, `data.discovery`, `data.models`, `data.splitting` |
| `datasets/nbaiot.py` | N-BaIoT adapter with canonical device mapping. | `NBaiotAdapter` | `core.constants`, `core.enums`, `core.exceptions`, `core.ids`, `data.adapter`, `data.discovery`, `data.models`, `data.splitting` |
| `discovery.py` | Filesystem discovery helpers for adapters. | `DatasetDiscovery` | `core.exceptions` |
| `eligibility.py` | Pre-outcome client eligibility evaluation. | `ClientEligibilityEvaluator` | `config.models`, `core.enums`, `data.datasets.diad`, `data.models` |
| `feature_sensitivity.py` | DIAD R14 numeric-safe feature derivation. | `ClientTrainingRowHash`, `NumericSafeFeatureContract`, `derive_numeric_safe_features` | `core.ids`, `data.manifests` |
| `integrity.py` | Split disjointness validation. | `validate_split_disjointness` | `core.enums`, `core.exceptions`, `data.models` |
| `manifests.py` | Typed dataset/eligibility/split/cache provenance models. | `SourceFileManifest`, `EligibilityManifest`, `CalibrationRoleManifest`, `ClientCalibrationManifest`, `CalibrationAssignmentManifest`, `RoleArtifactManifest`, `ClientDatasetManifest`, `CalibrationAssignmentReference`, `PreparedDatasetManifest`, `hash_file`, `source_file_manifest`, `hash_row_ids` | `core.enums`, `core.ids`, `data.models` |
| `models.py` | Typed client data/partitions/calibration assignments/eligibility records. | `ClientData`, `RoleFrame`, `ClientSplits`, `RolePositions`, `CalibrationRoleAssignment`, `EligibilityRecord` | `core.enums`, `core.ids` |
| `preprocessing.py` | Frozen train-only preprocessing for detector inputs. | `model_feature_columns`, `ClientPreprocessingParameters`, `PreprocessingModel`, `ClientPreprocessingStatistics`, `FederatedPreprocessor` | `core.enums`, `core.exceptions`, `core.ids`, `data.models`, `data.manifests` |
| `splitting.py` | Deterministic base partitioning + calibration-reservoir assignment. | `hash_seed`, `calibration_rng`, `attack_rng`, `stable_row_id`, `attack_group_counts`, `allocate_nbaiot_attack_development`, `allocate_diad_attack_development`, `CalibrationAssignmentBuilder`, `DataSplitter` | `config.models`, `core.constants`, `core.enums`, `core.exceptions`, `core.ids`, `data.integrity`, `data.manifests`, `data.models` |

`data/models.py` name is on prompt.md's forbidden-vague-list, though target tree does keep `data/` mostly flat (`nbaiot.py`, `diad.py`, `prepare.py`, `splits.py`, `preprocessing.py`, `eligibility.py`, `feature_sensitivity.py`) — no `models.py`/`manifests.py`/`adapter.py`/`audit.py`/`discovery.py`/`integrity.py` listed explicitly; these need to be folded into the listed files or justified as cohesive exceptions per the "do not artificially split a cohesive implementation" clause.

### `detectors/`

| File | Responsibility | Public symbols | Internal imports |
|---|---|---|---|
| `__init__.py` | Empty. | — | none |
| `autoencoder.py` | Feed-forward autoencoder detector. | `activation_module`, `Autoencoder` | `config.models`, `core.enums`, `detectors.base` |
| `base.py` | Detector model contract (ABC). | `DetectorModel` | none |
| `deep_svdd.py` | Deep-SVDD one-class detector. | `DeepSvdd` | `config.models`, `core.enums`, `detectors.base` |
| `factory.py` | Detector construction from config. | `DetectorFactory` | `config.models`, `detectors.autoencoder`, `detectors.base`, `detectors.deep_svdd` |

Maps cleanly to target (`detector.py` ← `base.py`, `create_detector.py` ← `factory.py`). `base.py` and `factory.py` are both forbidden-name-list entries by name, though target explicitly keeps `create_detector.py` (renamed) and a `detector.py` (renamed from `base.py`) — so the rename is expected, not a foundational architecture problem.

### `experiments/`

| File | Responsibility | Public symbols | Internal imports |
|---|---|---|---|
| `__init__.py` | Empty. | — | none |
| `completion.py` | Reconciles generated evidence against the expected workload roadmap. | `ExperimentCompletion`, `ExperimentCompletionAuditor` | `artifacts.layout`, `artifacts.manifest`, `core.enums`, `experiments.definitions`, `experiments.registry` |
| `definitions.py` | Full pre-registered S1-S6/R1-R14 experiment catalogue. | `ALL_POLICIES`, `SECOND_DETECTOR_POLICIES`, `PRIMARY_REQUIRED`, `axis`, `setting`, `target_fpr_cell`, `definitions` | `core.enums`, `experiments.models` |
| `dependencies.py` | Evaluates dependencies + deterministic topological order. | `DependencyResolver` | `core.enums`, `experiments.registry` |
| `executor.py` | Executes experiments with dependency/status semantics — **second orchestration engine**. | `ExperimentExecutor` | `core.enums`, `experiments.dependencies`, `experiments.lifecycle`, `experiments.models`, `experiments.registry` |
| `lifecycle.py` | Allowed experiment status transitions. | `assert_transition` | `core.enums` |
| `models.py` | Typed experiment catalogue/parameter grid/plan/execution-state models. | `ParameterAxis`, `ParameterSetting`, `ParameterCell`, `WorkloadExpectation`, `ExperimentDefinition`, `ExperimentPlan`, `ExperimentExecution`(Generic) | `core.enums`, `core.ids`; local import `experiments.lifecycle` |
| `planner.py` | Builds an `ExperimentPlan` from a validated config. | `ExperimentPlanner` | `config.models`, `core.enums`, `core.ids`, `experiments.models`, `experiments.registry` |
| `real_sensitivity.py` | Real-score sensitivity helper kernels (R2-R9, R12). | `source_order_blocks`, `contaminate_benign_scores` | none |
| `registry.py` | Lookup/validate the frozen experiment catalogue. | `ExperimentRegistry` | `core.enums`, `experiments.definitions`, `experiments.models` |
| `synthetic.py` | Statistical kernels for pre-registered synthetic experiments. | `SyntheticCoverageResult`, `MismatchPowerResult`, `draw_distribution`, `distribution_cdf`, `iid_readiness_validation`, `contamination_validation`, `exact_mismatch_power` | `core.enums`, `core.types`, `protocol.mismatch`, `protocol.readiness` |

**Flag:** `experiments/executor.py` (`ExperimentExecutor`) appears to be **dead/unused** — it is never imported by `application/*`, `cli/*`, or tests except `tests/unit/experiments/test_models.py`, which is the only consumer. This is a second, unused orchestration engine parallel to `application.run_experiment.RunExperiment`. Candidate for removal or consolidation into `pipeline/run_experiment.py`. `registry.py` name is forbidden by prompt.md's vague-name list, though target tree doesn't list a `registry.py` under `experiments/` at all — its responsibility (`ExperimentRegistry`) must fold into `experiments/experiment_definition.py` or `execution.py`.

### `federated/`

| File | Responsibility | Public symbols | Internal imports |
|---|---|---|---|
| `__init__.py` | Empty. | — | none |
| `aggregation.py` | Equal-client parameter aggregation. | `equal_client_mean`, `EqualMeanAggregator` | `detectors.base` |
| `client.py` | Local client training with deterministic per-epoch shuffling. | `epoch_seed`, `FederatedClient` | `config.models`, `core.ids`, `detectors.autoencoder`, `detectors.base`, `detectors.deep_svdd`, `federated.models` |
| `models.py` | Typed federated-training diagnostics/results. | `ClientRoundResult`, `RoundResult`, `TrainingResult` | `core.ids` |
| `sampling.py` | Deterministic client participation sampling. | `ClientSampler` | `core.ids` |
| `scheduling.py` | Learning-rate scheduling (cosine). | `cosine_learning_rate` | none |
| `server.py` | Federated server state + aggregation call-out. | `FederatedServer` | `detectors.base`, `federated.aggregation` |
| `trainer.py` | Deterministic federated training orchestration. | `FederatedTrainer` | `config.models`, `core.ids`, `core.logging`, `detectors.base`, `federated.client`, `federated.models`, `federated.sampling`, `federated.scheduling`, `federated.server` |

Maps cleanly onto target `federation/` (rename package only; `models.py`→`training_results.py`, `sampling.py`→`participation.py`, `scheduling.py`→`learning_rate.py`). No duplication found.

### `metrics/` (target: replaced by `evaluation/`)

| File | Responsibility | Public symbols | Internal imports |
|---|---|---|---|
| `__init__.py` | Empty. | — | none |
| `admission.py` | Federation summaries of operating-point admission states. | `AdmissionSummary`, `summarize_admission` | `core.enums`, `protocol.results` |
| `attack_balanced.py` | Attack-group-balanced utility metric. | `attack_balanced_tpr` | none |
| `classification.py` | Strict-threshold classification metrics (confusion matrix, fpr/tpr/precision/recall/f1/balanced_accuracy). | `ConfusionMatrix`, `confusion_matrix`, `fpr`, `tpr`, `precision`, `recall`, `f1`, `balanced_accuracy` | none |
| `federation.py` | Equal-client federation aggregation + utility-margin evaluation. | `LOCKED_UTILITY_MARGIN`, `utility_margin_satisfied`, `aggregate_policy`, `utility_anchor`, `utility_preserved`, `assert_ranking_metric_invariance`, `assess_utility` | `core.enums`, `core.ids`, `metrics.results` |
| `operating_band.py` | Operating-band reliability metrics. | `band_error`, `high_excess`, `band_violation`, `absolute_fpr_error` | `core.types` |
| `ranking.py` | Threshold-independent ranking metrics. | `auroc`, `auprc` | none |
| `results.py` | Typed client/federation metric records. | `ClientMetrics`, `PolicyEvaluation`, `FederationMetrics`, `EvaluationBundle`, `UtilityAssessment` | `core.enums`, `core.ids`, `core.types`, `protocol.results` |

Maps well onto target `evaluation/`: `classification.py`→`classification_metrics.py`+`confusion_matrix.py`, `operating_band.py`→`operating_band_metrics.py`, `attack_balanced.py`→`attack_balanced_metrics.py`, `admission.py`→`admission_metrics.py`, `ranking.py`→`ranking_metrics.py`, `federation.py`→`federation_evaluation.py`, `results.py`→`evaluation_results.py`. `results.py` name is forbidden-list; must rename.

### `policies/` (target: split into `thresholds/` comparators + `method/` for FedCRG itself)

| File | Responsibility | Public symbols | Internal imports |
|---|---|---|---|
| `__init__.py` | Empty. | — | none |
| `attack_aware.py` | Locked supervised-development comparators (dev-local/global F1, summary-statistic, supervised-global-F1). | `dev_local_global`, `summary_statistic_threshold`, `supervised_global_f1` | `core.constants`, `metrics.classification`, `policies.base` |
| `base.py` | Typed policy evidence views + shared finite-sample quantile rule. | `BenignPolicyEvidence`, `SupervisedDevelopmentEvidence`, `FinalTestEvidence`, `empirical_quantile` | `core.ids`, `protocol.results` |
| `oracle.py` | Final-test oracle (unattainable diagnostic ceiling). | `oracle_choice` | `core.types`, `metrics.classification`, `metrics.operating_band`, `policies.base` |
| `personalized.py` | Benign-only ablations (readiness-only, mismatch-only). | `readiness_only`, `mismatch_only` | `core.enums`, `policies.base` |
| `quantile.py` | Benign-only global/local/three-sigma quantile comparators. | `global_quantile`, `local_quantile`, `three_sigma` | `policies.base` |
| `registry.py` | The one authoritative policy registry + federation policy selector. | `InformationRegime`, `PolicyDefinition`, `ClientPolicyThreshold`, `UndefinedPolicyReason`, `PolicyThresholdSet`, `PolicyRegistry`, `FederationPolicySelector` | `config.models`, `core.enums`, `core.ids`, `policies.attack_aware`, `policies.base`, `policies.personalized`, `policies.quantile`, `policies.shrinkage` |
| `shrinkage.py` | Threshold-space partial-pooling comparator + tuning rule. | `tune_shrinkage`, `shrinkage` | `core.constants`, `policies.base` |

**Flag:** `registry.py` name is doubly forbidden — it's both on the vague-name list and prompt.md says explicitly "Do not create a policy registry. Selection should be explicit and typed." This whole file must be dissolved into typed comparator selection under `thresholds/`. `base.py` also forbidden-name. No file here contains the actual FedCRG method (that's in `protocol/`) — confirms prompt.md's complaint that "FedCRG itself must not be hidden among generic policy implementations"; `protocol/` and `policies/` are two halves of one conceptual pipeline that the target explicitly separates into `method/` (FedCRG) vs `thresholds/` (comparators), which is *not* how the current split is drawn (current split is protocol-vs-policy, not method-vs-comparators — needs conscious remap, not a blind rename).

### `protocol/` (target: replaced by `method/`)

| File | Responsibility | Public symbols | Internal imports |
|---|---|---|---|
| `__init__.py` | Empty. | — | none |
| `decision.py` | Deterministic threshold-deployment decision state machine. | `ThresholdDecisionEngine` | `core.enums`, `protocol.results` |
| `mismatch.py` | Exact benign reference-mismatch evidence + fleet-level sensitivity (Bonferroni/Holm). | `clopper_pearson_interval`, `minimum_bidirectional_sample_count`, `ReferenceMismatchEvaluator`, `FleetMismatchDecision`, `bonferroni_fleet_sensitivity`, `holm_directional_fleet_sensitivity` | `core.enums`, `core.ids`, `core.types`, `protocol.results` |
| `readiness.py` | Finite-sample local-readiness planning, lookup cache, continuity diagnostics. | `coverage_probability`, `ReadinessPlanBuilder`, `ReadinessPlanCache`, `CalibrationReadinessEvaluator`, `continuity_diagnostics`, `familywise_readiness_assurance` | `core.enums`, `core.types`, `protocol.results` |
| `reference.py` | Federation reference-threshold estimation. | `reference_rank`, `ReferenceThresholdEstimator` | `core.ids`, `protocol.results` |
| `results.py` | Typed outputs for the operating-point governance protocol. | `ReferenceThreshold`, `ReadinessPlan`, `ContinuityDiagnostics`, `CalibrationReadiness`, `MismatchEvidence`, `ThresholdDecision`, `ClientProtocolResult` | `core.enums`, `core.ids`, `core.types` |
| `service.py` | Composition service wiring reference/mismatch/readiness/decision together. | `FedCRGProtocol` | `config.models`, `core.ids`, `protocol.decision`, `protocol.mismatch`, `protocol.readiness`, `protocol.reference`, `protocol.results` |

Maps well onto target `method/`: `reference.py`→`reference_threshold.py`, `readiness.py`→`calibration_readiness.py`, `mismatch.py`→`mismatch_detection.py`, `decision.py`→`threshold_decision.py`, `service.py`→`client_evaluation.py`, `results.py`→`results.py` (already good name here, matches target). `service.py` and `results.py` are both forbidden-vague-names per the generic list, but `results.py` is explicitly kept in the target tree for `method/results.py`, so only `service.py` needs rename/dissolution (its content, `FedCRGProtocol`, should become the `client_evaluation.py` step).

### `scoring/`

| File | Responsibility | Public symbols | Internal imports |
|---|---|---|---|
| `__init__.py` | Re-exports `ScoreCache`, `ScoreManifest`, `CalibrationScoreViewBuilder`, `CalibrationScoreViews`. | `ScoreCache`, `ScoreManifest`, `CalibrationScoreViewBuilder`, `CalibrationScoreViews` | `scoring.cache`, `scoring.models`, `scoring.views` |
| `cache.py` | Atomic, hash-finalized Parquet score-cache persistence. | `ScoreCacheIdentity`, `ScoreRoleCacheRecord`, `ScoreCacheDescriptor`, `ScoreCache` | `artifacts.hashing`, `artifacts.serialization`, `core.enums`, `core.exceptions`, `core.ids`, `scoring.integrity`, `scoring.models` |
| `computer.py` | Frozen-model anomaly-score computation. | `ScoreComputer` | `core.enums`, `core.ids`, `detectors.base`, `scoring.models` |
| `integrity.py` | Score-cache integrity + detector-invariance checks. | `validate_score_manifest` | `core.enums`, `core.ids`, `scoring.models` |
| `models.py` | Typed anomaly-score inputs/immutable score-cache contents. | `RoleScoreInput`, `ClientScoreInput`, `RoleScores`, `ClientScoreSet`, `ScoreManifest` | `core.enums`, `core.ids` |
| `views.py` | Deterministic calibration-role views over one frozen score cache. | `ClientCalibrationScores`, `CalibrationScoreViews`, `CalibrationScoreViewBuilder`, `truncate_view` | `artifacts.dataset`, `config.models`, `core.enums`, `core.ids`, `data.manifests`, `data.splitting`, `scoring.cache`, `scoring.models` |

**Flag:** `scoring/__init__.py` re-exports symbols — mild `__init__.py`-as-dumping-ground pattern the prompt warns against ("`__init__.py` files must not become compatibility/re-export dumping grounds"); this is the only package `__init__.py` in the repo that isn't empty. `models.py` is forbidden-vague-name; target renames to `calibration_scores.py`/`score_records.py`. Maps otherwise cleanly: `cache.py`✓, `computer.py`→`compute.py`, `integrity.py`→`validation.py`, `views.py`→ split across `calibration_scores.py`/`score_records.py`.

---

## 2. Test inventory

### `tests/contract/` — architectural/domain invariant enforcement (should mostly survive rewritten against new module paths)

| File | Tests | Exercises |
|---|---|---|
| `test_architecture_boundaries.py` | AST-based scan: forbidden source fragments (TODO/FIXME/personal paths/`__import__`), legacy dirs absent. | filesystem/AST scan of `src/fedcrg`, no runtime imports |
| `test_cache_identity.py` | Config-resolver-driven cache identity behavior. | `config.resolver` |
| `test_configuration_profiles.py` | Cross-profile config resolution + validation for all dataset/detector/experiment combos. | `config.resolver`, `config.validation`, `core.constants`, `core.enums` |
| `test_deep_svdd_center_contract.py` | Deep-SVDD center computation contract. | `config.models`, `detectors.deep_svdd` |
| `test_diad_label_firewall.py` | DIAD adapter never leaks label columns into features. | `data.datasets.diad` |
| `test_domain_immutability.py` | AST scan: dataclasses that should be frozen are frozen, no mutable-container annotations outside declared boundaries. | AST scan of `src/fedcrg` |
| `test_domain_value_types.py` | Field types on manifests/cache records are domain value types, not primitives. | `artifacts.manifest`, `artifacts.references`, `core.ids`, `scoring.cache`, `scoring.models` |
| `test_enums.py` | Enum membership/values sanity. | `core.enums` |
| `test_experiment_catalogue_exact.py` | Exact experiment code catalogue match. | `core.enums`, `experiments.registry` |
| `test_experiment_registry.py` | Registry lookup/validation. | `core.enums`, `experiments.registry` |
| `test_frozen_readiness_runtime_contract.py` | Readiness plan cache frozen-contract behavior. | `core.types`, `protocol.readiness` |
| `test_information_boundaries.py` | Dataclass field introspection: no cross-role information leakage. | `policies.base`, `protocol.service` |
| `test_no_legacy_algorithm_names.py` | Forbids specific legacy filenames (`gate_a.py`, `gate_b.py`, `states.py`, `fedcrg.py`). | filesystem scan |
| `test_output_layout_contract.py` | Output directory layout contract. | `artifacts.layout`, `core.ids` |
| `test_preprocessing_artifact_contract.py` | Preprocessing manifest contract. | `artifacts.preprocessing` |
| `test_protocol_numerical_ledger.py` | Numerical exactness of protocol computations (mismatch/readiness/reference). | `core.enums`, `core.types`, `protocol.mismatch`, `protocol.readiness`, `protocol.reference` |
| `test_repository_hygiene.py` | Forbidden filenames/dirs, no personal paths, no semicolon-compressed statements. | filesystem scan of `src/fedcrg` |
| `test_run_identity_uniqueness.py` | Run identity uniqueness across configs/seeds. | `application.feature_sensitivity`, `artifacts.identity`, `config.resolver`, `core.enums`, `core.ids`, `data.feature_sensitivity` |

### `tests/integration/`

| File | Tests | Exercises |
|---|---|---|
| `test_application_run.py` | End-to-end `RunExperiment` lifecycle. | `application.run_experiment`, `artifacts.manifest`, `config.models`, `core.enums`, `experiments.models` |
| `test_artifact_verification.py` | Run/layout/manifest verification pipeline. | `artifacts.layout`, `artifacts.manifest`, `artifacts.serialization`, `artifacts.verification`, `core.enums`, `core.ids`, `experiments.models` |
| `test_cli.py` | CLI smoke tests via `CliRunner`. | `cli.main` (and transitively everything it wires) |
| `test_config_resolution.py` | Config resolve+validate integration. | `config.resolver`, `config.validation`, `core.enums` |
| `test_end_to_end_cache_pipeline.py` | Full train→score→evaluate cache pipeline smoke test. | `application.evaluate`, `application.score`, `application.train`, `artifacts.dataset`, `artifacts.hashing`, `artifacts.serialization`, `config.models`, `core.constants`, `core.enums`, `core.ids`, `data.manifests`, `protocol.readiness` |
| `test_outputs.py` | Output layout + manifest immutability behaviors. | `artifacts.layout`, `artifacts.manifest`, `core.enums`, `core.exceptions`, `core.ids` |
| `test_prepare_data.py` | `PrepareData` orchestration integration. | `application.prepare_data`, `config.models`, `core.enums`, `core.ids`, `data.adapter`, `data.models` |

### `tests/regression/`

| File | Tests | Exercises |
|---|---|---|
| `test_decision_regressions.py` | Regression cases for prior threshold-decision defects. | `core.enums`, `core.types`, `protocol.decision`, `protocol.results` |
| `test_dependency_regression.py` | Experiment dependency-resolution regression. | `core.enums`, `experiments.dependencies`, `experiments.registry` |
| `test_federated_server_update_contract.py` | Federated server update-contract regression. | `detectors.base`, `federated.server` |
| `test_r14_stream_feature_regression.py` | R14 numeric-safe feature derivation regression. | `core.ids`, `data.feature_sensitivity` |

### `tests/unit/` (per-module unit tests)

| File | Tests | Exercises |
|---|---|---|
| `unit/analysis/test_communication_and_stability.py` | Communication accounting + stability summaries. | `analysis.communication`, `analysis.stability`, `config.resolver`, `core.enums` |
| `unit/analysis/test_tables_sensitivity.py` | Publication table builder sensitivity rows. | `analysis.tables` |
| `unit/data/test_dataset_contracts.py` | DIAD/N-BaIoT adapter contracts. | `core.constants`, `core.enums`, `core.ids`, `data.datasets.diad`, `data.datasets.nbaiot` (reaches into private `_CANONICAL_DEVICES`) |
| `unit/data/test_diad_splitting.py` | DIAD-specific splitting behavior. | `config.models`, `core.constants`, `core.enums`, `core.ids`, `data.models`, `data.splitting` |
| `unit/data/test_preprocessing.py` | Federated preprocessing. | `core.enums`, `core.exceptions`, `core.ids`, `data.models`, `data.preprocessing` |
| `unit/data/test_splitting.py` | Base/calibration splitting. | `config.models`, `core.enums`, `core.ids`, `data.models`, `data.splitting` |
| `unit/detectors/test_architectures.py` | Autoencoder architecture. | `config.models`, `detectors.autoencoder` |
| `unit/experiments/test_completion.py` | Experiment completion auditing. | `core.enums`, `experiments.completion` |
| `unit/experiments/test_models.py` | Experiment models/dependencies/**executor**/registry. | `core.enums`, `core.ids`, `experiments.dependencies`, `experiments.executor`, `experiments.models`, `experiments.registry` (only consumer of `experiments.executor`) |
| `unit/federated/test_aggregation_deep_svdd.py` | Aggregation with Deep-SVDD/Autoencoder detectors. | `config.models`, `detectors.autoencoder`, `detectors.deep_svdd`, `detectors.factory`, `federated.aggregation` |
| `unit/federated/test_training.py` | Federated trainer + LR scheduling. | `config.models`, `core.ids`, `detectors.autoencoder`, `federated.scheduling`, `federated.trainer` |
| `unit/metrics/test_metrics.py` | Classification + operating-band metrics. | `core.types`, `metrics.classification`, `metrics.operating_band` |
| `unit/metrics/test_ranking_attack.py` | Ranking + attack-balanced metrics. | `metrics.attack_balanced`, `metrics.ranking` |
| `unit/policies/test_policies.py` | Policy evidence + registry selection. | `config.models`, `core.enums`, `core.ids`, `core.types`, `policies.base`, `policies.registry`, `protocol.results` |
| `unit/policies/test_registry.py` | Policy registry/information-regime behavior. | `core.enums`, `policies.registry` |
| `unit/protocol/test_decision.py` | Threshold decision engine. | `core.enums`, `core.types`, `protocol.decision`, `protocol.results` |
| `unit/protocol/test_mismatch.py` | Mismatch evaluator + sample-count math. | `core.enums`, `core.types`, `protocol.mismatch` |
| `unit/protocol/test_readiness.py` | Readiness evaluator/plan builder. | `core.enums`, `core.types`, `protocol.readiness` |
| `unit/protocol/test_reference.py` | Reference threshold estimator. | `core.ids`, `protocol.reference` |
| `unit/protocol/test_service.py` | `FedCRGProtocol` composition service. | `config.models`, `core.enums`, `core.ids`, `protocol.service` |
| `unit/scoring/test_cache.py` | Score cache persistence. | `core.enums`, `core.ids`, `scoring.cache`, `scoring.models` |
| `unit/scoring/test_computer.py` | Score computer. | `config.models`, `core.enums`, `core.ids`, `detectors.autoencoder`, `scoring.computer`, `scoring.models` |

**Test-layer flags:**
- No test file directly exercises most of `application/*` (`benchmark.py`, `federation_cell.py`, `pipeline.py`, `policy_cell.py`, `precompute.py`, `preflight.py`, `report.py`, `research_pipeline.py`, `sensitivity.py`, `source_order.py`, `synthetic.py`, `verify.py`) or most of `analysis/*` (`benchmark.py`, `claims.py`, `decision_architecture.py`, `figures.py`, `primary.py`, `publication.py`, `robustness.py`, `statistics.py`) — these orchestration/reporting layers are effectively covered only transitively via `test_cli.py` and `test_end_to_end_cache_pipeline.py` (which are smoke tests, not targeted coverage). Migration will need new targeted tests for the `pipeline/`, `analysis/`, and `reporting/` equivalents.
- `experiments.executor` is exercised only incidentally inside `unit/experiments/test_models.py` — consistent with the dead-code flag above.
- `unit/data/test_dataset_contracts.py` reaches into `nbaiot._CANONICAL_DEVICES`, a private module attribute — a minor test-hygiene smell (testing internals) to fix when the file is renamed/relocated.

---

## 3. Explicit flags

### 3.1 Duplicate/overlapping responsibilities

| Concept | Competing files | Notes |
|---|---|---|
| "Run everything" orchestration | `application/run_experiment.py:RunExperiment`, `application/pipeline.py:ExecuteFrozenWorkload`, `application/research_pipeline.py:ExecuteResearchPipeline`, `experiments/executor.py:ExperimentExecutor` | Four parallel orchestration entry points at different granularities; prompt.md explicitly forbids this pattern ("application pipeline / experiment executor / research pipeline / workload runner / pipeline runner"). |
| "Results"/"models" dataclass containers | `protocol/results.py`, `metrics/results.py`, `federated/models.py`, `data/models.py`, `experiments/models.py`, `scoring/models.py`, `config/models.py` | Seven distinct `results.py`/`models.py` files across packages, all forbidden-vague-names; each holds a legitimately distinct domain but the *naming convention itself* is the duplication (no reader can distinguish them by name alone). |
| FedCRG method vs. generic policies | `protocol/*` (method) vs `policies/*` (comparators) | Currently two separate packages that don't cleanly map to target's `method/` (FedCRG) vs `thresholds/` (comparators) split — `policies/base.py` and `policies/registry.py` reference `protocol/results.py` types, so the packages are already coupled and need conscious re-partition, not a rename. |
| Policy "registry" pattern | `policies/registry.py:PolicyRegistry`, `experiments/registry.py:ExperimentRegistry` | Two separate "registry" abstractions; prompt.md explicitly forbids "Do not create a policy registry. Selection should be explicit and typed," and forbids the word "registry" generally. |
| Environment capture vs. lock | `artifacts/environment.py:capture_environment`, `artifacts/environment_lock.py:EnvironmentLocker` | Both about environment reproducibility; target lists a single `environment.py`. |
| Sensitivity execution split across three files | `application/sensitivity.py`, `application/source_order.py`, `experiments/real_sensitivity.py` | R2-R9/R12 sensitivity logic spread across app-layer orchestration + experiments-layer kernels; target keeps sensitivity as `experiments/definitions/sensitivity.py`. |

### 3.2 Dead-code candidates

- `experiments/executor.py` (`ExperimentExecutor`) — only referenced by one test file (`tests/unit/experiments/test_models.py`); not used by any `application/*` or `cli/*` production path (which instead call `application.run_experiment.RunExperiment`).
- `application/robustness.py` (`RunRobustness`) — thin 25-line wrapper solely calling `TrainDetector`; only consumer is `cli/research.py:train_deep_svdd`. Candidate for inlining rather than a standalone file.

### 3.3 Compatibility-wrapper / re-export dumping grounds

- `scoring/__init__.py` — the only non-empty package `__init__.py` in `src/fedcrg`; re-exports `ScoreCache`, `ScoreManifest`, `CalibrationScoreViewBuilder`, `CalibrationScoreViews`. Violates "`__init__.py` files must not become compatibility/re-export dumping grounds."
- `data/datasets/__init__.py` — re-exports `DiadAdapter`, `NBaiotAdapter`; same smell, smaller scale.
- No true legacy compatibility shims/aliases found elsewhere (repo has no `fedcrg/` legacy dir, no `scripts/`, no `setup.py` — contract tests `test_architecture_boundaries.py`/`test_repository_hygiene.py` already assert their absence, suggesting a prior cleanup already removed those).

### 3.4 Circular-import risk / boundary violations (against prompt.md's target dependency direction)

- `config/validation.py` imports `policies/registry.py` (`config` → `policies`) — config sits above data/detectors/... in the target dependency chain; validating against `PolicyRegistry` from `config` is a downward/sideways dependency that will need inversion once `policies/` becomes `thresholds/`.
- `application/*` (to become `pipeline/`) imports heavily from `analysis/*` (`application/benchmark.py`→`analysis.benchmark`, `application/claims.py`→`analysis.claims`+`analysis.primary`, `application/synthetic.py`→`analysis.robustness`, `application/report.py`→`analysis.primary`) — but target dependency order places `pipeline` **above** `analysis` (`pipeline ↓ analysis ↓ reporting`), so pipeline importing analysis is backwards versus the target chain; this content needs to be inverted or the analysis content that pipeline actually needs (kernels like `analysis.robustness`, `analysis.benchmark`) relocated below `pipeline` (e.g., into `experiments/` or `pipeline/` itself), consistent with the note that `analysis/` "must not execute experiments."
- `analysis/tables.py`, `analysis/publication.py` import `config.models` directly and are themselves rendering code destined for `reporting/`; once relocated, verify `reporting` isn't then imported from something lower like `experiments`/`pipeline` (currently nothing in `application/`/`experiments/` imports `analysis.tables`/`analysis.publication`/`analysis.figures`/`analysis.decision_architecture` directly — only `cli/evaluation.py` does — so this particular boundary looks currently clean once those four files move to `reporting/`).
- Numerous local (function-body, not module-top) imports inside `cli/data.py`, `cli/evaluation.py`, `cli/experiments.py`, `cli/research.py`, `application/claims.py`, `application/feature_sensitivity.py`, `experiments/models.py` — these are classic "avoid circular import at module load time" workarounds and should be eliminated during migration by fixing the underlying dependency direction rather than preserved.

### 3.5 Filenames that don't match prompt.md's target tree / forbidden-name list

Files whose current name is on prompt.md's explicit forbidden list (`base.py`, `models.py`, `service.py`, `registry.py`, `identity.py`, `shared.py`, `references.py`, `variants.py`, `factory.py`, `pipeline.py`, `research.py`):

| Current path | Forbidden term |
|---|---|
| `detectors/base.py` | `base.py` |
| `policies/base.py` | `base.py` |
| `config/models.py` | `models.py` |
| `data/models.py` | `models.py` |
| `experiments/models.py` | `models.py` |
| `federated/models.py` | `models.py` |
| `scoring/models.py` | `models.py` |
| `protocol/service.py` | `service.py` |
| `policies/registry.py` | `registry.py` |
| `experiments/registry.py` | `registry.py` |
| `artifacts/identity.py` | `identity.py` |
| `cli/shared.py` | `shared.py` |
| `artifacts/references.py` | `references.py` |
| `config/variants.py` | `variants.py` |
| `detectors/factory.py` | `factory.py` |
| `application/pipeline.py` | `pipeline.py` |
| `cli/research.py` | `research.py` |

Plus, `metrics/results.py` and `protocol/results.py` use `results.py`, which is **not** forbidden by name (target explicitly keeps `results.py` under `method/`, `experiments/`, `thresholds/`) but there are currently 7 different `results.py`/`models.py`-style containers — flagged under 3.1 above for the discoverability problem even where the literal name is technically permitted.

---

## 4. Tooling configuration

All in `/home/naslouby/Projects/FedCRG/pyproject.toml` (no separate `ruff.toml`, `mypy.ini`, `pytest.ini`, `setup.cfg`, `tox.ini`, `Makefile`, or `noxfile.py` found; no `conftest.py` anywhere in `tests/`).

- **Ruff**: `[tool.ruff]` `target-version = "py311"`, `line-length = 100`, `src = ["src", "tests"]`; `[tool.ruff.lint]` `select = ["E", "F", "UP", "B"]`, `ignore = ["E501"]`.
  Command: `ruff check .` (or `ruff format` for formatting; no explicit `[tool.ruff.format]` section, so defaults apply).
- **Mypy**: `[tool.mypy]` `python_version = "3.11"`, `packages = ["fedcrg"]`, `ignore_missing_imports = true`, `disallow_untyped_defs = true`, `check_untyped_defs = true`, `warn_return_any = true`, `warn_unused_ignores = true`, `no_implicit_optional = true`. No Pyright/Pylance config present (`pyrightconfig.json` absent) — prompt.md asks for "Pyright/Pylance-compatible type checking" but the repo only wires up mypy; this is a gap to note for the validation-cadence step.
  Command: `mypy` (uses `packages = ["fedcrg"]`, so run from repo root with `src` on path, e.g. via `mypy` after `pip install -e .`).
- **Pytest**: `[tool.pytest.ini_options]` `addopts = "-q --strict-markers --disable-warnings"`, `testpaths = ["tests"]`, `pythonpath = ["src"]`. No custom markers registered (none used in the suite either, so `--strict-markers` is currently a no-op safety net). No `conftest.py` exists anywhere.
  Command: `pytest` (plain). `pytest-xdist` (3.8.0) is installed in the environment but **not** declared in `pyproject.toml`'s `dev` extras and no `-n auto`/`-p xdist` flag is wired into `addopts`; parallel execution would currently require manually passing `pytest -n auto` — the "pytest in parallel where supported" validation step from prompt.md is not yet automated in project config.
- **Coverage**: `[tool.coverage.run]` `branch = true`, `source = ["fedcrg"]`; `[tool.coverage.report]` `show_missing = true`, `skip_covered = true`, `fail_under = 80`.
- Console entry point: `[project.scripts] fedcrg = "fedcrg.cli.main:cli"`.