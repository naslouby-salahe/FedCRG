# Migration map (old path -> new path)

Source of truth: prompt.md target tree. Built from current_state.md inventory.
Legend: [1]..[9] = migration phase (see remaining_work.md), all under src/fedcrg/.

## Phase 1: domain + config

- core/enums.py -> domain/enums.py
- core/constants.py -> domain/constants.py
- core/exceptions.py -> domain/errors.py
- core/ids.py -> domain/identifiers.py (ClientId, RowId, AttackGroupId, Sha256, RunId)
- core/types.py -> domain/values.py (OperatingBand, ConfidenceInterval) + any Seed value types
- core/logging.py -> runtime.py (process/runtime concern, not domain)
- core/ (package) -> deleted
- config/models.py -> split: config/dataset_config.py (DatasetConfig, SplitConfig),
  config/training_config.py (TrainingConfig, AutoencoderConfig, DeepSvddConfig, DetectorConfig, RandomnessConfig),
  config/method_config.py (ProtocolConfig),
  config/experiment_config.py (ExperimentConfig) + FrozenModel base -> keep in experiment_config.py or values module
- config/loader.py -> config/load.py
- config/resolver.py -> config/resolve.py
- config/validation.py -> config/validate.py (must stop importing policies/thresholds; move
  any threshold-id validation to use domain enums only, or validate at higher layer)
- config/variants.py -> experiments/definitions/* (experiment variant construction is
  experiment-catalogue responsibility, not config responsibility) -- re-examine on phase 6

## Phase 2: data + detectors

- data/models.py -> fold into data/prepare.py / data/splits.py (typed records colocated with
  the logic that produces them; avoid a renamed "models.py")
- data/manifests.py -> fold into data/prepare.py (manifest dataclasses colocated with prepare)
- data/adapter.py -> fold into data/prepare.py (DatasetAdapter ABC) or keep as small
  cohesive piece inside nbaiot.py/diad.py common base -- decide during implementation
- data/audit.py -> data/eligibility.py or data/prepare.py (preflight audit of prepared data)
- data/discovery.py -> fold into nbaiot.py/diad.py (each adapter owns its own file discovery)
- data/integrity.py -> data/splits.py (split disjointness validation lives with splitting)
- data/eligibility.py -> data/eligibility.py (keep)
- data/feature_sensitivity.py -> data/feature_sensitivity.py (keep)
- data/preprocessing.py -> data/preprocessing.py (keep)
- data/splitting.py -> data/splits.py
- data/datasets/nbaiot.py -> data/nbaiot.py
- data/datasets/diad.py -> data/diad.py
- data/datasets/__init__.py -> deleted (no re-export dumping ground)
- detectors/base.py -> detectors/detector.py
- detectors/factory.py -> detectors/create_detector.py
- detectors/autoencoder.py -> detectors/autoencoder.py (keep)
- detectors/deep_svdd.py -> detectors/deep_svdd.py (keep)

## Phase 3: federation + scoring

- federated/ (package) -> federation/ (package rename)
- federated/models.py -> federation/training_results.py
- federated/sampling.py -> federation/participation.py
- federated/scheduling.py -> federation/learning_rate.py
- federated/client.py -> federation/client.py (keep)
- federated/server.py -> federation/server.py (keep)
- federated/aggregation.py -> federation/aggregation.py (keep)
- federated/trainer.py -> federation/training.py
- scoring/computer.py -> scoring/compute.py
- scoring/integrity.py -> scoring/validation.py
- scoring/models.py -> split: scoring/calibration_scores.py (role/client score views+inputs),
  scoring/score_records.py (ScoreManifest persistence-facing record)
- scoring/views.py -> fold into scoring/calibration_scores.py
- scoring/cache.py -> scoring/cache.py (keep)
- scoring/__init__.py -> emptied (no re-exports)

## Phase 4: method (was protocol) + thresholds (was policies)

- protocol/ (package) -> method/ (package)
- protocol/reference.py -> method/reference_threshold.py
- protocol/readiness.py -> method/calibration_readiness.py
- protocol/mismatch.py -> method/mismatch_detection.py
- protocol/decision.py -> method/threshold_decision.py
- protocol/service.py -> method/client_evaluation.py (FedCRGProtocol -> ClientEvaluation service)
- protocol/results.py -> method/results.py
- policies/ (package) -> thresholds/ (package) for comparators; FedCRG-specific bits already
  moved to method/ above
- policies/base.py -> thresholds/evidence.py (BenignPolicyEvidence, SupervisedDevelopmentEvidence,
  FinalTestEvidence, empirical_quantile) -- shared threshold evidence/quantile helper
- policies/quantile.py -> thresholds/comparators/global_quantile.py + local_quantile.py + three_sigma.py
  (split three comparators into three files per target tree)
- policies/personalized.py -> thresholds/comparators/readiness_only.py + mismatch_only.py
- policies/attack_aware.py -> thresholds/comparators/development_f1.py (dev_local_global),
  thresholds/comparators/summary_statistic.py (summary_statistic_threshold),
  thresholds/comparators/supervised_f1.py (supervised_global_f1)
- policies/shrinkage.py -> thresholds/comparators/shrinkage.py
- policies/oracle.py -> thresholds/comparators/oracle_test.py
- policies/registry.py -> DISSOLVED. PolicyRegistry/FederationPolicySelector logic becomes
  explicit typed selection: thresholds/selection.py (selects/builds ClientPolicyThreshold sets
  per PolicyId, calling the appropriate comparator function directly -- no registry indirection).
  PolicyDefinition/UndefinedPolicyReason/PolicyThresholdSet/InformationRegime -> thresholds/results.py
  (or evidence.py if purely evidence-shaped) per actual usage.
- thresholds/comparators/reference_quantile.py -- NEW: currently REFERENCE_QUANTILE policy
  (REF-Q99-R) is FedCRG's own reference-threshold path; verify whether it is truly a distinct
  comparator or just method/reference_threshold.py exposed as a policy option. Decide in Phase 4.

## Phase 5: evaluation (was metrics) + analysis/reporting split

- metrics/ (package) -> evaluation/ (package)
- metrics/classification.py -> evaluation/classification_metrics.py + evaluation/confusion_matrix.py
  (ConfusionMatrix + confusion_matrix() -> confusion_matrix.py; fpr/tpr/precision/recall/f1/
  balanced_accuracy -> classification_metrics.py)
- metrics/operating_band.py -> evaluation/operating_band_metrics.py
- metrics/attack_balanced.py -> evaluation/attack_balanced_metrics.py
- metrics/admission.py -> evaluation/admission_metrics.py
- metrics/ranking.py -> evaluation/ranking_metrics.py
- metrics/federation.py -> evaluation/federation_evaluation.py
- metrics/results.py -> evaluation/evaluation_results.py
- (new) evaluation/client_evaluation.py -- per-client evaluation orchestration currently
  embedded in application/evaluate.py; extract pure evaluation-computation part here,
  leave orchestration/fan-out for pipeline/evaluate_policies.py (Phase 6)
- (new) evaluation/utility_evaluation.py -- utility_margin_satisfied/utility_anchor/
  utility_preserved/assess_utility from metrics/federation.py if warranting separate file
  from federation_evaluation.py; decide by cohesion during implementation
- analysis/decision_architecture.py -> reporting/decision_figure.py
- analysis/figures.py -> reporting/figures.py
- analysis/publication.py -> reporting/publication.py
- analysis/tables.py -> reporting/tables.py
- analysis/benchmark.py -> analysis/computational_benchmark.py
- analysis/claims.py -> analysis/claim_gates.py
- analysis/communication.py -> analysis/communication_cost.py
- analysis/primary.py -> split: analysis/policy_contrasts.py (contrasts) +
  analysis/split_stability.py callers / data loading -- decide exact split during impl;
  load_federation_results likely belongs with artifacts or analysis entry
- analysis/robustness.py -> analysis/robustness_analysis.py
- analysis/stability.py -> analysis/split_stability.py
- analysis/statistics.py -> analysis/descriptive_statistics.py + analysis/paired_bootstrap.py

## Phase 6: application/ removed -> pipeline/ + experiments/definitions/

- application/prepare_data.py -> pipeline/prepare_dataset.py
- application/train.py -> pipeline/train_detector.py
- application/score.py -> pipeline/compute_scores.py
- application/evaluate.py -> pipeline/evaluate_policies.py (uses thresholds/selection.py +
  evaluation/* instead of policies.registry)
- application/run_experiment.py -> pipeline/run_experiment.py
- application/pipeline.py (ExecuteFrozenWorkload) -> MERGE into pipeline/run_experiment.py /
  pipeline/run_all_experiments.py -- collapse the 3 orchestration layers into one
- application/research_pipeline.py (ExecuteResearchPipeline) -> MERGE into
  pipeline/run_all_experiments.py (preflight + full workload)
- application/federation_cell.py -> pipeline/run_policy_evaluation.py
- application/policy_cell.py -> fold into pipeline/run_policy_evaluation.py
- application/precompute.py -> pipeline/select_thresholds.py or experiments/definitions/
  (finite-sample protocol table precompute -- statistical precompute belongs near thresholds)
- application/preflight.py -> pipeline/preflight.py
- application/verify.py -> pipeline/verify_outputs.py
- application/report.py -> reporting/report.py
- application/claims.py -> analysis/claim_gates.py (merge orchestration into analysis, since
  it derives claim gates from evidence -- no execution of experiments involved)
- application/benchmark.py -> analysis/computational_benchmark.py (merge with analysis/benchmark.py)
- application/synthetic.py -> experiments/definitions/synthetic.py
- application/sensitivity.py + experiments/real_sensitivity.py -> experiments/definitions/sensitivity.py
- application/source_order.py -> experiments/definitions/sensitivity.py (source-order is a
  sensitivity variant) or robustness.py -- decide by ExperimentType during impl
- application/robustness.py -> experiments/definitions/robustness.py (inline the thin wrapper)
- application/feature_sensitivity.py -> experiments/definitions/sensitivity.py (R14 contract-build)
  + pipeline/prepare_dataset.py (the PrepareDiadFeatureSensitivity part)
- application/ (package) -> deleted entirely
- experiments/executor.py -> DELETED (dead code, superseded by pipeline/run_experiment.py)
- experiments/definitions.py -> experiments/experiment_definition.py (catalogue data) split
  into experiments/definitions/{synthetic,primary,sensitivity,robustness,external_validation,
  computational_benchmark}.py by ExperimentType
- experiments/registry.py -> fold lookup/validate into experiments/experiment_definition.py
  (no "registry" concept; ExperimentRegistry becomes plain functions/typed lookup)
- experiments/planner.py -> experiments/planning.py
- experiments/dependencies.py -> experiments/dependencies.py (keep)
- experiments/lifecycle.py -> fold into experiments/execution.py (status transitions belong
  with execution state)
- experiments/models.py -> experiments/experiment_definition.py (catalogue/plan types) +
  experiments/execution.py (ExperimentExecution generic + lifecycle transitions)
- experiments/completion.py -> experiments/completion.py (keep)
- experiments/synthetic.py -> experiments/definitions/synthetic.py (statistical kernels
  merge with catalogue definitions for synthetic experiments)
- experiments/real_sensitivity.py -> experiments/definitions/sensitivity.py

## Phase 7: artifacts consolidation

- artifacts/layout.py -> artifacts/paths.py
- artifacts/identity.py -> fold into artifacts/paths.py (RunIdentityFactory builds RunId/paths)
- artifacts/manifest.py -> artifacts/manifests.py
- artifacts/dataset.py -> fold manifest-store classes into artifacts/manifests.py
- artifacts/preprocessing.py -> fold into artifacts/manifests.py
- artifacts/training.py -> fold into artifacts/manifests.py
- artifacts/experiment_results.py -> fold into artifacts/manifests.py or artifacts/records.py
- artifacts/records.py -> artifacts/records.py (keep; ThresholdRecord/MetricRecord)
- artifacts/references.py -> fold into artifacts/records.py (CacheReference is a record type)
- artifacts/hashing.py -> fold into artifacts/integrity.py
- artifacts/verification.py -> artifacts/integrity.py
- artifacts/serialization.py -> artifacts/json_io.py
- artifacts/environment.py + artifacts/environment_lock.py -> artifacts/environment.py (merge)

## Phase 8: reporting + cli

- cli/shared.py -> fold load_config into cli/main.py or runtime.py
- cli/research.py -> split: cli/scoring.py (train/score-adjacent precompute?), cli/benchmark.py
  (benchmark_command), remaining synthetic/robustness/sensitivity commands fold into
  cli/experiments.py (they all invoke experiments/pipeline machinery)
- cli/claims.py -> cli/claims.py (keep)
- cli/data.py -> cli/data.py (keep)
- cli/environment.py -> cli/environment.py (keep)
- cli/evaluation.py -> cli/evaluation.py + cli/reporting.py (report/publication commands move
  to reporting.py; evaluate_command stays in evaluation.py)
- cli/experiments.py -> cli/experiments.py (keep, absorb research.py experiment-adjacent cmds)
- cli/training.py -> cli/training.py (keep)
- cli/verification.py -> cli/verification.py (keep)
- cli/main.py -> cli/main.py (keep, update registrations)

## Phase 9: final audit
- repository-wide grep for stale names (core., application., protocol., policies., metrics.,
  federated.), stale test imports, dead re-exports; fix; run full validation.
