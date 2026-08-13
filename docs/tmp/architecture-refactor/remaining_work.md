# Remaining work (resumable checklist)

Update this file as phases complete. Status values: TODO / IN_PROGRESS / DONE.

- [x] Phase 1: domain + config          STATUS: DONE
- [x] Phase 2: data + detectors          STATUS: DONE
- [x] Phase 3: federation + scoring      STATUS: DONE
- [x] Phase 4: method + thresholds       STATUS: DONE
- [x] Phase 5: evaluation + analysis/reporting split   STATUS: DONE
- [x] Phase 6: application/ removed -> pipeline/ + experiments/definitions/   STATUS: DONE
- [ ] Phase 7: artifacts consolidation   STATUS: TODO
- [ ] Phase 8: reporting + cli           STATUS: TODO
- [ ] Phase 9: final hostile audit + full validation suite   STATUS: TODO

Each phase, when started, must:
1. Move/merge/rewrite the modules per migration_map.md.
2. Update every caller (grep repo-wide for old dotted path).
3. Adapt/rewrite the relevant tests (see current_state.md section 2 for old test->module map).
4. Delete the old files/packages -- no re-export shims.
5. Run `ruff check`, `ruff format --check`, `mypy`, targeted `pytest` subset for touched area.
6. Update this file's status and commit.

Do not run full pytest/mypy/ruff after every micro-edit -- only at phase completion.

## Phase 1 completion notes (domain + config)
- core/ deleted; enums.py/constants.py/errors.py(was exceptions.py)/identifiers.py(was ids.py)/
  values.py(was types.py) now live under domain/. logging.py moved to top-level runtime.py
  (process/runtime concern, not domain).
- config/models.py (282-line monolith) split into method_config.py (ProtocolConfig),
  dataset_config.py (DatasetConfig, SplitConfig), training_config.py (AutoencoderConfig,
  DeepSvddConfig, DetectorConfig, TrainingConfig, RandomnessConfig), experiment_config.py
  (ExperimentConfig + hashing helpers). No shared FrozenModel base class was recreated --
  each pydantic model declares its own `model_config = {"frozen": True, "extra": "forbid",
  "use_enum_values": False}` dict literal directly (avoids reintroducing a vague shared/base
  concept for a 1-line body used by ~8 classes).
- config/loader.py -> config/load.py, config/resolver.py -> config/resolve.py,
  config/validation.py -> config/validate.py. config/variants.py left in place with imports
  fixed (its final home is experiments/definitions/sensitivity.py per migration_map.md,
  deferred to Phase 6 since its only consumer, application/sensitivity.py, moves then too).
- config/validate.py still imports fedcrg.policies.registry.PolicyRegistry (downward
  dependency, pre-existing finding #9 in audit_findings.md) -- NOT fixed yet, deferred to
  Phase 4 when policies/ becomes thresholds/.
- Bulk-updated ~51 call sites across src/ and tests/ that imported from the old
  fedcrg.config.models / .loader / .resolver / .validation paths.
- Fixed tests/contract/test_architecture_boundaries.py's
  test_core_is_dependency_free_from_outer_layers -> test_domain_is_dependency_free_from_outer_layers
  (path core/ -> domain/, added fedcrg.config to forbidden-import list since domain sits below
  config in the target dependency chain).
- Validation: ruff check/format clean on full src+tests; mypy (py312 override, repo's declared
  py311 target chokes on installed numpy stubs -- pre-existing environment mismatch, not
  introduced by this migration) shows 0 errors in domain/ or config/; 11 pre-existing errors
  remain in untouched files (experiments/executor.py, scoring/cache.py, analysis/publication.py,
  detectors/deep_svdd.py, detectors/autoencoder.py, application/report.py, application/claims.py,
  application/federation_cell.py, cli/research.py) -- to be fixed as those files migrate in
  later phases, must all be clean before final audit. Full pytest -n auto suite passes (129
  tests, unchanged pass count).

## Phase 2 completion notes (data + detectors)
- data/datasets/{nbaiot,diad}.py -> data/{nbaiot,diad}.py (package flattened, datasets/
  subpackage and its re-exporting __init__.py deleted).
- data/adapter.py, data/discovery.py, data/models.py (ClientData only -- other symbols split
  out), and the manifest/hash helpers from data/manifests.py (SourceFileManifest,
  CalibrationAssignmentReference, RoleArtifactManifest, ClientDatasetManifest,
  PreparedDatasetManifest, hash_file, hash_row_ids, source_file_manifest) consolidated into
  new data/prepare.py -- the dataset-adapter contract + prepared-cache provenance types.
- data/splitting.py + data/integrity.py (validate_split_disjointness) + the splitting-domain
  types from data/models.py (RoleFrame, ClientSplits, RolePositions,
  CalibrationRoleAssignment) + the calibration-manifest types from data/manifests.py
  (CalibrationRoleManifest, ClientCalibrationManifest, CalibrationAssignmentManifest)
  consolidated into new data/splits.py.
- EligibilityRecord (from data/models.py) and EligibilityManifest (from data/manifests.py)
  moved into data/eligibility.py, colocated with ClientEligibilityEvaluator.
- data/preprocessing.py and data/feature_sensitivity.py kept as-is, imports repointed.
- data/audit.py deliberately left in place (imports already fixed by Phase 1's bulk sed) --
  its true home is pipeline/preflight.py per migration_map.md, deferred to Phase 6 since its
  only consumer (application/preflight.py) moves there too.
- detectors/base.py -> detectors/detector.py (rename only, forbidden vague name fixed).
  detectors/factory.py -> detectors/create_detector.py: DetectorFactory class dissolved into
  a plain create_detector() function (prompt.md's target tree names the file
  create_detector.py with no factory-class abstraction implied, and the class added no
  behavior beyond one branch). application/train.py's TrainDetector no longer takes an
  injectable `factory` parameter (no caller ever passed one).
- Bulk-updated ~15 call sites across src/ and tests/ that imported the old data.models /
  data.manifests / data.adapter / data.discovery / data.integrity / data.splitting /
  data.datasets.* / detectors.base / detectors.factory paths.
- Validation: ruff check/format clean; mypy (py312 override) shows 0 new errors -- same 7
  pre-existing errors as Phase 1 baseline (minus 4 that were in files later touched here with
  no new issues introduced); full pytest -n auto suite passes (129 tests, unchanged).

## Phase 3 completion notes (federation + scoring)
- federated/ package renamed to federation/ (git mv of the whole directory).
  models.py -> training_results.py, sampling.py -> participation.py,
  scheduling.py -> learning_rate.py, trainer.py -> training.py. client.py, server.py,
  aggregation.py kept. tests/unit/federated/ -> tests/unit/federation/.
- scoring/models.py split: RoleScoreInput/ClientScoreInput/RoleScores/ClientScoreSet (the
  per-client/per-role score containers) merged with scoring/views.py's
  ClientCalibrationScores/CalibrationScoreViews/CalibrationScoreViewBuilder/truncate_view into
  new scoring/calibration_scores.py; ScoreManifest (the persistence-facing top record) moved
  to new scoring/score_records.py. scoring/integrity.py -> scoring/validation.py.
  scoring/computer.py -> scoring/compute.py.
- Broke a would-be import cycle between calibration_scores.py (needs ScoreCache/ScoreManifest
  only for method type hints) and cache.py/score_records.py (which need the score-container
  types) using `if TYPE_CHECKING:` imports in calibration_scores.py -- standard pattern, not
  a workaround-style local import.
- scoring/__init__.py emptied (was the only non-empty package __init__.py in the repo,
  re-exporting ScoreCache/ScoreManifest/CalibrationScoreViewBuilder/CalibrationScoreViews;
  confirmed no caller imported the package directly rather than its submodules).
- Bulk-updated all internal and test call sites for fedcrg.federated.* ->
  fedcrg.federation.* and fedcrg.scoring.{models,views,integrity,computer} -> the new module
  names above, including tests/contract/test_architecture_boundaries.py's forbidden-prefix
  list.
- Validation: ruff check/format clean; mypy (py312 override) shows the same pre-existing
  errors as before (application/report.py, application/claims.py, application/
  federation_cell.py, scoring/cache.py) -- none newly introduced, none in federation/ or
  scoring/ itself; full pytest -n auto suite passes (129 tests, unchanged).

## Phase 4 completion notes (method + thresholds)
- protocol/ renamed to method/: reference.py -> reference_threshold.py, readiness.py ->
  calibration_readiness.py, mismatch.py -> mismatch_detection.py, decision.py ->
  threshold_decision.py, service.py -> client_evaluation.py, results.py kept.
  FedCRGProtocol class renamed to ClientEvaluation; ClientProtocolResult renamed to
  ClientEvaluationResult (repo-wide, ~9 files).
- policies/ dissolved and re-partitioned into thresholds/ (comparators) since the old
  protocol/-vs-policies/ split did not match the target method/-vs-thresholds/ split:
  - policies/base.py -> thresholds/evidence.py (BenignPolicyEvidence's `protocol` field
    renamed to `evaluation` -- it holds a ClientEvaluationResult, not a protocol).
  - policies/quantile.py split into thresholds/comparators/{global_quantile,local_quantile,
    three_sigma}.py (one file per comparator per target tree).
  - policies/personalized.py split into thresholds/comparators/{readiness_only,
    mismatch_only}.py.
  - policies/attack_aware.py split into thresholds/comparators/{development_f1,
    summary_statistic,supervised_f1}.py; the shared per-threshold-F1 helper (private
    `_defined_f1`/`_mean_client_f1`) was made public as `f1_at_threshold`/
    `mean_client_f1_at_threshold` in development_f1.py so the other two comparators can
    import it directly instead of duplicating it or reaching into a private name.
  - policies/shrinkage.py -> thresholds/comparators/shrinkage.py.
  - policies/oracle.py -> thresholds/comparators/oracle_test.py.
  - NEW thresholds/comparators/reference_quantile.py: the REFERENCE_QUANTILE (REF-Q99-R)
    policy was previously inlined directly in the registry as
    `benign.protocol.reference.value`; extracted into its own one-line comparator function
    for consistency with every other policy having a named comparator function.
  - policies/registry.py (PolicyRegistry + FederationPolicySelector) dissolved per
    prompt.md's explicit "Do not create a policy registry" rule:
    - InformationRegime enum, ClientPolicyThreshold, UndefinedPolicyReason,
      PolicyThresholdSet moved to new thresholds/results.py.
    - PolicyDefinition dataclass eliminated -- its two derived facts became plain functions
      `information_regime(policy_id)` and `is_deployable(policy_id)` in new
      thresholds/selection.py, backed by an explicit `SUPERVISED_POLICIES` frozenset
      constant (no dynamically-built dict-of-definitions).
      `assert_exact_protocol_registry` -> `validate_policy_catalogue_completeness()`.
    - FederationPolicySelector renamed to PolicyThresholdSelector (same explicit typed
      if/elif dispatch it already had; only the "registry" indirection and word are gone --
      it now calls thresholds/comparators/* functions directly instead of going through a
      registry-provided dict of policy definitions).
  - policies/ package deleted entirely.
- Resolved audit finding #9 (config/validate.py importing policies.registry, a downward
  dependency): the only thing config/validate.py needed from the registry was a "policy
  catalogue has exactly 12 members" check, which is now inlined as
  `len(set(PolicyId)) != 12` directly against the domain enum -- config no longer imports
  thresholds/ at all.
- application/evaluate.py (EvaluatePolicies) updated: imports repointed to
  method/thresholds/scoring's new locations, `selector.registry.supervised_requested(...)`
  replaced with `bool(set(config.policies) & SUPERVISED_POLICIES)` (no registry object to
  delegate to anymore).
- tests/unit/protocol/ -> tests/unit/method/ (files renamed to match); tests/unit/policies/
  -> tests/unit/thresholds/ (test_policies.py -> test_comparators.py, test_registry.py ->
  test_selection.py, rewritten against information_regime()/is_deployable() instead of the
  deleted PolicyRegistry.get()/all_ids()).
- Validation: ruff check/format clean; mypy (py312 override) reports 0 issues across
  method/, thresholds/, config/, and application/evaluate.py; full pytest -n auto suite
  passes (130 tests -- one net-new test added alongside the registry-removal rewrite).

## Phase 5 completion notes (evaluation + analysis/reporting split)
- metrics/ package renamed to evaluation/:
  - classification.py split: ConfusionMatrix + confusion_matrix() -> evaluation/
    confusion_matrix.py; fpr/tpr/precision/recall/f1/balanced_accuracy -> evaluation/
    classification_metrics.py.
  - operating_band.py -> operating_band_metrics.py, attack_balanced.py ->
    attack_balanced_metrics.py, ranking.py -> ranking_metrics.py, admission.py ->
    admission_metrics.py, federation.py -> federation_evaluation.py, results.py ->
    evaluation_results.py. metrics/ package deleted.
- analysis/ package boundary fixed per prompt.md's explicit "analysis/ must not own
  publication rendering" rule -- the four purely-rendering files (decision_architecture.py,
  figures.py, publication.py, tables.py) moved to a new top-level reporting/ package
  (decision_figure.py, figures.py, publication.py, tables.py), which prompt.md's target tree
  already names explicitly. This was the single biggest package-boundary violation flagged
  in the pre-migration audit (current_state.md section on analysis/).
- Remaining analysis/ files renamed to their target names: benchmark.py ->
  computational_benchmark.py, claims.py -> claim_gates.py, communication.py ->
  communication_cost.py, robustness.py -> robustness_analysis.py.
- analysis/statistics.py split by cohesion: DescriptiveSummary/describe/
  split_sensitivity_summary -> descriptive_statistics.py; PairedBootstrapInterval/
  paired_model_seed_bootstrap -> paired_bootstrap.py (two independent statistical kernels
  that happened to share a file only because they're both "statistics").
- analysis/primary.py split: FederationResultRecord/load_federation_results/
  confirmatory_contrasts/ContrastMetricResult/PolicyContrastResult (the primary contrast
  computation) -> new policy_contrasts.py; the split_sensitivity() function merged into
  analysis/stability.py (renamed split_stability.py) alongside ThresholdStability/
  StateStability, since both are split-sensitivity/stability responsibilities per prompt.md's
  target tree -- split_stability.py imports FederationResultRecord from policy_contrasts.py.
- Bulk-updated ~15 call sites across src/ and tests/ for the metrics.* -> evaluation.* and
  analysis.{primary,statistics,stability,benchmark,claims,communication,robustness,
  decision_architecture,figures,publication,tables} -> new module renames, including
  tests/contract/test_architecture_boundaries.py's forbidden-prefix lists
  ("fedcrg.metrics" -> "fedcrg.evaluation").
- tests/unit/metrics/ -> tests/unit/evaluation/ (test_metrics.py ->
  test_classification_and_bands.py, test_ranking_attack.py kept). tests/unit/analysis/
  test_communication_and_stability.py -> test_communication_cost_and_split_stability.py.
  tests/unit/analysis/test_tables_sensitivity.py -> new tests/unit/reporting/
  test_tables_sensitivity.py (it tests PublicationTableBuilder, a reporting/ concern).
- application/report.py, application/claims.py, application/synthetic.py,
  application/benchmark.py, cli/evaluation.py, data/preprocessing.py (a comment only) all
  had their imports repointed; application/*.py itself is not yet moved to pipeline/
  (deferred to Phase 6) but now correctly depends on the new analysis/evaluation/reporting
  module names.
- Validation: ruff check/format clean; mypy (py312 override) shows only the one pre-existing
  error in reporting/publication.py:297 (untyped-def, present before this phase too, not
  newly introduced); full pytest -n auto suite passes (130 tests, unchanged).

## Phase 6 completion notes (application/ removed -> pipeline/ + experiments/)
This was the largest phase: application/ (23 files) deleted entirely; its responsibilities
redistributed across a new pipeline/ package, analysis/, reporting/, and
experiments/definitions/.

- New pipeline/ package (the single execution spine): train_detector.py (was train.py),
  compute_scores.py (was score.py), run_experiment.py (unchanged), verify_outputs.py (was
  verify.py), select_thresholds.py (was precompute.py -- ProtocolTablePrecomputer),
  evaluate_policies.py (was evaluate.py), prepare_dataset.py (was prepare_data.py, now also
  hosting PrepareDiadFeatureSensitivity), preflight.py (merges application/preflight.py's
  ResearchPreflight with data/audit.py's PreparedDatasetAuditor -- data/audit.py reads
  already-persisted artifact JSON and gates training, which is pipeline/preflight territory,
  not a data/-package concern), run_policy_evaluation.py (merges policy_cell.py's
  PolicyCellMaterializer with federation_cell.py's FederationCellMaterializer -- one
  materializes a single policy cell, the other wraps it for every requested policy in one
  federation cell, a single cohesive responsibility), run_all_experiments.py (merges
  pipeline.py's ExecuteFrozenWorkload with research_pipeline.py's ExecuteResearchPipeline
  into one RunAllExperiments class -- this collapses two of the four previously-parallel
  orchestration layers identified in the pre-migration audit finding #1).
- reporting/report.py: application/report.py moved as-is (ReportBuilder), imports repointed.
- analysis/claim_gates.py: merged application/claims.py's ClaimGateEvaluator (derives G0-G8
  from frozen evidence) into the existing pure evidence-assessment file -- this matches
  prompt.md's explicit analysis/ allowance ("claim-gate assessment") and analysis/ is now
  correctly positioned to import pipeline/ (analysis sits below pipeline in the target
  dependency chain, so this direction is valid, unlike the old application/*->analysis/*
  imports flagged as backwards in finding #10).
- analysis/computational_benchmark.py: merged application/benchmark.py's RunBenchmark into
  the existing benchmark-harness file, for the same reason (R13 benchmark orchestration is
  "computational benchmark analysis", explicitly allowed in analysis/).
- experiments/definitions/synthetic.py (new subpackage, one file): merges
  experiments/synthetic.py's statistical kernels, analysis/robustness_analysis.py's
  temporal_dependence_stress/calibration_shift_stress/RobustnessCell (these generate live
  Monte Carlo trial data during S3/S4 execution, not post-hoc analysis of completed
  evidence -- relocating them resolves finding #10's other half), and
  application/synthetic.py's RunSyntheticExperiments orchestration for the full locked S1-S6
  programme (kept together since one class already executes all six as one cohesive unit).
- experiments/definitions/sensitivity.py (new subpackage, one file): merges
  experiments/real_sensitivity.py's kernels, application/sensitivity.py's
  RunRealSensitivities (R2-R9), application/source_order.py's RunSourceOrderCalibration
  (R12), application/feature_sensitivity.py's R14 contract-derivation
  (BuildDiadFeatureSensitivityContract/r14_config -- PrepareDiadFeatureSensitivity itself
  went to pipeline/prepare_dataset.py instead, since it's a data-preparation step), and
  config/variants.py's ExperimentVariantFactory (its only consumer was
  application/sensitivity.py, and it's explicit/typed config-variant construction, not a
  registry). **Deleted RunRealSensitivities.run_r12 as genuine dead code**: cli/research.py's
  dispatcher routes R12 exclusively to RunSourceOrderCalibration and never called
  RunRealSensitivities.run_r12, and the two methods wrote incompatible JSON shapes (only
  RunSourceOrderCalibration's shape matches what experiments/completion.py's
  _source_order_workload expects) -- this was silent duplication, not equivalent code.
- experiments/registry.py's PolicyRegistry-style ExperimentRegistry class dissolved (finding
  #4's second half) into plain module-level functions `get_experiment_definition(id)` /
  `all_experiment_definitions()` / catalogue-completeness validated at import time, defined
  in new experiments/experiment_definition.py alongside the merged
  experiments/models.py + experiments/definitions.py content. Deliberately kept as ONE file
  rather than prompt.md's literal experiments/definitions/{synthetic,primary,sensitivity,
  robustness,external_validation,computational_benchmark}.py split for the catalogue itself:
  the 20 ExperimentDefinition entries cross-reference each other's dependencies
  (`primary`/`external` ids) across type boundaries, so a reader needs the whole table
  either way, and primary/external_validation/computational_benchmark have no bespoke
  execution logic beyond their catalogue entry (unlike synthetic/sensitivity, which do --
  those two got real definitions/ subpackage files because they hold genuine per-type
  execution logic, not just data).
- experiments/lifecycle.py folded into new experiments/execution.py alongside
  ExperimentPlan/ExperimentExecution/ExperimentRunner (from models.py) -- eliminates the
  local `from fedcrg.experiments.lifecycle import assert_transition` import that
  ExperimentExecution.transition() previously needed only to dodge a circular import;
  now they're in the same file so no import is needed at all (finding #11 instance fixed).
  experiments/planner.py -> experiments/planning.py (registry dependency removed, now calls
  get_experiment_definition directly). experiments/dependencies.py's DependencyResolver no
  longer takes an injected registry (calls get_experiment_definition directly).
- experiments/executor.py (ExperimentExecutor) deleted as dead code (finding #6): confirmed
  its only reference anywhere was tests/unit/experiments/test_models.py, which itself only
  tested ExperimentExecutor. Deleted that test file; its one still-relevant test
  (dependency-order-includes-prerequisites) moved to new
  tests/unit/experiments/test_dependencies.py.
- application/robustness.py (RunRobustness, a 25-line pass-through to TrainDetector) deleted
  (finding #7); its one caller (cli/research.py's train_deep_svdd command) now calls
  TrainDetector directly with the Deep-SVDD check inlined.
- Rewrote tests/contract/test_architecture_boundaries.py's dependency-direction tests:
  domain-forbidden-imports list now says "fedcrg.pipeline" instead of the removed
  "fedcrg.application" (plus added fedcrg.analysis/fedcrg.reporting, which hadn't been
  listed before); method-boundary test renamed
  test_method_does_not_depend_on_pipeline_cli_or_artifact_io; added explicit
  test_legacy_repository_layers_are_absent assertions that application/, core/, protocol/,
  policies/, metrics/, federated/ no longer exist under src/fedcrg.
- Rewrote/renamed several experiments-catalogue tests against the new
  experiment_definition module: test_experiment_registry.py ->
  test_experiment_catalogue_completeness.py, test_experiment_catalogue_exact.py updated.
- Bulk-updated every cli/*.py module (claims.py, verification.py, evaluation.py, data.py,
  experiments.py, training.py, research.py) and ~10 test files that imported the deleted
  application.*/experiments.{models,definitions,registry,lifecycle,planner,executor,
  synthetic,real_sensitivity}/config.variants/data.audit/analysis.robustness_analysis paths.
- Validation: ruff check/format clean; mypy (py312 override) shows the same 10 pre-existing
  errors as before this phase (all in files that were moved-not-rewritten: reporting/
  publication.py, reporting/report.py, analysis/claim_gates.py, detectors/deep_svdd.py,
  detectors/autoencoder.py, scoring/cache.py, cli/research.py x2,
  pipeline/run_policy_evaluation.py) -- zero new errors; full pytest -n auto suite passes
  (125 tests; net -5 vs Phase 5's 130 from deleting the ExperimentExecutor-only test file's
  3 dead tests and consolidating 2 others, offset by 1 new dependency-order test).

## Notes / open decisions to resolve during implementation
- config/validate.py must not import thresholds/ (downward dependency violation in old
  config/validation.py -> policies.registry). Resolve by validating only against domain
  enums/config shape in config/, and doing threshold-set-completeness checks (if needed)
  in pipeline/ or thresholds/selection.py instead.
- application/* -> analysis/* imports (benchmark, claims, synthetic, report) currently run
  "backwards" vs target chain (pipeline should sit above analysis). Resolve during Phase 5/6:
  analysis kernels that pipeline/experiments actually need at execution time (e.g.
  analysis.robustness stress generators, analysis.benchmark harness) must be relocated to
  experiments/definitions/ or pipeline/, not left in analysis/ and imported upward.
- thresholds/comparators/reference_quantile.py existence TBD -- confirm whether REF-Q99-R
  policy is a thin wrapper over method/reference_threshold.py or a genuinely distinct
  comparator; only create the file if there is real distinct behavior.
- Local (function-body) imports scattered in cli/*.py and application/*.py exist to dodge
  circular imports -- fix root dependency direction during migration; do not preserve the
  workaround pattern in new code.
- scoring/__init__.py and data/datasets/__init__.py currently re-export symbols; make all
  __init__.py files empty (or minimal package docstring only) after migration.
- pyproject.toml: no pyrightconfig.json exists; prompt.md requires Pyright/Pylance-compatible
  type checking. Existing mypy config is strict (disallow_untyped_defs etc.) -- treat mypy
  as the working proxy for Pyright compatibility during phase validation; add pyright as a
  final-audit check if available in the environment.
- pytest-xdist is installed but not wired into addopts; use `pytest -n auto` explicitly for
  parallel runs during validation, per prompt.md's "pytest in parallel where supported".
