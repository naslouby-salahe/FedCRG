# Audit findings log

Append one dated section per audit pass (after each phase, and the final hostile audit).
Each finding: description, severity, resolution (fixed / accepted-with-reason).

## Pre-migration baseline audit (from repo inventory, see current_state.md)

See current_state.md section 3 for full detail. Summary of findings to be fixed during migration:

1. Four parallel orchestration entry points (application/run_experiment.py,
   application/pipeline.py, application/research_pipeline.py, experiments/executor.py) --
   must collapse to single pipeline/ spine. [Phase 6]
2. Seven results.py/models.py-named dataclass containers across packages -- rename to
   describe actual content per package. [Phases 1-7, per-package]
3. protocol/ vs policies/ split does not match target method/ vs thresholds/ split --
   requires conscious re-partition. [Phase 4]
4. policies/registry.py + experiments/registry.py -- "registry" pattern forbidden by
   prompt.md; dissolve into explicit typed selection. [Phase 4, Phase 6]
5. artifacts/environment.py + artifacts/environment_lock.py duplicate reproducibility
   capture concept -- merge. [Phase 7]
6. experiments/executor.py is dead code (only referenced by one test). [Phase 6 - delete]
7. application/robustness.py is a 25-line pass-through wrapper. [Phase 6 - inline]
8. scoring/__init__.py and data/datasets/__init__.py are re-export dumping grounds.
   [Phases 2-3 - empty them]
9. config/validation.py imports policies/registry.py -- downward dependency violation.
   [Phase 1/4 - invert]
10. application/* imports analysis/* (backwards vs target chain pipeline > analysis).
    [Phase 5/6 - relocate analysis kernels used at execution time]
11. Numerous function-body-local imports to dodge circular imports in cli/*.py,
    application/*.py, experiments/models.py -- fix dependency direction, remove workaround.
    [ongoing, all phases]

## Phase 1-4 resolutions

- Finding #3 (protocol/ vs policies/ split mismatched target method/ vs thresholds/ split):
  RESOLVED in Phase 4. Re-partitioned into method/ (FedCRG's own steps) and thresholds/
  (comparators), see remaining_work.md Phase 4 notes.
- Finding #4 (policies/registry.py + experiments/registry.py "registry" pattern forbidden):
  policies/registry.py RESOLVED in Phase 4 (dissolved into thresholds/selection.py's
  explicit functions + PolicyThresholdSelector). experiments/registry.py still pending,
  deferred to Phase 6 (application/ -> pipeline/ collapse touches experiments/ too).
- Finding #8 (scoring/__init__.py, data/datasets/__init__.py re-export dumping grounds):
  RESOLVED in Phases 2-3. scoring/__init__.py emptied; data/datasets/ package deleted
  entirely (nbaiot.py/diad.py flattened into data/ directly, no re-exporting __init__.py
  remains).
- Finding #9 (config/validation.py importing policies.registry, downward dependency):
  RESOLVED in Phase 4. The only real dependency was a "policy catalogue has exactly 12
  members" check, now inlined directly against the domain PolicyId enum -- config/validate.py
  no longer imports thresholds/ (or any package below it) at all.
## Phase 6 resolutions

- Finding #1 (four parallel orchestration layers): RESOLVED. application/run_experiment.py,
  pipeline.py (ExecuteFrozenWorkload), and research_pipeline.py (ExecuteResearchPipeline)
  collapsed into pipeline/run_experiment.py + pipeline/run_all_experiments.py (one class,
  RunAllExperiments, wrapping the training/scoring/materialization grid with preflight).
  experiments/executor.py deleted as dead code (see finding #6). One coherent spine remains:
  cli -> pipeline -> experiments/artifacts.
- Finding #4 (registries), second half: RESOLVED. experiments/registry.py's
  ExperimentRegistry dissolved into plain functions in experiments/experiment_definition.py.
- Finding #6 (experiments/executor.py dead code): RESOLVED, deleted.
- Finding #7 (application/robustness.py thin wrapper): RESOLVED, deleted and inlined.
- Finding #10 (application/* importing analysis/* backwards vs target chain): RESOLVED.
  analysis/robustness_analysis.py's live-execution kernels relocated to
  experiments/definitions/synthetic.py (used during S3/S4 execution, not evidence analysis).
  analysis/claim_gates.py and analysis/computational_benchmark.py now correctly import
  pipeline/ (valid direction: analysis sits below pipeline in the target chain).
- Finding #11 (function-body-local imports dodging circular imports): substantially
  resolved. experiments/execution.py no longer needs a local lifecycle import (merged into
  the same file). config/validate.py no longer imports thresholds/ at all (Phase 4). One
  new, deliberate local import remains (pipeline/prepare_dataset.py ->
  experiments/definitions/sensitivity.py inside PrepareDiadFeatureSensitivity.prepare) --
  this is a genuine mutual reference between two peer areas (R14's DIAD contract-builder
  needs general dataset preparation; general dataset preparation needs the R14-specific
  config transform), not a workaround for a fixable dependency-direction mistake; documented
  here per the requirement to justify any remaining exception rather than silently keep it.
- Finding #2 (results.py/models.py naming): largely resolved through Phases 1-6's renames
  (config/experiment_config.py, data/prepare.py+splits.py, evaluation/evaluation_results.py,
  experiments/experiment_definition.py, federation/training_results.py,
  scoring/score_records.py, thresholds/results.py, method/results.py -- the last two are
  explicitly permitted by the target tree). No remaining forbidden-named files.
- Finding #5 (artifacts/environment.py + environment_lock.py duplication): RESOLVED in
  Phase 7, merged into one artifacts/environment.py.

## Phase 7 resolutions

- Finding #5: RESOLVED (see above).
- artifacts/ package reduced from 14 files to prompt.md's exact 6 names (paths.py,
  manifests.py, records.py, json_io.py, integrity.py, environment.py). All findings from
  the pre-migration audit are now resolved except the minor sha256_file/hash_file
  cross-package duplication noted in remaining_work.md's Phase 7 notes (accepted, not a
  meaningful architectural issue -- different type contracts for different package layers).

Status: 7 of 9 phases complete (domain+config, data+detectors, federation+scoring,
method+thresholds, evaluation+analysis/reporting, application-removed+pipeline, artifacts
consolidation). See remaining_work.md for per-phase detail. Remaining: Phase 8
(reporting+cli boundaries), Phase 9 (final hostile audit + full validation).
