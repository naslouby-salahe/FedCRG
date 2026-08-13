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
- Findings #1 (four orchestration layers), #2 (results.py/models.py naming), #5
  (environment.py/environment_lock.py duplication), #6 (experiments/executor.py dead code),
  #7 (application/robustness.py thin wrapper), #10 (application/* importing analysis/*
  backwards), #11 (function-body-local imports dodging circular imports): still open,
  targeted for Phase 5/6.

Status: 4 of 9 phases complete (domain+config, data+detectors, federation+scoring,
method+thresholds). See remaining_work.md for per-phase detail.
