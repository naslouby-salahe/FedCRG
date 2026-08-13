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

Status: none yet resolved -- migration not started.
