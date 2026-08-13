# Goal

Refactor the FedCRG repository into a clean, explicit, easy-to-navigate architecture.

This is a **real architectural migration**, not a compatibility-preserving reorganization.

## Non-Negotiable Rules

* **NO backwards compatibility.**
* Do not preserve old imports, old module paths, deprecated names, aliases, redirect modules, compatibility wrappers, re-exports, shims, or migration bridges.
* When code moves, **move it completely** and update every caller.
* Delete obsolete files immediately once their responsibilities have been migrated.
* Do not leave empty compatibility packages behind.
* Do not introduce `canonical`, `canonical_*`, `registry`, `catalogue`, `identity`, `shared`, `common`, `utils`, `helpers`, `misc`, or similarly vague architectural concepts merely to relocate existing complexity.
* Avoid vague filenames such as:

  * `base.py`
  * `models.py`
  * `service.py`
  * `registry.py`
  * `identity.py`
  * `shared.py`
  * `references.py`
  * `variants.py`
  * `factory.py`
  * `pipeline.py`
  * `research.py`
* Filenames must describe the responsibility they actually contain.
* Prefer domain-specific names over architectural jargon.
* No primitive leakage where a domain enum/value/dataclass already exists or should exist.
* Use enums instead of stringly-typed choices.
* Use frozen/slotted dataclasses for domain records where appropriate.
* Do not use untyped dictionaries as domain interfaces.
* Do not introduce `Any` to make the migration easier.
* Search for existing reusable behavior before creating new abstractions.
* Merge duplicate concepts instead of wrapping them.
* Remove dead code.
* Remove stale tests rather than preserving obsolete APIs for them.
* Adapt/rewrite tests to the new architecture.
* Do not add weird explanatory AI-generated comments.
* Keep public APIs as small as possible.
* `__init__.py` files must not become compatibility/re-export dumping grounds.

# First Step

Before changing code:

1. Inspect the complete repository.
2. Read the project roadmap and existing architecture/audit documentation if present.
3. Map every current module, class, enum, dataclass, function, test, CLI entry point, artifact type, and import relationship to its actual responsibility.
4. Identify:

   * duplicate responsibilities;
   * vague modules;
   * circular or upward dependencies;
   * compatibility code;
   * dead modules;
   * redundant abstractions;
   * concepts split across multiple packages;
   * modules whose names do not describe their behavior.
5. Create a temporary migration workspace under:

```text
docs/tmp/architecture-refactor/
```

Use it only for migration tracking. Maintain at minimum:

```text
current_state.md
migration_map.md
remaining_work.md
audit_findings.md
```

Do not create hashing/checkpoint bureaucracy. These files are simply resumable working notes.

# Target Architecture

Refactor toward this architecture:

```text
src/
└── fedcrg/
    ├── __init__.py
    ├── __main__.py
    │
    ├── domain/
    │   ├── __init__.py
    │   ├── enums.py
    │   ├── identifiers.py
    │   ├── values.py
    │   ├── constants.py
    │   └── errors.py
    │
    ├── config/
    │   ├── __init__.py
    │   ├── dataset_config.py
    │   ├── training_config.py
    │   ├── method_config.py
    │   ├── experiment_config.py
    │   ├── load.py
    │   ├── resolve.py
    │   └── validate.py
    │
    ├── data/
    │   ├── __init__.py
    │   ├── nbaiot.py
    │   ├── diad.py
    │   ├── prepare.py
    │   ├── splits.py
    │   ├── preprocessing.py
    │   ├── eligibility.py
    │   └── feature_sensitivity.py
    │
    ├── detectors/
    │   ├── __init__.py
    │   ├── detector.py
    │   ├── autoencoder.py
    │   ├── deep_svdd.py
    │   └── create_detector.py
    │
    ├── federation/
    │   ├── __init__.py
    │   ├── client.py
    │   ├── server.py
    │   ├── aggregation.py
    │   ├── participation.py
    │   ├── learning_rate.py
    │   ├── training.py
    │   └── training_results.py
    │
    ├── scoring/
    │   ├── __init__.py
    │   ├── compute.py
    │   ├── cache.py
    │   ├── calibration_scores.py
    │   ├── score_records.py
    │   └── validation.py
    │
    ├── method/
    │   ├── __init__.py
    │   ├── reference_threshold.py
    │   ├── calibration_readiness.py
    │   ├── mismatch_detection.py
    │   ├── threshold_decision.py
    │   ├── client_evaluation.py
    │   └── results.py
    │
    ├── thresholds/
    │   ├── __init__.py
    │   ├── evidence.py
    │   ├── selection.py
    │   ├── results.py
    │   │
    │   └── comparators/
    │       ├── __init__.py
    │       ├── reference_quantile.py
    │       ├── global_quantile.py
    │       ├── local_quantile.py
    │       ├── readiness_only.py
    │       ├── mismatch_only.py
    │       ├── shrinkage.py
    │       ├── three_sigma.py
    │       ├── development_f1.py
    │       ├── summary_statistic.py
    │       ├── supervised_f1.py
    │       └── oracle_test.py
    │
    ├── evaluation/
    │   ├── __init__.py
    │   ├── confusion_matrix.py
    │   ├── classification_metrics.py
    │   ├── ranking_metrics.py
    │   ├── operating_band_metrics.py
    │   ├── attack_balanced_metrics.py
    │   ├── admission_metrics.py
    │   ├── client_evaluation.py
    │   ├── federation_evaluation.py
    │   ├── utility_evaluation.py
    │   └── evaluation_results.py
    │
    ├── experiments/
    │   ├── __init__.py
    │   ├── experiment_definition.py
    │   ├── planning.py
    │   ├── dependencies.py
    │   ├── execution.py
    │   ├── completion.py
    │   ├── results.py
    │   │
    │   └── definitions/
    │       ├── __init__.py
    │       ├── synthetic.py
    │       ├── primary.py
    │       ├── sensitivity.py
    │       ├── robustness.py
    │       ├── external_validation.py
    │       └── computational_benchmark.py
    │
    ├── analysis/
    │   ├── __init__.py
    │   ├── descriptive_statistics.py
    │   ├── paired_bootstrap.py
    │   ├── policy_contrasts.py
    │   ├── split_stability.py
    │   ├── communication_cost.py
    │   ├── robustness_analysis.py
    │   ├── computational_benchmark.py
    │   └── claim_gates.py
    │
    ├── artifacts/
    │   ├── __init__.py
    │   ├── paths.py
    │   ├── manifests.py
    │   ├── records.py
    │   ├── json_io.py
    │   ├── integrity.py
    │   └── environment.py
    │
    ├── pipeline/
    │   ├── __init__.py
    │   ├── preflight.py
    │   ├── prepare_dataset.py
    │   ├── train_detector.py
    │   ├── compute_scores.py
    │   ├── select_thresholds.py
    │   ├── evaluate_policies.py
    │   ├── run_policy_evaluation.py
    │   ├── run_experiment.py
    │   ├── run_all_experiments.py
    │   └── verify_outputs.py
    │
    ├── reporting/
    │   ├── __init__.py
    │   ├── report.py
    │   ├── publication.py
    │   ├── tables.py
    │   ├── figures.py
    │   └── decision_figure.py
    │
    ├── cli/
    │   ├── __init__.py
    │   ├── main.py
    │   ├── data.py
    │   ├── training.py
    │   ├── scoring.py
    │   ├── evaluation.py
    │   ├── experiments.py
    │   ├── reporting.py
    │   ├── claims.py
    │   ├── benchmark.py
    │   ├── environment.py
    │   └── verification.py
    │
    └── runtime.py
```

This tree is a **target responsibility model**, not an excuse to create empty files.

Do not create a listed file unless there is actual behavior that belongs there. If repository inspection proves that two listed files would merely split one cohesive implementation artificially, keep the implementation cohesive and document the reason in the migration notes.

Conversely, do not retain an old file simply because its exact responsibility is not shown above.

# Required Structural Changes

## Remove `application/`

There must be no separate `application` architectural layer.

Move its responsibilities into the appropriate domain package or into the single `pipeline/` execution spine.

Anything equivalent to:

```text
PrepareData
TrainDetector
ComputeScores
EvaluatePolicies
RunExperiment
ExecuteResearchPipeline
ExecuteFrozenWorkload
materialize*
```

must be placed according to actual responsibility, normally under `pipeline/`.

Do not keep wrappers preserving the old application interfaces.

## Replace `core/`

Do not retain a generic `core/` dumping ground.

Move:

* domain enums → `domain/enums.py`;
* identifiers → `domain/identifiers.py`;
* value objects → `domain/values.py`;
* true global constants → `domain/constants.py`;
* domain exceptions → `domain/errors.py`;
* runtime-specific behavior → `runtime.py` or the directly responsible package.

## Replace `protocol/`

The FedCRG scientific decision method belongs under:

```text
method/
```

It should expose the actual conceptual steps:

```text
reference threshold
→ mismatch detection
→ calibration readiness
→ threshold decision
→ client evaluation/result
```

Do not keep a generic protocol-service architecture.

## Replace `policies/`

Threshold methods belong under:

```text
thresholds/
```

FedCRG itself must not be hidden among generic policy implementations.

FedCRG method logic belongs in `method/`.

Comparator threshold strategies belong in:

```text
thresholds/comparators/
```

Do not create a policy registry.

Selection should be explicit and typed.

## Replace `metrics/`

Use:

```text
evaluation/
```

This layer computes scientific run-level metrics and evaluations.

It must not perform cross-run statistical analysis or manuscript reporting.

## Simplify `artifacts/`

Do not maintain one file for every individual artifact dataclass.

Consolidate artifact infrastructure into cohesive responsibilities:

```text
paths.py
manifests.py
records.py
json_io.py
integrity.py
environment.py
```

Strong typing remains mandatory.

Consolidation does **not** mean replacing typed records with dictionaries.

## Clarify `analysis/`

`analysis/` may consume completed experimental evidence and perform:

* descriptive statistics;
* paired bootstrap;
* contrasts;
* robustness analysis;
* communication analysis;
* split sensitivity/stability analysis;
* claim-gate assessment;
* computational benchmark analysis.

It must not execute experiments.

It must not own publication rendering.

## Clarify `reporting/`

`reporting/` converts existing evidence into:

* reports;
* publication packages;
* tables;
* figures;
* decision diagrams.

Scientific calculations that define experimental results belong elsewhere.

Reporting must not silently recompute or alter scientific outcomes.

## One Execution Spine

There must be exactly one obvious execution path:

```text
CLI
 ↓
pipeline
 ↓
experiments / capabilities
 ↓
artifacts
```

Do not retain parallel execution frameworks.

Do not have both:

```text
application pipeline
experiment executor
research pipeline
workload runner
pipeline runner
```

performing overlapping orchestration.

The CLI must remain thin.

# Dependency Direction

Target dependency direction:

```text
domain
  ↓
config
  ↓
data / detectors / federation / scoring / method / thresholds
  ↓
evaluation
  ↓
experiments
  ↓
pipeline
  ↓
analysis
  ↓
reporting
  ↓
cli
```

This is conceptual dependency direction, not permission for unnecessary coupling.

Enforce these boundaries:

* `domain` imports nothing from higher layers.
* `config` must not depend on pipeline/CLI/reporting.
* `data` must not import pipeline or CLI.
* `detectors` must not import experiments/pipeline/reporting/CLI.
* `federation` must not import experiments/pipeline/reporting/CLI.
* `scoring` must not import pipeline/reporting/CLI.
* `method` must not import pipeline/reporting/CLI.
* `thresholds` must not import pipeline/reporting/CLI.
* `evaluation` must not import reporting or CLI.
* `experiments` define experimental work but do not become a second orchestration framework.
* `analysis` analyzes completed evidence and must not launch experiments.
* `reporting` consumes evidence and must not define experimental science.
* `cli` contains no scientific/business logic.

Detect and eliminate circular dependencies instead of hiding them through local imports or import tricks.

# Naming Rules

A developer should understand a file's responsibility from its path alone.

Prefer:

```text
training_results.py
calibration_scores.py
threshold_decision.py
classification_metrics.py
operating_band_metrics.py
communication_cost.py
run_experiment.py
verify_outputs.py
```

over:

```text
models.py
service.py
shared.py
common.py
registry.py
base.py
manager.py
engine.py
handler.py
processor.py
helpers.py
utils.py
```

Do not blindly rename existing concepts. Rename them according to what they actually do.

Do not invent unnecessary nouns merely to satisfy architectural symmetry.

# Migration Procedure

Perform the refactor incrementally by large coherent responsibility groups.

For each large group:

1. inspect the current implementation;
2. identify the authoritative implementation;
3. identify duplication and stale variants;
4. decide the final owner/package;
5. move/merge/refactor the implementation;
6. update all imports and consumers;
7. adapt tests;
8. delete old modules;
9. search the entire repository for stale imports/names;
10. update `docs/tmp/architecture-refactor/`;
11. commit the completed coherent migration.

Do **not** create compatibility modules between steps.

The repository is allowed to temporarily have broken old imports during the migration because old APIs are not being preserved. Complete each coherent migration before moving to unrelated work.

# Validation Cadence

Do not run the entire test/lint/type-check stack after every tiny edit.

Create, adapt, rewrite, or remove tests continuously as code changes.

After a **large coherent chunk** of the migration is complete, run the appropriate full validation suite, including:

* formatting;
* Ruff;
* Pyright/Pylance-compatible type checking;
* pytest in parallel where supported.

Fix all resulting issues before declaring that migration chunk complete.

Do not weaken typing, add ignores, introduce `Any`, or add compatibility layers merely to make validation pass.

At the end, run the complete quality suite again.

# Audit After Every Major Phase

After each major package migration, perform an architecture audit.

Check:

1. Does every remaining module have one clear responsibility?
2. Does its filename explain that responsibility?
3. Is any old package still retained solely for compatibility?
4. Is functionality duplicated?
5. Are there wrapper classes/functions that merely forward calls?
6. Are there stale re-exports?
7. Are any imports pointing in the wrong dependency direction?
8. Are there circular dependencies?
9. Are there string primitives where enums should be used?
10. Are domain records being represented as dictionaries?
11. Are dataclasses duplicated between packages?
12. Does pipeline orchestration exist anywhere outside `pipeline/`?
13. Does scientific analysis execute experiments?
14. Does reporting calculate scientific outcomes?
15. Does CLI contain business/scientific logic?
16. Is there dead code?
17. Are old architectural names still referenced in tests/docs/code?
18. Is any concept unnecessarily split across several tiny files?
19. Are any modules generic dumping grounds?
20. Would a new developer know where to find a responsibility without repository-wide search?

Fix findings immediately before continuing.

# Final Audit

Do not stop when files have merely been moved.

The goal is reached only when:

* the old architecture is gone;
* old imports are gone;
* compatibility aliases are gone;
* compatibility modules are gone;
* duplicate implementations are gone;
* dead code is gone;
* vague architectural files are gone unless genuinely justified;
* responsibilities match the new package boundaries;
* there is one execution spine;
* tests reflect only the new architecture;
* repository-wide searches find no stale architectural names;
* type checking passes;
* Ruff passes;
* formatting passes;
* tests pass;
* the architecture is materially simpler to understand than before.

Perform a final package-by-package audit and record the result in:

```text
docs/tmp/architecture-refactor/audit_findings.md
```

If the audit discovers a structural issue, fix it and rerun the relevant audit. Do not merely document unresolved architectural debt.

# Git Discipline

Commit after each **major coherent migration**, not after every small edit.

Commits should correspond to meaningful architecture changes such as:

```text
refactor domain and configuration boundaries
refactor data and detector packages
consolidate federation and scoring
separate FedCRG method from threshold comparators
consolidate evaluation and analysis
replace application layer with pipeline execution spine
simplify artifact infrastructure
refactor reporting and CLI boundaries
complete architecture cleanup and final audit
```

Do not commit knowingly broken half-migrations unless unavoidable.

# Idempotency

This goal is idempotent.

On every invocation:

1. inspect the current repository state;
2. read `docs/tmp/architecture-refactor/` if it exists;
3. compare the current codebase against this target architecture and responsibility model;
4. do not redo completed migrations;
5. verify previous work rather than assuming it is correct;
6. continue from the highest-priority remaining architectural problem;
7. remove any regression toward the old architecture;
8. continue until the final audit passes.

Do not stop merely because the target directories exist.

**The goal is semantic architectural convergence, not matching a file-tree screenshot.**
