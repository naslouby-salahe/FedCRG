# FedCRG Continuous Repository Completion Goal

Work continuously on:

`/home/naslouby/Projects/FedCRG`

Your goal is to bring the repository to the target architecture and quality level defined below.

This is an idempotent goal.

Every time this prompt is executed:

1. Inspect the current repository state.
2. Read the scientific specification.
3. Read the current audit matrix.
4. Determine what is already correct.
5. Determine what remains incomplete, incorrect, duplicated, weakly typed, hardcoded, poorly named, misplaced, dead, or architecturally inconsistent.
6. Select a substantial coherent batch of work.
7. Implement it completely.
8. Update all callers and tests.
9. Audit the repository again.
10. Update the matrix.
11. Commit substantial completed work.
12. Continue.

Do not stop because one phase or package is complete.

Do not declare completion while known actionable issues remain.

---

# 1. Current Repository Only

Work from the current `main` branch and the current filesystem.

Do not inspect Git history to reconstruct:

- previous architectures
- previous temporary folders
- deleted audit matrices
- deleted implementation plans
- deleted source files
- old prompts
- abandoned approaches

Do not use:

- `git log` for architectural archaeology
- `git show` to recover deleted implementations
- `git blame` to reconstruct previous decisions
- old branches as implementation references

The current repository is the implementation state.

Git may still be used normally for:

- `git status`
- inspecting the current diff
- checking the current branch
- committing completed coherent work
- pushing when appropriate

Do not discard unrelated user changes.

There is no backwards compatibility requirement.

Do not preserve obsolete:

- APIs
- modules
- imports
- names
- aliases
- compatibility layers
- redirects
- re-export modules
- configuration formats
- wrappers

Update all callers and tests and remove obsolete code.

---

# 2. First Required Action: Create a Fresh Audit Matrix

Before performing the main refactor, thoroughly read:

`/home/naslouby/Projects/FedCRG/docs/FedCRG Roadmap.md`

Then inspect the complete current repository.

Create a completely fresh file:

`/home/naslouby/Projects/FedCRG/docs/FedCRG Audit Matrix.md`

Do not search for an older matrix.

Do not recover one from Git.

Do not copy previous temporary work.

Build the matrix only from:

- the current roadmap
- the current repository
- the current configuration
- the current tests
- the current implementation

The matrix must cover all major scientific and engineering requirements, including:

- configuration
- datasets
- preprocessing
- dataset reuse
- detectors
- federated training
- scoring
- FedCRG decision logic
- threshold policies
- evaluation
- statistical analysis
- experiments
- sensitivity experiments
- robustness experiments
- synthetic experiments
- benchmarks
- artifacts
- caching
- publication outputs
- reporting
- CLI
- logging
- monitoring
- GPU execution
- RAM and VRAM safety
- reproducibility
- tests
- typing
- architecture
- repository hygiene

Each row should contain at least:

- requirement
- expected implementation
- expected location
- configuration ownership
- current state
- identified problem
- required action
- verification criteria
- status

Use only clear statuses:

- `NOT_IMPLEMENTED`
- `PARTIAL`
- `INCORRECT`
- `IMPLEMENTED`
- `VERIFIED`
- `BLOCKED`

Do not use vague percentages.

The matrix must remain a living implementation control document.

Update it after each substantial batch.

Do not fill it with migration history.

---

# 3. Fresh Working Folder

Create:

`docs/work/`

Use this directory only for the current implementation effort.

Maintain concise files such as:

- `docs/work/current_state.md`
- `docs/work/current_violations.md`
- `docs/work/next_actions.md`
- `docs/work/verification.md`

Do not recreate timestamped files after every iteration.

Update the existing files.

Production code must never depend on `docs/work/`.

---

# 4. Target Repository Structure

Converge toward the following architecture.

Small evidence-based consolidations are allowed when two files would otherwise become meaningless wrappers, but do not drift away from the responsibility boundaries.

    FedCRG/
    ├── .github/
    │   └── workflows/
    │       └── ci.yml
    │
    ├── configs/
    │   ├── method/
    │   │   └── fedcrg.yaml
    │   ├── datasets/
    │   │   ├── nbaiot.yaml
    │   │   └── diad.yaml
    │   ├── detectors/
    │   │   ├── nbaiot_autoencoder.yaml
    │   │   ├── diad_autoencoder.yaml
    │   │   └── nbaiot_deep_svdd.yaml
    │   ├── training/
    │   │   ├── nbaiot_autoencoder.yaml
    │   │   ├── diad_autoencoder.yaml
    │   │   └── nbaiot_deep_svdd.yaml
    │   ├── randomness/
    │   │   ├── primary.yaml
    │   │   ├── external_validation.yaml
    │   │   └── synthetic.yaml
    │   ├── statistics/
    │   │   └── confirmatory.yaml
    │   └── experiments/
    │       ├── primary/
    │       │   └── nbaiot.yaml
    │       ├── external/
    │       │   └── diad.yaml
    │       ├── robustness/
    │       │   ├── second_detector.yaml
    │       │   ├── temporal_dependence.yaml
    │       │   ├── calibration_shift.yaml
    │       │   ├── calibration_contamination.yaml
    │       │   ├── real_contamination.yaml
    │       │   ├── source_order_test.yaml
    │       │   └── source_order_calibration.yaml
    │       ├── sensitivity/
    │       │   ├── readiness_sample_size.yaml
    │       │   ├── mismatch_sample_size.yaml
    │       │   ├── tolerance.yaml
    │       │   ├── target_fpr.yaml
    │       │   ├── assurance.yaml
    │       │   ├── multiplicity.yaml
    │       │   └── diad_features.yaml
    │       ├── synthetic/
    │       │   ├── readiness_theorem.yaml
    │       │   ├── target_fpr.yaml
    │       │   └── mismatch_power.yaml
    │       └── benchmark/
    │           └── computational.yaml
    │
    ├── data/
    │   ├── raw/
    │   └── preprocessed/
    │
    ├── src/
    │   └── fedcrg/
    │       ├── __init__.py
    │       ├── __main__.py
    │       │
    │       ├── domain/
    │       │   ├── enums.py
    │       │   ├── identifiers.py
    │       │   ├── parameters.py
    │       │   └── errors.py
    │       │
    │       ├── configuration/
    │       │   ├── method_config.py
    │       │   ├── dataset_config.py
    │       │   ├── detector_config.py
    │       │   ├── training_config.py
    │       │   ├── statistics_config.py
    │       │   ├── experiment_config.py
    │       │   └── config_loader.py
    │       │
    │       ├── datasets/
    │       │   ├── client_dataset.py
    │       │   ├── dataset_split.py
    │       │   ├── eligibility.py
    │       │   ├── preprocessing.py
    │       │   ├── nbaiot.py
    │       │   └── diad.py
    │       │
    │       ├── detectors/
    │       │   ├── anomaly_detector.py
    │       │   ├── autoencoder.py
    │       │   └── deep_svdd.py
    │       │
    │       ├── federation/
    │       │   ├── trainer.py
    │       │   ├── aggregation.py
    │       │   └── training_result.py
    │       │
    │       ├── scoring/
    │       │   ├── score_batch.py
    │       │   ├── detector_scoring.py
    │       │   └── score_cache.py
    │       │
    │       ├── decision/
    │       │   ├── readiness.py
    │       │   ├── mismatch.py
    │       │   ├── threshold_decision.py
    │       │   ├── policy_evidence.py
    │       │   ├── policy_selection.py
    │       │   └── policies/
    │       │       ├── quantile.py
    │       │       ├── shrinkage.py
    │       │       ├── supervised.py
    │       │       ├── summary_statistic.py
    │       │       ├── three_sigma.py
    │       │       └── oracle.py
    │       │
    │       ├── evaluation/
    │       │   ├── detection_metrics.py
    │       │   ├── ranking_metrics.py
    │       │   ├── operating_band_metrics.py
    │       │   ├── federation_metrics.py
    │       │   └── communication_metrics.py
    │       │
    │       ├── experiments/
    │       │   ├── experiment_plan.py
    │       │   ├── experiment_runner.py
    │       │   ├── synthetic_experiments.py
    │       │   ├── sensitivity_experiments.py
    │       │   ├── robustness_experiments.py
    │       │   └── computational_benchmark.py
    │       │
    │       ├── artifacts/
    │       │   ├── artifact_paths.py
    │       │   ├── run_manifest.py
    │       │   ├── dataset_manifest.py
    │       │   ├── preprocessing_manifest.py
    │       │   ├── training_manifest.py
    │       │   ├── score_manifest.py
    │       │   ├── serialization.py
    │       │   └── integrity.py
    │       │
    │       ├── analysis/
    │       │   ├── claim_gates.py
    │       │   ├── policy_contrasts.py
    │       │   ├── paired_bootstrap.py
    │       │   └── split_stability.py
    │       │
    │       ├── reporting/
    │       │   ├── publication_tables.py
    │       │   ├── publication_figures.py
    │       │   └── claim_report.py
    │       │
    │       ├── runtime/
    │       │   └── logging.py
    │       │
    │       └── cli/
    │           ├── app.py
    │           ├── data_commands.py
    │           ├── experiment_commands.py
    │           ├── analysis_commands.py
    │           └── report_commands.py
    │
    ├── outputs/
    │   ├── logs/
    │   ├── monitoring/
    │   ├── cache/
    │   │   ├── models/
    │   │   ├── scores/
    │   │   └── analysis/
    │   └── runs/
    │
    ├── results/
    │
    ├── tests/
    │   ├── unit/
    │   ├── integration/
    │   ├── contract/
    │   └── regression/
    │
    ├── docs/
    ├── Makefile
    ├── noxfile.py
    ├── pyproject.toml
    ├── requirements.lock
    ├── README.md
    └── LICENSE

There must be one obvious execution spine.

`experiments/` owns experiment and campaign execution.

Do not create another orchestration layer.

---

# 5. Configuration Must Be the Scientific Source of Truth

This is a strict rule.

YAML owns scientific choices.

Python owns:

- typing
- validation
- algorithms
- formulas
- execution
- derived values
- persistence
- reporting

Computed runtime artifacts own computed values.

Never declare the same scientific setting independently in YAML and Python.

Move scientific choices out of Python, including where applicable:

- model seeds
- calibration seeds
- bootstrap seeds
- synthetic seeds
- rounds
- local epochs
- batch size
- learning rates
- optimizer parameters
- weight decay
- client fraction
- detector dimensions
- activation
- target FPR
- tolerance
- confidence
- assurance
- bootstrap replicate count
- experiment axes
- sample sizes
- contamination fractions
- sensitivity values
- benchmark repetitions
- benchmark warmups
- Monte Carlo repetitions
- split sizes
- calibration sizes
- selected policies
- dataset expected counts

Pydantic configuration classes validate these values.

They must not silently create scientific values through Python defaults.

Scientific fields should generally be required.

Do not solve hardcoding by moving values into a Python `constants.py`.

If a value belongs to configuration, place it in configuration.

If a value is derived, compute it.

If a value is categorical and finite, use an enum.

---

# 6. Continuously Audit Primitive Leakage

Repeatedly search for inappropriate semantic use of:

- `str`
- `int`
- `float`
- `bool`
- `object`
- `Any`
- `dict`
- raw unnamed tuples
- untyped lists

Audit:

- function parameters
- return types
- class fields
- dataclass fields
- Pydantic fields
- manifests
- configuration
- results
- experiment definitions
- cache descriptors
- public APIs
- CLI/application boundaries

Important domain concepts must have semantic types.

Examples include:

- `ClientId`
- `RunId`
- `DatasetId`
- `ExperimentId`
- `PolicyId`
- `ModelSeed`
- `CalibrationSeed`
- `FeatureCount`
- `RowCount`
- `RoundCount`
- `EpochCount`
- `BatchSize`
- `Probability`
- `TargetFpr`
- `ConfidenceLevel`
- `ReadinessAssurance`
- `Tolerance`
- `LearningRate`
- `ByteCount`
- `Sha256`

Use:

- enums for closed categorical sets
- frozen dataclasses for immutable records
- Pydantic models for validated boundary/configuration/artifact models

Do not use `Any` in production code.

Do not use `object` as an escape hatch.

Do not use `dict[str, object]` as an internal transport format.

Raw dictionaries may exist temporarily when parsing an external format but must immediately become validated typed models.

Numerical arrays inside NumPy, SciPy, PyTorch, scikit-learn, Arrow, Polars or pandas are normal.

Do not create pointless wrappers around every numerical element.

The goal is to eliminate primitive leakage at semantic boundaries.

Create permanent contract tests for primitive leakage.

Use an extremely small explicit allowlist only for genuine third-party boundaries.

---

# 7. Enums

Use enums for every finite categorical domain concept.

Examples:

- dataset
- detector
- feature contract
- optimizer
- aggregation strategy
- compute device
- policy
- experiment
- experiment type
- data role
- calibration mode
- readiness state
- mismatch outcome
- decision state
- threshold source
- failure reason
- eligibility state
- artifact type
- claim level
- multiplicity method
- contamination direction
- distribution
- chronology state
- attack family
- attack subtype
- run status

Do not compare domain concepts through arbitrary strings.

Use descriptive enum values.

Do not use opaque runtime names such as:

- `R1`
- `R14`
- `S3`
- `REF-Q99-R`
- `GATE-A-ONLY`
- `DEV-F1-LG-SELECT`

Use names such as:

- `primary_nbaiot`
- `diad_feature_sensitivity`
- `reference_quantile`
- `readiness_only`
- `development_f1_selection`

Publication-specific labels belong in reporting.

---

# 8. Forbidden Vocabulary and Naming

The word `canonical` is forbidden in production source.

Remove it from:

- filenames
- class names
- function names
- method names
- variables
- comments
- docstrings
- log messages

Use precise alternatives such as:

- `identity_payload`
- `stable_representation`
- `resolved_configuration`
- `device_directory_rules`
- `serialized_payload`

Production source must not refer to:

- roadmap
- prompt
- migration phases
- agent instructions
- previous architecture
- backwards compatibility work

Comments should describe scientific or engineering reasons only.

Do not write strange AI-style explanatory comments.

Avoid vague filenames such as:

- `utils.py`
- `helpers.py`
- `common.py`
- `shared.py`
- `manager.py`
- `handler.py`
- `processor.py`
- `engine.py`
- `service.py`
- `base.py`
- `models.py`
- `registry.py`
- `factory.py`

unless there is a very strong responsibility-specific justification.

Improve unclear variable names.

Avoid application-level names such as:

- `cfg`
- `res`
- `obj`
- `tmp`
- `cid`
- `val`
- `item`
- `data`

when a precise name is available.

Short mathematical symbols are acceptable inside narrow formulas.

---

# 9. Continuously Audit Hardcoded Values and Defaults

Search the entire production source for literals and defaults.

Do not rely only on grep.

Use AST inspection where useful.

Every significant literal should be classified as:

- mathematical constant
- programming/structural constant
- external serialization requirement
- library requirement
- configuration-owned value
- runtime-derived value

Move configuration-owned values into YAML.

Compute runtime-derived values.

Do not move a hardcoded number into `constants.py` and consider it fixed.

Search repeatedly for:

- Pydantic field defaults
- function defaults
- constructor defaults
- hidden statistical defaults
- CLI defaults
- detector defaults
- experiment defaults
- hardcoded seeds
- fixed counts
- repeated scientific literals

Tests must not become another scientific configuration source.

True mathematical regression fixtures may remain hardcoded when they are testing an expected mathematical result.

---

# 10. Prefer Mature Libraries

Before implementing custom infrastructure, check whether a mature library already solves the problem correctly.

Evaluate and use where appropriate:

- Pydantic v2
- Pydantic `TypeAdapter`
- Pandera
- PyArrow
- Polars
- NumPy
- SciPy
- scikit-learn
- statsmodels
- PyTorch
- Rich
- structlog
- psutil
- NVML bindings
- filelock
- pytest
- pytest-xdist
- Hypothesis where valuable

Do not add dependencies without value.

Do not maintain custom implementations of common functionality when a mature library provides a clearer and safer implementation.

FedCRG-specific scientific algorithms and formulas should remain explicit and tested.

---

# 11. Preprocessed Data Must Live Under `data/preprocessed`

This is mandatory.

Preprocessed datasets must be materialized under:

`data/preprocessed/`

Do not store preprocessed datasets under:

`outputs/cache/datasets/`

Use a deterministic preprocessing identity.

A suitable structure is:

    data/preprocessed/
    └── <dataset_id>/
        └── <preprocessing_identity>/
            ├── manifest.json
            ├── preprocessing.json
            └── clients/
                └── <client_id>/
                    ├── train.parquet
                    ├── reservoir.parquet
                    ├── benign_test.parquet
                    ├── attack_dev.parquet
                    └── attack_test.parquet

The exact physical representation may be improved if scientifically and computationally justified, but the root remains:

`data/preprocessed/`

The preprocessing identity must include all inputs that change prepared data, including where relevant:

- dataset ID
- dataset source version
- source-file checksums
- parser version
- feature contract
- split configuration
- preprocessing configuration
- deterministic split seeds

It must not include unrelated:

- policy configuration
- reporting settings
- publication settings
- analysis settings

Before preprocessing:

1. Resolve the expected preprocessing identity.
2. Check whether that materialization already exists.
3. Validate its manifest and integrity.
4. Reuse it if valid.
5. Do not preprocess again.

Compatible experiments must reuse the same preprocessed dataset.

Add integration tests proving this reuse.

Do not duplicate a complete preprocessed dataset merely because the calibration assignment changed.

When possible, calibration assignments should be views or persisted row assignments over the stable reservoir.

Preprocessed data should be immutable after successful materialization.

Use an explicit overwrite/rebuild command when rebuilding is genuinely necessary.

---

# 12. Raw Data

Raw input belongs under:

`data/raw/`

If the repository uses a shared raw-data directory and `data/raw` does not exist, create the appropriate symlink instead of copying the dataset.

Never write derived data into raw data.

Treat raw data as immutable.

---

# 13. Runtime Output Architecture

Use:

    outputs/
    ├── logs/
    ├── monitoring/
    ├── cache/
    │   ├── models/
    │   ├── scores/
    │   └── analysis/
    └── runs/

Responsibilities:

`outputs/logs/`

- persistent execution logs
- structured logs
- failure diagnostics

`outputs/monitoring/`

- RAM telemetry
- CPU telemetry
- GPU utilization
- GPU memory
- stage timings

`outputs/cache/models/`

- reusable trained models
- training manifests
- detector artifacts

`outputs/cache/scores/`

- reusable immutable score caches

`outputs/cache/analysis/`

- reusable expensive analysis results where appropriate

`outputs/runs/`

- run manifests
- status
- provenance
- references to shared reusable artifacts

Do not duplicate huge artifacts into every run directory.

---

# 14. Publication Results Folder

Create a top-level:

`results/`

This is a publication output area, not a cache.

Implement an explicit CLI command:

`fedcrg results build [CAMPAIGN_ID]`

Campaign execution must also automatically invoke the exact same results-building implementation after all required work completes successfully.

Do not maintain two different implementations.

A publication bundle should look approximately like:

    results/
    └── <campaign_id>/
        ├── manifest.json
        ├── checksums.json
        ├── resolved_configs/
        ├── metrics/
        ├── statistics/
        ├── tables/
        ├── figures/
        ├── reports/
        └── provenance/

Include compact paper/reviewer-ready evidence such as:

- resolved configurations
- campaign manifest
- run identities
- dataset identity
- preprocessing identity
- detector identity
- training identity
- model seeds
- calibration seeds where relevant
- threshold decisions
- evaluation metrics
- aggregated statistics
- sensitivity results
- robustness results
- claim evidence
- publication tables
- publication figures
- machine-readable JSON
- useful CSV files
- environment information
- provenance
- checksums

Do not copy:

- raw datasets
- huge preprocessed datasets
- unnecessary caches
- temporary files

into `results/`.

Implement:

`fedcrg results verify [CAMPAIGN_ID]`

Verification must check:

- required files
- manifest consistency
- hashes
- provenance
- configuration identities
- result completeness
- required publication figures/tables/reports

A campaign is not complete if required publication results do not verify.

---

# 15. Reuse Expensive Artifacts

Before expensive work, compute the typed identity of the artifact.

Check whether a valid immutable artifact already exists.

Reuse when scientifically valid.

This applies to:

- preprocessed data
- trained detector models
- score caches
- expensive derived analysis

Do not use filenames alone to determine reuse.

Validate:

- scientific identity
- configuration identity
- provenance
- checksums
- required metadata

Examples:

A reporting change must not retrain the detector.

A threshold-policy change must not preprocess the data again.

A threshold-policy change must not retrain the detector if detector inputs are unchanged.

A calibration-assignment change must not regenerate seed-independent preprocessing.

A calibration-assignment change should not recompute seed-independent base scores when the required scores already exist.

Design artifact identities accordingly.

---

# 16. CLI

Keep the CLI thin.

Scientific logic must not live in CLI commands.

Converge toward a command structure similar to:

    fedcrg
    ├── data
    │   ├── preprocess [DATASET_ID]
    │   └── status [DATASET_ID]
    ├── experiment
    │   ├── validate [EXPERIMENT_ID]
    │   ├── plan [EXPERIMENT_ID]
    │   └── run <EXPERIMENT_ID>
    ├── campaign
    │   ├── run
    │   └── status
    ├── results
    │   ├── build [CAMPAIGN_ID]
    │   └── verify [CAMPAIGN_ID]
    ├── report
    └── monitor

Refine command grouping only if inspection proves a clearer interface exists.

Do not add redundant synonyms.

All important workflows must be properly wired.

---

# 17. Console Output and Logging

Long-running experiments must clearly show what is happening.

Use Rich where useful.

The console should show meaningful information such as:

- campaign ID
- experiment ID
- current stage
- dataset
- detector
- model seed
- calibration seed where relevant
- current policy where relevant
- preprocessing cache hit
- model cache hit
- score cache hit
- cache miss
- stage start
- stage completion
- elapsed duration
- experiment progress
- campaign progress
- current resource usage
- failures
- blocked experiments
- artifact paths
- final result bundle path

Do not print per-row or per-batch noise.

Persist useful logs under:

`outputs/logs/`

Use structured contextual logging.

Do not scatter `print()` throughout production code.

---

# 18. Resource Monitoring

Add practical runtime monitoring.

Track at least:

- process RAM
- available system RAM
- CPU utilization
- GPU utilization
- GPU memory usage
- GPU memory availability
- active CUDA device
- stage duration

Persist telemetry under:

`outputs/monitoring/`

Provide periodic console summaries during long-running workloads.

Implement a CLI command such as:

`fedcrg monitor`

where practical.

Campaign execution should automatically record telemetry.

Use maintained libraries where possible.

---

# 19. GPU Usage

Use GPU aggressively where GPU execution materially benefits the workload.

Detector training should use CUDA when the configured experiment requires CUDA.

Detector inference/scoring should use CUDA where beneficial.

Do not silently fall back from a required CUDA experiment to CPU.

Log:

- CUDA availability
- selected GPU
- GPU name
- VRAM capacity
- peak allocated VRAM where possible

Use appropriate PyTorch practices:

- bounded batches
- pinned host memory when useful
- non-blocking copies when safe
- `torch.inference_mode()` for inference
- avoid repeated unnecessary tensor conversions
- avoid unnecessary CPU/GPU synchronization

Keep CPU-native statistical operations on CPU when using GPU would provide little benefit.

Reproducibility takes priority over nondeterministic speed hacks.

---

# 20. RAM and VRAM Safety

Use RAM effectively but avoid OOM.

Prefer:

- Parquet
- Arrow
- Polars lazy scanning where beneficial
- streaming
- chunking
- bounded batches
- memory mapping
- iterators
- partitioned reads
- dropping large intermediates promptly

Do not repeatedly duplicate large dataframes.

Do not build giant Python lists for serialization when Arrow/Parquet operations can be used.

Use GPU batches conservatively.

If adaptive batch sizing is used, it must not alter scientific results and must be logged.

Never catch an OOM and silently modify the scientific experiment.

---

# 21. One Experiment Execution Spine

There must be one obvious execution flow.

Do not create:

runner → service → executor → pipeline → another runner.

Use one experiment runner that coordinates meaningful capabilities.

Conceptually:

    configuration
    → preflight validation
    → preprocessing identity and reuse
    → model identity and reuse
    → scoring identity and reuse
    → calibration/evidence
    → FedCRG decision
    → policy evaluation
    → statistical analysis
    → reporting
    → publication results
    → verification

Do not use a DAG framework.

Keep the execution path understandable in normal Python.

---

# 22. Classes and Abstractions

Use classes only when they own meaningful:

- state
- invariants
- behavior
- lifecycle
- dependency composition

Use frozen dataclasses or Pydantic models for immutable records.

Do not convert every function into a class.

Remove unnecessary:

- factories
- registries
- managers
- services
- wrappers
- façades
- adapters with no translation
- one-method classes

Do not preserve a wrapper whose entire implementation is equivalent to:

`return dependency.same_operation(...)`

Search existing code before creating new implementations.

Merge duplicate behavior.

---

# 23. Serialization

Use typed serialization.

Do not pass `dict[str, object]` throughout the application.

Prefer:

- Pydantic v2
- `TypeAdapter`
- typed manifests
- explicit boundary models

Avoid repeated manual reconstruction using:

- `str(...)`
- `int(...)`
- `float(...)`
- custom `as_json_dict`
- custom `as_json_list`
- custom primitive narrowing helpers

unless absolutely required by a third-party boundary.

JSON, CSV and Parquet are external representation formats.

Internal code should consume typed objects.

Validate every loaded manifest.

Fail clearly on corruption or integrity mismatch.

---

# 24. Dataset and Dataframe Contracts

Define explicit dataset schemas.

Evaluate Pandera or an equivalent mature solution.

Validate:

- required columns
- feature columns
- feature ordering
- finite values
- finite-rate contracts
- metadata fields
- row IDs
- label boundaries
- attack-group fields
- client identity

Do not scatter important column-name strings across unrelated code.

Dataset-specific filename and directory matching rules should live in typed dataset configuration where appropriate.

Use enums for finite attack families and attack subtypes.

---

# 25. Metrics and Statistical Libraries

Use mature implementations for standard statistical and ML operations.

Prefer existing libraries for:

- confusion matrices
- F1
- precision
- recall
- balanced accuracy
- AUROC
- standard ranking metrics
- multiple-testing correction
- standard schedulers

FedCRG-specific formulas should remain explicit.

All statistical choices affecting conclusions must come from configuration.

Do not hide defaults such as:

- `alpha=0.05`
- `confidence=0.95`
- `replicates=10000`
- hardcoded bootstrap seeds

inside scientific functions.

---

# 26. Makefile

Create or update the root:

`Makefile`

It should expose convenient commands such as:

- `make help`
- `make install`
- `make format`
- `make lint`
- `make typecheck`
- `make test`
- `make test-unit`
- `make test-integration`
- `make test-contract`
- `make test-regression`
- `make audit`
- `make validate`
- `make preprocess`
- `make plan`
- `make run`
- `make campaign`
- `make status`
- `make monitor`
- `make results`
- `make verify-results`
- `make quality`

Use variables where needed, such as:

- `EXPERIMENT=...`
- `DATASET=...`
- `CAMPAIGN=...`

Do not put scientific configuration values in the Makefile.

The Makefile is only an interface to real commands.

---

# 27. Nox

Create or update:

`noxfile.py`

Provide sessions such as:

- `format`
- `lint`
- `typecheck`
- `unit`
- `integration`
- `contract`
- `regression`
- `audit`
- `quality`

Use `pyproject.toml` as the underlying tool configuration source.

Do not duplicate Ruff, type-checking or pytest configuration in Nox unnecessarily.

Do not execute the complete quality session after every tiny edit.

---

# 28. Permanent Architecture Contract Tests

Create tests that prevent these problems from returning.

Enforce at least:

- no old package paths
- no compatibility modules
- no thin redirect modules
- no forbidden vague filenames without explicit justification
- no `canonical`
- no production references to roadmap/prompt/migration
- no `Any`
- no inappropriate `object`
- no `dict[str, object]` internal transport
- no scientific defaults in configuration models
- no important primitive leakage
- no duplicate scientific configuration ownership
- correct dependency direction
- one execution spine
- preprocessed data root is `data/preprocessed`
- compatible experiments reuse preprocessing
- immutable shared artifacts are validated before reuse
- publication result bundles verify correctly

Prefer AST-based tests when grep would produce false positives.

Do not make the checks reject normal mathematical primitives inside numerical calculations.

Focus on semantic boundaries.

---

# 29. Typing and Quality

Maintain strict typing.

Use:

- Pyright/Pylance-compatible typing
- Ruff
- Ruff formatting
- pytest
- pytest-xdist

Do not add `Any`.

Do not broadly suppress type errors.

Do not weaken type-checker configuration to make failures disappear.

Use a narrow `cast()` only when a third-party typing defect genuinely requires one.

Do not add unexplained `# type: ignore`.

Fix the implementation instead.

---

# 30. Test Execution Strategy

Continuously create, update, rewrite and delete tests as architecture changes.

Do not run the full suite after every small change.

Work in substantial coherent batches.

Use narrow validation while developing difficult logic.

After a substantial batch, run the appropriate combination of:

- formatting
- Ruff
- Pyright/type checking
- relevant unit tests
- relevant contract tests
- relevant integration tests

After a very large batch, run broader verification.

Run tests in parallel when safe.

Before final completion, run the complete quality gate.

---

# 31. Continuous Audit Loop

Repeat this process continuously:

    READ ROADMAP AND CURRENT MATRIX
    ↓
    INSPECT CURRENT REPOSITORY
    ↓
    IDENTIFY HIGHEST-IMPACT VIOLATIONS
    ↓
    SELECT ONE COHERENT IMPLEMENTATION BATCH
    ↓
    IMPLEMENT / MOVE / MERGE / DELETE / RENAME / REFACTOR
    ↓
    UPDATE CALLERS
    ↓
    UPDATE TESTS
    ↓
    RUN TARGETED VALIDATION
    ↓
    AUDIT PRIMITIVES
    ↓
    AUDIT ANY / OBJECT / DICT
    ↓
    AUDIT HARDCODED VALUES
    ↓
    AUDIT DEFAULTS
    ↓
    AUDIT ENUM USAGE
    ↓
    AUDIT NAMING
    ↓
    AUDIT CANONICAL TERMINOLOGY
    ↓
    AUDIT WRAPPERS
    ↓
    AUDIT DUPLICATION
    ↓
    AUDIT LIBRARY REUSE
    ↓
    AUDIT CONFIGURATION OWNERSHIP
    ↓
    AUDIT PREPROCESSING REUSE
    ↓
    AUDIT MODEL REUSE
    ↓
    AUDIT SCORE REUSE
    ↓
    AUDIT ARTIFACT INTEGRITY
    ↓
    AUDIT OUTPUT STRUCTURE
    ↓
    AUDIT LOGGING AND PROGRESS
    ↓
    AUDIT RESOURCE MONITORING
    ↓
    AUDIT DEPENDENCY DIRECTIONS
    ↓
    UPDATE MATRIX
    ↓
    COMMIT IF SUBSTANTIAL
    ↓
    REPEAT

An audit is not merely a report.

Fix the findings.

---

# 32. Repeated Repository Searches

Regularly run searches equivalent to:

    rg '\bAny\b' src tests
    rg '\bobject\b' src
    rg 'dict\[' src
    rg '\bcanonical\b' src tests
    rg -i 'roadmap|prompt|migration|legacy|backward.?compat' src
    rg 'Field\(default' src/fedcrg
    rg 'default\s*=' src/fedcrg

Also inspect:

- primitive type annotations
- function defaults
- numeric literals
- string literals used as identities
- duplicate values from YAML
- duplicate path construction
- duplicate serialization code
- duplicate policy dispatch
- duplicate experiment dispatch
- single-method classes
- test-only production APIs
- dead files
- unreachable functions
- local imports hiding circular dependencies

Do not blindly replace grep matches.

Classify each match based on its semantic role.

Use AST-based inspection where appropriate.

---

# 33. Navigation Quality

Regularly ask:

Can a new developer correctly predict where this responsibility lives?

If not, improve the architecture.

Avoid:

- overlapping packages
- giant unrelated modules
- dozens of one-function files
- generic dumping grounds
- repeated result classes with vague names
- duplicate execution paths

Optimize for:

- scientific traceability
- readability
- discoverability
- responsibility boundaries
- reuse

---

# 34. Runtime Progress and Status

Campaign execution must expose persistent status.

A user must be able to determine:

- current campaign
- current experiment
- current stage
- completed experiments
- pending experiments
- failed experiments
- blocked experiments
- preprocessing reuse
- model reuse
- score reuse
- current model seed
- current calibration seed
- elapsed time
- current CPU use
- current RAM use
- current GPU use
- current VRAM use
- important artifact paths
- results path

Implement a useful:

`fedcrg campaign status`

Provide clear Rich console output during execution.

---

# 35. Failure Handling

A failed experiment must not destroy valid shared artifacts.

Persist failure information.

Independent experiments may continue when dependencies allow it.

Do not continue an experiment whose required dependency failed.

Clearly distinguish:

- failed
- blocked
- skipped
- complete

Do not build a supposedly complete publication bundle when required confirmatory evidence is missing.

---

# 36. Scientific Integrity

Do not alter scientific formulas because implementation is inconvenient.

When the implementation and specification disagree:

1. Inspect the roadmap carefully.
2. Inspect the relevant equations and tests.
3. Inspect current configuration.
4. Research authoritative primary literature if required.
5. Resolve the ambiguity explicitly.
6. Record the decision briefly in `docs/work/`.
7. Implement one behavior.

Do not keep multiple behaviors for compatibility.

Do not run expensive experiments merely to guess what the intended formula is.

---

# 37. Avoid Full Research Runs During Refactoring

Do not launch the complete research campaign merely to validate architecture.

Prefer:

- static analysis
- unit tests
- contract tests
- targeted integration tests
- synthetic fixtures
- smoke workflows
- tiny development datasets where appropriate

All actual commands must still be fully implemented and wired.

---

# 38. Final Hostile Audit

When the matrix appears complete, do not stop.

Perform a fresh hostile repository audit as if reviewing somebody else's research software.

Recheck everything from scratch:

- roadmap coverage
- architecture
- configuration ownership
- scientific defaults
- hardcoded values
- primitive leakage
- enums
- duplicate values
- duplicate implementations
- library reuse
- class quality
- wrapper methods
- dead code
- test-only production code
- naming
- forbidden vocabulary
- serialization
- dataset contracts
- preprocessing storage
- preprocessing reuse
- model reuse
- score reuse
- cache integrity
- GPU use
- RAM safety
- VRAM safety
- logging
- console progress
- monitoring
- outputs
- publication results
- CLI
- Makefile
- Nox
- dependency direction
- Pyright
- Ruff
- formatting
- tests

Fix every actionable problem found.

Then perform another hostile audit.

Repeat until an entire hostile audit produces no actionable finding.

---

# 39. Completion Criteria

The goal is complete only when all of the following are simultaneously true:

- the fresh audit matrix contains no unresolved `NOT_IMPLEMENTED`
- the fresh audit matrix contains no unresolved `PARTIAL`
- the fresh audit matrix contains no unresolved `INCORRECT`
- there are no unjustified `BLOCKED` requirements
- the repository closely follows the target architecture
- there is one experiment execution spine
- YAML owns all scientific choices
- Python configuration models do not silently provide scientific defaults
- no significant scientific parameter is duplicated between YAML and Python
- no inappropriate hardcoded scientific values remain
- no inappropriate primitive leakage remains
- no `Any` remains in production source
- no inappropriate `object` remains
- no inappropriate `dict[str, object]` transport remains
- finite categorical concepts use enums
- important semantic scalar concepts use validated value types
- no `canonical` terminology remains
- no production code refers to roadmap, prompt, migration or previous architecture
- no backwards-compatibility wrappers remain
- no redirect modules remain
- no unnecessary re-export modules remain
- no unnecessary wrapper classes remain
- no dead code remains
- no important production implementation exists only for tests
- mature libraries are used instead of unnecessary custom infrastructure
- preprocessed datasets live under `data/preprocessed/`
- compatible experiments demonstrably reuse preprocessing
- reusable trained models are reused when scientifically valid
- reusable score caches are reused when scientifically valid
- reusable artifacts are validated by identity and integrity
- runtime logs live under `outputs/logs/`
- runtime monitoring lives under `outputs/monitoring/`
- models, scores and analysis caches have clear homes under `outputs/cache/`
- run metadata has a clear home under `outputs/runs/`
- long-running experiments provide good console progress
- campaign status is inspectable
- GPU-capable detector work uses GPU correctly
- RAM and VRAM usage are bounded and OOM-conscious
- `fedcrg results build` exists
- campaign completion automatically builds results using the same implementation
- `fedcrg results verify` exists
- publication-ready results are written under `results/<campaign_id>/`
- result bundles contain required JSON, metrics, statistics, tables, figures, reports and provenance
- the Makefile is complete and current
- the Nox configuration is complete and current
- architecture contract tests prevent the main drift classes from returning
- Pyright/Pylance-compatible type checking passes
- Ruff passes
- formatting passes
- the complete test suite passes
- the final hostile audit finds no actionable defect

Only then mark the matrix items as `VERIFIED`.

Until then, continue working.