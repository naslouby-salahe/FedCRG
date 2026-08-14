# FedCRG Architecture Simplification and Full Implementation Goal

Work directly on the current FedCRG repository until the repository satisfies this goal completely.

This is an idempotent long-running goal.

Do not stop after one cleanup pass.
Do not stop because the repository currently passes tests.
Do not stop because an existing audit says something is complete.
Do not trust previous architectural audits.

Continuously:

1. inspect,
2. audit,
3. simplify,
4. refactor,
5. implement,
6. remove,
7. validate,
8. re-audit,
9. repeat,

until the repository converges to the target architecture and there are no meaningful violations remaining.

The objective is not merely to make the current architecture pass tests.

The objective is to make FedCRG substantially smaller, clearer, more strongly typed, less repetitive, easier to navigate, scientifically faithful, configuration-driven, restart-safe, observable during execution, and easy to reproduce for publication.

---

# 1. Operating Scope

Repository:

    /home/naslouby/Projects/FedCRG

Work on the current working tree and current `main`.

Do not inspect git history to recover:

- previous matrices;
- previous temporary folders;
- removed architecture;
- old implementations;
- migration paths;
- deleted compatibility APIs;
- previous prompts.

Do not use git history as architectural guidance.

The current repository, current authoritative roadmap under `docs/`, current data, current tests, and this goal are the inputs.

No backwards compatibility is required.

Do not preserve old APIs, old paths, old classes, old filenames, old serialized structures, old configuration layouts, or compatibility redirects merely because they already existed.

When an old design is inferior, delete it and migrate the repository cleanly.

---

# 2. First Cycle Must Start Fresh

On the first cycle only:

1. Locate the authoritative current FedCRG roadmap under `docs/`.
   - Prefer `docs/FedCRG Roadmap.md` if it exists.
   - Otherwise use the current FedCRG roadmap document that exists in `docs/`.
   - Do not inspect git history to locate older versions.

2. Read the complete roadmap before making scientific decisions.

3. Create a brand-new audit/implementation matrix:

       docs/FedCRG Audit Matrix.md

4. The matrix must be freshly derived from:
   - the current roadmap;
   - the current working tree;
   - the target architecture in this prompt.

5. Do not reuse an old matrix.
   If a matrix currently exists, rebuild it from scratch rather than trusting its statuses.

6. Create a fresh restart-safe tracking directory:

       docs/work/

   It may contain structured working-state files such as:

       docs/work/state.json
       docs/work/violations.json
       docs/work/actions.json
       docs/work/verification.json

   Prefer JSON for machine state.

   Do not create a collection of temporary Markdown status documents.

7. Tracking state must allow another agent process to resume from the current repository state without relying on conversation history.

After the initial cycle, update the matrix and JSON tracking state instead of recreating them unless they become structurally invalid.

---

# 3. Roadmap Usage Rule

The roadmap is scientific input, not a runtime dependency.

Read it thoroughly and extract its requirements into the fresh audit matrix.

Production source code must not contain comments, docstrings, class names, function names, CLI messages, filenames, serialized fields, runtime reports, or implementation terminology referring to:

- roadmap;
- matrix;
- prompt;
- migration;
- legacy;
- compatibility;
- old implementation;
- old architecture;
- phase numbers used only for implementation tracking.

The implementation must express the scientific concept directly.

The audit/tracking documents under `docs/` may record requirement provenance, but production code must stand on its own.

---

# 4. Mandatory Target Repository Architecture

Converge toward this architecture.

Do not preserve additional packages merely because they already exist.

A deviation is acceptable only when there is a strong technical or scientific reason and the alternative is demonstrably simpler.

    FedCRG/
    ├── README.md
    ├── pyproject.toml
    ├── Makefile
    ├── noxfile.py
    │
    ├── config/
    │   ├── study.yaml
    │   ├── datasets.yaml
    │   └── experiments.yaml
    │
    ├── docs/
    │   ├── <authoritative FedCRG roadmap>
    │   ├── FedCRG Audit Matrix.md
    │   └── work/
    │       ├── state.json
    │       ├── violations.json
    │       ├── actions.json
    │       └── verification.json
    │
    ├── data/
    │   ├── raw/
    │   └── preprocessed/
    │
    ├── src/
    │   └── fedcrg/
    │       ├── __init__.py
    │       ├── __main__.py
    │       ├── cli.py
    │       ├── types.py
    │       ├── config.py
    │       │
    │       ├── data/
    │       │   ├── __init__.py
    │       │   ├── datasets.py
    │       │   └── preprocessing.py
    │       │
    │       ├── learning/
    │       │   ├── __init__.py
    │       │   ├── detectors.py
    │       │   ├── federated.py
    │       │   └── scores.py
    │       │
    │       ├── thresholding/
    │       │   ├── __init__.py
    │       │   ├── readiness.py
    │       │   ├── policies.py
    │       │   └── metrics.py
    │       │
    │       ├── experiments/
    │       │   ├── __init__.py
    │       │   ├── runner.py
    │       │   └── analyses.py
    │       │
    │       ├── evidence/
    │       │   ├── __init__.py
    │       │   ├── models.py
    │       │   └── store.py
    │       │
    │       ├── reporting.py
    │       └── runtime.py
    │
    ├── tests/
    │   ├── contract/
    │   ├── integration/
    │   └── unit/
    │
    ├── outputs/
    │   ├── logs/
    │   ├── monitoring/
    │   ├── cache/
    │   │   ├── models/
    │   │   ├── scores/
    │   │   └── analysis/
    │   ├── runs/
    │   ├── campaigns/
    │   ├── figures/
    │   └── reports/
    │
    └── results/

Avoid additional package levels unless they remove more complexity than they introduce.

Do not create packages containing one trivial file.

Do not create one file for every tiny strategy or one-line function.

At the same time, do not create uncontrolled 3,000-line god modules.

Split by genuine capability and cohesion.

---

# 5. Architecture Simplification Rules

Aggressively identify and eliminate:

- redirect modules;
- import-only modules;
- one-method service classes;
- stateless factories;
- stateless managers;
- stateless handlers;
- unnecessary stores;
- wrapper methods that only forward arguments;
- wrapper classes around a single function;
- duplicate result classes;
- duplicate manifest representations;
- duplicate configuration representations;
- duplicate serializers;
- duplicate runners;
- duplicate planners;
- duplicate validators;
- duplicate policy-selection metadata;
- duplicate experiment catalogues;
- duplicate experiment group lists;
- duplicate path definitions;
- duplicate filesystem abstractions;
- duplicate status models;
- dead code;
- unused code;
- unreachable branches;
- old compatibility aliases;
- compatibility shims;
- transitional redirects.

Names such as the following require strong justification:

- utils
- helpers
- common
- manager
- handler
- processor
- engine
- service
- base
- models
- registry
- factory

Prefer names describing the scientific or application responsibility.

Do not create `FooFactory` when a constructor or function is sufficient.

Do not create `FooStore` for two lines of Pydantic persistence unless it owns meaningful behavior.

Do not create an interface solely to wrap one implementation.

---

# 6. No "Canonical" Terminology

Production code must contain no unnecessary use of:

- canonical
- canonicalize
- canonicalized
- canonical_name
- canonical_payload
- canonical representation

Use names describing what actually happens, for example:

- serialized
- normalized
- validated
- resolved
- stable
- sorted
- encoded
- persisted
- deterministic

Add a repository architecture/hygiene test that scans production source and fails if forbidden canonical terminology reappears unless a third-party API absolutely requires the exact term.

---

# 7. Primitive Leakage Elimination

This is a major objective.

Continuously search the production source for leaked primitive annotations and weakly typed structures involving:

- `float`
- `int`
- `str`
- `object`
- `Any`
- `dict`
- `list`
- untyped tuples
- arbitrary mappings
- generic JSON dictionaries

Do not blindly replace every internal arithmetic scalar with a wrapper object.

The rule is:

Important domain, scientific, configuration, experiment, evidence, persistence, orchestration, and public API boundaries must not communicate using ambiguous primitives when a meaningful constrained type, enum, model, or typed collection can express the concept.

Primitive use is acceptable only at legitimate boundaries such as:

- NumPy;
- PyTorch;
- pandas/Polars/PyArrow;
- filesystem APIs;
- JSON/YAML library input before validation;
- CLI input before validation;
- external library APIs;
- low-level arithmetic inside a scientifically clear implementation.

Convert into typed domain values immediately after the boundary.

Prefer Pydantic-constrained aliases for scalar semantics rather than creating dozens of pointless wrapper classes.

Examples of concepts that should have constrained aliases or equivalent strong types:

- ClientId
- RunId
- CampaignId
- Sha256
- ModelSeed
- CalibrationSeed
- AnalysisSeed
- Probability
- Alpha
- Fpr
- Tpr
- ConfidenceLevel
- Assurance
- LearningRate
- PositiveCount
- NonNegativeCount
- SampleCount
- RoundCount
- EpochCount
- BatchSize
- FeatureCount
- ReplicateCount
- ByteCount
- Threshold
- Score
- Fraction

Use constructs such as:

    Annotated[float, Field(...)]
    Annotated[int, Field(...)]
    Annotated[str, StringConstraints(...)]

Use enums for closed identity/state sets.

Use Pydantic models where multiple fields form an invariant.

Do not create a class merely to hold one unrestricted primitive.

---

# 8. Mandatory Primitive-Leakage Tests

Add AST-based architecture tests.

Do not rely only on grep.

The tests must scan `src/fedcrg/` and identify annotations containing inappropriate use of:

- float
- int
- str
- object
- Any
- dict
- bare list
- weak generic mappings

The tests must have an explicit, small boundary allowlist where primitive annotations are unavoidable.

The allowlist must be documented in the test itself by architectural reason, not by convenience.

The test must fail when a new primitive leak is introduced outside approved boundary locations.

Also scan for:

    dict[str, object]
    dict[str, Any]
    Mapping[str, object]
    Mapping[str, Any]
    list[dict[...]]
    -> object
    -> Any
    Any
    typing.Any

Do not weaken the test simply to make violations disappear.

Fix the production types first.

---

# 9. Pydantic v2 Must Own Structured Boundaries

Use Pydantic v2 extensively and consistently.

Prefer:

- `BaseModel`
- `ConfigDict`
- frozen models
- `Annotated`
- `Field`
- `StringConstraints`
- discriminated unions
- `TypeAdapter`
- `model_validate`
- `model_validate_json`
- `model_dump`
- `model_dump_json`
- computed fields where appropriate

Create one shared frozen-model configuration rather than repeating identical model configuration everywhere.

Replace handwritten serialization and conversion infrastructure where Pydantic already provides the behavior.

Remove custom infrastructure equivalent to:

- recursive arbitrary-object-to-JSON converters;
- `as_json_int`;
- `as_json_float`;
- `as_json_dict`;
- `as_json_list`;
- manual enum decoding;
- manual nested dataclass reconstruction;
- repeated `json.loads` plus dozens of casts.

Keep only genuinely useful atomic-write/checksum helpers.

Do not mix dataclasses, dictionaries, Pydantic models, and arbitrary JSON representations for the same persisted concept.

Persisted scientific evidence must have a clear schema.

---

# 10. Configuration Must Have One Source of Truth

The target is:

    config/study.yaml
    config/datasets.yaml
    config/experiments.yaml

Eliminate the current fragmented configuration hierarchy when possible.

Remove:

- config files that only redirect to another config;
- `extends` chains;
- configuration inheritance graphs;
- Python deep-merge implementations;
- multiple files containing the same scientific values;
- Python experiment catalogues duplicating YAML experiment definitions.

`study.yaml` should own shared study-level scientific configuration such as:

- protocol parameters;
- statistical parameters;
- randomness;
- detector/training profiles where appropriate;
- non-dataset study-wide settings.

`datasets.yaml` should own dataset contracts.

`experiments.yaml` must own the experiment catalogue, including as appropriate:

- experiment id;
- experiment category;
- dataset/profile;
- detector/profile;
- policies;
- axes;
- coupled cells;
- repetitions;
- dependencies;
- required evidence;
- expected workload;
- confirmatory/diagnostic classification.

Do not maintain the same experiment grid in Python and configuration.

Python executes a typed `ExperimentSpec`.

It must not separately redefine what the experiment contains.

---

# 11. Hardcoded Scientific Values Audit

Continuously inspect Python source for hardcoded scientific values.

Search AST literals and manually inspect suspicious literals.

Examples include:

- probabilities;
- alpha values;
- rho values;
- confidence levels;
- assurances;
- seed values;
- sample counts;
- dataset expected counts;
- minimum row counts;
- repetitions;
- experiment axis values;
- contamination fractions;
- thresholds;
- tolerance values;
- learning rates;
- rounds;
- epochs;
- batch sizes;
- detector dimensions;
- hidden-layer sizes;
- percentages;
- benchmark warmups;
- workload sizes;
- scientific numerical tolerances.

Every scientific value must be one of:

1. owned by typed configuration;
2. mathematically derived from configured values;
3. an inherent mathematical constant clearly justified by the algorithm.

Do not move meaningless constants into config merely to satisfy a test.

But do not hide configurable scientific assumptions in source.

---

# 12. Mandatory Config-vs-Source Drift Tests

Create architecture tests that detect scientific values duplicated between configuration and Python source.

The test should:

1. parse the structured configuration;
2. collect configured numeric/string scientific values where practical;
3. scan production AST literals;
4. report suspicious occurrences of configured values repeated directly in production code.

This is a drift detector, not an excuse to create a huge allowlist.

If a configured value is duplicated in source, first attempt to remove the source literal and use the typed config value.

Allow genuine mathematical constants only when justified.

Also test that:

- experiment axes are not duplicated in Python;
- configured seed lists are not duplicated in Python;
- configured sample sizes are not duplicated in Python;
- configured model/training values are not duplicated in Python.

---

# 13. No Hidden Scientific Defaults

Search continuously for Python defaults.

Scientific configuration values must not silently appear because Python supplied a default.

Audit:

- Pydantic defaults;
- dataclass defaults;
- function argument defaults;
- CLI defaults;
- fallback values using `.get(..., default)`;
- `or <value>` fallback logic;
- constants used as fallback configuration;
- implicit behavior based on missing fields.

A scientific value must either:

- be explicitly configured; or
- be truly invariant and therefore not pretend to be configurable.

Examples such as optimizer choice, aggregation rule, early stopping, deterministic execution, training sizes, statistical parameters, or experimental settings must not silently become scientific defaults.

Structural/runtime conveniences may have reasonable defaults only when they cannot alter scientific interpretation.

---

# 14. Dataset Preprocessing Must Be Reusable

This is mandatory.

All deterministic preprocessed datasets must live under:

    data/preprocessed/

Use a clear cache identity such as:

    data/preprocessed/<dataset-id>/<preprocessing-id>/

The preprocessing identity must be derived from everything that changes the preprocessed data, for example as applicable:

- dataset identity;
- source file identities/hashes;
- feature contract;
- split contract;
- preprocessing configuration;
- parser version;
- preprocessing implementation/schema version;
- relevant deterministic randomness.

Do not include unrelated experiment values.

A valid existing preprocessed dataset must be reused.

Campaigns and individual experiments must not preprocess the same unchanged raw data again.

Implement explicit validation before reuse:

- manifest exists;
- expected files exist;
- hashes/checksums match;
- schema matches;
- source identity matches;
- preprocessing identity matches.

If valid, reuse it and log clearly:

    Reusing preprocessed dataset ...

If invalid or scientifically incompatible, rebuild it.

Do not silently reuse stale data.

Do not silently rebuild valid data.

Use atomic finalization or locking so interrupted preprocessing cannot be mistaken for a valid cache.

Where multiple experiment definitions share the same preprocessing identity, they must reference the same preprocessed artifacts.

---

# 15. Preprocessing CLI

Provide a clear command such as:

    fedcrg preprocess [DATASET_ID]

Support an explicit overwrite/rebuild option only if useful:

    fedcrg preprocess [DATASET_ID] --overwrite

The normal behavior must be reuse-first.

Campaign execution must invoke the exact same preprocessing capability.

There must not be a campaign-only alternate preprocessing implementation.

---

# 16. Output Architecture

Runtime/generated artifacts belong under:

    outputs/

Use:

    outputs/logs/
    outputs/monitoring/
    outputs/cache/models/
    outputs/cache/scores/
    outputs/cache/analysis/
    outputs/runs/
    outputs/campaigns/
    outputs/figures/
    outputs/reports/

Do not commit generated scientific outputs except required empty-directory markers when needed.

Do not duplicate cached models or scores into every policy run.

Reference immutable upstream evidence instead.

Model caches should be reused when their full scientific training identity matches.

Score caches should be reused when dataset/model/scoring identity matches.

Analysis caches should be reused when their full analysis identity matches.

Cache reuse must always be validated.

---

# 17. Run Evidence Should Be Simple

Avoid deeply nested per-run folder structures containing mostly references.

Prefer a concise structure such as:

    outputs/runs/<run-id>/
    ├── run.json
    ├── thresholds.parquet
    ├── metrics.parquet
    └── verification.json

Add files only when scientifically required.

Do not create folders just because previous architecture had them.

Do not encode the entire scientific experiment into a human-formatted directory name.

Prefer a stable typed run identity/hash and store structured metadata in `run.json`.

---

# 18. JSON, JSONL and Parquet Rules

Markdown is documentation.

It is not runtime state.

Use:

- JSON for structured manifests, status, provenance, summaries, configs after resolution, verification, environment records;
- JSONL for streaming logs/telemetry where appropriate;
- Parquet for tabular scientific evidence such as client metrics, threshold records and large result tables;
- CSV only as an interoperability/publication export;
- PDF/PNG/SVG for figures as appropriate;
- Markdown only for human-authored documentation or an explicitly requested presentation export.

Do not generate Markdown files as the primary machine-readable source of truth.

Runtime logic must never parse generated Markdown.

---

# 19. Publication Results Bundle

Implement a dedicated publication-results capability.

Provide a command such as:

    fedcrg results build [CAMPAIGN_ID]

and:

    fedcrg results verify [CAMPAIGN_ID]

or an equivalently clear CLI.

Campaign completion must use the exact same results builder.

Do not maintain one manual CLI builder and another campaign builder.

Publication-ready bundles must live under:

    results/<campaign-id>/

Suggested structure:

    results/<campaign-id>/
    ├── manifest.json
    ├── summary.json
    ├── configuration.json
    ├── environment.json
    ├── provenance.json
    ├── checksums.json
    ├── statistics/
    │   └── *.json
    ├── tables/
    │   ├── *.parquet
    │   └── *.csv
    └── figures/
        ├── *.pdf
        └── *.png

The exact contents must follow the scientific roadmap.

The publication bundle must:

- include only finalized evidence;
- reference the exact campaign;
- include resolved scientific configuration;
- include environment/provenance;
- include checksums;
- include claim-bearing tables;
- include statistical outputs;
- include publication figures;
- be reproducible from immutable run evidence;
- never fabricate missing experiments;
- clearly represent incomplete evidence;
- be verifiable independently.

The results builder must not retrain models.

It must not recompute scientific experiments.

It must project already generated immutable evidence into a paper/publication bundle.

---

# 20. Policy Architecture

Do not keep a directory containing one file per trivial policy.

Merge tightly related policy implementations into:

    thresholding/policies.py

Use a typed policy specification.

A policy definition should centrally describe, as appropriate:

- PolicyId;
- evidence regime;
- deployability;
- required inputs;
- evaluator;
- supervised/unsupervised status;
- result semantics.

Do not separately maintain:

- supervised-policy sets;
- deployability helpers;
- information-regime helpers;
- giant selector `if/elif` ladders;
- separate CLI policy metadata;
- separate experiment policy metadata.

One typed policy definition should drive all of these where possible.

---

# 21. Experiment Architecture

There must be one execution spine.

Converge toward:

    experiments/runner.py
    experiments/analyses.py

Remove unnecessary distinctions between:

- runner;
- experiment runner;
- execution;
- materializer;
- planner;
- policy-cell runner;
- campaign runner wrappers;
- completion service;
- preflight wrapper;
- verification wrapper.

Keep a separate object/function only if it owns a coherent responsibility and meaningfully reduces complexity.

A campaign should orchestrate the same experiment runner used for individual execution.

Do not create parallel implementations.

Use the standard library where possible.

For dependency ordering prefer:

    graphlib.TopologicalSorter

rather than maintaining a custom topological-sort implementation.

---

# 22. Evaluation and Analysis

Merge trivial metric modules when they are strongly related.

Do not create one module for every 10-line metric.

Use NumPy, SciPy, scikit-learn and other trusted libraries where they exactly implement the locked scientific definition.

Do not reimplement standard mathematics simply to avoid a dependency that already exists.

However, do not replace scientifically specific procedures with library defaults if doing so changes semantics.

Exact roadmap definitions always win.

Document meaningful scientific deviations from library defaults in code by describing the scientific reason, not the implementation history.

---

# 23. GPU and Memory Use

Use GPU aggressively for workloads that benefit materially from GPU execution.

Training and score generation should use CUDA when configured and available.

Do not silently fall back from requested CUDA execution to CPU.

Log:

- selected compute device;
- GPU name;
- CUDA availability;
- total VRAM;
- allocated VRAM;
- peak VRAM where available;
- CPU usage;
- process RAM;
- available system RAM;
- relevant stage durations.

Use:

- bounded batch sizes;
- streaming;
- Parquet scanning;
- lazy/streaming dataframe execution where appropriate;
- `torch.inference_mode()` for inference;
- careful tensor/device lifetimes;
- explicit cache cleanup only where technically appropriate.

Do not load an entire large dataset into RAM merely for convenience.

Avoid OOM.

Prefer throughput that remains safely below RAM/VRAM exhaustion rather than maximally filling memory.

Never duplicate large arrays unnecessarily.

---

# 24. Runtime Observability

Running experiments must be easy to understand from the terminal.

Use Rich or equivalent existing dependency appropriately.

Provide clean progress output showing information such as:

- campaign id;
- experiment id;
- dataset;
- current stage;
- current model seed;
- current calibration seed where relevant;
- policy or analysis;
- cache hit/miss/reuse;
- preprocessing reuse;
- elapsed time;
- completed/total work;
- CPU/RAM usage;
- GPU/VRAM usage;
- warnings;
- failures;
- current output location.

Do not scatter `print()` across production source.

Use structured logging.

Persist logs under:

    outputs/logs/

Persist resource telemetry under:

    outputs/monitoring/

Long-running campaign execution must produce enough information to determine what the program is currently doing without opening source code.

---

# 25. Monitoring Command

Provide a command such as:

    fedcrg monitor

It should show current or recent runtime resource state and, where possible, campaign/run progress.

It should expose:

- CPU;
- RAM;
- GPU;
- VRAM;
- active campaign/run;
- current stage;
- elapsed time;
- cache activity.

Use structured telemetry.

Do not invent a second monitoring state separate from campaign state.

---

# 26. CLI Simplification

Keep the public CLI simple.

Prefer a surface around:

    fedcrg validate [EXPERIMENT_ID]
    fedcrg preprocess [DATASET_ID] [--overwrite]
    fedcrg plan [EXPERIMENT_ID]
    fedcrg run [EXPERIMENT_ID] [--overwrite]
    fedcrg campaign [--overwrite]
    fedcrg status [EXPERIMENT_ID]
    fedcrg monitor
    fedcrg report [CAMPAIGN_ID]
    fedcrg results build [CAMPAIGN_ID]
    fedcrg results verify [CAMPAIGN_ID]

Adapt names only where a clearly better interface exists.

Do not expose internal implementation stages as top-level CLI commands merely because functions exist for them.

CLI functions must remain thin.

Scientific behavior belongs in typed application code.

Avoid repetitive option parsing and repetitive JSON construction.

---

# 27. Makefile

Rebuild/update the Makefile to reflect the final CLI and quality workflow.

It should be clean, short and useful.

Provide appropriate targets such as:

    help
    install
    format
    lint
    typecheck
    test
    test-unit
    test-integration
    test-contract
    test-regression
    audit
    validate
    preprocess
    plan
    run
    campaign
    status
    monitor
    report
    results
    verify-results
    quality

Do not put scientific constants in Makefile.

Do not duplicate application logic in Makefile.

Targets should call the real CLI or development tools.

Remove stale targets.

---

# 28. Nox

Maintain an updated `noxfile.py`.

Provide clean sessions such as:

    format
    lint
    typecheck
    unit
    integration
    contract
    regression
    audit
    quality

Reuse `pyproject.toml` configuration.

Do not duplicate Ruff/Pyright/pytest configuration in Python when configuration already exists in `pyproject.toml`.

Remove stale sessions.

---

# 29. Static Quality

Strengthen quality enforcement.

Use Ruff comprehensively rather than only minimal E/F checks.

Enable useful rules covering areas such as:

- pyflakes;
- pycodestyle;
- pyupgrade;
- bugbear;
- simplify;
- import sorting;
- naming;
- comprehensions;
- pathlib;
- exception quality;
- unnecessary constructs.

Keep intentional scientific code readable.

Move Pyright toward strict typing.

Production code should not rely on:

- Any;
- object;
- unchecked dictionaries;
- broad type ignores;
- blanket noqa;
- casts hiding a bad model.

Fix the underlying type model.

---

# 30. Architecture Tests

Delete or rewrite tests that preserve the old architecture.

Do not keep tests asserting that obsolete packages/files must exist.

Create contract tests for the target architecture.

They must verify at least:

- only intended production packages exist;
- obsolete package names do not return;
- no pipeline/orchestration duplicate layer exists;
- no config inheritance/redirect structure exists;
- no forbidden canonical terminology exists;
- production source contains no roadmap/matrix/prompt references;
- no compatibility shims exist;
- no vague files such as generic utils/helpers appear without explicit allowance;
- no primitive leakage appears outside approved boundaries;
- no `Any` appears in production code;
- no weak generic dict/object result payloads leak through application/domain boundaries;
- no experiment catalogue is duplicated in Python;
- scientific config values are not duplicated as source literals;
- preprocessed data root is exactly `data/preprocessed`;
- output roots conform to the target structure;
- publication results root is exactly `results/`;
- campaign and direct commands share preprocessing/execution/results implementations.

---

# 31. Test Migration

As files and classes move:

- update relevant tests;
- rewrite tests around behavior rather than old internal names;
- delete tests for removed behavior;
- delete tests for deleted wrappers;
- delete compatibility tests;
- preserve scientific regression tests;
- add regression tests before risky scientific simplifications where necessary.

Do not create duplicate tests for the same behavior solely because old and new APIs coexist.

The old API should normally be removed.

---

# 32. Preprocessing Reuse Tests

Add tests proving:

1. first call builds preprocessing;
2. second equivalent call reuses it;
3. campaign execution reuses the same artifact;
4. two experiments sharing preprocessing identity reuse the same artifact;
5. changed preprocessing config changes the identity;
6. changed source identity invalidates reuse;
7. incomplete/interrupted preprocessing is not reused;
8. corrupted files are not reused;
9. overwrite explicitly rebuilds;
10. valid cache reuse does not rewrite artifacts.

---

# 33. Cache Reuse Tests

Similarly test model/score/analysis cache identity and reuse.

A calibration-only or policy-only change must not retrain/rescore when the underlying model/score identity is unchanged.

A training-specification change must invalidate the model cache.

A model change must invalidate the dependent score cache.

A report/results rebuild must not retrain or rescore.

---

# 34. Publication Bundle Tests

Add tests that verify:

- results bundle creation;
- deterministic manifest structure;
- checksums;
- missing evidence remains missing/incomplete;
- finalized evidence is included;
- expected tables exist;
- expected figures exist;
- provenance/config/environment files exist;
- campaign automatically invokes the same results builder;
- direct `results build` and campaign result creation are equivalent;
- `results verify` detects tampering;
- building results never invokes training or scoring.

---

# 35. Runtime/Console Tests

Test important runtime behavior without overfitting Rich rendering internals.

Verify that:

- stages are reported;
- cache reuse is reported;
- preprocessing reuse is reported;
- failures include useful context;
- resource telemetry is persisted;
- campaign status survives restart;
- logs are persisted;
- no scattered production `print()` calls exist.

---

# 36. Naming Audit

Continuously audit names.

Rename vague or misleading variables.

Examples of names that should generally be improved:

- `x`
- `n`
- `v`
- `d`
- `data`
- `value`
- `item`
- `result`
- `record`
- `obj`
- `thing`
- `tmp`
- `raw`

Short mathematical names are acceptable inside a small mathematical formula implementation when their meaning is standard and locally obvious.

Structured application logic should use descriptive names such as:

- exceedance_count;
- sample_count;
- reference_threshold;
- local_threshold;
- applied_threshold;
- calibration_score;
- client_metrics;
- experiment_spec;
- model_seed;
- calibration_seed;
- campaign_state;
- preprocessing_identity.

Do not rename standard mathematical quantities into awkward prose merely for length.

Optimize for clarity.

---

# 37. Comments and Docstrings

Remove comments that merely narrate syntax.

Remove AI-style commentary.

Remove implementation-history commentary.

Do not mention:

- old code;
- migration;
- previous structure;
- roadmap;
- matrix;
- prompt;
- backward compatibility.

Comments should explain only useful scientific or engineering rationale that cannot be understood from the code itself.

Docstrings should describe contracts and semantics.

---

# 38. README

Keep README clean and short.

It should explain:

- what FedCRG is;
- installation;
- data placement;
- preprocessing;
- core commands;
- campaign execution;
- status/monitoring;
- results bundle generation;
- output locations;
- scientific reproducibility expectations.

Do not make README an architecture migration diary.

Do not reference implementation prompts or audit progress.

Do not duplicate the scientific roadmap inside README.

---

# 39. Continuous Audit Loop

After every substantial implementation chunk:

1. inspect the current tree;
2. update `docs/FedCRG Audit Matrix.md`;
3. update `docs/work/state.json`;
4. update `docs/work/violations.json`;
5. search again for:
   - primitive leaks;
   - Any;
   - object;
   - raw dicts;
   - weak mappings;
   - hardcoded scientific values;
   - hidden defaults;
   - duplicate configuration;
   - duplicate classes;
   - wrappers;
   - redirects;
   - dead code;
   - stale imports;
   - stale tests;
   - vague names;
   - canonical terminology;
   - roadmap/matrix/prompt references;
   - generated Markdown used as state;
   - unnecessary directories;
   - cache duplication;
   - preprocessing duplication.

6. fix newly exposed problems;
7. continue.

Do not assume a previous scan was complete.

Refactoring frequently exposes additional duplication.

Repeat until successive hostile audits find no meaningful new issue.

---

# 40. Validation Cadence

Do not waste time running the entire quality suite after every tiny edit.

During a coherent refactor chunk, use targeted checks as needed.

After a substantial coherent chunk, run the relevant quality checks.

Before declaring a cycle complete, run the complete quality suite.

The final cycle must include:

    ruff format
    ruff check
    pyright
    pytest
    nox -s quality

or the equivalent final commands defined by the repository.

Tests should use parallel execution where safe.

Fix failures rather than weakening quality settings.

---

# 41. CLI and Runtime Smoke Validation

Actually invoke the implemented commands to verify their wiring.

At minimum exercise safe commands such as:

    fedcrg --help
    fedcrg validate ...
    fedcrg plan ...
    fedcrg preprocess ...
    fedcrg status ...
    fedcrg monitor ...
    fedcrg results --help

Use small fixtures/smoke execution where required to validate training/scoring/campaign wiring.

Do not launch an enormous confirmatory experiment merely to prove a refactor works if a scientifically representative smoke execution is sufficient.

When a GPU smoke workload exists, exercise the CUDA path.

Monitor CPU, RAM, GPU and VRAM behavior during realistic smoke execution.

Look for avoidable memory duplication.

---

# 42. Do Not Game Audits

Never make an audit pass by:

- broadening an allowlist;
- hiding a primitive behind an alias with no constraint or semantics;
- moving a hardcoded value to another Python file;
- renaming a wrapper;
- putting duplicated values in constants;
- adding `# noqa`;
- adding `type: ignore`;
- skipping files;
- excluding packages;
- deleting meaningful tests;
- lowering type-checking strictness;
- replacing a failing scientific assertion with a weaker assertion.

Resolve the underlying problem.

---

# 43. Scientific Integrity

Architecture simplification must not cause scientific drift.

For every significant scientific refactor:

1. identify the roadmap requirement;
2. identify the current scientifically meaningful behavior;
3. preserve the correct mathematical/data semantics;
4. update tests;
5. verify against locked numerical/regression expectations where available.

Do not change an algorithm merely because a library offers something similar.

Reuse libraries only when semantics match.

Do not invent missing science.

If the roadmap and implementation genuinely conflict, use the current authoritative roadmap as scientific authority and update implementation accordingly.

Record the resolution in the audit matrix.

---

# 44. Deletion Is Expected

This goal should result in substantial deletion.

Do not measure success by how much new code is written.

Prefer:

- fewer modules;
- fewer classes;
- fewer wrappers;
- fewer config files;
- fewer serializers;
- fewer duplicate representations;
- fewer directories;
- fewer commands;
- fewer internal abstractions;
- stronger types;
- clearer responsibilities.

If 500 lines of generic infrastructure can be replaced safely with 50 lines using Pydantic, PyArrow, NumPy, SciPy, stdlib or another already appropriate dependency, do so.

---

# 45. Completion Criteria

Do not declare the goal complete until all of the following are true:

- the source tree closely matches the target architecture;
- obsolete package architecture is gone;
- the configuration hierarchy is simplified;
- there is one experiment source of truth;
- no meaningful scientific config is duplicated in Python;
- primitive leakage tests pass;
- Any/object/raw-dict leakage is eliminated outside legitimate boundaries;
- hardcoded scientific-value drift tests pass;
- hidden scientific defaults are removed;
- canonical terminology is absent;
- production roadmap/matrix/prompt references are absent;
- trivial wrappers and redirects are gone;
- duplicate runners/planners/materializers are gone;
- policies have a clear centralized architecture;
- preprocessed data lives in `data/preprocessed`;
- preprocessing reuse works and is proven by tests;
- model and score reuse works when scientifically valid;
- outputs are cleanly organized under `outputs`;
- logs and monitoring are useful;
- GPU execution is used where appropriate;
- RAM/VRAM use is bounded;
- run status is restart-safe;
- `results/<campaign-id>/` publication bundles work;
- campaign execution automatically uses the publication results builder when appropriate;
- results verification detects corruption;
- Makefile matches the final architecture;
- noxfile matches the final architecture;
- CLI matches the final architecture;
- README matches the final architecture;
- old architecture tests are removed;
- new architecture tests exist;
- scientific regression tests pass;
- Ruff passes;
- formatting passes;
- Pyright passes at the strongest practical strictness;
- pytest passes;
- nox quality passes;
- repeated hostile audits discover no substantive remaining cleanup issue.

---

# 46. Loop Termination Rule

At the end of each cycle ask internally:

- Is the target tree actually achieved?
- Can any package be removed or merged?
- Can any class become a type/model/function?
- Can any wrapper disappear?
- Is any scientific value still hardcoded?
- Is any configured value duplicated in source?
- Is any primitive leaking through an important boundary?
- Is any `Any`, `object`, raw `dict`, or weak mapping still present?
- Does any hidden default remain?
- Is any experiment definition duplicated?
- Is any config file merely redirecting?
- Is any cache being recomputed unnecessarily?
- Is preprocessing always reused when valid?
- Are model and score caches reused correctly?
- Is runtime output understandable?
- Is resource monitoring useful?
- Is the publication bundle complete and verifiable?
- Are tests preserving obsolete architecture?
- Are names clear?
- Does source contain canonical terminology?
- Does production source reference roadmap/matrix/prompt?
- Can existing libraries replace more custom infrastructure?
- Is there scientific drift?

If any answer reveals a meaningful problem, continue working.

Only stop when repeated fresh audits converge with no substantive issue remaining.

Do not stop merely because the matrix says VERIFIED.

The repository itself is the evidence.