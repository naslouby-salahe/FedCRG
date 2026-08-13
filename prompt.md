# FedCRG — Final Roadmap Alignment and Implementation Completion

You are responsible for completing the existing FedCRG repository as production-quality research software.

This is an **idempotent continuation task**, not a greenfield rewrite.

The repository already contains a large architectural refactor in GitHub PR #1:

* repository: `naslouby-salahe/FedCRG`
* PR: `#1`
* branch: `agent/rework-fedcrg-architecture`
* target branch: `main`

The scientific authority is the current FedCRG roadmap under `docs/`.

Your goal is to:

1. merge the existing architectural PR into `main` first;
2. continue all remaining work directly from the resulting `main`;
3. audit the actual implementation against the roadmap;
4. fix, complete, consolidate, remove, and wire whatever remains;
5. leave one coherent implementation with every supported workflow reachable from the CLI;
6. finish with a clean static/test verification of the implementation.

Do not redesign the project merely because another architecture is theoretically possible. Reuse and improve what the PR already established.

---

## 1. FIRST ACTION: MERGE THE EXISTING PR INTO `main`

Before doing new implementation work, inspect the repository and preserve all existing work.

Run the equivalent of:

```bash
git status --short --branch
git remote -v
git fetch --all --prune --tags
git log --oneline --decorate --graph -30
```

Inspect PR #1 and its branch.

If there are already uncommitted or unpushed changes belonging to that PR, preserve them and include them before the merge. Do not discard work.

Then merge PR #1 into `main`.

If GitHub refuses because the PR is still marked draft, mark it ready and merge it. Do not use draft status as a reason to postpone the requested merge.

After the merge:

```bash
git checkout main
git pull --ff-only
```

Confirm that the PR head is contained in `main`.

**Only after this merge should new implementation work begin.**

Continue working from `main`.

Do not create another long-lived refactor branch unless an actual Git restriction forces it.

Commit occasionally after substantial coherent changes or completed phases. Do not create a commit for every tiny edit.

---

## 2. READ THE ROADMAP, THEN AUDIT THE ACTUAL CODE

Read the complete current FedCRG roadmap before modifying the implementation.

Treat it as the authoritative scientific contract for:

* algorithms and formulas;
* thresholds and inequalities;
* finite-sample rules;
* data roles and leakage boundaries;
* N-BaIoT and DIAD handling;
* preprocessing;
* detector training;
* score caching;
* policy information regimes;
* metrics;
* S1–S6;
* R1–R14;
* statistical analysis;
* artifacts and provenance;
* reproducibility;
* required tables and figures;
* failure states;
* CLI behavior;
* claim discipline.

Then inspect the repository as it exists **after the PR merge**.

Also read:

```text
docs/protocol_implementation_ledger.md
docs/architecture.md
docs/experiment_protocol.md
docs/reproducibility.md
```

Do not trust `IMPLEMENTED` in the ledger merely because it is written there.

For each major roadmap requirement, determine whether it is:

* correctly implemented and wired;
* implemented but incomplete;
* duplicated;
* stale;
* scientifically wrong;
* unreachable;
* placeholder-only;
* implemented but missing tests;
* implemented but missing CLI integration;
* legitimately dependent on future experimental evidence.

Update the existing implementation ledger when necessary.

Do **not** create another enormous duplicate audit matrix.

Maintain only a small resumable progress record under `docs/tmp/` if useful so another session can immediately identify:

* what was audited;
* what was fixed;
* what remains;
* the next major work item.

Keep this lightweight.

---

## 3. IMPLEMENTATION COMPLETENESS IS NOT EXPERIMENTAL COMPLETION

Do not waste compute by running the entire publication campaign during this coding task.

The repository must make the complete roadmap **executable**, but this task does not require generating all publication evidence such as:

* the full 970,000-trial synthetic campaign;
* all five-seed real-data detector trainings;
* all 50 calibration-role permutations;
* every R1–R14 production run;
* complete DIAD external evidence;
* full Deep-SVDD publication runs;
* final machine-specific benchmarks;
* final publication tables from expensive experiments.

Implement and wire all of those paths.

Use small deterministic fixtures, unit tests, contract tests, integration tests, and smoke executions to establish that they work.

Do not fake, synthesize, or mark publication evidence complete merely to make `fedcrg verify` green.

If required experimental artifacts have not genuinely been generated, `fedcrg verify` should truthfully report them as incomplete.

The distinction must remain explicit:

```text
implementation complete != experiment evidence complete
```

---

## 4. NO DAG

Do not introduce or retain a DAG architecture.

No:

* DAG executor;
* task graph;
* node/edge workflow model;
* graph orchestration engine;
* graph scheduler;
* generic topological workflow abstraction;
* Dagster;
* Airflow-style abstraction;
* NetworkX workflow representation.

The roadmap already provides an understandable scientific execution sequence. Use straightforward application orchestration.

Experiment prerequisites may be represented as typed dependency metadata when scientifically necessary, but dependency checking must remain simple and explicit.

Do not model the programme as a generic graph merely to order experiments.

In particular, audit the current experiment dependency implementation that performs topological ordering and remove unnecessary graph/DAG semantics.

Prefer:

```text
validated experiment definition
→ prerequisite checks
→ deterministic execution sequence
→ execution
→ verification
```

over a workflow graph.

---

## 5. USE RESPONSIBILITY-BASED NAMES, NOT PAPER BOOKKEEPING AS ARCHITECTURE

The roadmap necessarily contains manuscript identifiers such as:

```text
Gate A
Gate B
S1–S6
R1–R14
B0–B10
```

These are scientific/publication identifiers.

They must **not** become the canonical architecture of the Python code.

Use names that describe what the implementation actually does.

Prefer concepts such as:

```text
readiness
reference mismatch
reference threshold
decision
feature sensitivity
calibration assignment
policy evaluation
experiment execution
artifact verification
```

Do not create canonical implementation concepts such as:

```text
gate_a.py
gate_b.py
GateAService
GateBManager
r14.py
R14Builder
run_r14()
s1_runner.py
B7Handler
CanonicalGate
CanonicalExperiment
```

unless an exact serialized research identifier genuinely has to be preserved.

Research codes may remain in:

* the experiment registry;
* configuration IDs;
* manifests;
* artifact identities;
* tables/figures;
* publication-facing metadata;
* tests verifying roadmap correspondence.

They should not determine class, module, service, or application-layer architecture.

The existing PR already moved the protocol toward:

```text
protocol/reference.py
protocol/readiness.py
protocol/mismatch.py
protocol/decision.py
protocol/service.py
```

Continue that direction consistently.

### Known cleanup target

The current PR contains overlapping R14-specific implementation in areas such as:

```text
application/r14.py
application/feature_sensitivity.py
data/r14_feature_contract.py
data/feature_sensitivity.py
```

Audit these together.

There should be one clear **DIAD feature-sensitivity** implementation, not parallel implementations named after the paper experiment number.

Perform the same overlap audit elsewhere, especially around:

```text
pipeline.py
research_pipeline.py
run_experiment.py

layout.py
experiment_layout.py

environment.py
environment_lock.py
```

Merge responsibilities where they are actually duplicated.

Do not preserve duplicates for compatibility.

---

## 6. ONE EXECUTION SPINE

Converge on one understandable execution path:

```text
CLI
→ application service
→ experiment/protocol/domain capability
→ artifact persistence
```

The CLI must not bypass the application layer and call internal scientific formulas directly.

The experiment layer must not duplicate application orchestration.

Artifact code must persist and verify evidence; it must not independently implement experiment logic.

Scientific formulas belong in the appropriate protocol/metrics/analysis modules.

There must not be three different ways to run the same experiment.

Search for reuse before creating new code.

When equivalent code exists:

1. reuse it;
2. refactor it if necessary;
3. merge duplicate responsibilities;
4. update callers;
5. delete the redundant path.

---

## 7. ALL CLI COMMANDS MUST BE REAL AND FULLY WIRED

The CLI is a first-class deliverable.

Every supported command must:

* be registered;
* parse and validate inputs;
* resolve typed configuration;
* invoke the real application service;
* execute the intended workflow;
* persist the correct artifacts;
* return meaningful success/failure;
* be covered by CLI/integration tests.

No command may exist merely as:

* an echo;
* placeholder;
* stub;
* partial wrapper;
* TODO;
* disconnected command group.

At minimum ensure the roadmap workflows are available end-to-end for:

```text
fedcrg doctor
fedcrg data prepare
fedcrg tables precompute-readiness
fedcrg synthetic run
fedcrg train
fedcrg score
fedcrg evaluate
fedcrg robustness deep-svdd
fedcrg benchmark
fedcrg report build
fedcrg verify
```

Also audit the experiment, sensitivity, robustness, reporting, and verification subcommands currently exposed by the refactored CLI.

If a command exists, it must work.

If a command is obsolete, remove it.

If functionality exists only in an internal module or old script but belongs to a supported research workflow, wire it through the proper CLI/application service.

CLI code must remain thin. It must not contain:

* scientific formulas;
* detector implementation;
* policy implementation;
* experiment grids;
* manual manifest interpretation;
* dataset inference from filenames;
* duplicated configuration rules.

Use responsibility-based CLI names rather than paper shorthand unless the identifier is genuinely part of the publication-facing contract.

Keep documentation and `--help` synchronized with the final command tree.

---

## 8. NO BACKWARD COMPATIBILITY

There is no backward-compatibility requirement.

Delete obsolete architecture instead of maintaining adapters to it.

Remove:

* the old root package;
* compatibility imports;
* forwarding modules;
* deprecated aliases;
* duplicate classes;
* obsolete config models;
* stale CLI commands;
* old experiment runners;
* duplicate registries;
* compatibility wrappers;
* transitional shims;
* dead tests written only for removed APIs.

After completion there must be one canonical source package:

```text
src/fedcrg/
```

Do not leave two ways of doing the same thing.

---

## 9. STRONG DOMAIN TYPES — NO PRIMITIVE OBSESSION

Domain concepts must not travel through the system as arbitrary primitive values.

Finite categorical domains must use enums.

Semantic identities should use explicit value objects.

Structured domain state should use frozen typed records.

Prefer:

```python
DatasetId
ExperimentId
PolicyId
ClientId
RunId
ModelSeed
CalibrationSeed
DecisionState
DecisionReason
ArtifactType
Sha256
```

over raw:

```python
str
int
tuple[str, ...]
```

when those primitives carry domain meaning.

Use:

* enums for finite choices;
* `@dataclass(frozen=True, slots=True)` for immutable domain records;
* explicit value objects for identities;
* Pydantic v2 models for validated configuration/boundary parsing.

A mathematical scalar may remain numeric when it genuinely represents mathematics, such as:

* probability;
* FPR;
* TPR;
* sample count;
* loss;
* threshold;
* learning rate.

Do not wrap every mathematical operation in meaningless classes.

The rule is to eliminate **primitive domain identity/state**, not mathematics.

`bool` is acceptable when it genuinely represents a boolean condition.

---

## 10. NO `dict` DOMAIN MODELS

Do not use dictionaries as scientific/domain schemas.

Do not expose APIs based on:

```python
dict[str, Any]
dict[str, object]
dict[ClientId, ...]
```

for persistent domain state.

Do not hide mutable dictionaries inside frozen dataclasses.

Use typed immutable records with explicit lookup behavior.

Generic mappings are acceptable only at unavoidable external boundaries such as YAML/JSON parsing or third-party APIs, and must be converted immediately into typed models.

Temporary implementation-local mappings are acceptable only when they:

* do not escape;
* are not the domain schema;
* are not serialized as the scientific contract;
* are immediately converted into typed structures.

Do not scatter serialization dictionaries throughout application code.

---

## 11. NO `Any`, BROAD CASTS, OR TYPE-SUPPRESSION DRIFT

Remove unjustified:

```text
Any
cast(Any, ...)
# type: ignore
# pyright: ignore
```

Do not disable useful Pyright diagnostics globally merely to achieve zero errors.

Third-party boundaries may require narrow, documented exceptions, but the internal FedCRG API should remain strongly typed.

Audit all public method inputs and outputs.

No vague `object` return types where a proper result model is known.

---

## 12. CONFIGURATION HAS ONE SOURCE OF TRUTH

Use the roadmap-approved configuration flow:

```text
YAML
→ validated Pydantic v2 model
→ cross-field validation
→ resolved immutable experiment configuration
→ deterministic scientific hashes
→ execution
```

Do not duplicate scientific values across:

* YAML;
* constants;
* experiment registries;
* CLI defaults;
* tests;
* runner code.

Do not infer dataset or experiment identity from a filename.

Do not mutate validated models with unchecked update shortcuts when scientific invariants could be bypassed.

Construct variants through explicit validated factories/builders and revalidate the complete configuration.

Filesystem locations must not silently alter scientific identity.

---

## 13. EXPERIMENT REGISTRY IS THE RESEARCH REGISTRY

There must be one authoritative typed registry describing the experiment programme.

It must faithfully represent the roadmap's:

```text
S1–S6
R1–R14
```

without turning those codes into application architecture.

The registry owns publication identity and locked experiment definition.

Runner modules must not independently recreate the same experiment grids.

Pay particular attention to coupled parameter cells such as S2. Do not accidentally replace a locked coupled design with a Cartesian product.

Experiment dependencies must be explicit but simple. Again: **no DAG abstraction**.

---

## 14. SCIENTIFIC CONTRACTS MUST NOT DRIFT

Never change scientific behavior merely to simplify code or satisfy an existing test.

Protect exactly:

* strict comparison direction;
* quantile conventions;
* finite-sample rank selection;
* readiness probability;
* exact Clopper–Pearson mismatch evidence;
* dynamic minimum evidence sizes;
* client eligibility;
* R/G/C role disjointness;
* attack-label firewall;
* deterministic score reuse;
* frozen detector semantics;
* information regimes for each policy;
* federation coupling;
* metric definitions;
* undefined-metric behavior;
* statistical unit of analysis;
* artifact provenance;
* reproducibility contracts.

If implementation and tests disagree with the roadmap, correct the implementation and the tests.

Do not weaken the roadmap to preserve stale code.

If a roadmap passage contains a stale **implementation name** but its scientific meaning is clear, use the responsibility-based implementation and surgically synchronize the implementation-facing documentation without changing the locked scientific protocol.

---

## 15. DATA AND LEAKAGE BOUNDARIES

Inspect the actual available raw-data layout rather than assuming filenames or schemas.

Verify the real:

* N-BaIoT files;
* DIAD files;
* schemas;
* features;
* client identities;
* source ordering;
* benign/attack labels;
* row provenance.

Do not load massive datasets merely for inspection when headers/schema/sample rows suffice.

The implementation must structurally prevent information leakage.

In particular:

* training rows cannot leak into held-out roles;
* threshold policies cannot access forbidden attack evidence;
* final-test labels remain unavailable to non-oracle policies;
* score caches retain immutable physical identity;
* calibration seeds must not cause detector retraining/rescoring when the roadmap requires frozen scores;
* preprocessing must fit only on the permitted training evidence.

Leakage protection should be encoded through typed APIs and contracts, not merely comments.

---

## 16. ARTIFACTS AND REPRODUCIBILITY

Artifacts must be deterministic, typed, traceable, and semantically verifiable.

Verify:

```text
dataset
→ preprocessing
→ training
→ score cache
→ calibration view
→ policy decision
→ metrics
→ analysis/report
```

Hashes alone are insufficient.

The verifier must also confirm that artifacts belong to the correct:

* dataset;
* client;
* split;
* model seed;
* calibration seed where applicable;
* experiment;
* policy;
* scientific configuration;
* upstream artifact lineage.

Completed immutable artifacts must not be silently overwritten.

Do not make domain/scientific modules filesystem-aware merely for convenience.

---

## 17. COMMENTS AND DOCSTRINGS

Do not generate verbose AI-style commentary throughout the repository.

Remove:

* obvious comments;
* narration of what the next line does;
* historical migration notes;
* temporary explanations;
* stale TODO prose;
* comments describing removed architecture.

Use comments/docstrings only where they explain something genuinely non-obvious, especially:

* scientific invariants;
* leakage boundaries;
* numerical conventions;
* determinism;
* reproducibility;
* counter-intuitive implementation decisions;
* strict inequality/tie behavior.

Names and types should explain ordinary code.

---

## 18. TESTING AND STATIC ANALYSIS

Create, adapt, merge, or remove tests while implementing.

Tests must cover:

* scientific formulas;
* regression cases;
* architecture boundaries;
* information firewalls;
* deterministic identity;
* score-cache behavior;
* configuration validation;
* experiment registry;
* CLI wiring;
* artifact verification;
* end-to-end smoke workflows.

Do **not** run the entire test/static-analysis matrix after every small edit or commit.

Work in substantial coherent chunks.

During a chunk, run only focused tests when useful.

After a major implementation phase, run the appropriate broader checks, fix the discovered issues, and continue.

At major milestones and finally run:

```bash
python -m compileall src tests
ruff check src tests
ruff format --check src tests
pyright
pytest -n auto
git diff --check
```

If `pytest-xdist` is unavailable, use the existing supported equivalent rather than inventing another test framework.

Pyright is the CLI type-checking authority.

Keep Pylance strict configuration aligned with Pyright so the repository is also clean in VS Code.

Target:

```text
0 Pyright errors
0 Ruff errors
all tests passing
clean formatting
```

Do not obtain a clean result by weakening diagnostics.

---

## 19. PERFORMANCE AND REUSE

Search before creating new abstractions.

Prefer reuse and consolidation.

Remove unnecessary repeated:

* file scans;
* dataset loading;
* dataframe conversion;
* hashing;
* scoring;
* model loading;
* serialization;
* experiment resolution.

Do not add caching that weakens provenance.

Do not introduce clever abstraction for trivial operations.

Optimize the implementation where there is a real cost, especially dataset processing, score reuse, repeated experiment evaluation, and test execution.

Use GPU where the configured detector workflow requires it, but do not redesign scientific behavior around hardware shortcuts.

---

## 20. KEEP THE REPOSITORY CLEAN

By completion, remove:

* dead modules;
* orphaned imports;
* stale configuration;
* unused tests;
* duplicate source trees;
* obsolete scripts;
* temporary migration files;
* compatibility layers;
* paper-code implementation modules;
* accidental generated outputs;
* unneeded dependencies.

Research execution must live in the package and be reachable through the CLI.

Do not let `scripts/` become a second application layer.

Developer-only utilities may exist under `tools/` when genuinely appropriate.

---

## 21. AUDIT IN LARGE PASSES, NOT ENDLESS MICRO-AUDITS

Do not spend days repeatedly auditing the same matrix row by row.

Use a bounded implementation loop:

### Pass 1 — Structural audit

Identify:

* duplicates;
* stale naming;
* primitive/domain leaks;
* DAG/graph orchestration;
* dead architecture;
* incomplete CLI wiring;
* multiple execution spines.

Fix the substantial findings.

Commit.

### Pass 2 — Scientific wiring audit

Trace representative roadmap workflows end-to-end:

```text
config
→ data
→ training
→ scoring
→ policy
→ metrics
→ artifacts
→ analysis/report
```

Trace synthetic, primary N-BaIoT, DIAD, Deep-SVDD, sensitivity, and benchmark paths.

Fix missing or incorrect wiring.

Commit.

### Pass 3 — Quality audit

Run static analysis and the test suite after the large implementation work.

Fix all legitimate findings.

Commit.

### Pass 4 — Final roadmap reconciliation

Compare the final implementation against the roadmap and the implementation ledger once more.

Classify anything remaining as either:

```text
implementation defect
```

or:

```text
future experimental evidence
```

There should be no unresolved implementation defects.

Do not repeatedly restart the audit from zero after every minor edit.

---

## 22. AUTONOMY

Do not stop to ask for routine implementation decisions.

Inspect the repository and choose the strongest maintainable solution consistent with the roadmap.

Do not merely produce an audit report when you can fix the issue.

Continue implementing until the repository has no known implementation gap.

Only stop for a genuine external blocker such as unavailable required raw data, invalid repository authentication, or an inaccessible external dependency that cannot be substituted without scientific drift.

---

## 23. COMMIT STRATEGY

The PR merge happens first.

After that, work on `main` and commit after **large coherent changes**, for example:

* architecture/naming/DAG cleanup;
* CLI completion;
* data or scientific-contract completion;
* experiment/pipeline consolidation;
* artifact/verification completion;
* final quality fixes.

Use clear commit messages.

Do not make dozens of tiny mechanical commits.

Do not leave a large completed phase uncommitted.

---

## 24. FINAL COMPLETION CONDITIONS

The coding task is complete only when all of the following are true:

* PR #1 is merged into `main`;
* one source package remains under `src/fedcrg`;
* no backward-compatibility architecture remains;
* no DAG/workflow-graph architecture remains;
* responsibility-based implementation naming is consistent;
* paper experiment codes are restricted to research identity/registry/artifact contexts;
* duplicate R14/feature-sensitivity implementation is consolidated;
* other overlapping application/artifact modules have been audited and consolidated where necessary;
* one execution spine exists;
* every supported CLI command is implemented and wired;
* no CLI command is a stub;
* experiment registry is authoritative;
* all S1–S6 and R1–R14 execution paths are implemented;
* configuration is fully validated;
* information firewalls are structural;
* score-cache identity and reuse follow the roadmap;
* scientific formulas and metrics match the roadmap;
* artifact verification is semantic as well as cryptographic;
* no unjustified primitive-domain APIs remain;
* no domain dictionaries remain;
* no unjustified `Any` or broad type suppression remains;
* obsolete/dead code is removed;
* Pyright is clean;
* Pylance configuration is strict and aligned;
* Ruff is clean;
* formatting is clean;
* tests pass;
* CLI integration tests pass;
* documentation reflects the actual final architecture;
* the implementation ledger truthfully distinguishes implemented code from evidence still requiring real experiments.

Do not require full publication-scale experiments merely to declare the **software implementation** complete.

Do not falsify experimental completion.

---

## 25. FINAL VERIFICATION

From `main`, perform the final implementation-quality check:

```bash
git status --short --branch
python -m compileall src tests
ruff check src tests
ruff format --check src tests
pyright
pytest -n auto
git diff --check
```

Run lightweight CLI smoke checks covering the real command tree.

Exercise `fedcrg verify`.

If publication evidence is still genuinely missing, its failure/incomplete result is acceptable **only if it correctly identifies the missing evidence rather than an implementation defect**.

Any implementation error surfaced by verification must be fixed.

Commit the final fixes.

Leave `main` clean.

---

## 26. FINAL RESPONSE

When finished, report concisely:

1. the PR merge result;
2. major architecture consolidations;
3. DAG/graph logic removed;
4. stale paper/canonical implementation names removed;
5. CLI commands completed and wired;
6. major roadmap implementation gaps fixed;
7. test/Pyright/Ruff results;
8. important commits created;
9. anything remaining that is **experimental evidence only**, not an implementation defect.

Do not claim experiments were completed if they were not actually executed.

The final repository should be a coherent, typed, deterministic, reproducible research implementation of the FedCRG roadmap — not a collection of partially connected modules that merely pass tests.
