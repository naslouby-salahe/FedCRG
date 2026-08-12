# Goal

Fully implement, validate, audit, execute, and complete the FedCRG project at:

```text
/home/naslouby/Projects/FedCRG
```

The authoritative scientific specification is:

```text
/home/naslouby/Projects/FedCRG/docs/FedCRG Roadmap.md
```

Work continuously toward **complete roadmap satisfaction**. This is an idempotent goal: every invocation must inspect the current repository state, preserve already-correct work, resume incomplete work, repair regressions, and continue until the roadmap is fully implemented and verified.

Do not stop after planning, auditing, scaffolding, partial implementation, a test pass, a commit, or identifying remaining work.

Do not ask me questions. Resolve ambiguity using the rules below, record the decision, and continue.

---

# 1. Absolute source of truth

The **first action** on every invocation is to read `docs/FedCRG Roadmap.md` completely.

Do not rely on memory, summaries, previous agent output, comments, existing code, tests, README files, or assumptions in place of the roadmap.

The roadmap is the scientific source of truth.

Respect its own internal precedence rules exactly, including its precedence among:

1. formulas and state-transition rules;
2. dataset-role definitions;
3. baseline definitions;
4. experiment registry;
5. generated configuration.

Never modify a LOCKED scientific value because implementation would be easier another way.

Never tune, reinterpret, weaken, silently skip, or replace a roadmap requirement after observing results.

If existing code disagrees with the roadmap, the code changes.

If a generic engineering preference below conflicts with an explicit scientific roadmap requirement, the roadmap wins.

Do not rewrite the roadmap unless implementation genuinely requires a separately requested protocol amendment. Treat it as normative input, not a working TODO document.

---

# 2. Build or validate the roadmap audit/implementation matrix

The **second action**, after reading the roadmap, is to inspect whether this exists:

```text
docs/FedCRG Audit and Implementation Matrix.md
```

If it does not exist, create it.

If it already exists, audit and update it rather than recreating or duplicating it.

The matrix must be a **lossless operational extraction of the roadmap before it becomes a repository-status report**.

Do not infer scientific requirements from the current implementation.

Every material roadmap requirement must receive a stable descriptive ID belonging to an appropriate family such as:

```text
GLOBAL-*
FORMULA-*
INVARIANT-*
DATASET-*
CLIENT-*
SPLIT-*
PREPROCESS-*
TRAIN-*
CHECKPOINT-*
SCORE-*
GATE-A-*
GATE-B-*
STATE-*
POLICY-*
BASELINE-*
METRIC-*
STAT-*
ROBUSTNESS-*
EXPERIMENT-*
CONFIG-*
ARTIFACT-*
PROVENANCE-*
CLI-*
TABLE-*
FIGURE-*
REPORT-*
CLAIM-*
FAILURE-*
NEGATIVE-*
TEST-*
BOUNDARY-*
```

Do not use opaque identifiers when a descriptive identifier is possible.

Each matrix row must be precise enough that another engineer could implement or audit it without guessing. Include, where applicable:

* stable requirement ID;
* exact roadmap section/reference;
* normative requirement;
* status such as LOCKED, DERIVED, DATA-DEPENDENT, EXPLORATORY, STOP, or implementation-only;
* formulas and numerical values;
* allowed and forbidden behavior;
* dataset/client/split applicability;
* experiment applicability;
* configuration ownership;
* expected implementation owner/module;
* expected runtime caller/execution path;
* inputs;
* outputs;
* required artifacts and provenance;
* metrics/statistics;
* edge-case behavior;
* failure state/error behavior;
* required positive tests;
* required negative/invariant/leakage tests;
* expected CLI exposure;
* report/table/figure/claim dependencies;
* current repository status;
* remediation required;
* completion evidence.

A section heading alone never counts as coverage.

A test existing alone never counts as implementation coverage.

A class existing but being unreachable from the real execution path never counts as implemented.

---

# 3. Audit the matrix before using it

Before large-scale repository implementation, perform **four distinct matrix audits**.

## Audit 1 — Lossless roadmap coverage

Walk the roadmap from beginning to end and prove that every:

* normative MUST/MUST NOT/SHALL/SHOULD contract;
* formula;
* numerical constant/grid;
* algorithm;
* state transition;
* eligibility rule;
* dataset role;
* baseline;
* policy;
* experiment;
* metric;
* statistical procedure;
* sensitivity;
* stress test;
* CLI requirement;
* artifact;
* manifest;
* table;
* figure;
* claim gate;
* prohibited claim;
* failure state;
* completion-checklist item

has an explicit matrix representation.

Fix omissions immediately.

## Audit 2 — Scientific-contract consistency

Independently audit the matrix for:

* formula fidelity;
* Gate-A mathematics;
* Gate-B exact-binomial semantics;
* threshold inequality semantics;
* independence/disjointness requirements;
* data-role leakage;
* calibration/test separation;
* detector freezing;
* score-cache invariance;
* policy counts and identities;
* statistical unit of analysis;
* paired/repeated-split semantics;
* global-threshold coupling;
* multiplicity handling;
* undefined-metric behavior;
* deterministic tie handling;
* all STOP/error states.

Fix every discrepancy before continuing.

## Audit 3 — Experimental and evidence completeness

Independently reconcile:

* complete synthetic registry;
* complete primary experiment registry;
* sensitivity experiments;
* robustness/assumption-stress experiments;
* external replication;
* second-detector analysis;
* all policies and comparators;
* required artifact hashes/manifests;
* tables;
* figures;
* reports;
* claim-strength gates;
* prohibited claims;
* reproducibility requirements.

No experiment or evidence output may disappear simply because it is inconvenient to implement.

## Audit 4 — Implementability and verification

Verify that every matrix requirement has a concrete path to:

```text
requirement
→ configuration/domain representation
→ implementation owner
→ runtime caller
→ artifact/result
→ test
→ verification evidence
```

Also reverse-audit the matrix:

```text
matrix → roadmap
```

Anything in the matrix that is not authorized by the roadmap must be removed or clearly classified as engineering-only support that cannot alter scientific behavior.

Do not begin major implementation until these four audits have converged without unresolved matrix omissions.

---

# 4. Crash-safe progress state

After the matrix is established, create and maintain:

```text
docs/.tmp/
```

This is the local durable working area for resuming after interruption.

At minimum maintain:

```text
docs/.tmp/CURRENT_STATE.md
docs/.tmp/PROGRESS.md
docs/.tmp/AUDIT_LOG.md
docs/.tmp/VALIDATION_LOG.md
docs/.tmp/BLOCKERS.md
docs/.tmp/NEXT_ACTION.md
```

Keep these concise and continuously current.

On every invocation:

1. read the roadmap;
2. read the matrix;
3. read `docs/.tmp/CURRENT_STATE.md`;
4. inspect Git history/status and the live repository;
5. verify that previously completed work is still valid;
6. resume from the highest-priority incomplete requirement.

Update the tracking files after every substantial implementation chunk, audit, validation run, and commit.

The tracking directory is not a replacement for Git commits or the audit matrix.

Do not create duplicate trackers on subsequent runs.

---

# 5. Raw-data entrypoint

The project must use:

```text
/home/naslouby/Projects/FedCRG/data/raw
```

as its canonical project-facing raw-data location.

The actual shared raw data resides at the external filesystem target I provided.

Inspect `data/raw` before changing it.

Rules:

* if the correct symlink already exists, do nothing;
* if the symlink is broken or points elsewhere, correct it;
* if `data/raw` is absent, create the parent directory as needed and create the symlink;
* if `data/raw` is only an empty/placeholder directory, safely replace the placeholder with the symlink;
* if it contains real non-placeholder data, do not destroy user data; inspect and resolve safely;
* never copy the raw datasets into the repository merely to avoid the symlink;
* never modify or delete the shared raw-data target;
* never commit raw data.

The external source pathname is filesystem infrastructure only. **Do not persist its project-name component in FedCRG source code or the audit matrix.** Code/configuration must see the project-local `data/raw` entrypoint.

---

# 6. Naming isolation

FedCRG must remain an independent project identity.

Outside the literal external filesystem path needed to create or verify the raw-data symlink:

* do not introduce the string `datp` into source code;
* do not introduce it into the audit/implementation matrix;
* do not use it in Python package names;
* do not use it in symbols, classes, enums, constants, test names, fixtures, artifact names, or implementation comments;
* do not borrow old scientific terminology from another project.

Use only terminology authorized by the FedCRG roadmap.

The Python namespace is `fedcrg`.

Remember the roadmap's distinction: **FedCRG names the post-training personalization-admission layer, not the trained detector itself.**

---

# 7. Live repository audit

Once the roadmap-derived matrix is complete, audit the live repository against it.

Inventory:

* packages;
* modules;
* classes;
* enums;
* dataclasses;
* Pydantic configuration models;
* functions;
* constants;
* configuration files;
* CLI commands;
* pipeline/execution stages;
* dataset readers;
* preprocessing;
* training;
* scoring;
* score caches;
* Gate A;
* Gate B;
* state transitions;
* threshold policies;
* comparators;
* metrics;
* statistics;
* robustness experiments;
* reports;
* artifact serializers;
* manifests;
* tests;
* fixtures;
* obsolete code;
* duplicated implementations;
* dead execution paths.

For every matrix row classify the repository state as something equivalent to:

```text
MISSING
PARTIAL
INCORRECT
PRESENT_UNVERIFIED
PRESENT_VERIFIED
STALE
UNAUTHORIZED
BLOCKED_BY_ROADMAP_STOP
```

Then perform the reverse mapping:

```text
repository implementation → authorizing roadmap requirement
```

Material scientific behavior without roadmap authorization must be removed.

Required behavior reachable only from tests but not the actual execution spine is incomplete.

---

# 8. Implementation strategy

Implement by dependency and scientific risk rather than by whichever file is easiest.

Prefer this broad progression unless the roadmap establishes a stricter dependency:

1. typed domain identities, enums, value objects and protocol contracts;
2. configuration parsing/validation and locked configuration files;
3. dataset discovery, integrity and deterministic preparation;
4. role assignment and leakage prevention;
5. preprocessing;
6. detector training;
7. immutable/deterministic scoring and score caches;
8. Gate-A precomputation and exact implementation;
9. Gate-B implementation;
10. FedCRG state machine and admission decision;
11. all roadmap policies and baselines;
12. metrics and edge-case semantics;
13. statistical analysis;
14. synthetic experiments;
15. primary experiment registry;
16. sensitivity experiments;
17. robustness and assumption-stress experiments;
18. external-dataset replication;
19. second-detector robustness check;
20. artifact/provenance/manifest system;
21. CLI;
22. reporting, figures and tables;
23. claim-strength verification;
24. final reproducibility and completeness verification.

Do not implement empty interfaces and call a phase complete.

Do not stop at architecture.

Do not stop at unit tests.

Required production behavior must be reachable through the intended FedCRG execution path.

---

# 9. No backwards compatibility

There is **no backwards compatibility requirement**.

Do not add or retain:

* compatibility modules;
* deprecated aliases;
* transitional imports;
* import redirects;
* forwarding modules;
* shims;
* wrappers whose only job is preserving an old API;
* duplicate legacy and new implementations;
* old names pointing to new names;
* fallback imports;
* compatibility configuration keys;
* versioned parallel APIs;
* dead migration paths.

When a design is superseded:

1. migrate every caller;
2. migrate or replace the tests;
3. delete the old implementation.

A migration is not complete while the obsolete execution path still works accidentally.

---

# 10. Type and domain-model discipline

Avoid primitive obsession.

Closed scientific vocabularies, method identities, states, failure codes, dataset IDs, policy IDs, experiment IDs, evidence roles, split roles, artifact kinds, and similar concepts must use enums or appropriately typed domain objects rather than arbitrary strings.

Use frozen/slot dataclasses where appropriate for immutable domain records.

Use Pydantic where the roadmap requires validated configuration/boundary parsing.

Do not use anonymous dictionaries as domain objects or as the normal internal API between components.

Configuration may enter through YAML/mapping parsing at the boundary, but convert it immediately to validated typed models.

Do not leak raw primitives across public/domain interfaces when a meaningful value type or enum exists.

Avoid:

* `Any`;
* untyped dictionaries;
* arbitrary `object`;
* stringly typed states;
* magic strings;
* magic numbers;
* boolean parameters whose meaning is unclear;
* positional tuples with undocumented semantics.

If a third-party API requires a primitive container, isolate that representation at the integration boundary and convert immediately to/from typed project structures.

Use explicit result types for legitimate unavailable/undefined outcomes where the roadmap defines them. Never silently coerce an undefined metric to zero.

---

# 11. Reuse, refactoring, and ownership

Before creating a new class, function, module, serializer, validation rule, metric, or utility:

1. search the repository;
2. inspect callers;
3. inspect tests;
4. determine whether existing behavior can be generalized or moved to the correct owner.

Prefer reuse and consolidation over duplication.

Actively:

* remove duplicate logic;
* merge equivalent abstractions;
* delete dead code;
* simplify over-engineered layers;
* improve algorithms and hot paths where semantics remain identical;
* make module ownership obvious;
* keep the CLI thin;
* keep scientific logic out of reporting/UI layers;
* keep reports from redefining metrics;
* keep configuration from duplicating scientific calculations that belong in typed domain logic.

Do not create speculative abstractions for hypothetical future requirements.

Do not create thin wrappers with no real responsibility.

---

# 12. Scientific implementation rules

Never improvise around the protocol.

In particular preserve all roadmap-defined:

* data-role disjointness;
* benign-only FedCRG admission inputs;
* attack-label restrictions;
* frozen-detector semantics;
* exact order-statistic conventions;
* Gate-A tie behavior;
* exact Clopper-Pearson Gate-B semantics;
* strict `score > threshold` decision rule;
* threshold-equality-benign rule;
* policy identities;
* policy counts;
* comparator information budgets;
* score-cache reuse;
* score/hash invariance;
* model/calibration seed semantics;
* experiment unit of analysis;
* bootstrap recomputation requirements for federation-coupled thresholds;
* robustness stress definitions;
* invalid/STOP states;
* claim limitations.

Do not silently rescue a failed scientific run by:

* changing hyperparameters;
* altering a seed;
* changing a sample count;
* adjusting a threshold;
* relaxing eligibility;
* dropping a client;
* changing a policy;
* switching an undefined metric to zero;
* modifying an analysis after seeing outcomes.

Fix implementation/data-integrity defects while keeping the locked protocol unchanged.

---

# 13. Tests must evolve with the implementation

For every substantial production change, add/adapt/remove the corresponding tests during the same implementation chunk.

Tests must cover, where relevant:

* unit behavior;
* exact numerical/formula parity;
* property/invariant behavior;
* configuration validation;
* negative cases;
* leakage prevention;
* disjointness;
* deterministic behavior;
* tie handling;
* metric undefined cases;
* failure-state transitions;
* serialization round trips;
* artifact hashes;
* score-cache immutability;
* threshold-policy score invariance;
* AUROC/AUPRC invariance across threshold policies;
* CLI contracts;
* experiment enumeration;
* required artifact completeness;
* report reconstruction.

Do not preserve obsolete tests merely for compatibility.

Do not weaken a valid test to make incorrect code pass.

---

# 14. Validation cadence: batch it

Do **not** run pytest, Ruff, formatting, Pyright/Pylance checks after every tiny edit or every small commit.

Create/adapt tests continuously, but execute expensive validation in batches after a **large coherent implementation chunk** or phase.

Typical large-chunk gate:

1. format the affected code;
2. run Ruff lint/checks;
3. run strict Pyright/Pylance-compatible type checking;
4. run the relevant targeted tests;
5. run related integration/contract tests;
6. fix all failures before moving materially forward.

Use `pytest-xdist` / parallel test execution when tests are safe to parallelize.

Do not parallelize tests or scientific jobs when doing so can violate deterministic state, shared artifacts, GPU ownership, or filesystem isolation.

At major phase boundaries, widen validation.

Near final completion, run the complete project validation suite.

Do not repeatedly spend minutes running the same full suite after trivial modifications.

---

# 15. Experiment execution order

Do not launch expensive confirmatory campaigns while foundational implementation is still unstable.

Use this progression:

## A. Static and contract readiness

Make the roadmap implementation structurally complete enough for meaningful validation.

Run configuration, unit, property, negative, leakage, serialization, deterministic-enumeration and formula-parity tests.

## B. Synthetic protocol validation

Run the roadmap-required synthetic programme and Gate-A precomputations.

Synthetic theoretical parity failures are implementation/audit failures, not permission to alter the theorem.

## C. Real-data preparation and integrity

Prepare the required datasets through the canonical `data/raw` project entrypoint.

Validate:

* natural-client identities;
* exact feature expectations;
* row counts;
* eligibility;
* role counts;
* disjointness;
* attack-development separation;
* manifests;
* deterministic preparation;
* source-order/chronology claims only where actually justified.

## D. Controlled scientific smoke

Before a full campaign, run the smallest roadmap-compatible end-to-end scientific path needed to establish:

```text
data
→ preparation
→ training
→ scoring
→ calibration roles
→ policies
→ metrics
→ artifacts
→ reporting
```

This is an implementation smoke only and cannot establish scientific claims.

## E. Full required execution

Only after implementation, integrity, synthetic, and smoke gates are valid, execute the roadmap-required:

* primary N-BaIoT programme;
* sensitivities;
* robustness/assumption-stress programme;
* external CIC IoT-DIAD replication;
* second-detector check;
* benchmarks;
* required report generation.

Do not invent reduced grids to save time.

Do not silently omit weak or negative results.

## F. Final report and verification

Build the required reports/tables/figures.

Run the complete verification contract, including the roadmap-prescribed:

```text
fedcrg verify
```

A missing experiment cell, hash, manifest field, leakage check, test requirement, or required artifact is a failure to finish, not a reason to declare completion.

---

# 16. Configuration discipline and anti-HARKing

Use the roadmap-prescribed typed configuration and YAML files.

Scientific constants belong in the locked configuration/domain contract, not scattered across code.

Do not expose confirmatory values as ad hoc CLI knobs when the roadmap says they come from configuration.

Once a confirmatory configuration/result-inspection boundary is crossed:

* do not mutate the locked scientific config based on observed results;
* do not silently change the policy registry;
* do not change statistical rules;
* do not change claim gates;
* do not cherry-pick seeds or clients;
* do not suppress valid negative outcomes.

A scientific negative result is not an implementation failure.

A roadmap-defined integrity failure is.

Keep those categories separate.

---

# 17. Failure handling without asking me questions

Do not stop to ask me what to do.

When uncertainty appears:

1. consult the exact roadmap section;
2. apply the roadmap's internal precedence hierarchy;
3. inspect related config and neighboring requirements;
4. inspect the live implementation and tests;
5. choose the narrowest interpretation that preserves all locked scientific invariants;
6. record the interpretation in `docs/.tmp/AUDIT_LOG.md`;
7. continue.

For genuine roadmap-defined STOP conditions:

* do not fabricate data or scientific semantics;
* record the exact condition and evidence in `docs/.tmp/BLOCKERS.md`;
* mark only the affected matrix items blocked;
* continue every independent unblocked task;
* return to the blocker when its prerequisite becomes available.

A blocker in one branch is not permission to abandon the entire goal.

---

# 18. Git discipline

Inspect Git status and recent history before editing.

Preserve valid existing work.

Do not wipe unrelated user modifications.

Do not use destructive history rewriting to simplify the task.

After every **major coherent change or completed phase**, create a commit.

Examples of commit boundaries:

* audit matrix completed and four audits passed;
* foundational domain/config architecture complete;
* dataset/preprocessing layer complete;
* training/scoring layer complete;
* Gate A/B + FedCRG decision layer complete;
* comparator/policy system complete;
* metric/statistics system complete;
* experiment/robustness programme complete;
* reporting/provenance/CLI complete;
* final verification/remediation complete.

A commit should represent a meaningful working checkpoint, not every tiny file edit.

Before each major commit, run the validation appropriate to that chunk.

Use clear descriptive commit messages.

Do not commit raw datasets, temporary scientific caches that should remain ignored, or accidental generated junk.

---

# 19. Code-quality rules

Keep the implementation professional and maintainable.

Required principles:

* no backwards compatibility;
* no dead code;
* no duplicate implementations;
* no unused abstractions;
* no unexplained magic values;
* no primitive/stringly typed domain identities;
* no `Any`;
* no anonymous dict-based domain contracts;
* no hidden fallback behavior;
* no silent exception swallowing;
* no misleading names;
* no fake generality;
* no circular scientific ownership;
* no test-only production implementation;
* no weird AI-generated comments;
* no comments that merely narrate obvious code;
* no TODO placeholders standing in for roadmap requirements;
* no premature “done” markers.

Optimize and refactor continuously where it improves clarity, correctness, runtime, memory use, or maintainability without changing scientific semantics.

---

# 20. Continuous audit loop

After every major implementation phase:

1. update the matrix;
2. rerun the relevant roadmap → matrix coverage check;
3. audit the new code against the matrix;
4. perform repository → roadmap reverse mapping for changed scientific code;
5. check runtime reachability;
6. check for dead or duplicate implementations;
7. check for primitive/domain leaks;
8. check for stale names or compatibility paths;
9. update tests;
10. run the batched validation gate;
11. update `docs/.tmp/`;
12. commit the coherent phase.

If the audit exposes an earlier architectural mistake, fix it immediately rather than building more code on top of it.

---

# 21. Final hostile audits

Before declaring completion, perform separate hostile audits for:

## Scientific fidelity

Try to find any deviation from formulas, state transitions, data roles, seeds, policy semantics, experiment grids, metric definitions, statistics, stress conditions, or claim constraints.

## Leakage and provenance

Try to prove that training/calibration/gate/test/attack-development roles leak into one another or that an artifact can be consumed with incompatible lineage.

Any successful attack must be fixed.

## Runtime completeness

Prove every required implementation is reachable from the intended CLI/execution path and not merely present in a module or test.

## Experiment completeness

Compare the roadmap experiment registry with materialized/available experiment coordinates and identify missing or unauthorized cells.

## Artifact completeness

Verify manifests, hashes, score caches, checkpoints, metrics, tables, figures, reports and required provenance.

## Engineering quality

Search for duplicate code, obsolete code, wrappers, aliases, dead modules, primitive obsession, dictionaries used as domain models, `Any`, magic constants, circular ownership, and unnecessary complexity.

## Idempotence

Run the goal mentally/operationally one more time from its beginning.

A completed repository should cause the process to verify existing work and make **no substantive implementation changes**.

---

# 22. Definition of complete

Do not declare the goal complete until all of the following are true:

* the roadmap has been fully read and remains the authority;
* the audit/implementation matrix exists;
* all four independent matrix audits pass;
* every roadmap requirement is represented;
* every actionable matrix requirement is implemented;
* every implementation maps back to legitimate roadmap authority;
* all required production code is runtime-reachable;
* no unauthorized scientific behavior remains;
* no compatibility layer remains;
* no stale implementation remains;
* no meaningful duplication remains;
* typed domain/configuration rules are satisfied;
* required tests exist;
* all final Ruff/format/type checks pass;
* the complete applicable pytest suite passes;
* all scientific invariants and leakage checks pass;
* required synthetic experiments pass;
* required datasets are prepared and validated;
* required scientific experiment cells are complete;
* required score/artifact/provenance checks pass;
* all required tables/figures/reports are generated;
* the roadmap's claim/integrity gates are evaluated correctly;
* `fedcrg verify` passes;
* final hostile audits find no unresolved issue;
* the audit matrix records completion evidence;
* `docs/.tmp/CURRENT_STATE.md` records the final verified state;
* major phases have corresponding Git commits;
* an idempotent rerun would produce no substantive changes.

Until then, continue working.

**Do not ask for confirmation. Do not ask what to do next. Do not stop because the task is large. Continue from the highest-priority incomplete or invalid requirement until the complete FedCRG roadmap is faithfully implemented, exercised, audited, and verified.**
