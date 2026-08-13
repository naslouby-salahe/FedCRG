# FedCRG Audit Matrix

Fresh audit matrix rebuilt from the v2.0 protocol (`docs/roadmap.md`), the current working tree, and the target architecture in the implementation goal. Statuses are re-derived by hostile re-audit; a VERIFIED entry is not trusted until the repository itself demonstrates it.

Legend: `OPEN` = not yet addressed · `IN_PROGRESS` = partially addressed · `VERIFIED` = repository evidence confirms · `N/A` = not applicable to the current milestone.

## A. Repository Architecture

| ID | Requirement (goal section) | Status | Evidence |
|----|----------------------------|--------|----------|
| A1 | Target tree: `config/`, `docs/`, `data/`, `src/fedcrg/`, `tests/{contract,integration,unit}`, `outputs/`, `results/` (§4) | OPEN | `configs/` hierarchy still present; `src/fedcrg` has 14 packages |
| A2 | No additional package levels without justification (§4) | OPEN | `decision/policies/`, `experiments/definitions/`, `configuration/` sub-packages exist |
| A3 | No one-file packages; no god modules (§4) | OPEN | `experiments/definitions/sensitivity.py` is 808 lines |
| A4 | No redirect/import-only modules, stateless factories/managers/handlers, wrapper classes (§5) | OPEN | to be audited |
| A5 | No `canonical` terminology in production code; hygiene test (§6) | OPEN | need AST test |
| A6 | No production references to roadmap/matrix/prompt/migration/legacy (§3) | OPEN | need AST test |
| A7 | Vague names (`utils`, `helpers`, `common`, `manager`, `handler`, `processor`, `engine`, `service`, `base`, `registry`, `factory`) absent (§5, §36) | OPEN | need audit |
| A8 | Naming audit: descriptive names, no `x`/`n`/`data`/`value`/`result` in structured logic (§36) | OPEN | need audit |

## B. Configuration

| ID | Requirement | Status | Evidence |
|----|-------------|--------|----------|
| B1 | One source of truth: `config/study.yaml`, `config/datasets.yaml`, `config/experiments.yaml` (§10) | OPEN | `configs/` tree with extends/merge |
| B2 | No `extends` chains, no inheritance graph, no Python deep-merge (§10) | OPEN | `configuration/resolve.py` implements `_deep_merge` and `extends` |
| B3 | `study.yaml` owns protocol/statistical/randomness/training values (§10) | OPEN | split across `configs/` |
| B4 | `datasets.yaml` owns dataset contracts (§10) | OPEN | `configs/datasets/*.yaml` |
| B5 | `experiments.yaml` owns the experiment catalogue (id, category, dataset/detector, policies, axes, coupled cells, repetitions, dependencies, required evidence, workload, confirmatory/diagnostic) (§10) | OPEN | catalogue in `experiments/definitions/*.py` |
| B6 | Python executes typed `ExperimentSpec`; no separate Python redefinition (§10) | OPEN | `experiment_definition.py` duplicates catalogue |
| B7 | Config-drift tests: configured scientific values not duplicated as source literals (§12) | OPEN | need test |
| B8 | No hidden scientific defaults (Pydantic/dataclass/function/CLI defaults, `.get` fallbacks, `or` fallbacks) (§13) | OPEN | need audit |
| B9 | Config validation: alpha/gamma/rho/band/seed/count/policy checks (§14.9) | OPEN | `configuration/validate.py` exists |
| B10 | `config/` root path and no `configs/` residue (§4) | OPEN | `configs/` present |

## C. Type System and Primitive Leakage

| ID | Requirement | Status | Evidence |
|----|-------------|--------|----------|
| C1 | Strong typed identifiers: ClientId, RunId, CampaignId, seeds, Probability, Alpha, Fpr, Tpr, ConfidenceLevel, Assurance, counts, Threshold, Score, Fraction (§7) | OPEN | `domain/identifiers.py`, `domain/values.py` partial |
| C2 | Constrained aliases via `Annotated[..., Field(...)]` / `StringConstraints` (§7) | OPEN | need audit |
| C3 | AST primitive-leakage tests with documented boundary allowlist (§8) | OPEN | need test |
| C4 | No `Any`, `object`, `dict[str, object]`, `Mapping[str, Any]`, `-> object`, `-> Any` outside boundaries (§8, §29) | OPEN | need audit |
| C5 | Pydantic v2 owns structured boundaries; frozen models, `ConfigDict`, `TypeAdapter`, discriminated unions (§9) | OPEN | need audit |
| C6 | No handwritten JSON converters / manual enum decoding / nested dict reconstruction (§9) | OPEN | `artifacts/json_io.py` to review |
| C7 | Domain enums for closed identity/state sets (§7) | PARTIAL | `domain/enums.py` exists and is used |

## D. Scientific Core (Preserved Semantics)

| ID | Requirement | Status | Evidence |
|----|-------------|--------|----------|
| D1 | Reference threshold: equal-count pooled R, `q_ref=min(N_R,ceil((N_R+1)(1-alpha)))`, strict `>` (§5.1, §9.1) | VERIFIED | unit tests pass |
| D2 | Gate A: `P_r = I_b(n+1-r,r) - I_a(n+1-r,r)`, argmax with larger-r tie, READY iff max >= gamma_A; rank precomputation table with n/rank/coverage/ready/alpha/rho/a/b/gamma_A (§5.2) | VERIFIED | `calibration_readiness.py` + exact tests |
| D3 | Gate-A exact values: n=1416 r*=1404 P=0.9500045311; n=1500 r*=1487 P=0.9573928914; n=2000 r*=1982 P=0.9805279151; tolerance 1e-10 (§14.2) | VERIFIED | unit tests pass |
| D4 | Gate B: two-sided Clopper-Pearson, LOW iff U<a, HIGH iff L>b, else NO_MATERIAL_MISMATCH_DEMONSTRATED; p_low/p_high diagnostics (§5.3) | VERIFIED | `mismatch_detection.py` + exact tests |
| D5 | Gate-B cutoffs: 736/1000/1500/2000/3000 rows (§14.2, G.2) | VERIFIED | unit tests pass |
| D6 | Gate-B minimum `n_G_min(a,gamma_B)` derived, primary 736; ONE_SIDED_BAND_BY_DESIGN when a=0 (§5.4.1) | VERIFIED | tests pass |
| D7 | Five decision states + CALIBRATION_ASSUMPTION_VIOLATION tie rule (§5.4) | VERIFIED | `threshold_decision.py` + tests |
| D8 | Multiplicity: Bonferroni readiness/mismatch, Holm directional, GATE_B_DIRECTION_CONTRADICTION (§6.1) | VERIFIED | tests pass |
| D9 | Metrics: MEBE, HighExcess, BandViolationRate, MAFE, ABMacroTPR, MacroTPR, AUROC/AUPRC invariance 1e-12 (§10) | VERIFIED | unit tests pass |
| D10 | Baselines B0-B10 with exact rules (§9): quantile rank ledger, shrinkage n0 grid, dev-F1 selector, Laridi-style SS, SUP-F1-1000, oracle | VERIFIED | policy tests pass |
| D11 | Federated training: AE 115-86-57-38-29-38-57-86-115 (36,626 params), 30 rounds, 120/20 local epochs, batch 64, tanh, Xavier gain 5/3, cosine LR 1e-3->1e-5, optimizer reset, deterministic shuffle (§8) | VERIFIED | unit tests pass |
| D12 | Deep-SVDD: encoder 115-64-32 tanh no bias, center from seed-initialized encoder, frozen center, 30 rounds x 20 epochs (§8.4) | VERIFIED | tests pass |
| D13 | Synthetic kernels: 4 distributions, AR(1), shift, contamination, exact mismatch power (§16) | VERIFIED | tests pass |
| D14 | Score caches: float64, hash-finalized, immutable before policy evaluation; AUROC invariance across policies (§14.1, §14.3) | VERIFIED | cache tests pass |
| D15 | Failure-code registry (§14.7) | VERIFIED | enums + tests |
| D16 | Metric edge-case rules: NA not 0, strict `>`, equality benign (§14.8) | VERIFIED | tests pass |

## E. Data and Preprocessing

| ID | Requirement | Status | Evidence |
|----|-------------|--------|----------|
| E1 | Preprocessed root exactly `data/preprocessed/` with cache identity `data/preprocessed/<dataset-id>/<preprocessing-id>/` (§14) | PARTIAL | root exists; identity scheme to verify |
| E2 | Reuse-first: valid cache reused with explicit validation (manifest, files, hashes, schema, source identity, preprocessing identity) (§14) | OPEN | `datasets/preprocessing.py` to audit |
| E3 | Atomic finalization / locking (no interrupted-cache reuse) (§14) | OPEN | to audit |
| E4 | `fedcrg preprocess [DATASET_ID] [--overwrite]`; campaign uses identical capability (§15) | OPEN | current CLI is `data preprocess` |
| E5 | N-BaIoT adapter: 9 clients, 115 features, source-order holdout, role partition, attack dev/test, row_id integrity (§7.1) | VERIFIED | `datasets/nbaiot.py` + tests |
| E6 | DIAD adapter: 86-feature allowlist, client-local imputation, eligibility rule, water-filling attack dev, stable ordering (§7.2-7.3) | VERIFIED | `datasets/diad.py` + tests |
| E7 | Preprocessing reuse tests (10 scenarios, §32) | OPEN | need tests |

## F. Experiments and Execution

| ID | Requirement | Status | Evidence |
|----|-------------|--------|----------|
| F1 | One execution spine: `experiments/runner.py` + `experiments/analyses.py`; no parallel runners/planners/materializers (§21) | OPEN | 18 files in `experiments/` |
| F2 | Campaign orchestrates same runner; no campaign-only implementations (§21) | OPEN | to audit |
| F3 | `graphlib.TopologicalSorter` for dependency ordering (§21) | OPEN | `dependencies.py` to audit |
| F4 | Experiment catalogue complete: exactly one entry per ExperimentId (S1-S6, R1-R14) (§11) | VERIFIED | catalogue test passes |
| F5 | Workload ledger expectations reconcile (§11.2) | VERIFIED | ledger tests pass |
| F6 | Restart-safe campaign state (JSON, not Markdown) (§2, §18) | OPEN | `completion.py`/`campaign.py` to audit |
| F7 | Run evidence simple: `outputs/runs/<run-id>/{run.json,thresholds.parquet,metrics.parquet,verification.json}` (§17) | OPEN | current layout to audit |
| F8 | Model/score/analysis cache reuse validated by identity (§16) | OPEN | to audit |

## G. Outputs, Monitoring, Results

| ID | Requirement | Status | Evidence |
|----|-------------|--------|----------|
| G1 | Output roots: logs/, monitoring/, cache/{models,scores,analysis}, runs/, campaigns/, figures/, reports/ (§16) | PARTIAL | `outputs/cache/datasets`, `cache/precomputed`, `reports/latest`, `reports/publication` exist — mismatch |
| G2 | Structured logging persisted under outputs/logs; no scattered `print()` (§24) | PARTIAL | `runtime/logging.py` exists; scan needed |
| G3 | Resource telemetry persisted under outputs/monitoring; `fedcrg monitor` (§25) | VERIFIED | `runtime/monitoring.py` + CLI |
| G4 | Publication results bundle `results/<campaign-id>/` with manifest/summary/configuration/environment/provenance/checksums/statistics/tables/figures (§19) | OPEN | `reporting/publication.py` to audit |
| G5 | `fedcrg results build` and `verify`; campaign uses identical builder; verify detects tampering (§19) | OPEN | need CLI + tests |
| G6 | Results builder never retrains/rescores (§19, §33) | OPEN | need test |

## H. CLI, Tooling, Docs

| ID | Requirement | Status | Evidence |
|----|-------------|--------|----------|
| H1 | CLI surface: validate, preprocess, plan, run, campaign, status, monitor, report, results build/verify (§26) | OPEN | current surface is `data preprocess`, `experiment execute-grid`, `config validate`, etc. |
| H2 | CLI functions thin; scientific behavior in typed application code (§26) | OPEN | `cli/experiment_commands.py` is 531 lines |
| H3 | Makefile targets match final CLI (§27) | OPEN | Makefile references old CLI |
| H4 | noxfile sessions: format/lint/typecheck/unit/integration/contract/regression/audit/quality (§28) | PARTIAL | exists but references old paths |
| H5 | Ruff comprehensive rules; Pyright strict (§29) | OPEN | ruff select is minimal; pyright standard |
| H6 | README matches final architecture (§38) | OPEN | references old layout |
| H7 | Architecture contract tests for target tree (§30) | OPEN | need rewrite |
| H8 | Old-architecture tests removed (§30) | OPEN | `test_architecture_contract.py` asserts old layout |

## I. Quality Gates

| ID | Requirement | Status | Evidence |
|----|-------------|--------|----------|
| I1 | ruff format + check pass | VERIFIED | baseline run |
| I2 | pyright passes | PARTIAL | 4 errors in `experiment_runner.py` |
| I3 | pytest passes | VERIFIED | 148 tests |
| I4 | nox -s quality passes | OPEN | not yet run |
| I5 | CLI smoke: help/validate/plan/preprocess/status/monitor/results (§41) | OPEN | to run |
| I6 | Repeated hostile audits converge (§39, §46) | OPEN | ongoing |
