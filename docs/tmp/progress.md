# FedCRG completion task — resumable progress record

Lightweight state per prompt.md section 2. Not a duplicate audit matrix —
see `docs/protocol_implementation_ledger.md` for the substantive findings.

## What was audited

- PR #1 merged into `main` (done, first action of the task).
- Pass 1-3 (structural / scientific-wiring / quality) completed earlier:
  ~30+ real defects found and fixed across CLI wiring, dict-domain
  violations, dead/duplicate code, Parquet serialization, orphaned
  subsystems. All static gates verified clean at that point.
- Pass 4 (final roadmap reconciliation), done in two rounds:
  - Round 1: orphan-symbol sweep across every `src/fedcrg/*` directory
    (grep-based, then manually verified each zero-external-reference
    candidate). Found and fixed: utility-margin formula duplication,
    unwired decision-architecture figure, unwired AdmissionSummary,
    unwired CalibrationAssignmentManifestStore (dict-shaped domain
    contract on a scientific-critical read path), several genuinely
    dead classes/functions.
  - Round 2: ran `fedcrg verify` end-to-end (not just static analysis)
    and traced the R2-R9/R12 workload-reconciliation path. Found a real
    scientific-contract drift: `ExperimentCompletionAuditor` read a JSON
    schema that none of the real R2-R9/R12/Table-5 producers ever wrote,
    so those workloads could never have verified as complete even with
    genuine evidence. Fixed producer/reader schema agreement, threaded
    `ExperimentCode` end-to-end where raw string literals hid the drift,
    deleted a dead `ExperimentCellEnvelope` class the broken reader
    appears to have originally been written against, added regression
    tests pinning the schemas.
- CLI command tree checked against prompt.md section 7's minimum list
  (`doctor`, `data prepare`, `tables precompute-readiness`,
  `synthetic run`, `train`, `score`, `evaluate`, `robustness deep-svdd`,
  `benchmark`, `report build`, `verify`) — all present and registered.

## Current state

- `pyright`: 0 errors. `ruff check`/`ruff format`: clean. `pytest -n auto`:
  126 passed. `python -m compileall`: clean. `git diff --check`: clean.
- `fedcrg doctor` and `fedcrg verify` run end-to-end against a live
  environment; `verify` truthfully reports all S1-S6/R1-R14 workloads
  incomplete against an empty `outputs/` (no fabricated evidence).

## What remains

- No known implementation defect at time of writing. Remaining `verify`
  failures are exclusively "experiment evidence missing", which is
  expected and correct per prompt.md section 3 — this task does not
  require running the publication-scale campaign.
- Checked: R11 (SECOND_DETECTOR) and R14 (DIAD_FEATURE_SENSITIVITY) both
  route through `_policy_run_workload`/`_external_policy_workload` in
  `experiments/completion.py`, i.e. the same `RunManifestStore`/
  `RunLayout.metric_records`-based mechanism already used and proven for
  R1/R10 — not a bespoke JSON envelope, so they do not share the R2-R9/
  R12 schema-drift bug class. No further action identified there.
- If resuming: the orphan-symbol sweep and the schema-drift trace (run
  the CLI end-to-end, follow one real workload's evidence from producer
  to reader) were the two techniques that found real bugs this session.
  Applying the same schema-drift trace to `report build-publication`'s
  figure/table pipeline against real (not synthetic-fixture) evidence
  would be the next place to look if genuine gaps are suspected.
