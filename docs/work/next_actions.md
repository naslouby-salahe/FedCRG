# Next Actions

Execution order. Each batch is a substantial coherent unit; run targeted validation,
update the audit matrix, and commit after each batch. Idempotent: re-run checks before
starting a batch; skip work already done.

## Batch 1 — Configuration ownership (matrix K01-K11, B1-B4)

DONE 2026-08-13 (commit after b1): method/training/randomness/statistics YAML layout,
required Pydantic fields, `domain/constants.py` deleted, parameter/byte counts derived,
statistics threaded through analysis, `validate.py` structural-only, tests updated.

## Batch 2 — Vocabulary and PolicyId (matrix R04, E07, A1, A2)

DONE 2026-08-13: `canonical` removed from production source; `PolicyId` enum values
renamed to descriptive snake_case (configs, reporting, tests updated); README rewritten
to match the actual tree and CLI.

## Batch 3 — Preprocessed root and outputs layout (matrix R06, R08, D05, D12, C2, C3)

1. Move prepared-dataset materialization to `data/preprocessed/<dataset>/<identity>/`.
2. Restructure outputs: add `logs/`, `monitoring/`; rename `cache/precomputed/` ->
   `cache/analysis/`; remove `cache/datasets/`.
3. Update integration/contract tests to assert the new roots.

## Batch 4 — Logging, monitoring, GPU (matrix L07, L08, L10, L11, D2-D5)

1. `runtime/logging.py` with file handler to `outputs/logs/`.
2. Monitoring sampler (psutil + torch CUDA) writing `outputs/monitoring/`; `fedcrg
   monitor`; campaign hook.
3. GPU: device logging, explicit CUDA guard, inference_mode, VRAM logging.
4. Add deps (psutil, rich) + requirements.lock update.

## Batch 5 — Campaign and results (matrix R09, L04, L05, P03-P05, C4)

1. Campaign concept: `fedcrg campaign run|status`; persistent status store.
2. `fedcrg results build [CAMPAIGN_ID]` and `results verify [CAMPAIGN_ID]` sharing one
   builder; campaign completion auto-invokes the same builder.
3. Publication bundle under `results/<campaign_id>/` (manifest, checksums, configs,
   metrics, statistics, tables, figures, reports, provenance).

## Batch 6 — Execution spine and package layout (matrix R01, R02, E09, C1, C5)

1. Fold `pipeline/` into `experiments/` (runner, preflight, verification) and capability
   modules; delete `pipeline/` package.
2. Rename `config/` -> `configuration/`, `data/` -> `datasets/`, `method/`+`thresholds/`
   -> `decision/`, `runtime.py` -> `runtime/logging.py` (module), `cli/` -> target files.
3. Move `analysis/communication_cost.py` -> `evaluation/communication_metrics.py`;
   `analysis/computational_benchmark.py` -> `experiments/computational_benchmark.py`.
4. Update all imports, CLI, tests.

## Batch 7 — CLI surface (matrix L01-L06, D1)

1. Consolidate CLI to `app.py`, `data_commands.py`, `experiment_commands.py`,
   `analysis_commands.py`, `report_commands.py`.
2. Add `data preprocess [DATASET_ID]`, `data status`, `experiment validate|run`,
   `campaign run|status`, `results build|verify`, `monitor`, `report`.
3. Rich console progress; keep CLI thin.

## Batch 8 — Makefile, Nox, CI, typing (matrix R10, R11, T03, T07, E2)

1. Add `Makefile` with all required targets.
2. Add `noxfile.py` sessions.
3. Wire pyright into pyproject + CI; keep ruff; update CI workflow.
4. Update requirements.lock.

## Batch 9 — Contract tests and typing audit (matrix T02, T06, D09, E1, E3)

1. Type `artifacts/records.py`, `environment.py` (typed models), integrity helpers.
2. Add missing contract tests (preprocessed root, no pipeline pkg, no constants.py,
   results verify, no scientific defaults, YAML ownership).
3. Update stale contract tests (output layout, preprocessing artifact).

## Batch 10 — Full quality gate and hostile audit (matrix X01, X02, F1-F4)

1. Run complete quality gate (format, lint, pyright, full pytest, nox).
2. Hostile audit per prompt §38; fix every actionable finding; re-audit until clean.
3. Update matrix to all VERIFIED; commit.
