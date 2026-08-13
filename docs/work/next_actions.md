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

DONE 2026-08-13: `data/preprocessed/<dataset_id>/<identity>/` materialization root
(`config.preprocessed_root`), `outputs/logs/` + `outputs/monitoring/` skeletons,
`cache/precomputed/` -> `cache/analysis/`, `outputs/cache/datasets/` removed; outputs
README/.gitignore updated.

## Batch 4 — Logging, monitoring, GPU (matrix L07, L08, L10, L11, D2-D5)

DONE 2026-08-13: `runtime/{logging,gpu,monitoring}.py` package; file logs to
`outputs/logs/`; `fedcrg monitor` + telemetry.jsonl; CUDA-required guard, device/VRAM
logging, `torch.inference_mode()` scoring; psutil dep; requirements.lock updated;
runtime unit tests added. Remaining: Rich progress (L08) lands with campaign work.

## Batch 5 — Campaign and results (matrix R09, L04, L05, P03-P05, C4)

DONE 2026-08-13: `experiments/campaign.py` (persistent CampaignStatusStore,
dependency-aware runner, telemetry per stage, auto results build on completion);
`reporting/results.py` (ResultsBuilder + ResultsVerifier; bundle layout matches §14);
CLI `campaign run|status|list` and `results build|verify`; unit + e2e tests.

## Batch 6 — Execution spine and package layout (matrix R01, R02, E09, C1, C5)

DONE 2026-08-13: `pipeline/` folded into `experiments/` (runner, preflight,
verification, experiment_runner, policy_cells, model_training, dataset_preparation,
table_precompute) and capability homes (scoring/compute_scores, evaluation/metrics);
`config/`->`configuration/`, `data/`->`datasets/`, `method/`+`thresholds/`->`decision/`
(with policies/), `analysis/communication_cost.py`->`evaluation/communication_metrics.py`,
`analysis/computational_benchmark.py`->`experiments/`; boundary contract updated;
README architecture section rewritten. Remaining: `cli/` consolidation (Batch 7).

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
