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

DONE 2026-08-13: consolidated to `cli/app.py` (root, doctor, monitor, config),
`data_commands.py` (data preprocess|status|prepare, environment),
`experiment_commands.py` (experiment validate|plan|run-policy-cell|
materialize-federation-cell|execute-grid, train, score, evaluate, synthetic,
robustness, sensitivity, benchmark, campaign run|status|list),
`analysis_commands.py` (claims, tables, verify), `report_commands.py` (report,
results build|verify); entry point `cli.app:cli`; all 14 legacy files deleted.

## Batch 8 — Makefile, Nox, CI, typing (matrix R10, R11, T03, T07, E2)

DONE 2026-08-13: Makefile with all 22 required targets (no scientific values);
`noxfile.py` with format/lint/typecheck/unit/integration/contract/regression/
audit/quality sessions (all green, quality green in clean venv);
`tools/audit_repository.py` (fast architecture re-audit, exit-code gated);
CI switched mypy -> pyright (+ format check, audit, pytest -n auto);
`[tool.pyright]` standard mode in pyproject; pandas-stub defects fixed with
narrow cast (publication groupby, nbaiot apply, cache column); `canonical`
removed from nbaiot (`_FIXED_DEVICES`); requirements.lock updated (pyright,
nox, pytest-xdist); tests/ made a package; pytest pythonpath includes ".".

## Batch 9 — Contract tests and typing audit (matrix T02, T06, D09, D10, E1, E3)

DONE 2026-08-13: `JsonValue`/`JsonObject` PEP-695 recursive alias; Pydantic now owns
`ExperimentResultEnvelope` (frozen BaseModel, cells/metadata as JsonObject) and
`CampaignStatus`/`ExperimentCampaignStatus` (model_dump / model_validate_json, manual
loaders deleted); `EnvironmentSnapshot` typed dataclass; all manifest/integrity/
reporting/cache JSON helpers typed `JsonValue` or `Mapping[str, JsonValue]`; zero
`dict[str, object]` and zero bare `object` returns remain in src; new
`tests/contract/test_architecture_contract.py` enforces §28 (canonical, redirect
modules, scientific defaults in config models, dict-transport, preprocessing identity
sharing). 146 tests, pyright 0/0, nox quality green.

## Batch 10 — Full quality gate and hostile audit (matrix X01, X02, F1-F4)

1. Run complete quality gate (format, lint, pyright, full pytest, nox).
2. Hostile audit per prompt §38; fix every actionable finding; re-audit until clean.
3. Update matrix to all VERIFIED; commit.
