# FedCRG v2.0 Protocol Implementation Ledger

This ledger maps the normative `docs/roadmap.md` contract to executable repository components. It deliberately distinguishes **implementation completeness** from **experimental completion**. A code path being implemented does not imply that the corresponding real-data or Monte-Carlo experiment has been executed.

## Status vocabulary

- **IMPLEMENTED** — the repository contains the executable production path and typed artifact contract.
- **IMPLEMENTED / EVIDENCE REQUIRED** — the production path exists, but the required real-data or large-scale experiment must still be run.
- **GENERATED DURING RUN** — the artifact cannot truthfully exist in source control because it is derived from acquired datasets, hardware, or experiment outputs.
- **SUBMISSION-TIME** — deliberately performed immediately before release/submission rather than during implementation.

## Statistical protocol

| Roadmap contract | Status | Implementation |
|---|---|---|
| Equal-count federation reference threshold and finite-sample quantile rank | IMPLEMENTED | `src/fedcrg/protocol/reference.py` |
| Pre-data finite-sample readiness rank optimization | IMPLEMENTED | `src/fedcrg/protocol/readiness.py` |
| Persistent readiness-rank cache | IMPLEMENTED | `ReadinessPlanCache`; `fedcrg tables precompute-readiness` |
| Exact two-sided Clopper-Pearson mismatch evidence | IMPLEMENTED | `src/fedcrg/protocol/mismatch.py` |
| Parameter-dependent minimum mismatch sample size | IMPLEMENTED | `minimum_bidirectional_sample_count` |
| One-sided `a=0` sensitivity behavior | IMPLEMENTED | high-side-only mismatch evidence |
| Exact directional diagnostic p-values | IMPLEMENTED | `MismatchEvidence.p_low` / `p_high` |
| Fleet Bonferroni interval sensitivity | IMPLEMENTED | `bonferroni_fleet_sensitivity` |
| Holm directional exact sensitivity | IMPLEMENTED | `holm_directional_fleet_sensitivity` |
| Five-state deployment decision | IMPLEMENTED | `src/fedcrg/protocol/decision.py` |
| Selected-threshold tie/continuity diagnostics | IMPLEMENTED | `ContinuityDiagnostics` |
| Strict `score > threshold` classification | IMPLEMENTED | policy/metric evaluation paths |

## Data and preprocessing

| Roadmap contract | Status | Implementation |
|---|---|---|
| N-BaIoT nine natural-client mapping | IMPLEMENTED | `src/fedcrg/data/datasets/nbaiot.py` |
| N-BaIoT source-count cross-check | IMPLEMENTED / EVIDENCE REQUIRED | preparation manifest compares acquired files with locked ledger |
| N-BaIoT 115-feature numeric/finite contract | IMPLEMENTED | N-BaIoT adapter |
| DIAD 105 source identities before filtering | IMPLEMENTED / EVIDENCE REQUIRED | DIAD adapter and preparation stop gate |
| DIAD hashed natural client IDs | IMPLEMENTED | `DiadAdapter.public_client_id` |
| Locked 86-feature DIAD representation | IMPLEMENTED | `DIAD_FEATURES` allowlist |
| DIAD pre-outcome eligibility and exclusion precedence | IMPLEMENTED | `src/fedcrg/data/eligibility.py` |
| Deterministic source-order / verified-time ordering | IMPLEMENTED | DIAD adapter chronology branch |
| Seed-independent train/reservoir/final-test partition | IMPLEMENTED | `DataSplitter.split_base` |
| Deterministic calibration-role assignment | IMPLEMENTED | `CalibrationAssignmentBuilder` |
| Source-order calibration sensitivity | IMPLEMENTED | source-order assignment mode / R12 path |
| N-BaIoT balanced 500-record attack-development split | IMPLEMENTED | exact subtype allocation |
| DIAD capacity-aware 500-record water-filling split | IMPLEMENTED | exact lexical water filling |
| Training-only DIAD local-median imputation | IMPLEMENTED | `FederatedPreprocessor` |
| Training-only federated global min/max | IMPLEMENTED | `FederatedPreprocessor` |
| No post-scaling clipping | IMPLEMENTED | preprocessing transform |
| Preprocessing fit-row hashes | IMPLEMENTED | preprocessing manifest |
| DIAD R14 training-schema-only numeric-safe representation | IMPLEMENTED / EVIDENCE REQUIRED | `src/fedcrg/data/feature_sensitivity.py` |
| Dataset/source SHA-256 manifests | GENERATED DURING RUN | `fedcrg data prepare` |

## Detector and federated training

| Roadmap contract | Status | Implementation |
|---|---|---|
| N-BaIoT 115-86-57-38-29 symmetric AE | IMPLEMENTED | typed config + mirrored autoencoder |
| DIAD 86-64-43-28-21 symmetric AE | IMPLEMENTED | typed config + mirrored autoencoder |
| Xavier uniform with tanh gain 5/3 and zero bias | IMPLEMENTED | detector initialization |
| Exact 30-round cosine LR schedule | IMPLEMENTED | federated scheduling |
| Fresh Adam state every round | IMPLEMENTED | `FederatedClient.train` |
| Deterministic `(model_seed, client, round, epoch)` shuffle | IMPLEMENTED | epoch hash seed |
| Ordinary final mini-batch / `drop_last=false` | IMPLEMENTED | local DataLoader |
| Equal-client model aggregation | IMPLEMENTED | federated server/aggregation |
| Per-round mean/min/max loss | IMPLEMENTED | `RoundResult` |
| Parameter-update norm | IMPLEMENTED | `FederatedTrainer` |
| Tensor communication accounting | IMPLEMENTED | training manifest + `analysis/communication.py` |
| DIAD round-20 versus final training-score correlation | IMPLEMENTED | optional locked diagnostic |
| Mandatory Deep-SVDD second score generator | IMPLEMENTED / EVIDENCE REQUIRED | detector + R11 configuration/workload |
| Five primary AE model trainings | IMPLEMENTED / EVIDENCE REQUIRED | `experiment execute-grid` |
| Three Deep-SVDD trainings | IMPLEMENTED / EVIDENCE REQUIRED | R11 grid |

## Score cache and policy information regimes

| Roadmap contract | Status | Implementation |
|---|---|---|
| One immutable score cache per dataset/model seed | IMPLEMENTED | `src/fedcrg/scoring/cache.py` |
| Calibration seed excluded from physical score-cache identity | IMPLEMENTED | data/training hash separation + calibration views |
| Float64 stored scores after detector inference | IMPLEMENTED | score computer/cache |
| Row provenance in score cache | IMPLEMENTED | Parquet `row_id` records |
| SHA-256 finalized before policy evaluation | IMPLEMENTED | score cache validation |
| Benign-only method API has no attack-label input | IMPLEMENTED | `FedCRGProtocol` / `BenignPolicyEvidence` |
| B0-B6 benign-only evidence separation | IMPLEMENTED | typed policy evidence |
| B7-B9 supervised 500/500 development evidence | IMPLEMENTED | `SupervisedDevelopmentEvidence` |
| B10 final-test oracle isolated as diagnostic | IMPLEMENTED | `FinalTestEvidence` / oracle policy |
| All 12 registered policy IDs | IMPLEMENTED | `PolicyId` / `PolicyRegistry` |
| Exact shared/local finite-sample quantile convention | IMPLEMENTED | policy quantile helper |
| Exact shrinkage candidate/tie rule | IMPLEMENTED | shrinkage module |
| Published-style three-sigma comparator | IMPLEMENTED | quantile comparator module |
| Development local/global selector | IMPLEMENTED | attack-aware policy module |
| Laridi-style summary-statistic/F1 comparator | IMPLEMENTED | attack-aware policy module |
| 1,000-candidate supervised F1 comparator | IMPLEMENTED | attack-aware policy module |

## Metrics and statistical analysis

| Roadmap contract | Status | Implementation |
|---|---|---|
| Client FPR/TPR/precision/F1/balanced accuracy | IMPLEMENTED | `src/fedcrg/metrics/` |
| Undefined metric values remain `NA`/`None` | IMPLEMENTED | classification metric rules |
| MEBE / HighExcess / BandViolationRate / MAFE | IMPLEMENTED | federation aggregation |
| Attack-balanced per-client and federation macro TPR | IMPLEMENTED | attack-balanced metrics |
| AUROC/AUPRC score invariance check | IMPLEMENTED | federation metric assertions |
| Exact final-benign CP reference interval | IMPLEMENTED | evaluation path |
| Utility anchor and -3 pp margin | IMPLEMENTED | federation utility assessment |
| -1 pp / -5 pp sensitivity support | IMPLEMENTED | margin parameterization |
| Threshold SD/IQR | IMPLEMENTED | `analysis/stability.py` |
| State-transition frequency | IMPLEMENTED | `analysis/stability.py` |
| Five-seed fixed-federation summaries | IMPLEMENTED / EVIDENCE REQUIRED | primary analysis |
| 10,000-replicate paired seed bootstrap | IMPLEMENTED / EVIDENCE REQUIRED | statistical analysis |
| Split-sensitivity summaries without pseudo-replication | IMPLEMENTED / EVIDENCE REQUIRED | primary analysis |

## Experiment registry and execution

| Roadmap contract | Status | Implementation |
|---|---|---|
| Exact S1-S6 / R1-R14 registry | IMPLEMENTED | `src/fedcrg/experiments/definitions.py` |
| Exact synthetic trial/cell ledgers | IMPLEMENTED | typed workload expectations |
| R1 5×50×12 policy-cell workload | IMPLEMENTED / EVIDENCE REQUIRED | workload reconciliation |
| R10 5×20×12 external workload | IMPLEMENTED / EVIDENCE REQUIRED | workload reconciliation |
| R11 five-policy interpretation and 1,350 client-policy cells | IMPLEMENTED / EVIDENCE REQUIRED | detector spec/workload-consistent interpretation |
| R2-R9 frozen-score sensitivity paths | IMPLEMENTED / EVIDENCE REQUIRED | sensitivity application |
| R12 N-BaIoT and DIAD source-order role sensitivity | IMPLEMENTED / EVIDENCE REQUIRED | source-order application/config profiles |
| R13 100 warm-up / 1,000 measured calls | IMPLEMENTED / EVIDENCE REQUIRED | benchmark application |
| R14 derived-feature DIAD training path | IMPLEMENTED / EVIDENCE REQUIRED | feature sensitivity application |
| Train once / score once / evaluate many calibration-policy cells | IMPLEMENTED | `ExecuteFrozenWorkload` |
| Exact experiment completion reconciliation | IMPLEMENTED | `ExperimentCompletionAuditor` |

## Artifacts and reproducibility

| Roadmap contract | Status | Implementation |
|---|---|---|
| Immutable `outputs/runs/<run_id>` | IMPLEMENTED | artifact layout/manifests |
| Reusable dataset/model/score caches separated from run evidence | IMPLEMENTED | `outputs/cache/` contract |
| Canonical run identity with protocol parameters and policy | IMPLEMENTED | `RunId.for_policy_cell` |
| Dataset/preprocessing/training/score provenance chain | IMPLEMENTED | SHA-256 manifests and cache references |
| Threshold JSONL evidence | IMPLEMENTED | typed threshold records |
| Metric JSONL evidence | IMPLEMENTED | typed metric records |
| Environment/Git-state capture | IMPLEMENTED | artifact environment module |
| Semantic as well as cryptographic verification | IMPLEMENTED | `ArtifactVerifier` |
| Full workload reconciliation in `fedcrg verify` | IMPLEMENTED | repository verifier |
| Reports generated only from immutable evidence | IMPLEMENTED | report builder |
| Environment lock/pin | GENERATED / MUST BE FROZEN BEFORE CONFIRMATORY RUNS | packaging environment must be frozen at first successful protocol validation |
| Novelty search within seven days of submission | SUBMISSION-TIME | Appendix D procedure; not an implementation-time result |

## Experiments not claimed as completed

The repository must not mark S1-S6 or R1-R14 complete merely because their code exists. Real dataset acquisition, five-seed federated training, 970,000 locked Monte-Carlo trials, DIAD eligibility evaluation, Deep-SVDD training, R13 benchmarking, and final publication tables/figures require execution in the intended environment. `fedcrg verify` is designed to remain failing/incomplete until those evidence ledgers reconcile.

## Post-merge implementation audit (this session)

PR #1 was merged into `main` and then audited end to end. The merge left several
CLI/application call sites broken (calling renamed or nonexistent methods) and
a handful of real scientific/typing defects that the merged test suite had
never exercised. All were fixed, not papered over:

- `fedcrg evaluate`, R12 source-order sensitivity, and R14 config derivation
  were calling nonexistent or unsafe APIs (`EvaluatePolicies.evaluate`,
  `model_copy(update=...)` bypassing full config revalidation). Fixed to use
  `evaluate_from_cache` and a full-payload `model_validate` rebuild.
- `PrepareData` called a nonexistent `DatasetAdapter.iter_clients`; `fedcrg data
  prepare` could not run for either dataset before this fix.
- `capture_environment()`'s dict keys didn't match what `run_experiment.py`
  read, so every `RunExperiment.execute()` call raised `KeyError`.
- The score cache's Parquet schema drifted across roles because the
  label/attack-family columns lacked stable dtypes; fixed with explicit
  nullable dtypes so streaming multi-role writes are schema-consistent.
- `PublicationTableBuilder` was missing 6 of its ~10 table builders
  (`literature_boundary`, `primary_policy_results`, `admission_states_from_runs`,
  `ablations`, `sensitivity`, `external_replication`); Tables 1/4/5/6/7/8 always
  silently reported "unavailable" regardless of evidence. Implemented all six.
- Consolidated the duplicate R14/DIAD-feature-sensitivity implementation
  (`application/r14.py` + `data/r14_feature_contract.py` deleted; canonical
  path is `data/feature_sensitivity.py` + `application/feature_sensitivity.py`).
- Deleted dead code: `application/research_pipeline.py`'s preflight gate was
  unwired — instead of deleting it, it is now the real `execute-grid` path
  (`ExecuteResearchPipeline` audits prepared data and precomputes readiness/
  mismatch tables before every frozen-workload execution); `artifacts/
  experiment_layout.py` was genuinely dead and removed.
- Wired the previously dead `EnvironmentLocker` behind `fedcrg environment
  freeze`, and implemented `VerifyOutputs.verify_protocol_precompute` (the G1
  claim-gate dependency referenced but never implemented).
- Renamed `robustness second-detector` to `robustness deep-svdd` to match the
  roadmap-required command name and the actual algorithm name.
- Replaced dict-shaped frozen-dataclass fields with typed tuple-of-record
  models plus lookup helpers across `scoring/`, `data/`, `artifacts/`, and
  `analysis/`, per the no-dict-domain-model rule.
- `pyright`: 0 errors (down from 89 once the merge's actual state was measured).
  `ruff check`/`ruff format`: clean across the whole tree, which had never been
  formatted since the refactor. Full suite: 121 passed, 0 failed.

None of this changes the experimental-evidence status above: `fedcrg verify`
against an empty `outputs/` still truthfully reports all S1-S6/R1-R14 workloads
incomplete, as intended.
