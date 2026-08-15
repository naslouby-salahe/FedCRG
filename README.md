# FedCRG

**Federated Calibration Readiness Gate** — evidence-admitted client-specific thresholding for federated IoT anomaly detection.

FedCRG is a **post-training operating-point governance protocol**. It does not introduce a new federated optimizer or anomaly detector. A client-specific threshold is admitted only when two independent benign evidence conditions hold:

1. the federation reference threshold is shown to be materially outside the client’s pre-registered benign FPR band; and
2. an independent local calibration sample is large enough to construct an order-statistic threshold with the locked finite-sample in-band readiness assurance.

## Repository architecture

```text
config/                  YAML profiles: study, datasets, experiments catalogue
src/fedcrg/
  config.py              typed configuration models and study resolution
  types.py               constrained aliases and closed enums
  runtime.py             structured logging, resource monitoring, CUDA guards
  reporting.py           publication tables/figures, reports, results bundles
  cli.py                 thin research command surface
  data/                  natural-client adapters, splitting, eligibility, preprocessing
  learning/              detectors (AE/Deep-SVDD), federated training, score caches
  thresholding/          readiness, mismatch, threshold decision, policy evidence,
                         policy selection, and per-policy comparators, metrics
  evidence/              immutable layouts, manifests, hashes, environment evidence
  experiments/           one execution spine: runner, preflight, verification,
                         campaign, table precompute, and the pre-registered
                         experiment catalogue
tests/                   contract, unit, and integration verification
outputs/                 generated caches, runs, campaign status, evidence, reports
data/preprocessed/       deterministic prepared datasets (reused by identity)
results/                 publication bundles per campaign
tools/                   optional developer/release utilities only
```

## Evidence layout

Generated evidence is deliberately separated by lifecycle:

```text
outputs/
├── cache/
│   ├── models/
│   ├── scores/
│   └── analysis/        readiness-plan and mismatch-cutoff tables
├── runs/<immutable-run-id>/
├── campaigns/
├── monitoring/
└── reports/{publication,benchmark.json}/
```

Physical detector scores are cached **once per dataset/model seed**. Calibration seeds only create deterministic R/G/C/guard views over the same frozen reservoir scores; they do not retrain or rescore the detector. Completed run directories are immutable and reference upstream caches by relative path plus SHA-256.

## Install

Python 3.12 is supported.

```bash
uv sync --extra dev
```

After the first validated protocol environment, freeze the exact installed runtime versions:

```bash
python tools/freeze_environment.py
```

The generated `requirements.lock` is then versioned as part of the protocol freeze rather than silently regenerated mid-study.

## Research workflow

Representative commands:

```bash
fedcrg doctor
fedcrg validate primary_nbaiot
fedcrg preprocess nbaiot
fedcrg plan primary_nbaiot
fedcrg run primary_nbaiot
fedcrg campaign
fedcrg status
fedcrg monitor
fedcrg report
fedcrg results build
fedcrg results verify
```

`fedcrg preprocess` without a dataset argument prepares every raw dataset. The CLI takes no path or configuration options: repository layout and the campaign identity are owned by `config/study.yaml`. `fedcrg campaign` executes every registered experiment under the configured campaign identity; pass `--overwrite` to restart it from scratch.

Every completed experiment run writes its own Markdown summary into the run directory (`outputs/runs/<run>/reports/summary.md`), and `fedcrg run` additionally refreshes the aggregate repository report. `fedcrg report` regenerates the same deliverables on demand — use it standalone whenever report code changes.

The high-level research application path performs a prepared-data audit and freezes statistical lookup tables before model training. Lower-level services remain available for reproducible component work, but confirmatory execution should use the audited path.

## Scientific invariants

- Natural device identities are the federated clients; no Dirichlet pseudo-clients are used in confirmatory real-data experiments.
- Threshold comparison is always **`score > threshold`**; equality is benign.
- Reference, mismatch, and local-calibration evidence streams are disjoint.
- FedCRG admission accepts benign evidence only. Attack development labels are exposed only to the explicitly supervised B7-B9 comparators; final labels are evaluation/oracle evidence only.
- N-BaIoT uses the locked 115-dimensional AE and 30×120 training scale; DIAD uses the locked 86-dimensional AE and 30×20 scale.
- Federated preprocessing is fitted from benign training rows only; DIAD imputation is client-local and min/max scaling exchanges extrema rather than centralizing rows.
- Every score cache is float64, hash-finalized, and immutable before threshold-policy evaluation.
- Global-threshold uncertainty procedures must recompute the shared threshold inside a resampled federation; already-computed client metrics are not treated as independent observations.

## Implementation is not experimental completion

The codebase implements the registered S1-S6 / R1-R14 paths and evidence contracts. That does **not** mean the locked experiments have already been executed. Dataset acquisition/source hashes, five-seed federated training, the 970,000 S1-S5 Monte-Carlo trials, S6 exact power cells, DIAD eligibility/results, Deep-SVDD runs, R13 machine-specific benchmarking, and final manuscript artifacts must be generated in the intended environment.

`fedcrg results verify` is expected to remain incomplete until those evidence ledgers reconcile. It must never infer or fabricate missing experiment results.

## Development rules

- Keep source modules named for responsibilities/capabilities rather than manuscript shorthand.
- Closed states/identities belong in enums or value objects; JSON/YAML/string conversion occurs at boundaries.
- Do not add compatibility shims for deleted prototype APIs.
- Do not place scientific verification in ad-hoc `scripts/`; use production validation or the structured test suites.
- Comments explain scientific or engineering rationale, not obvious syntax.
- Do not manually type numerical manuscript results that can be generated from immutable artifacts.

## Status

The implementation is being hardened against the v2.0 roadmap, tracked in `docs/FedCRG Audit Matrix.md` and `docs/work/`. See the audit matrix for the current distinction between executable code and evidence that still requires experimental execution.
