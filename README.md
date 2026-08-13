# FedCRG

**Federated Calibration Readiness Gate** — evidence-admitted client-specific thresholding for federated IoT anomaly detection.

FedCRG is a **post-training operating-point governance protocol**. It does not introduce a new federated optimizer or anomaly detector. A client-specific threshold is admitted only when two independent benign evidence conditions hold:

1. the federation reference threshold is shown to be materially outside the client’s pre-registered benign FPR band; and
2. an independent local calibration sample is large enough to construct an order-statistic threshold with the locked finite-sample in-band readiness assurance.

The normative research specification is [`docs/roadmap.md`](docs/roadmap.md). The live roadmap-to-code control document is [`docs/FedCRG Audit Matrix.md`](docs/FedCRG%20Audit%20Matrix.md).

## Repository architecture

```text
configs/                 YAML profiles: method, training, randomness, statistics,
                         datasets, detectors, and per-experiment compositions
src/fedcrg/
  domain/                enums, identifiers, value objects, errors
  configuration/         typed configuration resolution and validation
  datasets/              natural-client adapters, splitting, eligibility, preprocessing
  detectors/             autoencoder and Deep-SVDD score generators
  federation/            deterministic client/server training and aggregation
  scoring/               immutable score caches, calibration views, score computation
  decision/              readiness, mismatch, threshold decision, policy evidence,
                         policy selection, and per-policy comparators
  evaluation/            client/federation reliability and utility metrics
  experiments/           one execution spine: runner, preflight, verification,
                         campaign, table precompute, and the S1-S6/R1-R14 catalogue
  analysis/              statistics, contrasts, stability, claim gates
  artifacts/             immutable layouts, manifests, hashes, environment evidence
  reporting/             publication tables/figures, reports, results bundles
  runtime/               structured logging, resource monitoring, CUDA guards
  cli/                   thin research command surface
tests/                   contract, regression, unit, and integration verification
outputs/                 generated caches, runs, experiment evidence, and reports
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
│   └── precomputed/
├── experiments/<S1..S6,R1..R14>/
├── runs/<immutable-run-id>/
└── reports/{latest,publication}/
```

Physical detector scores are cached **once per dataset/model seed**. Calibration seeds only create deterministic R/G/C/guard views over the same frozen reservoir scores; they do not retrain or rescore the detector. Completed run directories are immutable and reference upstream caches by relative path plus SHA-256.

## Install

Python 3.11 or 3.12 is supported.

```bash
python -m pip install -e '.[dev]'
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
fedcrg config validate --path configs/experiments/primary/nbaiot.yaml
fedcrg data prepare --config configs/experiments/primary/nbaiot.yaml --data-root /path/to/nbaiot
fedcrg tables precompute-readiness --config configs/experiments/primary/nbaiot.yaml
fedcrg train --config configs/experiments/primary/nbaiot.yaml --prepared-root <preprocessed-root> --model-seed 11
fedcrg score --config configs/experiments/primary/nbaiot.yaml --prepared-root <preprocessed-root> --model-path ... --model-seed 11
fedcrg experiment execute-grid --config configs/experiments/primary/nbaiot.yaml --prepared-root <preprocessed-root>
fedcrg report build-repository --outputs outputs --config configs/experiments/primary/nbaiot.yaml
fedcrg verify --outputs outputs
```

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
- Negative/non-replicating scientific results remain reportable. Claim strength is reduced; the method is not redesigned on the same confirmatory outcomes.

## Implementation is not experimental completion

The codebase implements the registered S1-S6 / R1-R14 paths and evidence contracts. That does **not** mean the locked experiments have already been executed. Dataset acquisition/source hashes, five-seed federated training, the 970,000 S1-S5 Monte-Carlo trials, S6 exact power cells, DIAD eligibility/results, Deep-SVDD runs, R13 machine-specific benchmarking, and final manuscript artifacts must be generated in the intended environment.

`fedcrg verify` is expected to remain incomplete until those evidence ledgers reconcile. It must never infer or fabricate missing experiment results.

## Development rules

- Keep source modules named for responsibilities/capabilities rather than manuscript shorthand.
- Closed states/identities belong in enums or value objects; JSON/YAML/string conversion occurs at boundaries.
- Do not add compatibility shims for deleted prototype APIs.
- Do not place scientific verification in ad-hoc `scripts/`; use production validation or the structured test suites.
- Comments explain scientific or engineering rationale, not obvious syntax.
- Do not manually type numerical manuscript results that can be generated from immutable artifacts.

## Status

The implementation is being hardened against the v2.0 roadmap, tracked in `docs/FedCRG Audit Matrix.md` and `docs/work/`. See the audit matrix for the current distinction between executable code and evidence that still requires experimental execution.
