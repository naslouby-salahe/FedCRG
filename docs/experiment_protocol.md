# Experiment Protocol

The implementation treats the frozen research protocol as configuration plus typed domain rules. A run is reproducible only when its resolved configuration, dataset manifest, split manifest, preprocessing state, model artifact, score cache, decisions, metrics, and verification hashes agree.

## Information boundaries

The fitting path receives benign training, reference, mismatch-evidence, and calibration roles only. Supervised comparator development data are exposed only to the supervised policy modules. Final benign and attack test roles are evaluation-only. The final-test oracle is isolated as a diagnostic policy and must never influence deployable policy selection.

## Natural-client datasets

N-BaIoT uses the nine physical devices as clients. CIC IoT-DIAD uses stable hashed device identities established before model tensors are created. Dataset adapters discover source material, enforce the feature contract, emit stable row identifiers, and return typed client records. Partitioning is a separate deterministic service.

## Training

The primary autoencoder uses the dataset-specific architecture and training schedule from the frozen protocol. Global min/max scaling is fitted only from client training rows through extrema aggregation. Local optimizers are reset each round. Client participation and aggregation are explicit configuration values. Confirmatory execution does not perform early stopping or outcome-driven model selection.

## Threshold governance

The operating-point protocol is composed of four responsibilities:

1. reference-threshold estimation;
2. local calibration-readiness evaluation;
3. reference-mismatch evidence evaluation;
4. a single typed threshold-decision engine.

No other layer is allowed to recreate this decision logic.

## Comparators

Policies are selected at federation scope through `FederationPolicySelector`. Each policy is registered with its information regime. Benign-only, supervised-development, and final-test-oracle policies are therefore auditable as distinct classes of evidence access.

## Artifacts

Every run receives a deterministic `RunId` derived from experiment identity, resolved-config hash, model seed, and calibration seed. Completed run directories are immutable evidence. Reusable caches live outside `runs/` so cache lifecycle cannot be confused with publication evidence.
