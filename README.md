# FedCRG

FedCRG is a post-training operating-point governance layer for federated IoT anomaly detection. It determines whether a client should retain a federation reference threshold or use a client-specific threshold from independent benign calibration evidence.

## Design principles

- One typed representation for each closed domain concept.
- One implementation of each protocol decision.
- Statistical functions are pure and deterministic.
- Configuration is validated before execution and resolved into an immutable run specification.
- Every completed run is self-contained under `outputs/runs/<run_id>/`.
- Production modules contain no embedded test programs or developer-specific paths.
- Experiment dependencies are explicit: failed prerequisites block dependants.

## Repository layout

- `src/fedcrg/`: production package.
- `configs/`: composable protocol, dataset, detector, and experiment profiles.
- `tests/`: unit, regression, contract, and integration tests.
- `outputs/`: runtime outputs; completed run directories are immutable.
- `docs/`: architecture and reproducibility documentation.

## Development

```bash
python -m pip install -e '.[dev]'
pytest
ruff check src tests
mypy src/fedcrg
```

## CLI

```bash
fedcrg config validate --config configs/experiments/primary/nbaiot.yaml
fedcrg experiment plan --config configs/experiments/primary/nbaiot.yaml
fedcrg verify --outputs outputs
```

See `docs/architecture.md` and `docs/reproducibility.md` for the execution and artifact contracts.
