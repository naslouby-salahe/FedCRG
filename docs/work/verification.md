# Verification

Baseline (before any changes in this effort, 2026-08-13):

- `python3 -m pytest` -> 125 passed
- `pyright src/fedcrg` -> 0 errors, 0 warnings
- `git status` -> clean on `main`

## Verification commands

- Unit: `python3 -m pytest tests/unit -q`
- Contract: `python3 -m pytest tests/contract -q`
- Integration: `python3 -m pytest tests/integration -q`
- Regression: `python3 -m pytest tests/regression -q`
- Full: `python3 -m pytest -q`
- Type: `pyright src/fedcrg` (via `~/.nvm/versions/node/v25.8.2/bin/pyright`)
- Lint: `ruff check src tests`
- Format: `ruff format --check src tests`

## Batch verification log

| Batch | Command | Result | Date |
|-------|---------|--------|------|
| baseline | pytest; pyright | 125 passed; 0 errors | 2026-08-13 |
| b1 (config ownership) | pytest; pyright; ruff check; ruff format --check | 125 passed; 0 errors; 0 warnings; clean | 2026-08-13 |
| b2 (vocabulary) | pytest; pyright; ruff | 125 passed; clean | 2026-08-13 |
| b3 (preprocessed root + outputs layout) | pytest; pyright; ruff | 126 passed; clean | 2026-08-13 |
| b4 (logging/monitoring/GPU) | pytest; pyright; ruff; fedcrg monitor --samples 2 | 133 passed; telemetry.jsonl + fedcrg.log written | 2026-08-13 |
