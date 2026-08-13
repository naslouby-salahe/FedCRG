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
