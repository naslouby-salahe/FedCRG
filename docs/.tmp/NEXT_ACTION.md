# Next Action

**Last updated:** 2026-08-12
**Current time:** 2026-08-12

## Immediate Next Action

**Priority:** HIGH   
**Task:** Implement data infrastructure (fedcrg/data/)  
**File:** `fedcrg/data/` module  
**Status:** NOT STARTED

### Specific Steps

1. **Implement DatasetID enum** - Already in config.py, verify usage
2. **Create data adapter base class** - Abstract base for N-BaIoT and DIAD
3. **Implement N-BaIoT adapter** - Read from data/raw, handle 9 clients, 115 features
4. **Implement DIAD adapter** - Read from data/raw, handle eligibility filtering, 86 features
5. **Implement role-based splitting** - R, G, C, guard partitions per roadmap Section 7
6. **Implement manifest system** - SHA-256 hashing, row counts, eligibility tracking

### Why This is Next

According to prompt.md Section 8 (Implementation strategy):
- Phase 1: Domain model (DONE - config.py has enums)
- Phase 2: Configuration (DONE - Pydantic models + YAML)
- Phase 3: Dataset discovery, integrity and deterministic preparation (NEXT)

The CLI commands in Section 14.10 require:
- `fedcrg data prepare --config configs/nbaiot_primary.yaml`
- `fedcrg data prepare --config configs/diad_external.yaml`

These commands need the data module to be implemented.

### Blocking Dependencies

None. This task can proceed immediately.

## After Data Infrastructure

1. Implement preprocessing and feature engineering (fedcrg/data/preprocess.py)
2. Implement detector models (fedcrg/models/)
3. Implement federated training (fedcrg/fl/)
4. Implement scoring and score caching

## Alternative Paths

None - data infrastructure is the clear next priority per implementation strategy.

## Time Estimate

- Data adapters: 2-4 hours
- Role splitting: 1-2 hours
- Manifest system: 1-2 hours
- Total: ~4-8 hours

## Resources Needed

- Access to data/raw symlink (to be set up)
- Roadmap Section 7 (Dataset and Data-Partition Protocol)
- Roadmap Section 7.1 (N-BaIoT), Section 7.2 (DIAD)
