# Next Action

**Last updated:** 2026-08-12
**Current time:** 2026-08-12

## Immediate Next Action

**Priority:** HIGH   
**Task:** Implement federated training (fedcrg/fl/)  
**File:** `fedcrg/fl/` module  
**Status:** NOT STARTED

### Specific Steps

1. **Create fedcrg/fl/__init__.py** - Package initialization
2. **Implement FL trainer** - Per Section 8.2
   - Training state machine with 30 rounds
   - Adam optimizer with betas=(0.9, 0.999), eps=1e-8, weight_decay=0
   - Cosine LR schedule: eta_t = 1e-5 + 0.5*(1e-3-1e-5)*(1+cos(pi*t/29))
   - Deterministic shuffling per (model_seed, client_id, round, local_epoch)
   - Batch size 64, drop_last=false
   - Equal arithmetic mean aggregation
   - Fresh optimizer at each round (moments reset)
   - Score storage as float64 after float32 forward pass
3. **Implement server** - Model aggregation and broadcast
4. **Implement client trainer** - Local training loop

### Why This is Next

According to prompt.md Section 8 (Implementation strategy):
- Phase 1: Domain model (DONE)
- Phase 2: Configuration (DONE)
- Phase 3: Dataset discovery, integrity and deterministic preparation (DONE)
- Phase 4: Role assignment and leakage prevention (DONE in data module)
- Phase 5: Preprocessing (NEXT after fl, or parallel)
- Phase 6: Detector training (REQUIRES fl module)

The CLI commands in Section 14.10 require:
- `fedcrg train --config configs/nbaiot_primary.yaml`

This command needs the FL training implementation.

### Blocking Dependencies

None. Detector models are complete. This task can proceed immediately.

## After Federated Training

1. Implement scoring and score caching (fedcrg/scoring/)
2. Implement baseline suite (fedcrg/baselines/)
3. Implement preprocessing and feature engineering (fedcrg/data/preprocess.py)

## Time Estimate

- FL state machine and trainer: 4-6 hours
- Server/client implementation: 3-4 hours
- Total: ~7-10 hours

## Resources Needed

- Roadmap Section 8.2 (Federated training state machine)
- Roadmap Section 8.2.1 (Exact local batch semantics)
- Roadmap Section 8.1 (AE training parameters)
- Roadmap Section 8.1.1 (Learning-rate schedule)
