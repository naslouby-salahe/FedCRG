# Next Action

**Last updated:** 2026-08-12
**Current time:** 2026-08-12

## Immediate Next Action

**Priority:** HIGH   
**Task:** Implement detector models (fedcrg/models/)  
**File:** `fedcrg/models/` module  
**Status:** NOT STARTED

### Specific Steps

1. **Create fedcrg/models/__init__.py** - Package initialization
2. **Implement Autoencoder model** - Per Section 8.2 (AE architecture)
   - Architecture: 115-86-57-38-29-38-57-86-115 (per Section 8.2.1)
   - Activation: tanh (per FedDetect paper)
   - Loss: MSE reconstruction
   - Input: 115 features for N-BaIoT, 86 for DIAD
3. **Implement Deep-SVDD model** - Per Section 8.4
   - Encoder: [115, 64, 32] for N-BaIoT
   - Center computation: equal_mean_of_client_initial_embeddings
   - Loss: SVDD objective
4. **Create model base class** - Common interface for all models

### Why This is Next

According to prompt.md Section 8 (Implementation strategy):
- Phase 1: Domain model (DONE)
- Phase 2: Configuration (DONE)
- Phase 3: Dataset discovery, integrity and deterministic preparation (DONE)
- Phase 4: Role assignment and leakage prevention (DONE in data module)
- Phase 5: Preprocessing (NEXT after models, or parallel)
- Phase 6: Detector training (REQUIRES models)

The CLI commands in Section 14.10 require:
- `fedcrg train --config configs/nbaiot_primary.yaml`

This command needs the model implementations to exist.

### Blocking Dependencies

None. This task can proceed immediately.

## After Detector Models

1. Implement federated training (fedcrg/fl/)
2. Implement preprocessing and feature engineering (fedcrg/data/preprocess.py)
3. Implement scoring and score caching

## Alternative Paths

Could also implement preprocessing first, but detector models are needed for
training, which is a prerequisite for most other phases.

## Time Estimate

- Autoencoder model: 2-3 hours
- Deep-SVDD model: 2-3 hours
- Total: ~4-6 hours

## Resources Needed

- Roadmap Section 8 (Frozen Detector and Federated Training)
- Roadmap Section 8.2 (AE architecture)
- Roadmap Section 8.4 (Deep-SVDD)
- FedDetect paper for AE hyperparameters (batch_size=64, 120 epochs, 30 rounds)
