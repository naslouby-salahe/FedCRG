# Next Action

**Last updated:** 2026-08-12
**Current time:** 2026-08-12

## Immediate Next Action

**Priority:** HIGH   
**Task:** Implement experiment registry and execution pipeline  
**Module:** `fedcrg/experiments/`  
**Status:** NOT STARTED

### Specific Steps

Implement the experiment registry per Section 11 of the roadmap:

1. **Synthetic experiments (S1-S6):**
   - S1: IID Gate-A theorem validation
   - S2: Global threshold scaling
   - S3: Band width sensitivity
   - S4: Gate-B sensitivity
   - S5: Gate-A assurance sensitivity
   - S6: tied-scores assumption violation

2. **Real data experiments (R1-R14):**
   - R1: N-BaIoT primary FedCRG
   - R2: N-BaIoT GATE-A-ONLY
   - R3: N-BaIoT FEDDETECT-3SIGMA
   - R4-R6: N-BaIoT Q99 baselines
   - R7-R8: N-BaIoT attack-aware baselines
   - R9: N-BaIoT shrinkage
   - R10: N-BaIoT oracle
   - R11-R14: DIAD and external validation

3. **Files to create:**
   - `fedcrg/experiments/__init__.py`
   - `fedcrg/experiments/registry.py` - Experiment registry
   - `fedcrg/experiments/synthetic.py` - S1-S6 synthetic experiments
   - `fedcrg/experiments/real_data.py` - R1-R14 real data experiments
   - `fedcrg/experiments/executor.py` - Experiment execution engine
   - `fedcrg/experiments/results.py` - Result collection and serialization

4. **Key requirements:**
   - Deterministic execution
   - Score cache reuse across experiments
   - Artifact hashing and provenance
   - Results serialization in JSON/CSV

### Why This is Next

Experiments are needed to:
- Validate FedCRG on synthetic data (theorem verification)
- Run primary N-BaIoT experiments
- Run external DIAD validation
- Generate results for tables and figures
- Support reproducibility and auditing

### Blocking Dependencies

None. All core modules complete (FedCRG algorithm, metrics, preprocessing, baselines, scoring).

### After Experiments

1. Implement CLI commands
2. Create and populate detailed matrix files
3. Perform four matrix audits
4. Implement raw data symlink (data/raw)
5. End-to-end integration testing
6. Run all experiments

### Time Estimate

8-12 hours

### Resources Needed

- Roadmap Section 11 (Experiment Registry)
- Roadmap Section 14 (Expected Values)
- Existing FedCRG, metrics, preprocessing, baselines modules
- numpy, pandas for data handling
