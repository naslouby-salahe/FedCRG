# FedCRG Audit and Implementation Matrix

**Version:** 1.0  
**Created:** 2026-08-12  
**Status:** Initial extraction - structured by functional domain
**Source:** `docs/FedCRG Roadmap.md` v2.0

---

## Matrix Organization

This matrix is organized by **functional domain** and split across multiple files to manage size. Each file contains requirements from specific roadmap sections.

### Matrix Files

1. **Core Requirements** → `docs/matrix/01_core_requirements.md`
   - Global constants and identities (GLOBAL-*, PROTOCOL-*)
   - Mathematical formulas (FORMULA-*)
   - Decision states (STATE-*)

2. **Statistical Core** → `docs/matrix/02_statistical_core.md`
   - Gate A requirements (GATE-A-*)
   - Gate B requirements (GATE-B-*)
   - Precomputation and numerical requirements

3. **Dataset Specifications** → `docs/matrix/03_dataset_requirements.md`
   - N-BaIoT specifications (DATASET-*, SPLIT-*)
   - CIC IoT-DIAD specifications
   - Feature contracts (PREPROCESS-*)

4. **Training Specifications** → `docs/matrix/04_training_requirements.md`
   - Detector architecture and training (TRAIN-*)
   - Deep-SVDD specifications
   - Federated training state machine

5. **Baseline Suite** → `docs/matrix/05_baseline_requirements.md`
   - Baseline definitions (BASELINE-*)
   - Deterministic quantile conventions
   - Shrinkage and comparator specifications

6. **Metrics and Evaluation** → `docs/matrix/06_metrics_requirements.md`
   - Metric definitions (METRIC-*)
   - Attack-balanced recall
   - Edge cases and NA handling

7. **Experiments** → `docs/matrix/07_experiment_requirements.md`
   - Synthetic experiments (EXPERIMENT-*, S1-S6)
   - Real data experiments (R1-R14)
   - Randomness registry

8. **Implementation and Artifacts** → `docs/matrix/08_implementation_requirements.md`
   - Package structure (IMPLEMENT-*)
   - Artifact schemas (ARTIFACT-*)
   - Configuration (CONFIG-*)
   - CLI contract (CLI-*)
   - Python API (API-*)

9. **Testing and Validation** → `docs/matrix/09_testing_requirements.md`
   - Unit tests (TEST-*)
   - Leakage tests
   - Determinism requirements

10. **Failure States and Claims** → `docs/matrix/10_failure_claims.md`
    - Failure registry (FAILURE-*)
    - Claim strength gates (CLAIM-*)
    - Required tables and figures (TABLE-*, FIGURE-*)

---

## Summary Statistics

| Category | Total Requirements | Current Status |
|----------|-------------------|----------------|
| Core | ~50 | 66 IMPLEMENTED & VERIFIED |
| Statistical Core | ~40 | 48 IMPLEMENTED & VERIFIED |
| Dataset | ~80 | PENDING |
| Training | ~60 | PENDING |
| Baselines | ~30 | PENDING |
| Metrics | ~40 | PENDING |
| Experiments | ~50 | PENDING |
| Implementation | ~40 | PENDING |
| Testing | ~30 | PENDING |
| Failure/Claims | ~60 | PENDING |
| **Total** | **~420** | **114 IMPLEMENTED & VERIFIED** |

---

## Current Implementation Status

**Repository State:** Significant implementation progress

- Core requirements: COMPLETE (66/66 implemented and verified)
- Statistical core: COMPLETE (48/48 implemented and verified)
- Matrix extraction: IN PROGRESS (2 matrix files created)
- Implementation status: Core modules, CLI, data infrastructure, experiments registry COMPLETE
- CLI commands: FIXED and WORKING
- Raw data symlink: CREATED and VERIFIED
- Synthetic experiments: S1 and S6 COMPLETED and VERIFIED

---

## Next Steps

1. Complete matrix extraction into individual files
2. Perform Audit 1: Lossless roadmap coverage verification
3. Perform Audit 2: Scientific-contract consistency check
4. Perform Audit 3: Experimental and evidence completeness
5. Perform Audit 4: Implementability and verification mapping
6. Begin implementation following priority order (Section 8 of prompt.md)

---

## Quick Reference: Critical Requirements

### Must-Have Before Any Implementation
- [ ] GLOBAL-001 to GLOBAL-005: Identity constants
- [ ] PROTOCOL-001 to PROTOCOL-007: Core parameters
- [ ] FORMULA-001 to FORMULA-012: Reference threshold formulas
- [ ] GATE-A-001 to GATE-A-012: Gate A core
- [ ] GATE-B-001 to GATE-B-015: Gate B core
- [ ] STATE-001 to STATE-006: State machine

### Must-Have Before Data Loading
- [ ] DATASET-001 to DATASET-004: N-BaIoT identity
- [ ] SPLIT-001 to SPLIT-007: N-BaIoT partition
- [ ] INVARIANT-001 to INVARIANT-015: N-BaIoT integrity

### Must-Have Before Training
- [ ] TRAIN-001 to TRAIN-034: Architecture and hyperparameters
- [ ] TRAIN-035 to TRAIN-044: Training state machine

### Must-Have Before Policy Evaluation
- [ ] BASELINE-001 to BASELINE-014: Baseline definitions
- [ ] METRIC-001 to METRIC-015: Metric definitions

### Must-Have Before Verification
- [ ] TEST-001 to TEST-022: Normative unit tests
- [ ] TEST-013 to TEST-021: Leakage and integrity tests
- [ ] CLI-001 to CLI-016: CLI commands
- [ ] CLAIM-001 to CLAIM-009: Claim gates

---

## Matrix Maintenance Rules

1. **Update Frequency:** After each major implementation phase
2. **Versioning:** Increment version on significant changes
3. **Audit Trail:** All changes logged in AUDIT_LOG.md
4. **Reverse Mapping:** Verify repo -> roadmap after each update
5. **Status Updates:** Update status as implementation progresses

---

## See Also

- `docs/.tmp/CURRENT_STATE.md` - Current working state
- `docs/.tmp/PROGRESS.md` - Implementation progress tracking
- `docs/.tmp/AUDIT_LOG.md` - Audit and decision log
- `docs/.tmp/VALIDATION_LOG.md` - Validation results
- `docs/.tmp/BLOCKERS.md` - Blockers and issues
- `docs/.tmp/NEXT_ACTION.md` - Immediate next steps

---

*Matrix files will be created as the extraction progresses. This index will be updated with links as files are created.*
