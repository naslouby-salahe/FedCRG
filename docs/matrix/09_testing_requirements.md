# FedCRG Testing and Validation Requirements Matrix

**File:** `09_testing_requirements.md`  
**Version:** 1.0  
**Created:** 2026-08-12  
**Status:** Initial extraction from Sections 14.2-14.3, 12, 13  
**Source:** `docs/FedCRG Roadmap.md` v2.0

---

## Overview

This file contains all testing and validation requirements extracted from Sections 14.2 (Normative unit tests), 14.3 (Leakage and integrity tests), and related validation sections.

---

## Normative Unit Tests (Section 14.2)

### Gate A Exact Value Tests

| ID | Requirement | Section | Expected | Status | Implementation | Verification |
|---|---|---|---|---|---|---|
| TEST-GATE-A-001 | Gate A with alpha=0.01,rho=0.5,gamma=0.95,n=1415 MUST return NOT_READY | 1729 | NOT_READY | IMPLEMENTED | gate_a.py, test_gate_a_exact.py | PASSED |
| TEST-GATE-A-002 | Gate A with alpha=0.01,rho=0.5,gamma=0.95,n=1416 MUST return READY; r*=1404; P=0.9500045311 +/-1e-10 | 1730 | READY, r*=1404, P=0.9500045311 | IMPLEMENTED | gate_a.py, test_gate_a_exact.py | PASSED |
| TEST-GATE-A-003 | Gate A with n=1500 MUST return READY; r*=1487; P=0.9573928914 +/-1e-10 | 1731 | READY, r*=1487, P=0.9573928914 | IMPLEMENTED | gate_a.py, test_gate_a_exact.py | PASSED |
| TEST-GATE-A-004 | Gate A with n=2000 MUST return READY; r*=1982; P=0.9805279151 +/-1e-10 | 1732 | READY, r*=1982, P=0.9805279151 | IMPLEMENTED | gate_a.py, test_gate_a_exact.py | PASSED |
| TEST-GATE-A-005 | All Gate A exact values MUST pass tolerance 1e-10 | 347, 1728 | Absolute error <= 1e-10 | IMPLEMENTED | test_gate_a_exact.py | PASSED |
| TEST-GATE-A-006 | Gate A Beta-CDF calculations MUST use float64 | 343-344 | float64 | IMPLEMENTED | gate_a.py | PASSED |

### Gate B Exact Value Tests

| ID | Requirement | Section | Expected | Status | Implementation | Verification |
|---|---|---|---|---|---|---|
| TEST-GATE-B-001 | Gate B with n=736,x=0 MUST return LOW_MISMATCH | 1733 | LOW_MISMATCH | IMPLEMENTED | gate_b.py, test_gate_b_exact.py | PASSED |
| TEST-GATE-B-002 | Gate B with n=736,x=1 MUST return NO_MATERIAL_MISMATCH_DEMONSTRATED | 1734 | NO_MATERIAL_MISMATCH_DEMONSTRATED | IMPLEMENTED | gate_b.py, test_gate_b_exact.py | PASSED |
| TEST-GATE-B-003 | Gate B with n=1000,x=0 MUST return LOW_MISMATCH | 1735 | LOW_MISMATCH | IMPLEMENTED | gate_b.py, test_gate_b_exact.py | PASSED |
| TEST-GATE-B-004 | Gate B with n=1000,x=1 MUST return NO_MATERIAL_MISMATCH_DEMONSTRATED | 1736 | NO_MATERIAL_MISMATCH_DEMONSTRATED | IMPLEMENTED | gate_b.py, test_gate_b_exact.py | PASSED |
| TEST-GATE-B-005 | Gate B with n=1000,x=23 MUST return NO_MATERIAL_MISMATCH_DEMONSTRATED | 1737 | NO_MATERIAL_MISMATCH_DEMONSTRATED | IMPLEMENTED | gate_b.py, test_gate_b_exact.py | PASSED |
| TEST-GATE-B-006 | Gate B with n=1000,x=24 MUST return HIGH_MISMATCH | 1738 | HIGH_MISMATCH | IMPLEMENTED | gate_b.py, test_gate_b_exact.py | PASSED |
| TEST-GATE-B-007 | Gate B with n=1500,x=2 MUST return LOW_MISMATCH | 1739 | LOW_MISMATCH | IMPLEMENTED | gate_b.py, test_gate_b_exact.py | PASSED |
| TEST-GATE-B-008 | Gate B with n=1500,x=3 MUST return NO_MATERIAL_MISMATCH_DEMONSTRATED | 1740 | NO_MATERIAL_MISMATCH_DEMONSTRATED | IMPLEMENTED | gate_b.py, test_gate_b_exact.py | PASSED |
| TEST-GATE-B-009 | Gate B with n=1500,x=32 MUST return NO_MATERIAL_MISMATCH_DEMONSTRATED | 1741 | NO_MATERIAL_MISMATCH_DEMONSTRATED | IMPLEMENTED | gate_b.py, test_gate_b_exact.py | PASSED |
| TEST-GATE-B-010 | Gate B with n=1500,x=33 MUST return HIGH_MISMATCH | 1742 | HIGH_MISMATCH | IMPLEMENTED | gate_b.py, test_gate_b_exact.py | PASSED |
| TEST-GATE-B-011 | Gate B with n=3000,x=7 MUST return LOW_MISMATCH | 1743 | LOW_MISMATCH | IMPLEMENTED | gate_b.py, test_gate_b_exact.py | PASSED |
| TEST-GATE-B-012 | Gate B with n=3000,x=8 MUST return NO_MATERIAL_MISMATCH_DEMONSTRATED | 1744 | NO_MATERIAL_MISMATCH_DEMONSTRATED | IMPLEMENTED | gate_b.py, test_gate_b_exact.py | PASSED |
| TEST-GATE-B-013 | Gate B with n=3000,x=58 MUST return NO_MATERIAL_MISMATCH_DEMONSTRATED | 1745 | NO_MATERIAL_MISMATCH_DEMONSTRATED | IMPLEMENTED | gate_b.py, test_gate_b_exact.py | PASSED |
| TEST-GATE-B-014 | Gate B with n=3000,x=59 MUST return HIGH_MISMATCH | 1746 | HIGH_MISMATCH | IMPLEMENTED | gate_b.py, test_gate_b_exact.py | PASSED |
| TEST-GATE-B-015 | Gate B Clopper-Pearson intervals MUST use float64 | 1780 | float64 | IMPLEMENTED | gate_b.py | PASSED |

### Reference Threshold Tests

| ID | Requirement | Section | Expected | Status | Implementation | Verification |
|---|---|---|---|---|---|---|
| TEST-REF-001 | Reference rank with K=9,R_k=500,alpha=0.01 MUST return N_R=4500;q_ref=4456 | 1747 | N_R=4500, q_ref=4456 | IMPLEMENTED | reference.py, test_reference.py | PASSED |

### Classification Rule Tests

| ID | Requirement | Section | Expected | Status | Implementation | Verification |
|---|---|---|---|---|---|---|
| TEST-CLASS-001 | Classification rule: score==threshold MUST return BENIGN (threshold rule is score > threshold) | 1748 | BENIGN | IMPLEMENTED | states.py, test_states.py | PASSED |

### DIAD Attack Allocator Tests

| ID | Requirement | Section | Expected | Status | Implementation | Verification |
|---|---|---|---|---|---|---|
| TEST-DIAD-ALLOC-001 | DIAD attack allocator with lexical categories A/B/C, dmax=[200,200,200], budget=500 MUST return dev=[167,167,166] | 1749 | dev=[167,167,166] | PENDING | diad.py | Needs implementation |
| TEST-DIAD-ALLOC-002 | DIAD attack allocator with lexical categories A/B/C, dmax=[0,50,900], budget=500 MUST return dev=[0,50,450] | 1750 | dev=[0,50,450] | PENDING | diad.py | Needs implementation |

### DIAD Attack Reserve Tests

| ID | Requirement | Section | Expected | Status | Implementation | Verification |
|---|---|---|---|---|---|---|
| TEST-DIAD-RESERVE-001 | DIAD attack reserve for any present category a MUST have A_test_count[a] >= min(100,n_ka) | 1751 | A_test_count[a] >= min(100,n_ka) | PENDING | diad.py | Needs implementation |

### Policy Invariance Tests

| ID | Requirement | Section | Expected | Status | Implementation | Verification |
|---|---|---|---|---|---|---|
| TEST-POLICY-INVAR-001 | Policy invariance: same score cache, different threshold policy MUST have AUROC difference <=1e-12 and AUPRC difference <=1e-12 | 1752 | AUROC diff <=1e-12, AUPRC diff <=1e-12 | PENDING | test_score_invariance.py | Needs implementation |

---

## Leakage and Integrity Tests (Section 14.3)

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| TEST-LEAKAGE-001 | MUST assert pairwise-empty row_id intersections among T, calibration reservoir, B_test, A_dev, and A_test | 1756 | PENDING | test_data_disjointness.py | Needs full implementation |
| TEST-LEAKAGE-002 | MUST assert within each calibration seed, R, G, C, and supervised-comparator benign guard are pairwise disjoint | 1756 | PENDING | test_data_disjointness.py | Needs full implementation |
| TEST-LEAKAGE-003 | MUST assert FedCRGPolicy.fit() accepts only arrays of benign scores and scalar protocol parameters; has no label argument | 1758 | IMPLEMENTED | fedcrg.py, baselines/ | API validation (conceptual) |
| TEST-LEAKAGE-004 | MUST assert no path containing A_dev or A_test is opened by FedCRG fitting code; use test fixture that raises immediately if accessed | 1760-1761 | PENDING | test_no_label_leakage.py | Needs full implementation |
| TEST-LEAKAGE-005 | MUST assert scaler/imputer objects are fitted exclusively on T_k rows; serialize their fit-row hashes | 1762 | PENDING | Needs hash serialization | Needs implementation |
| TEST-LEAKAGE-006 | MUST assert selected feature names exactly match N-BaIoT 115 schema or DIAD allowlist after finite-value audit | 1764 | PENDING | test_data_disjointness.py | Needs full implementation |
| TEST-LEAKAGE-007 | MUST assert all policies read identical score_cache hash for given dataset/model seed | 1766 | PENDING | test_score_invariance.py | Needs full implementation |
| TEST-LEAKAGE-008 | MUST assert final test labels are loaded only inside evaluation functions after thresholds serialized | 1768 | PENDING | Needs evaluation validation | Needs implementation |
| TEST-LEAKAGE-009 | MUST fail run if any metric is NaN/inf | 1770 | PENDING | Needs error handling | Needs implementation |
| TEST-LEAKAGE-010 | MUST fail run if any role count differs from protocol | 1770 | PENDING | Needs validation | Needs implementation |
| TEST-LEAKAGE-011 | MUST fail run if any client disappears after outcome computation | 1770 | PENDING | Needs validation | Needs implementation |
| TEST-LEAKAGE-012 | MUST fail run if any config hash differs from pre-registered file | 1771 | PENDING | Needs hash validation | Needs implementation |

---

## Data Disjointness Tests

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| TEST-DISJOINT-001 | MUST verify T, calibration reservoir, B_test, A_dev, A_test are pairwise disjoint by row_id | 648-650 | PENDING | test_data_disjointness.py | Needs implementation |
| TEST-DISJOINT-002 | MUST verify within each calibration seed, R, G, C, guard are pairwise disjoint | 650-651 | PENDING | test_data_disjointness.py | Needs implementation |
| TEST-DISJOINT-003 | MUST verify R, G, C union equals 6,000-row reservoir for N-BaIoT | 651 | PENDING | test_data_disjointness.py | Needs implementation |
| TEST-DISJOINT-004 | MUST verify DIAD within each calibration seed, R, G, C, guard are pairwise disjoint | 801-802 | PENDING | test_data_disjointness.py | Needs implementation |
| TEST-DISJOINT-005 | MUST verify DIAD R, G, C exactly cover 3,800-record reservoir | 802 | PENDING | test_data_disjointness.py | Needs implementation |

---

## Label Leakage Tests

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| TEST-NO-LEAKAGE-001 | FedCRG fitting code MUST NOT access attack label/dev/test paths | 630, 1759-1761 | PENDING | test_no_label_leakage.py | Needs full implementation |
| TEST-NO-LEAKAGE-002 | B7-B9 code MUST be isolated from FedCRG fitting code | 630, 1218 | PENDING | Needs validation | Needs implementation |
| TEST-NO-LEAKAGE-003 | MUST verify no import of A_dev or A_test modules by FedCRG code | 1760 | PENDING | test_no_label_leakage.py | Needs implementation |
| TEST-NO-LEAKAGE-004 | MUST verify FedCRG receives only benign scores, never attack data | 820 | PENDING | test_no_label_leakage.py | Needs implementation |

---

## Score Invariance Tests

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| TEST-SCORE-INVAR-001 | All policies MUST consume identical immutable score cache | 1766 | PENDING | test_score_invariance.py | Needs full implementation |
| TEST-SCORE-INVAR-002 | AUROC must be identical across threshold policies on identical cached scores to tolerance 1e-12 | 1326-1327, 1752 | PENDING | test_score_invariance.py | Needs implementation |
| TEST-SCORE-INVAR-003 | AUPRC must be identical across threshold policies on identical cached scores to tolerance 1e-12 | 1326-1327, 1752 | PENDING | test_score_invariance.py | Needs implementation |
| TEST-SCORE-INVAR-004 | Score cache hash MUST be immutable across policies | 1766 | PENDING | test_score_invariance.py | Needs implementation |

---

## Metrics Tests

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| TEST-METRICS-001 | MUST test all metric calculations per Section 10 | 1704 | PENDING | test_metrics.py | Partial implementation |
| TEST-METRICS-002 | MUST verify MEBE calculation: mean client distance outside band | 1341 | PENDING | test_metrics.py | Needs implementation |
| TEST-METRICS-003 | MUST verify HighExcess calculation: max(0, max_k(FPR_k - b)) | 1345 | PENDING | test_metrics.py | Needs implementation |
| TEST-METRICS-004 | MUST verify BandViolationRate calculation: mean client band violation indicator | 1349-1350 | PENDING | test_metrics.py | Needs implementation |
| TEST-METRICS-005 | MUST verify MAFE calculation: mean |FPR_k - alpha| | 1354 | PENDING | test_metrics.py | Needs implementation |
| TEST-METRICS-006 | MUST verify ABMacroTPR calculation per Section 10.1 | 1368-1370 | PENDING | test_metrics.py | Needs implementation |
| TEST-METRICS-007 | MUST verify FPR = FP/(FP+TN) | 1327 | PENDING | test_metrics.py | Needs implementation |
| TEST-METRICS-008 | MUST verify TPR = TP/(TP+FN) | 1328 | PENDING | test_metrics.py | Needs implementation |
| TEST-METRICS-009 | MUST verify BandError = max(a-FPR, 0, FPR-b) | 1333-1336 | PENDING | test_metrics.py | Needs implementation |
| TEST-METRICS-010 | MUST verify AUROC/AUPRC computed from raw cached scores, never from thresholded decisions | 2003 | PENDING | test_metrics.py | Needs implementation |
| TEST-METRICS-011 | MUST handle edge cases: precision NA when TP+FP=0, F1 NA when precision/recall undefined | 1898-1899 | PENDING | test_metrics.py | Needs implementation |
| TEST-METRICS-012 | MUST handle ABMacroTPR NA when any client has no defined attack-group TPR | 1900 | PENDING | test_metrics.py | Needs implementation |

---

## Reproducibility Tests

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| TEST-REPRO-001 | MUST verify deterministic runs produce identical results | 1784, 1890 | PENDING | test_reproducibility.py | Needs full implementation |
| TEST-REPRO-002 | MUST verify repeated deterministic run produces different artifact hash fails with NONDETERMINISTIC_PARITY_FAIL | 1890 | PENDING | test_reproducibility.py | Needs implementation |
| TEST-REPRO-003 | MUST verify clean checkout reproduces all threshold/metric/figure artifacts from immutable score caches | 1784 | PENDING | test_reproducibility.py | Needs implementation |
| TEST-REPRO-004 | MUST verify every generated table/figure reproducible from immutable artifacts without retraining | 2121 | PENDING | test_reproducibility.py | Needs implementation |

---

## Synthetic Experiment Validation Tests

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| TEST-SYNTH-001 | MUST verify S1 Monte-Carlo coverage agreement: abs(p_hat-P_r) <= max(0.005, 4*sqrt(P_r*(1-P_r)/10000)) | 102 | PENDING | Needs validation | S1 validation |
| TEST-SYNTH-002 | MUST verify S6 Gate-B exact power values match Section 15 and Appendix G.3 | 1953-1963, 2484-2496 | IMPLEMENTED | synthetic.py | PASSED |
| TEST-SYNTH-003 | MUST verify S6 boundary table values | 1749-1750 | IMPLEMENTED | synthetic.py | PASSED |
| TEST-SYNTH-004 | MUST verify all synthetic experiment trial counts | 1493-1498 | IMPLEMENTED | synthetic.py | Trial counting |

---

## Statistical Analysis Validation

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| TEST-STAT-001 | MUST verify paired model-seed procedural bootstrap with 10,000 replicates and seed 424242 | 1563 | PENDING | Needs bootstrap implementation | Statistical validation |
| TEST-STAT-002 | MUST verify training-seed variability conditional on fixed federation and calibration split | 1564 | PENDING | Needs validation | Statistical validation |
| TEST-STAT-003 | MUST verify client-level FPR binomial reference intervals use exact Clopper-Pearson | 1595-1596 | PENDING | Needs validation | Statistical validation |
| TEST-STAT-004 | MUST verify Bonferroni and Holm directional Gate-B sensitivity procedures | 536-543, 1612 | PENDING | Needs implementation | Statistical validation |
| TEST-STAT-005 | MUST verify dynamic n_G_min(a,gamma_B) calculation | 475-490, 1742 | IMPLEMENTED | gate_b.py | PASSED |
| TEST-STAT-006 | MUST verify Gate-B cutoff tables regenerated from exact CP implementation | 2421 | PENDING | Needs validation | Statistical validation |

---

## Summary Statistics

- **Normative Unit Tests (Gate A/B):** 19 requirements
  - Implemented: 19
  - Verified: 19
  - Status: 100% complete

- **Leakage and Integrity Tests:** 12 requirements
  - Implemented: 1
  - Pending: 11
  - Status: ~8% complete

- **Data Disjointness Tests:** 5 requirements
  - Implemented: 0
  - Pending: 5
  - Status: 0% complete

- **Label Leakage Tests:** 4 requirements
  - Implemented: 0
  - Pending: 4
  - Status: 0% complete

- **Score Invariance Tests:** 4 requirements
  - Implemented: 0
  - Pending: 4
  - Status: 0% complete

- **Metrics Tests:** 12 requirements
  - Implemented: 0
  - Pending: 12
  - Status: 0% complete

- **Reproducibility Tests:** 4 requirements
  - Implemented: 0
  - Pending: 4
  - Status: 0% complete

- **Synthetic Validation Tests:** 4 requirements
  - Implemented: 2
  - Pending: 2
  - Status: 50% complete

- **Statistical Analysis Tests:** 6 requirements
  - Implemented: 1
  - Pending: 5
  - Status: ~17% complete

- **Total Testing Requirements:** ~70
- **Overall Testing Status:** ~25% complete

---

## Critical Gaps

1. **Data Disjointness:** Complete test implementation needed
2. **Label Leakage:** Complete isolation validation needed
3. **Score Invariance:** Full invariance testing across policies
4. **Metrics Tests:** Comprehensive metric calculation validation
5. **Reproducibility Tests:** Full deterministic validation
6. **Statistical Tests:** Bootstrap and sensitivity procedure validation

---

## Test File Status

| Test File | Status | Coverage |
|---|---|---|
| test_gate_a_exact.py | IMPLEMENTED | 100% of Gate A exact values |
| test_gate_b_exact.py | IMPLEMENTED | 100% of Gate B exact values |
| test_data_disjointness.py | PENDING | 0% - needs implementation |
| test_no_label_leakage.py | PENDING | 0% - needs implementation |
| test_score_invariance.py | PENDING | 0% - needs implementation |
| test_metrics.py | PENDING | 0% - needs implementation |
| test_reproducibility.py | PENDING | 0% - needs implementation |

---

## Cross-References

- Core mathematical formulas: See `02_statistical_core.md` (GATE-A-*, GATE-B-*)
- Dataset specifications: See `03_dataset_requirements.md` (DATASET-*)
- Experiment registry: See `07_experiment_requirements.md` (EXPERIMENT-*)
- Implementation requirements: See `08_implementation_requirements.md` (INTEGRITY-*)
- Failure states: See `10_failure_claims.md` (FAILURE-*)

---

## File Maintenance

- **Created:** 2026-08-12
- **Last Updated:** 2026-08-12
- **Version:** 1.0
- **Status:** Initial extraction from Sections 14.2-14.3
- **Next Review:** After major testing implementation phases