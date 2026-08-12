# FedCRG Experiment Requirements Matrix

**File:** `07_experiment_requirements.md`  
**Version:** 1.0  
**Created:** 2026-08-12  
**Status:** Initial extraction from Section 11  
**Source:** `docs/FedCRG Roadmap.md` v2.0, Section 11 (Experiment Registry)

---

## Overview

This file contains all experiment-related requirements extracted from Section 11 of the FedCRG Roadmap. Experiments are divided into **Synthetic (S1-S6)** and **Real Data (R1-R14)** categories per Section 11.

---

## Synthetic Experiments (S1-S6)

### S1: IID Gate-A Theorem Validation

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| EXPERIMENT-S1-001 | S1: IID Gate-A theorem validation MUST run 4 distributions x 8 n_C values x 10,000 repetitions | 11, 1493 | PENDING | fedcrg/experiments/synthetic.py | Unit tests |
| EXPERIMENT-S1-002 | Distributions MUST include: Normal(0,1), LogNormal(0,1), Gamma(shape=2,scale=1), 0.9N(0,1)+0.1N(3,1) | 1419 | PENDING | synthetic.py | Formula parity |
| EXPERIMENT-S1-003 | n_C values MUST include: 500, 1000, 1400, 1415, 1416, 1500, 2000, 3000 | 1419 | PENDING | synthetic.py | Value validation |
| EXPERIMENT-S1-004 | alpha=0.01, rho=0.5, gamma_A=0.95 for S1 | 1419 | PENDING | synthetic.py | Config validation |
| EXPERIMENT-S1-005 | In every i.i.d.-continuous S1 cell, Monte-Carlo coverage must agree with exact Gate-A probability: abs(p_hat-P_r) <= max(0.005, 4*sqrt(P_r*(1-P_r)/10000)) | 102 | PENDING | synthetic.py | H1 validation |
| EXPERIMENT-S1-006 | S1 MUST verify Gate A exact values per Section 14.2 (tolerance 1e-10) | 347, 102 | IMPLEMENTED | gate_a.py:verify_gate_a_exact_values | PASSED |
| EXPERIMENT-S1-007 | S1 MUST verify exact values: n=1415->NOT_READY, n=1416->READY r*=1404 P=0.9500045311, n=1500->READY r*=1487 P=0.9573928914, n=2000->READY r*=1982 P=0.9805279151 | 357-360, 1729-1732 | IMPLEMENTED | gate_a.py:_EXPECTED_VALUES | PASSED |
| EXPERIMENT-S1-008 | S1 total trials: 4 distributions x 8 n_C values x 10,000 = 320,000 Monte-Carlo trials | 1493 | PENDING | synthetic.py | Trial counting |
| EXPERIMENT-S1-009 | S1 MUST use synthetic master seed 123456 | 1448 | PENDING | synthetic.py | Seed validation |

### S2: Target-FPR Sensitivity

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| EXPERIMENT-S2-001 | S2: Target-FPR sensitivity MUST run 3 non-primary alpha values x 3 n values x 4 distributions x 10,000 | 1420 | PENDING | synthetic.py | Unit tests |
| EXPERIMENT-S2-002 | alpha values MUST include: 0.005, 0.02, 0.05 | 1420 | PENDING | synthetic.py | Config validation |
| EXPERIMENT-S2-003 | For alpha=0.005, n values MUST include: 2860, 2861, 5722 | 1420 | PENDING | synthetic.py | Value validation |
| EXPERIMENT-S2-004 | For alpha=0.02, n values MUST include: 693, 694, 1388 | 1420 | PENDING | synthetic.py | Value validation |
| EXPERIMENT-S2-005 | For alpha=0.05, n values MUST include: 269, 270, 540 | 1420 | PENDING | synthetic.py | Value validation |
| EXPERIMENT-S2-006 | S2 MUST use rho=0.5, gamma_A=0.95 for all cells | 1420 | PENDING | synthetic.py | Config validation |
| EXPERIMENT-S2-007 | S2 total trials: 3 x 3 x 4 x 10,000 = 360,000 trials | 1494 | PENDING | synthetic.py | Trial counting |

### S3: Temporal Dependence Stress

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| EXPERIMENT-S3-001 | S3: Temporal-dependence stress MUST run 4 AR(1) phi x 3 n_C x 10,000 | 1422, 1971 | PENDING | synthetic.py | Unit tests |
| EXPERIMENT-S3-002 | AR(1) phi values MUST include: 0, 0.3, 0.6, 0.9 | 1422 | PENDING | synthetic.py | Value validation |
| EXPERIMENT-S3-003 | n_C values for S3 MUST include: 1416, 2000, 3000 | 1422 | PENDING | synthetic.py | Value validation |
| EXPERIMENT-S3-004 | S3 MUST use marginal N(0,1) distribution | 1422 | PENDING | synthetic.py | Distribution validation |
| EXPERIMENT-S3-005 | S3 MUST evaluate theoretical future marginal exceedance | 1971 | PENDING | synthetic.py | Formula validation |
| EXPERIMENT-S3-006 | S3 MUST plot realized in-band coverage vs phi | 1971 | PENDING | synthetic.py | Visualization |
| EXPERIMENT-S3-007 | S3 exact theorem claimed ONLY for phi=0 (independent) condition | 1971 | PENDING | synthetic.py | Documentation |
| EXPERIMENT-S3-008 | S3 total trials: 4 x 3 x 10,000 = 120,000 trials | 1495 | PENDING | synthetic.py | Trial counting |

### S4: Calibration-to-Test Shift

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| EXPERIMENT-S4-001 | S4: Calibration-to-test shift MUST run 5 mean shifts x 10,000 | 1423, 1975 | PENDING | synthetic.py | Unit tests |
| EXPERIMENT-S4-002 | Calibration scores MUST be N(0,1), n_C=2000 | 1423 | PENDING | synthetic.py | Distribution validation |
| EXPERIMENT-S4-003 | Future benign distributions MUST use N(mu,1) for mu={0, 0.10, 0.25, 0.50, 1.00} | 1423 | PENDING | synthetic.py | Value validation |
| EXPERIMENT-S4-004 | S4 MUST quantify how rapidly static contract fails when calibration no longer represents deployment | 1975 | PENDING | synthetic.py | Analysis |
| EXPERIMENT-S4-005 | S4 total trials: 5 x 10,000 = 50,000 trials | 1496 | PENDING | synthetic.py | Trial counting |

### S5: Calibration Contamination

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| EXPERIMENT-S5-001 | S5: Calibration contamination MUST run 6 rates x 2 directions x 10,000 | 1424, 1978 | PENDING | synthetic.py | Unit tests |
| EXPERIMENT-S5-002 | n_C=2000 for S5 | 1424 | PENDING | synthetic.py | Value validation |
| EXPERIMENT-S5-003 | Contamination q values MUST include: 0, 0.001, 0.005, 0.01, 0.02, 0.05 | 1424 | PENDING | synthetic.py | Value validation |
| EXPERIMENT-S5-004 | Contamination MUST use high-tail N(3,1) and low-tail N(-3,1) | 1424 | PENDING | synthetic.py | Distribution validation |
| EXPERIMENT-S5-005 | S5 MUST NOT claim contamination-robust unless future method explicitly addresses this threat | 1979 | PENDING | synthetic.py | Documentation |
| EXPERIMENT-S5-006 | S5 total trials: 6 x 2 x 10,000 = 120,000 trials | 1497 | PENDING | synthetic.py | Trial counting |

### S6: Gate-B Exact Power

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| EXPERIMENT-S6-001 | S6: Gate-B exact power MUST run 5 n_G x 9 true FPR values | 1424, 1498 | IMPLEMENTED | synthetic.py | COMPLETED |
| EXPERIMENT-S6-002 | n_G values MUST include: 736, 1000, 1500, 2000, 3000 | 1424 | IMPLEMENTED | synthetic.py | PASSED |
| EXPERIMENT-S6-003 | True FPR p values MUST include: 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03 | 1424 | IMPLEMENTED | synthetic.py | PASSED |
| EXPERIMENT-S6-004 | S6 MUST use exact binomial calculation, no Monte Carlo | 1424 | IMPLEMENTED | gate_b.py:compute_gate_b | PASSED |
| EXPERIMENT-S6-005 | S6 MUST compute low_mismatch_prob, high_mismatch_prob, none_prob for each cell | 1424 | IMPLEMENTED | synthetic.py | PASSED |
| EXPERIMENT-S6-006 | S6 MUST verify boundary table: n_g=736 -> low_x_max=0, high_x_min=19; n_g=1000 -> low_x_max=0, high_x_min=24; n_g=1500 -> low_x_max=2, high_x_min=33; n_g=2000 -> low_x_max=3, high_x_min=42; n_g=3000 -> low_x_max=7, high_x_min=59 | 1749-1750 | IMPLEMENTED | synthetic.py | PASSED |
| EXPERIMENT-S6-007 | S6 MUST use alpha=0.01, rho=0.5, gamma_b=0.95, a=0.005, b=0.015, seed=123456 | 1424, 1448 | IMPLEMENTED | synthetic.py | PASSED |
| EXPERIMENT-S6-008 | S6 total: 5 x 9 = 45 cells, zero Monte-Carlo trials | 1498 | IMPLEMENTED | synthetic.py | PASSED |
| EXPERIMENT-S6-009 | S6 MUST verify exact power values per Section 15 and Appendix G.3 | 1953-1963, 2484-2496 | IMPLEMENTED | synthetic.py | PASSED |

---

## Real Data Experiments (R1-R14)

### R1: N-BaIoT Primary

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| EXPERIMENT-R1-001 | R1: N-BaIoT primary MUST run 9 natural clients x 5 model seeds x 50 calibration seeds | 1425, 1458-1460 | PENDING | real_data.py | Integration test |
| EXPERIMENT-R1-002 | R1 MUST use alpha=0.01, rho=0.5, gamma_A=0.95, gamma_B=0.95 | 1425 | PENDING | real_data.py | Config validation |
| EXPERIMENT-R1-003 | R1 MUST use all mandatory policies | 1425 | PENDING | real_data.py | Policy registry |
| EXPERIMENT-R1-004 | R1 client count: 9 natural clients (nb01-nb09) | 556-578 | PENDING | nbaiot.py | Client validation |
| EXPERIMENT-R1-005 | R1 model seeds: 11, 22, 33, 44, 55 | 1425, 1444 | PENDING | real_data.py | Seed validation |
| EXPERIMENT-R1-006 | R1 calibration seeds: 1000-1049 (1000 named primary split) | 1445, 1425 | PENDING | real_data.py | Seed validation |
| EXPERIMENT-R1-007 | R1 MUST create 5 model checkpoints (30 rounds each) | 1457 | PENDING | fl/trainer.py | Training validation |
| EXPERIMENT-R1-008 | R1 policy cells: 5 model seeds x 50 calibration seeds x 12 policies x 9 clients = 27,000 client-policy cells | 1461 | PENDING | real_data.py | Cell counting |
| EXPERIMENT-R1-009 | R1 FedCRG decisions: 5 x 50 x 9 = 2,250 client state decisions | 1463 | PENDING | real_data.py | Decision counting |
| EXPERIMENT-R1-010 | R1 reference thresholds: 5 model seeds x 50 calibration seeds = 250 | 1464-1466 | PENDING | real_data.py | Reference counting |
| EXPERIMENT-R1-011 | R1 MUST compute all metrics per Section 10 for all policies | 1425 | PENDING | real_data.py | Metrics validation |
| EXPERIMENT-R1-012 | R1 MUST use deterministic cosine LR schedule | 956-967 | PENDING | fl/lr_schedule.py | Schedule validation |
| EXPERIMENT-R1-013 | R1 MUST use 30 rounds x 120 local epochs for N-BaIoT | 944-945 | PENDING | fl/trainer.py | Training validation |
| EXPERIMENT-R1-014 | R1 MUST freeze final global weights after round 29 | 1065 | PENDING | fl/trainer.py | Finalization validation |
| EXPERIMENT-R1-015 | R1 MUST cache scores as float64 | 1068 | PENDING | scoring/cache.py | Score validation |

### R2: Gate-A Sample-Size Sweep

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| EXPERIMENT-R2-001 | R2: Gate-A sample-size sweep MUST use n_C={500, 1000, 1400, 1415, 1416, 1500, 2000} | 1426 | PENDING | real_data.py | Value validation |
| EXPERIMENT-R2-002 | R2 MUST use n_G=3000 fixed | 1426 | PENDING | real_data.py | Config validation |
| EXPERIMENT-R2-003 | R2 MUST use same frozen scores as R1 | 1426 | PENDING | real_data.py | Score reuse |
| EXPERIMENT-R2-004 | R2 MUST show Gate-A readiness changes across n_C values | 1426 | PENDING | real_data.py | Readiness analysis |

### R3: Gate-B Sample-Size Sweep

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| EXPERIMENT-R3-001 | R3: Gate-B sample-size sweep MUST use n_G={736, 1000, 1500, 2000, 3000} | 1427 | PENDING | real_data.py | Value validation |
| EXPERIMENT-R3-002 | R3 MUST use n_C=2000 fixed | 1427 | PENDING | real_data.py | Config validation |
| EXPERIMENT-R3-003 | R3 MUST use same frozen scores as R1 | 1427 | PENDING | real_data.py | Score reuse |
| EXPERIMENT-R3-004 | R3 MUST show Gate-B mismatch evidence changes across n_G values | 1427 | PENDING | real_data.py | Mismatch analysis |

### R4: Operating Tolerance Sensitivity

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| EXPERIMENT-R4-001 | R4: Operating tolerance sensitivity MUST use rho={0.25, 0.50, 1.00} | 1428 | PENDING | real_data.py | Value validation |
| EXPERIMENT-R4-002 | R4 MUST use alpha=0.01 | 1428 | PENDING | real_data.py | Config validation |
| EXPERIMENT-R4-003 | R4 MUST show data cost of narrower operational contracts | 1428 | PENDING | real_data.py | Cost analysis |

### R5: Target-FPR Sensitivity

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| EXPERIMENT-R5-001 | R5: Target-FPR sensitivity MUST use alpha={0.005, 0.01, 0.02, 0.05} | 1429 | PENDING | real_data.py | Value validation |
| EXPERIMENT-R5-002 | R5 MUST use rho=0.50 | 1429 | PENDING | real_data.py | Config validation |
| EXPERIMENT-R5-003 | R5 MUST show Gate A may correctly declare insufficient evidence at alpha=0.005 with n_C=2000 | 1429 | PENDING | real_data.py | Insufficient evidence validation |

### R6: Assurance Sensitivity

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| EXPERIMENT-R6-001 | R6: Assurance sensitivity MUST use gamma_A={0.90, 0.95, 0.99} | 1430 | PENDING | real_data.py | Value validation |
| EXPERIMENT-R6-002 | R6 MUST use gamma_B=0.95 | 1430 | PENDING | real_data.py | Config validation |
| EXPERIMENT-R6-003 | R6 MUST use primary band | 1430 | PENDING | real_data.py | Band validation |

### R7: Multiplicity Sensitivity

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| EXPERIMENT-R7-001 | R7: Multiplicity sensitivity MUST use gamma_A=1-.05/9 for N-BaIoT | 1431, 531 | PENDING | real_data.py | Value validation |
| EXPERIMENT-R7-002 | R7 MUST compute Gate-B Bonferroni sensitivity | 1431, 536 | PENDING | real_data.py | Bonferroni implementation |
| EXPERIMENT-R7-003 | R7 MUST compute Gate-B Holm directional-exact sensitivity | 1431, 538-543 | PENDING | real_data.py | Holm implementation |
| EXPERIMENT-R7-004 | R7 MUST use per-client confidence 1-0.05/K for Bonferroni | 537 | PENDING | real_data.py | Confidence calculation |
| EXPERIMENT-R7-005 | R7 MUST show N-BaIoT derived Bonferroni cutoff: K=9, n_G=3000, per-client confidence 0.994444 yields LOW_MISMATCH for x<=5 and HIGH_MISMATCH for x>=65 | 547 | PENDING | real_data.py | Cutoff validation |
| EXPERIMENT-R7-006 | R7 primary unadjusted cutoffs: x<=7 and x>=59 | 547 | PENDING | real_data.py | Cutoff validation |
| EXPERIMENT-R7-007 | R7 MUST report how many primary mismatch declarations survive each sensitivity | 545 | PENDING | real_data.py | Reporting |
| EXPERIMENT-R7-008 | R7 MUST NOT reinterpret non-surviving declaration as proof reference is in-band | 545 | PENDING | real_data.py | Documentation |
| EXPERIMENT-R7-009 | R7 MUST stop with GATE_B_DIRECTION_CONTRADICTION if both tails rejected for same cell | 543 | PENDING | real_data.py | Error handling |

### R8: Source-Order Test Segmentation

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| EXPERIMENT-R8-001 | R8: Source-order test segmentation MUST use 5 equal source-order benign-test blocks per client | 1432 | PENDING | real_data.py | Block segmentation |
| EXPERIMENT-R8-002 | R8 MUST report block-wise FPR without re-fitting | 1432 | PENDING | real_data.py | FPR reporting |
| EXPERIMENT-R8-003 | R8 MUST call this temporal drift only when dataset provenance verifies chronological order | 1432 | PENDING | real_data.py | Chronology validation |
| EXPERIMENT-R8-004 | R8 MUST NOT use "chronological" without verified time provenance | 1432 | PENDING | real_data.py | Documentation |

### R9: Real-Score Calibration Contamination

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| EXPERIMENT-R9-001 | R9: Real-score calibration contamination MUST use q={0.001, 0.005, 0.01, 0.02, 0.05} | 1433 | PENDING | real_data.py | Value validation |
| EXPERIMENT-R9-002 | R9 MUST replace q fraction of benign C/G with A_dev scores | 1979 | PENDING | real_data.py | Contamination implementation |
| EXPERIMENT-R9-003 | R9 MUST keep detector frozen | 1979 | PENDING | real_data.py | Detector freeze validation |
| EXPERIMENT-R9-004 | R9 MUST NOT claim contamination-robust | 1979 | PENDING | real_data.py | Documentation |

### R10: CIC IoT-DIAD External Replication

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| EXPERIMENT-R10-001 | R10: CIC IoT-DIAD external replication MUST run all eligible natural clients x 5 model seeds x 20 calibration seeds | 1434, 1475-1480 | PENDING | real_data.py | Integration test |
| EXPERIMENT-R10-002 | R10 MUST use same alpha/rho/confidence as N-BaIoT primary | 1434 | PENDING | real_data.py | Config validation |
| EXPERIMENT-R10-003 | R10 MUST use dataset-specific fixed data counts per Section 7.2 | 1434 | PENDING | real_data.py | Count validation |
| EXPERIMENT-R10-004 | R10 DIAD policy cells: 5 x 20 x 12 x K_D = 1200*K_D | 1480-1481 | PENDING | real_data.py | Cell counting |
| EXPERIMENT-R10-005 | R10 FedCRG decisions: 5 x 20 x K_D = 100*K_D | 1484 | PENDING | real_data.py | Decision counting |
| EXPERIMENT-R10-006 | R10 reference thresholds: 5 x 20 = 100 | 1485 | PENDING | real_data.py | Reference counting |
| EXPERIMENT-R10-007 | R10 MUST use 86-feature allowlist per Section 7.3 | 1434 | PENDING | diad.py | Feature validation |
| EXPERIMENT-R10-008 | R10 MUST use DIAD eligibility rule per Section 7.2.4 | 1434, 772-775 | PENDING | diad.py | Eligibility validation |
| EXPERIMENT-R10-009 | R10 MUST emit diad_eligibility.json with exact counts and exclusion reasons | 782-784 | PENDING | diad.py | Manifest generation |
| EXPERIMENT-R10-010 | R10 MUST use per-client hash-seeded PCG64 for calibration permutation | 707-710 | PENDING | diad.py | Seed validation |
| EXPERIMENT-R10-011 | R10 DIAD training: 30 rounds x 20 local epochs | 945, 1015 | PENDING | fl/trainer.py | Training validation |
| EXPERIMENT-R10-012 | If K_D < 10, R10 MUST be labeled EXTERNAL_DATASET_INSUFFICIENT_CLIENTS | 811-813 | PENDING | real_data.py | Error handling |
| EXPERIMENT-R10-013 | R10 MUST NOT claim confirmatory two-dataset replication if K_D < 10 | 813 | PENDING | real_data.py | Claim discipline |

### R11: Second-Detector Check

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| EXPERIMENT-R11-001 | R11: Second-detector check MUST use Federated Deep-SVDD | 1136, 1435 | PENDING | models/deep_svdd.py | Model validation |
| EXPERIMENT-R11-002 | R11 MUST run on N-BaIoT | 1435 | PENDING | real_data.py | Dataset validation |
| EXPERIMENT-R11-003 | R11 model seeds: 11, 22, 33 | 1155, 1435 | PENDING | real_data.py | Seed validation |
| EXPERIMENT-R11-004 | R11 calibration seeds: 1000-1009 (1000 named split, rest sensitivity) | 1156, 1435 | PENDING | real_data.py | Seed validation |
| EXPERIMENT-R11-005 | R11 MUST only use policies: B1, B2, B5, FEDCRG | 1435 | PENDING | real_data.py | Policy validation |
| EXPERIMENT-R11-006 | R11 Deep-SVDD encoder: 115-64-32, tanh, biases disabled | 1143, 1144 | PENDING | models/deep_svdd.py | Architecture validation |
| EXPERIMENT-R11-007 | R11 Deep-SVDD embedding dimension: 32 | 1144 | PENDING | models/deep_svdd.py | Dimension validation |
| EXPERIMENT-R11-008 | R11 Deep-SVDD center initialization: initialize encoder from model seed; each client computes mean embedding on T_k; server equal-averages client means; center frozen | 1145-1146 | PENDING | models/deep_svdd.py | Center initialization |
| EXPERIMENT-R11-009 | R11 Deep-SVDD loss: mean squared distance to center | 1146 | PENDING | models/deep_svdd.py | Loss validation |
| EXPERIMENT-R11-010 | R11 Deep-SVDD anomaly score: L2 distance to center | 1147 | PENDING | models/deep_svdd.py | Score validation |
| EXPERIMENT-R11-011 | R11 Deep-SVDD rounds: 30, local epochs: 20 | 1148-1149 | PENDING | fl/trainer.py | Training validation |
| EXPERIMENT-R11-012 | If qualitative pattern fails to replicate, scope claim to reconstruction-error systems | 1163-1165 | PENDING | real_data.py | Documentation |

### R12: Calibration-Role Source-Order Sensitivity

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| EXPERIMENT-R12-001 | R12: Calibration-role source-order sensitivity MUST run on N-BaIoT + DIAD | 1436 | PENDING | real_data.py | Dataset validation |
| EXPERIMENT-R12-002 | R12 MUST use fixed source-order roles, no within-reservoir permutation | 1436 | PENDING | real_data.py | Role validation |
| EXPERIMENT-R12-003 | R12 N-BaIoT source-order: first 500 R, next 3000 G, next 2000 C, final 500 supervised guard | 1436 | PENDING | real_data.py | Order validation |
| EXPERIMENT-R12-004 | R12 DIAD source-order: first 300 R, next 1500 G, next 1500 C, final 500 supervised guard | 1436 | PENDING | real_data.py | Order validation |
| EXPERIMENT-R12-005 | R12 MUST use same frozen detectors as primary | 1436 | PENDING | real_data.py | Detector reuse |
| EXPERIMENT-R12-006 | R12 MUST use same final tests as primary | 1436 | PENDING | real_data.py | Test reuse |
| EXPERIMENT-R12-007 | R12 MUST NOT make chronology claim without verified time provenance | 1436 | PENDING | real_data.py | Documentation |

### R13: Computational/Communication Overhead

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| EXPERIMENT-R13-001 | R13: Computational/communication overhead MUST run 100 warm-ups + 1000 measured repetitions per primitive on one CPU thread | 1437, 1813 | IMPLEMENTED | real_data.py | COMPLETED |
| EXPERIMENT-R13-002 | R13 MUST measure: reference construction, cached Gate-A rank lookup + order statistic, Gate B count/interval, full policy decision | 1437 | IMPLEMENTED | real_data.py | Primitive implementation |
| EXPERIMENT-R13-003 | R13 MUST report median/p95 wall time and peak memory | 1437 | IMPLEMENTED | real_data.py | Metrics reporting |
| EXPERIMENT-R13-004 | R13 MUST pin benchmark to one CPU thread | 1813 | IMPLEMENTED | real_data.py | Thread validation |
| EXPERIMENT-R13-005 | R13 MUST record CPU model/OS/Python/NumPy/SciPy versions | 1813 | IMPLEMENTED | real_data.py | Environment recording |
| EXPERIMENT-R13-006 | R13 MUST NOT claim hardware-independent latency | 1813 | IMPLEMENTED | real_data.py | Documentation |
| EXPERIMENT-R13-007 | R13 primitive payloads per Section 14.5.1: Reference R, Gate A, Gate B, GLOBAL-Q99-FULL/FEDDETECT-3SIGMA, LARIDI-STYLE-SS moments, LARIDI/SUP-F1 candidate evaluation | 1802-1810 | IMPLEMENTED | real_data.py | Payload validation |

### R14: DIAD Feature-Contract Sensitivity

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| EXPERIMENT-R14-001 | R14: DIAD feature-contract sensitivity MUST use one training-schema-derived numeric-safe feature representation | 1438, 1852 | PENDING | real_data.py | Feature validation |
| EXPERIMENT-R14-002 | R14 MUST start from official packet schema, remove direct identifiers, labels, IP/MAC addresses, ports, non-numeric fields | 1852 | PENDING | real_data.py | Schema processing |
| EXPERIMENT-R14-003 | R14 MUST retain column only if every eligible client has >=99.0% finite values for that column | 1852 | PENDING | real_data.py | Finite rate validation |
| EXPERIMENT-R14-004 | R14 feature list and dimension d_R14 MUST be frozen before any calibration/test score is evaluated | 1852 | PENDING | real_data.py | Freeze validation |
| EXPERIMENT-R14-005 | R14 MUST use deterministic symmetric AE with architecture: d -> floor(0.75d) -> floor(0.50d) -> floor(d/3) -> floor(0.25d) -> floor(d/3) -> floor(0.50d) -> floor(0.75d) -> d | 1855 | PENDING | models/autoencoder.py | Architecture validation |
| EXPERIMENT-R14-006 | R14 AE hidden width MUST be lower-bounded at 1 | 1855 | PENDING | models/autoencoder.py | Width validation |
| EXPERIMENT-R14-007 | R14 MUST use 30 rounds x 20 local epochs for DIAD | 1857 | PENDING | fl/trainer.py | Training validation |
| EXPERIMENT-R14-008 | R14 MUST compare FedCRG, GLOBAL-Q99-FULL, LOCAL-Q99-FULL, SHRINKAGE | 1438 | PENDING | real_data.py | Policy comparison |
| EXPERIMENT-R14-009 | R14 is EXPLORATORY and CANNOT replace the 86-feature R10 result | 1438 | PENDING | real_data.py | Claim discipline |
| EXPERIMENT-R14-010 | R14 adds 5 exploratory DIAD trainings with DATA-DEPENDENT feature dimension | 1486 | PENDING | real_data.py | Training counting |
| EXPERIMENT-R14-011 | R14 policy cells: 5 x 1 x 4 x K_D = 20*K_D | 1487 | PENDING | real_data.py | Cell counting |

---

## Randomness Registry

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| RANDOMNESS-001 | Model seeds MUST be: 11, 22, 33, 44, 55 | 1444, 2204 | IMPLEMENTED | config.py | Config validation |
| RANDOMNESS-002 | N-BaIoT calibration seeds MUST be: 1000-1049; 1000 is named primary split | 1445 | IMPLEMENTED | config.py | Config validation |
| RANDOMNESS-003 | DIAD calibration seeds MUST be: 2000-2019; 2000 is named primary split | 1446 | IMPLEMENTED | config.py | Config validation |
| RANDOMNESS-004 | Attack dev/test stratification seed MUST be: 9001 | 1447 | IMPLEMENTED | config.py | Config validation |
| RANDOMNESS-005 | Synthetic Monte Carlo master seed MUST be: 123456 | 1448 | IMPLEMENTED | synthetic.py | Seed validation |
| RANDOMNESS-006 | Optional device-population bootstrap seed MUST be: 424242 | 1449, 1574-1590 | IMPLEMENTED | config.py | Config validation |
| RANDOMNESS-007 | Deep-SVDD model seeds MUST be: 11, 22, 33 | 1450 | IMPLEMENTED | config.py | Config validation |
| RANDOMNESS-008 | N-BaIoT primary calibration seed: 1000 | 1425 | IMPLEMENTED | nbaiot.py | Seed validation |
| RANDOMNESS-009 | DIAD primary calibration seed: 2000 | 1434 | IMPLEMENTED | diad.py | Seed validation |

---

## Experiment Workload Accounting

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| WORKLOAD-S1 | S1 total: 320,000 Monte-Carlo trials | 1493 | PENDING | synthetic.py | Trial counting |
| WORKLOAD-S2 | S2 total: 360,000 trials | 1494 | PENDING | synthetic.py | Trial counting |
| WORKLOAD-S3 | S3 total: 120,000 trials | 1495 | PENDING | synthetic.py | Trial counting |
| WORKLOAD-S4 | S4 total: 50,000 trials | 1496 | PENDING | synthetic.py | Trial counting |
| WORKLOAD-S5 | S5 total: 120,000 trials | 1497 | PENDING | synthetic.py | Trial counting |
| WORKLOAD-S6 | S6 total: 45 cells, zero Monte-Carlo trials | 1498 | IMPLEMENTED | synthetic.py | PASSED |
| WORKLOAD-SYNTHETIC-TOTAL | Total locked Monte-Carlo trials S1-S5: 970,000 | 1500 | PENDING | synthetic.py | Trial counting |
| WORKLOAD-R1-TRAIN | R1 detector trainings: 5 complete FL trainings | 1457 | PENDING | fl/trainer.py | Training counting |
| WORKLOAD-R1-POLICY | R1 policy evaluations: 5 x 50 x 12 x 9 = 27,000 client-policy cells | 1461 | PENDING | real_data.py | Cell counting |
| WORKLOAD-R1-DECISIONS | R1 FedCRG decisions: 5 x 50 x 9 = 2,250 client state decisions | 1463 | PENDING | real_data.py | Decision counting |
| WORKLOAD-R1-REFERENCES | R1 reference thresholds: 250 | 1464-1466 | PENDING | real_data.py | Reference counting |
| WORKLOAD-DIAD-TRAIN | DIAD detector trainings: 5 | 1479 | PENDING | fl/trainer.py | Training counting |
| WORKLOAD-DIAD-POLICY | DIAD policy cells: 1200*K_D | 1481 | PENDING | real_data.py | Cell counting |
| WORKLOAD-DIAD-EXPLORE | R14 adds 5 exploratory DIAD trainings | 1487 | PENDING | real_data.py | Training counting |

---

## Experiment Ledger Verification

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| LEDGER-001 | Run-verification script MUST reconcile actual artifacts against expected ledger before statistics | 1488-1489 | PENDING | results.py | Ledger reconciliation |
| LEDGER-002 | Comparator-only cells for DIAD: 5 x 20 x 11 x K_D = 1100*K_D | 1482-1483 | PENDING | results.py | Cell counting |
| LEDGER-003 | FedCRG state decisions for DIAD: 100*K_D | 1484 | PENDING | results.py | Decision counting |
| LEDGER-004 | Reference threshold constructions for DIAD: 100 | 1485 | PENDING | results.py | Reference counting |
| LEDGER-005 | All policies MUST read identical score_cache hash for given dataset/model seed | 1766 | PENDING | results.py | Hash validation |

---

## Summary Statistics

- **Synthetic Experiments:** 6 experiment IDs (S1-S6)
  - Total synthetic requirements: ~100
  - Implemented: ~30
  - Verified: ~25
  - Pending: ~70

- **Real Data Experiments:** 9 experiment IDs (R1-R14)  
  - Total real data requirements: ~150
  - Implemented: ~10
  - Verified: ~5
  - Pending: ~140

- **Total Experiment Requirements:** ~250
- **Overall Implementation Status:** ~15% complete for experiments

---

## Cross-References

- Core mathematical formulas: See `02_statistical_core.md` (GATE-A-*, GATE-B-*)
- Dataset specifications: See `03_dataset_requirements.md` (DATASET-*, SPLIT-*)
- Baseline definitions: See `05_baseline_requirements.md` (BASELINE-*)
- Metric definitions: See `06_metrics_requirements.md` (METRIC-*)
- Failure states: See `10_failure_claims.md` (FAILURE-*, CLAIM-*)
- CLI commands: See `08_implementation_requirements.md` (CLI-*)

---

## File Maintenance

- **Created:** 2026-08-12
- **Last Updated:** 2026-08-12
- **Version:** 1.0
- **Status:** Initial extraction from Section 11
- **Next Review:** After S1-S6 implementation completion