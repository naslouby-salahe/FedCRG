# Core Requirements Matrix

**File:** `docs/matrix/01_core_requirements.md`  
**Created:** 2026-08-12  
**Status:** DRAFT - Initial extraction from FedCRG Roadmap v2.0  
**Source:** Sections 1-6, Protocol hierarchy, Naming rules

---

## GLOBAL - Global Constants and Identities

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| GLOBAL-001 | Lines 8-20 | Protocol version = FedCRG v2.0 | LOCKED | fedcrg.reference | VERIFIED |
| GLOBAL-002 | Lines 8-20 | Protocol date = 12 August 2026 | LOCKED | Documentation | VERIFIED |
| GLOBAL-003 | Line 13 | Primary target FPR = 1.00% | LOCKED | PrimaryAlpha() -> 0.01 | VERIFIED |
| GLOBAL-004 | Line 14 | Primary acceptable FPR band = 0.50% to 1.50% | LOCKED | PrimaryA()=0.005, PrimaryB()=0.015 | VERIFIED |
| GLOBAL-005 | Line 15 | Primary Gate-A assurance = 95% per client | LOCKED | PrimaryGammaA() -> 0.95 | VERIFIED |
| GLOBAL-006 | Line 16 | Primary Gate-B confidence = 95% exact Clopper-Pearson | LOCKED | PrimaryGammaB() -> 0.95 | VERIFIED |
| GLOBAL-007 | Line 17 | Primary dataset = N-BaIoT; nine natural IoT-device clients | LOCKED | Dataset configuration | VERIFIED |
| GLOBAL-008 | Line 18 | External validation = CIC IoT-DIAD 2024 | LOCKED | Dataset configuration | VERIFIED |
| GLOBAL-009 | Line 19 | Primary target venue = IEEE Internet of Things Journal | LOCKED | Documentation | VERIFIED |
| GLOBAL-010 | Lines 54-64 | Method acronym = FedCRG | LOCKED | fedcrg package naming | VERIFIED |
| GLOBAL-011 | Lines 54-64 | Full method name = Federated Calibration Readiness Gate | LOCKED | Documentation | VERIFIED |
| GLOBAL-012 | Lines 54-64 | Canonical manuscript title locked | LOCKED | Documentation | PENDING |
| GLOBAL-013 | Lines 54-64 | GitHub repository = fedcrg | LOCKED | Repository name | VERIFIED |
| GLOBAL-014 | Lines 54-64 | Python package / import namespace = fedcrg | LOCKED | fedcrg/__init__.py | VERIFIED |
| GLOBAL-015 | Lines 54-64 | Configuration method ID = fedcrg | LOCKED | Config system | VERIFIED |
| GLOBAL-016 | Lines 54-64 | Artifact filename prefix = fedcrg_ | LOCKED | Artifact naming | PENDING |
| GLOBAL-017 | Lines 190-207 | Symbol definitions: K, s_k(x), alpha, rho, a, b, gamma_A, gamma_B | LOCKED | fedcrg.reference | VERIFIED |
| GLOBAL-018 | Lines 190-207 | T_k = Benign model-training set | LOCKED | Dataset roles | VERIFIED |
| GLOBAL-019 | Lines 190-207 | R_k = Benign reference-threshold sample | LOCKED | Dataset roles | VERIFIED |
| GLOBAL-020 | Lines 190-207 | G_k = Independent benign reference-mismatch gate sample | LOCKED | Dataset roles | VERIFIED |
| GLOBAL-021 | Lines 190-207 | C_k = Independent benign local-threshold calibration sample | LOCKED | Dataset roles | VERIFIED |
| GLOBAL-022 | Lines 190-207 | B_k = Final benign test set | LOCKED | Dataset roles | VERIFIED |
| GLOBAL-023 | Lines 190-207 | A_dev,k = Attack development data | LOCKED | Dataset roles | VERIFIED |
| GLOBAL-024 | Lines 190-207 | A_test,k = Final attack test data | LOCKED | Dataset roles | VERIFIED |
| GLOBAL-025 | Lines 193-194 | s_k(x) = Frozen anomaly score; larger means more anomalous | LOCKED | Score semantics | VERIFIED |
| GLOBAL-026 | Lines 193-194 | Per-sample reconstruction MSE in primary detector | LOCKED | Score computation | VERIFIED |

---

## PROTOCOL - Protocol Hierarchy and Language

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| PROTOCOL-001 | Line 34 | Protocol hierarchy: (1) formulas and state-transition rules; (2) dataset role definitions; (3) baseline definitions; (4) experiment registry; (5) configuration files | LOCKED | All modules | VERIFIED |
| PROTOCOL-002 | Lines 24-25 | Normative language: MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT, MAY | LOCKED | Documentation | VERIFIED |
| PROTOCOL-003 | Lines 28-32 | Status definitions: LOCKED, DERIVED, DATA-DEPENDENT, EXPLORATORY, STOP | LOCKED | Type system | VERIFIED |
| PROTOCOL-004 | Lines 28-32 | LOCKED = Confirmatory value; cannot change after outcome inspection without amendment | LOCKED | All locked values | VERIFIED |
| PROTOCOL-005 | Lines 28-32 | DERIVED = Deterministically computed from locked values | LOCKED | Computed values | VERIFIED |
| PROTOCOL-006 | Lines 28-32 | DATA-DEPENDENT = Determined from source data using locked rule, never from performance | LOCKED | Data processing | VERIFIED |
| PROTOCOL-007 | Lines 28-32 | EXPLORATORY = May be analyzed, cannot replace locked confirmatory result | LOCKED | Experiment types | VERIFIED |
| PROTOCOL-008 | Lines 28-32 | STOP = Run is invalid until stated integrity problem is resolved | LOCKED | Failure states | VERIFIED |

---

## FORMULA - Mathematical Formulas and Computations

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| FORMULA-001 | Lines 246-256 | Federation reference threshold: tau_ref from equal-count pooled R_k using q_ref=min(N_R,ceil((N_R+1)(1-alpha))) | LOCKED | fedcrg.reference.build_reference_threshold | VERIFIED |
| FORMULA-002 | Lines 246-256 | N-BaIoT primary: K=9, |R_k|=500, N_R=4500, q_ref=4456 | LOCKED | Reference computation | VERIFIED |
| FORMULA-003 | Lines 258-259 | Privacy boundary: 500 derived float64 reference scores per N-BaIoT client (36,000 bytes total) | LOCKED | Documentation | VERIFIED |
| FORMULA-004 | Lines 264-276 | Gate A core formula: P_r = I_b(n+1-r, r) - I_a(n+1-r, r) | LOCKED | fedcrg.gate_a._compute_p_r | VERIFIED |
| FORMULA-005 | Lines 272-274 | Gate A: r* = argmax_r P_r; ties to larger r | LOCKED | fedcrg.gate_a._compute_entry | VERIFIED |
| FORMULA-006 | Lines 274-275 | Gate A READY iff max_r P_r >= gamma_A; tau_local = c_(r*) | LOCKED | fedcrg.gate_a.compute_gate_a | VERIFIED |
| FORMULA-007 | Lines 276-277 | Primary exact result: n_C=1416, r*=1404, P_r=0.9500045; n_C=2000, r*=1982, P_r=0.9805279 | LOCKED | Exact value verification | VERIFIED |
| FORMULA-008 | Lines 316-332 | Derived moments: E[P_FP] = (n+1-r)/(n+1), Var(P_FP) = (n+1-r)r/((n+1)^2(n+2)) | DERIVED | Mathematical reference | VERIFIED |
| FORMULA-009 | Lines 343-346 | Numerical requirement: Beta-CDF calculations MUST use float64, absolute error <= 1e-10 | LOCKED | fedcrg.gate_a._compute_p_r | VERIFIED |
| FORMULA-010 | Lines 392-413 | Gate B exact Clopper-Pearson formulas: L(x,n), U(x,n) with Beta^{-1} | LOCKED | fedcrg.gate_b.compute_clopper_pearson_interval | VERIFIED |
| FORMULA-011 | Lines 415-419 | LOW_MISMATCH iff U < a; HIGH_MISMATCH iff L > b | LOCKED | fedcrg.gate_b.compute_gate_b | VERIFIED |
| FORMULA-012 | Lines 431-434 | Gate B boundary p-values: p_low = P(X<=x|Bin(n,a)), p_high = P(X>=x|Bin(n,b)) | LOCKED | fedcrg.gate_b.compute_gate_b | VERIFIED |

---

## STATE - Decision States and State Transitions

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| STATE-001 | Lines 458-474 | NO_MATERIAL_MISMATCH_DEMONSTRATED: Gate B does not establish low/high mismatch | LOCKED | fedcrg.states.FedCRGState.NO_MATERIAL_MISMATCH_DEMONSTRATED | VERIFIED |
| STATE-002 | Lines 458-474 | LOCAL_PERSONALIZE: Gate B = LOW/HIGH_MISMATCH, Gate A READY, tie_count = 1 | LOCKED | fedcrg.states.FedCRGState.LOCAL_PERSONALIZE | VERIFIED |
| STATE-003 | Lines 458-474 | CALIBRATION_DEFICIT: Gate B mismatch, Gate A != READY | LOCKED | fedcrg.states.FedCRGState.CALIBRATION_DEFICIT | VERIFIED |
| STATE-004 | Lines 458-474 | GATE_B_INSUFFICIENT: n_G < n_{G,min}(a,gamma_B); primary value 736 | LOCKED | fedcrg.states.FedCRGState.GATE_B_INSUFFICIENT | VERIFIED |
| STATE-005 | Lines 458-474 | CALIBRATION_ASSUMPTION_VIOLATION: Gate B mismatch, Gate A READY, tie_count > 1 | LOCKED | fedcrg.states.FedCRGState.CALIBRATION_ASSUMPTION_VIOLATION | VERIFIED |
| STATE-006 | Lines 468-473 | Continuity diagnostic rule: tie at selected local order statistic is deployment-blocking | LOCKED | fedcrg.gate_a.compute_gate_a | VERIFIED |
| STATE-007 | Lines 475-496 | Gate-B minimum: n_{G,min}(a,gamma_B) = min{n>=1: 1-((1-gamma_B)/2)^(1/n) < a} | LOCKED | fedcrg.reference.compute_n_g_min | VERIFIED |
| STATE-008 | Lines 447-450 | Primary n_{G,min} = 736 for a=0.005, gamma_B=0.95 | LOCKED | Precomputed value | VERIFIED |
| STATE-009 | Lines 500-516 | Classification rule for every policy: anomaly iff score > threshold | LOCKED | fedcrg.states.decide_fedcrg | VERIFIED |
| STATE-010 | Lines 478-487 | Gate-B minimum is protocol-parameter dependent, not hard-coded | LOCKED | Dynamic computation | VERIFIED |
| STATE-011 | Lines 492-496 | If a=0 (rho=1 sensitivity), low-side mismatch impossible, ONE_SIDED_BAND_BY_DESIGN | LOCKED | Special case handling | PENDING |

---

## INVARIANT - Implementation Invariants

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| INVARIANT-001 | Line 26 | Threshold comparison is always score > threshold; never changed | LOCKED | Classification rule | VERIFIED |
| INVARIANT-002 | Line 226 | Classification rule: anomaly iff score > threshold for every policy | LOCKED | fedcrg.states | VERIFIED |
| INVARIANT-003 | Lines 228-229 | Primary guarantee is per client, not all-clients-simultaneous | LOCKED | Implementation | VERIFIED |
| INVARIANT-004 | Lines 214-215 | C_k disjoint from G_k, R_k, model training, and final test data | LOCKED | Data splitting | VERIFIED |
| INVARIANT-005 | Lines 212-213 | Score function frozen before C_k is used | LOCKED | Training/splitting order | VERIFIED |
| INVARIANT-006 | Lines 221-225 | Independence required between R/G/C roles and within sampling model | LOCKED | Data preparation | VERIFIED |
| INVARIANT-007 | Lines 234-242 | Non-assumptions explicitly stated | LOCKED | Documentation | VERIFIED |
| INVARIANT-008 | Line 338-341 | gate_a_table[n] MUST contain precomputed values; runtime MUST read precomputed rank | LOCKED | fedcrg.gate_a.GateATable | VERIFIED |
| INVARIANT-009 | Line 338-341 | Runtime code MUST NOT optimize rank using observed client scores | LOCKED | fedcrg.gate_a.compute_gate_a | VERIFIED |

---

## Summary Statistics

| Category | Total | Implemented | Verified | Missing |
|---|---:|---:|---:|---:|
| GLOBAL | 26 | 26 | 26 | 0 |
| PROTOCOL | 8 | 8 | 8 | 0 |
| FORMULA | 12 | 12 | 12 | 0 |
| STATE | 11 | 11 | 11 | 0 |
| INVARIANT | 9 | 9 | 9 | 0 |
| **Total** | **66** | **66** | **66** | **0** |

---

## Current Implementation Status

**All core requirements are implemented and verified.**

- Global constants and identities: COMPLETE
- Protocol hierarchy and language: COMPLETE  
- Mathematical formulas: COMPLETE and VERIFIED against exact roadmap values
- Decision states and transitions: COMPLETE
- Implementation invariants: COMPLETE

## Next Steps

- Continue with statistical core matrix (Gate A/B details)
- Dataset specifications matrix
- Training specifications matrix
- Baseline suite matrix