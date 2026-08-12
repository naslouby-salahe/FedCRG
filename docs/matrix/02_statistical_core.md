# Statistical Core Matrix

**File:** `docs/matrix/02_statistical_core.md`  
**Created:** 2026-08-12  
**Status:** DRAFT - Initial extraction from FedCRG Roadmap v2.0  
**Source:** Sections 5.2-5.4, Gate A/B mathematics

---

## GATE-A - Local Operating-Band Readiness

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| GATE-A-001 | Lines 260-262 | Correction: Gate A is two-sided in operating FPR | LOCKED | fedcrg.gate_a | VERIFIED |
| GATE-A-002 | Lines 264-270 | C_k sorted: c_(1) <= ... <= c_(n); tau_r = c_(r) | LOCKED | fedcrg.gate_a.compute_gate_a | VERIFIED |
| GATE-A-003 | Lines 266-270 | P_r = I_b(n+1-r, r) - I_a(n+1-r, r) under i.i.d.-continuous model | LOCKED | fedcrg.gate_a._compute_p_r | VERIFIED |
| GATE-A-004 | Line 272 | r* = argmax_r P_r; ties resolved to larger r (more conservative threshold) | LOCKED | fedcrg.gate_a._compute_entry | VERIFIED |
| GATE-A-005 | Line 274 | Gate A = READY iff max_r P_r >= gamma_A | LOCKED | fedcrg.gate_a.compute_gate_a | VERIFIED |
| GATE-A-006 | Line 274 | tau_local = c_(r*) (r*-th order statistic) | LOCKED | fedcrg.gate_a.compute_gate_a | VERIFIED |
| GATE-A-007 | Lines 276-277 | Primary exact results verified: n=1416, r*=1404, P_r=0.9500045; n=2000, r*=1982, P_r=0.9805279 | LOCKED | fedcrg.gate_a.verify_gate_a_exact_values | VERIFIED |
| GATE-A-008 | Lines 278-280 | Interpretation discipline: Gate-A readiness is sample-size/contract gate | LOCKED | Documentation | VERIFIED |
| GATE-A-009 | Lines 281-283 | Evidence quantity is amount of independent presumed-benign calibration data | LOCKED | Documentation | VERIFIED |
| GATE-A-010 | Lines 284-290 | Theorem: P_FP(C_{k,(r)}) ~ Beta(n+1-r, r) | LOCKED | Mathematical foundation | VERIFIED |
| GATE-A-011 | Lines 316-320 | Implementation invariant: precomputed rank MUST be used, not optimized from observed scores | LOCKED | fedcrg.gate_a.compute_gate_a | VERIFIED |
| GATE-A-012 | Lines 334-336 | Mean FPR at n=2000,r*=1982: 19/2001=0.0094952524 | DERIVED | Mathematical verification | VERIFIED |
| GATE-A-013 | Lines 338-342 | gate_a_table[n] MUST contain: n, rank_r, coverage_probability, ready, alpha, rho, a, b, gamma_A | LOCKED | fedcrg.gate_a.GateATableEntry | VERIFIED |
| GATE-A-014 | Lines 343-347 | Numerical requirement: float64, tested special-function, absolute error <= 1e-10 | LOCKED | fedcrg.gate_a._compute_p_r | VERIFIED |
| GATE-A-015 | Lines 349-367 | Precomputation table for various n values | LOCKED | fedcrg.gate_a.precompute_primary_gate_a_table | VERIFIED |
| GATE-A-016 | Lines 350-360 | Minimum n_C for 95% Gate A at various alpha/rho: alpha=0.5%->2861, alpha=1.0%->1416, alpha=2.0%->694, alpha=5.0%->270 | LOCKED | Precomputed values | VERIFIED |
| GATE-A-017 | Lines 356-360 | Minimum n_C for various assurance levels: gamma_A=90%->1000, gamma_A=95%->1416, gamma_A=99%->2435 | LOCKED | Precomputed values | VERIFIED |
| GATE-A-018 | Lines 362-367 | Minimum n_C for various tolerances: rho=0.25->5970, rho=0.50->1416, rho=1.00->149 | LOCKED | Precomputed values | VERIFIED |
| GATE-A-019 | Lines 504-506 | Pseudocode steps 8-11: compute P_r for every rank, choose largest-r tie, check max(P_r) < gamma_A | LOCKED | fedcrg.gate_a.compute_gate_a | VERIFIED |
| GATE-A-020 | Lines 513-514 | Pseudocode steps 10-11: tau_local = sorted(C_k)[r*-1], tie_count = multiplicity(tau_local in C_k) | LOCKED | fedcrg.gate_a.compute_gate_a | VERIFIED |
| GATE-A-021 | Lines 511-512 | Pseudocode step 12: If tie_count > 1: return CALIBRATION_ASSUMPTION_VIOLATION | LOCKED | fedcrg.states.decide_fedcrg | VERIFIED |

---

## GATE-B - Independent Evidence of Reference Threshold Mismatch

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| GATE-B-001 | Lines 368-369 | Gate B tests exact deployment quantity: benign exceedance probability of tau_ref | LOCKED | fedcrg.gate_b | VERIFIED |
| GATE-B-002 | Lines 372-373 | x_k = sum_{g in G_k} 1[g > tau_ref], n_G = |G_k| | LOCKED | fedcrg.gate_b.compute_gate_b | VERIFIED |
| GATE-B-003 | Lines 374-376 | Compute two-sided 95% exact Clopper-Pearson interval [L_k, U_k] for p_ref,k | LOCKED | fedcrg.gate_b.compute_clopper_pearson_interval | VERIFIED |
| GATE-B-004 | Lines 376-379 | LOW_MISMATCH if U_k < a; HIGH_MISMATCH if L_k > b; otherwise NO_MATERIAL_MISMATCH_DEMONSTRATED | LOCKED | fedcrg.gate_b.compute_gate_b | VERIFIED |
| GATE-B-005 | Lines 377-379 | Interpretation: NO_MATERIAL_MISMATCH_DEMONSTRATED is not equivalence claim | LOCKED | Documentation | VERIFIED |
| GATE-B-006 | Lines 380-388 | n_G thresholds: n_G=736 (x=0 and x>=19), n_G=1000 (x<=0 and x>=24), n_G=1500 (x<=2 and x>=33), n_G=2000 (x<=3 and x>=42), n_G=3000 (x<=7 and x>=59) | LOCKED | Documentation | VERIFIED |
| GATE-B-007 | Lines 388-389 | Gate-B minimum evidence rule: n_G >= n_G_min(a,gamma_B)=736 for primary contract | LOCKED | fedcrg.reference.compute_n_g_min | VERIFIED |
| GATE-B-008 | Lines 392-413 | Exact Clopper-Pearson formulas: L(x,n) = Beta^{-1}(delta_B/2; x, n-x+1) for x>0, 0 for x=0 | LOCKED | fedcrg.gate_b.compute_clopper_pearson_interval | VERIFIED |
| GATE-B-009 | Lines 392-413 | U(x,n) = 1 for x=n, Beta^{-1}(1-delta_B/2; x+1, n-x) for x<n | LOCKED | fedcrg.gate_b.compute_clopper_pearson_interval | VERIFIED |
| GATE-B-010 | Lines 415-419 | Decision rules: LOW_MISMATCH iff U < a; HIGH_MISMATCH iff L > b | LOCKED | fedcrg.gate_b.compute_gate_b | VERIFIED |
| GATE-B-011 | Lines 421-426 | Decision is conditional on realized reference threshold | LOCKED | Implementation | VERIFIED |
| GATE-B-012 | Lines 422-425 | R_k and G_k are disjoint, Gate B never reuses reference-construction scores | LOCKED | Data splitting | VERIFIED |
| GATE-B-013 | Lines 427-434 | MUST log exact one-sided boundary p-values p_low and p_high | LOCKED | fedcrg.gate_b.compute_gate_b | VERIFIED |
| GATE-B-014 | Lines 436-438 | Normative state transition based on Clopper-Pearson interval rule | LOCKED | fedcrg.gate_b.compute_gate_b | VERIFIED |
| GATE-B-015 | Lines 441-455 | Derivation: n_{G,min} from U(0,n) = 1-(delta_B/2)^(1/n) < a | LOCKED | fedcrg.reference.compute_n_g_min | VERIFIED |
| GATE-B-016 | Lines 447-450 | At n=735, U=0.0050063101; at n=736, U=0.0049995250 | LOCKED | Mathematical verification | VERIFIED |
| GATE-B-017 | Lines 507-510 | Pseudocode steps 3-6: compute n_G,min, x = count(g > tau_ref), [L,U] = exact Clopper-Pearson | LOCKED | fedcrg.gate_b.compute_gate_b | VERIFIED |
| GATE-B-018 | Lines 509-510 | Pseudocode step 5: mismatch = LOW if U < a; HIGH if L > b; otherwise NONE | LOCKED | fedcrg.gate_b.compute_gate_b | VERIFIED |

---

## PRECOMPUTATION - Precomputation Requirements

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| PRECOMPUTE-001 | Lines 14.5 | For fixed (n, a, b, gamma_A), r* and P_r determined BEFORE observing scores | LOCKED | fedcrg.gate_a.GateATable | VERIFIED |
| PRECOMPUTE-002 | Lines 14.5 | Runtime code MUST read precomputed rank and MUST NOT optimize rank using observed scores | LOCKED | fedcrg.gate_a.compute_gate_a | VERIFIED |
| PRECOMPUTE-003 | Lines 347-348 | Absolute error against reference values MUST be <= 1e-10 | LOCKED | fedcrg.gate_a.verify_gate_a_exact_values | VERIFIED |
| PRECOMPUTE-004 | Lines 14.5.1 | R13: Measure wall time and memory for threshold-policy primitives | LOCKED | fedcrg.experiments.real_data.run_r13 | VERIFIED |

---

## NUMERICAL - Numerical Requirements and Precision

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| NUMERICAL-001 | Lines 343-346 | Beta-CDF calculations MUST use float64 | LOCKED | numpy.float64, scipy.special.betainc | VERIFIED |
| NUMERICAL-002 | Lines 343-346 | Tested special-function implementation | LOCKED | scipy.special tested | VERIFIED |
| NUMERICAL-003 | Lines 345-346 | Absolute error <= 1e-10 for every locked unit-test cell | LOCKED | fedcrg.gate_a.verify_gate_a_exact_values | VERIFIED |
| NUMERICAL-004 | Lines 347-348 | Reference values in Section 14.2 used for verification | LOCKED | _EXPECTED_VALUES in gate_a.py | VERIFIED |
| NUMERICAL-005 | Lines 434-435 | Boundary p-values use binomial CDF | LOCKED | scipy.stats.binom | VERIFIED |

---

## SUMMARY - Statistical Core Summary

| Category | Total | Implemented | Verified | Missing |
|---|---:|---:|---:|---:|
| GATE-A | 21 | 21 | 21 | 0 |
| GATE-B | 18 | 18 | 18 | 0 |
| PRECOMPUTATION | 4 | 4 | 4 | 0 |
| NUMERICAL | 5 | 5 | 5 | 0 |
| **Total** | **48** | **48** | **48** | **0** |

---

## Current Implementation Status

**All statistical core requirements are implemented and verified.**

- Gate A mathematics: COMPLETE and VERIFIED against exact roadmap values
- Gate B mathematics: COMPLETE and VERIFIED 
- Precomputation requirements: COMPLETE
- Numerical precision requirements: COMPLETE and VERIFIED

## Verification Evidence

- `fedcrg.gate_a.verify_gate_a_exact_values()`: PASSED (tolerance 1e-10)
- All expected values from Section 349-367 verified
- Primary contract values: n=1416, r*=1404, P_r=0.9500045; n=2000, r*=1982, P_r=0.9805279
- Minimum n_C values for various parameters verified

## Next Steps

- Create dataset specifications matrix (03_dataset_requirements.md)
- Create training specifications matrix (04_training_requirements.md)
- Create baseline suite matrix (05_baseline_requirements.md)
- Create metrics and evaluation matrix (06_metrics_requirements.md)