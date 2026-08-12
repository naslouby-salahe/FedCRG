# Next Action

**Last updated:** 2026-08-12
**Current time:** 2026-08-12

## Immediate Next Action

**Priority:** HIGH   
**Task:** Implement baseline suite (fedcrg/baselines/)  
**File:** `fedcrg/baselines/` module  
**Status:** NOT STARTED

### Specific Steps

Implement all baselines from Section 9:

1. **B0 REF-Q99-R** - Reference threshold from R only
   - Pool all R_k with equal per-client counts
   - q = min(N, ceil((N+1)(1-alpha))) with alpha=0.01
   - Use sorted pooled scores, anomaly iff score > threshold

2. **B1 GLOBAL-Q99-FULL** - Strong full benign-policy-budget always-shared
   - Pool R+G+C (5,500 N-BaIoT; 3,300 DIAD) with equal per-client counts
   - Same quantile formula

3. **B2 LOCAL-Q99-FULL** - Strong full benign-policy-budget always-local
   - Per client R+G+C
   - Same quantile formula

4. **B3 GATE-A-ONLY** - Ablates Gate-B personalization-necessity
   - If Gate A READY and multiplicity 1: use tau_local
   - Otherwise: use tau_ref

5. **B4 GATE-B-ONLY** - Ablates finite-sample readiness
   - If Gate B mismatch: use C_(q_C) with q_C=min(n_C, ceil((n_C+1)(1-alpha)))
   - Otherwise: tau_ref

6. **B5 SHRINKAGE** - Required due adjacent shrinkage literature
   - tau_shr = w * tau_local,Q99 + (1-w) * tau_ref
   - w = n_C / (n_C + n0)
   - Candidate n0 grid: {100, 300, 1000, 3000, 10000}
   - Choose n0 with minimum mean error on G_k

7. **B6 FEDDETECT-3SIGMA** - Published-style federated AE threshold
   - Pool R+G+C scores
   - threshold = global mean + 3 * sqrt(mean((s-mean)^2)) with ddof=0

8. **B7 DEV-F1-LG-SELECT** - Attack-aware selector
   - Use 500 benign guard + 500 A_dev per client (50:50 development)
   - Compute F1 for B1 and B2 on development set
   - Select B2 only if strictly larger F1; ties select B1

9. **B8 LARIDI-STYLE-SS** - Closest-prior comparator
   - Compute per-class summary statistics on development set
   - Server pools statistics and computes overlap interval
   - Generate 1000 equally spaced thresholds
   - Equal-client mean F1 selection

10. **B9 SUP-F1-1000** - Strong attack-aware candidate-search
    - 1000 federation-wide candidates spanning development-score range
    - Equal-client mean F1, maximize F1

11. **B10 ORACLE-TEST** - Unattainable diagnostic ceiling
    - For each client choose whichever of B1, B2, or FedCRG gives smallest final-test band error
    - Tie: higher TPR

### Why This is Next

According to prompt.md Section 8 (Implementation strategy):
- Phase 1: Domain model (DONE)
- Phase 2: Configuration (DONE)
- Phase 3: Dataset discovery, integrity and deterministic preparation (DONE)
- Phase 4: Role assignment and leakage prevention (DONE in data module)
- Phase 5: Preprocessing (NEXT after baselines, or parallel)
- Phase 7: Detector training (DONE)
- Phase 8: Scoring (DONE)

The baselines are needed for the experiment registry (Section 11) and
for comparison against FedCRG.

### Blocking Dependencies

None. Scoring module is complete. This task can proceed immediately.

### After Baselines

1. Implement FedCRG gate execution (fedcrg/fedcrg.py)
2. Implement experiment registry (fedcrg/experiments/)
3. Implement metrics (fedcrg/metrics/)
4. Implement preprocessing (fedcrg/data/preprocess.py)

### Time Estimate

- B0-B6 (quantile baselines): 4-6 hours
- B7-B10 (attack-aware baselines): 6-8 hours
- Total: ~10-14 hours

### Resources Needed

- Roadmap Section 9 (Baseline Suite and Information-Regime Fairness)
- Roadmap Section 9.1 (Deterministic quantile-rank ledger)
- Roadmap Section 9.2 (Shrinkage baseline)
- Roadmap Section 9.3 (Attack-aware local-versus-global selector)
- Roadmap Section 9.4 (Laridi et al. 2024 closest-prior comparator)
