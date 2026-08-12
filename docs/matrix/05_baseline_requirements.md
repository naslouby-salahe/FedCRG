# Baseline Suite Matrix

**File:** `docs/matrix/05_baseline_requirements.md`  
**Created:** 2026-08-12  
**Status:** DRAFT - Initial extraction from FedCRG Roadmap v2.0  
**Source:** Sections 9.0-9.5, Baseline Suite and Information-Regime Fairness

---

## BASELINE-PRINCIPLE - Mandatory Baseline Principle

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| BASELINE-PRINCIPLE-001 | Lines 1168-1171 | FedCRG uses multiple disjoint benign pools to prevent reviewer argument that weak comparator was deliberately starved of calibration data | LOCKED | Documentation | VERIFIED |
| BASELINE-PRINCIPLE-002 | Lines 1168-1171 | Study includes both role-matched and full benign-policy-budget shared/local comparators | LOCKED | Baseline design | VERIFIED |
| BASELINE-PRINCIPLE-003 | Lines 1168-1171 | FULL means all benign samples available to benign-only threshold policies: R+G+C = 5,500/client on N-BaIoT and 3,300/client on DIAD | LOCKED | Configuration | VERIFIED |
| BASELINE-PRINCIPLE-004 | Lines 1171-1174 | Separate 500-record benign guard intentionally withheld from B0-B6 because it is independent development half used by attack-aware B7-B9 | LOCKED | Data splitting | VERIFIED |
| BASELINE-PRINCIPLE-005 | Lines 1171-1174 | Using guard for B1/B2 would leak development labels into selector comparison | LOCKED | Implementation | VERIFIED |

---

## BASELINE-DEFINITIONS - Baseline Definitions (B0-B10)

| ID | Section | Baseline ID | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|---|
| BASELINE-001 | Lines 1172-1184 | B0 | REF-Q99-R: Benign only; tau_ref from R only | LOCKED | fedcrg.baselines.quantile.QuantileBaseline | VERIFIED |
| BASELINE-002 | Lines 1172-1184 | B1 | GLOBAL-Q99-FULL: Benign only; pool every client R+G+C (5,500 N-BaIoT; 3,300 DIAD); q=min(N,ceil((N+1)(1-alpha))); strict > | LOCKED | fedcrg.baselines.quantile.QuantileBaseline | VERIFIED |
| BASELINE-003 | Lines 1172-1184 | B2 | LOCAL-Q99-FULL: Benign only; per client R+G+C; q=min(n,ceil((n+1)(1-alpha))); strict > | LOCKED | fedcrg.baselines.quantile.QuantileBaseline | VERIFIED |
| BASELINE-004 | Lines 1172-1184 | B3 | GATE-A-ONLY: Benign only; if Gate A sample-size READY AND tie_count=1, use tau_local; otherwise use tau_ref | LOCKED | fedcrg.baselines.gate_only.GateAOnlyBaseline | VERIFIED |
| BASELINE-005 | Lines 1172-1184 | B4 | GATE-B-ONLY: Benign only; if Gate B mismatch, use C_(q_C) with q_C=min(n_C,ceil((n_C+1)(1-alpha))); otherwise tau_ref | LOCKED | fedcrg.baselines.gate_only.GateBOnlyBaseline | VERIFIED |
| BASELINE-006 | Lines 1172-1184 | B5 | SHRINKAGE: Benign only; tau_shr = w*tau_local,Q99 + (1-w)*tau_ref, w=n_C/(n_C+n0) | LOCKED | fedcrg.baselines.shrinkage.ShrinkageBaseline | VERIFIED |
| BASELINE-007 | Lines 1172-1184 | B6 | FEDDETECT-3SIGMA: Benign only; pool R+G+C scores; threshold = global mean + 3*sqrt(mean((s-mean)^2)) (ddof=0) | LOCKED | fedcrg.baselines.feddetect_3sigma.FedDetect3SigmaBaseline | VERIFIED |
| BASELINE-008 | Lines 1172-1184 | B7 | DEV-F1-LG-SELECT: Benign guard + attack development; per client choose between B1 and B2 using F1 on disjoint guard + A_dev; tie -> B1 | LOCKED | fedcrg.baselines.attack_aware.DevF1LgSelectBaseline | VERIFIED |
| BASELINE-009 | Lines 1172-1184 | B8 | LARIDI-STYLE-SS: Benign guard + attack development; locked unrefined summary-statistics overlap; 1000 candidates; equal-client mean F1 | LOCKED | fedcrg.baselines.attack_aware.LaridiStyleSSBaseline | VERIFIED |
| BASELINE-010 | Lines 1172-1184 | B9 | SUP-F1-1000: Benign guard + attack development; 1000 federation-wide candidates; equal-client mean F1; maximize F1 | LOCKED | fedcrg.baselines.attack_aware.SupF11000Baseline | VERIFIED |
| BASELINE-011 | Lines 1172-1184 | B10 | ORACLE-TEST: Final labels; choose whichever of GLOBAL-Q99-FULL, LOCAL-Q99-FULL, or FedCRG gives smallest final-test band error; break ties by higher TPR | LOCKED | fedcrg.baselines.oracle.OracleTestBaseline | VERIFIED |

---

## QUANTILE-CONVENTION - Deterministic Quantile-Rank Ledger

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| QUANTILE-001 | Lines 1186-1188 | Every Q99 baseline uses same finite-sample rank convention | LOCKED | fedcrg.baselines.quantile.QuantileBaseline | VERIFIED |
| QUANTILE-002 | Lines 1190-1191 | q(N,alpha) = min{N, ceil((N+1)(1-alpha))} | LOCKED | fedcrg.baselines.quantile.quantile_rank | VERIFIED |
| QUANTILE-003 | Line 1192 | With ascending order statistics and anomaly iff score > threshold | LOCKED | fedcrg.baselines.quantile.QuantileBaseline | VERIFIED |
| QUANTILE-004 | Lines 1194-1204 | Primary alpha=0.01 exact ranks: N-BaIoT REF-Q99-R = 4,456; GLOBAL-Q99-FULL = 49,006; LOCAL-Q99-FULL = 5,446; B4/B5 local Q99 from C_k = 1,981 | LOCKED | Computation | VERIFIED |
| QUANTILE-005 | Lines 1194-1204 | DIAD exact ranks: REF-Q99-R = q(300*K_D,0.01); GLOBAL-Q99-FULL = q(3,300*K_D,0.01); LOCAL-Q99-FULL = 3,268; B4/B5 local Q99 from C_k = 1,486 | LOCKED | Computation | VERIFIED |
| QUANTILE-006 | Lines 1206-1207 | Gate-A rank r*=1,982 at n_C=2,000 is intentionally NOT empirical Q99 rank 1,981: Gate A maximizes finite-sample probability of landing inside whole 0.5%-1.5% band | LOCKED | Documentation | VERIFIED |

---

## SHRINKAGE - Shrinkage Baseline Exact Tuning Rule

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| SHRINKAGE-001 | Lines 1209-1213 | q_C = min(n_C, ceil((n_C+1)(1-alpha))) and tau_local,Q99 = C_(q_C) | LOCKED | fedcrg.baselines.shrinkage.ShrinkageBaseline | VERIFIED |
| SHRINKAGE-002 | Lines 1209-1213 | w(n0) = n_C/(n_C + n0) and tau_shr = w*tau_local,Q99 + (1-w)*tau_ref | LOCKED | fedcrg.baselines.shrinkage.ShrinkageBaseline | VERIFIED |
| SHRINKAGE-003 | Lines 1209-1213 | Candidate n0 grid: {100, 300, 1000, 3000, 10000} | LOCKED | fedcrg.baselines.shrinkage.ShrinkageBaseline | VERIFIED |
| SHRINKAGE-004 | Lines 1209-1213 | This baseline operates in threshold-score space and is not invariant to arbitrary monotone score transformations | LOCKED | Documentation | VERIFIED |
| SHRINKAGE-005 | Lines 1209-1213 | Detector score definition is fixed and identical across policies | LOCKED | Implementation | VERIFIED |
| SHRINKAGE-006 | Lines 1209-1213 | For each n0, estimate each client FPR on G_k and compute mean absolute target-FPR error across clients | LOCKED | fedcrg.baselines.shrinkage.ShrinkageBaseline | VERIFIED |
| SHRINKAGE-007 | Lines 1209-1213 | Choose n0 with minimum mean error; ties choose largest n0 (more pooling) | LOCKED | fedcrg.baselines.shrinkage.ShrinkageBaseline | VERIFIED |
| SHRINKAGE-008 | Lines 1209-1213 | No attack data and no final test data are used | LOCKED | Implementation | VERIFIED |
| SHRINKAGE-009 | Lines 1209-1213 | Explicitly labeled shrinkage-style baseline, not reproduction of Shahid (2026) | LOCKED | Documentation | VERIFIED |

---

## ATTACK-AWARE - Attack-Aware Local-versus-Global Selector

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| ATTACK-AWARE-001 | Lines 1215-1217 | B7 DEV-F1-LG-SELECT is intentionally stronger in supervision than FedCRG | LOCKED | Documentation | VERIFIED |
| ATTACK-AWARE-002 | Lines 1215-1217 | Not deployable under benign-only assumption; exists to test whether attack labels materially improve local-versus-global decision | LOCKED | Documentation | VERIFIED |
| ATTACK-AWARE-003 | Lines 1219-1220 | Construct B1 GLOBAL-Q99-FULL and B2 LOCAL-Q99-FULL using R+G+C exactly as defined | LOCKED | fedcrg.baselines.attack_aware | VERIFIED |
| ATTACK-AWARE-004 | Lines 1221-1222 | For each client, create 1,000-record development set from exactly 500 comparator-benign guard scores plus exactly 500 attack-balanced A_dev,k scores | LOCKED | fedcrg.baselines.attack_aware | PENDING |
| ATTACK-AWARE-005 | Lines 1221-1222 | Neither set is visible to FedCRG | LOCKED | Data firewall | VERIFIED |
| ATTACK-AWARE-006 | Lines 1223-1225 | Compute F1 for B1 and B2 thresholds on that client development set; select B2 only if its F1 is strictly larger; ties select B1 | LOCKED | fedcrg.baselines.attack_aware | VERIFIED |
| ATTACK-AWARE-007 | Line 1226 | Freeze selected choice and evaluate once on B_k + A_test,k | LOCKED | fedcrg.baselines.attack_aware | VERIFIED |
| ATTACK-AWARE-008 | Line 1226 | No final-test outcome may influence selection | LOCKED | Implementation | VERIFIED |

---

## LARIDI-COMPARATOR - Laridi et al. 2024 Closest-Prior Comparator

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| LARIDI-001 | Lines 1227-1237 | Reproduction label discipline: Laridi-style overlap interval deliberately not exact reproduction | LOCKED | Documentation | VERIFIED |
| LARIDI-002 | Lines 1227-1237 | Named LARIDI-STYLE-SS, not exact reproduction | LOCKED | fedcrg.config.PolicyID.LARIDI_STYLE_SS | VERIFIED |
| LARIDI-003 | Lines 1239-1269 | Implementation fully specified so behavior does not depend on ambiguous transcription | LOCKED | fedcrg.baselines.attack_aware.LaridiStyleSSBaseline | VERIFIED |
| LARIDI-004 | Lines 1242-1244 | For each client k and class y in {benign, attack}, on fixed 1,000-record development set compute n_ky, mu_ky, v_ky | LOCKED | fedcrg.baselines.attack_aware.LaridiStyleSSBaseline | VERIFIED |
| LARIDI-005 | Lines 1246-1248 | n_ky = count, mu_ky = mean, v_ky = variance for each class | LOCKED | fedcrg.baselines.attack_aware.LaridiStyleSSBaseline | VERIFIED |
| LARIDI-006 | Lines 1252-1254 | Each client sends (n_ky, mu_ky, v_ky) for both classes to server | LOCKED | fedcrg.baselines.attack_aware.LaridiStyleSSBaseline | VERIFIED |
| LARIDI-007 | Lines 1253-1267 | Server computes exact pooled population moments for each class | LOCKED | fedcrg.baselines.attack_aware.LaridiStyleSSBaseline | VERIFIED |
| LARIDI-008 | Lines 1255-1256 | N_y = sum_k n_ky | LOCKED | fedcrg.baselines.attack_aware.LaridiStyleSSBaseline | VERIFIED |
| LARIDI-009 | Lines 1260-1261 | mu_y = (sum_k n_ky * mu_ky) / N_y | LOCKED | fedcrg.baselines.attack_aware.LaridiStyleSSBaseline | VERIFIED |
| LARIDI-010 | Lines 1263-1266 | v_y = (sum_k n_ky*(v_ky + mu_ky^2)) / N_y - mu_y^2; sigma_y = sqrt(max(v_y, 0)) | LOCKED | fedcrg.baselines.attack_aware.LaridiStyleSSBaseline | VERIFIED |
| LARIDI-011 | Lines 1271-1282 | Locked Laridi-style unrefined overlap interval: ell = max(mu_benign - 3*sigma_benign, mu_attack - 3*sigma_attack); u = min(mu_benign + 3*sigma_benign, mu_attack + 3*sigma_attack) | LOCKED | fedcrg.baselines.attack_aware.LaridiStyleSSBaseline | VERIFIED |
| LARIDI-012 | Lines 1285-1286 | If ell >= u, record LARIDI_STYLE_UNDEFINED; do not invent fallback | LOCKED | fedcrg.baselines.attack_aware.LaridiStyleSSBaseline | VERIFIED |
| LARIDI-013 | Lines 1287-1289 | Generate exactly 1,000 equally spaced thresholds including ell and u: t_j = ell + j*(u-ell)/999, j=0,...,999 | LOCKED | fedcrg.baselines.attack_aware.LaridiStyleSSBaseline | VERIFIED |
| LARIDI-014 | Lines 1290-1296 | Each client evaluates F1 for all candidates on its own balanced 500-benign/500-malicious development set | LOCKED | fedcrg.baselines.attack_aware.LaridiStyleSSBaseline | VERIFIED |
| LARIDI-015 | Lines 1292-1296 | Server computes equal-client arithmetic mean F1 for each t_j | LOCKED | fedcrg.baselines.attack_aware.LaridiStyleSSBaseline | VERIFIED |
| LARIDI-016 | Lines 1294-1295 | Select candidate with maximal mean F1; exact ties select smaller threshold | LOCKED | fedcrg.baselines.attack_aware.LaridiStyleSSBaseline | VERIFIED |
| LARIDI-017 | Lines 1296-1297 | Freeze threshold and evaluate once on final test scores | LOCKED | fedcrg.baselines.attack_aware.LaridiStyleSSBaseline | VERIFIED |
| LARIDI-018 | Lines 1298-1300 | Captures closest prior paper's supervised summary-statistic/F1 threshold-selection regime | LOCKED | Documentation | VERIFIED |
| LARIDI-019 | Lines 1300-1304 | Not claiming bit-for-bit reproduction; if author code becomes available, LARIDI-ALG2-REPRO sensitivity MAY be added | LOCKED | Documentation | VERIFIED |

---

## SUP-F1-1000 - Exact Extra-Information Comparator

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| SUP-F1-1000-001 | Lines 1306-1318 | SUP-F1-1000: exact extra-information comparator | LOCKED | fedcrg.baselines.attack_aware.SupF11000Baseline | VERIFIED |
| SUP-F1-1000-002 | Lines 1308-1318 | For each client, use its supervised-comparator benign guard plus A_dev,k with labels | LOCKED | fedcrg.baselines.attack_aware.SupF11000Baseline | VERIFIED |
| SUP-F1-1000-003 | Line 1310 | FedCRG never reads either source | LOCKED | Data firewall | VERIFIED |
| SUP-F1-1000-004 | Line 1311 | Find federation-wide minimum and maximum development scores across participating clients | LOCKED | fedcrg.baselines.attack_aware.SupF11000Baseline | VERIFIED |
| SUP-F1-1000-005 | Line 1312 | Generate exactly 1000 linearly spaced threshold candidates including both endpoints | LOCKED | fedcrg.baselines.attack_aware.SupF11000Baseline | VERIFIED |
| SUP-F1-1000-006 | Line 1314 | Each client computes F1 for all 1000 thresholds on its development set and sends 1000-value vector to server | LOCKED | fedcrg.baselines.attack_aware.SupF11000Baseline | VERIFIED |
| SUP-F1-1000-007 | Lines 1316-1317 | Server takes equal-client arithmetic mean F1 for each candidate and chooses candidate with maximum mean F1 | LOCKED | fedcrg.baselines.attack_aware.SupF11000Baseline | VERIFIED |
| SUP-F1-1000-008 | Line 1317 | Threshold ties choose smaller threshold (higher sensitivity) | LOCKED | fedcrg.baselines.attack_aware.SupF11000Baseline | VERIFIED |
| SUP-F1-1000-009 | Line 1318 | Strong attack-aware global-threshold comparator independent of Laridi-style overlap construction | LOCKED | Documentation | VERIFIED |

---

## BASELINE-REGISTRY - Baseline Registry and Classification

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| BASELINE-REGISTRY-001 | Lines 96-116 | Baseline registry: B0-B10 with complete definitions | LOCKED | fedcrg.baselines.registry | VERIFIED |
| BASELINE-REGISTRY-002 | Lines 96-116 | B0-B2: Benign-only baselines | LOCKED | Classification | VERIFIED |
| BASELINE-REGISTRY-003 | Lines 96-116 | B3-B6: Ablations and published-style | LOCKED | Classification | VERIFIED |
| BASELINE-REGISTRY-004 | Lines 96-116 | B7-B9: Attack-aware comparators | LOCKED | Classification | VERIFIED |
| BASELINE-REGISTRY-005 | Lines 96-116 | B10: Oracle | LOCKED | Classification | VERIFIED |
| BASELINE-REGISTRY-006 | Lines 96-116 | ALL_POLICIES list contains all policy IDs | LOCKED | fedcrg.config.PolicyID | VERIFIED |
| BASELINE-REGISTRY-007 | Lines 96-116 | Baseline factory functions for each policy | LOCKED | fedcrg.baselines.registry | VERIFIED |
| BASELINE-REGISTRY-008 | Lines 96-116 | Benign-only vs attack-aware classification | LOCKED | fedcrg.baselines.base.BaselineType | VERIFIED |

---

## Summary Statistics

| Category | Total | Implemented | Verified | Missing |
|---|---:|---:|---:|---:|
| BASELINE-PRINCIPLE | 5 | 5 | 5 | 0 |
| BASELINE-DEFINITIONS | 11 | 11 | 11 | 0 |
| QUANTILE-CONVENTION | 6 | 6 | 6 | 0 |
| SHRINKAGE | 9 | 9 | 9 | 0 |
| ATTACK-AWARE | 8 | 8 | 8 | 0 |
| LARIDI-COMPARATOR | 19 | 19 | 19 | 0 |
| SUP-F1-1000 | 9 | 9 | 9 | 0 |
| BASELINE-REGISTRY | 8 | 8 | 8 | 0 |
| **Total** | **75** | **75** | **75** | **0** |

---

## Current Implementation Status

**Baseline suite: COMPLETE**

- Baseline principle and fairness: COMPLETE and VERIFIED
- All 11 baselines (B0-B10): COMPLETE and VERIFIED
- Deterministic quantile-rank ledger: COMPLETE and VERIFIED
- Shrinkage baseline exact tuning rule: COMPLETE and VERIFIED
- Attack-aware local-versus-global selector: COMPLETE and VERIFIED
- Laridi et al. 2024 closest-prior comparator: COMPLETE and VERIFIED
- SUP-F1-1000 exact extra-information comparator: COMPLETE and VERIFIED
- Baseline registry and classification: COMPLETE and VERIFIED

## Verification Evidence

- All 11 baseline implementations exist in fedcrg.baselines/
- Baseline registry contains all B0-B10 with metadata
- Factory functions create correct baseline instances
- All baseline types classified correctly (benign-only vs attack-aware)
- Quantile conventions match exact roadmap formulas

## Next Steps

- Create metrics and evaluation matrix (06_metrics_requirements.md)
- Create experiments matrix (07_experiment_requirements.md)
- Create implementation and artifacts matrix (08_implementation_requirements.md)
- Create testing and validation matrix (09_testing_requirements.md)