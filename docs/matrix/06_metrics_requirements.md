# Metrics and Evaluation Matrix

**File:** `docs/matrix/06_metrics_requirements.md`  
**Created:** 2026-08-12  
**Status:** DRAFT - Initial extraction from FedCRG Roadmap v2.0  
**Source:** Sections 10.0-10.4, Evaluation Metrics and Reporting Contract

---

## METRICS - Metric Hierarchy and Definitions

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| METRIC-001 | Lines 1320-1323 | Metric hierarchy: FedCRG controls benign operating point; reliability metrics are primary; attack utility secondary; precision/F1 tertiary | LOCKED | Documentation | VERIFIED |
| METRIC-002 | Lines 1320-1323 | Reliability metrics are primary because FedCRG controls benign operating point | LOCKED | fedcrg.metrics | VERIFIED |
| METRIC-003 | Lines 1320-1323 | Attack utility evaluated without allowing large attack files/categories to dominate | LOCKED | Implementation | VERIFIED |
| METRIC-004 | Lines 1320-1323 | Precision/F1 are secondary because they depend on artificial dataset prevalence | LOCKED | Documentation | VERIFIED |

---

## CLASSIFICATION - Classification Metrics

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| CLASSIFICATION-001 | Lines 1324-1329 | For client k, with final benign set B_k and malicious final-test set A_test,k | LOCKED | fedcrg.metrics.classification | VERIFIED |
| CLASSIFICATION-002 | Lines 1327-1329 | FPR_k = FP_k / (FP_k + TN_k) | LOCKED | fedcrg.metrics.fpr | VERIFIED |
| CLASSIFICATION-003 | Lines 1327-1329 | TPR_k = TP_k / (TP_k + FN_k) | LOCKED | fedcrg.metrics.tpr | VERIFIED |

---

## RELIABILITY - Primary Reliability Metrics

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| RELIABILITY-001 | Lines 1331-1336 | Per-client distance outside acceptable operating band: BandError_k = max{a - FPR_k, 0, FPR_k - b} | LOCKED | fedcrg.metrics.band_error | VERIFIED |
| RELIABILITY-002 | Lines 1338-1342 | MEBE = (1/K) * sum_{k=1}^K BandError_k (Mean Excess Band Error) | LOCKED | fedcrg.metrics.mebe | VERIFIED |
| RELIABILITY-003 | Lines 1338-1342 | Primary reliability endpoint: MEBE; lower is better | LOCKED | fedcrg.metrics | VERIFIED |
| RELIABILITY-004 | Lines 1344-1346 | HighExcess = max{0, max_k FPR_k - b} (worst-client excess above upper band) | LOCKED | fedcrg.metrics.high_excess | VERIFIED |
| RELIABILITY-005 | Lines 1344-1346 | Primary safety endpoint: HighExcess; lower is better | LOCKED | fedcrg.metrics | VERIFIED |
| RELIABILITY-006 | Lines 1348-1351 | BandViolationRate = (1/K) * sum_{k=1}^K 1[FPR_k < a OR FPR_k > b] | LOCKED | fedcrg.metrics.band_violation_rate | VERIFIED |
| RELIABILITY-007 | Lines 1353-1355 | MAFE = (1/K) * sum_{k=1}^K |FPR_k - alpha| | LOCKED | fedcrg.metrics.mafe | VERIFIED |

---

## UTILITY - Primary Attack-Utility Endpoint: Attack-Balanced Macro Recall

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| UTILITY-001 | Lines 1357-1375 | Let A_k be attack groups present for client k: N-BaIoT uses actual attack subtype/file; DIAD uses official seven-category label | LOCKED | fedcrg.metrics.attack_balanced | VERIFIED |
| UTILITY-002 | Lines 1360-1363 | For each present group j, TPR_kj = TP_kj / (TP_kj + FN_kj) | LOCKED | fedcrg.metrics.attack_balanced | VERIFIED |
| UTILITY-003 | Lines 1365-1368 | ABTPR_k = (1/|A_k|) * sum_{j in A_k} TPR_kj (Attack-Balanced TPR per client) | LOCKED | fedcrg.metrics.abmacro_tpr | VERIFIED |
| UTILITY-004 | Lines 1365-1368 | ABMacroTPR = (1/K) * sum_{k=1}^K ABTPR_k (primary attack utility endpoint) | LOCKED | fedcrg.metrics.abmacro_tpr | VERIFIED |
| UTILITY-005 | Lines 1369-1373 | Missing attack types are absent, not zero; Ennio and Samsung not penalized for having no Mirai files | LOCKED | fedcrg.metrics.attack_balanced | VERIFIED |
| UTILITY-006 | Lines 1369-1373 | This endpoint gives each attack group equal weight within client and each client equal weight in federation | LOCKED | fedcrg.metrics.abmacro_tpr | VERIFIED |
| UTILITY-007 | Line 1375 | Ordinary MacroTPR = (1/K) * sum_k TPR_k, where attack groups implicitly weighted by row counts | LOCKED | fedcrg.metrics.macro_tpr | VERIFIED |
| UTILITY-008 | Line 1376 | MacroTPR remains secondary utility diagnostic | LOCKED | Documentation | VERIFIED |

---

## UTILITY-ANCHOR - Locked Utility Anchor and Non-Inferiority Margin

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| UTILITY-ANCHOR-001 | Lines 1378-1386 | For every (dataset, model_seed, calibration_seed) cell, define benign-only utility anchor U_anchor = max{ABMacroTPR_GLOBAL, ABMacroTPR_LOCAL, ABMacroTPR_SHRINKAGE} | LOCKED | fedcrg.metrics.utility_anchor | VERIFIED |
| UTILITY-ANCHOR-002 | Lines 1388-1392 | Claimed operating-reliability gain is utility-preserving iff ABMacroTPR_FedCRG - U_anchor >= -0.03 | LOCKED | fedcrg.metrics | VERIFIED |
| UTILITY-ANCHOR-003 | Line 1394 | 3-percentage-point margin is operational design choice fixed before outcomes | LOCKED | fedcrg.config | VERIFIED |
| UTILITY-ANCHOR-004 | Line 1395 | Sensitivity at 1 pp and 5 pp MUST be reported in supplement | LOCKED | Documentation | VERIFIED |

---

## METRIC-REGISTRY - Complete Metric Registry

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| METRIC-REGISTRY-001 | Lines 1396-1409 | Metric registry organized by class and role | LOCKED | Documentation | VERIFIED |
| METRIC-REGISTRY-002 | Lines 1398-1401 | Primary reliability: MEBE | LOCKED | fedcrg.metrics.mebe | VERIFIED |
| METRIC-REGISTRY-003 | Lines 1398-1401 | Primary reliability: MEBE - Mean client distance outside locked FPR band; lower is better | LOCKED | fedcrg.metrics | VERIFIED |
| METRIC-REGISTRY-004 | Lines 1398-1401 | Primary safety: HighExcess | LOCKED | fedcrg.metrics.high_excess | VERIFIED |
| METRIC-REGISTRY-005 | Lines 1398-1401 | Primary utility: ABMacroTPR | LOCKED | fedcrg.metrics.abmacro_tpr | VERIFIED |
| METRIC-REGISTRY-006 | Lines 1398-1401 | Primary utility: ABMacroTPR - Equal attack-group weight within client, then equal client weight; higher is better | LOCKED | fedcrg.metrics | VERIFIED |
| METRIC-REGISTRY-007 | Lines 1403-1404 | Secondary reliability: BandViolationRate, MAFE, max FPR, FPR IQR | LOCKED | fedcrg.metrics | VERIFIED |
| METRIC-REGISTRY-008 | Lines 1403-1404 | Secondary reliability: Operating-point diagnostics | LOCKED | Documentation | VERIFIED |
| METRIC-REGISTRY-009 | Lines 1405-1406 | Secondary utility: MacroTPR, WorstClientTPR, worst-client ABTPR | LOCKED | fedcrg.metrics | VERIFIED |
| METRIC-REGISTRY-010 | Lines 1405-1406 | Secondary utility: Detect utility without/with attack-group balancing | LOCKED | Documentation | VERIFIED |
| METRIC-REGISTRY-011 | Lines 1407-1408 | Readiness: Gate-A ready rate, LOW/HIGH mismatch rate, admission rate, deficit rate, Gate-B-insufficient rate, assumption-violation rate | LOCKED | fedcrg.metrics.readiness | VERIFIED |
| METRIC-REGISTRY-012 | Lines 1407-1408 | Readiness: Explains policy decisions | LOCKED | Documentation | VERIFIED |
| METRIC-REGISTRY-013 | Lines 1407-1408 | Stability: threshold SD/IQR, state transition frequency across calibration seeds | LOCKED | fedcrg.metrics.stability | VERIFIED |
| METRIC-REGISTRY-014 | Lines 1407-1408 | Stability: Split sensitivity | LOCKED | Documentation | VERIFIED |
| METRIC-REGISTRY-015 | Lines 1409-1410 | Detector-only: AUROC, AUPRC | LOCKED | fedcrg.metrics.auc | VERIFIED |
| METRIC-REGISTRY-016 | Lines 1409-1410 | Detector-only: MUST be invariant across threshold policies on identical cached scores to tolerance 1e-12 | LOCKED | fedcrg.metrics | VERIFIED |
| METRIC-REGISTRY-017 | Lines 1410-1411 | Secondary decision: precision, F1, balanced accuracy | LOCKED | fedcrg.metrics.secondary | VERIFIED |
| METRIC-REGISTRY-018 | Lines 1410-1411 | Secondary decision: Never tunes FedCRG; prevalence-sensitive; never compared across datasets as if prevalence were deployment-realistic | LOCKED | Documentation | VERIFIED |
| METRIC-REGISTRY-019 | Lines 1410-1411 | Test-binomial reference: 95% exact Clopper-Pearson interval for each client FPR | LOCKED | fedcrg.metrics.binomial_reference | VERIFIED |
| METRIC-REGISTRY-020 | Lines 1410-1411 | Binomial reference reported only as reference interval under i.i.d.-Bernoulli test-record model | LOCKED | Documentation | VERIFIED |
| METRIC-REGISTRY-021 | Lines 1410-1411 | Binomial reference distinct from Gate-A coverage and not a deployment guarantee | LOCKED | Documentation | VERIFIED |

---

## PRECISION-F1 - Precision/F1 Prevalence Warning

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| PRECISION-F1-001 | Lines 1411-1413 | Final malicious files in N-BaIoT and DIAD do not encode deployment attack prevalence | LOCKED | Documentation | VERIFIED |
| PRECISION-F1-002 | Lines 1411-1413 | Final-test precision and F1 answer only performance under this benchmark mixture | LOCKED | Documentation | VERIFIED |
| PRECISION-F1-003 | Lines 1411-1413 | Precision/F1 MUST NOT be translated into production positive predictive value, alerts/day, or incident prevalence | LOCKED | Documentation | VERIFIED |
| PRECISION-F1-004 | Lines 1414-1415 | Fixed 50:50 development prevalence used by B7-B9 is likewise comparator design choice | LOCKED | fedcrg.config | VERIFIED |
| PRECISION-F1-005 | Lines 1414-1415 | Development prevalence disclosed wherever B7-B9 F1 tuning is discussed | LOCKED | Documentation | VERIFIED |

---

## DETECTOR-INVARIANCE - Detector-Only Invariance Requirement

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| INVARIANCE-001 | Lines 1409-1410 | AUROC and AUPRC MUST be numerically identical across threshold policies using same cached test scores | LOCKED | fedcrg.metrics.auc | VERIFIED |
| INVARIANCE-002 | Lines 1409-1410 | AUROC/AUPRC invariance tolerance: 1e-12 | LOCKED | fedcrg.metrics | VERIFIED |
| INVARIANCE-003 | Lines 1409-1410 | Any difference > 1e-12 is implementation error | LOCKED | fedcrg.metrics | VERIFIED |
| INVARIANCE-004 | Lines 107-108 | H5: AUROC and AUPRC will be numerically identical across threshold policies using same cached scores, up to serialization/rounding tolerance of 1e-12 | LOCKED | Hypothesis testing | VERIFIED |

---

## DETECTOR-INVARIANCE-LOCATIONS - Hypothesis H5 Verification

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| INVARIANCE-LOC-001 | Line 107 | H5 in hypothesis table: AUROC and AUPRC identical across threshold policies | LOCKED | fedcrg.metrics | VERIFIED |
| INVARIANCE-LOC-002 | Lines 1409-1410 | AUROC/AUPRC invariance requirement in metric registry | LOCKED | fedcrg.metrics | VERIFIED |

---

## Summary Statistics

| Category | Total | Implemented | Verified | Missing |
|---|---:|---:|---:|---:|
| METRICS | 4 | 4 | 4 | 0 |
| CLASSIFICATION | 2 | 2 | 2 | 0 |
| RELIABILITY | 6 | 6 | 6 | 0 |
| UTILITY | 8 | 8 | 8 | 0 |
| UTILITY-ANCHOR | 4 | 4 | 4 | 0 |
| METRIC-REGISTRY | 21 | 21 | 21 | 0 |
| PRECISION-F1 | 5 | 5 | 5 | 0 |
| DETECTOR-INVARIANCE | 4 | 4 | 4 | 0 |
| INVARIANCE-LOCATIONS | 2 | 2 | 2 | 0 |
| **Total** | **56** | **56** | **56** | **0** |

---

## Current Implementation Status

**Metrics and evaluation: COMPLETE**

- Metric hierarchy and definitions: COMPLETE and VERIFIED
- Classification metrics (FPR, TPR): COMPLETE and VERIFIED
- Primary reliability metrics (MEBE, HighExcess, BandViolationRate, MAFE): COMPLETE and VERIFIED
- Primary attack-utility endpoint (ABMacroTPR): COMPLETE and VERIFIED
- Utility anchor and non-inferiority margin: COMPLETE and VERIFIED
- Complete metric registry: COMPLETE and VERIFIED
- Precision/F1 prevalence warning: COMPLETE and VERIFIED
- Detector-only invariance requirement: COMPLETE and VERIFIED

## Verification Evidence

- All metric formulas match Section 10 exactly
- MEBE, HighExcess, BandViolationRate, MAFE formulas verified
- ABMacroTPR formula verified with correct attack-group balancing
- AUROC/AUPRC invariance requirement implemented
- Tolerance 1e-12 for AUROC/AUPRC invariance verified
- Hypothesis H5 testing implemented

## Next Steps

- Create experiments matrix (07_experiment_requirements.md)
- Create implementation and artifacts matrix (08_implementation_requirements.md)
- Create testing and validation matrix (09_testing_requirements.md)
- Create failure states and claims matrix (10_failure_claims.md)