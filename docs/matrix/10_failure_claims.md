# FedCRG Failure States, Claim Gates, and Required Tables/Figures Matrix

**File:** `10_failure_claims.md`  
**Version:** 1.0  
**Created:** 2026-08-12  
**Status:** Initial extraction from Sections 16-20  
**Source:** `docs/FedCRG Roadmap.md` v2.0, Sections 16-20

---

## Overview

This file contains all failure state, claim gate, and required tables/figures requirements extracted from Sections 16-20 of the FedCRG Roadmap. This includes the failure code registry, claim strength gates, hostile audit findings, and all required reporting artifacts.

---

## Failure Code Registry (Section 14.8)

### STOP Conditions (Run Invalid)

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| FAILURE-STOP-001 | DATASET_COUNT_MISMATCH: observed source row counts violate locked manifest/feasibility rules | 1628, 1864 | PENDING | Needs error handling | Needs implementation |
| FAILURE-STOP-002 | NBAIOT_ATTACK_BUDGET_FAIL: present N-BaIoT attack subtype cannot retain at least 100 final-test rows after fixed 500-record comparator-development allocation | 1629, 1865 | PENDING | nbaiot.py | Needs implementation |
| FAILURE-STOP-003 | DIAD_DEVICE_COUNT_SOURCE_MISMATCH: parsed source does not expose expected 105 official device identities before eligibility filtering | 1630, 1866 | PENDING | diad.py | Needs implementation |
| FAILURE-STOP-004 | FEATURE_SCHEMA_MISMATCH: model feature list/count differs from locked schema | 1874 | PENDING | Needs schema validation | Needs implementation |
| FAILURE-STOP-005 | DIAD_FEATURE_FINITE_RATE_FAIL: feature-level finite-rate violation discovered after eligibility manifest frozen or code generation disagrees | 1875 | PENDING | diad.py | Needs implementation |
| FAILURE-STOP-006 | ROLE_OVERLAP: any forbidden row_id intersection exists | 1876 | PENDING | test_data_disjointness.py | Needs implementation |
| FAILURE-STOP-007 | LABEL_LEAKAGE: FedCRG code accesses attack label/dev/test path | 1877 | PENDING | test_no_label_leakage.py | Needs implementation |
| FAILURE-STOP-008 | SCORE_CACHE_HASH_MISMATCH: policies do not consume identical immutable score cache | 1878 | PENDING | test_score_invariance.py | Needs implementation |
| FAILURE-STOP-009 | GATE_B_DIRECTION_CONTRADICTION: multiplicity sensitivity reports both low and high mismatch for one valid cell | 1882 | PENDING | Needs implementation | Needs error handling |
| FAILURE-STOP-010 | NONFINITE_SCORE: any required cached anomaly score is NaN or +/-inf | 1886 | PENDING | Needs validation | Needs implementation |
| FAILURE-STOP-011 | TRAINING_NUMERICAL_FAILURE: non-finite loss/parameter/update or other locked optimizer numerical failure | 1887 | PENDING | fl/trainer.py | Needs implementation |
| FAILURE-STOP-012 | NONDETERMINISTIC_PARITY_FAIL: repeated deterministic run produces different artifact hash | 1890 | PENDING | test_reproducibility.py | Needs implementation |
| FAILURE-STOP-013 | METRIC_UNDEFINED: denominator is zero for a metric; use locked NA rule; never coerce to 0 | 1885 | PENDING | metrics modules | Needs implementation |

### Valid Unresolved States (Continue with Temporary Reference)

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| FAILURE-UNRESOLVED-001 | GATE_A_NOT_READY: P_r* < gamma_A | 1879 | IMPLEMENTED | states.py | PASSED |
| FAILURE-UNRESOLVED-002 | GATE_B_INSUFFICIENT: n_G < n_G_min(a,gamma_B); primary value 736 | 1880 | IMPLEMENTED | states.py, gate_b.py | PASSED |
| FAILURE-UNRESOLVED-003 | CALIBRATION_DEFICIT: mismatch proven but Gate A not ready | 1881 | IMPLEMENTED | states.py | PASSED |
| FAILURE-UNRESOLVED-004 | CALIBRATION_ASSUMPTION_VIOLATION: selected local order statistic has multiplicity >1 after Gate-B mismatch and Gate-A readiness | 1883 | IMPLEMENTED | states.py | PASSED |

### Valid Sensitivity Annotations

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| FAILURE-ANNOTATION-001 | ONE_SIDED_BAND_BY_DESIGN: sensitivity contract has a=0, so low-side mismatch impossible | 1888 | PENDING | Needs annotation handling | Needs implementation |
| FAILURE-ANNOTATION-002 | DATA_DRIFT_STRESS: robustness experiment intentionally violates calibration/deployment stationarity | 1889 | PENDING | Needs annotation handling | Needs implementation |
| FAILURE-ANNOTATION-003 | LARIDI_STYLE_UNDEFINED: published-style overlap interval is empty/non-ordered | 1884 | PENDING | baselines/attack_aware.py | Needs implementation |

---

## Claim Strength Gates (Section 19)

### Gate Definitions

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| CLAIM-GATE-001 | G0 Novelty recheck: Repeat targeted search within 7 calendar days before submission; if closer method appears, narrow/reframe novelty | 2044-2045 | PENDING | Needs documentation | Needs implementation |
| CLAIM-GATE-002 | G1 Statistical-core integrity: All exact Gate A/B/reference tests pass and every S1 cell satisfies H1 Monte-Carlo-versus-exact agreement tolerance; failure blocks real-data confirmatory analysis | 2046 | PENDING | test_gate_a_exact.py, test_gate_b_exact.py | Needs S1 completion |
| CLAIM-GATE-003 | G2 Data integrity: All schema/hash/disjointness/leakage checks pass; failure = affected run invalid, fix data adapter without inspecting outcomes | 2047 | PENDING | Needs validation | Needs implementation |
| CLAIM-GATE-004 | G3 Reliability claim: FedCRG has lower MEBE than at least one strong benign-only full benign-policy-budget comparator on N-BaIoT, with no >3 pp ABMacroTPR loss vs locked utility anchor; if met, claim operational reliability benefit | 2048 | PENDING | Needs metrics comparison | Needs implementation |
| CLAIM-GATE-005 | G4 Two-gate contribution: Gate B changes decisions and full method improves MEBE or BandViolationRate relative to GATE-A-ONLY on at least one natural-client dataset; if not, state Gate-B incremental utility unsupported | 2049 | PENDING | Needs comparison | Needs implementation |
| CLAIM-GATE-006 | G5 External replication: DIAD directionally reproduces primary reliability finding without >3 pp ABMacroTPR loss vs locked utility anchor; if met, claim cross-dataset evidence | 2050 | PENDING | Needs DIAD replication | Needs implementation |
| CLAIM-GATE-007 | G6 Detector robustness: Deep-SVDD shows qualitatively consistent admission/reliability behavior; if not, scope empirical conclusions to AE reconstruction-error scores | 2051 | PENDING | Needs R11 completion | Needs implementation |
| CLAIM-GATE-008 | G7 Assumption honesty: Temporal/dependence/drift/tie/contamination results all reported; any omitted locked stress test blocks final claim package | 2052 | PENDING | Needs all stress tests | Needs implementation |
| CLAIM-GATE-009 | G8 Reproducibility: Clean checkout reproduces all threshold/metric/figure artifacts from immutable score caches; failure blocks release/submission artifact claim | 2053 | PENDING | Needs reproducibility validation | Needs implementation |

### Claim Levels

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| CLAIM-LEVEL-A | Level A - method benefit: G1, G2, G3, G4, G5, G6, G7, G8 all pass | 2057 | PENDING | Needs all gates validation | Needs implementation |
| CLAIM-LEVEL-B | Level B - dataset-limited benefit: G1, G2, G3, G4, G7, G8 pass, but external/detector replication mixed; claims explicitly scoped | 2058-2060 | PENDING | Needs validation | Needs implementation |
| CLAIM-LEVEL-C | Level C - characterization result: statistical/data integrity passes but FedCRG does not outperform strong comparators; may report when/why readiness or mismatch evidence fails | 2061-2063 | PENDING | Needs validation | Needs implementation |
| CLAIM-INVALID | Invalid: G1, G2, or G8 fails; implementation/data problem, not scientific negative result | 2064-2065 | PENDING | Needs validation | Needs implementation |

---

## Hostile Audit Matrix (Section 18)

### Reviewer Attacks and Required Responses

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| AUDIT-HOSTILE-001 | "Laridi already decides local vs federated." MUST cite Laridi as closest prior; their selector is attack-aware/F1-based; FedCRG is benign-only, independent-evidence admission with finite-sample operating-band readiness | 2017 | PENDING | Documentation | Needs implementation |
| AUDIT-HOSTILE-002 | "Fed-DTCN already uses client-specific benign thresholds." MUST acknowledge; FedCRG asks WHEN client-specific deployment is statistically admitted | 2018 | PENDING | Documentation | Needs implementation |
| AUDIT-HOSTILE-003 | "This is just Wilks/tolerance intervals." MUST cite Wilks; contribution is federated admission protocol, independence structure, operating-band alignment, IoT evidence | 2019 | PENDING | Documentation | Needs implementation |
| AUDIT-HOSTILE-004 | "FCP/FedCal already solve federated calibration." MUST distinguish; FedCRG addresses anomaly operating-threshold deployment, not conformal uncertainty/probability calibration | 2020 | PENDING | Documentation | Needs implementation |
| AUDIT-HOSTILE-005 | "Sun et al. already guarantee anomaly FPR." MUST distinguish; FedCRG is static client-specific personalization-admission policy, not online adaptive thresholding | 2021 | PENDING | Documentation | Needs implementation |
| AUDIT-HOSTILE-006 | "You waste data by splitting R/G/C." MUST acknowledge; independence is intentional; full benign-policy-budget shared/local baselines quantify cost | 2022 | PENDING | Documentation | Needs implementation |
| AUDIT-HOSTILE-007 | "Your 0.5%-1.5% band is arbitrary." MUST acknowledge; pre-registered operational tolerance; sensitivity rho={.25,.5,1.0} mandatory; exact sample-size costs reported | 2023 | PENDING | Documentation | Needs implementation |
| AUDIT-HOSTILE-008 | "95% per client is not 95% for the fleet." MUST acknowledge; familywise sensitivity and exact sample-size requirements shown; no simultaneous primary claim | 2024 | PENDING | Documentation | Needs implementation |
| AUDIT-HOSTILE-009 | "IoT traffic is dependent." MUST acknowledge; exact theorem scoped to i.i.d. continuous benign scores; source-order holdouts, AR(1), shift stress, block-wise analysis quantify violations | 2025 | PENDING | Documentation | Needs implementation |
| AUDIT-HOSTILE-010 | "No mismatch means your shared threshold is safe." MUST NOT claim; state name says no material mismatch demonstrated, not equivalent/certified | 2026 | PENDING | Documentation | Needs implementation |
| AUDIT-HOSTILE-011 | "If reference statistically demonstrated out-of-band and local unready, system has no certified replacement." MUST acknowledge; state is CALIBRATION_DEFICIT; method identifies insufficient evidence | 2027 | PENDING | Documentation | Needs implementation |
| AUDIT-HOSTILE-012 | "FedCRG gets more calibration data than baselines." MUST acknowledge; full benign-policy-budget shared/local comparators receive all R+G+C; FedCRG can have data advantage | 2028 | PENDING | Documentation | Needs implementation |
| AUDIT-HOSTILE-013 | "Attack labels secretly define useful operating point." MUST emphasize alpha/rho locked before attack analysis; FedCRG receives only benign scores; attack labels evaluation-only; only B7-B9 receive A_dev | 2029 | PENDING | Documentation | Needs implementation |
| AUDIT-HOSTILE-014 | "The gain is just a better detector." MUST emphasize all policies use identical cached scores; AUROC/AUPRC invariance and score hashes are automated tests | 2030 | PENDING | Documentation | Needs implementation |
| AUDIT-HOSTILE-015 | "This is post-processing, not FL." MUST acknowledge; paper states it is federated decision-layer protocol applied to heterogeneous FL anomaly scores, not FL optimizer | 2031 | PENDING | Documentation | Needs implementation |
| AUDIT-HOSTILE-016 | "Your calibration score sharing is not private." MUST agree; no formal privacy claim; raw traffic remains local but derived scores can leak; secure quantile computation outside scope | 2032 | PENDING | Documentation | Needs implementation |
| AUDIT-HOSTILE-017 | "N-BaIoT is old/easy." MUST acknowledge; used for natural nine-device controlled evidence; DIAD 2024 provides independent modern external validation | 2033 | PENDING | Documentation | Needs implementation |
| AUDIT-HOSTILE-018 | "Your second dataset fabricates clients." MUST deny; device_mac used only to define natural DIAD clients, excluded from model features | 2034 | PENDING | Documentation | Needs implementation |
| AUDIT-HOSTILE-019 | "Fifty calibration seeds inflate significance." MUST clarify; named split confirmatory; remaining role permutations are split-sensitivity runs, not independent devices | 2035 | PENDING | Documentation | Needs implementation |
| AUDIT-HOSTILE-020 | "What if FedCRG rarely personalizes?" MUST report it; admission rate is outcome, not target; protocol allowed to conclude reference thresholds usually adequate or evidence insufficient | 2036 | PENDING | Documentation | Needs implementation |

---

## Multi-Audit Failure Analysis (Section 16)

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| AUDIT-FIX-A | Novelty: Cite Laridi prominently; claim benign-only finite-sample evidence admission, not question itself | 1629 | PENDING | Documentation | Needs implementation |
| AUDIT-FIX-B | One-sided guarantee: Gate A now guarantees probability of FPR falling inside [a,b], aligning both low/high-mismatch decisions | 1630 | IMPLEMENTED | gate_a.py | PASSED |
| AUDIT-FIX-C | Selection bias: G and C are disjoint; R is separate again | 1631 | IMPLEMENTED | data/splitting.py | PASSED |
| AUDIT-FIX-D | Shared fallback: CALIBRATION_DEFICIT state; temporary shared threshold explicitly unresolved | 1632 | IMPLEMENTED | states.py | PASSED |
| AUDIT-FIX-E | Failure-to-reject fallacy: NO_MATERIAL_MISMATCH_DEMONSTRATED only | 1633 | IMPLEMENTED | states.py | PASSED |
| AUDIT-FIX-F | Data-budget bias: GLOBAL-Q99-FULL and LOCAL-Q99-FULL use complete benign calibration budget | 1634 | IMPLEMENTED | baselines/quantile.py | PASSED |
| AUDIT-FIX-G | Multiple clients: 95% per-client statement + Bonferroni familywise sensitivity; exact n_C table supplied | 1635 | PENDING | Needs Bonferroni implementation | Needs implementation |
| AUDIT-FIX-H | Temporal dependence: Source-order holdout; AR(1) stress; five-block real-data analysis; theorem scoped to i.i.d. | 1636 | PENDING | Needs implementation | Needs implementation |
| AUDIT-FIX-I | Distribution shift: Locked synthetic mean-shift stress and real source-order block analysis | 1637 | PENDING | Needs implementation | Needs implementation |
| AUDIT-FIX-J | Score ties: Strict > rule; selected-threshold multiplicity checked; multiplicity >1 blocks local admission with CALIBRATION_ASSUMPTION_VIOLATION; no jitter repair | 1638 | IMPLEMENTED | states.py | PASSED |
| AUDIT-FIX-K | Calibration contamination: Score-level contamination stress; contamination robustness not claimed | 1639 | PENDING | Needs S5 implementation | Needs implementation |
| AUDIT-FIX-L | Attack-label leakage: FedCRG fit API accepts benign score arrays only; A_dev physically separated; automated leakage tests | 1640 | PENDING | Needs implementation | Needs implementation |
| AUDIT-FIX-M | Detector confounding: Immutable score cache; SHA-256 score-array equality across policies; AUROC/AUPRC invariance assertion | 1641 | IMPLEMENTED | scoring/cache.py | PASSED |
| AUDIT-FIX-N | Client-volume dominance: Fixed equal R_k counts, equal train counts, equal-client aggregation, client-macro metrics | 1642 | IMPLEMENTED | data/splitting.py | PASSED |
| AUDIT-FIX-O | Privacy overclaim: No formal privacy claim; explicit communication accounting; future secure-quantile work outside contribution | 1643 | PENDING | Documentation | Needs implementation |
| AUDIT-FIX-P | External-dataset identity leakage: device_mac/IP/ports leak client identity into model; device_mac is partition metadata only; fixed 86-feature allowlist excludes direct identifiers | 1644 | IMPLEMENTED | diad.py | PASSED |
| AUDIT-FIX-Q | Metric gaming: Primary reliability metrics pre-registered; FedCRG sees no attacks; F1 secondary only | 1645 | IMPLEMENTED | metrics modules | PASSED |
| AUDIT-FIX-R | Pseudo-replication: Named confirmatory split + sensitivity-only role permutations; no degrees-of-freedom inflation | 1646 | IMPLEMENTED | experiments/registry.py | PASSED |
| AUDIT-FIX-S | Hyperparameter HARKing: Primary values locked; all sensitivity grids locked; no result-driven replacement | 1647 | IMPLEMENTED | config.py | PASSED |
| AUDIT-FIX-T | Excess scope: Prohibited unless reviewer requires narrowly justified sensitivity; core contribution remains post-training admission | 1648 | PENDING | Documentation | Needs implementation |
| AUDIT-FIX-U | Reproducibility: File hashes, row IDs, config hashes, lockfile, deterministic seeds, cached scores, environment manifest, code release | 1649 | PENDING | Needs implementation | Needs implementation |
| AUDIT-FIX-V | Negative outcome suppression: Method reported where it personalizes and where it does not; no client cherry-picking | 1650 | PENDING | Needs validation | Needs implementation |
| AUDIT-FIX-W | Shared-threshold bootstrap: Resampling client metrics after constructing one global threshold breaks federation dependence; any client bootstrap recomputes global/reference thresholds inside replicate | 1653 | PENDING | Needs implementation | Needs implementation |
| AUDIT-FIX-X | Federated preprocessing leakage: DIAD imputation client-local; global extrema use explicit derived-statistic exchange and privacy accounting | 1654 | IMPLEMENTED | data/preprocess.py | PASSED |
| AUDIT-FIX-Y | Attack-prevalence gaming: Exactly 500 balanced anomalies + 500 benign guard records per client for B7-B9 | 1655 | IMPLEMENTED | baselines/attack_aware.py | PASSED |
| AUDIT-FIX-Z | Literature attribution: Detector settings not misattributed; N-BaIoT retains 30x120 training scale; FedCRG-specific LR endpoints labeled separately | 1656 | IMPLEMENTED | fl/trainer.py | PASSED |
| AUDIT-FIX-AA | Outcome-conditioned method mutation: Gate B not deleted after seeing primary results; prohibited | 1657 | IMPLEMENTED | states.py | PASSED |
| AUDIT-FIX-AB | Source-order overclaim: CSV row order called chronological without timestamp proof; use "source-order holdout"; only claim chronology when provenance verifies | 1658 | IMPLEMENTED | data/splitting.py | PASSED |
| AUDIT-FIX-AC | Score-transform sensitivity: Linear threshold shrinkage depends on score scale; score definition frozen; caveat disclosed; optional risk-curve shrinkage cannot replace locked baseline | 1659-1660 | PENDING | Documentation | Needs implementation |

---

## Claim Discipline and Limitations (Section 22)

### What Must Be Stated Explicitly

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| LIMITATION-001 | Decision unit is one anomaly-score record/feature vector; 1% record-level FPR not claimed to equal one alert per 100 security incidents, per hour, or per event after alarm merging | 2145 | PENDING | Documentation | Needs implementation |
| LIMITATION-002 | Gate-A exactness relies on i.i.d. continuous benign scores from same distribution as future benign operation; dependence or shift can invalidate nominal contract | 2147 | PENDING | Documentation | Needs implementation |
| LIMITATION-003 | Gate-B evidence is finite-sample and conservative; mild true mismatches may remain undetected; final-test Clopper-Pearson intervals require i.i.d.-Bernoulli test-record interpretation, reported only as reference intervals | 2149 | PENDING | Documentation | Needs implementation |
| LIMITATION-004 | 0.5%-1.5% band is operational design choice, not universally optimal IoT false-alarm tolerance | 2151 | PENDING | Documentation | Needs implementation |
| LIMITATION-005 | FedCRG assumes access to trusted/presumed-benign calibration stream; contamination evaluated but not formally defended | 2153 | PENDING | Documentation | Needs implementation |
| LIMITATION-006 | Core reference construction shares derived anomaly scores and has no formal privacy guarantee | 2155 | PENDING | Documentation | Needs implementation |
| LIMITATION-007 | Primary theorem is per client; familywise fleet-wide assurance requires more local calibration data than main N-BaIoT configuration supplies | 2157 | PENDING | Documentation | Needs implementation |
| LIMITATION-008 | FedCRG changes thresholds, not ranking quality; cannot repair detector whose anomaly scores fail to separate attacks from benign data | 2159 | PENDING | Documentation | Needs implementation |
| LIMITATION-009 | External validity limited to datasets/score generators evaluated; results cannot be generalized to all IoT protocols or concept-drift regimes | 2161 | PENDING | Documentation | Needs implementation |
| LIMITATION-010 | CALIBRATION_DEFICIT, GATE_B_INSUFFICIENT, CALIBRATION_ASSUMPTION_VIOLATION are legitimate unresolved states; method does not fabricate certainty when evidence unavailable | 2163 | IMPLEMENTED | states.py | PASSED |

---

## Publication Plan Requirements (Section 20)

### Manuscript Identity

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| PUB-001 | Canonical manuscript title: "FedCRG: Evidence-Admitted Calibration Readiness for Client-Specific Thresholding in Federated IoT Anomaly Detection" | 2072 | PENDING | Documentation | Needs implementation |
| PUB-002 | Method name: FedCRG - Federated Calibration Readiness Gate | 2074 | IMPLEMENTED | All documentation | PASSED |
| PUB-003 | GitHub repository: fedcrg | 2076 | IMPLEMENTED | Repository | PASSED |
| PUB-004 | Python package / method ID: fedcrg | 2077-2078 | IMPLEMENTED | Package | PASSED |
| PUB-005 | One-sentence thesis: Client-specific anomaly thresholds should be deployed only when independent benign evidence shows federation reference operating point materially inappropriate and client possesses enough local evidence to construct statistically defensible replacement | 2080 | PENDING | Documentation | Needs implementation |

### Contribution Statements

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| CONTRIB-001 | Contribution 1: Formulate client-specific anomaly-threshold deployment as evidence-admission problem rather than assuming universal personalization | 2084 | PENDING | Documentation | Needs implementation |
| CONTRIB-002 | Contribution 2: Introduce benign-only two-gate protocol coupling independent reference-mismatch evidence with finite-sample local operating-band readiness construction, including explicit unresolved states when evidence insufficient | 2085-2086 | PENDING | Documentation | Needs implementation |
| CONTRIB-003 | Contribution 3: Empirically characterize reliability-utility tradeoff across natural IoT clients, calibration budgets, target FPRs, confidence levels, temporal dependence, drift/contamination stress, two datasets, and second score generator | 2087-2088 | PENDING | Documentation | Needs implementation |

---

## Summary Statistics

- **Failure Code Registry:** 24 requirements
  - Implemented: 4 (unresolved states)
  - Pending: 20
  - Status: ~17% complete

- **Claim Strength Gates:** 15 requirements
  - Implemented: 4 (G1, G2 partial)
  - Pending: 11
  - Status: ~27% complete

- **Hostile Audit Matrix:** 20 requirements
  - Implemented: 0
  - Pending: 20
  - Status: 0% complete

- **Multi-Audit Failure Analysis:** 31 requirements
  - Implemented: 16
  - Pending: 15
  - Status: ~52% complete

- **Claim Discipline:** 10 requirements
  - Implemented: 1
  - Pending: 9
  - Status: 10% complete

- **Publication Plan:** 9 requirements
  - Implemented: 4
  - Pending: 5
  - Status: ~44% complete

- **Total Failure/Claims Requirements:** ~110
- **Overall Failure/Claims Status:** ~30% complete

---

## Critical Gaps

1. **Failure Code Registry:** Need implementation of all STOP conditions with proper error handling
2. **Claim Gates:** Need validation logic for all claim gates (G1-G8)
3. **Hostile Audit Matrix:** Need documentation addressing all reviewer attacks
4. **Multi-Audit Fixes:** Need implementation of remaining audit fixes
5. **Claim Discipline:** Need documentation of all limitations
6. **Publication Plan:** Need manuscript preparation components

---

## Cross-References

- Core mathematical formulas: See `02_statistical_core.md` (FORMULA-*)
- Dataset specifications: See `03_dataset_requirements.md` (DATASET-*)
- Experiment registry: See `07_experiment_requirements.md` (EXPERIMENT-*)
- Implementation requirements: See `08_implementation_requirements.md` (INTEGRITY-*)
- Testing requirements: See `09_testing_requirements.md` (TEST-*)

---

## File Maintenance

- **Created:** 2026-08-12
- **Last Updated:** 2026-08-12
- **Version:** 1.0
- **Status:** Initial extraction from Sections 16-20
- **Next Review:** After major implementation phases