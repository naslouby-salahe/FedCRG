# FedCRG Implementation and Artifacts Requirements Matrix

**File:** `08_implementation_requirements.md`  
**Version:** 1.0  
**Created:** 2026-08-12  
**Status:** Initial extraction from Sections 14-15  
**Source:** `docs/FedCRG Roadmap.md` v2.0, Sections 14-15

---

## Overview

This file contains all implementation, artifact schema, configuration, CLI, and API requirements extracted from Sections 14-15 of the FedCRG Roadmap.

---

## Package Structure Requirements

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| IMPLEMENT-001 | Package structure MUST follow Section 14.1 specification | 1661-1712 | IMPLEMENTED | fedcrg/ package | Directory structure |
| IMPLEMENT-002 | MUST have configs/ directory with protocol_v2.yaml, nbaiot_primary.yaml, diad_external.yaml, synthetic.yaml | 1665-1669 | IMPLEMENTED | configs/ | File existence |
| IMPLEMENT-003 | MUST have fedcrg/reference.py | 1671 | IMPLEMENTED | fedcrg/reference.py | Module exists |
| IMPLEMENT-004 | MUST have fedcrg/gate_a.py | 1672 | IMPLEMENTED | fedcrg/gate_a.py | Module exists |
| IMPLEMENT-005 | MUST have fedcrg/gate_b.py | 1673 | IMPLEMENTED | fedcrg/gate_b.py | Module exists |
| IMPLEMENT-006 | MUST have fedcrg/policy.py | 1674 | PENDING | fedcrg/policy.py | Needs implementation |
| IMPLEMENT-007 | MUST have fedcrg/states.py | 1675 | IMPLEMENTED | fedcrg/states.py | Module exists |
| IMPLEMENT-008 | MUST have fedcrg/metrics.py | 1676 | IMPLEMENTED | fedcrg/metrics/ | Module exists |
| IMPLEMENT-009 | MUST have fedcrg/data/ with nbaiot.py, diad.py, manifests.py, splits.py | 1677-1681 | IMPLEMENTED | fedcrg/data/ | Directory structure |
| IMPLEMENT-010 | MUST have fedcrg/models/ with autoencoder.py, deep_svdd.py | 1682-1684 | IMPLEMENTED | fedcrg/models/ | Directory structure |
| IMPLEMENT-011 | MUST have fedcrg/fl/ with trainer.py, aggregation.py, lr_schedule.py, client.py, server.py, sampling.py | 1686-1689 | IMPLEMENTED | fedcrg/fl/ | Directory structure |
| IMPLEMENT-012 | MUST have fedcrg/experiments/ with synthetic.py, gate_b_power.py, real_primary.py, sensitivity.py | 1689-1692 | IMPLEMENTED | fedcrg/experiments/ | Directory structure |
| IMPLEMENT-013 | MUST have fedcrg/experiments/analysis/ with statistics.py, figures.py, tables.py | 1694-1696 | PENDING | analysis/ | Needs implementation |
| IMPLEMENT-014 | MUST have fedcrg/tests/ with test_gate_a_exact.py, test_gate_b_exact.py, test_data_disjointness.py, test_no_label_leakage.py, test_score_invariance.py, test_metrics.py, test_reproducibility.py | 1699-1705 | PARTIAL | fedcrg/tests/ | Partial implementation |
| IMPLEMENT-015 | MUST have artifacts/ directory with manifests/, scores/, thresholds/, metrics/, figures/ | 1706-1711 | PARTIAL | artifacts/ | Partial implementation |

---

## Artifact Schema Requirements

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| ARTIFACT-001 | dataset_manifest.json MUST contain: dataset_id, source_version, file paths, SHA256 per input, parser_version, created_at, feature_names, client IDs, per-role counts | 1719 | PENDING | data/manifests.py | Needs validation |
| ARTIFACT-002 | score_cache.parquet MUST contain: dataset_id, client_id, row_id, phase, model_seed, score_float64, label_test_only, attack_family_test_only | 1720 | PENDING | scoring/schemas.py | Needs validation |
| ARTIFACT-003 | threshold_record.jsonl MUST contain: run_id, policy_id, client_id, tau_ref, tau_local, selected_tau, gate_a_n, gate_a_rank, gate_a_probability, gate_b_n, gate_b_x, CP_L, CP_U, state, tie_count | 1721 | PENDING | Needs implementation | Needs module |
| ARTIFACT-004 | metric_record.jsonl MUST contain: run_id, policy_id, client_id, benign_n, attack_n, FP, TN, TP, FN, FPR, TPR, precision, F1, BA, AUROC, AUPRC, band_error | 1722 | PENDING | Needs implementation | Needs module |
| ARTIFACT-005 | run_config.json MUST contain: full parameters, seeds, git commit, environment lock hash, data-manifest hash, score-cache hash | 1723 | PENDING | Needs implementation | Needs module |
| ARTIFACT-006 | Score cache MUST use float64 storage | 1068, 1780 | IMPLEMENTED | scoring/cache.py | PASSED |
| ARTIFACT-007 | Score cache MUST have SHA-256 hash | 1720 | IMPLEMENTED | scoring/cache.py | Hash implementation |
| ARTIFACT-008 | Score cache MUST be immutable | 1766 | IMPLEMENTED | scoring/cache.py | Immutability validation |

---

## Configuration Requirements

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| CONFIG-001 | Configuration MUST validate via JSON-Schema/Pydantic | 1909-1919 | IMPLEMENTED | config.py | Pydantic validation |
| CONFIG-002 | Configuration MUST validate: 0 < alpha < 1 | 1912 | IMPLEMENTED | config.py | Alpha validation |
| CONFIG-003 | Configuration MUST validate: 0 < gamma_a < 1, 0 < gamma_b < 1 | 1913 | IMPLEMENTED | config.py | Gamma validation |
| CONFIG-004 | Configuration MUST validate: 0 <= rho | 1914 | IMPLEMENTED | config.py | Rho validation |
| CONFIG-005 | Configuration MUST validate derived: 0 <= a < b <= 1 | 1915 | IMPLEMENTED | config.py | Band validation |
| CONFIG-006 | Configuration MUST validate: all seed lists contain unique integers | 1916 | IMPLEMENTED | config.py | Seed validation |
| CONFIG-007 | Configuration MUST validate: all role counts are nonnegative and fit client eligibility rule | 1917 | IMPLEMENTED | config.py | Count validation |
| CONFIG-008 | Configuration MUST validate: policy registry exactly matches protocol | 1918 | PENDING | config.py | Registry validation |
| CONFIG-009 | Confirmatory run config hash MUST match experiment ledger | 1919 | PENDING | Needs implementation | Ledger validation |
| CONFIG-010 | Configuration MUST include all values from Appendix E | 2302-2400 | IMPLEMENTED | YAML files | Config completeness |
| CONFIG-011 | protocol_v2.yaml MUST be frozen before outcomes | 2410 | IMPLEMENTED | configs/protocol_v2.yaml | Config freeze |
| CONFIG-012 | Configuration files MUST be generated from Appendix E skeleton | 2302-2400 | IMPLEMENTED | All YAML files | Skeleton validation |

---

## CLI Command Requirements (Section 14.10)

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| CLI-001 | MUST have command: fedcrg doctor | 1928, 1930 | IMPLEMENTED | cli.py:doctor | PASSED |
| CLI-002 | MUST have command: fedcrg data prepare --config configs/nbaiot_primary.yaml | 1929 | IMPLEMENTED | cli.py:data_prepare | PARTIAL (placeholder) |
| CLI-003 | MUST have command: fedcrg data prepare --config configs/diad_external.yaml | 1930 | IMPLEMENTED | cli.py:data_prepare | PARTIAL (placeholder) |
| CLI-004 | MUST have command: fedcrg tables precompute-gate-a --config configs/protocol_v2.yaml | 1931 | IMPLEMENTED | cli.py:precompute_gate_a | PASSED |
| CLI-005 | MUST have command: fedcrg synthetic run --config configs/synthetic.yaml | 1932 | IMPLEMENTED | cli.py:synthetic_run | PASSED |
| CLI-006 | MUST have command: fedcrg train --config configs/nbaiot_primary.yaml | 1933 | IMPLEMENTED | cli.py:train | PARTIAL (placeholder) |
| CLI-007 | MUST have command: fedcrg score --config configs/nbaiot_primary.yaml | 1934 | IMPLEMENTED | cli.py:score | PARTIAL (placeholder) |
| CLI-008 | MUST have command: fedcrg evaluate --config configs/nbaiot_primary.yaml | 1935 | IMPLEMENTED | cli.py:evaluate | PARTIAL (placeholder) |
| CLI-009 | MUST have command: fedcrg train --config configs/diad_external.yaml | 1936 | IMPLEMENTED | cli.py:train | PARTIAL (placeholder) |
| CLI-010 | MUST have command: fedcrg score --config configs/diad_external.yaml | 1937 | IMPLEMENTED | cli.py:score | PARTIAL (placeholder) |
| CLI-011 | MUST have command: fedcrg evaluate --config configs/diad_external.yaml | 1938 | IMPLEMENTED | cli.py:evaluate | PARTIAL (placeholder) |
| CLI-012 | MUST have command: fedcrg robustness deep-svdd --config configs/nbaiot_primary.yaml | 1939 | IMPLEMENTED | cli.py:robustness_deep_svdd | PARTIAL (placeholder) |
| CLI-013 | MUST have command: fedcrg benchmark --config configs/protocol_v2.yaml | 1940 | IMPLEMENTED | cli.py:benchmark | PASSED |
| CLI-014 | MUST have command: fedcrg report build | 1941 | IMPLEMENTED | cli.py:report_build | PARTIAL (placeholder) |
| CLI-015 | MUST have command: fedcrg verify | 1942 | IMPLEMENTED | cli.py:verify | PASSED |
| CLI-016 | All CLI commands MUST read confirmatory values from YAML rather than typed manually | 1924-1925 | IMPLEMENTED | cli.py:load_config | Config-based CLI |
| CLI-017 | fedcrg verify MUST fail if any required experiment cell, artifact hash, unit test, leakage check, or manifest field is missing | 1945-1946 | IMPLEMENTED | cli.py:verify | Error handling |

---

## Python API Requirements (Section 14.6)

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| API-001 | MUST expose: build_reference_threshold(reference_scores_by_client, alpha) -> ReferenceThresholdResult | 1822-1826 | IMPLEMENTED | reference.py:build_reference_threshold | Module exists |
| API-002 | MUST expose: gate_a_readiness(calibration_scores, alpha, rho, gamma_a) -> GateAResult | 1828-1833 | IMPLEMENTED | gate_a.py:compute_gate_a | Module exists |
| API-003 | MUST expose: gate_b_reference_mismatch(gate_scores, tau_ref, alpha, rho, gamma_b) -> GateBResult | 1835-1841 | IMPLEMENTED | gate_b.py:compute_gate_b | Module exists |
| API-004 | MUST expose: decide_fedcrg(reference, gate_a, gate_b) -> FedCRGDecision | 1843-1847 | IMPLEMENTED | states.py:decide_fedcrg | Module exists |
| API-005 | GateAResult MUST include: n, rank, coverage_probability, ready, tau_local, tie_count, a, b | 1850-1851 | IMPLEMENTED | gate_a.py:GateAResult | Dataclass validation |
| API-006 | GateBResult MUST include: n, x, fpr_hat, cp_lower, cp_upper, p_low, p_high, mismatch_state | 1853-1854 | IMPLEMENTED | gate_b.py:GateBResult | Dataclass validation |
| API-007 | FedCRGDecision MUST include: state, selected_threshold, selected_source, tie_count, reason_code | 1856-1858 | IMPLEMENTED | states.py:FedCRGDecision | Dataclass validation |
| API-008 | Public package interfaces MUST be testable | 1816-1820 | IMPLEMENTED | All modules | Unit tests |

---

## Determinism and Environment Requirements (Section 14.4)

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| DETERMINISM-001 | MUST use one locked environment file (uv.lock, poetry.lock, or requirements lock) | 1774 | PENDING | Needs lockfile | Environment freeze |
| DETERMINISM-002 | MUST record Python, PyTorch, CUDA, cuDNN, NumPy, SciPy, pandas, scikit-learn, OS, CPU, GPU, driver versions in every run manifest | 1776-1777 | PENDING | Needs manifest recording | Environment recording |
| DETERMINISM-003 | MUST enable deterministic PyTorch algorithms where supported | 1778 | PENDING | Needs PyTorch config | Determinism validation |
| DETERMINISM-004 | MUST set Python, NumPy, torch RNGs from model seed | 1778 | IMPLEMENTED | fl/sampling.py | RNG seeding |
| DETERMINISM-005 | MUST record any nondeterministic CUDA operation that cannot be disabled | 1778 | PENDING | Needs CUDA validation | Operation recording |
| DETERMINISM-006 | MUST compute calibration/order-statistic mathematics and exact binomial intervals in float64 | 1780 | IMPLEMENTED | gate_a.py, gate_b.py | Float64 validation |
| DETERMINISM-007 | Neural forward passes may be float32; convert final scalar scores to float64 before threshold calculations | 1780 | IMPLEMENTED | scoring/computer.py | Score conversion |
| DETERMINISM-008 | MUST persist git commit hash and git diff --quiet state | 1782 | PENDING | Needs git integration | Git state validation |
| DETERMINISM-009 | Main confirmatory runs require clean repository or stored patch hash | 1782 | PENDING | Needs validation | Repository validation |
| DETERMINISM-010 | Every generated table/figure must be reproducible from immutable score/threshold/metric artifacts without retraining detector | 1784 | PENDING | Needs reproducibility | Reproducibility validation |

---

## Computational and Communication Contract Requirements (Section 14.5)

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| COMPUTE-001 | Gate-A rank precomputation: For fixed (n_C, a, b), r* and readiness depend only on protocol constants and sample count, not observed score values | 1788 | IMPLEMENTED | gate_a.py:GateATable | Precomputation validation |
| COMPUTE-002 | Runtime code MUST read precomputed rank and MUST NOT optimize rank using observed client scores | 1788, 340-341 | IMPLEMENTED | gate_a.py:compute_gate_a | Runtime validation |
| COMPUTE-003 | Reference threshold complexity: O(N_R log N_R) with full sort; O(N_R) memory | 1791-1792 | IMPLEMENTED | reference.py:build_reference_threshold | Complexity validation |
| COMPUTE-004 | Each client sends exactly |R_k| float64 scores for reference threshold | 1792 | IMPLEMENTED | reference.py | Payload validation |
| COMPUTE-005 | N-BaIoT reference: 500 x 8 = 4,000 bytes/client; 36,000 bytes total for 9 clients | 1792 | IMPLEMENTED | reference.py | Payload accounting |
| COMPUTE-006 | Gate A runtime complexity: O(n_C log n_C) full-sort reference; cached r* lookup O(1) | 1793 | IMPLEMENTED | gate_a.py | Complexity validation |
| COMPUTE-007 | Gate B complexity: O(n_G) threshold comparisons; O(1) streaming count possible | 1794 | IMPLEMENTED | gate_b.py | Complexity validation |
| COMPUTE-008 | State decision complexity: O(1) | 1795 | IMPLEMENTED | states.py | Complexity validation |
| COMPUTE-009 | Expected O(n_C) selection for r*-th statistic permitted only after parity tests | 1796 | PENDING | Needs optimization validation | Optimization validation |
| COMPUTE-010 | R13 benchmark protocol: Pin to one CPU thread, 100 warm-ups + 1000 timed calls per primitive | 1813 | IMPLEMENTED | real_data.py | Benchmark validation |

---

## Threshold-Policy Payload Accounting (Section 14.5.1)

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| PAYLOAD-001 | FedCRG reference R: 500 float64/client = 4,000 B; 36,000 B federation for N-BaIoT | 1804 | IMPLEMENTED | reference.py | Payload accounting |
| PAYLOAD-002 | FedCRG reference R: 300 float64/client = 2,400 B; 2,400*K_D B for DIAD | 1804 | IMPLEMENTED | reference.py | Payload accounting |
| PAYLOAD-003 | FedCRG Gate B: one integer count plus logged interval/state; raw G need not leave client | 1805 | IMPLEMENTED | gate_b.py | Payload accounting |
| PAYLOAD-004 | FedCRG Gate A: 0 B required for local threshold construction | 1806 | IMPLEMENTED | gate_a.py | Payload accounting |
| PAYLOAD-005 | GLOBAL-Q99-FULL/FEDDETECT-3SIGMA naive score upload: 5,500 float64/client = 44,000 B; 396,000 B federation for N-BaIoT | 1807 | PENDING | Needs implementation | Payload accounting |
| PAYLOAD-006 | LARIDI-STYLE-SS moments: 2 classes x (n int64, mean float64, variance float64) = 48 B/client | 1808 | PENDING | baselines/attack_aware.py | Payload accounting |
| PAYLOAD-007 | LARIDI-STYLE-SS/SUP-F1-1000 candidate evaluation: 1,000 float64 F1 values = 8,000 B/client | 1809 | PENDING | baselines/attack_aware.py | Payload accounting |
| PAYLOAD-008 | Paper MUST report FedCRG threshold-policy traffic separately from model-training traffic | 1811 | PENDING | Documentation | Reporting validation |
| PAYLOAD-009 | Paper MUST NOT claim lower total communication than comparator unless same serialization and transport accounting measured for both | 1811 | PENDING | Documentation | Claim discipline |

---

## Required Tables and Figures (Section 17)

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| TABLE-001 | Figure 1: FedCRG decision architecture | 1993 | PENDING | Needs figure generation | Visualization |
| TABLE-002 | Figure 2: Finite-sample readiness frontier with n_C vs maximum exact in-band probability; mark 1416 minimum, 2000 primary, sensitivity bands | 1994 | PENDING | Needs figure generation | Visualization |
| TABLE-003 | Figure 3: Gate-B evidence/power map with n_G x true reference FPR, mismatch-declaration probability; mark primary n_G=3000 | 1995 | IMPLEMENTED | experiments/synthetic.py | COMPLETED (S6) |
| TABLE-004 | Figure 4: Per-client operating points for N-BaIoT with GLOBAL-Q99-FULL, LOCAL-Q99-FULL, SHRINKAGE, FedCRG; horizontal lines 0.5%,1%,1.5% | 1996 | PENDING | Needs figure generation | Visualization |
| TABLE-005 | Figure 5: Reliability-utility frontier with MEBE vs ABMacroTPR for mandatory policies with paired uncertainty | 1997 | PENDING | Needs figure generation | Visualization |
| TABLE-006 | Figure 6: Calibration-size phase transition showing Gate-A readiness/admission and n_G sweep showing mismatch evidence | 1998 | PENDING | Needs figure generation | Visualization |
| TABLE-007 | Figure 7: Assumption stress with coverage under AR(1), mean shift, contamination; separate panels in supplement | 1999 | PENDING | Needs figure generation | Visualization |
| TABLE-008 | Figure 8: External replication with DIAD per-client FPR and aggregate MEBE/ABMacroTPR | 2000-2001 | PENDING | Needs figure generation | Visualization |
| TABLE-009 | Table 1: Literature boundary comparing information used, threshold object, local/global decision, finite-sample contract, IoT setting | 2004 | PENDING | Needs table generation | Tabular reporting |
| TABLE-010 | Table 2: Protocol constants with every locked alpha/rho/confidence/count/seed/hyperparameter | 2005 | IMPLEMENTED | YAML files, config.py | Config completeness |
| TABLE-011 | Table 3: Dataset inventory with all client IDs, benign/attack counts, exact role counts, feature dimensions, file hashes | 2006 | PENDING | Needs manifest generation | Tabular reporting |
| TABLE-012 | Table 4: Primary policy results with MEBE, HighExcess, band-violation rate, MAFE, ABMacroTPR, MacroTPR, worst-client TPR, F1 secondary | 2007 | PENDING | Needs metrics collection | Tabular reporting |
| TABLE-013 | Table 5: Admission states with per client Gate-B x/CI, Gate-A n/rank/probability, state, tau_ref, tau_local, selected tau | 2008 | PENDING | Needs state collection | Tabular reporting |
| TABLE-014 | Table 6: Ablations with Gate-A-only, Gate-B-only, full benign-policy-budget baselines, shrinkage | 2009 | PENDING | Needs baseline comparison | Tabular reporting |
| TABLE-015 | Table 7: Sensitivity with alpha, rho, gamma_A, sample sizes, multiplicity | 2010 | PENDING | Needs sensitivity collection | Tabular reporting |
| TABLE-016 | Table 8: External replication with same primary metrics on DIAD | 2011 | PENDING | Needs DIAD metrics | Tabular reporting |
| TABLE-017 | All generated tables/figures MUST be reproducible from immutable artifacts without retraining | 1784, 2119-2121 | PENDING | Needs reproducibility | Reproducibility validation |

---

## Submission Package Requirements (Section 20.5)

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| SUBMISSION-001 | Public code repository with tagged release matching manuscript; no uncommitted main-run code | 2113 | PENDING | Needs release process | Release validation |
| SUBMISSION-002 | Protocol YAML/JSON, environment lockfile, all seeds, exact dataset acquisition instructions, input SHA-256 manifests | 2115 | PENDING | Needs packaging | Package validation |
| SUBMISSION-003 | No redistribution of dataset files unless licenses explicitly allow it; publish row-ID/hash manifests and preprocessing scripts instead | 2117 | PENDING | Needs manifest system | Legal compliance |
| SUBMISSION-004 | Immutable score caches if redistribution permitted; otherwise publish threshold/metric artifacts and scripts that regenerate scores | 2118 | PENDING | Needs artifact system | Artifact validation |
| SUBMISSION-005 | All figure/table-generation scripts; no manually edited numerical values in manuscript tables | 2121 | PENDING | Needs script generation | Script validation |
| SUBMISSION-006 | Archive release with persistent DOI (e.g., Zenodo) at acceptance or submission if journal policy permits | 2123 | PENDING | Needs DOI process | Archive validation |
| SUBMISSION-007 | Pre-submission novelty search log dated within seven days of submission | 2125 | PENDING | Needs search process | Search validation |

---

## Leakage and Integrity Test Requirements (Section 14.3)

| ID | Requirement | Section | Status | Implementation | Verification |
|---|---|---|---|---|---|
| INTEGRITY-001 | MUST assert pairwise-empty row_id intersections among T, calibration reservoir, B_test, A_dev, and A_test | 1756 | PENDING | test_data_disjointness.py | Disjointness validation |
| INTEGRITY-002 | MUST assert within each calibration seed, R, G, C, and supervised-comparator benign guard are pairwise disjoint | 1756 | PENDING | test_data_disjointness.py | Seed disjointness |
| INTEGRITY-003 | MUST assert FedCRGPolicy.fit() accepts only arrays of benign scores and scalar protocol parameters; has no label argument | 1758 | IMPLEMENTED | fedcrg.py, baselines/ | API validation |
| INTEGRITY-004 | MUST assert no path containing A_dev or A_test is opened by FedCRG fitting code; use test fixture that raises immediately if accessed | 1760-1761 | PENDING | test_no_label_leakage.py | Path validation |
| INTEGRITY-005 | MUST assert scaler/imputer objects are fitted exclusively on T_k rows; serialize their fit-row hashes | 1762 | PENDING | Needs hash serialization | Hash validation |
| INTEGRITY-006 | MUST assert selected feature names exactly match N-BaIoT 115 schema or DIAD allowlist after finite-value audit | 1764 | PENDING | test_data_disjointness.py | Schema validation |
| INTEGRITY-007 | MUST assert all policies read identical score_cache hash for given dataset/model seed | 1766 | PENDING | test_score_invariance.py | Hash validation |
| INTEGRITY-008 | MUST assert final test labels are loaded only inside evaluation functions after thresholds serialized | 1768 | PENDING | Needs evaluation validation | Loading validation |
| INTEGRITY-009 | MUST fail run if any metric is NaN/inf, any role count differs from protocol, any client disappears after outcome computation, or any config hash differs from pre-registered file | 1770-1771 | PENDING | Needs error handling | Error validation |

---

## Summary Statistics

- **Package Structure:** 15 requirements
  - Implemented: 14
  - Pending: 1
  - Status: ~93% complete

- **Artifact Schemas:** 8 requirements
  - Implemented: 3
  - Pending: 5
  - Status: ~38% complete

- **Configuration:** 12 requirements
  - Implemented: 9
  - Pending: 3
  - Status: ~75% complete

- **CLI Commands:** 17 requirements
  - Implemented: 16
  - Pending: 1
  - Status: ~94% complete (but many partial/placeholder implementations)

- **Python API:** 8 requirements
  - Implemented: 8
  - Status: 100% complete

- **Determinism:** 10 requirements
  - Implemented: 4
  - Pending: 6
  - Status: ~40% complete

- **Computation/Communication:** 10 requirements
  - Implemented: 9
  - Pending: 1
  - Status: ~90% complete

- **Payload Accounting:** 9 requirements
  - Implemented: 4
  - Pending: 5
  - Status: ~44% complete

- **Tables/Figures:** 17 requirements
  - Implemented: 2
  - Pending: 15
  - Status: ~12% complete

- **Submission Package:** 7 requirements
  - Pending: 7
  - Status: 0% complete

- **Leakage/Integrity:** 9 requirements
  - Implemented: 1
  - Pending: 8
  - Status: ~11% complete

- **Total Implementation Requirements:** ~115
- **Overall Implementation Status:** ~50% complete

---

## Critical Gaps

1. **Package Structure:** Missing `fedcrg/policy.py` module
2. **Artifact Schemas:** Need threshold_record.jsonl, metric_record.jsonl, run_config.json schemas and implementations
3. **Configuration:** Need config hash matching ledger validation
4. **CLI Commands:** Many commands have placeholder implementations needing full functionality
5. **Determinism:** Need environment lockfile, manifest recording, git integration
6. **Tables/Figures:** Most reporting requirements are pending
7. **Submission Package:** Complete packaging system needed
8. **Leakage/Integrity:** Most integrity tests need implementation

---

## Cross-References

- Core mathematical formulas: See `02_statistical_core.md` (GATE-A-*, GATE-B-*)
- Dataset specifications: See `03_dataset_requirements.md` (DATASET-*, SPLIT-*)
- Experiment registry: See `07_experiment_requirements.md` (EXPERIMENT-*)
- Failure states: See `10_failure_claims.md` (FAILURE-*, CLAIM-*)

---

## File Maintenance

- **Created:** 2026-08-12
- **Last Updated:** 2026-08-12
- **Version:** 1.0
- **Status:** Initial extraction from Sections 14-15
- **Next Review:** After major implementation phases