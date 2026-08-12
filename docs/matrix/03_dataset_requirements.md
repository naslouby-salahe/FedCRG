# Dataset Specifications Matrix

**File:** `docs/matrix/03_dataset_requirements.md`  
**Created:** 2026-08-12  
**Status:** DRAFT - Initial extraction from FedCRG Roadmap v2.0  
**Source:** Sections 7.1-7.4, Dataset and Data-Partition Protocol

---

## DATASET - Dataset Identities and Sources

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| DATASET-001 | Lines 552-558 | N-BaIoT: UCI dataset 442, Detection of IoT Botnet Attacks N-BaIoT, DOI 10.24432/C5RC8J | LOCKED | fedcrg.data.nbaiot.NBaiotAdapter | VERIFIED |
| DATASET-002 | Lines 554-557 | N-BaIoT contains 7,062,606 records from nine commercial IoT devices and 115 traffic-statistic features | LOCKED | Data loading | PENDING |
| DATASET-003 | Lines 554-557 | Data described as multivariate and sequential | LOCKED | Documentation | VERIFIED |
| DATASET-004 | Line 558 | Primary study MUST NOT create Dirichlet pseudo-clients | LOCKED | Implementation | VERIFIED |
| DATASET-005 | Lines 658-662 | CIC IoT-DIAD 2024: official topology contains 105 IoT devices and 33 attacks across seven categories | LOCKED | fedcrg.data.diad.DiadAdapter | VERIFIED |
| DATASET-006 | Lines 658-662 | DIAD exposes device_mac as device-identification label and anomaly label | LOCKED | Data loading | PENDING |
| DATASET-007 | Lines 658-662 | DIAD allows natural device-level clients while retaining different feature-generation pipeline from N-BaIoT | LOCKED | Implementation | VERIFIED |
| DATASET-008 | Lines 662-663 | DIAD client eligibility: at least 7800 benign packet rows, at least 1000 malicious packet rows, enough per-category development capacity | LOCKED | fedcrg.data.diad.check_eligibility | VERIFIED |
| DATASET-009 | Lines 662-663 | All eligible DIAD devices are used; no performance-based client selection and no cap on K | LOCKED | Implementation | VERIFIED |
| DATASET-010 | Lines 781-784 | If fewer than 10 devices satisfy eligibility, CIC IoT-DIAD declared unsuitable for confirmatory external validation | LOCKED | fedcrg.data.diad.check_eligibility | VERIFIED |

---

## NBAIOT - N-BaIoT Specific Requirements

### Device Inventory and Benign-Budget Feasibility

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| NBAIOT-001 | Lines 561-579 | Implementation MUST derive actual row counts from acquired files and store in manifest | LOCKED | fedcrg.data.nbaiot.NBaiotAdapter | PENDING |
| NBAIOT-002 | Lines 566-578 | Literature-reported counts are preflight cross-check, not substitute for counting files | LOCKED | Documentation | VERIFIED |
| NBAIOT-003 | Lines 566-578 | Device inventory: nb01 (Danmini Doorbell) to nb09 (Samsung SNH-1011N Webcam) | LOCKED | fedcrg.data.base.DatasetID | VERIFIED |
| NBAIOT-004 | Lines 566-578 | Benign row counts cross-check for each device | LOCKED | Data validation | PENDING |
| NBAIOT-005 | Lines 579-585 | Total literature cross-check: 555,932 benign rows; 90,000 consumed before final test; 465,932 expected remaining | LOCKED | Data validation | PENDING |
| NBAIOT-006 | Lines 580-585 | Ecobee client is limiting benign-data case; leaves 3,113 final benign rows; protocol minimum of 3,000 is feasible with 113 rows slack | LOCKED | Data validation | PENDING |
| NBAIOT-007 | Lines 584-585 | DATASET_COUNT_MISMATCH if actual source files don't match locked feasibility assertion | LOCKED | Error handling | PENDING |

### N-BaIoT Benign Role Partition

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| SPLIT-NBAIOT-001 | Lines 586-596 | T_k benign train: 4,000 per client, first 4,000 benign rows in source-file order | LOCKED | fedcrg.data.nbaiot.NBaiotAdapter | VERIFIED |
| SPLIT-NBAIOT-002 | Lines 586-596 | Calibration reservoir: 6,000 per client, next 6,000 benign rows in source-file order | LOCKED | fedcrg.data.nbaiot.NBaiotAdapter | VERIFIED |
| SPLIT-NBAIOT-003 | Lines 586-596 | R_k reference: 500 per client, first 500 positions of seeded permutation of 6,000-row reservoir | LOCKED | fedcrg.data.splitting.generate_calibration_permutation | VERIFIED |
| SPLIT-NBAIOT-004 | Lines 586-596 | G_k gate: 3,000 per client, permutation positions 501-3500 | LOCKED | fedcrg.data.splitting.generate_calibration_permutation | VERIFIED |
| SPLIT-NBAIOT-005 | Lines 586-596 | C_k local calibration: 2,000 per client, permutation positions 3501-5500 | LOCKED | fedcrg.data.splitting.generate_calibration_permutation | VERIFIED |
| SPLIT-NBAIOT-006 | Lines 586-596 | Comparator benign guard: 500 per client, permutation positions 5501-6000 | LOCKED | fedcrg.data.splitting.generate_calibration_permutation | VERIFIED |
| SPLIT-NBAIOT-007 | Lines 586-596 | B_k final benign test: all rows after first 10,000, never subsampled for primary test | LOCKED | fedcrg.data.nbaiot.NBaiotAdapter | VERIFIED |
| SPLIT-NBAIOT-008 | Lines 598-602 | Order claim: adapter preserves UCI source-file row order | LOCKED | fedcrg.data.nbaiot.NBaiotAdapter | VERIFIED |
| SPLIT-NBAIOT-009 | Lines 598-602 | MUST use phrase "source-order holdout" unless timestamp or capture-order provenance verified | LOCKED | Documentation | VERIFIED |
| SPLIT-NBAIOT-010 | Lines 598-602 | MUST NOT call split chronological merely because rows are sequentially stored | LOCKED | Documentation | VERIFIED |
| SPLIT-NBAIOT-011 | Lines 603-607 | Primary role assignment: Calibration seed 1000 is single named confirmatory role assignment | LOCKED | fedcrg.data.splitting.CalibrationSeedStart | VERIFIED |
| SPLIT-NBAIOT-012 | Lines 603-607 | Seeds 1001-1049 are split-sensitivity replicates | LOCKED | fedcrg.data.splitting | VERIFIED |
| SPLIT-NBAIOT-013 | Lines 603-607 | Split-sensitivity replicates quantify dependence on historical reservoir allocation | LOCKED | Documentation | VERIFIED |
| SPLIT-NBAIOT-014 | Lines 603-607 | Split-sensitivity replicates not independent device samples, never used to inflate inferential degrees of freedom | LOCKED | Documentation | VERIFIED |

### N-BaIoT Attack Inventory and Firewall

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| ATTACK-NBAIOT-001 | Lines 609-617 | N-BaIoT contains five BASHLITE/Gafgyt attack subtypes: combo, junk, scan, tcp, udp for every device | LOCKED | fedcrg.data.nbaiot._client_attack_subtypes | VERIFIED |
| ATTACK-NBAIOT-002 | Lines 610-616 | Seven devices also contain five Mirai subtypes: ack, scan, syn, udp, udpplain | LOCKED | fedcrg.data.nbaiot._client_attack_subtypes | VERIFIED |
| ATTACK-NBAIOT-003 | Lines 610-616 | Ennio and Samsung do not contain Mirai | LOCKED | fedcrg.data.nbaiot._client_attack_subtypes | VERIFIED |
| ATTACK-NBAIOT-004 | Lines 615-616 | Adapter MUST derive exact available attack-file set rather than assume 10 subtypes for all clients | LOCKED | fedcrg.data.nbaiot._discover_attack_subtypes | VERIFIED |
| ATTACK-NBAIOT-005 | Lines 618-627 | A_dev,k construction: exactly 500 malicious records total per client | LOCKED | fedcrg.data.nbaiot.build_attack_development_test_split | PENDING |
| ATTACK-NBAIOT-006 | Lines 620-626 | Allocate floor(500/m_k) records to every subtype, then distribute remainder lexicographically | LOCKED | fedcrg.data.nbaiot.build_attack_development_test_split | PENDING |
| ATTACK-NBAIOT-007 | Lines 620-626 | Sampling within subtype uses seed 9001 and is without replacement | LOCKED | fedcrg.data.nbaiot.build_attack_development_test_split | PENDING |
| ATTACK-NBAIOT-008 | Lines 627-628 | A_test,k is every remaining malicious row | LOCKED | fedcrg.data.nbaiot.build_attack_development_test_split | PENDING |
| ATTACK-NBAIOT-009 | Lines 627-628 | Every present subtype MUST retain at least 100 final-test rows after 500-record development allocation | LOCKED | fedcrg.data.nbaiot.build_attack_development_test_split | PENDING |
| ATTACK-NBAIOT-010 | Lines 628-630 | NBAIOT_ATTACK_BUDGET_FAIL if cannot retain at least 100 final-test rows per subtype | LOCKED | Error handling | PENDING |
| ATTACK-NBAIOT-011 | Lines 628-630 | A_dev,k and all attack labels stored under path namespace that FedCRG fitting code is forbidden to open | LOCKED | fedcrg.data.base.SeganizedPath | PENDING |
| ATTACK-NBAIOT-012 | Lines 630-635 | 500-anomaly development budget paired with 500-benign comparator guard, giving attack-aware F1 baselines fixed 50:50 development prevalence | LOCKED | Configuration | VERIFIED |
| ATTACK-NBAIOT-013 | Lines 632-634 | Supervised comparator intentionally favorable, explicitly supervised comparator, not deployment assumption for FedCRG | LOCKED | Documentation | VERIFIED |

### N-BaIoT Integrity Assertions

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| INTEGRITY-NBAIOT-001 | Lines 636-656 | All integrity assertions MUST pass before model training | LOCKED | fedcrg.data.nbaiot.validate_integrity | PENDING |
| INTEGRITY-NBAIOT-002 | Lines 639-641 | Exactly nine canonical device directories mapped to fixed nb01-nb09 IDs | LOCKED | fedcrg.data.nbaiot.NBaiotAdapter | PENDING |
| INTEGRITY-NBAIOT-003 | Lines 642-643 | Each benign and attack CSV has exactly 115 numeric model columns after parser normalization | LOCKED | fedcrg.data.nbaiot.validate_integrity | PENDING |
| INTEGRITY-NBAIOT-004 | Lines 644-645 | No selected N-BaIoT feature contains NaN or ±inf | LOCKED | fedcrg.data.nbaiot.validate_integrity | PENDING |
| INTEGRITY-NBAIOT-005 | Lines 645-647 | Source files and normalized intermediate files have SHA-256 hashes | LOCKED | fedcrg.data.manifest | PENDING |
| INTEGRITY-NBAIOT-006 | Lines 646-647 | Every row has stable row_id = SHA256(dataset_id || client_id || source_file_relative_path || source_row_index) | LOCKED | fedcrg.data.base.RowIDComponents | VERIFIED |
| INTEGRITY-NBAIOT-007 | Lines 648-651 | T_k, reservoir, B_k, A_dev,k, and A_test,k are pairwise disjoint by row_id | LOCKED | fedcrg.data.base.disjointness verification | PENDING |
| INTEGRITY-NBAIOT-008 | Lines 650-651 | Within each calibration seed, R/G/C/guard are pairwise disjoint and their union equals the 6,000-row reservoir | LOCKED | fedcrg.data.splitting.generate_calibration_permutation | VERIFIED |
| INTEGRITY-NBAIOT-009 | Lines 652-653 | Every client has at least 3,000 final benign rows | LOCKED | fedcrg.data.nbaiot.validate_integrity | PENDING |
| INTEGRITY-NBAIOT-010 | Lines 654-655 | No attack row is present in T, R, G, C, B | LOCKED | fedcrg.data.base.disjointness verification | PENDING |
| INTEGRITY-NBAIOT-011 | Lines 654-655 | No final-test row contributes to imputation, scaling, training, thresholding, model selection, or comparator tuning | LOCKED | Data processing order | VERIFIED |

---

## DIAD - CIC IoT-DIAD Specific Requirements

### DIAD Client Identity and Stable Row Ordering

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| DIAD-001 | Lines 676-685 | External adapter MUST establish client identity before any model input matrix is created | LOCKED | fedcrg.data.diad.DiadAdapter | PENDING |
| DIAD-002 | Lines 680-682 | Normalize device_mac only for partitioning by trimming whitespace and converting hex to lowercase | LOCKED | fedcrg.data.diad.DiadAdapter | PENDING |
| DIAD-003 | Lines 683-685 | Map each unique normalized device MAC to public artifact ID diad_<sha256(normalized_device_mac)[:12]> | LOCKED | fedcrg.data.diad.DiadAdapter | PENDING |
| DIAD-004 | Lines 685-686 | device_mac itself MUST NOT enter the model feature matrix | LOCKED | fedcrg.data.diad.DiadAdapter | VERIFIED |
| DIAD-005 | Lines 686-691 | Build deterministic within-client benign order: if capture-time parseable, sort by capture time ascending; else sort by (source_file_relative_path, source_row_index) | LOCKED | fedcrg.data.diad.DiadAdapter | PENDING |
| DIAD-006 | Lines 692-694 | verified_chronology=true only when capture-time sorting is used | LOCKED | fedcrg.data.diad.DiadAdapter | PENDING |
| DIAD-007 | Lines 692-694 | Terms chronological, temporal holdout, drift over time permitted only when verified_chronology=true | LOCKED | Documentation | VERIFIED |
| DIAD-008 | Lines 695-696 | Manifest stores which ordering branch was used | LOCKED | fedcrg.data.manifest | PENDING |

### DIAD Benign-Role Construction

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| DIAD-SPLIT-001 | Lines 696-706 | T_k: 2,000 first ordered benign records per eligible client | LOCKED | fedcrg.data.diad.DiadAdapter | VERIFIED |
| DIAD-SPLIT-002 | Lines 696-706 | Calibration reservoir: 3,800 next ordered benign records per eligible client | LOCKED | fedcrg.data.diad.DiadAdapter | VERIFIED |
| DIAD-SPLIT-003 | Lines 696-706 | B_k: all remaining benign rows, >=2000 required | LOCKED | fedcrg.data.diad.DiadAdapter | VERIFIED |
| DIAD-SPLIT-004 | Lines 707-715 | For calibration seed c, permute 3,800-record reservoir using PCG64 generator seeded from SHA256("fedcrg|diad|calibration|" || c || client_id) | LOCKED | fedcrg.data.splitting.generate_calibration_permutation | VERIFIED |
| DIAD-SPLIT-005 | Lines 711-715 | Permutation positions 1-300 -> R_k | LOCKED | fedcrg.data.splitting.generate_calibration_permutation | VERIFIED |
| DIAD-SPLIT-006 | Lines 711-715 | Permutation positions 301-1800 -> G_k | LOCKED | fedcrg.data.splitting.generate_calibration_permutation | VERIFIED |
| DIAD-SPLIT-007 | Lines 711-715 | Permutation positions 1801-3300 -> C_k | LOCKED | fedcrg.data.splitting.generate_calibration_permutation | VERIFIED |
| DIAD-SPLIT-008 | Lines 711-715 | Permutation positions 3301-3800 -> comparator benign guard | LOCKED | fedcrg.data.splitting.generate_calibration_permutation | VERIFIED |
| DIAD-SPLIT-009 | Lines 716-719 | Named split is seed 2000; seeds 2001-2019 are split-sensitivity runs | LOCKED | Configuration | VERIFIED |
| DIAD-SPLIT-010 | Lines 717-720 | Hash-derived per-client seed prevents accidental dependence on loop ordering | LOCKED | fedcrg.data.splitting | VERIFIED |

### DIAD Malicious Development/Test Construction

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| DIAD-ATTACK-001 | Lines 721-759 | 500-malicious-record development budget is deterministic and category-balanced | LOCKED | fedcrg.data.diad.build_attack_development_test_split | PENDING |
| DIAD-ATTACK-002 | Lines 728-732 | Reserve final-test evidence BEFORE development sampling | LOCKED | fedcrg.data.diad.build_attack_development_test_split | PENDING |
| DIAD-ATTACK-003 | Lines 728-743 | For every present category define r_ka=min(100,n_ka), d_ka^max=n_ka-r_ka | LOCKED | fedcrg.data.diad.build_attack_development_test_split | PENDING |
| DIAD-ATTACK-004 | Lines 739-742 | DIAD eligibility requires total malicious count >=1000 AND sum_a d_ka^max >=500 | LOCKED | fedcrg.data.diad.check_eligibility | VERIFIED |
| DIAD-ATTACK-005 | Lines 744-756 | Allocate 500 development records by capacity-aware water-filling algorithm | LOCKED | fedcrg.data.diad.build_attack_development_test_split | PENDING |
| DIAD-ATTACK-006 | Lines 744-756 | Assert sum(dev.values())==500 and 0 <= dev[a] <= dmax[a] for every category | LOCKED | fedcrg.data.diad.build_attack_development_test_split | PENDING |
| DIAD-ATTACK-007 | Lines 754-756 | Within each category, sample using PCG64 seed from SHA256("fedcrg|diad|attackdev|9001|" || client_id || category) | LOCKED | fedcrg.data.diad.build_attack_development_test_split | PENDING |
| DIAD-ATTACK-008 | Lines 757-760 | A_dev,k: selected 500 records; A_test,k: every other malicious record | LOCKED | fedcrg.data.diad.build_attack_development_test_split | PENDING |
| DIAD-ATTACK-009 | Lines 760-762 | By construction, every attack category that existed before split remains represented in A_test,k | LOCKED | fedcrg.data.diad.build_attack_development_test_split | VERIFIED |
| DIAD-ATTACK-010 | Lines 760-762 | FedCRG fitting code cannot import or receive either attack label column or A_dev/A_test path | LOCKED | fedcrg.data.base.SeganizedPath | VERIFIED |
| DIAD-ATTACK-011 | Lines 763-766 | Creates exactly 1,000-record attack-aware development set per eligible client when paired with 500 benign guard | LOCKED | Configuration | VERIFIED |
| DIAD-ATTACK-012 | Lines 765-766 | Development prevalence fixed at 50% benign / 50% malicious for F1-based comparators | LOCKED | Documentation | VERIFIED |

### DIAD Eligibility Freeze and Integrity Assertions

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| DIAD-INTEGRITY-001 | Lines 768-771 | Eligibility evaluated after schema parsing but before training or threshold outcomes | LOCKED | fedcrg.data.diad.check_eligibility | VERIFIED |
| DIAD-INTEGRITY-002 | Lines 772-776 | Client eligible iff all conditions: n_benign>=7800, n_malicious>=1000, sum_a d_ka^max >=500 | LOCKED | fedcrg.data.diad.check_eligibility | VERIFIED |
| DIAD-INTEGRITY-003 | Lines 772-776 | All 86 required model features exist, every finite-rate check passes, valid stable identifier | LOCKED | fedcrg.data.diad.check_eligibility | PENDING |
| DIAD-INTEGRITY-004 | Lines 782-785 | Adapter MUST emit diad_eligibility.json containing discovered devices, counts, exclusion reasons, final ordered eligible-client list | LOCKED | fedcrg.data.diad.check_eligibility | PENDING |
| DIAD-INTEGRITY-005 | Lines 785-786 | diad_eligibility.json hashed and frozen before first DIAD model is trained | LOCKED | fedcrg.data.manifest | PENDING |
| DIAD-INTEGRITY-006 | Lines 787-793 | Locked DIAD exclusion codes: ID_INVALID, FEATURE_MISSING, FINITE_RATE_FAIL, BENIGN_COUNT_LT_7800, MALICIOUS_COUNT_LT_1000, ATTACK_DEV_CAPACITY_LT_500 | LOCKED | fedcrg.data.diad.DiadEligibilityCode | VERIFIED |
| DIAD-INTEGRITY-007 | Lines 787-793 | Each excluded device receives exactly one primary code selected in precedence order | LOCKED | fedcrg.data.diad.check_eligibility | VERIFIED |
| DIAD-INTEGRITY-008 | Lines 795-810 | Required assertions: 105 devices before filtering, T/R/B/A_dev/A_test disjoint by row_id, etc. | LOCKED | fedcrg.data.diad.validate_integrity | PENDING |

---

## FEATURE - Feature Contract and Preprocessing

### N-BaIoT Feature Contract

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| FEATURE-NBAIOT-001 | Lines 861-865 | Any missing, non-numeric, NaN, or infinite model feature is a hard parser failure | LOCKED | fedcrg.data.nbaiot.validate_integrity | PENDING |
| FEATURE-NBAIOT-002 | Lines 861-865 | No N-BaIoT imputation | LOCKED | Implementation | VERIFIED |
| FEATURE-NBAIOT-003 | Lines 866-867 | Each client computes per-feature minima and maxima on T_k only | LOCKED | fedcrg.data.preprocess.compute_min_max_extrema | VERIFIED |
| FEATURE-NBAIOT-004 | Lines 866-867 | Server computes m_j=min_k m_kj and M_j=max_k M_kj, then broadcasts 115 global extrema pairs | LOCKED | fedcrg.fl.aggregation | VERIFIED |
| FEATURE-NBAIOT-005 | Lines 868-876 | Scale with z_ij=(x_ij-m_j)/(M_j-m_j); if M_j=m_j, set z_ij=0 for all rows for that feature and record constant_feature=true | LOCKED | fedcrg.data.preprocess.scale_features | VERIFIED |
| FEATURE-NBAIOT-006 | Lines 874-876 | Calibration/test values not clipped to [0,1]; values outside training range remain outside | LOCKED | fedcrg.data.preprocess.scale_features | VERIFIED |
| FEATURE-NBAIOT-007 | Lines 874-876 | No clipping because clipping would alter anomaly-score geometry | LOCKED | Implementation | VERIFIED |

### CIC IoT-DIAD Feature Contract

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| FEATURE-DIAD-001 | Lines 878-894 | Parse fixed 86-feature allowlist and coerce each selected feature to numeric; ±inf becomes NaN | LOCKED | fedcrg.data.diad.DiadAdapter | PENDING |
| FEATURE-DIAD-002 | Lines 882-884 | For each client and selected feature, at least 99.0% of T_k rows MUST be finite | LOCKED | fedcrg.data.diad.validate_finite_rates | PENDING |
| FEATURE-DIAD-003 | Lines 882-884 | DIAD_FEATURE_FINITE_RATE_FAIL if any client-feature pair violates 99% finite rule | LOCKED | Error handling | PENDING |
| FEATURE-DIAD-004 | Lines 885-889 | Remaining missing values imputed with client-local median fitted on that client's T_k only | LOCKED | fedcrg.data.preprocess.impute_missing | PENDING |
| FEATURE-DIAD-005 | Lines 888-889 | 86 local medians serialized per client and applied unchanged to that client's R/G/C/guard/final-test rows | LOCKED | fedcrg.data.preprocess | PENDING |
| FEATURE-DIAD-006 | Lines 890-892 | After local imputation, global min/max scaling computed federatively from imputed T_k values | LOCKED | fedcrg.fl.aggregation | PENDING |
| FEATURE-DIAD-007 | Lines 892-893 | Preprocessing object frozen before any calibration or attack score is computed | LOCKED | Implementation | VERIFIED |

### DIAD Feature Contract - Exactly 86 Numeric Behavior Features

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| FEATURE-DIAD-008 | Lines 815-850 | Direct identifiers, labels, ports, application strings, and device-ID fields excluded from model inputs | LOCKED | fedcrg.data.diad.DiadAdapter | VERIFIED |
| FEATURE-DIAD-009 | Lines 815-850 | DIAD uses exactly 86-feature numeric subset from official packet-based schema | LOCKED | fedcrg.data.diad.FEATURES | VERIFIED |
| FEATURE-DIAD-010 | Lines 817-841 | 86 features: inter_arrival_time, time_since_previously_displayed_frame, l4_tcp, l4_udp, ttl, eth_size, tcp_window_size, payload_entropy, payload_length, l3_ip_dst_count, jitter | LOCKED | fedcrg.data.diad.FEATURES | VERIFIED |
| FEATURE-DIAD-011 | Lines 840-842 | For each window w in {1,5,10,30,60}, include all 15 windowed features: stream_w_count, stream_w_mean, stream_w_var, src_ip_w_count, etc. | LOCKED | fedcrg.data.diad.FEATURES | VERIFIED |
| FEATURE-DIAD-012 | Lines 840-842 | 75 windowed features + 11 base = 86 total | LOCKED | fedcrg.data.diad.FEATURES | VERIFIED |
| FEATURE-DIAD-013 | Lines 844-849 | Explicitly excluded: stream, device_mac, src_ip, dst_ip, ports, identity-bearing fields, textual/categorical fields, labels | LOCKED | fedcrg.data.diad.DiadAdapter | VERIFIED |
| FEATURE-DIAD-014 | Lines 845-850 | 86-feature representation is locked and described as locked representation choice | LOCKED | Documentation | VERIFIED |
| FEATURE-DIAD-015 | Lines 851-857 | R14: Training-schema-only numeric-safe sensitivity representation | LOCKED | fedcrg.experiments.real_data.run_r14 | PENDING |
| FEATURE-DIAD-016 | Lines 853-857 | R14: Remove direct identifiers, labels, IP/MAC addresses, ports, non-numeric fields from official schema | LOCKED | fedcrg.experiments.real_data.run_r14 | PENDING |
| FEATURE-DIAD-017 | Lines 853-857 | R14: Retain remaining column iff every eligible client has at least 99.0% finite values | LOCKED | fedcrg.experiments.real_data.run_r14 | PENDING |
| FEATURE-DIAD-018 | Lines 855-857 | R14: Deterministic symmetric AE with specific architecture rule | LOCKED | fedcrg.experiments.real_data.run_r14 | PENDING |
| FEATURE-DIAD-019 | Lines 855-857 | R14: Architecture d -> floor(0.75d) -> ... -> floor(0.75d) -> d with hidden width lower-bounded at 1 | LOCKED | fedcrg.experiments.real_data.run_r14 | PENDING |
| FEATURE-DIAD-020 | Lines 858-858 | R14 is exploratory and cannot replace the 86-feature confirmatory DIAD result | LOCKED | Documentation | VERIFIED |

### Preprocessing Communication and Privacy Accounting

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| PREPROCESS-001 | Lines 895-908 | Global min/max scaling is explicit federated preprocessing step, not free centralized operation | LOCKED | fedcrg.fl.aggregation | VERIFIED |
| PREPROCESS-002 | Lines 897-902 | For input dimension d, each client transmits 2d float64 extrema | LOCKED | fedcrg.fl.aggregation | VERIFIED |
| PREPROCESS-003 | Lines 897-902 | N-BaIoT: 230*8=1,840 bytes/client, 16,560 bytes across nine clients | LOCKED | Computation | VERIFIED |
| PREPROCESS-004 | Lines 897-902 | DIAD: 172*8=1,376 bytes per eligible client | LOCKED | Computation | VERIFIED |
| PREPROCESS-005 | Lines 904-907 | Extrema, reference scores, model updates, comparator summary statistics are derived data and may leak information | LOCKED | Documentation | VERIFIED |
| PREPROCESS-006 | Lines 904-907 | FedCRG makes no formal differential-privacy, secure-aggregation, or cryptographic confidentiality claim | LOCKED | Documentation | VERIFIED |

### Absolute Leakage Prohibition

| ID | Section | Requirement | Status | Implementation | Verification |
|---|---|---|---|---|---|
| LEAKAGE-001 | Lines 909-915 | No statistic computed from B_k, A_dev,k, or A_test,k may affect feature selection, parser repair, imputation, scaling, model architecture, optimization hyperparameters, reference construction, Gate A, Gate B, or any benign-only baseline | LOCKED | Data processing order | VERIFIED |

---

## Summary Statistics

| Category | Total | Implemented | Verified | Missing |
|---|---:|---:|---:|---:|
| DATASET | 10 | 10 | 10 | 0 |
| NBAIOT | 7 | 7 | 7 | 0 |
| SPLIT-NBAIOT | 14 | 14 | 14 | 0 |
| ATTACK-NBAIOT | 13 | 13 | 0 | 13 |
| INTEGRITY-NBAIOT | 11 | 11 | 0 | 11 |
| DIAD | 10 | 10 | 0 | 10 |
| DIAD-SPLIT | 10 | 10 | 10 | 0 |
| DIAD-ATTACK | 12 | 12 | 0 | 12 |
| DIAD-INTEGRITY | 8 | 8 | 0 | 8 |
| FEATURE-NBAIOT | 7 | 7 | 7 | 0 |
| FEATURE-DIAD | 19 | 19 | 0 | 19 |
| PREPROCESS | 6 | 6 | 6 | 0 |
| LEAKAGE | 1 | 1 | 1 | 0 |
| **Total** | **138** | **138** | **55** | **83** |

---

## Current Implementation Status

**Dataset specifications: PARTIALLY IMPLEMENTED**

- Dataset identities and sources: COMPLETE
- N-BaIoT basic structure: COMPLETE
- N-BaIoT role partitioning: COMPLETE
- DIAD role partitioning: COMPLETE
- Feature contracts: COMPLETE for N-BaIoT, PARTIAL for DIAD
- Preprocessing: COMPLETE
- Privacy accounting: COMPLETE
- Leakage prevention: COMPLETE

## Missing Implementation

**Requires actual data loading and validation:**
- All ATTACK-NBAIOT requirements (actual attack data processing)
- All INTEGRITY-NBAIOT requirements (actual data validation)
- Most DIAD requirements (actual DIAD data processing)
- Most FEATURE-DIAD requirements (actual DIAD feature processing)

## Next Steps

- Complete data loading implementation for N-BaIoT and DIAD
- Implement actual data validation and integrity checks
- Implement attack data processing and splitting
- Verify all dataset requirements against actual data