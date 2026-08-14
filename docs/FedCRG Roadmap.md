# FedCRG — Federated Calibration Readiness Gate
## Official Research, Implementation, and Pre-Registration Protocol

> **Document status.** Implementation-grade, pre-registration-ready research protocol. This specification is normative for the confirmatory FedCRG study. It fixes the scientific claim, novelty boundary, data roles, statistical contract, algorithms, baselines, metrics, experiment registry, implementation interfaces, tests, failure states, and claim discipline before confirmatory outcome analysis. Any deviation requires a versioned amendment recorded before the affected outcome is inspected.


| **Control item**            | **Locked value**                                   |
|-----------------------------|----------------------------------------------------|
| Version                     | FedCRG v2.0                                        |
| Protocol date               | 12 August 2026                                     |
| Literature cutoff           | 12 August 2026                                     |
| Primary method              | FedCRG — Federated Calibration Readiness Gate      |
| Primary target FPR          | 1.00%                                              |
| Primary acceptable FPR band | 0.50% to 1.50%                                     |
| Primary Gate-A assurance    | 95% per client                                     |
| Primary Gate-B confidence   | 95% exact Clopper-Pearson                          |
| Primary dataset             | N-BaIoT; nine natural IoT-device clients           |
| External validation         | CIC IoT-DIAD 2024; eligible natural device clients |
| Primary target venue        | IEEE Internet of Things Journal                    |


## Document Control and Normative Language

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHALL NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** are used normatively.

| **Status** | **Meaning in this protocol** |
|---|---|
| LOCKED | Confirmatory value. Changing it after outcome inspection requires a protocol amendment and the affected analysis becomes exploratory. |
| DERIVED | Deterministically computed from locked values. |
| DATA-DEPENDENT | Determined from source data using a locked rule, never from performance outcomes. |
| EXPLORATORY | May be analyzed, but cannot replace a locked confirmatory result. |
| STOP | The run is invalid until the stated integrity problem is resolved. |

**Protocol hierarchy.** In case of conflict: (1) formulas and state-transition rules in Sections 4–6; (2) dataset role definitions in Section 7; (3) baseline definitions in Section 9; (4) experiment registry in Section 11; (5) configuration files generated from Appendix E. Comments, README prose, notebooks, and figure scripts cannot override this document.

### v2.0 hostile-audit corrections

FedCRG v2.0 supersedes v1.0. The most material corrections are:

1. The detector-training audit is resolved against the **primary FedDetect paper**. FedDetect explicitly reports a hyperparameter search followed by **batch size 64, 120 local epochs, 30 global rounds, and tanh**; its algorithm also uses local Adam and a cross-round cosine scheduler. The N-BaIoT primary detector therefore retains the literature-faithful 30-round/120-local-epoch scale. The exact learning-rate endpoints, initialization, deterministic shuffling, optimizer-reset semantics, and score-cache contract remain FedCRG-specific engineering choices because those details are not all fixed by the paper text.
2. Calibration-role permutations are explicitly treated as **split-sensitivity analyses**, not independent experimental subjects. Confirmatory inference cannot count 50 calibration seeds as 50 independent clients.
3. Global thresholds couple clients within a federation. Any client bootstrap used for population-style sensitivity MUST recompute the federation threshold inside the bootstrap replicate; resampling already-computed client metrics is prohibited.
4. DIAD missing-value imputation is local to each client’s benign training data. The previous “global median” wording silently required centralizing training rows or adding an unspecified secure quantile protocol.
5. Attack-aware comparators now use a **fixed, balanced 500-anomaly development budget per client** rather than an uncontrolled 10% of very large attack files. This prevents F1 threshold choice from being dominated by arbitrary attack prevalence and equalizes label advantage across clients.
6. Outcome-contingent method mutation is prohibited. Gate B is not removed after seeing primary outcomes. If its ablation shows no value, that is reported as a negative component result rather than used to redesign the method on the same confirmatory data.
7. Publication “go/no-go” thresholds are replaced by **claim-strength gates**. Negative or non-replicating results remain reportable; the protocol never suppresses a valid study because the method underperforms.
8. The N-BaIoT source-order language is corrected: the CSV order is preserved and treated as **source order**. It is called chronological only when a verifiable timestamp/order provenance exists.
9. Federated preprocessing communication and privacy are made explicit. Global min/max scaling requires per-feature extrema exchange; this is derived information and is not formally private.
10. Model parameter counts, training communication, artifact schemas, exact quantile conventions, tie handling, and failure codes are now specified so two independent implementations can be parity-tested.


## Canonical Research Identity

| **Identity item**                 | **Locked value**                                                                                                    |
|-----------------------------------|---------------------------------------------------------------------------------------------------------------------|
| Method acronym                    | FedCRG                                                                                                              |
| Full method name                  | Federated Calibration Readiness Gate                                                                                |
| Canonical manuscript title        | FedCRG: Evidence-Admitted Calibration Readiness for Client-Specific Thresholding in Federated IoT Anomaly Detection |
| GitHub repository                 | fedcrg                                                                                                              |
| Python package / import namespace | fedcrg                                                                                                              |
| Configuration method ID           | fedcrg                                                                                                              |
| Artifact filename prefix          | fedcrg\_                                                                                                            |
| Primary paper shorthand           | FedCRG; define once as Federated Calibration Readiness Gate                                                         |

> **Naming rule.** The trained anomaly detector is not called FedCRG. FedCRG names only the post-training threshold-personalization admission layer. This prevents the paper from conflating detector learning with operating-point governance.


No author name is inserted because none was supplied. The document is a protocol/specification, not a claim that experiments have already been run.

# 1. Executive Decision and Research Position

**Decision: PROCEED, with the corrected specification in this document.** The research question remains publishable only if it is framed as evidence-admitted deployment of a client-specific anomaly operating point. It must not be framed as invention of local thresholds, federated calibration, adaptive thresholding, personalized federated learning, conformal prediction, or finite-sample order-statistic theory.

> **Central scientific question.** For a heterogeneous federated IoT client, when is there enough independent benign evidence to justify replacing a federation reference threshold with a client-specific threshold whose future benign false-positive rate is itself supported by a finite-sample operating-band contract?


**Final added value.** FedCRG couples two logically separate requirements: (A) local capability - the client possesses enough benign calibration evidence to construct a local threshold whose future benign FPR falls inside a pre-registered operating band with a specified finite-sample assurance; and (B) personalization necessity - an independent benign gate sample provides exact-binomial evidence that the federation reference threshold is materially outside that same operating band for the client. Personalization is admitted only when both conditions hold.

- The detector is frozen before threshold-policy evaluation. FedCRG is a post-training decision layer, not a new FL optimizer or anomaly model.

- The core protocol uses benign data only for threshold admission and threshold construction. Attack labels are test-only for FedCRG.

- The formal guarantee is per-client and conditional on the stated **i.i.d. continuous benign-score sampling assumptions**. Generic dependence or finite exchangeability alone is not asserted to preserve the Beta/binomial laws. A simultaneous all-client guarantee is not claimed in the primary experiment.

- The revised Gate A is two-sided with respect to the acceptable FPR band. This closes the previous loophole where a one-sided upper-FPR guarantee could not justify correcting an over-conservative shared threshold.

- If the reference threshold is shown to be materially wrong but the local threshold is not statistically ready, the system enters a CALIBRATION_DEFICIT state; it does not pretend that the fallback is certified.

# 2. Research Questions, Hypotheses, and Falsification Criteria

| **ID** | **Focus**                 | **Locked question**                                                                                                                                                                                           |
|--------|---------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| RQ1    | Finite-sample validity    | Under i.i.d. continuous benign scores, does the order-statistic readiness construction achieve its pre-specified probability of placing the future benign FPR inside the operating band?                |
| RQ2    | Need for admission        | Does requiring independent evidence of reference-threshold mismatch avoid unnecessary local personalization compared with always-local thresholding?                                                          |
| RQ3    | Operational reliability   | At frozen detector scores, does FedCRG reduce client-level FPR contract violations and high-FPR excess relative to strong shared, local, and shrinkage baselines without materially degrading anomaly recall? |
| RQ4    | Data sufficiency          | How do admission states change as local calibration and gate sample counts cross exact finite-sample readiness/power boundaries?                                                                              |
| RQ5    | Robustness of assumptions | How does the nominal contract degrade under temporal dependence, calibration-to-test drift, score ties, and benign-calibration contamination?                                                                 |
| RQ6    | External validity         | Does the direction of the primary reliability result replicate on a second dataset with natural IoT device identities and a different feature representation?                                                 |

| **ID** | **Pre-registered hypothesis / falsification rule**                                                                                                                                                            |
|--------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| H1     | In every i.i.d.-continuous S1 cell, Monte-Carlo coverage must agree with the exact Gate-A probability: `abs(p_hat-P_r) <= max(0.005, 4*sqrt(P_r*(1-P_r)/10000))`. A violation is treated first as an implementation/audit failure, not as permission to alter the theorem. |
| H2     | FedCRG will reduce Mean Excess Band Error relative to both the full benign-policy-budget shared Q99 baseline and the full benign-policy-budget local Q99 baseline on the primary dataset.                                                         |
| H3     | Any claimed FedCRG reliability gain must incur no more than a 3.0 percentage-point absolute loss in **Attack-Balanced Macro-TPR** relative to the locked benign-only utility anchor `max(GLOBAL-Q99-FULL, LOCAL-Q99-FULL, SHRINKAGE)` on that experiment cell. |
| H4     | Gate B provides non-redundant incremental value only if it changes admission decisions and improves MEBE or BandViolationRate relative to GATE-A-ONLY on at least one natural-client dataset; otherwise its incremental utility is reported as unsupported without retroactively changing the confirmatory method.                          |
| H5     | AUROC and AUPRC will be numerically identical across threshold policies using the same cached test scores, up to serialization/rounding tolerance of 1e-12; any larger difference is an implementation error. |

# 3. Adversarial Literature Audit and Novelty Boundary

**Search status.** A targeted hostile literature recheck was performed through 12 August 2026 against primary papers, publisher pages, PMLR proceedings, arXiv preprints, and official dataset/journal sources. The search specifically attempted to invalidate novelty by looking for: federated anomaly thresholds; local-versus-global threshold selection; benign-only client-specific calibration; finite-sample federated threshold guarantees; personalized FL-IDS; adaptive/conformal thresholding; low-sample local/global calibration borrowing; and exact-name collisions for FedCRG / Federated Calibration Readiness Gate. No exact FedCRG research-method collision was surfaced by the targeted searches. This is not a trademark or perpetual uniqueness guarantee; the search must be repeated immediately before public release and submission.

| **Prior work**                    | **What already exists**                                                                                                                                                                                                   | **Consequence for this study**                                                                                                                                                                                                         |
|-----------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Wilks, 1941                       | Nonparametric tolerance limits / sample-size logic from order statistics.                                                                                                                                                 | Kills any claim that the Beta/order-statistic mathematics is novel. FedCRG uses classical theory as a building block.                                                                                                                  |
| FedDetect, Zhang et al., 2021     | Federated IoT anomaly detection; N-BaIoT; global threshold from pooled reconstruction-error statistics.                                                                                                                   | Kills any claim that a federation-wide anomaly threshold or FAE+threshold pipeline is new.                                                                                                                                             |
| FCP, Lu et al., 2023              | Federated conformal prediction with partial exchangeability and finite-sample uncertainty guarantees.                                                                                                                     | Kills any claim that federation plus finite-sample uncertainty is new.                                                                                                                                                                 |
| FedCal, Peng et al., 2024         | Client-specific and global probability calibration via parameterized scalers.                                                                                                                                             | Kills generic "federated calibration" novelty.                                                                                                                                                                                         |
| Sun et al., ICML 2024             | Online adaptive anomaly thresholding with confidence-sequence guarantees on FPR/FNR and distribution shifts.                                                                                                              | Kills generic statistically guaranteed adaptive anomaly-threshold novelty.                                                                                                                                                             |
| Laridi et al., Sci. Rep. 2024     | Federated AE thresholding from normal+anomalous validation errors; follow-up analysis predicts local-vs-federated benefit using SVM/RF. Its benefit label is derived from the local-test F1 difference between federated and local thresholds. | Closest conceptual prior art. FedCRG cannot claim the local-vs-global question as new. The defensible distinction is prospective benign-only evidence admission, disjoint gate/calibration evidence, and a finite-sample operating-band contract rather than a learned retrospective F1-benefit selector. |
| Self-Aware PFL, Chen et al., 2022 | Adapts degree of model personalization from uncertainty/heterogeneity.                                                                                                                                                    | Kills broad "decide when to personalize" novelty at model level.                                                                                                                                                                       |
| PerFLID, 2025; pFedCross, 2025    | Personalized FL intrusion detection for heterogeneous clients.                                                                                                                                                            | Kills generic personalized FL-IDS novelty.                                                                                                                                                                                             |
| Fed-DTCN, Khan et al., 2026       | Unsupervised zero-day federated IoT anomaly detection with private/shared encoders and client-specific benign-calibrated thresholds.                                                                                      | Kills claim that benign-only client-specific thresholds in FL-IoT are new.                                                                                                                                                             |
| CF-HFC, 2026                      | Federated heterogeneous IoT IDS with adaptive conformal calibration and dynamically adjusted decision thresholds.                                                                                                         | Kills broad adaptive/conformal FL-IoT threshold novelty.                                                                                                                                                                               |
| Shahid, 2026 preprint             | Federated conformal risk control with risk-curve shrinkage; shows pooled-vs-local calibration poverty and shrinkage tradeoff.                                                                                             | Makes a partial-pooling/shrinkage comparator mandatory and narrows the contribution away from "small clients should borrow strength".                                                                                                  |
| FBID, Aug. 2026 preprint          | Server-controlled adaptive personalized FL for OOD IoT attack detection.                                                                                                                                                  | Kills broad adaptive personalization-control novelty; distinct because it controls model personalization, not threshold evidence admission.                                                                                            |
| Robalino-Diaz et al., 2026        | Shows federated non-IID anomaly models can retain AUC while fixed-threshold recall collapses; post-hoc calibration restores operating behavior.                                                                           | Strong motivation that score discrimination and operating-point reliability can diverge.                                                                                                                                               |
| Shi et al., Aug. 2026 preprint    | Explicitly separates detector representation quality from decision-rule/threshold quality.                                                                                                                                | Kills any standalone claim that separating detector quality from threshold quality is itself new.                                                                                                                                      |

## 3.1 The closest prior-art distinction that must appear in the manuscript

**Laridi et al. (2024) is the closest direct conceptual comparator.** That paper explicitly shows that some clients can benefit from local rather than federated thresholds and performs a follow-up SVM / Random-Forest analysis of that choice. The threshold calculation itself uses validation sets containing both normal and anomalous observations and F1-based selection; the follow-up benefit label is the local-test F1 difference between the best local threshold and the federated threshold. FedCRG therefore does not claim the question "local or global?" as new. The defensible distinction is prospective information and decision governance: FedCRG reads only benign evidence, contains no attack-aware F1 objective or learned local/global selector, uses disjoint R/G/C roles, and admits client-specific deployment only when independent mismatch evidence and a finite-sample operating-band readiness condition both hold.

**Fed-DTCN (2026) closes another potential loophole in the novelty claim.** It calibrates a client-specific anomaly threshold from held-out benign data to satisfy a target false-positive rate. Accordingly, neither "benign-only local threshold" nor "client-specific FPR threshold in federated IoT" can be presented as new. FedCRG adds the explicit decision of whether that local operating point is admissible at all, with a separate reference-mismatch gate, an independent local-readiness sample, a finite-sample in-band contract, and unresolved evidence states.

**Shahid (2026) makes partial pooling a required comparator.** Its federated conformal-risk study demonstrates the failure modes of both naive pooling and data-poor local calibration and proposes risk-curve shrinkage. FedCRG therefore must beat or meaningfully differ from a shrinkage-style baseline; it may not claim that borrowing strength under calibration scarcity is itself new.

## 3.2 Safe novelty claim

> **Permitted claim.** We formulate client-specific anomaly-threshold deployment in heterogeneous federated IoT detection as an evidence-admission problem. The proposed FedCRG protocol permits personalization only when independent benign evidence establishes that the federation reference threshold is materially outside a pre-registered client FPR band and a disjoint local benign calibration set is large enough to construct an order-statistic threshold satisfying a finite-sample in-band operating contract under stated assumptions.


## 3.3 Claims that are prohibited

- "First federated anomaly threshold" or "first client-specific threshold in federated IoT IDS."

- "First method deciding whether a federated client should personalize."

- "New finite-sample quantile/tolerance theory."

- "New federated conformal prediction/calibration algorithm."

- "Distribution-free guarantee under arbitrary temporal dependence or concept drift."

- "Privacy-preserving calibration" unless a formal privacy mechanism is later added and proved/evaluated.

- "End-to-end robust/secure federated IDS."

- "Improved detector representation" based only on threshold-policy results.

- Any use of "first" unless a fresh submission-week search confirms the exact claim.


## 3.4 2026 close-out audit: what changed after the earlier literature base

The 2026 literature makes three additional overclaim risks explicit.

1. **Adaptive thresholding in federated IoT is no longer a defensible
   broad novelty claim.** SecuFL-IoT and a 2026 federated Transformer-AE
   study both combine federated anomaly detection with dynamic/RL
   threshold adjustment.
2. **Client-specific benign calibration is already present.**
   Fed-DTCN performs local calibration and uses a client-specific
   decision threshold after benign representation learning.
3. **Calibration/threshold misalignment under heterogeneity is now an
   explicit research theme.** Recent work reports preserved ranking/AUC
   with severe fixed-threshold recall degradation under non-IID
   federation, strengthening the motivation for separating score quality
   from operating-point quality but eliminating novelty in that
   observation itself.

Accordingly, the manuscript's novelty paragraph MUST contain all four
qualifiers: **benign-only**, **independent evidence streams**,
**finite-sample operating-band readiness**, and **admission rather than
automatic personalization**. Removing any of these qualifiers makes the
claim materially easier for prior work to invalidate.


# 4. Formal Problem Definition and Locked Assumptions

| **Symbol** | **Definition**                                                 | **Primary value**                                                              |
|------------|----------------------------------------------------------------|--------------------------------------------------------------------------------|
| K          | Number of natural federated clients                            | 9 on N-BaIoT; K_DIAD determined by locked eligibility rule                     |
| s_k(x)     | Frozen anomaly score for client k; larger means more anomalous | Per-sample reconstruction MSE in primary detector                              |
| alpha      | Target benign false-positive rate                              | 0.0100                                                                         |
| rho        | Relative practical tolerance around alpha                      | 0.50                                                                           |
| a          | Lower acceptable FPR = max(0, alpha(1-rho))                    | 0.0050                                                                         |
| b          | Upper acceptable FPR = min(1, alpha(1+rho))                    | 0.0150                                                                         |
| gamma_A    | Required Gate-A in-band assurance                              | 0.95                                                                           |
| gamma_B    | Gate-B exact confidence level                                  | 0.95                                                                           |
| T_k        | Benign model-training set                                      | Dataset-specific fixed count                                                   |
| R_k        | Benign reference-threshold sample                              | 500 N-BaIoT; 300 DIAD                                                          |
| G_k        | Independent benign reference-mismatch gate sample              | 3000 N-BaIoT; 1500 DIAD                                                        |
| C_k        | Independent benign local-threshold calibration sample          | 2000 N-BaIoT; 1500 DIAD                                                        |
| B_k        | Final benign test set                                          | All remaining after fixed train/calibration reservoir; minimum specified below |
| A_dev,k    | Attack development data                                        | 500 malicious records/client, attack-subtype balanced; supervised comparators only |
| A_test,k   | Final attack test data                                         | All malicious records not assigned to the fixed 500-record `A_dev,k`; never used in threshold fitting                                 |

## 4.1 Assumptions required for the formal Gate-A statement

- Conditional on the frozen detector and preprocessing, benign calibration scores in \(C_k\) are assumed to be **i.i.d. draws from one continuous client-specific benign score distribution \(F_k\)** that also governs future benign scores for the formal Gate-A statement.

- The score function is frozen before C_k is used. Model weights, preprocessing parameters, and score definition cannot be changed after inspecting C_k or G_k outcomes.

- \(C_k\) is disjoint from \(G_k\), \(R_k\), model training, and final test data.

- Conditional on \(\tau_{\mathrm{ref}}\), indicators
  \(\mathbf 1[g_i>\tau_{\mathrm{ref}}]\) for \(g_i\in G_k\) are assumed
  i.i.d. Bernoulli with client reference exceedance probability
  \(p_{\mathrm{ref},k}\) for the exact Clopper-Pearson statement.

- Independence is required **between R/G/C roles and within the formal
  sampling model**. Dataset disjointness enforces non-reuse but cannot
  manufacture stochastic independence when real traffic is temporally
  dependent; this distinction MUST be stated in the manuscript.

- The threshold comparison is always score \> threshold; the inequality is never changed between methods or datasets.

- The primary guarantee is per client. It is not an all-clients-simultaneous 95% statement.

- Calibration data are presumed benign. Contamination robustness is evaluated empirically but is not part of the theorem.

## 4.2 Explicit non-assumptions

- FedCRG does not assume that the federation reference threshold is reliable for every client.

- FedCRG does not assume that local personalization is always beneficial.

- FedCRG does not require attack prevalence, attack labels, or attack samples for its own admission decision.

- FedCRG does not assume that real deployment traffic is i.i.d. The exact theorem is explicitly scoped to the i.i.d.-continuous reference model; temporal dependence, source-order effects, and distribution shift are stress-tested as assumption violations.

- FedCRG does not provide formal differential privacy or secure-aggregation guarantees for calibration scores.

# 5. Federated Calibration Readiness Gate (FedCRG): Exact Algorithm

## 5.1 Federation reference threshold

**Purpose.** The reference threshold is a federation-wide operating point against which each client is independently audited. It is not itself claimed to satisfy a per-client FPR guarantee.

R = union_k R_k, with \|R_k\| identical across clients.

N_R = sum_k \|R_k\|; q_ref = min(N_R, ceil((N_R + 1)(1 - alpha))).

tau_ref = R\_(q_ref), where R\_(j) is the j-th ascending pooled score.

**N-BaIoT primary value:** K=9, \|R_k\|=500, N_R=4500, q_ref=4456. Each client contributes the same number of scores, preventing traffic-volume dominance in reference construction.

**Privacy boundary:** the core research implementation transmits 500 derived float64 reference scores per N-BaIoT client (36,000 bytes total across nine clients, excluding transport overhead). Raw network records remain local, but score sharing is not formally private. The manuscript must not call this calibration stage privacy-preserving.

## 5.2 Gate A - finite-sample local operating-band readiness

**Correction relative to the preliminary roadmap.** Gate A is two-sided in operating FPR. A merely one-sided upper bound cannot justify personalization when the shared threshold is too conservative. The final Gate A therefore asks whether some order-statistic threshold can place the client FPR inside \[a,b\] with probability at least gamma_A.

Let C_k = {c_1,...,c_n}; sort c\_(1) \<= ... \<= c\_(n). For candidate rank r, tau_r = c\_(r).

Under the locked i.i.d.-continuous benign-score model: P_r = Pr\[a \<= FPR(tau_r) \<= b\]

P_r = I_b(n+1-r, r) - I_a(n+1-r, r),

where I_z(.,.) is the regularized incomplete beta function.

r\* = argmax_r P_r; ties are resolved in favor of the larger r (the more conservative threshold).

Gate A = READY iff max_r P_r \>= gamma_A; tau_local = c\_(r\*).

**Primary exact result.** For alpha=1%, rho=0.50 (band 0.5%-1.5%) and gamma_A=95%, the minimum benign local calibration size for any READY threshold is n_C=1416. At n_C=1416, r\*=1404 and P_r=0.9500045. At the primary N-BaIoT n_C=2000, r\*=1982 and P_r=0.9805279.

**Interpretation discipline.** Gate-A readiness is fundamentally a **sample-size/contract gate**. For fixed \((n,a,b,\gamma_A)\), readiness and \(r^*\) are determined before the score values are observed. The realized values in \(C_k\) determine only the numerical threshold \(C_{k,(r^*)}\). Consequently, \(P_r\) is a frequentist repeated-calibration-sample coverage probability under the locked i.i.d.-continuous model. It MUST NOT be written as a posterior probability, a confidence level conditional on the realized threshold, or “there is a 98.05% probability that this already-observed threshold is valid.”

**Why Gate A is still evidential.** The evidence quantity is the amount of independent presumed-benign calibration data available under a pre-specified sampling model. Score-shape diagnostics (ties, drift, contamination) can invalidate or qualify the theorem, but they are not allowed to move the rank after the data are seen.


### 5.2.1 Gate-A theorem, proof object, and implementation invariant

Let the benign score CDF for client \(k\) be \(F_k\), and let
\(C_{k,(r)}\) denote the \(r\)-th ascending order statistic from \(n\)
i.i.d. continuous benign scores. For the threshold
\(\tau=C_{k,(r)}\),

\[
U=F_k(C_{k,(r)}) \sim \operatorname{Beta}(r,n+1-r).
\]

Because the deployed false-positive probability under the strict rule
\(s>\tau\) is

\[
P_{\mathrm{FP}}(\tau)=1-F_k(\tau),
\]

the random future benign FPR induced by the order-statistic threshold is

\[
P_{\mathrm{FP}}(C_{k,(r)}) \sim
\operatorname{Beta}(n+1-r,r).
\]

Therefore the exact probability that the future benign FPR lies inside
the operating contract \([a,b]\) is

\[
\Pr\!\left[a\le P_{\mathrm{FP}}(C_{k,(r)})\le b\right]
=
I_b(n+1-r,r)-I_a(n+1-r,r).
\]

This is the quantity maximized by Gate A. No model score value enters the
rank optimization; only \(n,a,b\) do. The observed calibration scores
enter only when selecting the \(r^\*\)-th score after the rank has been
fixed.

**Derived moments.** For rank \(r\),

\[
\mathbb{E}[P_{\mathrm{FP}}]=\frac{n+1-r}{n+1},
\]

\[
\operatorname{Var}(P_{\mathrm{FP}})
=
\frac{(n+1-r)r}{(n+1)^2(n+2)}.
\]

For the locked N-BaIoT cell \(n=2000,r^\*=1982\), the induced mean FPR is
\(19/2001=0.0094952524\), close to the 1% target, while the exact
in-band probability is 0.9805279151.

**Implementation invariant.** `gate_a_table[n]` MUST contain:
`n`, `rank_r`, `coverage_probability`, `ready`, `alpha`, `rho`, `a`,
`b`, and `gamma_A`. Runtime code MUST read the precomputed rank and MUST
NOT optimize rank using observed client scores.

**Numerical requirement.** Beta-CDF calculations MUST use float64 and a
tested special-function implementation. For every locked unit-test cell,
absolute error against the reference values in Section 14.2 MUST be
\(\le 10^{-10}\).


| **Target alpha** | **Band at rho=0.50** | **Minimum n_C for 95% Gate A** |
|------------------|----------------------|--------------------------------|
| 0.5%             | 0.25%-0.75%          | 2861                           |
| 1.0%             | 0.50%-1.50%          | 1416                           |
| 2.0%             | 1.00%-3.00%          | 694                            |
| 5.0%             | 2.50%-7.50%          | 270                            |

| **Assurance gamma_A** | **Minimum n_C** | **Rank r\* at minimum** | **Exact P_r** |
|-----------------------|-----------------|-------------------------|---------------|
| 90%                   | 1000            | 991                     | 0.9001416     |
| 95%                   | 1416            | 1404                    | 0.9500045     |
| 99%                   | 2435            | 2413                    | 0.9900230     |

| **Tolerance** | **Band for alpha=1%** | **Minimum n_C at 95%** |
|---------------|-----------------------|------------------------|
| rho=0.25      | 0.75%-1.25%           | 5970                   |
| rho=0.50      | 0.50%-1.50%           | 1416                   |
| rho=1.00      | 0%-2.00%              | 149                    |

## 5.3 Gate B - independent evidence that the reference threshold is materially wrong

**Gate B no longer uses Wasserstein distance or a generic distribution-divergence threshold.** It tests the exact deployment quantity of interest: the benign exceedance probability of tau_ref on an independent client sample.

x_k = sum\_{g in G_k} 1\[g \> tau_ref\], n_G = \|G_k\|.

Compute the two-sided 95% exact Clopper-Pearson interval \[L_k, U_k\] for p_ref,k.

LOW_MISMATCH if U_k \< a; HIGH_MISMATCH if L_k \> b; otherwise NO_MATERIAL_MISMATCH_DEMONSTRATED.

**Important interpretation.** NO_MATERIAL_MISMATCH_DEMONSTRATED is not an equivalence claim and not a certification that p_ref,k lies in-band. It means only that the sample does not establish a material mismatch at the locked confidence level.

| **n_G** | **LOW_MISMATCH** | **HIGH_MISMATCH** | **Use**                                                                           |
|---------|------------------|-------------------|-----------------------------------------------------------------------------------|
| 736     | x=0              | x\>=19            | Minimum size at which even zero exceedances can establish LOW_MISMATCH for a=0.5% |
| 1000    | x\<=0            | x\>=24            | Sensitivity grid                                                                  |
| 1500    | x\<=2            | x\>=33            | External DIAD primary                                                             |
| 2000    | x\<=3            | x\>=42            | Sensitivity grid                                                                  |
| 3000    | x\<=7            | x\>=59            | N-BaIoT primary                                                                   |

> **Gate-B minimum evidence rule.** The deployable primary FedCRG contract requires `n_G >= n_G_min(a,gamma_B)=736` for the 0.5%–1.5% band at 95% two-sided confidence. Sensitivity contracts recompute this minimum from their own `a` and `gamma_B`; 736 is not reused blindly. If the applicable minimum is not met, return `GATE_B_INSUFFICIENT` rather than treating lack of evidence as support for the reference threshold.



### 5.3.1 Exact Clopper-Pearson formulas and directional tests

Let \(\delta_B=1-\gamma_B=0.05\). For \(x\) exceedances among \(n\)
independent gate scores:

\[
L(x,n)=
\begin{cases}
0,&x=0\\
\operatorname{Beta}^{-1}(\delta_B/2;\;x,n-x+1),&x>0
\end{cases}
\]

and

\[
U(x,n)=
\begin{cases}
1,&x=n\\
\operatorname{Beta}^{-1}(1-\delta_B/2;\;x+1,n-x),&x<n.
\end{cases}
\]

`LOW_MISMATCH` is declared iff \(U<a\); `HIGH_MISMATCH` is declared iff
\(L>b\). This is equivalent to requiring a two-sided 95% exact interval
to lie wholly outside the acceptable band in the corresponding
direction. The gate is therefore deliberately conservative near either
band boundary.

The decision is **conditional on the realized reference threshold**.
Because \(R_k\) and \(G_k\) are disjoint and Gate B never reuses the
reference-construction scores, the binomial calculation remains valid
conditional on \(\tau_{\mathrm{ref}}\) under the stated i.i.d. benign gate-score
assumption.

For auditability the implementation MUST additionally log the exact
one-sided boundary p-values

\[
p_{\mathrm{low}}=\Pr_{X\sim\mathrm{Bin}(n,a)}[X\le x],
\qquad
p_{\mathrm{high}}=\Pr_{X\sim\mathrm{Bin}(n,b)}[X\ge x].
\]

These p-values are diagnostics; the normative state transition is based
on the locked Clopper-Pearson interval rule above.

### 5.3.2 Why 736 gate observations is the minimum bidirectional budget

At \(x=0\), the 95% two-sided Clopper-Pearson upper endpoint is

\[
U(0,n)=1-(0.025)^{1/n}.
\]

The smallest \(n\) satisfying \(U(0,n)<a=0.005\) is \(n=736\).
At \(n=735\), \(U=0.0050063101\) and a low-FPR mismatch cannot be
demonstrated even with zero exceedances; at \(n=736\),
\(U=0.0049995250\), so \(x=0\) becomes sufficient.

This minimum is driven by the low-FPR direction. High-FPR evidence can
sometimes be established with fewer observations, but the deployable
primary protocol requires a gate sample large enough to support either
direction and therefore uses the symmetric minimum \(n_G\ge736\).


## 5.4 Final decision states and operational fallback

| **Deployment state** | **Condition** | **Threshold used** | **Interpretation** |
|---|---|---|---|
| NO_MATERIAL_MISMATCH_DEMONSTRATED | Gate B does not establish low/high mismatch | Use \(\tau_{\mathrm{ref}}\) | Reference retained; not called certified or equivalent. |
| LOCAL_PERSONALIZE | Gate B = LOW/HIGH_MISMATCH, Gate A is sample-size READY, and the selected local order statistic has multiplicity 1 | Use \(\tau_{\mathrm{local}}\) | Only state counted as admitted personalization. |
| CALIBRATION_DEFICIT | Gate B establishes LOW/HIGH_MISMATCH and Gate A != READY | Temporary \(\tau_{\mathrm{ref}}\) | Mismatch is statistically demonstrated at the locked Gate-B confidence, but a locally certified replacement is unavailable; collect more \(C_k\). |
| GATE_B_INSUFFICIENT | \(n_G<n_{G,\min}(a,\gamma_B)\); primary value 736 | Temporary \(\tau_{\mathrm{ref}}\) | Reference status is not evaluable under the primary bidirectional evidence budget; collect more \(G_k\). |
| CALIBRATION_ASSUMPTION_VIOLATION | Gate B establishes mismatch, Gate A is sample-size READY, but the selected \(C_{k,(r^*)}\) has multiplicity >1 | Temporary \(\tau_{\mathrm{ref}}\) | An observed tie directly violates the continuity model at the proposed operating point; local deployment is not admitted under the exact theorem. |

**Continuity diagnostic rule.** A tie at the selected local order statistic
is a deployment-blocking assumption violation in the confirmatory method,
not merely a cosmetic flag. Absence of an observed tie does **not** prove
that the underlying score law is continuous; continuity remains a stated
model assumption. `NONFINITE_SCORE`, `DATA_DRIFT_STRESS`, and similar
diagnostics are logged separately from the five-state policy machine.

### 5.4.1 Gate-B minimum is protocol-parameter dependent

The constant 736 is **not hard-coded as a universal statistical
constant**. For a two-sided Clopper-Pearson confidence level
\(\gamma_B\) and a strictly positive lower band limit \(a\), define

\[
n_{G,\min}(a,\gamma_B)=
\min\left\{n\ge1:
1-\left(\frac{1-\gamma_B}{2}\right)^{1/n}<a
\right\}.
\]

This is the minimum sample size for which even \(x=0\) can establish a
low-side mismatch. For the primary
\(a=0.005,\gamma_B=0.95\), it equals **736**.

If \(a=0\) (e.g. the \(\rho=1\) sensitivity band), a low-side mismatch
is mathematically impossible and no finite bidirectional minimum
exists. In that sensitivity, Gate B is high-side only and the result
must be labeled `ONE_SIDED_BAND_BY_DESIGN`. The primary algorithm and
deployment claims always use the locked positive-lower-bound band.

## 5.5 Pseudocode - normative implementation

```text
INPUT: alpha=0.01, rho=0.50, gamma_A=0.95, gamma_B=0.95,
federation reference samples {R_k}, client benign gate G_k,
client benign local calibration C_k
1. a = max(0, alpha * (1-rho)); b = min(1, alpha * (1+rho))
2. Build tau_ref from equal-count pooled R_k using q_ref=min(N_R,ceil((N_R+1)(1-alpha))).
3. Compute n_G,min from (a,gamma_B). For the primary contract it is 736. If |G_k| < n_G,min: return GATE_B_INSUFFICIENT, threshold=tau_ref.
4. x = count(g > tau_ref for g in G_k).
5. [L,U] = exact two-sided Clopper-Pearson(x, |G_k|, confidence=gamma_B).
6. mismatch = LOW if U<a; HIGH if L>b; otherwise NONE.
7. If mismatch == NONE: return NO_MATERIAL_MISMATCH_DEMONSTRATED, threshold=tau_ref.
8. For n=|C_k|, compute P_r for every rank r; choose largest-r tie among argmax(P_r).
9. If max(P_r) < gamma_A: return CALIBRATION_DEFICIT, threshold=tau_ref.
10. tau_local = sorted(C_k)[r*-1].
11. tie_count = multiplicity(tau_local in C_k). If tie_count > 1: return CALIBRATION_ASSUMPTION_VIOLATION, threshold=tau_ref.
12. return LOCAL_PERSONALIZE, threshold=tau_local.
CLASSIFICATION RULE FOR EVERY POLICY: anomaly iff score > threshold.
```


# 6. Per-Client Versus Federation-Wide Guarantees

**Primary claim: per-client.** The 95% Gate-A assurance applies separately to each admitted client. It does not imply that all K admitted client thresholds simultaneously satisfy the band with probability 95%.

| **K** | **Bonferroni per-client assurance for 95% familywise target** | **Minimum n_C** | **r\* at minimum** | **Exact P_r** |
|---:|---:|---:|---:|---:|
| 9 | 99.444444% | 2810 | 2785 | 0.9944483352 |
| 20 | 99.750000% | 3341 | 3311 | 0.9975000839 |
| 50 | 99.900000% | 3971 | 3935 | 0.9990007947 |
| 105 | 99.952381% | 4470 | 4430 | 0.9995240961 |

**Mandatory sensitivity.** For N-BaIoT, repeat Gate-A readiness using \(\gamma_A=1-0.05/9=0.994444\ldots\). Because the primary \(n_C=2000\) is below the simultaneous minimum \(n_C=2810\), the manuscript must explicitly show that the primary design supports a per-client contract but not a 95% all-nine simultaneous contract. The deployable algorithm itself remains per-client.

## 6.1 Exact Gate-B multiplicity sensitivities

Two federation-level sensitivities are REQUIRED; neither replaces the primary per-client Gate B.

1. **Bonferroni interval sensitivity.** For \(K\) clients, recompute every Clopper-Pearson interval at per-client confidence \(1-0.05/K\). Apply the same strict rules \(U<a\) and \(L>b\). This controls the probability of at least one false mismatch declaration by the union bound without requiring independence across clients.
2. **Holm directional-exact sensitivity.** For each client compute the one-sided exact binomial p-values
   \[
   p_{k,\mathrm{low}}=\Pr_{X\sim\mathrm{Bin}(n_G,a)}(X\le x_k),\qquad
   p_{k,\mathrm{high}}=\Pr_{X\sim\mathrm{Bin}(n_G,b)}(X\ge x_k).
   \]
   Apply Holm's step-down procedure at familywise \(0.05\) across all \(2K\) directional hypotheses. A client receives a multiplicity-sensitive LOW/HIGH label only if the corresponding directional null is rejected. If both were ever rejected because of a software/numerical defect, stop with `GATE_B_DIRECTION_CONTRADICTION`; a valid binomial count cannot support both tails simultaneously at these separated boundaries.

The manuscript reports how many primary mismatch declarations survive each sensitivity. It MUST NOT reinterpret a non-surviving declaration as proof that the reference is in-band.

**N-BaIoT derived Bonferroni cutoff.** With \(K=9,n_G=3000\), per-client confidence \(0.994444\ldots\) yields LOW_MISMATCH for \(x\le5\) and HIGH_MISMATCH for \(x\ge65\). The primary unadjusted cutoffs remain \(x\le7\) and \(x\ge59\). This difference MUST appear in the multiplicity sensitivity table.

# 7. Dataset and Data-Partition Protocol

## 7.1 N-BaIoT — primary natural-client experiment

**Authoritative dataset identity.** UCI dataset 442, *Detection of IoT
Botnet Attacks N-BaIoT*, DOI 10.24432/C5RC8J. The official release
contains 7,062,606 records from nine commercial IoT devices and 115
traffic-statistic features. The data are described as multivariate and
sequential. Each physical device is one federated client; the primary
study MUST NOT create Dirichlet pseudo-clients.

### 7.1.1 Device inventory and benign-budget feasibility

The implementation MUST derive actual row counts from the acquired
files and store them in the manifest. The following literature-reported
counts are a **preflight cross-check**, not a substitute for counting the
files:

| Client ID | Device | Benign rows cross-check | Benign rows consumed before final test | Expected remaining benign rows |
|---|---|---:|---:|---:|
| nb01 | Danmini Doorbell | 49,548 | 10,000 | 39,548 |
| nb02 | Ennio Doorbell | 39,100 | 10,000 | 29,100 |
| nb03 | Ecobee Thermostat | 13,113 | 10,000 | 3,113 |
| nb04 | Philips B120N/10 Baby Monitor | 175,240 | 10,000 | 165,240 |
| nb05 | Provision PT-737E Security Camera | 62,154 | 10,000 | 52,154 |
| nb06 | Provision PT-838 Security Camera | 98,514 | 10,000 | 88,514 |
| nb07 | SimpleHome XCS7-1002-WHT Security Camera | 46,585 | 10,000 | 36,585 |
| nb08 | SimpleHome XCS7-1003-WHT Security Camera | 19,528 | 10,000 | 9,528 |
| nb09 | Samsung SNH-1011N Webcam | 52,150 | 10,000 | 42,150 |
| **Total** |  | **555,932** | **90,000** | **465,932** |

The Ecobee client is the limiting benign-data case. The fixed design
leaves 3,113 final benign rows, so the protocol's minimum of 3,000 is
feasible with only 113 rows of slack. Any mismatch between the actual
source files and the locked feasibility assertion triggers
`DATASET_COUNT_MISMATCH` and MUST be investigated before training.

### 7.1.2 Benign role partition

| Role | Per-client count | Exact construction | Permitted use |
|---|---:|---|---|
| \(T_k\) benign train | 4,000 | First 4,000 benign rows in source-file order | Model/scaler fitting only |
| calibration reservoir | 6,000 | Next 6,000 benign rows in source-file order | Source for R/G/C/guard |
| \(R_k\) reference | 500 | First 500 positions of seeded permutation of the 6,000-row reservoir | Federation reference only |
| \(G_k\) gate | 3,000 | Permutation positions 501–3500 | Gate B only |
| \(C_k\) local calibration | 2,000 | Permutation positions 3501–5500 | Gate A/local threshold only |
| comparator benign guard | 500 | Permutation positions 5501–6000 | Attack-aware baselines only |
| \(B_k\) final benign test | all rows after first 10,000 | Never subsampled for primary test | Final evaluation only |

**Order claim.** The adapter preserves UCI source-file row order. The
paper MUST use the phrase **source-order holdout** unless a timestamp or
capture-order provenance is independently verified. It MUST NOT call
the split chronological merely because rows are sequentially stored.

**Primary role assignment.** Calibration seed 1000 is the single named
confirmatory role assignment. Seeds 1001–1049 are split-sensitivity
replicates. They quantify dependence on how the historical reservoir is
allocated to R/G/C/guard; they are not independent device samples and
are never used to inflate inferential degrees of freedom.

### 7.1.3 Attack inventory and attack-label firewall

N-BaIoT contains five BASHLITE/Gafgyt attack subtypes
`combo`, `junk`, `scan`, `tcp`, `udp` for every device. Seven devices
also contain five Mirai subtypes `ack`, `scan`, `syn`, `udp`,
`udpplain`; Ennio and Samsung do not contain Mirai. The adapter MUST
derive the exact available attack-file set rather than assume ten
subtypes for all clients.

For each client:

1. Let \(m_k\) be the number of attack subtypes present, normally 10 or
   5.
2. Construct `A_dev,k` with exactly **500 malicious records** total.
   Allocate \(\lfloor500/m_k\rfloor\) records to every subtype, then
   distribute the remainder one record at a time in lexicographic
   subtype order. Sampling within subtype uses seed 9001 and is without
   replacement.
3. `A_test,k` is every remaining malicious row. Every present subtype MUST retain at least 100 final-test rows after the 500-record development allocation; otherwise stop with `NBAIOT_ATTACK_BUDGET_FAIL` rather than consume a rare subtype for comparator tuning.
4. `A_dev,k` and all attack labels are stored under a path namespace
   that FedCRG fitting code is forbidden to open.
5. The 500-anomaly development budget is paired with the 500-benign
   comparator guard, giving the attack-aware F1 baselines a fixed 50:50
   development prevalence. This is an intentionally favorable,
   explicitly supervised comparator and is not a deployment assumption
   for FedCRG.

### 7.1.4 N-BaIoT integrity assertions

Before model training, all of the following MUST pass:

- exactly nine canonical device directories are mapped to the fixed
  `nb01`–`nb09` IDs;
- each benign and attack CSV has exactly 115 numeric model columns
  after parser normalization;
- no selected N-BaIoT feature contains NaN or ±inf;
- source files and normalized intermediate files have SHA-256 hashes;
- every row has a stable `row_id = SHA256(dataset_id || client_id ||
  source_file_relative_path || source_row_index)`;
- \(T_k\), reservoir, \(B_k\), `A_dev,k`, and `A_test,k` are pairwise
  disjoint by `row_id`;
- within each calibration seed, R/G/C/guard are pairwise disjoint and
  their union equals the 6,000-row reservoir;
- every client has at least 3,000 final benign rows;
- no attack row is present in \(T,R,G,C,B\);
- no final-test row contributes to imputation, scaling, training,
  thresholding, model selection, or comparator tuning.


## 7.2 CIC IoT-DIAD 2024 - external natural-device validation

> **Data-acquisition amendment (post-freeze).** The acquired CIC IoT-DIAD
> 2024 packet-based feature release exposes **115** unique `device_mac`
> identities across all source CSVs, confirmed by direct scan (raw and
> MAC-normalized counts agree), not the 105 assumed below. The dataset's
> own documentation describes this release as covering "approximately 45
> device categories" for identification, which is consistent with a
> device_mac count that does not equal the topology's device total. This
> is a data-identity count, not a scientific formula, threshold, or
> leakage boundary; it is corrected here rather than silently overriding
> `DATASET_COUNT_MISMATCH`/`DIAD_DEVICE_COUNT_SOURCE_MISMATCH`, which
> exist precisely to force this kind of discrepancy into view before
> training. `expected_source_clients` is 115 in the frozen configuration
> and `DIAD_EXPECTED_SOURCE_CLIENTS` in code; the client eligibility rule,
> role construction, and all other locked DIAD values below are
> unaffected.

**Dataset rationale.** The official CIC IoT-DIAD 2024 topology contains 105 IoT devices and 33 attacks across seven categories. The packet-based DI_AD representation exposes device_mac as a device-identification label and an anomaly label, allowing natural device-level clients while retaining a different feature-generation pipeline from N-BaIoT.

**Client eligibility is locked before outcome analysis:** a device is eligible only if it has at least 7800 benign packet rows, at least 1000 malicious packet rows after schema cleaning, and enough per-category development capacity to reserve final-test attack evidence as specified in Section 7.2.3. All eligible devices are used; no performance-based client selection and no cap on K are permitted. If fewer than 10 devices satisfy the rule, CIC IoT-DIAD is declared unsuitable for confirmatory external validation and the manuscript cannot claim two-dataset external replication until a replacement natural-client dataset is pre-specified before inspecting outcome results.

| **Role**                           | **Per eligible DIAD client**           | **Use**                                                                   |
|------------------------------------|----------------------------------------|---------------------------------------------------------------------------|
| T_k                                | 2000 benign                            | Training only                                                             |
| Calibration reservoir              | 3800 benign                            | Role permutation source                                                   |
| R_k                                | 300 benign                             | Reference threshold                                                       |
| G_k                                | 1500 benign                            | Gate B; LOW if x\<=2, HIGH if x\>=33                                      |
| C_k                                | 1500 benign                            | Gate A; r\*=1487; exact in-band probability 0.9573929                     |
| Supervised-comparator benign guard | 500 benign                             | Never visible to FedCRG; used only with A_dev by attack-aware comparators |
| B_k                                | All remaining benign; \>=2000 required | Final benign test                                                         |
| A_dev,k                            | 500 malicious records, category-balanced | Supervised comparators only; fixed 50:50 dev prevalence with 500 benign guard |
| A_test,k                           | All remaining malicious rows; >=500 required | Final attack test; every originally present category is preserved by the reserve rule |

### 7.2.1 DIAD client identity and stable row ordering

The external adapter MUST establish client identity before any model
input matrix is created.

1. Normalize `device_mac` only for partitioning by trimming whitespace
   and converting hex characters to lowercase.
2. Map each unique normalized device MAC to a public artifact ID
   `diad_<sha256(normalized_device_mac)[:12]>`.
3. `device_mac` itself MUST NOT enter the model feature matrix.
4. Build a deterministic within-client benign order:
   - if the official packet schema supplies a parseable capture-time
     field for all retained rows, sort by capture time ascending and
     break ties with `(source_file_relative_path, source_row_index)`;
   - otherwise sort by `(source_file_relative_path,
     source_row_index)` only and set `verified_chronology=false`.
5. The manifest stores which ordering branch was used. Terms such as
   *chronological*, *temporal holdout*, and *drift over time* are
   permitted only when `verified_chronology=true`.

### 7.2.2 DIAD benign-role construction

For every eligible client, apply the stable benign ordering before any
calibration-role randomization:

| Segment | Count | Construction |
|---|---:|---|
| \(T_k\) | 2,000 | first 2,000 ordered benign records |
| calibration reservoir | 3,800 | next 3,800 ordered benign records |
| \(B_k\) | all remaining | final benign evaluation, never used upstream |

For calibration seed \(c\), permute only the 3,800-record reservoir
without replacement using a NumPy `PCG64` generator seeded from
`SHA256("fedcrg|diad|calibration|" || c || client_id)` reduced to an
unsigned 64-bit integer. Assign:

- permutation positions 1–300 → \(R_k\);
- positions 301–1800 → \(G_k\);
- positions 1801–3300 → \(C_k\);
- positions 3301–3800 → comparator benign guard.

The named split is seed 2000. Seeds 2001–2019 are split-sensitivity
runs. The hash-derived per-client seed prevents accidental dependence
on loop ordering or Python's process-randomized `hash()` function.

### 7.2.3 DIAD malicious development/test construction

The 500-malicious-record development budget is deterministic and
category-balanced without assuming every device exhibits all seven
attack categories.

For client \(k\):

1. Let \(\mathcal A_k\) be the set of official non-benign attack
   categories with at least one retained record, and let \(n_{ka}\) be
   each category count.
2. Reserve final-test evidence **before** development sampling. For every
   present category define
   \[
   r_{ka}=\min(100,n_{ka}),\qquad d_{ka}^{\max}=n_{ka}-r_{ka}.
   \]
   Thus a category with at most 100 records is used only for final test,
   while a larger category keeps at least 100 final-test records.
3. DIAD eligibility requires total malicious count \(\ge1000\) **and**
   \(\sum_a d_{ka}^{\max}\ge500\). These two checks guarantee at
   least 500 final malicious rows overall and make the fixed 500-record
   supervised development budget feasible without deleting a present
   attack category from final evaluation.
4. Allocate the 500 development records by the following exact
   capacity-aware water-filling algorithm. Initialize `dev[a]=0` for
   every category. For allocation step `j=1,...,500`, form
   `E={a : dev[a] < dmax[a]}`. Eligibility guarantees `E` is non-empty.
   Let `m=min(dev[a] for a in E)` and choose the lexicographically first
   category in `{a in E : dev[a]=m}`; increment that category by one.
   This produces the most even deterministic allocation possible subject
   to the per-category capacity constraints. The implementation MUST
   assert `sum(dev.values())==500` and `0 <= dev[a] <= dmax[a]` for
   every category. No random category-ordering is permitted.
5. Within each category, sample the allocated development rows without
   replacement using a `PCG64` seed derived from
   `SHA256("fedcrg|diad|attackdev|9001|" || client_id || category)`.
6. The selected 500 records form `A_dev,k`; every other malicious
   record forms `A_test,k`. By construction, every attack category that
   existed before the split remains represented in `A_test,k`.
7. FedCRG fitting code cannot import or receive either the attack label
   column or an `A_dev/A_test` path. Only B7–B9 may access `A_dev,k`.

This creates an exactly 1,000-record attack-aware development set per
eligible client when paired with the 500 benign guard, fixing
development prevalence at 50% benign / 50% malicious for F1-based
comparators.

### 7.2.4 Eligibility freeze and integrity assertions

Eligibility is evaluated after schema parsing but before training or
threshold outcomes. A client is eligible iff all conditions hold:

\[
n_{\mathrm{benign}}\ge7800,\qquad
n_{\mathrm{malicious}}\ge1000,\qquad\sum_{a\in\mathcal A_k}\max(0,n_{ka}-\min(100,n_{ka}))\ge500,
\]

all 86 required model features exist, every training client-feature
finite-rate check in Section 7.4 passes, and the client has a valid
stable identifier.

The adapter MUST emit `diad_eligibility.json` containing every discovered
device, raw benign/malicious counts, exclusion reason if any, and the
final ordered eligible-client list. This file is hashed and frozen
before the first DIAD model is trained.

**Locked DIAD exclusion codes.** Each excluded device receives exactly one
primary code selected in the following precedence order: `ID_INVALID`,
`FEATURE_MISSING`, `FINITE_RATE_FAIL`, `BENIGN_COUNT_LT_7800`,
`MALICIOUS_COUNT_LT_1000`, `ATTACK_DEV_CAPACITY_LT_500`. Secondary
violations may be retained in a diagnostic list, but the primary code is
stable so eligibility manifests can be compared byte-for-byte across
implementations.

Required assertions:

- the official source exposes 105 device identities before protocol
  filtering; otherwise flag `DIAD_DEVICE_COUNT_SOURCE_MISMATCH`;
- \(T_k\), reservoir, \(B_k\), `A_dev,k`, and `A_test,k` are pairwise
  disjoint by stable row ID;
- for each calibration seed, R/G/C/guard are pairwise disjoint and
  exactly cover the 3,800-record reservoir;
- every included client retains at least 2,000 final benign rows;
- every included client has exactly 500 malicious development records and at least 500 final malicious records;
- for every attack category present before the split, `A_test,k` retains exactly `n_ka-dev[a] >= min(100,n_ka)` records;
- no direct identifier, label, port, application string, or excluded
  column appears in the 86-column model tensor;
- client eligibility is never revised because a client's detector or
  FedCRG performance is poor.

If fewer than ten clients remain, R10 is labeled
`EXTERNAL_DATASET_INSUFFICIENT_CLIENTS`. N-BaIoT results remain valid,
but the manuscript MUST NOT claim confirmatory two-dataset replication.

## 7.3 DIAD feature contract - exactly 86 numeric behavior features

**Direct identifiers, labels, ports, application strings, and device-ID fields are excluded from model inputs.** The external experiment uses the following fixed 86-feature numeric subset from the official packet-based schema:

- inter_arrival_time

- time_since_previously_displayed_frame

- l4_tcp

- l4_udp

- ttl

- eth_size

- tcp_window_size

- payload_entropy

- payload_length

- l3_ip_dst_count

- jitter

**For each window w in {1, 5, 10, 30, 60}, include all 15 of:** stream_w_count, stream_w_mean, stream_w_var, src_ip_w_count, src_ip_w_mean, src_ip_w_var, src_ip_mac_w_count, src_ip_mac_w_mean, src_ip_mac_w_var, channel_w_count, channel_w_mean, channel_w_var, stream_jitter_w_sum, stream_jitter_w_mean, stream_jitter_w_var. This contributes 75 features; 11 base + 75 windowed = 86 total.

**Explicitly excluded:** stream, device_mac, src_ip, dst_ip, src_port, dst_port, port_class_dst, all TLS/HTTP/DNS/OUI/user-agent/URI textual or identity-bearing fields, most_freq_spot, both labels, and every column not in the 86-feature allowlist. device_mac is retained only as client-partition metadata and never enters the model.

### 7.3.1 Why this 86-feature representation is locked

The official DIAD packet schema contains 134 columns and explicitly identifies stream/channel/jitter statistics over the 1, 5, 10, 30, and 60 windows as anomaly-detection-oriented behavior features. The confirmatory representation therefore uses **the complete 75-column multiscale behavior block** plus the 11 low-level numeric packet/traffic measurements listed above. This is a protocol-level feature rule fixed before detector outcomes; no univariate attack association, feature importance, AUROC, F1, or final-test statistic is used to select these columns.

The one-off aggregate block `min_et` through `iqr_p` (official columns 43–56), protocol/application metadata, ports, direct identifiers, textual/categorical fields, and labels are deliberately excluded from the confirmatory tensor. This prevents a partially mixed representation from making external validation depend on identity-bearing or protocol-specific fields while retaining the complete multiscale behavior family. The manuscript MUST describe this as a **locked representation choice**, not as an empirically optimal feature subset.

### 7.3.2 Feature-contract sensitivity

To expose any dependence on the 86-column choice, R14 constructs a **training-schema-only numeric-safe sensitivity representation**. Starting from the official packet schema, remove direct identifiers, labels, IP/MAC addresses, ports, and fields whose parser type is non-numeric. On \(T_k\) only, retain a remaining column iff every eligible client has at least 99.0% finite values for that column. The resulting feature names and dimension \(d_{R14}\) are frozen before any calibration/test score is evaluated. R14 uses the deterministic symmetric AE

`d -> floor(0.75d) -> floor(0.50d) -> floor(d/3) -> floor(0.25d) -> floor(d/3) -> floor(0.50d) -> floor(0.75d) -> d`,

with every hidden width lower-bounded at 1. All other DIAD optimizer/training settings remain 30 rounds × 20 local epochs. This architecture rule is specific to R14 and is not retroactively applied to the locked 86-dimensional confirmatory AE. R14 is exploratory and cannot replace the 86-feature confirmatory DIAD result.

## 7.4 Missing, non-finite, preprocessing, and scaling rules

### 7.4.1 N-BaIoT

- Any missing, non-numeric, NaN, or infinite model feature is a hard
  parser failure; there is no N-BaIoT imputation.
- Each client computes per-feature minima and maxima on \(T_k\) only.
  The server computes \(m_j=\min_k m_{kj}\) and
  \(M_j=\max_k M_{kj}\), then broadcasts the 115 global extrema pairs.
- Scale every train/calibration/test value with
  \[
  z_{ij}=\frac{x_{ij}-m_j}{M_j-m_j}.
  \]
  If \(M_j=m_j\), set \(z_{ij}=0\) for all rows for that feature and
  record `constant_feature=true`.
- Calibration/test values are **not clipped** to \([0,1]\). Values
  outside the training range remain outside the interval because
  clipping would alter anomaly-score geometry.

### 7.4.2 CIC IoT-DIAD

- Parse the fixed 86-feature allowlist and coerce each selected feature
  to numeric; ±inf becomes NaN.
- For each client and selected feature, at least 99.0% of \(T_k\) rows
  MUST be finite. If any client-feature pair violates this rule, return
  `DIAD_FEATURE_FINITE_RATE_FAIL`; do not drop that feature post hoc.
- Remaining missing values are imputed with the **client-local median
  fitted on that client's \(T_k\) only**. The 86 local medians are
  serialized per client and applied unchanged to that client's
  R/G/C/guard/final-test rows.
- After local imputation, global min/max scaling is computed
  federatively from the imputed \(T_k\) values using the same extrema
  exchange as N-BaIoT.
- The preprocessing object is frozen before any calibration or attack
  score is computed.

### 7.4.3 Preprocessing communication and privacy accounting

Global min/max scaling is an explicit federated preprocessing step, not
a free centralized operation. For input dimension \(d\), each client
transmits \(2d\) float64 extrema. N-BaIoT therefore transmits
\(230\times8=1,840\) bytes/client, or 16,560 client-upload bytes across
nine clients, excluding protocol overhead. DIAD transmits
\(172\times8=1,376\) bytes per eligible client.

These extrema, reference scores, model updates, and any comparator
summary statistics are **derived data** and may leak information.
FedCRG makes no formal differential-privacy, secure-aggregation, or
cryptographic confidentiality claim.

### 7.4.4 Absolute leakage prohibition

No statistic computed from \(B_k\), `A_dev,k`, or `A_test,k` may affect
feature selection, parser repair, imputation, scaling, model
architecture, optimization hyperparameters, reference construction,
Gate A, Gate B, or any benign-only baseline.


# 8. Frozen Detector and Federated Training Specification

**Scientific role.** The detector is deliberately conventional. FedCRG
is evaluated as a post-training operating-point governance layer. The
primary manuscript MUST NOT present the autoencoder architecture,
FedAvg-style aggregation, Adam optimizer, or reconstruction score as a
FedCRG contribution.

## 8.1 Primary federated autoencoder

| Parameter | N-BaIoT primary | DIAD external |
|---|---|---|
| Input dimension | 115 | 86 |
| Architecture | 115-86-57-38-29-38-57-86-115 | 86-64-43-28-21-28-43-64-86 |
| Trainable parameters implemented | 36,626 | 20,473 |
| Hidden activation | tanh | tanh |
| Output activation | linear | linear |
| Initialization | Xavier uniform, tanh gain \(5/3\); all biases zero | same |
| Objective | mean feature-wise reconstruction MSE | same |
| Per-sample anomaly score | \(\frac1d\sum_{j=1}^d(x_j-\hat x_j)^2\) | same |
| Local optimizer | Adam | Adam |
| Adam betas | (0.9, 0.999) | same |
| Adam epsilon | \(10^{-8}\) | same |
| Weight decay | 0 | 0 |
| Initial LR | \(10^{-3}\) | \(10^{-3}\) |
| Final LR | \(10^{-5}\) | \(10^{-5}\) |
| Batch size | 64 | 64 |
| Global rounds | **30** | **30** |
| Local epochs/round | **120** | **20** |
| Client participation | 100% | 100% of eligible clients |
| Local training rows | 4,000/client | 2,000/client |
| Aggregation | equal arithmetic mean of client parameter tensors | same |
| Optimizer state | reset at each round after global weights are loaded | same |
| Shuffle | deterministic by `(model_seed,client_id,round,local_epoch)` | same |
| Gradient clipping | none | none |
| Early stopping | prohibited in confirmatory run | prohibited |
| Mixed precision | off for primary | off for primary |
| Score storage | float64 after float32 forward pass | same |

### 8.1.1 Learning-rate schedule

For each dataset, for \(t\in\{0,\ldots,29\}\),

\[
\eta_t=\eta_{\min}+
\frac12(\eta_0-\eta_{\min})
\left[1+\cos\left(\frac{\pi t}{29}\right)\right],
\]

with \(\eta_0=10^{-3}\) and \(\eta_{\min}=10^{-5}\). The learning rate is
constant across all local epochs within round \(t\).

These endpoints are **FedCRG protocol choices**. FedDetect explicitly
uses a cross-round cosine scheduler but the paper text does not lock
these exact endpoint values; the manuscript MUST NOT attribute
\(10^{-3}\rightarrow10^{-5}\) to FedDetect unless the archived source
code used for reproduction verifies them.

### 8.1.2 Literature-faithful N-BaIoT relation

The primary FedDetect paper reports:

- N-BaIoT with nine commercial IoT devices;
- 115-dimensional inputs;
- a symmetric deep autoencoder using decreasing encoder dimensions;
- global min-max scaling fitted from benign training data;
- local Adam;
- a cross-round cosine learning-rate scheduler;
- hyperparameter search over learning rate, batch size, local epochs,
  rounds, and tanh/sigmoid;
- selected **batch size 64, 120 local epochs, 30 rounds, tanh**; and
- a federation global threshold computed from pooled reconstruction
  errors.

Accordingly, N-BaIoT uses the same reported **30×120** training scale to
avoid making detector tuning part of the FedCRG contribution.

**Parameter-count discrepancy.** The FedDetect paper reports **36,628**
autoencoder parameters. The explicit biased-linear architecture
`115-86-57-38-29-38-57-86-115` contains **36,626** parameters under the
standard formula

\[
P=\sum_{\ell=0}^{L-1}(d_\ell d_{\ell+1}+d_{\ell+1}).
\]

FedCRG uses the explicit layer dimensions as the normative
implementation and records the resulting 36,626 count. The two-parameter
difference is documented as a source-level reproducibility discrepancy;
the code MUST NOT add dummy parameters merely to force the paper's
reported count.

### 8.1.3 Why DIAD uses 30×20 rather than 30×120

The external dataset is not used to reproduce FedDetect. DIAD may have
up to 105 eligible natural clients and uses a different 86-feature
representation. To keep external validation computationally bounded
without outcome-based early stopping, DIAD uses the locked
**30-round/20-local-epoch** schedule from protocol freeze.

This is not tuned on DIAD test outcomes. The following diagnostics are
required for every DIAD model seed but cannot select a checkpoint:
round-wise mean client training loss, min/max client training loss,
parameter-update norm, and final-vs-round-20 score correlation on
training data. If optimization is numerically unstable, the run fails
with `TRAINING_NUMERICAL_FAILURE`; it is not silently retuned.

### 8.1.4 Published FedDetect settings versus FedCRG protocol settings

The N-BaIoT detector is **FedDetect-aligned, not a FedDetect reproduction**. The exact differences are pre-declared:

| Item | FedDetect paper | FedCRG N-BaIoT | Reason / claim discipline |
|---|---:|---:|---|
| benign training/client | 5,000 | 4,000 | FedCRG reserves a larger independent calibration budget; detector is a nuisance component, not the contribution |
| benign threshold/evaluation/client | 3,000 | R+G+C = 5,500 for benign-only policy work; 500 extra benign guard for supervised baselines | Required by separate reference, mismatch, readiness, and comparator roles |
| batch | 64 | 64 | literature-aligned |
| local epochs/round | 120 | 120 | literature-aligned |
| global rounds | 30 | 30 | literature-aligned |
| hidden activation | tanh | tanh | literature-aligned |
| global scaling | min/max from federation benign training | same statistic, implemented through extrema exchange | prevents silent centralization |
| threshold | pooled mean + \(3\sigma\) | FedCRG primary; mean+3σ retained as B6 only | thresholding is the research object |
| exact LR endpoints | not fixed by paper text | \(10^{-3}\to10^{-5}\) cosine | FedCRG engineering lock, not attributed to prior work |
| optimizer-state carry across rounds | not unambiguously fixed by paper text | reset every round | deterministic implementation lock |

The manuscript MUST NOT write “we reproduce FedDetect” unless a separate source-code parity experiment is performed. It MAY write that the detector architecture/training scale is **FedDetect-inspired/aligned** and cite the exact deviations above.

## 8.2 Federated training state machine

For each model seed:

1. Build and hash the dataset manifest.
2. Fit client-local imputers where applicable using \(T_k\) only.
3. Compute and federatively aggregate global min/max preprocessing
   extrema from \(T_k\) only.
4. Initialize one global AE from `model_seed`.
5. For rounds \(t=0,\ldots,29\):
   - broadcast the current global parameter tensors;
   - every client loads them into an identical model;
   - create a **fresh Adam optimizer** at \(\eta_t\); optimizer moments
     do not persist across global rounds;
   - deterministically shuffle that client's \(T_k\) separately for
     every local epoch;
   - train exactly 120 local epochs on N-BaIoT or 20 on DIAD;
   - use ordinary final mini-batches; `drop_last=false`;
   - upload the complete float32 parameter tensors;
   - server verifies tensor names, shapes, dtypes, and finite values;
   - server computes the unweighted arithmetic mean tensor-by-tensor;
   - serialize round hash and training diagnostics.
6. Freeze the final global weights after round 29.
7. Compute every benign and malicious score required by the experiment.
8. Convert stored scalar scores to IEEE-754 float64.
9. Serialize one immutable score cache per dataset/model seed.
10. Threshold-policy code may run only after the score cache receives a
    finalized SHA-256 hash.

### 8.2.1 Exact local batch semantics

For client \(k\) with \(n_k\) training records and batch size \(B=64\),

\[
N_{\text{batch},k}=\left\lceil\frac{n_k}{64}\right\rceil.
\]

With equal locked training counts:

- N-BaIoT: \(\lceil4000/64\rceil=63\) optimizer steps/local epoch,
  \(63\times120=7,560\) local optimizer steps/client/round, and
  \(226,800\) steps/client over 30 rounds.
- DIAD: \(\lceil2000/64\rceil=32\) optimizer steps/local epoch,
  \(32\times20=640\) local optimizer steps/client/round, and
  \(19,200\) steps/client over 30 rounds.

The smaller final mini-batch is included and its loss is averaged over
its actual record count. Epoch loss is the record-weighted mean MSE,
not the unweighted mean of batch means.

### 8.2.2 No hidden model selection

No validation loss, anomaly label, F1, AUROC, final-test metric, Gate
state, or threshold-policy result may select a training round,
architecture, learning rate, or checkpoint. The confirmatory detector
is the global model after round 29. Earlier checkpoints may be plotted
as training diagnostics only.

## 8.3 Exact model communication accounting

Ignoring serialization, optimizer state, transport headers, retries,
TLS/MQTT framing, and server-side storage:

- N-BaIoT model: 36,626 float32 parameters =
  **146,504 bytes** per complete model tensor payload.
- One full N-BaIoT round with 9 clients, counting one server→client
  model broadcast copy and one client→server upload per client:
  \[
  2\times9\times146,504=2,637,072\text{ bytes}.
  \]
- 30-round N-BaIoT training:
  \[
  30\times2,637,072=\mathbf{79,112,160}\text{ bytes}
  \]
  = **79.112160 MB decimal** of model tensors.
- Per N-BaIoT client over 30 rounds:
  \(2\times30\times146,504=8,790,240\) tensor bytes.
- DIAD model: 20,473 float32 parameters =
  **81,892 bytes** per complete model payload.
- For \(K_D\) eligible DIAD clients and 30 rounds:
  \[
  C_{\mathrm{DIAD}}=
  2\times K_D\times30\times81,892
  =4,913,520K_D\text{ bytes}.
  \]
- If all 105 official devices were eligible, the tensor-only upper
  reference would be **515,919,600 bytes = 515.919600 MB decimal**.

These counts are deterministic protocol accounting, not measured
network traffic. R13 separately measures serialized wall-time and
memory overhead.

## 8.4 Mandatory second score generator

The second detector is **mandatory and outcome-independent**; it is not
run only when the AE result is favorable.

| Parameter | Federated Deep-SVDD sensitivity |
|---|---|
| Dataset | N-BaIoT |
| Encoder | 115-64-32, tanh, biases disabled |
| Embedding dimension | 32 |
| Center initialization | initialize encoder from model seed; each client computes mean embedding on \(T_k\); server equal-averages nine client means; center then frozen |
| Loss | \(\frac1B\sum_i\|f_\theta(x_i)-c\|_2^2\) |
| Anomaly score | \(\|f_\theta(x)-c\|_2^2\) |
| Rounds | 30 |
| Local epochs/round | 20 |
| Batch size | 64 |
| Optimizer | Adam, betas=(0.9,0.999), eps=1e-8, weight_decay=0 |
| LR schedule | same 1e-3 to 1e-5 30-round cosine schedule |
| Client participation | 100% |
| Aggregation | equal client mean |
| Model seeds | 11, 22, 33 |
| Calibration seeds | 1000–1009; 1000 named split, rest sensitivity |
| Policies | GLOBAL-Q99-FULL, LOCAL-Q99-FULL, SHRINKAGE, GATE-A-ONLY, FedCRG |

The center \(c\) is computed once **before Deep-SVDD training** from the
seed-initialized encoder and is not recomputed after each FL round.
Threshold policies receive only the final frozen score cache.

If the qualitative pattern fails to replicate, the manuscript scopes
the empirical claim to reconstruction-error systems rather than
suppressing the second-detector result.


# 9. Baseline Suite and Information-Regime Fairness

**Mandatory principle.** FedCRG uses multiple disjoint benign pools. A reviewer must not be able to argue that a weak comparator was deliberately starved of calibration data. Therefore the study includes both role-matched and **full benign-policy-budget** shared/local comparators. Here `FULL` means all benign samples available to benign-only threshold policies, \(R+G+C\): 5,500/client on N-BaIoT and 3,300/client on DIAD. The separate 500-record benign guard is intentionally withheld from B0–B6 because it is the independent development half used by attack-aware B7–B9; using it to fit B1/B2 would leak those development labels into the selector comparison.

| **ID**              | **Information**                   | **Exact rule**                                                                                                                                | **Purpose**                                                                                        |
|---------------------|-----------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| B0 REF-Q99-R        | Benign only                       | tau_ref from R only                                                                                                                           | Reference used internally by FedCRG; not presented as strongest shared baseline.                   |
| B1 GLOBAL-Q99-FULL  | Benign only                       | Pool every client R+G+C (5,500 N-BaIoT; 3,300 DIAD) with equal per-client counts; `q=min(N,ceil((N+1)(1-alpha)))`; strict `>`                                            | Strong full benign-policy-budget always-shared comparator.                                                         |
| B2 LOCAL-Q99-FULL   | Benign only                       | Per client R+G+C; `q=min(n,ceil((n+1)(1-alpha)))`; strict `>`                                                                                     | Strong full benign-policy-budget always-local comparator.                                                          |
| B3 GATE-A-ONLY      | Benign only                       | If Gate A is sample-size READY **and** the selected local order statistic has multiplicity 1, use `tau_local`; otherwise use `tau_ref`                                           | Ablates Gate-B personalization-necessity evidence while retaining the same local-readiness and continuity preconditions as FedCRG. |
| B4 GATE-B-ONLY      | Benign only                       | If Gate B mismatch, use `C_(q_C)` with `q_C=min(n_C,ceil((n_C+1)(1-alpha)))`; otherwise tau_ref                                                                               | Ablates finite-sample readiness.                                                                   |
| B5 SHRINKAGE        | Benign only                       | tau_shr = w\*tau_local,Q99 + (1-w)\*tau_ref, w=n_C/(n_C+n0)                                                                                   | Required due adjacent shrinkage literature.                                                        |
| B6 FEDDETECT-3SIGMA | Benign only                       | Pool R+G+C scores; threshold = global mean + `3*sqrt(mean((s-mean)^2))` (`ddof=0`)                                                                                    | Published-style federated AE threshold baseline.                                                   |
| B7 DEV-F1-LG-SELECT | Benign guard + attack development | Per client choose between B1 and B2 using F1 on disjoint guard + A_dev; tie -\> B1                                                            | Closest simple attack-aware answer to "local or global?"; extra information is explicit.           |
| B8 LARIDI-STYLE-SS   | Benign guard + attack development | Locked unrefined summary-statistics overlap inspired by Laridi et al.; 1000 candidates; equal-client mean F1 | Closest-prior **style** comparator; deliberately not labeled an exact Algorithm-2 reproduction. |
| B9 SUP-F1-1000      | Benign guard + attack development | 1000 federation-wide candidates spanning development-score min/max; equal-client mean F1; maximize F1                                         | Strong attack-aware candidate-search comparator independent of Laridi overlap assumptions.         |
| B10 ORACLE-TEST     | Final labels                      | For each client choose whichever of GLOBAL-Q99-FULL, LOCAL-Q99-FULL, or FedCRG gives smallest final-test band error; break ties by higher TPR | Unattainable diagnostic ceiling; never described as deployable.                                    |

## 9.1 Deterministic quantile-rank ledger

Every `Q99` baseline uses the **same finite-sample rank convention**

\[
q(N,\alpha)=\min\{N,\lceil (N+1)(1-\alpha)\rceil\},
\]

with ascending order statistics and anomaly iff `score > threshold`. For the primary \(\alpha=0.01\):

| Object | Sample count | Exact rank |
|---|---:|---:|
| N-BaIoT `REF-Q99-R` | \(9\times500=4,500\) | 4,456 |
| N-BaIoT `GLOBAL-Q99-FULL` | \(9\times5,500=49,500\) | 49,006 |
| N-BaIoT `LOCAL-Q99-FULL` | 5,500/client | 5,446 |
| N-BaIoT B4/B5 local Q99 from \(C_k\) | 2,000/client | 1,981 |
| DIAD `REF-Q99-R` | \(300K_D\) | \(q(300K_D,0.01)\); if \(K_D=105\), 31,186 |
| DIAD `GLOBAL-Q99-FULL` | \(3,300K_D\) | \(q(3,300K_D,0.01)\); if \(K_D=105\), 343,036 |
| DIAD `LOCAL-Q99-FULL` | 3,300/client | 3,268 |
| DIAD B4/B5 local Q99 from \(C_k\) | 1,500/client | 1,486 |

The N-BaIoT Gate-A rank \(r^*=1,982\) at \(n_C=2,000\) is intentionally **not** the empirical Q99 rank 1,981: Gate A maximizes finite-sample probability of landing inside the whole 0.5%–1.5% band rather than estimating exactly one point quantile.

## 9.2 Shrinkage baseline - exact tuning rule

Let `q_C=min(n_C,ceil((n_C+1)(1-alpha)))` and `tau_local,Q99=C_(q_C)`. Then `w(n0)=n_C/(n_C+n0)` and `tau_shr=w*tau_local,Q99+(1-w)*tau_ref`.

**Candidate n0 grid:** {100, 300, 1000, 3000, 10000}. This baseline operates in threshold-score space and is therefore not invariant to arbitrary monotone score transformations. It is retained because the detector score definition is fixed and identical across policies. A supplementary risk-curve shrinkage implementation MAY be added, but it cannot replace this locked baseline. For each n0, estimate each client FPR on G_k and compute mean absolute target-FPR error across clients. Choose the n0 with minimum mean error; ties choose the largest n0 (more pooling). No attack data and no final test data are used. This is explicitly labeled a shrinkage-style baseline, not a reproduction of Shahid (2026).

## 9.3 Attack-aware local-versus-global selector - exact rule

- B7 DEV-F1-LG-SELECT is intentionally stronger in supervision than FedCRG. It is not deployable under the benign-only assumption and exists to test whether attack labels materially improve the local-versus-global decision.

- Construct B1 GLOBAL-Q99-FULL and B2 LOCAL-Q99-FULL using R+G+C exactly as defined above.

- For each client, create a 1,000-record development set from exactly 500 comparator-benign guard scores plus exactly 500 attack-balanced `A_dev,k` scores. Neither set is visible to FedCRG.

- Compute F1 for the B1 and B2 thresholds on that client development set. Select B2 only if its F1 is strictly larger; ties select B1 to avoid unnecessary personalization.

- Freeze the selected choice and evaluate it once on B_k + A_test,k. No final-test outcome may influence selection.

## 9.4 Laridi et al. 2024 closest-prior comparator

**Reproduction label discipline.** The primary Scientific Reports text
establishes the relevant information regime and algorithmic idea:
normal and anomalous validation reconstruction errors, client summary
statistics, an overlap region, candidate thresholds, and client F1
aggregation. It also states that skewness and kurtosis refine the
overlap bounds, while the exact rendered Algorithm 2 details are not
fully represented in accessible machine-readable text. Therefore the
locked baseline is deliberately named **`LARIDI-STYLE-SS`**, not an
exact reproduction.

The implementation is fully specified as follows so that its behavior
does not depend on an ambiguous transcription.

For each client \(k\) and class
\(y\in\{\mathrm{benign},\mathrm{attack}\}\), on the fixed 1,000-record
development set compute:

\[
n_{ky},\qquad
\mu_{ky}=\frac1{n_{ky}}\sum_i s_i,\qquad
v_{ky}=\frac1{n_{ky}}\sum_i(s_i-\mu_{ky})^2.
\]

The client sends \((n_{ky},\mu_{ky},v_{ky})\) for both classes. For each
class, the server computes the exact pooled population moments:

\[
N_y=\sum_k n_{ky},
\]

\[
\mu_y=\frac{\sum_k n_{ky}\mu_{ky}}{N_y},
\]

\[
v_y=
\frac{\sum_k n_{ky}(v_{ky}+\mu_{ky}^2)}{N_y}
-\mu_y^2,
\qquad
\sigma_y=\sqrt{\max(v_y,0)}.
\]

Define the **locked Laridi-style unrefined overlap interval**

\[
\ell=
\max(\mu_{\mathrm{benign}}-3\sigma_{\mathrm{benign}},
     \mu_{\mathrm{attack}}-3\sigma_{\mathrm{attack}}),
\]

\[
u=
\min(\mu_{\mathrm{benign}}+3\sigma_{\mathrm{benign}},
     \mu_{\mathrm{attack}}+3\sigma_{\mathrm{attack}}).
\]

- If \(\ell\ge u\), record `LARIDI_STYLE_UNDEFINED`; do not invent a
  fallback for this baseline.
- Otherwise generate exactly **1,000** equally spaced thresholds
  including \(\ell\) and \(u\):
  \(t_j=\ell+j(u-\ell)/999,\;j=0,\ldots,999\).
- Each client evaluates F1 for all candidates on its own balanced
  500-benign/500-malicious development set.
- The server computes the **equal-client arithmetic mean F1** for each
  \(t_j\).
- Select the candidate with maximal mean F1. Exact ties select the
  **smaller threshold**.
- Freeze that threshold and evaluate it once on final test scores.

This comparator captures the closest prior paper's supervised
summary-statistic/F1 threshold-selection regime without claiming a
bit-for-bit reproduction of the paper's skewness/kurtosis refinement.
If author code or a manually verified full Algorithm-2 transcription
becomes available before protocol freeze, an additional
`LARIDI-ALG2-REPRO` sensitivity MAY be added, but it cannot replace the
locked style comparator or alter FedCRG.

## 9.5 SUP-F1-1000 - exact extra-information comparator

- For each client, use its supervised-comparator benign guard plus A_dev,k with labels. FedCRG never reads either source.

- Find the federation-wide minimum and maximum development scores across participating clients.

- Generate exactly 1000 linearly spaced threshold candidates including both endpoints.

- Each client computes F1 for all 1000 thresholds on its development set and sends the 1000-value vector to the server.

- The server takes an equal-client arithmetic mean F1 for each candidate and chooses the candidate with maximum mean F1; threshold ties choose the smaller threshold (higher sensitivity).

- This is a strong attack-aware global-threshold comparator independent of the Laridi-style overlap construction.

# 10. Evaluation Metrics and Reporting Contract

**Metric hierarchy.** FedCRG controls a benign operating point. Reliability metrics are primary. Attack utility is evaluated without allowing large attack files/categories to dominate. Precision/F1 are secondary because they depend on the artificial dataset prevalence.

For client \(k\), with final benign set \(B_k\) and malicious final-test set \(A_{\mathrm{test},k}\),

\[
\mathrm{FPR}_k=\frac{FP_k}{FP_k+TN_k},\qquad
\mathrm{TPR}_k=\frac{TP_k}{TP_k+FN_k}.
\]

The per-client distance outside the acceptable operating band is

\[
\mathrm{BandError}_k
=\max\{a-\mathrm{FPR}_k,\,0,\,\mathrm{FPR}_k-b\}.
\]

The locked federation-level reliability endpoints are

\[
\mathrm{MEBE}=\frac1K\sum_{k=1}^{K}\mathrm{BandError}_k,
\]

\[
\mathrm{HighExcess}=\max\left\{0,\max_k\mathrm{FPR}_k-b\right\},
\]

\[
\mathrm{BandViolationRate}=\frac1K\sum_{k=1}^{K}
\mathbf 1[\mathrm{FPR}_k<a\ \lor\ \mathrm{FPR}_k>b],
\]

\[
\mathrm{MAFE}=\frac1K\sum_{k=1}^{K}|\mathrm{FPR}_k-\alpha|.
\]

## 10.1 Primary attack-utility endpoint: attack-balanced macro recall

Let \(\mathcal A_k\) be the attack groups present for client \(k\): N-BaIoT uses the actual attack subtype/file (for example `gafgyt_combo`, `mirai_syn`); DIAD uses the official seven-category label. For each present group \(j\),

\[
\mathrm{TPR}_{kj}=\frac{TP_{kj}}{TP_{kj}+FN_{kj}}.
\]

Define

\[
\mathrm{ABTPR}_k=\frac1{|\mathcal A_k|}\sum_{j\in\mathcal A_k}\mathrm{TPR}_{kj},
\qquad
\mathrm{ABMacroTPR}=\frac1K\sum_{k=1}^{K}\mathrm{ABTPR}_k.
\]

Missing attack types are **absent**, not zero. Thus Ennio and Samsung are not penalized for having no Mirai files. This endpoint gives each attack group equal weight within a client and each client equal weight in the federation.

Ordinary \(\mathrm{MacroTPR}=K^{-1}\sum_k\mathrm{TPR}_k\), in which attack groups are implicitly weighted by their row counts, remains a secondary utility diagnostic.

## 10.2 Locked utility anchor and non-inferiority margin

For every `(dataset, model_seed, calibration_seed)` cell, define the benign-only utility anchor

\[
U_{\mathrm{anchor}}=
\max\{\mathrm{ABMacroTPR}_{\mathrm{GLOBAL}},
      \mathrm{ABMacroTPR}_{\mathrm{LOCAL}},
      \mathrm{ABMacroTPR}_{\mathrm{SHRINKAGE}}\}.
\]

A claimed operating-reliability gain is called **utility-preserving** only when

\[
\mathrm{ABMacroTPR}_{\mathrm{FedCRG}}-U_{\mathrm{anchor}}\ge -0.03.
\]

The 3-percentage-point margin is an operational design choice fixed before outcomes; it is not a theorem and sensitivity at 1 pp and 5 pp MUST be reported in the supplement.

## 10.3 Metric registry

| Class | Metric | Exact role |
|---|---|---|
| Primary reliability | MEBE | Mean client distance outside the locked FPR band. Lower is better. |
| Primary safety | HighExcess | Worst-client excess above upper band \(b\). Lower is better. |
| Primary utility | **ABMacroTPR** | Equal attack-group weight within client, then equal client weight. Higher is better. |
| Secondary reliability | BandViolationRate, MAFE, max FPR, FPR IQR | Operating-point diagnostics. |
| Secondary utility | MacroTPR, WorstClientTPR, worst-client ABTPR | Detect utility without/with attack-group balancing. |
| Readiness | Gate-A ready rate, LOW/HIGH mismatch rate, admission rate, deficit rate, Gate-B-insufficient rate, assumption-violation rate | Explains policy decisions. |
| Stability | threshold SD/IQR, state transition frequency across calibration seeds | Split sensitivity. |
| Detector-only | AUROC, AUPRC | MUST be invariant across threshold policies on identical cached scores to tolerance \(10^{-12}\). |
| Secondary decision | precision, F1, balanced accuracy | Never tunes FedCRG. Prevalence-sensitive; never compared across datasets as if prevalence were deployment-realistic. |
| Test-binomial reference | 95% exact Clopper-Pearson interval for each client FPR | Reported only as a binomial reference interval under an i.i.d.-Bernoulli test-record model; distinct from Gate-A coverage and not a deployment guarantee. |

## 10.4 Precision/F1 prevalence warning

The final malicious files in N-BaIoT and DIAD do not encode a deployment attack prevalence. Therefore final-test precision and F1 answer only “performance under this benchmark mixture.” They MUST NOT be translated into production positive predictive value, alerts/day, or incident prevalence. The fixed 50:50 development prevalence used by B7–B9 is likewise a comparator design choice and is disclosed wherever their F1 tuning is discussed.

# 11. Experiment Registry - Locked Before Main Outcome Analysis

| **ID** | **Experiment**                          | **Scale**                                                                  | **Locked details**                                                                                                                                                                                                         |
|--------|-----------------------------------------|----------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| S1     | IID Gate-A theorem validation           | 4 distributions x 8 n_C values x 10,000 repetitions                        | Normal(0,1), LogNormal(0,1), Gamma(shape=2,scale=1), 0.9N(0,1)+0.1N(3,1); n_C={500,1000,1400,1415,1416,1500,2000,3000}; alpha=.01,rho=.5,gamma_A=.95.                                                                      |
| S2     | Target-FPR sensitivity                  | 3 non-primary alpha values x 3 n values x 4 distributions x 10,000         | alpha=.005: n={2860,2861,5722}; alpha=.02: n={693,694,1388}; alpha=.05: n={269,270,540}; rho=.5,gamma_A=.95.                                                                                                               |
| S3     | Temporal-dependence stress              | 4 AR(1) phi x 3 n_C x 10,000                                               | phi={0,.3,.6,.9}; n_C={1416,2000,3000}; marginal N(0,1); evaluate theoretical future marginal exceedance.                                                                                                             |
| S4     | Calibration-to-test shift               | 5 mean shifts x 10,000                                                     | C scores N(0,1), n_C=2000; future benign N(mu,1), mu={0,.10,.25,.50,1.00}.                                                                                                                                                 |
| S5     | Calibration contamination               | 6 rates x 2 directions x 10,000                                            | n_C=2000; contamination q={0,.001,.005,.01,.02,.05}; high-tail N(3,1) and low-tail N(-3,1).                                                                                                                                |
| S6     | Gate-B exact power                      | 5 n_G x 9 true FPR values                                                  | n_G={736,1000,1500,2000,3000}; p={.0025,.005,.0075,.01,.0125,.015,.02,.025,.03}; exact binomial calculation, no Monte Carlo.                                                                                               |
| R1     | N-BaIoT primary                         | 9 natural clients x 5 model seeds x 50 calibration seeds                   | alpha=.01,rho=.5,gamma_A=.95,gamma_B=.95; all mandatory policies.                                                                                                                                                          |
| R2     | Gate-A sample-size sweep                | n_C={500,1000,1400,1415,1416,1500,2000}                                    | n_G=3000 fixed; same frozen scores.                                                                                                                                                                                        |
| R3     | Gate-B sample-size sweep                | n_G={736,1000,1500,2000,3000}                                              | n_C=2000 fixed.                                                                                                                                                                                                            |
| R4     | Operating tolerance sensitivity         | rho={.25,.50,1.00}                                                         | alpha=.01; shows data cost of narrower operational contracts.                                                                                                                                                              |
| R5     | Target-FPR sensitivity                  | alpha={.005,.01,.02,.05}                                                   | rho=.50; Gate A may correctly declare insufficient evidence at alpha=.005 with n_C=2000.                                                                                                                                   |
| R6     | Assurance sensitivity                   | gamma_A={.90,.95,.99}                                                      | gamma_B=.95; primary band.                                                                                                                                                                                                 |
| R7     | Multiplicity sensitivity                | gamma_A=1-.05/9; Gate-B Bonferroni/Holm sensitivity                        | No familywise claim if readiness fails at available n_C.                                                                                                                                                                   |
| R8     | Source-order test segmentation          | 5 equal source-order benign-test blocks per client                         | Report block-wise FPR without re-fitting. Call this temporal drift only when dataset provenance verifies chronological order.                                                                                                                                                                |
| R9     | Real-score calibration contamination    | q={.001,.005,.01,.02,.05}                                                  | Replace q fraction of benign C/G with A_dev scores; detector frozen.                                                                                                                                                       |
| R10    | CIC IoT-DIAD external replication       | All eligible natural clients x 5 model seeds x 20 calibration seeds        | Same alpha/rho/confidence; dataset-specific fixed data counts.                                                                                                                                                             |
| R11    | Second-detector check                   | Federated Deep-SVDD; 3 model seeds x 10 calibration seeds                  | Only B1,B2,B5,FedCRG; N-BaIoT primary operating contract.                                                                                                                                                                  |
| R12    | Calibration-role source-order sensitivity | N-BaIoT + DIAD; fixed source-order roles, no within-reservoir permutation | N-BaIoT: first 500 R, next 3000 G, next 2000 C, final 500 supervised guard. DIAD: first 300 R, next 1500 G, next 1500 C, final 500 supervised guard. Same frozen detectors and final tests; no chronology claim without verified time provenance.                                |
| R13    | Computational/communication overhead    | 100 warm-ups + 1000 measured repetitions per primitive on one CPU thread   | Measure reference construction, cached Gate-A rank lookup + order statistic, Gate B count/interval, and full policy decision; report median/p95 wall time and peak memory. No invented hardware-independent latency claim. |
| R14    | DIAD feature-contract sensitivity       | One training-schema-derived numeric-safe feature representation × 5 model seeds × named calibration seed | Feature list is derived from training schema only by Section 7.3.2; compare FedCRG, GLOBAL-Q99-FULL, LOCAL-Q99-FULL, SHRINKAGE. Exploratory; cannot replace the 86-feature R10 result. |

## 11.1 Randomness registry

| **Purpose**                       | **Locked seed(s)**                         |
|-----------------------------------|--------------------------------------------|
| Primary detector model seeds      | 11, 22, 33, 44, 55                         |
| N-BaIoT calibration-role seeds    | 1000-1049; 1000 is the named primary split |
| DIAD calibration-role seeds       | 2000-2019; 2000 is the named primary split |
| Attack dev/test stratification    | 9001                                       |
| Synthetic Monte Carlo master seed | 123456                                     |
| Optional device-population bootstrap | 424242                                  |
| Deep-SVDD model seeds             | 11, 22, 33                                 |


## 11.2 Confirmatory workload accounting

### N-BaIoT primary AE

- Detector trainings: 5 model seeds = **5 complete FL trainings**.
- Primary calibration-policy evaluations: 5 model seeds × 1 named
  calibration split × 12 policy IDs × 9 clients =
  **540 client-policy cells**.
- Full split-sensitivity evaluations: 5 × 50 × 12 × 9 =
  **27,000 client-policy cells**.
- FedCRG itself yields 5 × 50 × 9 = **2,250 client state decisions**.
- Each calibration seed creates one federation reference threshold per
  model seed, so N-BaIoT R1 contains **250 reference-threshold
  constructions**.

### N-BaIoT Deep-SVDD

- 3 model seeds × 1 training each = **3 FL trainings**.
- 3 × 10 calibration seeds × 5 policies × 9 clients =
  **1,350 client-policy cells**.

### DIAD

Let \(K_D\) be the number of clients satisfying the locked
pre-outcome eligibility rule.

- Detector trainings: **5**.
- Policy cells for the 12 registered policy IDs (including FedCRG):
  \(5\times20\times12\times K_D=1200K_D\).
- Of these, comparator-only cells are
  \(5\times20\times11\times K_D=1100K_D\).
- FedCRG state decisions: \(5\times20\times K_D=100K_D\).
- Reference threshold constructions: \(5\times20=100\).
- R14 adds 5 exploratory DIAD trainings with a DATA-DEPENDENT feature dimension and 5 × 1 × 4 × \(K_D\) = \(20K_D\) policy cells; it is ledgered separately from R10.

These counts are ledger expectations. The run-verification script MUST
reconcile actual artifacts against them before statistics are produced.

## 11.3 Synthetic experiment reproducibility counts

- S1: \(4\times8\times10,000 = 320,000\) Monte-Carlo trials.
- S2: \(3\times3\times4\times10,000 = 360,000\) trials.
- S3: \(4\times3\times10,000 = 120,000\) trials.
- S4: \(5\times10,000 = 50,000\) trials.
- S5: \(6\times2\times10,000 = 120,000\) trials.
- S6 is exact binomial arithmetic: \(5\times9=45\) cells and **zero**
  Monte-Carlo trials.
- Total locked Monte-Carlo trials S1–S5: **970,000**.


# 12. Statistical Analysis Plan

## 12.1 Estimands

The primary estimand is the operating behavior of the **fixed natural
federation** defined by the dataset, not an abstract infinite
population of IoT devices.

For each policy \(p\), model seed \(m\), and calibration split \(c\),
compute the complete K-client federation metrics

\[
\mathrm{MEBE}_{p,m,c},\quad
\mathrm{HighExcess}_{p,m,c},\quad
\mathrm{ABMacroTPR}_{p,m,c}.
\]

Any global threshold used by a policy is constructed once from the
whole federation for that `(dataset,model_seed,calibration_seed)` cell
before per-client metrics are aggregated.

## 12.2 Confirmatory versus sensitivity randomness

- **Model seeds 11,22,33,44,55** represent stochastic model-training
  repetitions.
- **Calibration seed 1000** is the named N-BaIoT confirmatory role
  split; 2000 is the DIAD confirmatory split.
- Remaining calibration seeds quantify split sensitivity only. Because
  they repeatedly repartition the same finite historical reservoir,
  they MUST NOT be described as independent samples from the
  deployment population.
- The primary tables show the named split and all five model seeds.
  Split-sensitivity figures then show the full 50-seed/20-seed
  distributions.

## 12.3 Primary policy contrasts

Exactly four confirmatory contrasts are interpreted first:

1. FedCRG − GLOBAL-Q99-FULL
2. FedCRG − LOCAL-Q99-FULL
3. FedCRG − GATE-A-ONLY
4. FedCRG − SHRINKAGE

For each contrast report absolute difference in MEBE, HighExcess, and
ABMacroTPR. Relative improvement is reported only if the comparator
metric is strictly positive; otherwise write `NA` rather than divide by
zero.

The locked operational non-inferiority margin for **ABMacroTPR** is
**−0.03 absolute** relative to the utility anchor in Section 10.2. A reliability improvement is not described as utility-preserving if the paired difference is below −0.03. The supplement also reports −0.01 and −0.05 margin sensitivities.

## 12.4 Uncertainty without pseudo-replication

### Fixed-federation uncertainty

For the named calibration split:

1. compute each federation-level endpoint for all five paired model seeds;
2. report all five values, mean, standard deviation, median, minimum, and maximum;
3. compute a **paired model-seed procedural bootstrap** with 10,000 replicates and seed 424242 by resampling the five model-seed indices with replacement and preserving policy pairing; report the 2.5th/97.5th percentiles of the paired policy difference;
4. label this interval **training-seed variability conditional on the fixed federation and fixed calibration split**. It is not a population-of-devices confidence interval;
5. do not base the paper on a t-test over five seeds.

### Split-sensitivity uncertainty

For each model seed, summarize the 50 N-BaIoT calibration-role
permutations (20 for DIAD) using median, IQR, 5th/95th percentiles, and
state/admission frequencies. These are sensitivity distributions, not
confidence intervals for an independent-sample population.

### Optional device-population bootstrap

If a device-population CI is reported, use 10,000 paired bootstrap
replicates with seed 424242. In each replicate:

1. resample client IDs with replacement;
2. resample model seeds with replacement;
3. resample calibration seeds with replacement when the sensitivity
   estimand is intended;
4. **recompute every federation-level global/reference threshold from
   the resampled client score sets and multiplicities**;
5. recompute all policy states and endpoints;
6. preserve policy pairing throughout.

It is prohibited to resample already-computed client metrics for a
global-threshold policy because that breaks the dependence induced by
the shared reference threshold.

## 12.5 Client-level FPR binomial reference intervals

For every final benign test count \(n_k\) and observed false-positive
count \(x_k\), report the exact 95% Clopper-Pearson interval using the
same formulas as Gate B with `confidence=0.95`. The observed benchmark
FPR \(x_k/n_k\) is the primary descriptive quantity. The interval is
explicitly labeled a **binomial reference interval conditional on an
i.i.d.-Bernoulli test-record sampling model**. Source-order dependence,
repeated traffic behavior, or capture drift can invalidate its nominal
coverage, so it MUST NOT be presented as a confidence interval for a
future deployment stream unless that sampling assumption is defended.
It is also not a Gate-A calibration guarantee. R8 provides the required
block-wise dependence diagnostic.

## 12.6 Multiplicity and hypothesis-test discipline

There are **no confirmatory null-hypothesis p-values in the primary fixed-federation analysis**. Five model seeds are procedural repetitions, nine N-BaIoT devices are the complete fixed study federation, and global thresholds couple clients. Effect sizes, paired seed-variability intervals, client-level exact FPR intervals, component ablations, external replication, and the pre-registered utility margin carry the evidential burden.

Multiplicity is nevertheless controlled where the protocol itself performs multiple statistical declarations:

- Gate-B fleet sensitivity uses the exact Bonferroni/Holm procedures in Section 6.1.
- If a journal-requested exploratory p-value table is later added, it MUST be labeled exploratory, preserve federation-level pairing, apply Holm across the four primary policy contrasts, and MUST NOT change claim classification.
- Per-client, per-attack, or sensitivity-grid p-values are not used to declare the method successful.

## 12.7 No outcome-conditioned redesign

The confirmatory method, baselines, metrics, and dataset eligibility
rules are not changed because of unfavorable results. In particular,
failure of Gate B to improve the ablation does **not** authorize
removing Gate B and re-analyzing the same confirmatory data as a new
method. Such a simplification may be proposed for future work only.


# 13. Multi-Audit Failure Analysis and Mandatory Fixes

| **Audit**                            | **Failure mode**                                                                         | **Mandatory resolution**                                                                                                         |
|--------------------------------------|------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| A. Novelty                           | Reviewer says local-vs-global selection already exists.                                  | Cite Laridi prominently; claim benign-only finite-sample evidence admission, not the question itself.                            |
| B. One-sided guarantee               | Local threshold only guaranteed \<=alpha, yet low-FPR mismatch triggers personalization. | Fixed: Gate A now guarantees probability of FPR falling inside \[a,b\], aligning both low- and high-mismatch decisions.          |
| C. Selection bias                    | Same benign data used to decide personalization and fit local threshold.                 | Fixed: G and C are disjoint; R is separate again.                                                                                |
| D. Shared fallback                   | Reference mismatch is statistically demonstrated but Gate A fails; method silently calls shared reliable.          | Fixed: CALIBRATION_DEFICIT state; temporary shared threshold is explicitly unresolved.                                           |
| E. Failure-to-reject fallacy         | No Gate-B mismatch interpreted as equivalence.                                           | Fixed wording: NO_MATERIAL_MISMATCH_DEMONSTRATED only.                                                                           |
| F. Data-budget bias                  | FedCRG sees more benign data than baseline.                                              | Fixed: GLOBAL-Q99-FULL and LOCAL-Q99-FULL use the complete benign calibration budget.                                            |
| G. Multiple clients                  | 95% per-client statement misrepresented as federation-wide 95%.                          | Fixed: per-client claim plus Bonferroni familywise sensitivity; exact n_C table supplied.                                        |
| H. Temporal dependence               | IoT packet scores are not IID.                                                           | Source-order holdout; AR(1) stress; five-block real-data analysis; theorem scoped explicitly to i.i.d. continuous benign scores.     |
| I. Distribution shift                | Calibration and future benign distributions drift.                                       | Locked synthetic mean-shift stress and real source-order block analysis; temporal interpretation requires verified chronology.                      |
| J. Score ties                        | Beta order-statistic statement assumes continuity.                                       | Strict \> rule; selected-threshold multiplicity is checked. A multiplicity >1 blocks local admission with `CALIBRATION_ASSUMPTION_VIOLATION`; no jitter repair is allowed. |
| K. Calibration contamination         | Presumed-benign pool contains attacks/noise.                                             | Score-level contamination stress; contamination robustness not claimed.                                                          |
| L. Attack-label leakage              | Attack labels indirectly tune FedCRG.                                                    | FedCRG fit API accepts benign score arrays only; A_dev physically separated; automated leakage tests.                            |
| M. Detector confounding              | Threshold policy retrains or changes score generator.                                    | Immutable score cache; SHA-256 score-array equality across policies; AUROC/AUPRC invariance assertion.                           |
| N. Client-volume dominance           | Large clients dominate global threshold/metrics.                                         | Fixed equal R_k counts, equal train counts, equal-client aggregation, client-macro metrics.                                      |
| O. Privacy overclaim                 | Derived calibration scores can leak information.                                         | No formal privacy claim; explicit communication accounting; future secure-quantile work outside contribution.                    |
| P. External-dataset identity leakage | device_mac/IP/ports leak client identity into model.                                     | device_mac is partition metadata only; fixed 86-feature allowlist excludes direct identifiers and labels.                        |
| Q. Metric gaming                     | F1 tuned after seeing attack outcomes.                                                   | Primary reliability metrics pre-registered; FedCRG sees no attacks; F1 secondary only.                                           |
| R. Pseudo-replication                | 50 calibration seeds counted as independent observations.                                 | Named confirmatory split + sensitivity-only role permutations; no degrees-of-freedom inflation.                                  |
| S. Hyperparameter HARKing            | alpha/rho/confidence selected after results.                                             | Primary values locked now; all sensitivity grids locked now; no result-driven replacement.                                       |
| T. Excess scope                      | Paper adds RL, clustering, new FL optimizer, poisoning defense.                          | Prohibited unless a reviewer later requires a narrowly justified sensitivity; core contribution remains post-training admission. |
| U. Reproducibility                   | Undocumented dataset order, seeds, package drift, stochastic GPU.                        | File hashes, row IDs, config hashes, lockfile, deterministic seeds, cached scores, environment manifest, code release.           |
| V. Negative outcome suppression      | Method only reported where it personalizes.                                              | All clients/states reported, including no-mismatch and deficit; no client cherry-picking.                                        |


| W. Shared-threshold bootstrap | Resampling client metrics after constructing one global threshold breaks federation dependence. | Any client bootstrap recomputes global/reference thresholds inside the replicate. |
| X. Federated preprocessing leakage | “Global median/minmax” silently centralizes benign training data. | DIAD imputation is client-local; global extrema use explicit derived-statistic exchange and privacy accounting. |
| Y. Attack-prevalence gaming | F1 baselines see 10% of huge attack files, producing arbitrary class prevalence and unequal label budgets. | Exactly 500 balanced anomalies + 500 benign guard records per client. |
| Z. Literature attribution | Detector settings could be misattributed or changed on the basis of a secondary extraction. | Primary FedDetect full text was checked: N-BaIoT retains its reported 30 rounds × 120 local epochs; FedCRG-specific LR endpoints/initialization are labeled separately. |
| AA. Outcome-conditioned method mutation | Gate B could be deleted after seeing primary results. | Prohibited. Ablation weakness is reported; confirmatory method remains fixed. |
| AB. Source-order overclaim | CSV row order was called chronological without timestamp proof. | Use “source-order holdout”; only claim chronology when source provenance verifies it. |
| AC. Score-transform sensitivity | Linear threshold shrinkage depends on score scale. | Score definition is frozen; caveat disclosed; optional risk-curve shrinkage cannot replace the locked baseline. |

# 14. Implementation Specification

```text
project/
configs/
protocol_v2.yaml
nbaiot_primary.yaml
diad_external.yaml
synthetic.yaml
fedcrg/
reference.py
gate_a.py
gate_b.py
policy.py
states.py
metrics.py
data/
nbaiot.py
diad.py
manifests.py
splits.py
models/
autoencoder.py
deep_svdd.py
fl/
trainer.py
aggregation.py
lr_schedule.py
experiments/
synthetic_gate_a.py
gate_b_power.py
real_primary.py
sensitivity.py
analysis/
statistics.py
figures.py
tables.py
tests/
test_gate_a_exact.py
test_gate_b_exact.py
test_data_disjointness.py
test_no_label_leakage.py
test_score_invariance.py
test_metrics.py
test_reproducibility.py
artifacts/
manifests/
scores/
thresholds/
metrics/
figures/
```


## 14.1 Required data/artifact schemas

| **Artifact**           | **Required fields**                                                                                                                                         |
|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| dataset_manifest.json  | dataset_id, source_version, file paths, SHA256 per input, parser_version, created_at, feature_names, client IDs, per-role counts                            |
| score_cache.parquet    | dataset_id, client_id, row_id, phase, model_seed, score_float64, label_test_only, attack_family_test_only                                                   |
| threshold_record.jsonl | run_id, policy_id, client_id, tau_ref, tau_local, selected_tau, gate_a_n, gate_a_rank, gate_a_probability, gate_b_n, gate_b_x, CP_L, CP_U, state, tie_count |
| metric_record.jsonl    | run_id, policy_id, client_id, benign_n, attack_n, FP,TN,TP,FN,FPR,TPR,precision,F1,BA,AUROC,AUPRC, band_error                                               |
| run_config.json        | full parameters, seeds, git commit, environment lock hash, data-manifest hash, score-cache hash                                                             |

## 14.2 Normative unit tests - exact expected outputs

| **Component**     | **Input**                                    | **Expected**                                            |
|-------------------|----------------------------------------------|---------------------------------------------------------|
| Gate A            | alpha=.01,rho=.5,gamma=.95,n=1415            | NOT_READY                                               |
| Gate A            | alpha=.01,rho=.5,gamma=.95,n=1416            | READY; r\*=1404; P=0.9500045311 +/-1e-10                |
| Gate A            | n=1500                                       | READY; r\*=1487; P=0.9573928914 +/-1e-10                |
| Gate A            | n=2000                                       | READY; r\*=1982; P=0.9805279151 +/-1e-10                |
| Gate B            | n=736,x=0                                    | LOW_MISMATCH                                            |
| Gate B            | n=736,x=1                                    | NO_MATERIAL_MISMATCH_DEMONSTRATED                       |
| Gate B            | n=1000,x=0                                   | LOW_MISMATCH                                            |
| Gate B            | n=1000,x=1                                   | NO_MATERIAL_MISMATCH_DEMONSTRATED                       |
| Gate B            | n=1000,x=23                                  | NO_MATERIAL_MISMATCH_DEMONSTRATED                       |
| Gate B            | n=1000,x=24                                  | HIGH_MISMATCH                                           |
| Gate B            | n=1500,x=2                                   | LOW_MISMATCH                                            |
| Gate B            | n=1500,x=3                                   | NO_MATERIAL_MISMATCH_DEMONSTRATED                       |
| Gate B            | n=1500,x=32                                  | NO_MATERIAL_MISMATCH_DEMONSTRATED                       |
| Gate B            | n=1500,x=33                                  | HIGH_MISMATCH                                           |
| Gate B            | n=3000,x=7                                   | LOW_MISMATCH                                            |
| Gate B            | n=3000,x=8                                   | NO_MATERIAL_MISMATCH_DEMONSTRATED                       |
| Gate B            | n=3000,x=58                                  | NO_MATERIAL_MISMATCH_DEMONSTRATED                       |
| Gate B            | n=3000,x=59                                  | HIGH_MISMATCH                                           |
| Reference rank    | K=9,R_k=500,alpha=.01                        | N_R=4500;q_ref=4456                                     |
| Classification    | score==threshold                             | BENIGN because rule is score \> threshold               |
| DIAD attack allocator | lexical categories A/B/C; `dmax=[200,200,200]`; budget=500 | `dev=[167,167,166]` |
| DIAD attack allocator | lexical categories A/B/C; `dmax=[0,50,900]`; budget=500 | `dev=[0,50,450]` |
| DIAD attack reserve | any present category `a` | `A_test_count[a] >= min(100,n_ka)` |
| Policy invariance | same score cache, different threshold policy | AUROC difference \<=1e-12 and AUPRC difference \<=1e-12 |

## 14.3 Leakage and integrity tests

- Assert pairwise-empty row_id intersections among T, calibration reservoir, B_test, A_dev, and A_test; within each calibration seed, assert R, G, C, and supervised-comparator benign guard are pairwise disjoint.

- Assert FedCRGPolicy.fit() accepts only arrays of benign scores and scalar protocol parameters; it has no label argument.

- Assert no path containing A_dev or A_test is opened by FedCRG fitting code. Use a test fixture that raises immediately if accessed.

- Assert scaler/imputer objects are fitted exclusively on T_k rows; serialize their fit-row hashes.

- Assert selected feature names exactly match the N-BaIoT 115 schema or the DIAD allowlist after the finite-value audit.

- Assert all policies read the identical score_cache hash for a given dataset/model seed.

- Assert final test labels are loaded only inside evaluation functions after thresholds have been serialized.

- Fail a run if any metric is NaN/inf, any role count differs from the protocol, any client disappears after outcome computation, or any config hash differs from the pre-registered file.

## 14.4 Determinism and environment

- Use one locked environment file (uv.lock, poetry.lock, or fully pinned requirements lock). Freeze it at the first successful protocol test; do not update dependencies mid-study without a new protocol version.

- Record Python, PyTorch, CUDA, cuDNN, NumPy, SciPy, pandas, scikit-learn, OS, CPU, GPU, and driver versions in every run manifest.

- Enable deterministic PyTorch algorithms where supported; set Python, NumPy, and torch RNGs from the model seed; record any nondeterministic CUDA operation that cannot be disabled.

- Compute calibration/order-statistic mathematics and exact binomial intervals in float64. Neural forward passes may be float32; convert final scalar scores to float64 before threshold calculations.

- Persist git commit hash and \`git diff --quiet\` state. Main confirmatory runs require a clean repository or a stored patch hash.

- Every generated table/figure must be reproducible from immutable score/threshold/metric artifacts without retraining the detector.

## 14.5 Computational and communication contract

**Gate-A rank precomputation.** For fixed (n_C,a,b), the maximizing rank r\* and readiness probability depend only on the protocol constants and sample count, not on the observed score values. Precompute and cache this table. At runtime the client only selects the r\*-th order statistic and checks ties; it does not recompute every Beta probability.

| **Primitive**         | **Location**               | **Reference complexity**                                                            | **Communication / implementation rule**                                                                                                                 |
|-----------------------|----------------------------|-------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| Reference threshold   | Server                     | O(N_R log N_R) with full sort; O(N_R) memory in reference implementation            | Each client sends exactly \|R_k\| float64 scores. N-BaIoT: 500 x 8 = 4,000 bytes/client; 36,000 bytes total for 9 clients, excluding protocol overhead. |
| Gate A runtime        | Client                     | O(n_C log n_C) full-sort reference implementation; cached r\* lookup O(1)           | No communication is required for local threshold construction in the core design.                                                                       |
| Gate B                | Client                     | O(n_G) threshold comparisons; O(1) streaming count possible                         | Only the mismatch decision/statistics need be logged; no raw G scores need leave client in a distributed implementation.                                |
| State decision        | Client/server policy layer | O(1)                                                                                | Combines Gate A and Gate B states deterministically.                                                                                                    |
| Possible optimization | Client                     | Expected O(n_C) selection for r\*-th statistic is permitted only after parity tests | Optimization must return the identical threshold/tie state as the full-sort reference implementation.                                                   |

### 14.5.1 Threshold-policy payload accounting

The following payload counts are scalar payload only and exclude message headers, authentication, retries, encryption framing, and model-training communication.

| Policy primitive | N-BaIoT upload payload | DIAD upload payload | Notes |
|---|---:|---:|---|
| FedCRG reference R | 500 float64/client = 4,000 B; 36,000 B federation | 300 float64/client = 2,400 B; \(2,400K_D\) B | one reference construction per model/calibration split |
| FedCRG Gate B | one integer count plus logged interval/state; raw G need not leave client | same | exact serialized bytes implementation-dependent, so benchmark measured payload too |
| FedCRG Gate A | 0 B required for local threshold construction | 0 B | only selected threshold/state need be reported for experiment logging |
| GLOBAL-Q99-FULL / FEDDETECT-3SIGMA naive score upload | 5,500 float64/client = 44,000 B; 396,000 B federation | 3,300 float64/client = 26,400 B; \(26,400K_D\) B | strong comparator reference implementation; secure/distributed quantile optimization is outside scope |
| LARIDI-STYLE-SS moments | 2 classes × (`n` int64, mean float64, variance float64) = 48 B/client | 48 B/client | candidate F1 vectors below dominate |
| LARIDI-STYLE-SS / SUP-F1-1000 candidate evaluation | 1,000 float64 F1 values = 8,000 B/client | 8,000 B/client | thresholds broadcast not included in upload count |

The paper MUST report FedCRG threshold-policy traffic separately from model-training traffic. It MUST NOT claim lower total communication than a comparator unless the same serialization and transport accounting is measured for both.

**R13 benchmark protocol.** Pin the benchmark process to one CPU thread, record CPU model/OS/Python/NumPy/SciPy versions, run 100 warm-ups followed by 1000 timed calls per primitive and input size, and report median and 95th-percentile wall time plus peak resident memory. Runtime values are empirical deployment evidence, not mathematical constants.


## 14.6 Normative Python API

The public package interfaces MUST expose the following semantic
contracts. Names may be wrapped internally, but these functions and
return fields must be testable.

```python
build_reference_threshold(
    reference_scores_by_client: Mapping[str, NDArray[np.float64]],
    alpha: float
) -> ReferenceThresholdResult

gate_a_readiness(
    calibration_scores: NDArray[np.float64],
    alpha: float,
    rho: float,
    gamma_a: float
) -> GateAResult

gate_b_reference_mismatch(
    gate_scores: NDArray[np.float64],
    tau_ref: float,
    alpha: float,
    rho: float,
    gamma_b: float
) -> GateBResult

decide_fedcrg(
    reference: ReferenceThresholdResult,
    gate_a: GateAResult,
    gate_b: GateBResult
) -> FedCRGDecision
```

`GateAResult` MUST include `n`, `rank`, `coverage_probability`,
`ready`, `tau_local`, `tie_count`, `a`, and `b`.

`GateBResult` MUST include `n`, `x`, `fpr_hat`, `cp_lower`,
`cp_upper`, `p_low`, `p_high`, and `mismatch_state`.

`FedCRGDecision` MUST include `state`, `selected_threshold`,
`selected_source`, `tie_count`, and a machine-readable
`reason_code`.

## 14.7 Failure and reason-code registry

| Code | Trigger | Run status |
|---|---|---|
| DATASET_COUNT_MISMATCH | observed source row counts violate locked manifest/feasibility rules | STOP |
| NBAIOT_ATTACK_BUDGET_FAIL | a present N-BaIoT attack subtype cannot retain at least 100 final-test rows after the fixed 500-record comparator-development allocation | STOP / audit acquired dataset |
| DIAD_DEVICE_COUNT_SOURCE_MISMATCH | parsed source does not expose the expected 105 official device identities before eligibility filtering | STOP / audit parser or dataset version |
| ID_INVALID | DIAD device identifier cannot be mapped to a stable client ID | exclude device before outcome analysis |
| FEATURE_MISSING | one or more of the locked 86 DIAD model features are absent for the device/source schema | exclude device before outcome analysis; audit if systemic |
| FINITE_RATE_FAIL | one or more locked DIAD features fails the >=99% finite-value eligibility requirement on the device training rows | exclude device before outcome analysis; audit if systemic |
| BENIGN_COUNT_LT_7800 | DIAD device has fewer than 7,800 usable benign rows | exclude device before outcome analysis |
| MALICIOUS_COUNT_LT_1000 | DIAD device has fewer than 1,000 usable malicious rows | exclude device before outcome analysis |
| ATTACK_DEV_CAPACITY_LT_500 | after reserving `min(100,n_ka)` test rows per present attack category, fewer than 500 malicious rows remain available for development | exclude device before outcome analysis |
| EXTERNAL_DATASET_INSUFFICIENT_CLIENTS | fewer than 10 DIAD clients satisfy the pre-outcome eligibility rule | valid dataset-level outcome; R10 cannot support confirmatory external-replication claim |
| FEATURE_SCHEMA_MISMATCH | model feature list/count differs from locked schema | STOP |
| DIAD_FEATURE_FINITE_RATE_FAIL | a feature-level finite-rate violation is discovered after the eligibility manifest has been frozen, or eligibility/code generation disagrees across reruns | STOP; manifest/parser inconsistency |
| ROLE_OVERLAP | any forbidden row_id intersection exists | STOP |
| LABEL_LEAKAGE | FedCRG code accesses attack label/dev/test path | STOP |
| SCORE_CACHE_HASH_MISMATCH | policies do not consume the identical immutable score cache | STOP |
| GATE_A_NOT_READY | \(P_{r^\*}<\gamma_A\) | valid state |
| GATE_B_INSUFFICIENT | `n_G < n_G_min(a,gamma_B)`; primary value 736 | valid unresolved state |
| CALIBRATION_DEFICIT | mismatch proven but Gate A not ready | valid unresolved state |
| GATE_B_DIRECTION_CONTRADICTION | multiplicity sensitivity reports both low and high mismatch for one valid cell | STOP; implementation/numerical defect |
| CALIBRATION_ASSUMPTION_VIOLATION | selected local order statistic has multiplicity >1 after Gate-B mismatch and Gate-A sample-size readiness | valid unresolved state; use reference fallback; local admission blocked |
| LARIDI_STYLE_UNDEFINED | published-style overlap interval is empty/non-ordered | comparator cell missing-by-definition; report frequency |
| METRIC_UNDEFINED | denominator is zero for a metric | use locked NA rule; never coerce to 0 |
| NONFINITE_SCORE | any required cached anomaly score is NaN or ±inf | STOP affected model-seed run |
| TRAINING_NUMERICAL_FAILURE | non-finite loss/parameter/update or other locked optimizer numerical failure | STOP affected model-seed run; no silent retuning |
| ONE_SIDED_BAND_BY_DESIGN | sensitivity contract has `a=0`, so low-side mismatch is impossible | valid sensitivity annotation; high-side Gate B only |
| DATA_DRIFT_STRESS | robustness experiment intentionally violates calibration/deployment stationarity | valid stress-test annotation, not a primary-policy error |
| NONDETERMINISTIC_PARITY_FAIL | repeated deterministic run produces different artifact hash | STOP confirmatory run |

## 14.8 Metric edge-case rules

- `FPR = FP/(FP+TN)` requires at least one benign final-test record; the
  dataset protocol already enforces this.
- `TPR = TP/(TP+FN)` is `NA` if a client has no malicious final-test
  records; such a client is ineligible for DIAD by design.
- Precision is `NA` when `TP+FP=0`; F1 is `NA` when both precision and
  recall cannot be defined. Do not replace undefined values by zero.
- ABMacroTPR is `NA` if any included client has no defined attack-group TPR; this should be impossible under the eligibility rules and triggers an integrity warning.
- Attack balancing first averages only **present** attack groups within a client, then averages clients. Missing Mirai groups on Ennio/Samsung do not count as zero.
- Ordinary MacroTPR averages client TPRs formed from all malicious rows and is secondary because large attack groups receive more weight.
- AUROC/AUPRC are computed from raw cached scores and labels, never
  from thresholded decisions.
- All threshold comparisons use strict `score > threshold`.
- Threshold equality is benign by definition.

## 14.9 Configuration validation

A run starts only if a JSON-Schema/Pydantic validator confirms:

- `0 < alpha < 1`;
- `0 < gamma_a < 1`, `0 < gamma_b < 1`;
- `0 <= rho`;
- derived \(0\le a < b\le1\);
- all seed lists contain unique integers;
- all role counts are nonnegative and fit the client eligibility rule;
- the policy registry exactly matches the protocol;
- config hash matches the experiment ledger for confirmatory runs.

## 14.10 Command-line execution contract

The reference implementation SHOULD expose these parameterized research
commands, with all confirmatory values read from YAML rather than typed
manually:

```text
fedcrg doctor
fedcrg data prepare --config configs/nbaiot_primary.yaml
fedcrg data prepare --config configs/diad_external.yaml
fedcrg tables precompute-gate-a --config configs/protocol_v2.yaml
fedcrg synthetic run --config configs/synthetic.yaml
fedcrg train --config configs/nbaiot_primary.yaml
fedcrg score --config configs/nbaiot_primary.yaml
fedcrg evaluate --config configs/nbaiot_primary.yaml
fedcrg train --config configs/diad_external.yaml
fedcrg score --config configs/diad_external.yaml
fedcrg evaluate --config configs/diad_external.yaml
fedcrg robustness deep-svdd --config configs/nbaiot_primary.yaml
fedcrg benchmark --config configs/protocol_v2.yaml
fedcrg report build
fedcrg verify
```

`fedcrg verify` MUST fail if any required experiment cell, artifact hash,
unit test, leakage check, or manifest field is missing.


# 15. Gate-B Power and Evidence Interpretation

**Gate B is intentionally conservative.** At n_G=3000, a client whose true reference FPR is mildly outside the 1.5% boundary can still yield an inconclusive result. This is not a bug; the gate is designed to avoid personalization without strong independent evidence.

| **True reference FPR** | **Probability Gate B declares LOW/HIGH mismatch at n_G=3000** |
|------------------------|---------------------------------------------------------------|
| 0.25%                  | 52.45%                                                        |
| 0.50% boundary         | 1.78%                                                         |
| 0.75%                  | 0.013%                                                        |
| 1.00% target           | 0.00021%                                                      |
| 1.25%                  | 0.065%                                                        |
| 1.50% boundary         | 2.49%                                                         |
| 2.00%                  | 56.96%                                                        |
| 2.50%                  | 97.65%                                                        |
| 3.00%                  | 99.98%                                                        |

**Interpretation requirement.** Report Gate-B power curves next to admission results. If a client is not personalized, readers must be able to distinguish "reference appears adequate" from "the gate lacked power to prove a mild mismatch." The state label remains conservative: no mismatch demonstrated.

# 16. Robustness and Assumption-Stress Protocol

## 16.1 Temporal dependence

**AR(1) stress.** Generate `z_t = phi*z_(t-1) + sqrt(1-phi^2)*epsilon_t` for `phi in {0,0.3,0.6,0.9}`. Use marginal N(0,1), `n_C in {1416,2000,3000}`, and 10,000 repetitions per cell. For each threshold, compute its true marginal exceedance under N(0,1). Plot realized in-band coverage versus `phi`. The exact theorem is only claimed for the independent `phi=0` condition.

## 16.2 Distribution shift

**Shift stress.** C_k is sampled from N(0,1), n_C=2000. Future benign distribution is N(mu,1) for mu in {0,0.10,0.25,0.50,1.00}; 10,000 repetitions. This quantifies how rapidly the static contract fails when calibration no longer represents deployment.

## 16.3 Calibration contamination

**Synthetic.** Replace q in {0,0.001,0.005,0.01,0.02,0.05} of C_k with high-tail N(3,1) or low-tail N(-3,1) contamination; 10,000 repetitions. Real-score sensitivity replaces the same proportions of G/C benign scores with A_dev scores. FedCRG is not described as contamination-robust unless a future method explicitly addresses this threat.

## 16.4 Tie/continuity audit

- For every C_k, record unique_score_fraction, total duplicate count, selected-threshold multiplicity, and minimum positive score spacing.

- If selected-threshold multiplicity \>1, return `CALIBRATION_ASSUMPTION_VIOLATION` for the confirmatory policy. Do not inject random jitter or silently perturb scores in the primary analysis.

- A jitter sensitivity may be reported in supplementary material only if needed; any jitter rule must be fixed before use and never substituted silently for primary data.

# 17. Required Tables and Figures

| **Item** | **Title**                         | **Required content**                                                                                                         |
|----------|-----------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| Figure 1 | FedCRG decision architecture      | Reference threshold -\> independent Gate B -\> Gate A readiness/continuity check -\> five deployment states. Show data-role separation R/G/C. |
| Figure 2 | Finite-sample readiness frontier  | n_C vs maximum exact in-band probability. Mark 1416 minimum for 0.5%-1.5% / 95%, 2000 primary, and sensitivity bands.        |
| Figure 3 | Gate-B evidence/power map         | n_G x true reference FPR with mismatch-declaration probability; mark primary n_G=3000.                                       |
| Figure 4 | Per-client operating points       | For each N-BaIoT client, FPR under GLOBAL-Q99-FULL, LOCAL-Q99-FULL, SHRINKAGE, FedCRG; horizontal lines 0.5%,1%,1.5%.        |
| Figure 5 | Reliability-utility frontier      | MEBE vs ABMacroTPR for mandatory policies with paired uncertainty.                                                             |
| Figure 6 | Calibration-size phase transition | n_C sweep showing Gate-A readiness/admission and n_G sweep showing mismatch evidence.                                        |
| Figure 7 | Assumption stress                 | Coverage under AR(1), mean shift, and contamination; separate panels in supplement if main page budget is tight.             |
| Figure 8 | External replication              | DIAD per-client FPR and aggregate MEBE/ABMacroTPR using identical protocol values.                                             |

| **Item** | **Title**              | **Required content**                                                                                               |
|----------|------------------------|--------------------------------------------------------------------------------------------------------------------|
| Table 1  | Literature boundary    | Explicitly compare information used, threshold object, local/global decision, finite-sample contract, IoT setting. |
| Table 2  | Protocol constants     | Every locked alpha/rho/confidence/count/seed/hyperparameter.                                                       |
| Table 3  | Dataset inventory      | All client IDs, benign/attack counts, exact role counts, feature dimensions, file hashes.                          |
| Table 4  | Primary policy results | MEBE, HighExcess, band-violation rate, MAFE, ABMacroTPR, MacroTPR, worst-client TPR, F1 secondary.                             |
| Table 5  | Admission states       | Per client Gate-B x/CI, Gate-A n/rank/probability, state, tau_ref, tau_local, selected tau.                        |
| Table 6  | Ablations              | Gate-A-only, Gate-B-only, full benign-policy-budget baselines, shrinkage.                                                          |
| Table 7  | Sensitivity            | alpha, rho, gamma_A, sample sizes, multiplicity.                                                                   |
| Table 8  | External replication   | Same primary metrics on DIAD.                                                                                      |

# 18. Hostile Reviewer Matrix

| **Reviewer attack**                                                         | **Required response / design defense**                                                                                                                                                                                     |
|-----------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| "Laridi already decides local vs federated."                                | Correct. Cite it as closest prior. Their selector is attack-aware/F1-based and learned; FedCRG is benign-only, independent-evidence admission with a finite-sample operating-band readiness contract.                      |
| "Fed-DTCN already uses client-specific benign thresholds."                  | Correct. A client-specific threshold is not novel. FedCRG asks when client-specific deployment is statistically admitted versus retaining a reference threshold.                                                           |
| "This is just Wilks/tolerance intervals."                                   | The order-statistic theorem is classical and is cited as such. Contribution is the federated admission protocol, independence structure, operating-band alignment, and IoT evidence.                                       |
| "FCP/FedCal already solve federated calibration."                           | They address conformal uncertainty / probability calibration. FedCRG addresses anomaly operating-threshold deployment under a different target and information regime.                                                     |
| "Sun et al. already guarantee anomaly FPR."                                 | Sun et al. target online adaptive thresholding and distribution shifts. FedCRG is a static client-specific personalization-admission policy in heterogeneous federation; do not compete on adaptive threshold theory.      |
| "You waste data by splitting R/G/C."                                        | Yes; independence is intentional. Full benign-policy-budget shared/local baselines quantify the cost. Cross-fitting is not added because combining data-dependent admission and threshold estimation would complicate the clean guarantee. |
| "Your 0.5%-1.5% band is arbitrary."                                         | It is a pre-registered operational tolerance, not a theorem. Sensitivity rho={.25,.5,1.0} is mandatory and exact sample-size costs are reported.                                                                           |
| "95% per client is not 95% for the fleet."                                  | Explicitly agreed; familywise sensitivity and exact sample-size requirements are shown; no simultaneous primary claim.                                                                                                     |
| "IoT traffic is dependent."                                                 | Exact theorem is scoped to i.i.d. continuous benign scores; source-order holdouts, AR(1), shift stress, and block-wise real-data analysis quantify violations. Temporal claims require verified chronology.                                                                                      |
| "No mismatch means your shared threshold is safe."                          | Not claimed. State name says no material mismatch demonstrated, not equivalent/certified.                                                                                                                                  |
| "If the reference is statistically demonstrated out-of-band and local is unready, your system has no certified replacement." | Correct; state is CALIBRATION_DEFICIT. The method explicitly identifies insufficient evidence rather than hiding it.                                                                                                       |
| "FedCRG gets more calibration data than baselines."                         | Full benign-policy-budget shared/local comparators receive all R+G+C benign policy-calibration samples and can therefore have a data advantage.                                                                                                         |
| "Attack labels secretly define the useful operating point."                 | alpha/rho are locked before attack analysis. FedCRG receives only benign scores. Attack labels are evaluation-only; only B7-B9 receive A_dev and they are explicitly labeled extra-information comparators.                |
| "The gain is just a better detector."                                       | All policies use identical cached scores; AUROC/AUPRC invariance and score hashes are automated tests.                                                                                                                     |
| "This is post-processing, not FL."                                          | The paper says so. It is a federated decision-layer protocol applied to heterogeneous FL anomaly scores, not an FL optimizer.                                                                                              |
| "Your calibration score sharing is not private."                            | Agreed; no formal privacy claim. Raw traffic remains local but derived scores can leak; secure quantile computation is outside scope.                                                                                      |
| "N-BaIoT is old/easy."                                                      | It is used for natural nine-device controlled evidence; CIC IoT-DIAD 2024 provides independent modern external validation.                                                                                                 |
| "Your second dataset fabricates clients."                                   | No. device_mac is used only to define natural DIAD clients and is excluded from model features.                                                                                                                            |
| "Fifty calibration seeds inflate significance."                             | The named split is confirmatory; the remaining role permutations are split-sensitivity runs, not independent devices. Any optional client-population bootstrap must recompute global thresholds inside each replicate.                                                                                                                |
| "What if FedCRG rarely personalizes?"                                       | Report it. Admission rate is an outcome, not a target. The protocol is allowed to conclude that reference thresholds are usually adequate or evidence is insufficient.                                                     |

# 19. Claim-Strength, Integrity, and Stop Gates

These gates determine **what may be claimed**, not whether valid results
are hidden. The study is reported even when FedCRG underperforms.

| Gate | Required evidence | Consequence |
|---|---|---|
| G0 Novelty recheck | Repeat the targeted search within 7 calendar days before submission. | If a closer method appears, narrow/reframe novelty; do not omit the work. |
| G1 Statistical-core integrity | All exact Gate A/B/reference tests pass and every S1 cell satisfies the H1 Monte-Carlo-versus-exact agreement tolerance. | Failure = code/audit bug; real-data confirmatory analysis blocked until fixed. |
| G2 Data integrity | All schema/hash/disjointness/leakage checks pass. | Failure = affected run invalid; fix data adapter without inspecting outcomes. |
| G3 Reliability claim | FedCRG has lower MEBE than at least one strong benign-only full benign-policy-budget comparator on N-BaIoT, with no >3 pp ABMacroTPR loss versus the locked utility anchor. | If met, claim operational reliability benefit. If not, claim only characterization/conditional utility. |
| G4 Two-gate contribution | Gate B changes decisions and the full method improves MEBE or BandViolationRate relative to GATE-A-ONLY on at least one natural-client dataset. | If not met, state that Gate-B incremental utility was unsupported; do not retroactively redesign confirmatory method. |
| G5 External replication | DIAD directionally reproduces a primary reliability finding without >3 pp ABMacroTPR loss versus the locked utility anchor. | If met, claim cross-dataset evidence. If not, explicitly report non-replication and scope generality. |
| G6 Detector robustness | Deep-SVDD shows qualitatively consistent admission/reliability behavior. | If not, scope empirical conclusions to AE reconstruction-error scores. |
| G7 Assumption honesty | Temporal/dependence/drift/tie/contamination results are all reported. | Any omitted locked stress test blocks the final claim package. |
| G8 Reproducibility | Clean checkout reproduces all threshold/metric/figure artifacts from immutable score caches. | Failure blocks release/submission artifact claim. |

### 19.1 Claim levels

- **Level A — method benefit:** G1, G2, G3, G4, G5, G6, G7, G8 pass.
- **Level B — dataset-limited benefit:** G1, G2, G3, G4, G7, G8 pass,
  but external/detector replication is mixed. Claims are explicitly
  scoped.
- **Level C — characterization result:** statistical/data integrity
  passes but FedCRG does not outperform strong comparators. The paper
  may still report when/why readiness or mismatch evidence fails.
- **Invalid:** G1, G2, or G8 fails. This is an implementation/data
  problem, not a scientific negative result.


# 20. Publication Plan

## 20.1 Working manuscript identity

**Canonical manuscript title:** FedCRG: Evidence-Admitted Calibration Readiness for Client-Specific Thresholding in Federated IoT Anomaly Detection

**Method name:** FedCRG — Federated Calibration Readiness Gate.

**GitHub repository:** fedcrg

**Python package / method ID:** fedcrg

**One-sentence thesis:** Client-specific anomaly thresholds should be deployed only when independent benign evidence shows the federation reference operating point is materially inappropriate and the client possesses enough local evidence to construct a statistically defensible replacement.

## 20.2 Three contribution statements only

1.  Formulate client-specific anomaly-threshold deployment as an evidence-admission problem rather than assuming universal personalization.

2.  Introduce a benign-only two-gate protocol coupling independent reference-mismatch evidence with a finite-sample local operating-band readiness construction, including explicit unresolved states when evidence is insufficient.

3.  Empirically characterize the reliability-utility tradeoff across natural IoT clients, calibration budgets, target FPRs, confidence levels, temporal dependence, drift/contamination stress, two datasets, and a second score generator.

## 20.3 Target venue and manuscript budget

**Primary target: IEEE Internet of Things Journal.** As of the 12 August 2026 recheck, the journal scope includes IoT systems, applications, testbeds, enabling technologies, and IoT security-adjacent work. Its author guidelines require IEEE double-column format, a 150-250 word abstract, and mandatory page charges for published pages beyond the first eight. The working budget is therefore **<=8 published pages including references** where scientifically feasible; exhaustive protocol tables, proofs, audit ledgers, and sensitivity results belong in supplementary material. A longer paper is permitted when necessary, but it is explicitly treated as an overlength-cost decision rather than assumed to be free.

**Backup target: IEEE Transactions on Network and Service Management.** TNSM permits 10 pages without mandatory overlength charges and up to 16 pages with overlength charges; its scope supports reliability, management functions, applications, and performance/scalability analysis. This is a suitable backup if the paper is framed more strongly as reliability management of distributed IDS operating points.

## 20.4 Planned manuscript structure

| **Section** | **Content**                      | **Target two-column length** |
|-------------|----------------------------------|------------------------------|
| I           | Introduction + exact gap         | 0.7 pages                    |
| II          | Related work / novelty boundary  | 0.7                          |
| III         | Problem and FedCRG               | 1.3                          |
| IV          | Finite-sample analysis           | 0.8                          |
| V           | Experimental protocol            | 0.9                          |
| VI          | Primary results                  | 1.2                          |
| VII         | Robustness / external validation | 0.8                          |
| VIII        | Limitations                      | 0.3                          |
| IX          | Conclusion                       | 0.2                          |
| References  |                                  | target <=1.1 pages; compress prose before deleting necessary citations |

## 20.5 Reproducibility package required at submission

- Public code repository with tagged release matching the manuscript; no uncommitted main-run code.

- Protocol YAML/JSON, environment lockfile, all seeds, exact dataset acquisition instructions, input SHA-256 manifests, and deterministic split-generation code.

- No redistribution of dataset files unless licenses explicitly allow it; publish row-ID/hash manifests and preprocessing scripts instead.

- Immutable score caches if redistribution is permitted; otherwise publish threshold/metric artifacts and scripts that regenerate scores from official datasets.

- All figure/table-generation scripts; no manually edited numerical values in manuscript tables.

- Archive release with persistent DOI (for example Zenodo) at acceptance or submission if journal policy permits.

- Pre-submission novelty-search log dated within seven days of submission.

# 21. Implementation Sequence and Completion Criteria

| **Phase**                    | **Work**                                                                                                      | **Exit criterion**                                                          |
|------------------------------|---------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| P0 Protocol freeze           | Commit protocol_v2.yaml and this specification; create reference bibliography; lock primary parameter values. | No outcome data inspected.                                                  |
| P1 Statistical core          | Implement Gate A/B/reference/state machine and exact unit tests.                                              | All tests in Section 14.2 pass.                                             |
| P2 Synthetic validation      | Run S1-S6.                                                                                                    | S1 coverage acceptance met; exact power tables generated.                   |
| P3 Data adapters             | Implement N-BaIoT and DIAD manifests/splits/features/scaling.                                                 | All integrity/leakage assertions pass; dataset inventory table frozen.      |
| P4 Detector training         | Train five AE model seeds; cache all required scores.                                                         | All model checkpoints and score hashes frozen; no threshold policy run yet. |
| P5 Primary policy evaluation | Run R1-R9 and N-BaIoT component of R12 on cached N-BaIoT scores.                                              | Complete artifact ledger; no missing client/seed/policy cell.               |
| P6 External replication      | Run DIAD R10 plus the DIAD component of R12 unchanged.                                                        | Eligibility rule and all included clients documented before outcomes.       |
| P7 Second detector           | Run R11 regardless of primary direction; run R13 overhead benchmark before submission.                         | Qualitative robustness assessed without outcome-conditioned selection.       |
| P8 Statistical analysis      | Run fixed-federation primary summaries, split-sensitivity analysis, and any pre-specified optional federation-preserving bootstrap.                                                      | Tables 4-7 and main figures generated from scripts.                         |
| P9 Hostile self-review       | Re-run audit matrix against actual findings; remove claims unsupported by results.                            | Every reviewer attack has either evidence or explicit limitation.           |
| P10 Submission freeze        | Repeat literature search, generate manuscript, archive code/config/results.                                   | Submission package and supplement reproduce from clean checkout.            |

# 22. Claim Discipline and Limitations to State Explicitly

- The decision unit is one anomaly-score record / feature vector. A 1% record-level FPR is not claimed to equal one alert per 100 security incidents, per hour, or per event after alarm merging.

- Gate-A exactness relies on i.i.d. continuous benign scores from the same distribution as future benign operation; dependence or shift can invalidate the nominal contract.

- Gate-B evidence is finite-sample and conservative; mild true mismatches may remain undetected. Final-test Clopper-Pearson intervals likewise require an i.i.d.-Bernoulli test-record interpretation and are reported only as reference intervals when real source-order dependence is plausible.

- The selected 0.5%-1.5% band is an operational design choice, not a universally optimal IoT false-alarm tolerance.

- FedCRG assumes access to a trusted/presumed-benign calibration stream. Contamination is evaluated but not formally defended.

- The core reference construction shares derived anomaly scores and has no formal privacy guarantee.

- The primary theorem is per client; familywise fleet-wide assurance requires more local calibration data than the main N-BaIoT configuration supplies.

- FedCRG changes thresholds, not ranking quality. It cannot repair a detector whose anomaly scores fail to separate attacks from benign data.

- External validity is limited to the datasets/score generators evaluated; results cannot be generalized to all IoT protocols or concept-drift regimes.

- CALIBRATION_DEFICIT, GATE_B_INSUFFICIENT, and CALIBRATION_ASSUMPTION_VIOLATION are legitimate unresolved states. The method does not fabricate certainty when evidence is unavailable.

> **Final publication position.** If the implementation and experiments support the hypotheses, the strongest defensible paper is not "a new threshold algorithm." It is a statistically governed deployment protocol that distinguishes the right to personalize an anomaly operating point from the mere ability to compute one, and it quantifies when that right is supported, unsupported, or unresolved under heterogeneous federated IoT data.


# Appendix A. Exact Primary Constants

| **Constant**                                | **Value**      |
|---------------------------------------------|----------------|
| alpha                                       | 0.01           |
| rho                                         | 0.5            |
| a                                           | 0.005          |
| b                                           | 0.015          |
| gamma_A                                     | 0.95           |
| gamma_B                                     | 0.95           |
| Gate-A minimum n_C                          | 1416           |
| Gate-A primary N-BaIoT n_C                  | 2000           |
| Gate-A N-BaIoT rank r\*                     | 1982           |
| Gate-A N-BaIoT exact P_r                    | 0.9805279151   |
| Gate-A DIAD n_C                             | 1500           |
| Gate-A DIAD rank r\*                        | 1487           |
| Gate-A DIAD exact P_r                       | 0.9573928914   |
| Gate-B minimum n_G                          | 736            |
| Gate-B N-BaIoT n_G                          | 3000           |
| Gate-B N-BaIoT low cutoff                   | x\<=7          |
| Gate-B N-BaIoT high cutoff                  | x\>=59         |
| Gate-B DIAD n_G                             | 1500           |
| Gate-B DIAD low cutoff                      | x\<=2          |
| Gate-B DIAD high cutoff                     | x\>=33         |
| N-BaIoT R per client                        | 500            |
| N-BaIoT reference N_R                       | 4500           |
| N-BaIoT q_ref                               | 4456           |
| N-BaIoT train per client                    | 4000           |
| N-BaIoT calibration reservoir per client    | 6000           |
| N-BaIoT minimum final benign test assertion | 3000           |
| DIAD train per eligible client              | 2000           |
| DIAD calibration reservoir                  | 3800           |
| DIAD minimum final benign test              | 2000           |
| DIAD minimum final malicious test            | 500 overall; every present category reserved by Section 7.2.3 |
| DIAD benign eligibility minimum             | 7800           |
| DIAD malicious eligibility minimum          | 1000 total, plus development-capacity reserve rule |
| Model seeds                                 | 11,22,33,44,55 |
| Calibration seeds N-BaIoT                   | 1000-1049      |
| Calibration seeds DIAD                      | 2000-2019      |
| Bootstrap replicates                        | 10000          |
| Bootstrap seed                              | 424242         |


## Appendix A.1 Additional derived constants

| Derived item | Value |
|---|---:|
| N-BaIoT full local benign calibration budget \(R+G+C\) | 5,500/client |
| N-BaIoT LOCAL-Q99-FULL rank | 5,446 of 5,500 |
| N-BaIoT GLOBAL-Q99-FULL pooled N | 49,500 |
| N-BaIoT GLOBAL-Q99-FULL rank | 49,006 of 49,500 |
| N-BaIoT AE parameter count | 36,626 |
| N-BaIoT AE full-model float32 bytes | 146,504 |
| N-BaIoT 30-round tensor communication | 79,112,160 bytes |
| DIAD AE parameter count | 20,473 |
| DIAD AE full-model float32 bytes | 81,892 |
| N-BaIoT Gate-A induced mean FPR at n=2000,r=1982 | 0.0094952524 |
| Gate-B x=0 CP upper at n=735 | 0.0050063101 |
| Gate-B x=0 CP upper at n=736 | 0.0049995250 |


# Appendix B. Run-ID and Configuration Contract

```text
run_id =
{dataset}__{detector}__ms{model_seed}__cs{cal_seed}__
a{alpha_ppm}__r{rho_bp}__ga{gammaA_bp}__gb{gammaB_bp}__{policy}
Example:
nbaiot__ae__ms11__cs1000__a10000__r5000__ga9500__gb9500__fedcrg
alpha_ppm = round(alpha * 1,000,000)
rho_bp = round(rho * 10,000)
gamma_bp = round(gamma * 10,000)
Every run_id maps to exactly one immutable run_config.json SHA-256 hash.
```



# Appendix C. Literature and Source References

**\[1\]** S. S. Wilks, "Determination of Sample Sizes for Setting Tolerance Limits," Annals of Mathematical Statistics, 12(1), 91-96, 1941. DOI: 10.1214/aoms/1177731788. [<u>Source</u>](https://doi.org/10.1214/aoms/1177731788)

**\[2\]** Y. Meidan et al., "N-BaIoT: Network-based Detection of IoT Botnet Attacks Using Deep Autoencoders," 2018; canonical dataset: UCI Machine Learning Repository, DOI 10.24432/C5RC8J. [<u>Source</u>](https://archive.ics.uci.edu/dataset/442/detection%2Bof%2Biot%2Bbotnet%2Battacks%2Bn%2Bbaiot)

**\[3\]** T. Zhang et al., "Federated Learning for Internet of Things: A Federated Learning Framework for On-device Anomaly Data Detection," arXiv:2106.07976, 2021. [<u>Source</u>](https://arxiv.org/abs/2106.07976)

**\[4\]** C. Lu et al., "Federated Conformal Predictors for Distributed Uncertainty Quantification," ICML 2023, PMLR 202:22942-22964. [<u>Source</u>](https://proceedings.mlr.press/v202/lu23i.html)

**\[5\]** H. Peng et al., "FedCal: Achieving Local and Global Calibration in Federated Learning via Aggregated Parameterized Scaler," ICML 2024, PMLR 235:40331-40346. [<u>Source</u>](https://proceedings.mlr.press/v235/peng24g.html)

**\[6\]** S. H. Sun, A. Sankararaman, B. M. Narayanaswamy, "Online Adaptive Anomaly Thresholding with Confidence Sequences," ICML 2024, PMLR 235:47105-47132. [<u>Source</u>](https://proceedings.mlr.press/v235/sun24h.html)

**\[7\]** S. Laridi, G. Palmer, K.-M. M. Tam, "Enhanced federated anomaly detection through autoencoders using summary statistics-based thresholding," Scientific Reports 14, 26704, 2024. DOI: 10.1038/s41598-024-76961-2. [<u>Source</u>](https://doi.org/10.1038/s41598-024-76961-2)

**\[8\]** D. Chen et al., "Self-Aware Personalized Federated Learning," NeurIPS 2022. [<u>Source</u>](https://papers.nips.cc/paper_files/paper/2022/hash/8265d7bb2db42e86637001db2c46619f-Abstract-Conference.html)

**\[9\]** S. Sun et al., "Robust intrusion detection based on personalized federated learning for IoT environment," Computers & Security 154, 104442, 2025. DOI: 10.1016/j.cose.2025.104442. [<u>Source</u>](https://doi.org/10.1016/j.cose.2025.104442)

**\[10\]** H. Yan et al., "Global or Local Adaptation? Client-Sampled Federated Meta-Learning for Personalized IoT Intrusion Detection," IEEE Transactions on Information Forensics and Security, vol. 20, pp. 279-293, 2025, DOI: 10.1109/TIFS.2024.3516548. [<u>Source</u>](https://openreview.net/forum?id=qvHLtiDl19)

**\[11\]** M. A. Khan et al., "Fed-DTCN: A Federated Disentangled Learning Framework for Unsupervised Zero-Day Anomaly Detection in IoT with Semantic-Aware Augmentation," Sensors 26(6), 1918, 2026. DOI: 10.3390/s26061918. [<u>Source</u>](https://doi.org/10.3390/s26061918)

**\[12\]** S. Izadi, M. Ahmadi, "CF-HFC: Calibrated Federated based Hardware-aware Fuzzy Clustering for Intrusion Detection in Heterogeneous IoTs," arXiv:2602.12557, 2026. [<u>Source</u>](https://arxiv.org/abs/2602.12557)

**\[13\]** N. F. Shahid, "When Calibration Fails the Vulnerable Hospital: Federated Conformal Risk Control via Risk-Curve Shrinkage," arXiv:2606.20115, 2026. [<u>Source</u>](https://arxiv.org/abs/2606.20115)

**\[14\]** A. K. Bui et al., "FBID: Adaptive Personalized Federated Learning for Robust Out-of-Distribution Attack Detection in IoT Networks," arXiv:2608.04073, 2026. [<u>Source</u>](https://arxiv.org/abs/2608.04073)

**\[15\]** J. Robalino-Diaz et al., "Structural impact of non-IID heterogeneity on federated behavioral anomaly detection in IoT and IoMT systems," Frontiers in Artificial Intelligence 9:1825067, 2026. DOI: 10.3389/frai.2026.1825067. [<u>Source</u>](https://doi.org/10.3389/frai.2026.1825067)

**\[16\]** P. Shi et al., "What the Detector Can See: Evaluating CPS Anomaly Detectors Independently of the Decision Rule," arXiv:2608.02821, 2026. [<u>Source</u>](https://arxiv.org/abs/2608.02821)

**[16a]** A. Alqazzaz, "SecuFL-IoT: an adaptive privacy-preserving federated learning framework for anomaly detection in smart industrial networks," *Scientific Reports*, vol. 16, art. 4107, 2026. DOI: 10.1038/s41598-025-11883-1. This adjacent work includes reinforcement-learning-based threshold adjustment; it further prohibits any broad “adaptive thresholding in federated IoT” novelty claim. [<u>Source</u>](https://doi.org/10.1038/s41598-025-11883-1)

**[16b]** F. Ailabouni, J.-Á. Román-Gallego, M.-L. Pérez-Delgado, "Federated transformer autoencoder with reinforcement learning–tuned thresholding in IoT and network," *Internet of Things*, vol. 38, 101988, 2026. DOI: 10.1016/j.iot.2026.101988. The threshold is tuned by reinforcement learning against validation F1 and then frozen for test; FedCRG must distinguish itself by benign-only evidence admission and a finite-sample operating contract, not by adaptive threshold selection. [<u>Source</u>](https://doi.org/10.1016/j.iot.2026.101988)

**[16c]** M. N. Alatawi, "Secure adaptive federated learning for scalable anomaly detection in industrial IoT networks," *Scientific Reports*, 2026. DOI: 10.1038/s41598-026-37819-x. Adjacent adaptive federated anomaly-detection work; relevant to scope and deployment positioning, not a direct Gate-A/Gate-B predecessor. [<u>Source</u>](https://doi.org/10.1038/s41598-026-37819-x)


**\[17\]** Canadian Institute for Cybersecurity, "CIC IoT-DIAD 2024 dataset: A dual-function dataset for IoT device identification and anomaly detection," official dataset page; associated IEEE Internet of Things Journal paper, 2024. [<u>Source</u>](https://www.unb.ca/cic/datasets/iot-diad-2024.html)

**\[18\]** IEEE Internet of Things Journal, "Guidelines for Authors," accessed 12 August 2026. [<u>Source</u>](https://ieee-iotj.org/guidelines-for-authors/)

**\[19\]** IEEE Transactions on Network and Service Management, "Policies and Guidelines," accessed 12 August 2026. [<u>Source</u>](https://www.comsoc.org/publications/journals/ieee-tnsm/policies-guidelines)

# Appendix D. Literature Search Recheck Protocol for Submission Week

- Run exact phrase and semantic searches for: "federated anomaly threshold local global", "client-specific threshold federated anomaly detection", "benign-only federated calibration threshold", "finite-sample anomaly FPR federated", "personalization admission federated threshold", "IoT federated conformal threshold", and the method name "Calibration Readiness Gate".

- Search at minimum: IEEE Xplore/official IEEE pages, ACM Digital Library, Springer/Nature, Elsevier/ScienceDirect, PMLR/OpenReview, arXiv, and Google Scholar as an index. Prefer the primary publisher/preprint source for verification.

- Use publication/event date, not only search ranking. Record all papers from 2024 through the submission date that could invalidate the novelty claim.

- Update the related-work matrix and safe novelty wording before submission. If a closer work appears, change the claim; never hide it.

# Appendix E. Normative Configuration Skeleton

```yaml
protocol:
  id: fedcrg
  version: "2.0"
  alpha: 0.01
  rho: 0.50
  gate_a_assurance: 0.95
  gate_b_confidence: 0.95
  strict_threshold_operator: ">"
  gate_b_min_mode: derived_from_a_gamma_b
  primary_gate_b_min_n_expected: 736

training:
  model: autoencoder
  rounds: 30
  local_epochs_nbaiot: 120
  local_epochs_diad: 20
  batch_size: 64
  optimizer: adam
  lr_initial: 0.001
  lr_final: 0.00001
  betas: [0.9, 0.999]
  eps: 1.0e-8
  weight_decay: 0.0
  client_fraction: 1.0
  aggregation: equal_client_mean
  early_stopping: false
  mixed_precision: false

deep_svdd:
  rounds: 30
  local_epochs: 20
  batch_size: 64
  encoder: [115, 64, 32]
  activation: tanh
  bias: false
  optimizer: adam
  lr_initial: 0.001
  lr_final: 0.00001
  center_mode: equal_mean_of_client_initial_embeddings

nbaiot:
  clients: 9
  train_benign_per_client: 4000
  reservoir_benign_per_client: 6000
  reference_per_client: 500
  gate_per_client: 3000
  local_calibration_per_client: 2000
  comparator_benign_guard_per_client: 500
  min_final_benign_per_client: 3000
  attack_dev_per_client: 500
  min_attack_test_rows_per_present_subtype: 100
  primary_calibration_seed: 1000
  calibration_seeds: [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009,
                      1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019,
                      1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029,
                      1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039,
                      1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049]

diad:
  min_benign_rows: 7800
  min_malicious_rows: 1000
  min_final_attack_rows: 500
  min_attack_test_rows_per_present_category: 100
  min_clients: 10
  train_benign_per_client: 2000
  reservoir_benign_per_client: 3800
  reference_per_client: 300
  gate_per_client: 1500
  local_calibration_per_client: 1500
  comparator_benign_guard_per_client: 500
  min_final_benign_per_client: 2000
  attack_dev_per_client: 500
  primary_calibration_seed: 2000
  calibration_seed_start: 2000
  calibration_seed_end_inclusive: 2019

randomness:
  model_seeds: [11, 22, 33, 44, 55]
  attack_dev_seed: 9001
  synthetic_master_seed: 123456
  bootstrap_seed: 424242

policies:
  - REF-Q99-R
  - GLOBAL-Q99-FULL
  - LOCAL-Q99-FULL
  - GATE-A-ONLY
  - GATE-B-ONLY
  - SHRINKAGE
  - FEDDETECT-3SIGMA
  - DEV-F1-LG-SELECT
  - LARIDI-STYLE-SS
  - SUP-F1-1000
  - ORACLE-TEST
  - FEDCRG
```

**Important:** the policy registry contains 12 IDs including FedCRG itself.
Tables that count “11 policies” refer to the **11 comparator policies**
when FedCRG is discussed separately. Evaluation-ledger code MUST define
whether a count includes FedCRG and MUST never use an ambiguous
`num_policies` constant.

# Appendix F. Implementation Completion Checklist

- [ ] `protocol_v2.yaml` hash frozen.
- [ ] N-BaIoT UCI acquisition and SHA-256 manifest complete.
- [ ] Nine N-BaIoT clients and 115 features verified.
- [ ] N-BaIoT row counts counted from source and cross-check table reconciled.
- [ ] DIAD official acquisition and packet-schema manifest complete.
- [ ] DIAD 86-feature allowlist exact-match test passes.
- [ ] DIAD 86-feature selection rationale and R14 numeric-safe sensitivity manifest frozen before outcomes.
- [ ] DIAD client eligibility list frozen before outcome evaluation.
- [ ] All R/G/C/guard/test row-ID disjointness tests pass.
- [ ] Gate-A precomputation table regenerated from formulas and independently high-precision spot-checked.
- [ ] Gate-A repeated-sample interpretation text matches implementation; no conditional/posterior probability wording remains.
- [ ] Gate-B cutoff table regenerated from exact CP implementation.
- [ ] Dynamic `n_G_min(a,gamma_B)` and one-sided-band tests pass.
- [ ] Bonferroni and Holm directional Gate-B sensitivity tests pass.
- [ ] Exact unit tests pass to locked tolerances.
- [ ] Synthetic S1–S6 complete.
- [ ] Five N-BaIoT AE model seeds trained and frozen.
- [ ] N-BaIoT score-cache hashes frozen before policy evaluation.
- [ ] Primary calibration seed 1000 evaluated before sensitivity summaries.
- [ ] Remaining 49 calibration splits evaluated and labeled sensitivity.
- [ ] All comparator information-regime tests pass.
- [ ] DIAD five model seeds trained after eligibility freeze.
- [ ] DIAD score caches frozen before threshold policy evaluation.
- [ ] Deep-SVDD robustness experiment complete regardless of primary direction.
- [ ] Temporal/dependence/drift/contamination/tie stresses complete.
- [ ] R13 runtime/communication benchmark complete.
- [ ] No result table contains manually typed numerical values.
- [ ] `fedcrg verify` passes from a clean checkout.
- [ ] Submission-week novelty search log complete.
- [ ] Claims reduced to the highest level actually supported by results.

# Appendix G. Independent Numerical and Internal-Consistency Audit Ledger

This appendix records values regenerated independently from the normative formulas during the v2.0 hostile audit. It is a verification ledger, not an additional scientific result.

## G.1 Gate-A regenerated values

Using

\[
P_r=I_b(n+1-r,r)-I_a(n+1-r,r),\qquad
r^*=\arg\max_r P_r
\]

with larger-\(r\) tie breaking:

| Contract | Regenerated result |
|---|---|
| \(\alpha=.01,\rho=.5,\gamma_A=.95,n=1415\) | best \(r=1403\), \(P=0.9499884311\), NOT_READY |
| same, \(n=1416\) | \(r^*=1404\), \(P=0.9500045311\), READY |
| same, \(n=1500\) | \(r^*=1487\), \(P=0.9573928914\) |
| same, \(n=2000\) | \(r^*=1982\), \(P=0.9805279151\) |
| \(\gamma_A=.90\) minimum | \(n=1000,r^*=991,P=0.9001415746\) |
| \(\gamma_A=.99\) minimum | \(n=2435,r^*=2413,P=0.9900229803\) |
| \(\alpha=.005,\rho=.5,\gamma_A=.95\) minimum | \(n=2861\) |
| \(\alpha=.02,\rho=.5,\gamma_A=.95\) minimum | \(n=694\) |
| \(\alpha=.05,\rho=.5,\gamma_A=.95\) minimum | \(n=270\) |
| \(\alpha=.01,\rho=.25,\gamma_A=.95\) minimum | \(n=5970\) |
| \(\alpha=.01,\rho=1.0,\gamma_A=.95\) minimum | \(n=149\) |

## G.2 Gate-B regenerated cutoffs

For two-sided 95% Clopper-Pearson intervals and the primary band \\([0.005,0.015]\\):

| \(n_G\) | LOW iff | HIGH iff |
|---:|---:|---:|
| 736 | \(x=0\) | \(x\ge19\) |
| 1000 | \(x=0\) | \(x\ge24\) |
| 1500 | \(x\le2\) | \(x\ge33\) |
| 2000 | \(x\le3\) | \(x\ge42\) |
| 3000 | \(x\le7\) | \(x\ge59\) |

Boundary check: \\(U(0,735)=0.0050063101\\) and \\(U(0,736)=0.0049995250\\), establishing 736 as the primary bidirectional minimum.

## G.3 Gate-B exact power at the primary \(n_G=3000\)

| True reference FPR | Mismatch-declaration probability |
|---:|---:|
| 0.25% | 52.4547% |
| 0.50% | 1.7795% |
| 0.75% | 0.0133% |
| 1.00% | 0.0002109% |
| 1.25% | 0.06525% |
| 1.50% | 2.4889% |
| 2.00% | 56.9626% |
| 2.50% | 97.6504% |
| 3.00% | 99.9827% |

## G.4 Architecture and communication arithmetic

- N-BaIoT AE dimensions `115-86-57-38-29-38-57-86-115` yield **36,626** biased-linear trainable parameters, not the 36,628 printed in the FedDetect paper.
- N-BaIoT model tensor: \\(36,626\times4=146,504\\) bytes float32.
- N-BaIoT model exchange over 9 clients and 30 rounds: \\(2\times9\times30\times146,504=79,112,160\\) bytes.
- DIAD 86-dimensional AE yields **20,473** parameters and \\(20,473\times4=81,892\\) bytes/model.
- At the theoretical maximum 105 DIAD clients, 30-round full tensor exchange is **515,919,600 bytes**.
- N-BaIoT local optimizer steps/client: \\(\\lceil4000/64\\rceil\times120\times30=226,800\\).
- DIAD local optimizer steps/client: \\(\\lceil2000/64\\rceil\times20\times30=19,200\\).

## G.5 Audit outcome classification

| Audit dimension | v2.0 disposition |
|---|---|
| Mathematical constants | independently regenerated; locked tests supplied |
| Gate-A interpretation | tightened to pre-data repeated-sample coverage; no posterior interpretation |
| Sampling assumptions | tightened from generic exchangeability to the i.i.d.-continuous / i.i.d.-Bernoulli conditions actually used by the exact laws |
| Gate-B minimum | made parameter-dependent; 736 retained only for the primary contract |
| Multiple clients | exact Bonferroni Gate-A and Gate-B sensitivities plus Holm directional sensitivity specified |
| Baseline data fairness | `FULL` defined exactly as R+G+C; independent supervised guard protected from reuse |
| Attack imbalance | ABMacroTPR promoted to primary utility; comparator development prevalence fixed 50:50 |
| Prior-work reproduction | Laridi comparator downgraded from “exact reproduction” to fully specified style comparator unless source/code parity is established |
| FedDetect provenance | 30 rounds/120 epochs/batch64/tanh verified; protocol deviations explicitly listed |
| Data-order claim | source order distinguished from verified chronology |
| DIAD feature choice | confirmatory rationale plus pre-outcome numeric-safe sensitivity added |
| Privacy | score/extrema/model-derived communication explicitly non-private unless a formal mechanism is added |
| Outcome-conditioned redesign | prohibited |

**Audit conclusion.** No remaining numerical constant in the primary Gate-A/Gate-B contract is accepted solely because it appeared in the preliminary roadmap: the readiness cutoffs, ranks, Clopper-Pearson boundaries, reference ranks, parameter counts, optimizer-step counts, and tensor byte counts have explicit formulas and regression-test targets in this protocol.

