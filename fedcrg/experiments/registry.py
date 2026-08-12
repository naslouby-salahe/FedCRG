"""FedCRG Experiment Registry.

Implements the complete experiment registry per Section 11 of the FedCRG Roadmap v2.0.
All experiments are registered here with their locked parameters and metadata.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ExperimentType(Enum):
    """Classification of experiments by type."""
    SYNTHETIC = "synthetic"
    REAL_DATA = "real_data"
    SENSITIVITY = "sensitivity"
    ROBUSTNESS = "robustness"
    BENCHMARK = "benchmark"


class ExperimentID(Enum):
    """Complete enumeration of all FedCRG experiments per Section 11.
    
    Synthetic (S1-S6):
    - S1: IID Gate-A theorem validation
    - S2: Target-FPR sensitivity
    - S3: Temporal-dependence stress
    - S4: Calibration-to-test shift
    - S5: Calibration contamination
    - S6: Gate-B exact power
    
    Real Data (R1-R14):
    - R1: N-BaIoT primary
    - R2: Gate-A sample-size sweep
    - R3: Gate-B sample-size sweep
    - R4: Operating tolerance sensitivity
    - R5: Target-FPR sensitivity
    - R6: Assurance sensitivity
    - R7: Multiplicity sensitivity
    - R8: Source-order test segmentation
    - R9: Real-score calibration contamination
    - R10: CIC IoT-DIAD external replication
    - R11: Second-detector check (Deep-SVDD)
    - R12: Calibration-role source-order sensitivity
    - R13: Computational/communication overhead
    - R14: DIAD feature-contract sensitivity
    """
    # Synthetic experiments
    S1_GATE_A_THEOREM = "S1"
    S2_TARGET_FPR_SENSITIVITY = "S2"
    S3_TEMPORAL_DEPENDENCE = "S3"
    S4_CALIBRATION_SHIFT = "S4"
    S5_CONTAMINATION = "S5"
    S6_GATE_B_POWER = "S6"
    
    # Real data experiments - N-BaIoT
    R1_PRIMARY = "R1"
    R2_GATE_A_SWEEP = "R2"
    R3_GATE_B_SWEEP = "R3"
    R4_TOLERANCE_SENSITIVITY = "R4"
    R5_TARGET_FPR_SENSITIVITY = "R5"
    R6_ASSURANCE_SENSITIVITY = "R6"
    R7_MULTIPLICITY_SENSITIVITY = "R7"
    R8_SOURCE_ORDER = "R8"
    R9_REAL_CONTAMINATION = "R9"
    
    # Real data experiments - External
    R10_DIAD_REPLICATION = "R10"
    R11_SECOND_DETECTOR = "R11"
    R12_SOURCE_ORDER_ROLES = "R12"
    R13_COMPUTATIONAL_BENCHMARK = "R13"
    R14_DIAD_FEATURE_SENSITIVITY = "R14"


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for a single experiment.
    
    Attributes:
        experiment_id: Unique experiment identifier (S1, S2, ..., R14)
        experiment_type: Classification of the experiment
        name: Human-readable name
        description: Detailed description per roadmap
        scale: Scale description (e.g., "4 distributions x 8 n_C values x 10,000 repetitions")
        locked_details: Exact parameters and settings per roadmap
        dataset: Dataset used (None for synthetic)
        policies: List of policy IDs to evaluate
        model_seeds: Model seeds to use
        calibration_seeds: Calibration split seeds to use
        depends_on: List of experiment IDs that must complete first
        artifacts: List of artifact types produced
        metrics: List of metric types computed
        is_confirmatory: Whether this is a confirmatory experiment
    """
    experiment_id: ExperimentID
    experiment_type: ExperimentType
    name: str
    description: str
    scale: str
    locked_details: str
    dataset: Optional[str] = None
    policies: List[str] = field(default_factory=list)
    model_seeds: List[int] = field(default_factory=list)
    calibration_seeds: List[int] = field(default_factory=list)
    depends_on: List[ExperimentID] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    is_confirmatory: bool = False
    
    def __post_init__(self):
        """Validate the configuration."""
        if not self.experiment_id.value:
            raise ValueError("experiment_id cannot be empty")
        if not self.name:
            raise ValueError("name cannot be empty")
    
    @property
    def id_str(self) -> str:
        """Return the string representation of the experiment ID."""
        return self.experiment_id.value
    
    def compute_hash(self) -> str:
        """Compute a deterministic hash of the experiment configuration."""
        config_dict = {
            "experiment_id": self.experiment_id.value,
            "name": self.name,
            "description": self.description,
            "scale": self.scale,
            "locked_details": self.locked_details,
            "dataset": self.dataset,
            "policies": sorted(self.policies),
            "model_seeds": sorted(self.model_seeds),
            "calibration_seeds": sorted(self.calibration_seeds),
            "artifacts": sorted(self.artifacts),
            "metrics": sorted(self.metrics),
            "is_confirmatory": self.is_confirmatory,
        }
        config_str = str(sorted(config_dict.items()))
        return hashlib.sha256(config_str.encode('utf-8')).hexdigest()[:16]


class ExperimentRegistry:
    """Central registry of all FedCRG experiments.
    
    This class maintains the complete experiment registry per Section 11,
    including all synthetic (S1-S6) and real data (R1-R14) experiments.
    """
    
    # All policy IDs per Section 9 and Section 5
    ALL_POLICIES = [
        "B0_REF_Q99_R",
        "B1_GLOBAL_Q99_FULL",
        "B2_LOCAL_Q99_FULL",
        "B3_GATE_A_ONLY",
        "B4_GATE_B_ONLY",
        "B5_SHRINKAGE",
        "B6_FEDDETECT_3SIGMA",
        "B7_DEV_F1_LG_SELECT",
        "B8_LARIDI_STYLE_SS",
        "B9_SUP_F1_1000",
        "B10_ORACLE_TEST",
        "FEDCRG",
    ]
    
    # Primary reliability metrics
    PRIMARY_METRICS = [
        "MEBE",
        "HighExcess",
        "BandViolationRate",
        "MAFE",
        "ABMacroTPR",
        "MacroTPR",
        "AUROC",
        "AUPRC",
    ]
    
    # Secondary metrics
    SECONDARY_METRICS = [
        "FPR",
        "TPR",
        "Precision",
        "Recall",
        "F1",
        "WorstClientTPR",
        "worst_client_ABTPR",
    ]
    
    def __init__(self):
        """Initialize the experiment registry."""
        self._experiments: Dict[str, ExperimentConfig] = {}
        self._initialize_registry()
    
    def _initialize_registry(self):
        """Initialize all experiments from Section 11."""
        # S1: IID Gate-A theorem validation
        self._experiments["S1"] = ExperimentConfig(
            experiment_id=ExperimentID.S1_GATE_A_THEOREM,
            experiment_type=ExperimentType.SYNTHETIC,
            name="IID Gate-A Theorem Validation",
            description=(
                "Validate Gate-A exact probabilities under i.i.d. continuous benign scores. "
                "Tests that Monte-Carlo coverage agrees with exact Beta/CDF calculations."
            ),
            scale="4 distributions x 8 n_C values x 10,000 repetitions",
            locked_details=(
                "Normal(0,1), LogNormal(0,1), Gamma(shape=2,scale=1), "
                "0.9N(0,1)+0.1N(3,1); n_C={500,1000,1400,1415,1416,1500,2000,3000}; "
                "alpha=.01, rho=.5, gamma_A=.95"
            ),
            dataset=None,
            policies=["FEDCRG"],
            model_seeds=[],
            calibration_seeds=[],
            depends_on=[],
            artifacts=["gate_a_coverage", "probability_tables"],
            metrics=["coverage_error", "probability_accuracy"],
            is_confirmatory=True,
        )
        
        # S2: Target-FPR sensitivity
        self._experiments["S2"] = ExperimentConfig(
            experiment_id=ExperimentID.S2_TARGET_FPR_SENSITIVITY,
            experiment_type=ExperimentType.SENSITIVITY,
            name="Target-FPR Sensitivity",
            description=(
                "Sensitivity analysis for different target FPR values. "
                "Tests Gate-A minimum n_C and readiness across alpha values."
            ),
            scale="3 alpha values x 3 n values x 4 distributions x 10,000",
            locked_details=(
                "alpha=.005: n={2860,2861,5722}; alpha=.02: n={693,694,1388}; "
                "alpha=.05: n={269,270,540}; rho=.5, gamma_A=.95"
            ),
            dataset=None,
            policies=["FEDCRG"],
            model_seeds=[],
            calibration_seeds=[],
            depends_on=[],
            artifacts=["min_n_tables"],
            metrics=["min_n_C", "readiness_rate"],
            is_confirmatory=False,
        )
        
        # S3: Temporal-dependence stress
        self._experiments["S3"] = ExperimentConfig(
            experiment_id=ExperimentID.S3_TEMPORAL_DEPENDENCE,
            experiment_type=ExperimentType.ROBUSTNESS,
            name="Temporal Dependence Stress",
            description=(
                "AR(1) stress test for Gate-A coverage under temporal dependence. "
                "Quantifies degradation of i.i.d. contract under autocorrelation."
            ),
            scale="4 AR(1) phi x 3 n_C x 10,000",
            locked_details=(
                "phi={0,.3,.6,.9}; n_C={1416,2000,3000}; marginal N(0,1); "
                "evaluate theoretical future marginal exceedance"
            ),
            dataset=None,
            policies=["FEDCRG"],
            model_seeds=[],
            calibration_seeds=[],
            depends_on=[],
            artifacts=["coverage_vs_phi", "exceedance_distribution"],
            metrics=["coverage_probability", "marginal_exceedance"],
            is_confirmatory=False,
        )
        
        # S4: Calibration-to-test shift
        self._experiments["S4"] = ExperimentConfig(
            experiment_id=ExperimentID.S4_CALIBRATION_SHIFT,
            experiment_type=ExperimentType.ROBUSTNESS,
            name="Calibration-to-Test Shift",
            description=(
                "Distribution shift stress: calibration from N(0,1), "
                "test from N(mu,1) for various mu values."
            ),
            scale="5 mean shifts x 10,000",
            locked_details=(
                "C scores N(0,1), n_C=2000; future benign N(mu,1), "
                "mu={0,.10,.25,.50,1.00}"
            ),
            dataset=None,
            policies=["FEDCRG"],
            model_seeds=[],
            calibration_seeds=[],
            depends_on=[],
            artifacts=["coverage_vs_shift"],
            metrics=["in_band_probability", "fpr_distribution"],
            is_confirmatory=False,
        )
        
        # S5: Calibration contamination
        self._experiments["S5"] = ExperimentConfig(
            experiment_id=ExperimentID.S5_CONTAMINATION,
            experiment_type=ExperimentType.ROBUSTNESS,
            name="Calibration Contamination",
            description=(
                "Synthetic contamination stress: replace fraction of calibration "
                "scores with high/low-tail contamination."
            ),
            scale="6 rates x 2 directions x 10,000",
            locked_details=(
                "n_C=2000; contamination q={0,.001,.005,.01,.02,.05}; "
                "high-tail N(3,1) and low-tail N(-3,1)"
            ),
            dataset=None,
            policies=["FEDCRG"],
            model_seeds=[],
            calibration_seeds=[],
            depends_on=[],
            artifacts=["contamination_effect"],
            metrics=["assumption_violation_rate", "coverage_degradation"],
            is_confirmatory=False,
        )
        
        # S6: Gate-B exact power
        self._experiments["S6"] = ExperimentConfig(
            experiment_id=ExperimentID.S6_GATE_B_POWER,
            experiment_type=ExperimentType.SYNTHETIC,
            name="Gate-B Exact Power",
            description=(
                "Exact binomial power calculation for Gate-B mismatch detection. "
                "No Monte Carlo - uses exact Clopper-Pearson formulas."
            ),
            scale="5 n_G x 9 true FPR values",
            locked_details=(
                "n_G={736,1000,1500,2000,3000}; "
                "p={.0025,.005,.0075,.01,.0125,.015,.02,.025,.03}"
            ),
            dataset=None,
            policies=["FEDCRG"],
            model_seeds=[],
            calibration_seeds=[],
            depends_on=[],
            artifacts=["power_curves", "boundary_table"],
            metrics=["low_mismatch_prob", "high_mismatch_prob", "none_prob"],
            is_confirmatory=True,
        )
        
        # R1: N-BaIoT primary
        self._experiments["R1"] = ExperimentConfig(
            experiment_id=ExperimentID.R1_PRIMARY,
            experiment_type=ExperimentType.REAL_DATA,
            name="N-BaIoT Primary",
            description=(
                "Primary N-BaIoT experiment with all mandatory policies. "
                "Confirmatory evaluation of FedCRG on natural device clients."
            ),
            scale="9 natural clients x 5 model seeds x 50 calibration seeds",
            locked_details=(
                "alpha=.01, rho=.5, gamma_A=.95, gamma_B=.95; "
                "all mandatory policies B0-B10 + FedCRG"
            ),
            dataset="N-BaIoT",
            policies=self.ALL_POLICIES + ["FEDCRG"],
            model_seeds=[11, 22, 33, 44, 55],
            calibration_seeds=list(range(1000, 1050)),
            depends_on=[],
            artifacts=[
                "score_cache",
                "threshold_records",
                "metric_records",
                "state_decisions",
                "run_manifest",
            ],
            metrics=self.PRIMARY_METRICS + self.SECONDARY_METRICS,
            is_confirmatory=True,
        )
        
        # R2: Gate-A sample-size sweep
        self._experiments["R2"] = ExperimentConfig(
            experiment_id=ExperimentID.R2_GATE_A_SWEEP,
            experiment_type=ExperimentType.SENSITIVITY,
            name="Gate-A Sample-Size Sweep",
            description=(
                "N-BaIoT with varying calibration sample sizes. "
                "Shows how admission states change with n_C."
            ),
            scale="n_C={500,1000,1400,1415,1416,1500,2000}",
            locked_details=(
                "n_G=3000 fixed; same frozen scores across n_C values; "
                "alpha=.01, rho=.5, gamma_A=.95, gamma_B=.95"
            ),
            dataset="N-BaIoT",
            policies=["FEDCRG", "B3_GATE_A_ONLY"],
            model_seeds=[11],
            calibration_seeds=[1000],
            depends_on=[ExperimentID.R1_PRIMARY],
            artifacts=["admission_states", "readiness_curves"],
            metrics=["Gate_A_ready_rate", "admission_rate", "deficit_rate"],
            is_confirmatory=False,
        )
        
        # R3: Gate-B sample-size sweep
        self._experiments["R3"] = ExperimentConfig(
            experiment_id=ExperimentID.R3_GATE_B_SWEEP,
            experiment_type=ExperimentType.SENSITIVITY,
            name="Gate-B Sample-Size Sweep",
            description=(
                "N-BaIoT with varying gate sample sizes. "
                "Shows mismatch detection power vs n_G."
            ),
            scale="n_G={736,1000,1500,2000,3000}",
            locked_details=(
                "n_C=2000 fixed; same frozen scores across n_G values; "
                "alpha=.01, rho=.5, gamma_A=.95, gamma_B=.95"
            ),
            dataset="N-BaIoT",
            policies=["FEDCRG"],
            model_seeds=[11],
            calibration_seeds=[1000],
            depends_on=[ExperimentID.R1_PRIMARY],
            artifacts=["mismatch_states", "mismatch_curves"],
            metrics=["LOW_mismatch_rate", "HIGH_mismatch_rate", "none_rate"],
            is_confirmatory=False,
        )
        
        # R4: Operating tolerance sensitivity
        self._experiments["R4"] = ExperimentConfig(
            experiment_id=ExperimentID.R4_TOLERANCE_SENSITIVITY,
            experiment_type=ExperimentType.SENSITIVITY,
            name="Operating Tolerance Sensitivity",
            description=(
                "N-BaIoT with different rho values (band widths). "
                "Shows data cost of narrower operational contracts."
            ),
            scale="rho={.25,.50,1.00}",
            locked_details=(
                "alpha=.01; shows data cost of narrower operational contracts; "
                "n_C=2000, n_G=3000"
            ),
            dataset="N-BaIoT",
            policies=["FEDCRG"],
            model_seeds=[11],
            calibration_seeds=[1000],
            depends_on=[ExperimentID.R1_PRIMARY],
            artifacts=["tolerance_comparison"],
            metrics=["MEBE", "BandViolationRate", "assumption_violation_rate"],
            is_confirmatory=False,
        )
        
        # R5: Target-FPR sensitivity
        self._experiments["R5"] = ExperimentConfig(
            experiment_id=ExperimentID.R5_TARGET_FPR_SENSITIVITY,
            experiment_type=ExperimentType.SENSITIVITY,
            name="Target-FPR Sensitivity",
            description=(
                "N-BaIoT with different alpha (target FPR) values. "
                "Note: Gate A may declare insufficient evidence at alpha=.005 with n_C=2000."
            ),
            scale="alpha={.005,.01,.02,.05}",
            locked_details=(
                "rho=.50; Gate A may correctly declare insufficient evidence at alpha=.005 with n_C=2000; "
                "n_C=2000, n_G=3000"
            ),
            dataset="N-BaIoT",
            policies=["FEDCRG"],
            model_seeds=[11],
            calibration_seeds=[1000],
            depends_on=[ExperimentID.R1_PRIMARY],
            artifacts=["alpha_sensitivity"],
            metrics=["MEBE", "readiness_rate", "admission_rate"],
            is_confirmatory=False,
        )
        
        # R6: Assurance sensitivity
        self._experiments["R6"] = ExperimentConfig(
            experiment_id=ExperimentID.R6_ASSURANCE_SENSITIVITY,
            experiment_type=ExperimentType.SENSITIVITY,
            name="Assurance Sensitivity",
            description=(
                "N-BaIoT with different gamma_A (Gate-A assurance) values."
            ),
            scale="gamma_A={.90,.95,.99}",
            locked_details=(
                "gamma_B=.95; primary band; n_C=2000, n_G=3000"
            ),
            dataset="N-BaIoT",
            policies=["FEDCRG"],
            model_seeds=[11],
            calibration_seeds=[1000],
            depends_on=[ExperimentID.R1_PRIMARY],
            artifacts=["assurance_sensitivity"],
            metrics=["Gate_A_ready_rate", "admission_rate"],
            is_confirmatory=False,
        )
        
        # R7: Multiplicity sensitivity
        self._experiments["R7"] = ExperimentConfig(
            experiment_id=ExperimentID.R7_MULTIPLICITY_SENSITIVITY,
            experiment_type=ExperimentType.SENSITIVITY,
            name="Multiplicity Sensitivity",
            description=(
                "N-BaIoT with Bonferroni/Holm multiplicity adjustments for Gate B. "
                "No familywise claim if readiness fails at available n_C."
            ),
            scale="gamma_A=1-.05/9; Gate-B Bonferroni/Holm sensitivity",
            locked_details=(
                "With K=9, n_G=3000, per-client confidence 0.994444 yields "
                "LOW_MISMATCH for x<=5 and HIGH_MISMATCH for x>=65. "
                "Primary unadjusted cutoffs remain x<=7 and x>=59."
            ),
            dataset="N-BaIoT",
            policies=["FEDCRG"],
            model_seeds=[11],
            calibration_seeds=[1000],
            depends_on=[ExperimentID.R1_PRIMARY],
            artifacts=["multiplicity_table"],
            metrics=["mismatch_survival_rate", "GATE_B_DIRECTION_CONTRADICTION_check"],
            is_confirmatory=False,
        )
        
        # R8: Source-order test segmentation
        self._experiments["R8"] = ExperimentConfig(
            experiment_id=ExperimentID.R8_SOURCE_ORDER,
            experiment_type=ExperimentType.ROBUSTNESS,
            name="Source-Order Test Segmentation",
            description=(
                "N-BaIoT with 5 equal source-order benign-test blocks per client. "
                "Report block-wise FPR without re-fitting."
            ),
            scale="5 equal source-order benign-test blocks per client",
            locked_details=(
                "Report block-wise FPR without re-fitting. "
                "Call this temporal drift only when dataset provenance verifies chronological order."
            ),
            dataset="N-BaIoT",
            policies=self.ALL_POLICIES + ["FEDCRG"],
            model_seeds=[11],
            calibration_seeds=[1000],
            depends_on=[ExperimentID.R1_PRIMARY],
            artifacts=["block_fpr", "drift_diagnostics"],
            metrics=["block_wise_FPR", "FPR_IQR"],
            is_confirmatory=False,
        )
        
        # R9: Real-score calibration contamination
        self._experiments["R9"] = ExperimentConfig(
            experiment_id=ExperimentID.R9_REAL_CONTAMINATION,
            experiment_type=ExperimentType.ROBUSTNESS,
            name="Real-Score Calibration Contamination",
            description=(
                "N-BaIoT with real-score contamination: replace fraction of benign C/G "
                "with A_dev scores; detector frozen."
            ),
            scale="q={.001,.005,.01,.02,.05}",
            locked_details=(
                "Replace q fraction of benign C/G with A_dev scores; detector frozen"
            ),
            dataset="N-BaIoT",
            policies=["FEDCRG"],
            model_seeds=[11],
            calibration_seeds=[1000],
            depends_on=[ExperimentID.R1_PRIMARY],
            artifacts=["contamination_effect_real"],
            metrics=["coverage_degradation", "assumption_violation_rate"],
            is_confirmatory=False,
        )
        
        # R10: CIC IoT-DIAD external replication
        self._experiments["R10"] = ExperimentConfig(
            experiment_id=ExperimentID.R10_DIAD_REPLICATION,
            experiment_type=ExperimentType.REAL_DATA,
            name="CIC IoT-DIAD External Replication",
            description=(
                "External validation on CIC IoT-DIAD 2024 with eligible natural device clients. "
                "Same alpha/rho/confidence as N-BaIoT primary."
            ),
            scale="All eligible natural clients x 5 model seeds x 20 calibration seeds",
            locked_details=(
                "Same alpha/rho/confidence; dataset-specific fixed data counts; "
                "86-feature allowlist; eligibility locked before outcome analysis"
            ),
            dataset="CIC IoT-DIAD 2024",
            policies=self.ALL_POLICIES + ["FEDCRG"],
            model_seeds=[11, 22, 33, 44, 55],
            calibration_seeds=list(range(2000, 2020)),
            depends_on=[],
            artifacts=[
                "score_cache",
                "threshold_records",
                "metric_records",
                "state_decisions",
                "run_manifest",
            ],
            metrics=self.PRIMARY_METRICS + self.SECONDARY_METRICS,
            is_confirmatory=True,
        )
        
        # R11: Second-detector check (Deep-SVDD)
        self._experiments["R11"] = ExperimentConfig(
            experiment_id=ExperimentID.R11_SECOND_DETECTOR,
            experiment_type=ExperimentType.ROBUSTNESS,
            name="Second-Detector Check",
            description=(
                "Federated Deep-SVDD as second detector for robustness check. "
                "Only B1, B2, B5, FedCRG policies."
            ),
            scale="Federated Deep-SVDD; 3 model seeds x 10 calibration seeds",
            locked_details=(
                "Encoder: 115-64-32, tanh, biases disabled; embedding dim=32; "
                "center initialized from model seed, frozen; 30 rounds x 20 local epochs; "
                "Only B1, B2, B5, FedCRG"
            ),
            dataset="N-BaIoT",
            policies=["B1_GLOBAL_Q99_FULL", "B2_LOCAL_Q99_FULL", "B5_SHRINKAGE", "FEDCRG"],
            model_seeds=[11, 22, 33],
            calibration_seeds=list(range(1000, 1010)),
            depends_on=[ExperimentID.R1_PRIMARY],
            artifacts=["deep_svdd_scores", "deep_svdd_metrics"],
            metrics=self.PRIMARY_METRICS + self.SECONDARY_METRICS,
            is_confirmatory=False,
        )
        
        # R12: Calibration-role source-order sensitivity
        self._experiments["R12"] = ExperimentConfig(
            experiment_id=ExperimentID.R12_SOURCE_ORDER_ROLES,
            experiment_type=ExperimentType.SENSITIVITY,
            name="Calibration-Role Source-Order Sensitivity",
            description=(
                "N-BaIoT + DIAD with fixed source-order roles, no within-reservoir permutation. "
                "Same frozen detectors and final tests."
            ),
            scale="N-BaIoT + DIAD; fixed source-order roles",
            locked_details=(
                "N-BaIoT: first 500 R, next 3000 G, next 2000 C, final 500 supervised guard. "
                "DIAD: first 300 R, next 1500 G, next 1500 C, final 500 supervised guard. "
                "Same frozen detectors and final tests; no chronology claim without verified time provenance."
            ),
            dataset="N-BaIoT, CIC IoT-DIAD 2024",
            policies=["FEDCRG", "B1_GLOBAL_Q99_FULL", "B2_LOCAL_Q99_FULL", "B5_SHRINKAGE"],
            model_seeds=[11],
            calibration_seeds=[1000, 2000],
            depends_on=[ExperimentID.R1_PRIMARY, ExperimentID.R10_DIAD_REPLICATION],
            artifacts=["source_order_comparison"],
            metrics=["MEBE", "ABMacroTPR", "state_admission_rate"],
            is_confirmatory=False,
        )
        
        # R13: Computational/communication overhead
        self._experiments["R13"] = ExperimentConfig(
            experiment_id=ExperimentID.R13_COMPUTATIONAL_BENCHMARK,
            experiment_type=ExperimentType.BENCHMARK,
            name="Computational/Communication Overhead",
            description=(
                "Benchmark wall time and memory for threshold-policy primitives. "
                "100 warm-ups + 1000 measured repetitions per primitive on one CPU thread."
            ),
            scale="100 warm-ups + 1000 measured repetitions per primitive",
            locked_details=(
                "Measure reference construction, cached Gate-A rank lookup + order statistic, "
                "Gate B count/interval, and full policy decision; report median/p95 wall time and peak memory"
            ),
            dataset=None,
            policies=["FEDCRG", "B1_GLOBAL_Q99_FULL", "B2_LOCAL_Q99_FULL"],
            model_seeds=[],
            calibration_seeds=[],
            depends_on=[],
            artifacts=["benchmark_results", "memory_profiles"],
            metrics=["median_wall_time", "p95_wall_time", "peak_memory"],
            is_confirmatory=False,
        )
        
        # R14: DIAD feature-contract sensitivity
        self._experiments["R14"] = ExperimentConfig(
            experiment_id=ExperimentID.R14_DIAD_FEATURE_SENSITIVITY,
            experiment_type=ExperimentType.SENSITIVITY,
            name="DIAD Feature-Contract Sensitivity",
            description=(
                "DIAD with training-schema-only numeric-safe feature representation. "
                "Exploratory; cannot replace the 86-feature R10 result."
            ),
            scale="One training-schema-derived numeric-safe feature representation x 5 model seeds x named calibration seed",
            locked_details=(
                "Feature list derived from training schema only by Section 7.3.2; "
                "compare FedCRG, GLOBAL-Q99-FULL, LOCAL-Q99-FULL, SHRINKAGE; "
                "architecture: d -> floor(0.75d) -> floor(0.50d) -> floor(d/3) -> floor(0.25d) -> floor(d/3) -> floor(0.50d) -> floor(0.75d) -> d"
            ),
            dataset="CIC IoT-DIAD 2024",
            policies=["FEDCRG", "B1_GLOBAL_Q99_FULL", "B2_LOCAL_Q99_FULL", "B5_SHRINKAGE"],
            model_seeds=[11, 22, 33, 44, 55],
            calibration_seeds=[2000],
            depends_on=[ExperimentID.R10_DIAD_REPLICATION],
            artifacts=["feature_sensitivity_scores", "feature_sensitivity_metrics"],
            metrics=self.PRIMARY_METRICS + self.SECONDARY_METRICS,
            is_confirmatory=False,
        )
    
    def get(self, experiment_id: str) -> ExperimentConfig:
        """Get experiment configuration by ID string."""
        if experiment_id not in self._experiments:
            raise ValueError(f"Unknown experiment ID: {experiment_id}")
        return self._experiments[experiment_id]
    
    def get_by_enum(self, experiment_id: ExperimentID) -> ExperimentConfig:
        """Get experiment configuration by ExperimentID enum."""
        return self.get(experiment_id.value)
    
    def list_all(self) -> List[str]:
        """List all registered experiment IDs."""
        return sorted(self._experiments.keys())
    
    def list_synthetic(self) -> List[str]:
        """List all synthetic experiment IDs."""
        return sorted([
            exp_id for exp_id, config in self._experiments.items()
            if config.experiment_type == ExperimentType.SYNTHETIC
        ])
    
    def list_real_data(self) -> List[str]:
        """List all real data experiment IDs."""
        return sorted([
            exp_id for exp_id, config in self._experiments.items()
            if config.experiment_type == ExperimentType.REAL_DATA
        ])
    
    def list_confirmatory(self) -> List[str]:
        """List all confirmatory experiment IDs."""
        return sorted([
            exp_id for exp_id, config in self._experiments.items()
            if config.is_confirmatory
        ])
    
    def get_dependencies(self, experiment_id: str) -> List[str]:
        """Get list of experiment IDs that must complete before the given experiment."""
        config = self.get(experiment_id)
        return [dep.value for dep in config.depends_on]
    
    def validate_registry(self) -> Dict[str, Any]:
        """Validate the entire registry for consistency.
        
        Returns a dict with validation results and any issues found.
        """
        issues = []
        
        # Check all experiment IDs are unique
        all_ids = self.list_all()
        if len(all_ids) != len(set(all_ids)):
            issues.append("Duplicate experiment IDs found")
        
        # Check all enum values are registered
        for exp_enum in ExperimentID:
            if exp_enum.value not in self._experiments:
                issues.append(f"Enum {exp_enum.value} not in registry")
        
        # Check all registered IDs have enum
        for exp_id in all_ids:
            try:
                ExperimentID(exp_id)
            except ValueError:
                issues.append(f"Registered ID {exp_id} has no corresponding enum")
        
        # Check dependencies exist
        for exp_id in all_ids:
            for dep in self.get_dependencies(exp_id):
                if dep not in self._experiments:
                    issues.append(f"Experiment {exp_id} depends on non-existent {dep}")
        
        # Check model seeds are reasonable
        for exp_id in all_ids:
            config = self.get(exp_id)
            for seed in config.model_seeds:
                if not isinstance(seed, int) or seed <= 0:
                    issues.append(f"Invalid model seed in {exp_id}: {seed}")
        
        # Check calibration seeds are reasonable
        for exp_id in all_ids:
            config = self.get(exp_id)
            for seed in config.calibration_seeds:
                if not isinstance(seed, int) or seed <= 0:
                    issues.append(f"Invalid calibration seed in {exp_id}: {seed}")
        
        return {
            "valid": len(issues) == 0,
            "total_experiments": len(all_ids),
            "synthetic_count": len(self.list_synthetic()),
            "real_data_count": len(self.list_real_data()),
            "confirmatory_count": len(self.list_confirmatory()),
            "issues": issues,
        }


# Global registry singleton
_registry: Optional[ExperimentRegistry] = None


def get_registry() -> ExperimentRegistry:
    """Get the global experiment registry singleton."""
    global _registry
    if _registry is None:
        _registry = ExperimentRegistry()
    return _registry


def reset_registry() -> None:
    """Reset the global experiment registry (for testing)."""
    global _registry
    _registry = None
