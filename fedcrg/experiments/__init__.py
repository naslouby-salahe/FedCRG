"""FedCRG Experiment Registry and Execution.

This module implements the complete experiment registry per Section 11 of the
FedCRG Roadmap v2.0, including:
- Synthetic experiments S1-S6 for theorem validation and sensitivity
- Real data experiments R1-R14 for N-BaIoT and DIAD validation
- Experiment execution engine with score cache reuse
- Result serialization and provenance tracking
"""

from fedcrg.experiments.registry import (
    ExperimentID,
    ExperimentType,
    ExperimentConfig,
    ExperimentRegistry,
    get_registry,
)
from fedcrg.experiments.synthetic import (
    run_s1_gate_a_theorem,
    run_s2_target_fpr_sensitivity,
    run_s3_temporal_dependence,
    run_s4_calibration_shift,
    run_s5_contamination,
    run_s6_gate_b_power,
)
from fedcrg.experiments.real_data import (
    run_r1_primary,
    run_r2_gate_a_sweep,
    run_r3_gate_b_sweep,
    run_r4_tolerance_sensitivity,
    run_r5_target_fpr_sensitivity,
    run_r6_assurance_sensitivity,
    run_r7_multiplicity_sensitivity,
    run_r8_source_order,
    run_r9_real_contamination,
    run_r10_diad_replication,
    run_r11_second_detector,
    run_r12_source_order_roles,
    run_r13_computational_benchmark,
    run_r14_diad_feature_sensitivity,
)
from fedcrg.experiments.executor import (
    ExperimentExecutor,
    ExperimentResult,
    run_experiment,
    run_all_synthetic,
    run_all_real_data,
)
from fedcrg.experiments.results import (
    ResultCollector,
    serialize_results,
    deserialize_results,
    compute_aggregate_metrics,
)

__all__ = [
    # Registry
    "ExperimentID",
    "ExperimentType",
    "ExperimentConfig",
    "ExperimentRegistry",
    "get_registry",
    # Synthetic
    "run_s1_gate_a_theorem",
    "run_s2_target_fpr_sensitivity",
    "run_s3_temporal_dependence",
    "run_s4_calibration_shift",
    "run_s5_contamination",
    "run_s6_gate_b_power",
    # Real data
    "run_r1_primary",
    "run_r2_gate_a_sweep",
    "run_r3_gate_b_sweep",
    "run_r4_tolerance_sensitivity",
    "run_r5_target_fpr_sensitivity",
    "run_r6_assurance_sensitivity",
    "run_r7_multiplicity_sensitivity",
    "run_r8_source_order",
    "run_r9_real_contamination",
    "run_r10_diad_replication",
    "run_r11_second_detector",
    "run_r12_source_order_roles",
    "run_r13_computational_benchmark",
    "run_r14_diad_feature_sensitivity",
    # Executor
    "ExperimentExecutor",
    "ExperimentResult",
    "run_experiment",
    "run_all_synthetic",
    "run_all_real_data",
    # Results
    "ResultCollector",
    "serialize_results",
    "deserialize_results",
    "compute_aggregate_metrics",
]
