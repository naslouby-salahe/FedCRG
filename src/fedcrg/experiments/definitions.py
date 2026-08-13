"""The complete pre-registered S1-S6 / R1-R14 experiment catalogue."""

from __future__ import annotations

from fedcrg.core.enums import ArtifactType, ExperimentId, ExperimentType, PolicyId
from fedcrg.experiments.models import ExperimentDefinition, ParameterAxis, WorkloadExpectation

ALL_POLICIES = tuple(PolicyId)
SECOND_DETECTOR_POLICIES = (
    PolicyId.GLOBAL_QUANTILE,
    PolicyId.LOCAL_QUANTILE,
    PolicyId.SHRINKAGE,
    PolicyId.READINESS_ONLY,
    PolicyId.FEDCRG,
)
PRIMARY_REQUIRED = (
    ArtifactType.RESOLVED_CONFIG,
    ArtifactType.DATASET_MANIFEST,
    ArtifactType.PREPROCESSING_MANIFEST,
    ArtifactType.TRAINING_MANIFEST,
    ArtifactType.MODEL,
    ArtifactType.SCORE_MANIFEST,
    ArtifactType.THRESHOLD_RECORDS,
    ArtifactType.METRICS,
    ArtifactType.VERIFICATION,
)


def axis(name: str, *values: int | float | str) -> ParameterAxis:
    return ParameterAxis(name, tuple(values))


def definitions() -> tuple[ExperimentDefinition, ...]:
    primary = ExperimentId.PRIMARY_NBAIOT
    external = ExperimentId.EXTERNAL_DIAD
    return (
        ExperimentDefinition(
            ExperimentId.READINESS_THEOREM,
            "S1",
            ExperimentType.SYNTHETIC,
            axes=(
                axis("distribution", "normal", "lognormal", "gamma2", "normal_mixture"),
                axis("calibration_n", 500, 1000, 1400, 1415, 1416, 1500, 2000, 3000),
                axis("repetitions", 10000),
            ),
            workload=WorkloadExpectation(monte_carlo_trials=320000),
            confirmatory=True,
            description="IID finite-sample readiness theorem validation",
        ),
        ExperimentDefinition(
            ExperimentId.TARGET_FPR_SYNTHETIC,
            "S2",
            ExperimentType.SYNTHETIC,
            axes=(
                axis("alpha", 0.005, 0.02, 0.05),
                axis("distribution", "normal", "lognormal", "gamma2", "normal_mixture"),
                axis("alpha_0.005_n", 2860, 2861, 5722),
                axis("alpha_0.02_n", 693, 694, 1388),
                axis("alpha_0.05_n", 269, 270, 540),
                axis("repetitions", 10000),
            ),
            workload=WorkloadExpectation(monte_carlo_trials=360000),
        ),
        ExperimentDefinition(
            ExperimentId.TEMPORAL_DEPENDENCE,
            "S3",
            ExperimentType.ROBUSTNESS,
            axes=(axis("phi", 0.0, 0.3, 0.6, 0.9), axis("calibration_n", 1416, 2000, 3000), axis("repetitions", 10000)),
            workload=WorkloadExpectation(monte_carlo_trials=120000),
        ),
        ExperimentDefinition(
            ExperimentId.CALIBRATION_SHIFT,
            "S4",
            ExperimentType.ROBUSTNESS,
            axes=(axis("mean_shift", 0.0, 0.10, 0.25, 0.50, 1.0), axis("repetitions", 10000)),
            workload=WorkloadExpectation(monte_carlo_trials=50000),
        ),
        ExperimentDefinition(
            ExperimentId.CALIBRATION_CONTAMINATION,
            "S5",
            ExperimentType.ROBUSTNESS,
            axes=(axis("fraction", 0.0, 0.001, 0.005, 0.01, 0.02, 0.05), axis("direction", "high", "low"), axis("repetitions", 10000)),
            workload=WorkloadExpectation(monte_carlo_trials=120000),
        ),
        ExperimentDefinition(
            ExperimentId.MISMATCH_POWER,
            "S6",
            ExperimentType.SYNTHETIC,
            axes=(axis("mismatch_n", 736, 1000, 1500, 2000, 3000), axis("true_fpr", 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03)),
            workload=WorkloadExpectation(exact_cells=45),
            confirmatory=True,
        ),
        ExperimentDefinition(primary, "R1", ExperimentType.PRIMARY, policies=ALL_POLICIES, required_artifacts=PRIMARY_REQUIRED, workload=WorkloadExpectation(detector_trainings=5), confirmatory=True),
        ExperimentDefinition(ExperimentId.READINESS_SAMPLE_SIZE, "R2", ExperimentType.SENSITIVITY, axes=(axis("calibration_n", 500, 1000, 1400, 1415, 1416, 1500, 2000),), dependencies=(primary,), policies=ALL_POLICIES),
        ExperimentDefinition(ExperimentId.MISMATCH_SAMPLE_SIZE, "R3", ExperimentType.SENSITIVITY, axes=(axis("mismatch_n", 736, 1000, 1500, 2000, 3000),), dependencies=(primary,), policies=ALL_POLICIES),
        ExperimentDefinition(ExperimentId.TOLERANCE_SENSITIVITY, "R4", ExperimentType.SENSITIVITY, axes=(axis("rho", 0.25, 0.50, 1.0),), dependencies=(primary,), policies=ALL_POLICIES),
        ExperimentDefinition(ExperimentId.TARGET_FPR_REAL, "R5", ExperimentType.SENSITIVITY, axes=(axis("alpha", 0.005, 0.01, 0.02, 0.05),), dependencies=(primary,), policies=ALL_POLICIES),
        ExperimentDefinition(ExperimentId.ASSURANCE_SENSITIVITY, "R6", ExperimentType.SENSITIVITY, axes=(axis("readiness_assurance", 0.90, 0.95, 0.99),), dependencies=(primary,), policies=ALL_POLICIES),
        ExperimentDefinition(ExperimentId.MULTIPLICITY_SENSITIVITY, "R7", ExperimentType.SENSITIVITY, axes=(axis("procedure", "bonferroni_readiness", "bonferroni_mismatch", "holm_directional"),), dependencies=(primary,), policies=(PolicyId.FEDCRG,)),
        ExperimentDefinition(ExperimentId.SOURCE_ORDER_TEST, "R8", ExperimentType.ROBUSTNESS, axes=(axis("blocks", 5),), dependencies=(primary,), policies=ALL_POLICIES),
        ExperimentDefinition(ExperimentId.REAL_CONTAMINATION, "R9", ExperimentType.ROBUSTNESS, axes=(axis("fraction", 0.001, 0.005, 0.01, 0.02, 0.05),), dependencies=(primary,), policies=(PolicyId.FEDCRG,)),
        ExperimentDefinition(external, "R10", ExperimentType.EXTERNAL_VALIDATION, dependencies=(primary,), policies=ALL_POLICIES, required_artifacts=PRIMARY_REQUIRED, workload=WorkloadExpectation(detector_trainings=5), confirmatory=True),
        ExperimentDefinition(ExperimentId.SECOND_DETECTOR, "R11", ExperimentType.ROBUSTNESS, dependencies=(primary,), policies=SECOND_DETECTOR_POLICIES, workload=WorkloadExpectation(detector_trainings=3)),
        ExperimentDefinition(ExperimentId.SOURCE_ORDER_CALIBRATION, "R12", ExperimentType.ROBUSTNESS, axes=(axis("assignment", "source_order"),), dependencies=(primary,), policies=ALL_POLICIES),
        ExperimentDefinition(ExperimentId.COMPUTATIONAL_BENCHMARK, "R13", ExperimentType.BENCHMARK, axes=(axis("warmups", 100), axis("repetitions", 1000)), dependencies=(primary,)),
        ExperimentDefinition(ExperimentId.DIAD_FEATURE_SENSITIVITY, "R14", ExperimentType.SENSITIVITY, dependencies=(external,), policies=(PolicyId.GLOBAL_QUANTILE, PolicyId.LOCAL_QUANTILE, PolicyId.SHRINKAGE, PolicyId.FEDCRG), workload=WorkloadExpectation(detector_trainings=5)),
    )
