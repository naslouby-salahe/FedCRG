"""The complete pre-registered S1-S6 / R1-R14 experiment catalogue."""

from __future__ import annotations

from fedcrg.domain.enums import (
    ArtifactType,
    CalibrationAssignmentMode,
    ContaminationDirection,
    ExperimentAxisId,
    ExperimentCode,
    ExperimentId,
    ExperimentType,
    MultiplicityProcedure,
    PolicyId,
    SyntheticDistribution,
)
from fedcrg.experiments.models import (
    ExperimentDefinition,
    ParameterAxis,
    ParameterCell,
    ParameterSetting,
    WorkloadExpectation,
)

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


def axis(axis_id: ExperimentAxisId, *values: object) -> ParameterAxis:
    return ParameterAxis(axis_id, tuple(values))  # type: ignore[arg-type]


def setting(axis_id: ExperimentAxisId, value: int | float) -> ParameterSetting:
    return ParameterSetting(axis_id, value)


def target_fpr_cell(alpha: float, sample_count: int) -> ParameterCell:
    return ParameterCell(
        (
            setting(ExperimentAxisId.ALPHA, alpha),
            setting(ExperimentAxisId.CALIBRATION_N, sample_count),
        )
    )


def definitions() -> tuple[ExperimentDefinition, ...]:
    primary = ExperimentId.PRIMARY_NBAIOT
    external = ExperimentId.EXTERNAL_DIAD
    return (
        ExperimentDefinition(
            id=ExperimentId.READINESS_THEOREM,
            protocol_code=ExperimentCode.S1,
            type=ExperimentType.SYNTHETIC,
            axes=(
                axis(
                    ExperimentAxisId.DISTRIBUTION,
                    SyntheticDistribution.NORMAL,
                    SyntheticDistribution.LOGNORMAL,
                    SyntheticDistribution.GAMMA_SHAPE_2,
                    SyntheticDistribution.NORMAL_MIXTURE,
                ),
                axis(
                    ExperimentAxisId.CALIBRATION_N,
                    500,
                    1000,
                    1400,
                    1415,
                    1416,
                    1500,
                    2000,
                    3000,
                ),
                axis(ExperimentAxisId.REPETITIONS, 10_000),
            ),
            workload=WorkloadExpectation(monte_carlo_trials=320_000),
            confirmatory=True,
            description="IID finite-sample readiness theorem validation",
        ),
        ExperimentDefinition(
            id=ExperimentId.TARGET_FPR_SYNTHETIC,
            protocol_code=ExperimentCode.S2,
            type=ExperimentType.SYNTHETIC,
            axes=(
                axis(
                    ExperimentAxisId.DISTRIBUTION,
                    SyntheticDistribution.NORMAL,
                    SyntheticDistribution.LOGNORMAL,
                    SyntheticDistribution.GAMMA_SHAPE_2,
                    SyntheticDistribution.NORMAL_MIXTURE,
                ),
                axis(ExperimentAxisId.REPETITIONS, 10_000),
            ),
            coupled_cells=(
                target_fpr_cell(0.005, 2860),
                target_fpr_cell(0.005, 2861),
                target_fpr_cell(0.005, 5722),
                target_fpr_cell(0.02, 693),
                target_fpr_cell(0.02, 694),
                target_fpr_cell(0.02, 1388),
                target_fpr_cell(0.05, 269),
                target_fpr_cell(0.05, 270),
                target_fpr_cell(0.05, 540),
            ),
            workload=WorkloadExpectation(monte_carlo_trials=360_000),
            description="Target-FPR readiness sensitivity",
        ),
        ExperimentDefinition(
            id=ExperimentId.TEMPORAL_DEPENDENCE,
            protocol_code=ExperimentCode.S3,
            type=ExperimentType.ROBUSTNESS,
            axes=(
                axis(ExperimentAxisId.PHI, 0.0, 0.3, 0.6, 0.9),
                axis(ExperimentAxisId.CALIBRATION_N, 1416, 2000, 3000),
                axis(ExperimentAxisId.REPETITIONS, 10_000),
            ),
            workload=WorkloadExpectation(monte_carlo_trials=120_000),
            description="AR(1) dependence stress for readiness coverage",
        ),
        ExperimentDefinition(
            id=ExperimentId.CALIBRATION_SHIFT,
            protocol_code=ExperimentCode.S4,
            type=ExperimentType.ROBUSTNESS,
            axes=(
                axis(ExperimentAxisId.MEAN_SHIFT, 0.0, 0.10, 0.25, 0.50, 1.0),
                axis(ExperimentAxisId.REPETITIONS, 10_000),
            ),
            workload=WorkloadExpectation(monte_carlo_trials=50_000),
            description="Calibration-to-deployment mean-shift stress",
        ),
        ExperimentDefinition(
            id=ExperimentId.CALIBRATION_CONTAMINATION,
            protocol_code=ExperimentCode.S5,
            type=ExperimentType.ROBUSTNESS,
            axes=(
                axis(
                    ExperimentAxisId.FRACTION,
                    0.0,
                    0.001,
                    0.005,
                    0.01,
                    0.02,
                    0.05,
                ),
                axis(
                    ExperimentAxisId.DIRECTION,
                    ContaminationDirection.HIGH,
                    ContaminationDirection.LOW,
                ),
                axis(ExperimentAxisId.REPETITIONS, 10_000),
            ),
            workload=WorkloadExpectation(monte_carlo_trials=120_000),
            description="High-tail and low-tail calibration contamination stress",
        ),
        ExperimentDefinition(
            id=ExperimentId.MISMATCH_POWER,
            protocol_code=ExperimentCode.S6,
            type=ExperimentType.SYNTHETIC,
            axes=(
                axis(
                    ExperimentAxisId.MISMATCH_N,
                    736,
                    1000,
                    1500,
                    2000,
                    3000,
                ),
                axis(
                    ExperimentAxisId.TRUE_FPR,
                    0.0025,
                    0.005,
                    0.0075,
                    0.01,
                    0.0125,
                    0.015,
                    0.02,
                    0.025,
                    0.03,
                ),
            ),
            workload=WorkloadExpectation(exact_cells=45),
            confirmatory=True,
            description="Exact binomial mismatch-declaration power",
        ),
        ExperimentDefinition(
            id=primary,
            protocol_code=ExperimentCode.R1,
            type=ExperimentType.PRIMARY,
            policies=ALL_POLICIES,
            required_artifacts=PRIMARY_REQUIRED,
            workload=WorkloadExpectation(detector_trainings=5),
            confirmatory=True,
            description="N-BaIoT primary natural-client experiment",
        ),
        ExperimentDefinition(
            id=ExperimentId.READINESS_SAMPLE_SIZE,
            protocol_code=ExperimentCode.R2,
            type=ExperimentType.SENSITIVITY,
            axes=(
                axis(
                    ExperimentAxisId.CALIBRATION_N,
                    500,
                    1000,
                    1400,
                    1415,
                    1416,
                    1500,
                    2000,
                ),
            ),
            dependencies=(primary,),
            policies=ALL_POLICIES,
            description="Readiness evidence-budget sweep",
        ),
        ExperimentDefinition(
            id=ExperimentId.MISMATCH_SAMPLE_SIZE,
            protocol_code=ExperimentCode.R3,
            type=ExperimentType.SENSITIVITY,
            axes=(
                axis(
                    ExperimentAxisId.MISMATCH_N,
                    736,
                    1000,
                    1500,
                    2000,
                    3000,
                ),
            ),
            dependencies=(primary,),
            policies=ALL_POLICIES,
            description="Reference-mismatch evidence-budget sweep",
        ),
        ExperimentDefinition(
            id=ExperimentId.TOLERANCE_SENSITIVITY,
            protocol_code=ExperimentCode.R4,
            type=ExperimentType.SENSITIVITY,
            axes=(axis(ExperimentAxisId.RHO, 0.25, 0.50, 1.0),),
            dependencies=(primary,),
            policies=ALL_POLICIES,
            description="Operating-band tolerance sensitivity",
        ),
        ExperimentDefinition(
            id=ExperimentId.TARGET_FPR_REAL,
            protocol_code=ExperimentCode.R5,
            type=ExperimentType.SENSITIVITY,
            axes=(axis(ExperimentAxisId.ALPHA, 0.005, 0.01, 0.02, 0.05),),
            dependencies=(primary,),
            policies=ALL_POLICIES,
            description="Real-score target-FPR sensitivity",
        ),
        ExperimentDefinition(
            id=ExperimentId.ASSURANCE_SENSITIVITY,
            protocol_code=ExperimentCode.R6,
            type=ExperimentType.SENSITIVITY,
            axes=(axis(ExperimentAxisId.READINESS_ASSURANCE, 0.90, 0.95, 0.99),),
            dependencies=(primary,),
            policies=ALL_POLICIES,
            description="Readiness-assurance sensitivity",
        ),
        ExperimentDefinition(
            id=ExperimentId.MULTIPLICITY_SENSITIVITY,
            protocol_code=ExperimentCode.R7,
            type=ExperimentType.SENSITIVITY,
            axes=(
                axis(
                    ExperimentAxisId.PROCEDURE,
                    MultiplicityProcedure.BONFERRONI_READINESS,
                    MultiplicityProcedure.BONFERRONI_MISMATCH,
                    MultiplicityProcedure.HOLM_DIRECTIONAL,
                ),
            ),
            dependencies=(primary,),
            policies=(PolicyId.FEDCRG,),
            description="Familywise readiness and mismatch sensitivities",
        ),
        ExperimentDefinition(
            id=ExperimentId.SOURCE_ORDER_TEST,
            protocol_code=ExperimentCode.R8,
            type=ExperimentType.ROBUSTNESS,
            axes=(axis(ExperimentAxisId.BLOCKS, 5),),
            dependencies=(primary,),
            policies=ALL_POLICIES,
            description="Five-block source-order final-benign evaluation",
        ),
        ExperimentDefinition(
            id=ExperimentId.REAL_CONTAMINATION,
            protocol_code=ExperimentCode.R9,
            type=ExperimentType.ROBUSTNESS,
            axes=(axis(ExperimentAxisId.FRACTION, 0.001, 0.005, 0.01, 0.02, 0.05),),
            dependencies=(primary,),
            policies=(PolicyId.FEDCRG,),
            description="Real-score calibration contamination sensitivity",
        ),
        ExperimentDefinition(
            id=external,
            protocol_code=ExperimentCode.R10,
            type=ExperimentType.EXTERNAL_VALIDATION,
            dependencies=(primary,),
            policies=ALL_POLICIES,
            required_artifacts=PRIMARY_REQUIRED,
            workload=WorkloadExpectation(detector_trainings=5),
            confirmatory=True,
            description="CIC IoT-DIAD natural-client external replication",
        ),
        ExperimentDefinition(
            id=ExperimentId.SECOND_DETECTOR,
            protocol_code=ExperimentCode.R11,
            type=ExperimentType.ROBUSTNESS,
            dependencies=(primary,),
            policies=SECOND_DETECTOR_POLICIES,
            workload=WorkloadExpectation(detector_trainings=3),
            description="Mandatory Deep-SVDD second-score-generator check",
        ),
        ExperimentDefinition(
            id=ExperimentId.SOURCE_ORDER_CALIBRATION,
            protocol_code=ExperimentCode.R12,
            type=ExperimentType.ROBUSTNESS,
            axes=(
                axis(
                    ExperimentAxisId.ASSIGNMENT,
                    CalibrationAssignmentMode.SOURCE_ORDER,
                ),
            ),
            dependencies=(primary,),
            policies=ALL_POLICIES,
            description="Calibration-role source-order sensitivity",
        ),
        ExperimentDefinition(
            id=ExperimentId.COMPUTATIONAL_BENCHMARK,
            protocol_code=ExperimentCode.R13,
            type=ExperimentType.BENCHMARK,
            axes=(
                axis(ExperimentAxisId.WARMUPS, 100),
                axis(ExperimentAxisId.REPETITIONS, 1000),
            ),
            dependencies=(primary,),
            description="CPU primitive runtime and memory benchmark",
        ),
        ExperimentDefinition(
            id=ExperimentId.DIAD_FEATURE_SENSITIVITY,
            protocol_code=ExperimentCode.R14,
            type=ExperimentType.SENSITIVITY,
            dependencies=(external,),
            policies=(
                PolicyId.GLOBAL_QUANTILE,
                PolicyId.LOCAL_QUANTILE,
                PolicyId.SHRINKAGE,
                PolicyId.FEDCRG,
            ),
            workload=WorkloadExpectation(detector_trainings=5),
            description="Training-schema-derived DIAD numeric-safe feature sensitivity",
        ),
    )
