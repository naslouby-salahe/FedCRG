"""The complete pre-registered S1-S6 / R1-R14 experiment catalogue and lookup.

This is intentionally kept as one cohesive catalogue rather than split by
``ExperimentType`` per definition: dependencies are wired across type boundaries
(e.g. every sensitivity/robustness experiment depends on the primary R1 experiment),
so a reader must see the whole table to understand the dependency graph either way.
"""

from __future__ import annotations

from dataclasses import dataclass

from fedcrg.domain.enums import (
    ArtifactType,
    CalibrationAssignmentMode,
    ContaminationDirection,
    ExperimentAxisId,
    ExperimentId,
    ExperimentType,
    MultiplicityProcedure,
    PolicyId,
    SyntheticDistribution,
)

AxisValue = (
    int
    | float
    | SyntheticDistribution
    | ContaminationDirection
    | MultiplicityProcedure
    | CalibrationAssignmentMode
)


@dataclass(frozen=True, slots=True)
class ParameterAxis:
    """An independent axis whose values may be crossed with other independent axes."""

    id: ExperimentAxisId
    values: tuple[AxisValue, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError(f"Experiment axis {self.id.value} must contain values")
        if len(set(self.values)) != len(self.values):
            raise ValueError(f"Experiment axis {self.id.value} contains duplicate values")


@dataclass(frozen=True, slots=True)
class ParameterSetting:
    axis: ExperimentAxisId
    value: AxisValue


@dataclass(frozen=True, slots=True)
class ParameterCell:
    """A coupled combination that must be evaluated together, not Cartesian-crossed."""

    settings: tuple[ParameterSetting, ...]

    def __post_init__(self) -> None:
        if not self.settings:
            raise ValueError("A parameter cell must contain at least one setting")
        axes = tuple(item.axis for item in self.settings)
        if len(set(axes)) != len(axes):
            raise ValueError("A parameter cell cannot assign the same axis twice")

    def value(self, axis: ExperimentAxisId) -> AxisValue:
        for setting in self.settings:
            if setting.axis is axis:
                return setting.value
        raise KeyError(axis.value)


@dataclass(frozen=True, slots=True)
class WorkloadExpectation:
    monte_carlo_trials: int = 0
    exact_cells: int = 0
    detector_trainings: int = 0

    def __post_init__(self) -> None:
        if min(self.monte_carlo_trials, self.exact_cells, self.detector_trainings) < 0:
            raise ValueError("Workload expectations cannot be negative")


@dataclass(frozen=True, slots=True)
class ExperimentDefinition:
    id: ExperimentId
    type: ExperimentType
    axes: tuple[ParameterAxis, ...] = ()
    coupled_cells: tuple[ParameterCell, ...] = ()
    dependencies: tuple[ExperimentId, ...] = ()
    policies: tuple[PolicyId, ...] = ()
    required_artifacts: tuple[ArtifactType, ...] = ()
    workload: WorkloadExpectation = WorkloadExpectation()
    confirmatory: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        independent_axes = tuple(axis.id for axis in self.axes)
        if len(set(independent_axes)) != len(independent_axes):
            raise ValueError(f"Duplicate axis in {self.id.value}")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError(f"Duplicate dependency in {self.id.value}")
        if self.id in self.dependencies:
            raise ValueError(f"Experiment {self.id.value} cannot depend on itself")
        if len(set(self.policies)) != len(self.policies):
            raise ValueError(f"Duplicate policy in {self.id.value}")

    def axis(self, axis_id: ExperimentAxisId) -> ParameterAxis:
        for item in self.axes:
            if item.id is axis_id:
                return item
        raise KeyError(f"Experiment {self.id.value} has no independent {axis_id.value} axis")


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


def axis(axis_id: ExperimentAxisId, *values: AxisValue) -> ParameterAxis:
    return ParameterAxis(axis_id, tuple(values))


def setting(axis_id: ExperimentAxisId, value: int | float) -> ParameterSetting:
    return ParameterSetting(axis_id, value)


def target_fpr_cell(alpha: float, sample_count: int) -> ParameterCell:
    return ParameterCell(
        (
            setting(ExperimentAxisId.ALPHA, alpha),
            setting(ExperimentAxisId.CALIBRATION_N, sample_count),
        )
    )


def _catalogue() -> tuple[ExperimentDefinition, ...]:
    primary = ExperimentId.PRIMARY_NBAIOT
    external = ExperimentId.EXTERNAL_DIAD
    return (
        ExperimentDefinition(
            id=ExperimentId.READINESS_THEOREM,
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
            type=ExperimentType.PRIMARY,
            policies=ALL_POLICIES,
            required_artifacts=PRIMARY_REQUIRED,
            workload=WorkloadExpectation(detector_trainings=5),
            confirmatory=True,
            description="N-BaIoT primary natural-client experiment",
        ),
        ExperimentDefinition(
            id=ExperimentId.READINESS_SAMPLE_SIZE,
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
            type=ExperimentType.SENSITIVITY,
            axes=(axis(ExperimentAxisId.RHO, 0.25, 0.50, 1.0),),
            dependencies=(primary,),
            policies=ALL_POLICIES,
            description="Operating-band tolerance sensitivity",
        ),
        ExperimentDefinition(
            id=ExperimentId.TARGET_FPR_REAL,
            type=ExperimentType.SENSITIVITY,
            axes=(axis(ExperimentAxisId.ALPHA, 0.005, 0.01, 0.02, 0.05),),
            dependencies=(primary,),
            policies=ALL_POLICIES,
            description="Real-score target-FPR sensitivity",
        ),
        ExperimentDefinition(
            id=ExperimentId.ASSURANCE_SENSITIVITY,
            type=ExperimentType.SENSITIVITY,
            axes=(axis(ExperimentAxisId.READINESS_ASSURANCE, 0.90, 0.95, 0.99),),
            dependencies=(primary,),
            policies=ALL_POLICIES,
            description="Readiness-assurance sensitivity",
        ),
        ExperimentDefinition(
            id=ExperimentId.MULTIPLICITY_SENSITIVITY,
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
            type=ExperimentType.ROBUSTNESS,
            axes=(axis(ExperimentAxisId.BLOCKS, 5),),
            dependencies=(primary,),
            policies=ALL_POLICIES,
            description="Five-block source-order final-benign evaluation",
        ),
        ExperimentDefinition(
            id=ExperimentId.REAL_CONTAMINATION,
            type=ExperimentType.ROBUSTNESS,
            axes=(axis(ExperimentAxisId.FRACTION, 0.001, 0.005, 0.01, 0.02, 0.05),),
            dependencies=(primary,),
            policies=(PolicyId.FEDCRG,),
            description="Real-score calibration contamination sensitivity",
        ),
        ExperimentDefinition(
            id=external,
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
            type=ExperimentType.ROBUSTNESS,
            dependencies=(primary,),
            policies=SECOND_DETECTOR_POLICIES,
            workload=WorkloadExpectation(detector_trainings=3),
            description="Mandatory Deep-SVDD second-score-generator check",
        ),
        ExperimentDefinition(
            id=ExperimentId.SOURCE_ORDER_CALIBRATION,
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


_CATALOGUE: dict[ExperimentId, ExperimentDefinition] = {row.id: row for row in _catalogue()}
if len(_CATALOGUE) != len(ExperimentId) or set(_CATALOGUE) != set(ExperimentId):
    raise RuntimeError("The experiment catalogue must contain exactly one entry per ExperimentId")


def get_experiment_definition(experiment_id: ExperimentId) -> ExperimentDefinition:
    return _CATALOGUE[experiment_id]


def all_experiment_definitions() -> tuple[ExperimentDefinition, ...]:
    return tuple(_CATALOGUE.values())
