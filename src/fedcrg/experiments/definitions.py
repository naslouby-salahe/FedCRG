"""Authoritative experiment catalogue."""

from fedcrg.core.enums import ArtifactType, ExperimentId, ExperimentType
from fedcrg.experiments.models import ExperimentDefinition


def definitions() -> tuple[ExperimentDefinition, ...]:
    primary = ExperimentId.PRIMARY_NBAIOT
    return (
        ExperimentDefinition(ExperimentId.READINESS_THEOREM, ExperimentType.SYNTHETIC, confirmatory=True),
        ExperimentDefinition(ExperimentId.TARGET_FPR_SYNTHETIC, ExperimentType.SENSITIVITY),
        ExperimentDefinition(ExperimentId.TEMPORAL_DEPENDENCE, ExperimentType.ROBUSTNESS),
        ExperimentDefinition(ExperimentId.CALIBRATION_SHIFT, ExperimentType.ROBUSTNESS),
        ExperimentDefinition(ExperimentId.CALIBRATION_CONTAMINATION, ExperimentType.ROBUSTNESS),
        ExperimentDefinition(ExperimentId.MISMATCH_POWER, ExperimentType.SYNTHETIC, confirmatory=True),
        ExperimentDefinition(
            primary,
            ExperimentType.PRIMARY,
            required_artifacts=(
                ArtifactType.DATASET_MANIFEST,
                ArtifactType.TRAINING_MANIFEST,
                ArtifactType.SCORE_MANIFEST,
                ArtifactType.DECISIONS,
                ArtifactType.METRICS,
                ArtifactType.VERIFICATION,
            ),
            confirmatory=True,
        ),
        ExperimentDefinition(ExperimentId.READINESS_SAMPLE_SIZE, ExperimentType.SENSITIVITY, (primary,)),
        ExperimentDefinition(ExperimentId.MISMATCH_SAMPLE_SIZE, ExperimentType.SENSITIVITY, (primary,)),
        ExperimentDefinition(ExperimentId.TOLERANCE_SENSITIVITY, ExperimentType.SENSITIVITY, (primary,)),
        ExperimentDefinition(ExperimentId.TARGET_FPR_REAL, ExperimentType.SENSITIVITY, (primary,)),
        ExperimentDefinition(ExperimentId.ASSURANCE_SENSITIVITY, ExperimentType.SENSITIVITY, (primary,)),
        ExperimentDefinition(ExperimentId.MULTIPLICITY_SENSITIVITY, ExperimentType.SENSITIVITY, (primary,)),
        ExperimentDefinition(ExperimentId.SOURCE_ORDER_TEST, ExperimentType.ROBUSTNESS, (primary,)),
        ExperimentDefinition(ExperimentId.REAL_CONTAMINATION, ExperimentType.ROBUSTNESS, (primary,)),
        ExperimentDefinition(ExperimentId.EXTERNAL_DIAD, ExperimentType.EXTERNAL_VALIDATION, (primary,), confirmatory=True),
        ExperimentDefinition(ExperimentId.SECOND_DETECTOR, ExperimentType.ROBUSTNESS, (primary,)),
        ExperimentDefinition(ExperimentId.SOURCE_ORDER_CALIBRATION, ExperimentType.ROBUSTNESS, (primary,)),
        ExperimentDefinition(ExperimentId.COMPUTATIONAL_BENCHMARK, ExperimentType.BENCHMARK, (primary,)),
        ExperimentDefinition(ExperimentId.DIAD_FEATURE_SENSITIVITY, ExperimentType.SENSITIVITY, (ExperimentId.EXTERNAL_DIAD,)),
    )
