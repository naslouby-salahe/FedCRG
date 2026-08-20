"""Authoritative per-experiment artifact and completion contracts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from fedcrg.paths import LayoutArtifact
from fedcrg.types import (
    ArtifactType,
    CacheRole,
    DatasetId,
    DetectorId,
    ExperimentApplicability,
    ExperimentId,
    ExperimentType,
    JsonResultKind,
    ResultFormat,
    SharedCacheKind,
)

Frozen = ConfigDict(frozen=True, extra="forbid")

_EMPIRICAL_RUN_ARTIFACTS: tuple[ArtifactType, ...] = (
    ArtifactType.RESOLVED_CONFIG,
    ArtifactType.DATASET_MANIFEST,
    ArtifactType.PREPROCESSING_MANIFEST,
    ArtifactType.TRAINING_MANIFEST,
    ArtifactType.MODEL,
    ArtifactType.SCORE_MANIFEST,
    ArtifactType.THRESHOLD_RECORDS,
    ArtifactType.METRICS,
)

_SYNTHETIC_RUN_ARTIFACTS: tuple[ArtifactType, ...] = ()


class RequiredResultFile(BaseModel):
    """One required delivery artifact owned by an experiment."""

    model_config = Frozen

    format: ResultFormat
    filename: LayoutArtifact


class DetectorCacheIdentity(BaseModel):
    """Dataset and detector pair that keys a shared model/score cache family."""

    model_config = Frozen

    dataset_id: DatasetId
    detector_id: DetectorId


class ExperimentArtifactContract(BaseModel):
    """What must exist, validate, and be current before an experiment is fully passed."""

    model_config = Frozen

    experiment_id: ExperimentId
    category: ExperimentType
    applicability: ExperimentApplicability
    json_kind: JsonResultKind
    run_artifacts: tuple[ArtifactType, ...]
    json_files: tuple[RequiredResultFile, ...]
    csv_files: tuple[RequiredResultFile, ...]
    figure_files: tuple[RequiredResultFile, ...]
    requires_run_grid: bool
    requires_analysis_json: bool
    cache_kind: SharedCacheKind
    cache_role: CacheRole

    @model_validator(mode="after")
    def _validate(self) -> ExperimentArtifactContract:
        if not self.json_files:
            raise ValueError(f"{self.experiment_id} must declare a JSON result file")
        if self.cache_role is CacheRole.NONE and self.cache_kind is not SharedCacheKind.NONE:
            raise ValueError(f"{self.experiment_id} cache kind requires a cache role")
        if self.cache_role is not CacheRole.NONE and self.cache_kind is SharedCacheKind.NONE:
            raise ValueError(f"{self.experiment_id} cache role requires a cache kind")
        return self

    @property
    def inapplicable(self) -> bool:
        return self.applicability is ExperimentApplicability.INAPPLICABLE

    @property
    def optional(self) -> bool:
        return self.applicability is ExperimentApplicability.OPTIONAL


def detector_cache_identity(kind: SharedCacheKind) -> DetectorCacheIdentity | None:
    """Return the dataset/detector pair owned or consumed by `kind`."""
    match kind:
        case SharedCacheKind.NONE:
            return None
        case SharedCacheKind.NBAIOT_AUTOENCODER:
            return DetectorCacheIdentity(
                dataset_id=DatasetId.NBAIOT, detector_id=DetectorId.AUTOENCODER
            )
        case SharedCacheKind.NBAIOT_DEEP_SVDD:
            return DetectorCacheIdentity(
                dataset_id=DatasetId.NBAIOT, detector_id=DetectorId.DEEP_SVDD
            )
        case SharedCacheKind.DIAD_AUTOENCODER:
            return DetectorCacheIdentity(
                dataset_id=DatasetId.DIAD, detector_id=DetectorId.AUTOENCODER
            )


def _json_result() -> tuple[RequiredResultFile, ...]:
    return (
        RequiredResultFile(
            format=ResultFormat.JSON, filename=LayoutArtifact.EXPERIMENT_RESULT_JSON
        ),
    )


def _cells_csv() -> tuple[RequiredResultFile, ...]:
    return (
        RequiredResultFile(format=ResultFormat.CSV, filename=LayoutArtifact.EXPERIMENT_CELLS_CSV),
    )


def _policy_csv() -> tuple[RequiredResultFile, ...]:
    return (
        RequiredResultFile(format=ResultFormat.CSV, filename=LayoutArtifact.EXPERIMENT_POLICY_CSV),
    )


def _benchmark_csv() -> tuple[RequiredResultFile, ...]:
    return (
        RequiredResultFile(
            format=ResultFormat.CSV, filename=LayoutArtifact.EXPERIMENT_BENCHMARK_CSV
        ),
    )


def _synthetic(
    experiment_id: ExperimentId,
    category: ExperimentType,
    *,
    figure: LayoutArtifact | None,
    json_kind: JsonResultKind = JsonResultKind.SYNTHETIC_ENVELOPE,
) -> ExperimentArtifactContract:
    figures = (
        () if figure is None else (RequiredResultFile(format=ResultFormat.FIGURE, filename=figure),)
    )
    csv_files = _benchmark_csv() if json_kind is JsonResultKind.BENCHMARK_REPORT else _cells_csv()
    return ExperimentArtifactContract(
        experiment_id=experiment_id,
        category=category,
        applicability=ExperimentApplicability.REQUIRED,
        json_kind=json_kind,
        run_artifacts=_SYNTHETIC_RUN_ARTIFACTS,
        json_files=_json_result(),
        csv_files=csv_files,
        figure_files=figures,
        requires_run_grid=False,
        requires_analysis_json=True,
        cache_kind=SharedCacheKind.NONE,
        cache_role=CacheRole.NONE,
    )


def _empirical(
    experiment_id: ExperimentId,
    category: ExperimentType,
    *,
    cache_kind: SharedCacheKind,
    cache_role: CacheRole,
    figures: tuple[RequiredResultFile, ...] = (),
) -> ExperimentArtifactContract:
    return ExperimentArtifactContract(
        experiment_id=experiment_id,
        category=category,
        applicability=ExperimentApplicability.REQUIRED,
        json_kind=JsonResultKind.EMPIRICAL_RESULTS,
        run_artifacts=_EMPIRICAL_RUN_ARTIFACTS,
        json_files=_json_result(),
        csv_files=_policy_csv(),
        figure_files=figures,
        requires_run_grid=True,
        requires_analysis_json=False,
        cache_kind=cache_kind,
        cache_role=cache_role,
    )


_PRIMARY_FIGURES = (
    RequiredResultFile(
        format=ResultFormat.FIGURE, filename=LayoutArtifact.EXPERIMENT_OPERATING_POINTS_FIGURE
    ),
    RequiredResultFile(
        format=ResultFormat.FIGURE, filename=LayoutArtifact.EXPERIMENT_RELIABILITY_FIGURE
    ),
)
_EXTERNAL_FIGURES = (
    RequiredResultFile(
        format=ResultFormat.FIGURE, filename=LayoutArtifact.EXPERIMENT_REPLICATION_FIGURE
    ),
)


_DEFAULT_CONTRACTS: tuple[ExperimentArtifactContract, ...] = (
    _synthetic(
        ExperimentId.READINESS_THEOREM,
        ExperimentType.SYNTHETIC,
        figure=LayoutArtifact.EXPERIMENT_COVERAGE_FIGURE,
    ),
    _synthetic(
        ExperimentId.TARGET_FPR_SYNTHETIC,
        ExperimentType.SYNTHETIC,
        figure=LayoutArtifact.EXPERIMENT_COVERAGE_FIGURE,
    ),
    _synthetic(
        ExperimentId.TEMPORAL_DEPENDENCE,
        ExperimentType.ROBUSTNESS,
        figure=LayoutArtifact.EXPERIMENT_COVERAGE_FIGURE,
    ),
    _synthetic(
        ExperimentId.CALIBRATION_SHIFT,
        ExperimentType.ROBUSTNESS,
        figure=LayoutArtifact.EXPERIMENT_COVERAGE_FIGURE,
    ),
    _synthetic(
        ExperimentId.CALIBRATION_CONTAMINATION,
        ExperimentType.ROBUSTNESS,
        figure=LayoutArtifact.EXPERIMENT_COVERAGE_FIGURE,
    ),
    _synthetic(
        ExperimentId.MISMATCH_POWER,
        ExperimentType.SYNTHETIC,
        figure=LayoutArtifact.EXPERIMENT_POWER_FIGURE,
    ),
    _empirical(
        ExperimentId.PRIMARY_NBAIOT,
        ExperimentType.PRIMARY,
        cache_kind=SharedCacheKind.NBAIOT_AUTOENCODER,
        cache_role=CacheRole.OWNER,
        figures=_PRIMARY_FIGURES,
    ),
    _empirical(
        ExperimentId.READINESS_SAMPLE_SIZE,
        ExperimentType.SENSITIVITY,
        cache_kind=SharedCacheKind.NBAIOT_AUTOENCODER,
        cache_role=CacheRole.CONSUMER,
    ),
    _empirical(
        ExperimentId.MISMATCH_SAMPLE_SIZE,
        ExperimentType.SENSITIVITY,
        cache_kind=SharedCacheKind.NBAIOT_AUTOENCODER,
        cache_role=CacheRole.CONSUMER,
    ),
    _empirical(
        ExperimentId.TOLERANCE_SENSITIVITY,
        ExperimentType.SENSITIVITY,
        cache_kind=SharedCacheKind.NBAIOT_AUTOENCODER,
        cache_role=CacheRole.CONSUMER,
    ),
    _empirical(
        ExperimentId.TARGET_FPR_REAL,
        ExperimentType.SENSITIVITY,
        cache_kind=SharedCacheKind.NBAIOT_AUTOENCODER,
        cache_role=CacheRole.CONSUMER,
    ),
    _empirical(
        ExperimentId.ASSURANCE_SENSITIVITY,
        ExperimentType.SENSITIVITY,
        cache_kind=SharedCacheKind.NBAIOT_AUTOENCODER,
        cache_role=CacheRole.CONSUMER,
    ),
    _empirical(
        ExperimentId.MULTIPLICITY_SENSITIVITY,
        ExperimentType.SENSITIVITY,
        cache_kind=SharedCacheKind.NBAIOT_AUTOENCODER,
        cache_role=CacheRole.CONSUMER,
    ),
    _empirical(
        ExperimentId.SOURCE_ORDER_TEST,
        ExperimentType.ROBUSTNESS,
        cache_kind=SharedCacheKind.NBAIOT_AUTOENCODER,
        cache_role=CacheRole.CONSUMER,
    ),
    _empirical(
        ExperimentId.REAL_CONTAMINATION,
        ExperimentType.ROBUSTNESS,
        cache_kind=SharedCacheKind.NBAIOT_AUTOENCODER,
        cache_role=CacheRole.CONSUMER,
    ),
    _empirical(
        ExperimentId.EXTERNAL_DIAD,
        ExperimentType.EXTERNAL_VALIDATION,
        cache_kind=SharedCacheKind.DIAD_AUTOENCODER,
        cache_role=CacheRole.OWNER,
        figures=_EXTERNAL_FIGURES,
    ),
    _empirical(
        ExperimentId.SECOND_DETECTOR,
        ExperimentType.ROBUSTNESS,
        cache_kind=SharedCacheKind.NBAIOT_DEEP_SVDD,
        cache_role=CacheRole.OWNER,
    ),
    _empirical(
        ExperimentId.SOURCE_ORDER_CALIBRATION,
        ExperimentType.ROBUSTNESS,
        cache_kind=SharedCacheKind.NBAIOT_AUTOENCODER,
        cache_role=CacheRole.CONSUMER,
    ),
    _synthetic(
        ExperimentId.COMPUTATIONAL_BENCHMARK,
        ExperimentType.BENCHMARK,
        figure=None,
        json_kind=JsonResultKind.BENCHMARK_REPORT,
    ),
    _empirical(
        ExperimentId.DIAD_FEATURE_SENSITIVITY,
        ExperimentType.SENSITIVITY,
        cache_kind=SharedCacheKind.DIAD_AUTOENCODER,
        cache_role=CacheRole.CONSUMER,
    ),
)


class ExperimentContractCatalogue:
    """The fixed set of per-experiment completion contracts."""

    def __init__(self, contracts: tuple[ExperimentArtifactContract, ...] | None = None) -> None:
        resolved = contracts if contracts is not None else _DEFAULT_CONTRACTS
        ids = tuple(item.experiment_id for item in resolved)
        if len(set(ids)) != len(ids):
            raise ValueError("The experiment contract catalogue contains duplicate ids")
        if set(ids) != set(ExperimentId):
            missing = sorted(set(ExperimentId) - set(ids))
            extra = sorted(set(ids) - set(ExperimentId))
            raise ValueError(
                f"Experiment contract catalogue must cover every experiment id "
                f"(missing={missing}, extra={extra})"
            )
        self._contracts = resolved

    def all(self) -> tuple[ExperimentArtifactContract, ...]:
        return self._contracts

    def for_id(self, experiment_id: ExperimentId) -> ExperimentArtifactContract:
        for item in self._contracts:
            if item.experiment_id is experiment_id:
                return item
        raise KeyError(experiment_id)


_CATALOGUE = ExperimentContractCatalogue()


def experiment_contract(experiment_id: ExperimentId) -> ExperimentArtifactContract:
    """Return the authoritative completion contract for `experiment_id`."""
    return _CATALOGUE.for_id(experiment_id)


def all_experiment_contracts() -> tuple[ExperimentArtifactContract, ...]:
    """Every registered experiment completion contract."""
    return _CATALOGUE.all()
