"""Validated scientific configuration for the FedCRG study.

Three YAML documents are the single source of truth:

- ``config/study.yaml``       protocol, statistics, randomness, detector/training
                              profiles, and the policy registry;
- ``config/datasets.yaml``    dataset contracts (identity, feature contract,
                              eligibility, role sizing, seeds);
- ``config/experiments.yaml`` the locked experiment catalogue.

There is no inheritance graph and no deep-merge machinery. Python executes
typed ``ExperimentSpec`` values resolved from these documents and never
redefines the catalogue.

Raw primitives appear only at the YAML parse boundary (before validation)
and for free-text metadata such as descriptions and version strings.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal, Self, TypeAlias

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    TypeAdapter,
    ValidationInfo,
    field_validator,
    model_validator,
)

from fedcrg.types import (
    ActivationId,
    AdamBeta,
    AggregationId,
    Alpha,
    ArtifactType,
    Assurance,
    BatchSize,
    BootstrapSeed,
    CalibrationAssignmentMode,
    CalibrationSeed,
    CampaignId,
    CandidateCount,
    ClientFraction,
    ClientId,
    ComputeDeviceId,
    ConfidenceLevel,
    ConfigurationError,
    ContaminationDirection,
    DatasetFeatureContractId,
    DatasetId,
    Description,
    DeepSvddCenterMode,
    DetectorId,
    Dimension,
    EpochCount,
    ExperimentAxisId,
    ExperimentId,
    ExperimentType,
    FeatureCount,
    FeatureName,
    JsonValue,
    LearningRate,
    MetricDifference,
    ModelSeed,
    MultiplicityProcedure,
    NonNegativeCount,
    OperatingBand,
    OptimizerEpsilon,
    OptimizerId,
    PolicyId,
    PositiveCount,
    Probability,
    ProfileId,
    ProtocolId,
    ReplicateCount,
    Rho,
    RoundCount,
    SampleCount,
    Sha256,
    SyntheticDistribution,
    Tolerance,
    UtilityMargin,
    Version,
    WeightDecay,
    XavierGain,
)

FrozenModel = ConfigDict(frozen=True, extra="forbid", use_enum_values=False)

_CALIBRATION_SEED_ADAPTER = TypeAdapter(CalibrationSeed)


def _sha256_json(payload: object) -> Sha256:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


class ProtocolConfig(BaseModel):
    """Locked FedCRG decision-protocol parameters."""

    model_config = FrozenModel

    id: Literal[ProtocolId.FEDCRG] = ProtocolId.FEDCRG
    version: Literal["2.0"] = "2.0"
    alpha: Alpha
    rho: Rho
    readiness_assurance: Assurance
    mismatch_confidence: ConfidenceLevel
    strict_exceedance: Literal[True] = True
    reject_calibration_ties: Literal[True] = True

    @property
    def band(self) -> OperatingBand:
        return OperatingBand(
            lower=max(0.0, self.alpha * (1.0 - self.rho)),
            upper=min(1.0, self.alpha * (1.0 + self.rho)),
        )


class StatisticsConfig(BaseModel):
    """Locked confirmatory statistical-analysis choices."""

    model_config = FrozenModel

    bootstrap_replicates: ReplicateCount
    bootstrap_seed: BootstrapSeed
    utility_margin: UtilityMargin
    familywise_alpha: Probability
    ranking_invariance_tolerance: Tolerance
    shrinkage_n0_candidates: tuple[SampleCount, ...]
    supervised_threshold_candidates: CandidateCount

    @property
    def utility_margin_allowance(self) -> MetricDifference:
        """Signed allowance used by the locked non-inferiority comparison."""
        return -self.utility_margin


class RandomnessConfig(BaseModel):
    """Seed registry shared by real-data experiments."""

    model_config = FrozenModel

    model_seeds: tuple[ModelSeed, ...]
    attack_split_seed: ModelSeed
    synthetic_seed: ModelSeed

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if len(set(self.model_seeds)) != len(self.model_seeds):
            raise ValueError("Model seeds must be unique")
        return self


class TrainingConfig(BaseModel):
    """Frozen federated-training specification with no hidden scientific defaults."""

    model_config = FrozenModel

    rounds: RoundCount
    local_epochs: EpochCount
    batch_size: BatchSize
    optimizer: Literal[OptimizerId.ADAM] = OptimizerId.ADAM
    learning_rate_initial: LearningRate
    learning_rate_final: LearningRate
    adam_betas: tuple[AdamBeta, AdamBeta]
    adam_epsilon: OptimizerEpsilon
    weight_decay: WeightDecay
    client_fraction: ClientFraction
    aggregation: Literal[AggregationId.EQUAL_CLIENT_MEAN] = AggregationId.EQUAL_CLIENT_MEAN
    early_stopping: Literal[False] = False
    mixed_precision: Literal[False] = False
    deterministic_algorithms: Literal[True] = True
    record_round20_score_correlation: bool
    device: ComputeDeviceId

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.learning_rate_final > self.learning_rate_initial:
            raise ValueError("Final learning rate cannot exceed initial learning rate")
        beta1, beta2 = self.adam_betas
        if not 0.0 < beta1 < 1.0 or not 0.0 < beta2 < 1.0:
            raise ValueError("Adam betas must be in (0, 1)")
        return self


class AutoencoderConfig(BaseModel):
    """Frozen autoencoder architecture and initialization contract."""

    model_config = FrozenModel

    id: Literal[DetectorId.AUTOENCODER] = DetectorId.AUTOENCODER
    hidden_dims: tuple[Dimension, ...]
    activation: Literal[ActivationId.TANH] = ActivationId.TANH
    xavier_tanh_gain: XavierGain
    zero_bias: Literal[True] = True


class DeepSvddConfig(BaseModel):
    """Frozen Deep-SVDD encoder and center-mode contract."""

    model_config = FrozenModel

    id: Literal[DetectorId.DEEP_SVDD] = DetectorId.DEEP_SVDD
    hidden_dims: tuple[Dimension, ...]
    embedding_dim: Dimension
    activation: Literal[ActivationId.TANH] = ActivationId.TANH
    bias: Literal[False] = False
    center_mode: Literal[DeepSvddCenterMode.EQUAL_MEAN_OF_CLIENT_INITIAL_EMBEDDINGS] = (
        DeepSvddCenterMode.EQUAL_MEAN_OF_CLIENT_INITIAL_EMBEDDINGS
    )


DetectorConfig = Annotated[AutoencoderConfig | DeepSvddConfig, Field(discriminator="id")]


class StudyPaths(BaseModel):
    """Repository layout roots; the CLI accepts no path options."""

    model_config = FrozenModel

    data_root: Path
    preprocessed_root: Path
    outputs_root: Path
    results_root: Path


class StudyConfig(BaseModel):
    """Everything study-wide: protocol, statistics, randomness, profiles, policies."""

    model_config = FrozenModel

    protocol: ProtocolConfig
    statistics: StatisticsConfig
    randomness: RandomnessConfig
    detector_profiles: dict[ProfileId, DetectorConfig]
    training_profiles: dict[ProfileId, TrainingConfig]
    policies: tuple[PolicyId, ...]
    paths: StudyPaths
    campaign_id: CampaignId

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if len(set(self.policies)) != len(self.policies):
            raise ValueError("Policy registry must be unique")
        for profile_id, detector in self.detector_profiles.items():
            if detector.id is DetectorId.DEEP_SVDD and profile_id not in self.training_profiles:
                raise ValueError(f"Detector profile {profile_id!r} has no training profile")
        return self

    def detector_profile(self, profile_id: ProfileId) -> DetectorConfig:
        try:
            return self.detector_profiles[profile_id]
        except KeyError as exc:
            raise ConfigurationError(f"Unknown detector profile: {profile_id!r}") from exc

    def training_profile(self, profile_id: ProfileId) -> TrainingConfig:
        try:
            return self.training_profiles[profile_id]
        except KeyError as exc:
            raise ConfigurationError(f"Unknown training profile: {profile_id!r}") from exc


class ExpectedBenignCounts(RootModel[dict[ClientId, PositiveCount]]):
    """Per-client expected benign row counts, keyed by the typed client id."""

    def count_for(self, client_id: ClientId) -> PositiveCount | None:
        return self.root.get(client_id)

    def __len__(self) -> int:
        return len(self.root)


class SplitConfig(BaseModel):
    """Per-client benign/attack role sizing for one dataset contract."""

    model_config = FrozenModel

    train_benign: PositiveCount
    reference_benign: PositiveCount
    mismatch_benign: PositiveCount
    calibration_benign: PositiveCount
    benign_guard: NonNegativeCount
    min_benign_test: PositiveCount
    attack_dev: PositiveCount
    min_attack_test: NonNegativeCount
    min_attack_test_per_group: PositiveCount

    @property
    def reservoir_size(self) -> SampleCount:
        return (
            self.reference_benign
            + self.mismatch_benign
            + self.calibration_benign
            + self.benign_guard
        )

    @property
    def minimum_benign_rows(self) -> SampleCount:
        return self.train_benign + self.reservoir_size + self.min_benign_test


class DatasetConfig(BaseModel):
    """Identity, feature contract, eligibility rule, and role sizing of one dataset."""

    model_config = FrozenModel

    id: DatasetId
    source_directory: Path = Path(".")
    feature_contract: DatasetFeatureContractId
    source_version: Version
    parser_version: Version
    feature_count: FeatureCount
    feature_names: tuple[FeatureName, ...] = ()
    expected_clients: PositiveCount | None = None
    expected_source_clients: PositiveCount | None = None
    minimum_clients: PositiveCount
    minimum_benign_rows: PositiveCount | None = None
    minimum_malicious_rows: PositiveCount | None = None
    split: SplitConfig
    calibration_seeds: tuple[CalibrationSeed, ...]
    primary_calibration_seed: CalibrationSeed
    expected_benign_counts: ExpectedBenignCounts

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if not self.calibration_seeds:
            raise ValueError("At least one calibration seed is required")
        if len(set(self.calibration_seeds)) != len(self.calibration_seeds):
            raise ValueError("Calibration seeds must be unique")
        if self.primary_calibration_seed not in self.calibration_seeds:
            raise ValueError("Primary calibration seed must be in calibration_seeds")
        if self.feature_names and len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("Feature names must be unique")

        if self.id is DatasetId.NBAIOT:
            if self.feature_contract is not DatasetFeatureContractId.NBAIOT_LOCKED_115:
                raise ValueError("N-BaIoT must use the locked 115-feature contract")
            if self.expected_clients is None or self.expected_clients < 1:
                raise ValueError("N-BaIoT must declare its nine-client contract")
            if len(self.expected_benign_counts) != self.expected_clients:
                raise ValueError("N-BaIoT expected-benign-count ledger must match the client count")

        elif self.id is DatasetId.DIAD:
            if self.feature_contract not in {
                DatasetFeatureContractId.DIAD_LOCKED_86,
                DatasetFeatureContractId.DIAD_TRAINING_NUMERIC_SAFE,
            }:
                raise ValueError("DIAD uses either the locked or training-derived feature contract")
            if self.expected_source_clients is None or self.expected_source_clients < 1:
                raise ValueError("DIAD must declare its source identity count")
            if self.minimum_benign_rows is None or self.minimum_malicious_rows is None:
                raise ValueError("DIAD eligibility counts must be declared")
            if self.feature_contract is DatasetFeatureContractId.DIAD_LOCKED_86:
                if self.feature_names:
                    raise ValueError("Locked DIAD feature names are owned by the dataset adapter")
            else:
                if not self.feature_names or len(self.feature_names) != self.feature_count:
                    raise ValueError(
                        "Training-derived DIAD requires a frozen feature-name list matching feature_count"
                    )

        elif self.id is DatasetId.SYNTHETIC:
            if self.feature_contract is not DatasetFeatureContractId.SYNTHETIC:
                raise ValueError("Synthetic experiments must use the synthetic feature contract")
        return self


class DatasetCatalogue(RootModel[dict[DatasetId, DatasetConfig]]):
    """Frozen dataset-contract registry keyed by dataset identity."""

    """All dataset contracts, keyed by dataset id."""

    def contract(self, dataset_id: DatasetId) -> DatasetConfig:
        try:
            return self.root[dataset_id]
        except KeyError as exc:
            raise ConfigurationError(f"Unknown dataset contract: {dataset_id.value}") from exc


AxisValue: TypeAlias = (
    int
    | float
    | SyntheticDistribution
    | ContaminationDirection
    | MultiplicityProcedure
    | CalibrationAssignmentMode
)

_AXIS_VALUE_TYPES: dict[ExperimentAxisId, type[AxisValue] | None] = {
    ExperimentAxisId.DISTRIBUTION: SyntheticDistribution,
    ExperimentAxisId.DIRECTION: ContaminationDirection,
    ExperimentAxisId.PROCEDURE: MultiplicityProcedure,
    ExperimentAxisId.ASSIGNMENT: CalibrationAssignmentMode,
}


class ParameterAxis(BaseModel):
    """An independent axis whose values may be crossed with other independent axes."""

    model_config = ConfigDict(frozen=True)

    id: ExperimentAxisId
    values: tuple[AxisValue, ...]

    @field_validator("values", mode="before")
    @classmethod
    def _coerce_values(cls, value: object, info: ValidationInfo) -> object:
        axis_type = _AXIS_VALUE_TYPES.get(ExperimentAxisId(str(info.data["id"])))
        if axis_type is None or not isinstance(value, list):
            return value
        return [axis_type(item) if isinstance(item, str) else item for item in value]

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if not self.values:
            raise ValueError(f"Experiment axis {self.id.value} must contain values")
        if len(set(self.values)) != len(self.values):
            raise ValueError(f"Experiment axis {self.id.value} contains duplicate values")
        return self


class ParameterSetting(BaseModel):
    """One locked axis value in an experiment cell."""

    model_config = ConfigDict(frozen=True)

    axis: ExperimentAxisId
    value: AxisValue


class ParameterCell(BaseModel):
    """A coupled combination that must be evaluated together, not Cartesian-crossed."""

    model_config = ConfigDict(frozen=True)

    settings: tuple[ParameterSetting, ...]

    @model_validator(mode="before")
    @classmethod
    def _coerce_settings(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        return {
            "settings": [
                ParameterSetting(axis=ExperimentAxisId(key), value=item)
                for key, item in value.items()
            ]
        }

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if not self.settings:
            raise ValueError("A parameter cell must contain at least one setting")
        axes = tuple(item.axis for item in self.settings)
        if len(set(axes)) != len(axes):
            raise ValueError("A parameter cell cannot assign the same axis twice")
        return self

    def value(self, axis: ExperimentAxisId) -> AxisValue:
        for setting in self.settings:
            if setting.axis is axis:
                return setting.value
        raise KeyError(axis.value)


class WorkloadExpectation(BaseModel):
    """Locked workload ledger for one experiment."""

    model_config = ConfigDict(frozen=True)

    monte_carlo_trials: NonNegativeCount = 0
    exact_cells: NonNegativeCount = 0
    detector_trainings: NonNegativeCount = 0


class ExperimentSpec(BaseModel):
    """One locked catalogue entry; the only description of what an experiment contains."""

    model_config = FrozenModel

    id: ExperimentId
    category: ExperimentType
    dataset: DatasetId | None = None
    datasets: tuple[DatasetId, ...] = ()
    detector: ProfileId | None = None
    training: ProfileId | None = None
    model_seeds: tuple[ModelSeed, ...] = ()
    calibration_seeds: tuple[CalibrationSeed, ...] = ()
    policies: tuple[PolicyId, ...] = ()
    axes: tuple[ParameterAxis, ...] = ()
    coupled_cells: tuple[ParameterCell, ...] = ()
    dependencies: tuple[ExperimentId, ...] = ()
    required_evidence: tuple[ArtifactType, ...] = ()
    workload: WorkloadExpectation = WorkloadExpectation()
    confirmatory: bool = False
    description: Description = ""

    @field_validator("calibration_seeds", mode="before")
    @classmethod
    def _coerce_calibration_seeds(cls, value: object) -> object:
        if isinstance(value, list):
            return [_CALIBRATION_SEED_ADAPTER.validate_python(item) for item in value]
        return value

    @field_validator("axes", mode="before")
    @classmethod
    def _coerce_axes(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        return [
            ParameterAxis(id=ExperimentAxisId(key), values=items) for key, items in value.items()
        ]

    @field_validator("datasets", mode="before")
    @classmethod
    def _coerce_datasets(cls, value: object) -> object:
        if isinstance(value, str):
            return (DatasetId(value),)
        if isinstance(value, list):
            return [DatasetId(item) for item in value]
        return value

    @field_validator("required_evidence", mode="before")
    @classmethod
    def _coerce_required_evidence(cls, value: object) -> object:
        if isinstance(value, list):
            return [ArtifactType(item) for item in value]
        return value

    @model_validator(mode="after")
    def _validate(self) -> Self:
        independent_axes = tuple(axis.id for axis in self.axes)
        if len(set(independent_axes)) != len(independent_axes):
            raise ValueError(f"Duplicate axis in {self.id.value}")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError(f"Duplicate dependency in {self.id.value}")
        if self.id in self.dependencies:
            raise ValueError(f"Experiment {self.id.value} cannot depend on itself")
        if len(set(self.policies)) != len(self.policies):
            raise ValueError(f"Duplicate policy in {self.id.value}")
        if len(set(self.calibration_seeds)) != len(self.calibration_seeds):
            raise ValueError(f"Duplicate calibration seed in {self.id.value}")
        datasets = self.datasets or ((self.dataset,) if self.dataset else ())
        synthetic_only = bool(datasets) and all(item is DatasetId.SYNTHETIC for item in datasets)
        if synthetic_only and self.detector is not None:
            raise ValueError(f"Synthetic experiment {self.id.value} must not bind a detector")
        if not synthetic_only and self.detector is None:
            raise ValueError(f"Real-data experiment {self.id.value} requires a detector profile")
        if not datasets:
            raise ValueError(f"Experiment {self.id.value} must declare a dataset")
        return self

    def axis(self, axis_id: ExperimentAxisId) -> ParameterAxis:
        for item in self.axes:
            if item.id is axis_id:
                return item
        raise KeyError(f"Experiment {self.id.value} has no independent {axis_id.value} axis")

    @property
    def all_datasets(self) -> tuple[DatasetId, ...]:
        return self.datasets or ((self.dataset,) if self.dataset else ())


class ExperimentCatalogue(RootModel[tuple[ExperimentSpec, ...]]):
    """The locked experiment registry; exactly one entry per ExperimentId."""

    @field_validator("root", mode="before")
    @classmethod
    def _coerce_root(cls, value: object) -> object:
        if isinstance(value, list):
            return [ExperimentSpec.model_validate(item) for item in value]
        return value

    @model_validator(mode="after")
    def _validate(self) -> Self:
        rows = self.root
        if len({row.id for row in rows}) != len(rows):
            raise ValueError("The experiment catalogue contains duplicate ids")
        if set(ExperimentId) != {row.id for row in rows}:
            missing = sorted(set(ExperimentId) - {row.id for row in rows})
            raise ValueError(f"The experiment catalogue is missing entries: {missing}")
        if len(rows) != len(ExperimentId):
            raise ValueError("The experiment catalogue must contain exactly one entry per id")
        return self

    def spec(self, experiment_id: ExperimentId) -> ExperimentSpec:
        for row in self.root:
            if row.id is experiment_id:
                return row
        raise KeyError(experiment_id.value)

    def all(self) -> tuple[ExperimentSpec, ...]:
        return self.root


class ExperimentConfig(BaseModel):
    """Fully resolved, validated execution configuration for one experiment cell."""

    model_config = FrozenModel

    id: ExperimentId
    protocol: ProtocolConfig
    dataset: DatasetConfig
    detector: DetectorConfig | None = None
    training: TrainingConfig | None = None
    randomness: RandomnessConfig
    statistics: StatisticsConfig
    policies: tuple[PolicyId, ...]
    outputs_root: Path = Path("outputs")
    preprocessed_root: Path = Path("data/preprocessed")

    @model_validator(mode="after")
    def _validate(self) -> Self:
        policy_optional = (
            self.dataset.id is DatasetId.SYNTHETIC
            or self.id is ExperimentId.COMPUTATIONAL_BENCHMARK
        )
        if not policy_optional and not self.policies:
            raise ValueError("Real-data experiments require at least one policy")
        if len(set(self.policies)) != len(self.policies):
            raise ValueError("Policies must be unique")
        if self.id is ExperimentId.SECOND_DETECTOR:
            if self.detector is None or self.detector.id is not DetectorId.DEEP_SVDD:
                raise ValueError("Second-detector experiment requires Deep-SVDD")
        if self.id is ExperimentId.EXTERNAL_DIAD:
            if self.dataset.id is not DatasetId.DIAD:
                raise ValueError("External validation requires DIAD")
            if self.dataset.feature_contract is not DatasetFeatureContractId.DIAD_LOCKED_86:
                raise ValueError(
                    "Confirmatory DIAD external validation requires the locked 86-feature contract"
                )
        if self.id is ExperimentId.DIAD_FEATURE_SENSITIVITY:
            if self.dataset.id is not DatasetId.DIAD:
                raise ValueError("The training-schema-derived feature experiment requires DIAD")
            if (
                self.dataset.feature_contract
                is not DatasetFeatureContractId.DIAD_TRAINING_NUMERIC_SAFE
            ):
                raise ValueError(
                    "The feature experiment requires a frozen training-schema-derived feature contract"
                )
        elif self.dataset.feature_contract is DatasetFeatureContractId.DIAD_TRAINING_NUMERIC_SAFE:
            raise ValueError(
                "The training-schema-derived DIAD feature contract is exclusive to the "
                "feature-sensitivity experiment"
            )
        return self

    def serialized_payload(self) -> str:
        """Stable scientific configuration, independent of output location."""
        payload = self.model_dump(mode="json", exclude={"outputs_root", "preprocessed_root"})
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @property
    def config_hash(self) -> Sha256:
        return hashlib.sha256(self.serialized_payload().encode("utf-8")).hexdigest()

    @property
    def data_spec_hash(self) -> Sha256:
        """Hash only inputs that determine the seed-independent prepared dataset cache."""
        dataset_payload = self.dataset.model_dump(
            mode="json",
            exclude={"calibration_seeds", "primary_calibration_seed"},
        )
        return _sha256_json( #TODO: use pydantic's `model_dump_json` for a more direct approach
            {
                "dataset": dataset_payload,
                "attack_split_seed": self.randomness.attack_split_seed,
            }
        )

    @property
    def training_spec_hash(self) -> Sha256:
        """Hash the exact detector-training specification, excluding policy/protocol axes."""
        if self.detector is None or self.training is None:
            return self.data_spec_hash
        return _sha256_json( #TODO: use pydantic's `model_dump_json` for a more direct approach
            {
                "data_spec_hash": self.data_spec_hash,
                "detector": self.detector.model_dump(mode="json"),
                "training": self.training.model_dump(mode="json"),
            }
        )


def load_yaml_mapping(path: Path) -> dict[str, JsonValue]:
    """Load one configuration document before validation."""
    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Configuration file does not exist: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError(f"Configuration root must be a mapping: {path}")
    return {str(key): value for key, value in raw.items()}


def load_study(path: Path = Path("config/study.yaml")) -> StudyConfig:
    """Load and validate the study configuration."""
    return StudyConfig.model_validate(load_yaml_mapping(path))


def load_dataset_registry(path: Path = Path("config/datasets.yaml")) -> DatasetCatalogue:
    """Load and validate the dataset contract registry."""
    root = load_yaml_mapping(path)
    datasets = root.get("datasets")
    if not isinstance(datasets, dict):
        raise ConfigurationError("datasets.yaml must contain a 'datasets' mapping")
    return DatasetCatalogue.model_validate(datasets)


def load_experiment_catalogue(path: Path = Path("config/experiments.yaml")) -> ExperimentCatalogue:
    """Load and validate the experiment catalogue."""
    root = load_yaml_mapping(path)
    experiments = root.get("experiments")
    if not isinstance(experiments, list):
        raise ConfigurationError("experiments.yaml must contain an 'experiments' list")
    return ExperimentCatalogue.model_validate(experiments)


def resolve_experiment_config(
    spec: ExperimentSpec,
    study: StudyConfig,
    datasets: DatasetCatalogue,
    *,
    dataset_id: DatasetId | None = None,
) -> ExperimentConfig:
    """Bind one catalogue entry to its dataset contract and study profiles."""
    chosen = dataset_id or (spec.datasets[0] if spec.datasets else spec.dataset)
    if chosen is None:
        raise ConfigurationError(f"Experiment {spec.id.value} declares no dataset")
    dataset = datasets.contract(chosen)
    if spec.calibration_seeds:
        dataset = dataset.model_copy(update={"calibration_seeds": spec.calibration_seeds})
    detector = study.detector_profile(spec.detector) if spec.detector else None
    training = study.training_profile(spec.training) if spec.training else None
    randomness = study.randomness
    if spec.model_seeds:
        randomness = RandomnessConfig(
            model_seeds=spec.model_seeds,
            attack_split_seed=study.randomness.attack_split_seed,
            synthetic_seed=study.randomness.synthetic_seed,
        )
    return ExperimentConfig(
        id=spec.id,
        protocol=study.protocol,
        dataset=dataset,
        detector=detector,
        training=training,
        randomness=randomness,
        statistics=study.statistics,
        policies=spec.policies,
        outputs_root=study.paths.outputs_root,
        preprocessed_root=study.paths.preprocessed_root,
    )


class Study:
    """One loaded configuration set with catalogue lookup and resolution."""

    def __init__(
        self,
        study_config: StudyConfig,
        datasets: DatasetCatalogue,
        catalogue: ExperimentCatalogue,
    ) -> None:
        self.study_config = study_config
        self.datasets = datasets
        self.catalogue = catalogue

    @classmethod
    def load(
        cls,
        study_path: Path = Path("config/study.yaml"),
        datasets_path: Path = Path("config/datasets.yaml"),
        experiments_path: Path = Path("config/experiments.yaml"),
    ) -> Study:
        return cls(
            study_config=load_study(study_path),
            datasets=load_dataset_registry(datasets_path),
            catalogue=load_experiment_catalogue(experiments_path),
        )

    def spec(self, experiment_id: ExperimentId) -> ExperimentSpec:
        return self.catalogue.spec(experiment_id)

    @property
    def paths(self) -> StudyPaths:
        return self.study_config.paths

    @property
    def campaign_id(self) -> CampaignId:
        return self.study_config.campaign_id

    def resolve(
        self,
        experiment_id: ExperimentId,
        *,
        dataset_id: DatasetId | None = None,
    ) -> ExperimentConfig:
        return resolve_experiment_config(
            self.catalogue.spec(experiment_id),
            self.study_config,
            self.datasets,
            dataset_id=dataset_id,
        )


def validate_experiment_config(config: ExperimentConfig) -> None:
    """Cross-component invariants that span resolved configuration."""
    if config.dataset.id is not DatasetId.SYNTHETIC and not config.randomness.model_seeds:
        raise ConfigurationError("Real-data experiments require at least one model seed")
    if config.detector is not None and config.training is not None:
        if config.detector.id is DetectorId.DEEP_SVDD and config.training.local_epochs <= 0:
            raise ConfigurationError("Deep-SVDD requires positive local epochs")
    registered = set(PolicyId)
    if any(policy not in registered for policy in config.policies):
        raise ConfigurationError("Experiment contains an unregistered policy")
    if len(registered) != 12:
        raise ConfigurationError("Policy catalogue must contain exactly 12 protocol policies")
