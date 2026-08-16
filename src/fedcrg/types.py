"""Shared immutable value types, enums, and error classes used across the package."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

TIGHT_TOLERANCE = 1.0e-12
EXACT_BETA_TOLERANCE = 1.0e-10
COARSE_TOLERANCE = 1.0e-9

NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
AdamBeta = Annotated[float, Field(gt=0.0, lt=1.0)]

ModelSeed = NonNegativeInt
CalibrationSeed = NonNegativeInt
AnalysisSeed = NonNegativeInt
BootstrapSeed = NonNegativeInt
ReplicateCount = PositiveInt
RepetitionCount = PositiveInt

Alpha = Annotated[float, Field(gt=0.0, lt=1.0)]
Rho = NonNegativeFloat
Fpr = Probability
Tpr = Probability
PValue = Probability
CoverageProbability = Probability
ConfidenceLevel = Annotated[float, Field(gt=0.0, lt=1.0)]
Assurance = ConfidenceLevel
UtilityMargin = NonNegativeFloat
Tolerance = PositiveFloat
MetricDifference = Annotated[float, Field()]

LearningRate = PositiveFloat
OptimizerEpsilon = PositiveFloat
XavierGain = PositiveFloat
# Bounds collapse to a single point deliberately: weight decay is locked at 0 and client
# participation is locked at full (1.0) for the duration of the study.
WeightDecay = Annotated[float, Field(ge=0.0, le=0.0)]
ClientFraction = Annotated[float, Field(ge=1.0, le=1.0)]
RoundCount = PositiveInt
RoundIndex = NonNegativeInt
EpochCount = PositiveInt
BatchSize = PositiveInt
FeatureCount = PositiveInt
LayerWidth = PositiveInt
EmbeddingDimension = PositiveInt
SampleCount = PositiveInt
PositiveCount = PositiveInt
NonNegativeCount = NonNegativeInt
RowCount = NonNegativeInt
ByteCount = NonNegativeInt
ExceedanceCount = NonNegativeInt
BlockCount = PositiveInt
WarmupCount = PositiveInt
BootstrapReplicateCount = PositiveInt
CandidateCount = PositiveInt
Dimension = PositiveInt
Spacing = NonNegativeFloat
Threshold = Annotated[float, Field()]
Score = Annotated[float, Field()]
Fraction = Probability
ProtocolConstantValue = Fraction | Assurance | PositiveCount | LearningRate | ClientFraction
ContaminationFraction = Probability
TrueFpr = Probability

Percentage = Annotated[float, Field(ge=0.0, le=100.0)]

Position = NonNegativeInt

ParameterCount = PositiveInt

RngSeed = Annotated[int, Field(ge=0, le=0xFFFFFFFFFFFFFFFF)]

Loss = Annotated[float, Field(ge=0.0)]

Correlation = Annotated[float, Field(ge=-1.0, le=1.0)]

Version = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._+-]*$"),
]

Description = Annotated[str, StringConstraints(min_length=0, max_length=512)]

type JsonValue = (
    str | int | float | bool | None | tuple[JsonValue, ...] | list[JsonValue] | dict[str, JsonValue]
)

LogLevel = Annotated[
    str,
    StringConstraints(pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"),
]

Timestamp = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"),
]

Duration = Annotated[float, Field(ge=0.0)]

Identifier = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=128,
        strip_whitespace=True,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    ),
]
PathString = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=512,
        strip_whitespace=True,
    ),
]
DeviceName = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=128,
        strip_whitespace=True,
    ),
]
MacAddress = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=64,
        strip_whitespace=True,
    ),
]
ClientId = Identifier
AttackGroupId = Identifier
ProfileId = Identifier
FeatureName = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=128,
        strip_whitespace=True,
        pattern=r"^[a-z0-9][a-z0-9_]*$",
    ),
]


class CampaignStage(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class InformationRegime(StrEnum):
    BENIGN_ONLY = "benign_only"
    SUPERVISED_DEVELOPMENT = "supervised_development"
    FINAL_TEST_ORACLE = "final_test_oracle"


class PreparedColumn(StrEnum):
    ROW_ID = "row_id"
    ROLE = "role"
    LABEL = "label"
    ATTACK_GROUP = "attack_group"
    SOURCE_FILE = "source_file"
    SOURCE_ROW_INDEX = "source_row_index"
    CAPTURE_TIME = "capture_time"


class DecisionSource(StrEnum):
    REFERENCE = "reference"
    LOCAL = "local"
    ORACLE = "oracle"
    UNDEFINED = "undefined"


Metric = Annotated[float, Field()]

PlanKey = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=256,
        strip_whitespace=True,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._=|:-]*$",
    ),
]
RunId = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=256,
        strip_whitespace=True,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    ),
]
Sha256 = Annotated[
    str,
    StringConstraints(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"),
]
RowId = Sha256
ArtifactDigest = Sha256
GitCommit = Annotated[
    str,
    StringConstraints(
        min_length=7,
        max_length=40,
        strip_whitespace=True,
        pattern=r"^[0-9a-f]{7,40}$",
    ),
]
SourceFileId = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]


class OperatingBand(BaseModel):
    """A closed [lower, upper] false-positive-rate interval."""

    model_config = ConfigDict(frozen=True)

    lower: Fpr
    upper: Fpr

    @model_validator(mode="after")
    def _validate(self) -> OperatingBand:
        if self.lower >= self.upper:
            raise ValueError("Operating band must satisfy lower < upper")
        return self

    def contains(self, value: Fpr) -> bool:
        return self.lower <= value <= self.upper


class ConfidenceInterval(BaseModel):
    """A closed [lower, upper] confidence interval over a false-positive rate."""

    model_config = ConfigDict(frozen=True)

    lower: Fpr
    upper: Fpr

    @model_validator(mode="after")
    def _validate(self) -> ConfidenceInterval:
        if self.lower > self.upper:
            raise ValueError("Confidence interval must satisfy lower <= upper")
        return self


@dataclass(frozen=True, slots=True)
class BinomialCounts:
    """An exceedance count x out of n trials, with 0 <= x <= n enforced."""

    x: ExceedanceCount
    n: PositiveCount

    def __post_init__(self) -> None:
        if self.x > self.n:
            raise ValueError("Require 0 <= x <= n")

    @property
    def exceedance_rate(self) -> Fpr:
        return self.x / self.n


@dataclass(frozen=True, slots=True)
class ClassMoments:
    """Mean and standard deviation of a score distribution."""

    mean: Score
    std: Spacing


class ProtocolId(StrEnum):
    FEDCRG = "fedcrg"


class DatasetId(StrEnum):
    NBAIOT = "nbaiot"
    DIAD = "diad"
    SYNTHETIC = "synthetic"


class DatasetFeatureContractId(StrEnum):
    NBAIOT_LOCKED_115 = "nbaiot_locked_115"
    DIAD_LOCKED_86 = "diad_locked_86"
    DIAD_TRAINING_NUMERIC_SAFE = "diad_training_numeric_safe"
    SYNTHETIC = "synthetic"


class DetectorId(StrEnum):
    AUTOENCODER = "autoencoder"
    DEEP_SVDD = "deep_svdd"


class DataRole(StrEnum):
    TRAIN = "train"
    RESERVOIR = "reservoir"
    REFERENCE = "reference"
    MISMATCH = "mismatch"
    CALIBRATION = "calibration"
    BENIGN_GUARD = "benign_guard"
    BENIGN_TEST = "benign_test"
    ATTACK_DEV = "attack_dev"
    ATTACK_TEST = "attack_test"


class CalibrationAssignmentMode(StrEnum):
    SEEDED_PERMUTATION = "seeded_permutation"
    SOURCE_ORDER = "source_order"


class CalibrationReadinessState(StrEnum):
    READY = "READY"
    NOT_READY = "NOT_READY"


class MismatchOutcome(StrEnum):
    LOW = "LOW_MISMATCH"
    HIGH = "HIGH_MISMATCH"
    # An absence of demonstrated mismatch, not a finding that the reference threshold is correct.
    NO_MATERIAL_DIFFERENCE = "NO_MATERIAL_MISMATCH_DEMONSTRATED"
    INSUFFICIENT_EVIDENCE = "GATE_B_INSUFFICIENT"


class DecisionState(StrEnum):
    # Reference threshold stays in force because no material mismatch was demonstrated — this is
    # an absence of evidence, not evidence that the reference threshold is correct.
    REFERENCE_RETAINED = "NO_MATERIAL_MISMATCH_DEMONSTRATED"
    PERSONALIZED = "LOCAL_PERSONALIZE"
    CALIBRATION_DEFICIT = "CALIBRATION_DEFICIT"
    MISMATCH_EVIDENCE_INSUFFICIENT = "GATE_B_INSUFFICIENT"
    ASSUMPTION_VIOLATION = "CALIBRATION_ASSUMPTION_VIOLATION"


class ThresholdSource(StrEnum):
    REFERENCE = "reference"
    LOCAL_CALIBRATION = "local_calibration"


class DecisionReason(StrEnum):
    NO_MATERIAL_DIFFERENCE = "NO_MATERIAL_MISMATCH_DEMONSTRATED"
    LOCAL_PERSONALIZATION_ADMITTED = "local_personalization_admitted"
    CALIBRATION_NOT_READY = "calibration_not_ready"
    INSUFFICIENT_MISMATCH_EVIDENCE = "insufficient_mismatch_evidence"
    CALIBRATION_TIE = "calibration_tie"


class FailureCode(StrEnum):
    DATASET_COUNT_MISMATCH = "DATASET_COUNT_MISMATCH"
    NBAIOT_ATTACK_BUDGET_FAIL = "NBAIOT_ATTACK_BUDGET_FAIL"
    DIAD_DEVICE_COUNT_SOURCE_MISMATCH = "DIAD_DEVICE_COUNT_SOURCE_MISMATCH"
    ID_INVALID = "ID_INVALID"
    FEATURE_MISSING = "FEATURE_MISSING"
    FINITE_RATE_FAIL = "FINITE_RATE_FAIL"
    BENIGN_COUNT_LT_7800 = "BENIGN_COUNT_LT_7800"
    MALICIOUS_COUNT_LT_1000 = "MALICIOUS_COUNT_LT_1000"
    ATTACK_DEV_CAPACITY_LT_500 = "ATTACK_DEV_CAPACITY_LT_500"
    EXTERNAL_DATASET_INSUFFICIENT_CLIENTS = "EXTERNAL_DATASET_INSUFFICIENT_CLIENTS"
    FEATURE_SCHEMA_MISMATCH = "FEATURE_SCHEMA_MISMATCH"
    DIAD_FEATURE_FINITE_RATE_FAIL = "DIAD_FEATURE_FINITE_RATE_FAIL"
    ROLE_OVERLAP = "ROLE_OVERLAP"
    LABEL_LEAKAGE = "LABEL_LEAKAGE"
    SCORE_CACHE_HASH_MISMATCH = "SCORE_CACHE_HASH_MISMATCH"
    READINESS_NOT_READY = "GATE_A_NOT_READY"
    CALIBRATION_DEFICIT = "CALIBRATION_DEFICIT"
    MISMATCH_EVIDENCE_INSUFFICIENT = "GATE_B_INSUFFICIENT"
    DIRECTION_CONTRADICTION = "GATE_B_DIRECTION_CONTRADICTION"
    CALIBRATION_ASSUMPTION_VIOLATION = "CALIBRATION_ASSUMPTION_VIOLATION"
    SUMMARY_STATISTIC_COMPARATOR_UNDEFINED = "LARIDI_STYLE_UNDEFINED"
    METRIC_UNDEFINED = "METRIC_UNDEFINED"
    NONFINITE_SCORE = "NONFINITE_SCORE"
    TRAINING_NUMERICAL_FAILURE = "TRAINING_NUMERICAL_FAILURE"
    ONE_SIDED_BAND_BY_DESIGN = "ONE_SIDED_BAND_BY_DESIGN"
    DATA_DRIFT_STRESS = "DATA_DRIFT_STRESS"
    NONDETERMINISTIC_PARITY_FAIL = "NONDETERMINISTIC_PARITY_FAIL"


class EligibilityStatus(StrEnum):
    ELIGIBLE = "eligible"
    EXCLUDED = "excluded"


class ActivationId(StrEnum):
    TANH = "tanh"


class ComputeDeviceId(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"


class DeepSvddCenterMode(StrEnum):
    EQUAL_MEAN_OF_CLIENT_INITIAL_EMBEDDINGS = "equal_mean_of_client_initial_embeddings"


class ChronologyStatus(StrEnum):
    VERIFIED = "verified"
    SOURCE_ORDER_ONLY = "source_order_only"


class AggregationId(StrEnum):
    EQUAL_CLIENT_MEAN = "equal_client_mean"


class OptimizerId(StrEnum):
    ADAM = "adam"


class PolicyId(StrEnum):
    REFERENCE_QUANTILE = "reference_quantile"
    GLOBAL_QUANTILE = "global_quantile"
    LOCAL_QUANTILE = "local_quantile"
    READINESS_ONLY = "readiness_only"
    MISMATCH_ONLY = "mismatch_only"
    SHRINKAGE = "shrinkage"
    THREE_SIGMA = "three_sigma"
    DEV_F1_SELECT = "development_f1_selection"
    SUMMARY_STATISTIC_SELECT = "summary_statistic"
    SUPERVISED_F1 = "supervised_f1"
    ORACLE_TEST = "oracle_test"
    FEDCRG = "fedcrg"


class CampaignStatusValue(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class PolicyEvaluationStatus(StrEnum):
    EVALUATED = "evaluated"
    UNDEFINED = "undefined"


class ExperimentId(StrEnum):
    READINESS_THEOREM = "readiness_theorem"
    TARGET_FPR_SYNTHETIC = "target_fpr_synthetic"
    TEMPORAL_DEPENDENCE = "temporal_dependence"
    CALIBRATION_SHIFT = "calibration_shift"
    CALIBRATION_CONTAMINATION = "calibration_contamination"
    MISMATCH_POWER = "mismatch_power"
    PRIMARY_NBAIOT = "primary_nbaiot"
    READINESS_SAMPLE_SIZE = "readiness_sample_size"
    MISMATCH_SAMPLE_SIZE = "mismatch_sample_size"
    TOLERANCE_SENSITIVITY = "tolerance_sensitivity"
    TARGET_FPR_REAL = "target_fpr_real"
    ASSURANCE_SENSITIVITY = "assurance_sensitivity"
    MULTIPLICITY_SENSITIVITY = "multiplicity_sensitivity"
    SOURCE_ORDER_TEST = "source_order_test"
    REAL_CONTAMINATION = "real_contamination"
    EXTERNAL_DIAD = "external_diad"
    SECOND_DETECTOR = "second_detector"
    SOURCE_ORDER_CALIBRATION = "source_order_calibration"
    COMPUTATIONAL_BENCHMARK = "computational_benchmark"
    DIAD_FEATURE_SENSITIVITY = "diad_feature_sensitivity"


class ExperimentAxisId(StrEnum):
    DISTRIBUTION = "distribution"
    CALIBRATION_N = "calibration_n"
    REPETITIONS = "repetitions"
    ALPHA = "alpha"
    PHI = "phi"
    MEAN_SHIFT = "mean_shift"
    FRACTION = "fraction"
    DIRECTION = "direction"
    MISMATCH_N = "mismatch_n"
    TRUE_FPR = "true_fpr"
    RHO = "rho"
    READINESS_ASSURANCE = "readiness_assurance"
    PROCEDURE = "procedure"
    BLOCKS = "blocks"
    ASSIGNMENT = "assignment"
    WARMUPS = "warmups"


class SyntheticDistribution(StrEnum):
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    GAMMA_SHAPE_2 = "gamma2"
    NORMAL_MIXTURE = "normal_mixture"


class ContaminationDirection(StrEnum):
    HIGH = "high"
    LOW = "low"


class MultiplicityProcedure(StrEnum):
    BONFERRONI_READINESS = "bonferroni_readiness"
    BONFERRONI_MISMATCH = "bonferroni_mismatch"
    HOLM_DIRECTIONAL = "holm_directional"


class ExperimentType(StrEnum):
    SYNTHETIC = "synthetic"
    PRIMARY = "primary"
    SENSITIVITY = "sensitivity"
    ROBUSTNESS = "robustness"
    EXTERNAL_VALIDATION = "external_validation"
    BENCHMARK = "benchmark"


class ExperimentStatus(StrEnum):
    PENDING = "pending"
    VALIDATING = "validating"
    READY = "ready"
    RUNNING = "running"
    VERIFYING = "verifying"
    COMPLETE = "complete"
    FAILED = "failed"
    BLOCKED = "blocked"
    INVALID = "invalid"


class CompletionState(StrEnum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    EXECUTION_INCOMPLETE = "execution_incomplete"
    EVIDENCE_INCOMPLETE = "evidence_incomplete"
    FAILED = "failed"
    BLOCKED = "blocked"
    INVALID = "invalid"
    STALE = "stale"
    ANALYSIS_COMPLETE = "analysis_complete"
    FULLY_PASSED = "fully_passed"


class ExecutionOutcome(StrEnum):
    ALREADY_PASSED = "already_passed"
    COMPLETED = "completed"
    FAILED = "failed"


class ResultsGenerationStatus(StrEnum):
    ALREADY_GENERATED = "already_generated"
    BUILT = "built"
    REBUILT = "rebuilt"


class ExperimentApplicability(StrEnum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    INAPPLICABLE = "inapplicable"


class CacheRole(StrEnum):
    NONE = "none"
    OWNER = "owner"
    CONSUMER = "consumer"


class SharedCacheKind(StrEnum):
    NONE = "none"
    NBAIOT_AUTOENCODER = "nbaiot_autoencoder"
    NBAIOT_DEEP_SVDD = "nbaiot_deep_svdd"
    DIAD_AUTOENCODER = "diad_autoencoder"


class JsonResultKind(StrEnum):
    SYNTHETIC_ENVELOPE = "synthetic_envelope"
    BENCHMARK_REPORT = "benchmark_report"
    EMPIRICAL_RESULTS = "empirical_results"


class ResultFormat(StrEnum):
    JSON = "json"
    CSV = "csv"
    FIGURE = "figure"
    REPORT = "report"


class ArtifactType(StrEnum):
    RESOLVED_CONFIG = "resolved_config"
    DATASET_MANIFEST = "dataset_manifest"
    ELIGIBILITY_MANIFEST = "eligibility_manifest"
    SPLIT_MANIFEST = "split_manifest"
    PREPROCESSING_MANIFEST = "preprocessing_manifest"
    TRAINING_MANIFEST = "training_manifest"
    MODEL = "model"
    SCORE_MANIFEST = "score_manifest"
    THRESHOLD_RECORDS = "threshold_records"
    METRICS = "metrics"
    TABLE = "table"
    FIGURE = "figure"
    REPORT = "report"
    VERIFICATION = "verification"
    ANALYSIS_RESULT = "analysis_result"
    JSON_RESULT = "json_result"
    RESULTS_BUNDLE = "results_bundle"


class SupervisedClassLabel(IntEnum):
    BENIGN = 0
    ATTACK = 1


class FedCRGError(Exception):
    """Base class for errors raised by this package."""


class ConfigurationError(FedCRGError):
    """Raised for invalid or unresolvable configuration."""


class DataIntegrityError(FedCRGError):
    """Raised when prepared data fails an integrity check."""


class ImmutableRunError(FedCRGError):
    """Raised when code attempts to modify an artifact that must stay immutable once written."""
