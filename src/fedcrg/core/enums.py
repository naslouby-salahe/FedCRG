"""Closed domain identities used across FedCRG."""

from enum import StrEnum


class DatasetId(StrEnum):
    NBAIOT = "nbaiot"
    DIAD = "diad"
    SYNTHETIC = "synthetic"


class DetectorId(StrEnum):
    AUTOENCODER = "autoencoder"
    DEEP_SVDD = "deep_svdd"


class DataRole(StrEnum):
    TRAIN = "train"
    REFERENCE = "reference"
    MISMATCH = "mismatch"
    CALIBRATION = "calibration"
    BENIGN_GUARD = "benign_guard"
    BENIGN_TEST = "benign_test"
    ATTACK_DEV = "attack_dev"
    ATTACK_TEST = "attack_test"


class CalibrationReadinessState(StrEnum):
    READY = "ready"
    NOT_READY = "not_ready"


class MismatchOutcome(StrEnum):
    LOW = "low"
    HIGH = "high"
    NO_MATERIAL_DIFFERENCE = "no_material_difference"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class DecisionState(StrEnum):
    REFERENCE_RETAINED = "reference_retained"
    PERSONALIZED = "personalized"
    CALIBRATION_DEFICIT = "calibration_deficit"
    MISMATCH_EVIDENCE_INSUFFICIENT = "mismatch_evidence_insufficient"
    ASSUMPTION_VIOLATION = "assumption_violation"


class ThresholdSource(StrEnum):
    REFERENCE = "reference"
    LOCAL_CALIBRATION = "local_calibration"


class DecisionReason(StrEnum):
    NO_MATERIAL_DIFFERENCE = "no_material_difference"
    LOCAL_PERSONALIZATION_ADMITTED = "local_personalization_admitted"
    CALIBRATION_NOT_READY = "calibration_not_ready"
    INSUFFICIENT_MISMATCH_EVIDENCE = "insufficient_mismatch_evidence"
    CALIBRATION_TIE = "calibration_tie"


class ActivationId(StrEnum):
    TANH = "tanh"
    RELU = "relu"


class ComputeDeviceId(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"


class ScoreDtype(StrEnum):
    FLOAT64 = "float64"


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
    DEV_F1_SELECT = "dev_f1_select"
    SUMMARY_STATISTIC_SELECT = "summary_statistic_select"
    SUPERVISED_F1 = "supervised_f1"
    ORACLE_TEST = "oracle_test"
    FEDCRG = "fedcrg"


class PolicyEvaluationStatus(StrEnum):
    EVALUATED = "evaluated"
    UNDEFINED = "undefined"


class MetricId(StrEnum):
    FPR = "fpr"
    TPR = "tpr"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    AUROC = "auroc"
    AUPRC = "auprc"
    BAND_ERROR = "band_error"
    HIGH_EXCESS = "high_excess"
    BAND_VIOLATION = "band_violation"
    ABSOLUTE_FPR_ERROR = "absolute_fpr_error"
    ATTACK_BALANCED_TPR = "attack_balanced_tpr"


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


class ArtifactType(StrEnum):
    RESOLVED_CONFIG = "resolved_config"
    DATASET_MANIFEST = "dataset_manifest"
    SPLIT_MANIFEST = "split_manifest"
    TRAINING_MANIFEST = "training_manifest"
    MODEL = "model"
    SCORE_MANIFEST = "score_manifest"
    DECISIONS = "decisions"
    METRICS = "metrics"
    TABLE = "table"
    FIGURE = "figure"
    REPORT = "report"
    VERIFICATION = "verification"
