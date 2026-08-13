"""Closed domain identities used across the FedCRG implementation."""

from enum import StrEnum


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
    NO_MATERIAL_DIFFERENCE = "NO_MATERIAL_MISMATCH_DEMONSTRATED"
    INSUFFICIENT_EVIDENCE = "GATE_B_INSUFFICIENT"


class DecisionState(StrEnum):
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


class ScoreDtype(StrEnum):
    FLOAT64 = "float64"


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
    REFERENCE_QUANTILE = "REF-Q99-R"
    GLOBAL_QUANTILE = "GLOBAL-Q99-FULL"
    LOCAL_QUANTILE = "LOCAL-Q99-FULL"
    READINESS_ONLY = "GATE-A-ONLY"
    MISMATCH_ONLY = "GATE-B-ONLY"
    SHRINKAGE = "SHRINKAGE"
    THREE_SIGMA = "FEDDETECT-3SIGMA"
    DEV_F1_SELECT = "DEV-F1-LG-SELECT"
    SUMMARY_STATISTIC_SELECT = "LARIDI-STYLE-SS"
    SUPERVISED_F1 = "SUP-F1-1000"
    ORACLE_TEST = "ORACLE-TEST"
    FEDCRG = "FEDCRG"


class PolicyEvaluationStatus(StrEnum):
    EVALUATED = "evaluated"
    UNDEFINED = "undefined"


class MetricId(StrEnum):
    FPR = "fpr"
    TPR = "tpr"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    BALANCED_ACCURACY = "balanced_accuracy"
    AUROC = "auroc"
    AUPRC = "auprc"
    BAND_ERROR = "band_error"
    MEBE = "mebe"
    HIGH_EXCESS = "high_excess"
    BAND_VIOLATION_RATE = "band_violation_rate"
    MAFE = "mafe"
    ATTACK_BALANCED_TPR = "attack_balanced_tpr"
    ATTACK_BALANCED_MACRO_TPR = "attack_balanced_macro_tpr"
    MACRO_TPR = "macro_tpr"
    WORST_CLIENT_TPR = "worst_client_tpr"
    WORST_CLIENT_ATTACK_BALANCED_TPR = "worst_client_attack_balanced_tpr"


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


class ExperimentCode(StrEnum):
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    S4 = "S4"
    S5 = "S5"
    S6 = "S6"
    R1 = "R1"
    R2 = "R2"
    R3 = "R3"
    R4 = "R4"
    R5 = "R5"
    R6 = "R6"
    R7 = "R7"
    R8 = "R8"
    R9 = "R9"
    R10 = "R10"
    R11 = "R11"
    R12 = "R12"
    R13 = "R13"
    R14 = "R14"


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


class ClaimLevel(StrEnum):
    METHOD_BENEFIT = "method_benefit"
    DATASET_LIMITED_BENEFIT = "dataset_limited_benefit"
    CHARACTERIZATION = "characterization"
    INVALID = "invalid"
