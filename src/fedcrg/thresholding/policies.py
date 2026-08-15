from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from pydantic import BaseModel, ConfigDict

from fedcrg.thresholding.readiness import (
    CalibrationReadinessState,
    ClientEvaluationResult,
    MismatchOutcome,
    Threshold,
)
from fedcrg.thresholding.metrics import band_error, confusion_matrix, f1, tpr
from fedcrg.config import ExperimentConfig, ProtocolConfig, StatisticsConfig
from fedcrg.types import (
    Alpha,
    AttackGroupId,
    ByteCount,
    CandidateCount,
    ClientId,
    ClassMoments,
    FailureCode,
    Identifier,
    InformationRegime,
    Metric,
    NonNegativeCount,
    NonNegativeInt,
    OperatingBand,
    PolicyId,
    PositiveCount,
    SampleCount,
    Score,
    SupervisedClassLabel,
    Tpr,
)

Frozen = ConfigDict(frozen=True)

_FLOAT64_BYTES = 8
_INT64_BYTES = 8


class PolicyTrafficLedgerRow(BaseModel):
    model_config = Frozen

    policy: PolicyId
    upload_bytes_per_client: ByteCount


def threshold_policy_communication(
    config: ExperimentConfig,
    client_count: PositiveCount,
) -> tuple[PolicyTrafficLedgerRow, ...]:
    if client_count <= 0:
        raise ValueError("Policy traffic accounting requires a positive client count")
    split = config.dataset.split
    full_policy_budget = split.reference_benign + split.mismatch_benign + split.calibration_benign
    reference_payload = split.reference_benign * _FLOAT64_BYTES
    full_budget_payload = full_policy_budget * _FLOAT64_BYTES
    candidates = config.statistics.supervised_threshold_candidates
    moment_payload = 2 * (_INT64_BYTES + 2 * _FLOAT64_BYTES)
    candidate_payload = candidates * _FLOAT64_BYTES

    def upload_bytes(kind: PolicyUploadKind) -> ByteCount:
        if kind is PolicyUploadKind.NONE:
            return 0
        if kind is PolicyUploadKind.REFERENCE_SCORES:
            return reference_payload
        if kind is PolicyUploadKind.FULL_POLICY_BUDGET:
            return full_budget_payload
        if kind is PolicyUploadKind.SUMMARY_MOMENTS_AND_CANDIDATES:
            return moment_payload + candidate_payload
        if kind is PolicyUploadKind.SUPERVISED_CANDIDATES:
            return candidate_payload
        raise RuntimeError(f"Unhandled policy upload payload kind: {kind}")

    rows = tuple(
        PolicyTrafficLedgerRow(
            policy=policy,
            upload_bytes_per_client=upload_bytes(POLICIES[policy].upload_payload),
        )
        for policy in PolicyId
    )
    if len(rows) != len(PolicyId):
        raise RuntimeError("Policy traffic ledger must contain exactly one row per policy")
    return rows


class BenignPolicyEvidence:
    def __init__(
        self,
        client_id: ClientId,
        reference_scores: np.ndarray,
        mismatch_scores: np.ndarray,
        calibration_scores: np.ndarray,
        evaluation: ClientEvaluationResult,
    ) -> None:
        self.client_id = client_id
        self.reference_scores = _finite_vector(reference_scores, "reference_scores")
        self.mismatch_scores = _finite_vector(mismatch_scores, "mismatch_scores")
        self.calibration_scores = _finite_vector(calibration_scores, "calibration_scores")
        self.evaluation = evaluation

    @property
    def full_policy_budget(self) -> np.ndarray:
        return np.concatenate(
            (self.reference_scores, self.mismatch_scores, self.calibration_scores)
        )


class SupervisedDevelopmentEvidence:
    def __init__(
        self,
        benign: BenignPolicyEvidence,
        benign_guard_scores: np.ndarray,
        attack_dev_scores: np.ndarray,
    ) -> None:
        guard = np.asarray(benign_guard_scores, dtype=np.float64)
        attack = np.asarray(attack_dev_scores, dtype=np.float64)
        if len(guard) != 500 or len(attack) != 500:
            raise ValueError(
                "Supervised development must be exactly 500 benign + 500 malicious scores"
            )
        if not np.isfinite(guard).all() or not np.isfinite(attack).all():
            raise ValueError("Supervised development scores must be finite")
        self.benign = benign
        self.benign_guard_scores = guard
        self.attack_dev_scores = attack

    @property
    def client_id(self) -> ClientId:
        return self.benign.client_id

    @property
    def scores(self) -> np.ndarray:
        return np.concatenate((self.benign_guard_scores, self.attack_dev_scores))

    @property
    def labels(self) -> np.ndarray:
        return np.concatenate(
            (
                np.zeros(len(self.benign_guard_scores), dtype=np.int64),
                np.ones(len(self.attack_dev_scores), dtype=np.int64),
            )
        )


class FinalTestEvidence:
    def __init__(
        self,
        benign: BenignPolicyEvidence,
        benign_test_scores: np.ndarray,
        attack_test_scores: np.ndarray,
        attack_test_groups: tuple[AttackGroupId, ...],
    ) -> None:
        benign_values = np.asarray(benign_test_scores, dtype=np.float64)
        attack_values = np.asarray(attack_test_scores, dtype=np.float64)
        if len(benign_values) == 0 or len(attack_values) == 0:
            raise ValueError("Final evaluation requires benign and malicious evidence")
        if len(attack_values) != len(attack_test_groups):
            raise ValueError("Attack-test groups must align with attack-test scores")
        if not np.isfinite(benign_values).all() or not np.isfinite(attack_values).all():
            raise ValueError("Final test scores must be finite")
        self.benign = benign
        self.benign_test_scores = benign_values
        self.attack_test_scores = attack_values
        self.attack_test_groups = attack_test_groups


class ClientPolicyThreshold:
    def __init__(self, policy: PolicyId, client_id: ClientId, threshold: Threshold | None) -> None:
        self.policy = policy
        self.client_id = client_id
        self.threshold = threshold


class UndefinedPolicyReason:
    def __init__(self, policy: PolicyId, reason: FailureCode) -> None:
        self.policy = policy
        self.reason = reason


class PolicyThresholdSet:
    def __init__(
        self,
        entries: tuple[ClientPolicyThreshold, ...],
        undefined_reasons: tuple[UndefinedPolicyReason, ...],
        shrinkage_n0: NonNegativeInt | None,
    ) -> None:
        self.entries = entries
        self.undefined_reasons = undefined_reasons
        self.shrinkage_n0 = shrinkage_n0

    def for_client(self, policy: PolicyId, client_id: ClientId) -> Threshold | None:
        if policy is PolicyId.ORACLE_TEST:
            raise ValueError("oracle_test is not available before final-test evidence opens")
        for entry in self.entries:
            if entry.policy is policy and entry.client_id == client_id:
                return entry.threshold
        raise KeyError(f"No threshold for {policy}/{client_id}")


def _finite_vector(values: np.ndarray, name: Identifier) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite non-empty vector")
    return values


def empirical_quantile(scores: np.ndarray, alpha: Alpha) -> Threshold:
    """Order-statistic quantile using the same rank convention as the reference threshold, so comparator baselines are apples-to-apples."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Quantile thresholds require a non-empty one-dimensional array")
    rank = min(len(values), int(np.ceil((len(values) + 1) * (1.0 - alpha))))
    partitioned = np.partition(values, rank - 1)
    return float(partitioned[rank - 1])


def reference_quantile(client: BenignPolicyEvidence) -> Threshold:
    return client.evaluation.reference.value


def global_quantile(clients: tuple[BenignPolicyEvidence, ...], alpha: Alpha) -> Threshold:
    counts = {len(client.full_policy_budget) for client in clients}
    if len(counts) != 1:
        raise ValueError("Global quantile requires equal per-client benign policy budgets")
    pooled = np.concatenate(tuple(client.full_policy_budget for client in clients))
    return empirical_quantile(pooled, alpha)


def local_quantile(client: BenignPolicyEvidence, alpha: Alpha) -> Threshold:
    return empirical_quantile(client.full_policy_budget, alpha)


def readiness_only(client: BenignPolicyEvidence) -> Threshold:
    readiness = client.evaluation.readiness
    if (
        readiness.plan.state is CalibrationReadinessState.READY
        and readiness.tie_count == 1
        and readiness.threshold is not None
    ):
        return readiness.threshold
    return client.evaluation.reference.value


def mismatch_only(client: BenignPolicyEvidence, alpha: Alpha) -> Threshold:
    if client.evaluation.mismatch.outcome in {MismatchOutcome.LOW, MismatchOutcome.HIGH}:
        return empirical_quantile(client.calibration_scores, alpha)
    return client.evaluation.reference.value


def tune_shrinkage(
    clients: tuple[BenignPolicyEvidence, ...],
    alpha: Alpha,
    n0_candidates: tuple[SampleCount, ...],
) -> NonNegativeInt:
    """Pick the pseudo-count n0 minimizing mean absolute mismatch-set FPR error across clients over the candidate grid."""
    if not n0_candidates:
        raise ValueError("Shrinkage tuning requires at least one candidate n0")
    best_n0 = n0_candidates[0]
    best_error = float("inf")

    client_data: list[tuple[Threshold, Threshold, SampleCount, np.ndarray, SampleCount]] = []
    for client in clients:
        local_q = empirical_quantile(client.calibration_scores, alpha)
        ref_v = client.evaluation.reference.value
        n_cal = len(client.calibration_scores)
        mismatch_sorted = np.sort(np.asarray(client.mismatch_scores, dtype=np.float64))
        n_mismatch = len(mismatch_sorted)
        if n_mismatch == 0:
            raise ValueError("Mismatch evidence cannot be empty when tuning shrinkage")
        client_data.append((local_q, ref_v, n_cal, mismatch_sorted, n_mismatch))

    for n0 in n0_candidates:
        total_error = 0.0
        for local_q, ref_v, n_cal, mismatch_sorted, n_mismatch in client_data:
            weight = n_cal / (n_cal + n0)
            threshold = weight * local_q + (1.0 - weight) * ref_v
            false_positives = n_mismatch - np.searchsorted(mismatch_sorted, threshold, side="right")
            total_error += abs((false_positives / n_mismatch) - alpha)

        mean_error = total_error / len(client_data)
        if mean_error < best_error or (
            np.isclose(mean_error, best_error, rtol=0.0, atol=1e-15) and n0 > best_n0
        ):
            best_error = mean_error
            best_n0 = n0
    return best_n0


def shrinkage(client: BenignPolicyEvidence, alpha: Alpha, n0: NonNegativeInt) -> Threshold:
    """Blend local and reference thresholds, weighting the local estimate by n_calibration/(n_calibration+n0) so it dominates as calibration data grows."""
    local = empirical_quantile(client.calibration_scores, alpha)
    n_calibration = len(client.calibration_scores)
    weight = n_calibration / (n_calibration + n0)
    return weight * local + (1.0 - weight) * client.evaluation.reference.value


def three_sigma(clients: tuple[BenignPolicyEvidence, ...]) -> Threshold:
    pooled = np.concatenate(tuple(client.full_policy_budget for client in clients))
    return float(np.mean(pooled) + 3.0 * np.std(pooled))


def f1_at_threshold(client: SupervisedDevelopmentEvidence, threshold: Threshold) -> Metric:
    value = f1(confusion_matrix(client.scores, client.labels, threshold))
    return -1.0 if value is None else value


def dev_local_global(
    client: SupervisedDevelopmentEvidence,
    global_threshold: Threshold,
    local_threshold: Threshold,
) -> Threshold:
    global_score = f1_at_threshold(client, global_threshold)
    local_score = f1_at_threshold(client, local_threshold)
    return local_threshold if local_score > global_score else global_threshold


def mean_client_f1_at_threshold(
    clients: tuple[SupervisedDevelopmentEvidence, ...], threshold: Threshold
) -> Metric:
    return float(np.mean([f1_at_threshold(client, threshold) for client in clients]))


def _pooled_moments(
    clients: tuple[SupervisedDevelopmentEvidence, ...],
    label: SupervisedClassLabel,
) -> ClassMoments:
    """Combine per-client mean/variance into pooled moments without re-touching raw scores (law of total variance)."""
    counts: list[NonNegativeCount] = []
    means: list[Score] = []
    variances = []
    for client in clients:
        values = client.scores[client.labels == label]
        counts.append(len(values))
        means.append(float(np.mean(values)))
        variances.append(float(np.var(values, ddof=0)))
    total = sum(counts)
    mean = sum(count * local_mean for count, local_mean in zip(counts, means, strict=True)) / total
    variance = (
        sum(
            count * (local_variance + local_mean**2)
            for count, local_mean, local_variance in zip(counts, means, variances, strict=True)
        )
        / total
        - mean**2
    )
    return ClassMoments(mean=mean, std=float(np.sqrt(max(variance, 0.0))))


def summary_statistic_threshold(
    clients: tuple[SupervisedDevelopmentEvidence, ...],
    candidate_count: CandidateCount,
) -> Threshold | None:
    """Search for the best F1 threshold only inside the overlap of the benign and attack mean+-3*std envelopes; undefined if the classes don't overlap."""
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    benign = _pooled_moments(clients, SupervisedClassLabel.BENIGN)
    attack = _pooled_moments(clients, SupervisedClassLabel.ATTACK)

    lower = max(benign.mean - 3.0 * benign.std, attack.mean - 3.0 * attack.std)
    upper = min(benign.mean + 3.0 * benign.std, attack.mean + 3.0 * attack.std)
    if lower >= upper:
        return None

    candidates = np.linspace(lower, upper, candidate_count, dtype=np.float64)
    scores = np.asarray([mean_client_f1_at_threshold(clients, float(t)) for t in candidates])
    best = float(np.max(scores))
    return float(candidates[int(np.flatnonzero(scores == best)[0])])


def supervised_global_f1(
    clients: tuple[SupervisedDevelopmentEvidence, ...],
    candidate_count: CandidateCount,
) -> Threshold:
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    minimum = min(float(np.min(client.scores)) for client in clients)
    maximum = max(float(np.max(client.scores)) for client in clients)
    candidates = np.linspace(minimum, maximum, candidate_count, dtype=np.float64)
    scores = np.asarray([mean_client_f1_at_threshold(clients, float(t)) for t in candidates])
    best = float(np.max(scores))
    return float(candidates[int(np.flatnonzero(scores == best)[0])])


@dataclass(frozen=True, slots=True)
class _OracleCandidate:
    band_error: Metric
    tpr_rank: Tpr | None
    order: NonNegativeInt
    threshold: Threshold


def oracle_choice(
    client: FinalTestEvidence,
    candidates: tuple[Threshold, Threshold, Threshold],
    band: OperatingBand,
) -> Threshold:
    """Pick the candidate with least band error (ties broken by highest TPR) using held-out final-test evidence; not a deployable policy, only an upper-bound comparator."""
    ranked: list[_OracleCandidate] = []
    benign_labels = np.zeros(len(client.benign_test_scores), dtype=np.int64)
    attack_labels = np.ones(len(client.attack_test_scores), dtype=np.int64)
    for order, threshold in enumerate(candidates):
        benign_cm = confusion_matrix(client.benign_test_scores, benign_labels, threshold)
        attack_cm = confusion_matrix(client.attack_test_scores, attack_labels, threshold)
        client_fpr = benign_cm.fp / (benign_cm.fp + benign_cm.tn)
        client_tpr = tpr(attack_cm)
        ranked.append(
            _OracleCandidate(
                band_error=band_error(client_fpr, band),
                tpr_rank=client_tpr,
                order=order,
                threshold=threshold,
            )
        )
    best = min(
        ranked,
        key=lambda candidate: (
            candidate.band_error,
            -(candidate.tpr_rank if candidate.tpr_rank is not None else 1.0),
            candidate.order,
        ),
    )
    return best.threshold


class PolicyStrategy(StrEnum):
    REFERENCE = "reference"
    GLOBAL_QUANTILE = "global_quantile"
    LOCAL_QUANTILE = "local_quantile"
    READINESS = "readiness"
    MISMATCH = "mismatch"
    SHRINKAGE = "shrinkage"
    THREE_SIGMA = "three_sigma"
    DEV_F1 = "dev_f1"
    SUMMARY_STATISTIC = "summary_statistic"
    SUPERVISED_F1 = "supervised_f1"
    FEDCRG = "fedcrg"
    ORACLE = "oracle"


class ThresholdOrigin(StrEnum):
    CALIBRATION = "calibration"
    DEVELOPMENT = "development"
    FINAL_TEST = "final_test"


class PolicyUploadKind(StrEnum):
    NONE = "none"
    REFERENCE_SCORES = "reference_scores"
    FULL_POLICY_BUDGET = "full_policy_budget"
    SUMMARY_MOMENTS_AND_CANDIDATES = "summary_moments_and_candidates"
    SUPERVISED_CANDIDATES = "supervised_candidates"


class PolicySpec(BaseModel):
    model_config = Frozen

    policy: PolicyId
    strategy: PolicyStrategy
    regime: InformationRegime
    deployable: bool
    supervised: bool
    requires_reference: bool
    threshold_origin: ThresholdOrigin
    requires_final_test: bool
    requires_supervised_development: bool
    uses_global_budget: bool = False
    uses_local_budget: bool = False
    needs_method: bool = False
    uses_shrinkage_tuning: bool = False
    uses_three_sigma: bool = False
    uses_summary_statistic: bool = False
    uses_supervised_f1: bool = False
    candidate_policies: tuple[PolicyId, ...] = ()
    upload_payload: PolicyUploadKind


POLICIES: dict[PolicyId, PolicySpec] = {
    PolicyId.REFERENCE_QUANTILE: PolicySpec(
        policy=PolicyId.REFERENCE_QUANTILE,
        strategy=PolicyStrategy.REFERENCE,
        regime=InformationRegime.BENIGN_ONLY,
        deployable=True,
        supervised=False,
        requires_reference=True,
        threshold_origin=ThresholdOrigin.CALIBRATION,
        requires_final_test=False,
        requires_supervised_development=False,
        upload_payload=PolicyUploadKind.REFERENCE_SCORES,
    ),
    PolicyId.GLOBAL_QUANTILE: PolicySpec(
        policy=PolicyId.GLOBAL_QUANTILE,
        strategy=PolicyStrategy.GLOBAL_QUANTILE,
        regime=InformationRegime.BENIGN_ONLY,
        deployable=True,
        supervised=False,
        requires_reference=False,
        threshold_origin=ThresholdOrigin.CALIBRATION,
        requires_final_test=False,
        requires_supervised_development=False,
        uses_global_budget=True,
        upload_payload=PolicyUploadKind.FULL_POLICY_BUDGET,
    ),
    PolicyId.LOCAL_QUANTILE: PolicySpec(
        policy=PolicyId.LOCAL_QUANTILE,
        strategy=PolicyStrategy.LOCAL_QUANTILE,
        regime=InformationRegime.BENIGN_ONLY,
        deployable=True,
        supervised=False,
        requires_reference=False,
        threshold_origin=ThresholdOrigin.CALIBRATION,
        requires_final_test=False,
        requires_supervised_development=False,
        uses_local_budget=True,
        upload_payload=PolicyUploadKind.NONE,
    ),
    PolicyId.READINESS_ONLY: PolicySpec(
        policy=PolicyId.READINESS_ONLY,
        strategy=PolicyStrategy.READINESS,
        regime=InformationRegime.BENIGN_ONLY,
        deployable=True,
        supervised=False,
        requires_reference=True,
        threshold_origin=ThresholdOrigin.CALIBRATION,
        requires_final_test=False,
        requires_supervised_development=False,
        upload_payload=PolicyUploadKind.NONE,
    ),
    PolicyId.MISMATCH_ONLY: PolicySpec(
        policy=PolicyId.MISMATCH_ONLY,
        strategy=PolicyStrategy.MISMATCH,
        regime=InformationRegime.BENIGN_ONLY,
        deployable=True,
        supervised=False,
        requires_reference=True,
        threshold_origin=ThresholdOrigin.CALIBRATION,
        requires_final_test=False,
        requires_supervised_development=False,
        upload_payload=PolicyUploadKind.NONE,
    ),
    PolicyId.SHRINKAGE: PolicySpec(
        policy=PolicyId.SHRINKAGE,
        strategy=PolicyStrategy.SHRINKAGE,
        regime=InformationRegime.BENIGN_ONLY,
        deployable=True,
        supervised=False,
        requires_reference=True,
        threshold_origin=ThresholdOrigin.CALIBRATION,
        requires_final_test=False,
        requires_supervised_development=False,
        uses_shrinkage_tuning=True,
        upload_payload=PolicyUploadKind.NONE,
    ),
    PolicyId.THREE_SIGMA: PolicySpec(
        policy=PolicyId.THREE_SIGMA,
        strategy=PolicyStrategy.THREE_SIGMA,
        regime=InformationRegime.BENIGN_ONLY,
        deployable=True,
        supervised=False,
        requires_reference=False,
        threshold_origin=ThresholdOrigin.CALIBRATION,
        requires_final_test=False,
        requires_supervised_development=False,
        uses_global_budget=True,
        uses_three_sigma=True,
        upload_payload=PolicyUploadKind.FULL_POLICY_BUDGET,
    ),
    PolicyId.DEV_F1_SELECT: PolicySpec(
        policy=PolicyId.DEV_F1_SELECT,
        strategy=PolicyStrategy.DEV_F1,
        regime=InformationRegime.SUPERVISED_DEVELOPMENT,
        deployable=False,
        supervised=True,
        requires_reference=False,
        threshold_origin=ThresholdOrigin.DEVELOPMENT,
        requires_final_test=False,
        requires_supervised_development=True,
        uses_global_budget=True,
        uses_local_budget=True,
        upload_payload=PolicyUploadKind.NONE,
    ),
    PolicyId.SUMMARY_STATISTIC_SELECT: PolicySpec(
        policy=PolicyId.SUMMARY_STATISTIC_SELECT,
        strategy=PolicyStrategy.SUMMARY_STATISTIC,
        regime=InformationRegime.SUPERVISED_DEVELOPMENT,
        deployable=False,
        supervised=True,
        requires_reference=False,
        threshold_origin=ThresholdOrigin.DEVELOPMENT,
        requires_final_test=False,
        requires_supervised_development=True,
        uses_summary_statistic=True,
        upload_payload=PolicyUploadKind.SUMMARY_MOMENTS_AND_CANDIDATES,
    ),
    PolicyId.SUPERVISED_F1: PolicySpec(
        policy=PolicyId.SUPERVISED_F1,
        strategy=PolicyStrategy.SUPERVISED_F1,
        regime=InformationRegime.SUPERVISED_DEVELOPMENT,
        deployable=False,
        supervised=True,
        requires_reference=False,
        threshold_origin=ThresholdOrigin.DEVELOPMENT,
        requires_final_test=False,
        requires_supervised_development=True,
        uses_supervised_f1=True,
        upload_payload=PolicyUploadKind.SUPERVISED_CANDIDATES,
    ),
    PolicyId.ORACLE_TEST: PolicySpec(
        policy=PolicyId.ORACLE_TEST,
        strategy=PolicyStrategy.ORACLE,
        regime=InformationRegime.FINAL_TEST_ORACLE,
        deployable=False,
        supervised=False,
        requires_reference=False,
        threshold_origin=ThresholdOrigin.FINAL_TEST,
        requires_final_test=True,
        requires_supervised_development=False,
        uses_global_budget=True,
        uses_local_budget=True,
        needs_method=True,
        candidate_policies=(
            PolicyId.GLOBAL_QUANTILE,
            PolicyId.LOCAL_QUANTILE,
            PolicyId.FEDCRG,
        ),
        upload_payload=PolicyUploadKind.NONE,
    ),
    PolicyId.FEDCRG: PolicySpec(
        policy=PolicyId.FEDCRG,
        strategy=PolicyStrategy.FEDCRG,
        regime=InformationRegime.BENIGN_ONLY,
        deployable=True,
        supervised=False,
        requires_reference=True,
        threshold_origin=ThresholdOrigin.CALIBRATION,
        requires_final_test=False,
        requires_supervised_development=False,
        needs_method=True,
        upload_payload=PolicyUploadKind.REFERENCE_SCORES,
    ),
}

SUPERVISED_POLICIES = frozenset(spec.policy for spec in POLICIES.values() if spec.supervised)


def information_regime(policy_id: PolicyId) -> InformationRegime:
    return POLICIES[policy_id].regime


def is_deployable(policy_id: PolicyId) -> bool:
    return POLICIES[policy_id].deployable


@dataclass(frozen=True, slots=True)
class _PreparedThresholds:
    global_q: Threshold | None
    local_q: Mapping[ClientId, Threshold]
    shrinkage_n0: NonNegativeInt | None
    three_sigma_threshold: Threshold | None
    summary_threshold: Threshold | None
    supervised_threshold: Threshold | None
    supervised_by_client: Mapping[ClientId, SupervisedDevelopmentEvidence]


def _evaluate(
    strategy: PolicyStrategy,
    benign: BenignPolicyEvidence,
    prepared: _PreparedThresholds,
    alpha: Alpha,
) -> Threshold | None:
    match strategy:
        case PolicyStrategy.REFERENCE:
            return reference_quantile(benign)
        case PolicyStrategy.GLOBAL_QUANTILE:
            if prepared.global_q is None:
                raise RuntimeError("Global quantile was not materialized")
            return prepared.global_q
        case PolicyStrategy.LOCAL_QUANTILE:
            return prepared.local_q[benign.client_id]
        case PolicyStrategy.READINESS:
            return readiness_only(benign)
        case PolicyStrategy.MISMATCH:
            return mismatch_only(benign, alpha)
        case PolicyStrategy.SHRINKAGE:
            if prepared.shrinkage_n0 is None:
                raise RuntimeError("Shrinkage tuning was not materialized")
            return shrinkage(benign, alpha, prepared.shrinkage_n0)
        case PolicyStrategy.THREE_SIGMA:
            if prepared.three_sigma_threshold is None:
                raise RuntimeError("Three-sigma threshold was not materialized")
            return prepared.three_sigma_threshold
        case PolicyStrategy.DEV_F1:
            if prepared.global_q is None:
                raise RuntimeError("Global quantile was not materialized")
            return dev_local_global(
                prepared.supervised_by_client[benign.client_id],
                prepared.global_q,
                prepared.local_q[benign.client_id],
            )
        case PolicyStrategy.SUMMARY_STATISTIC:
            return prepared.summary_threshold
        case PolicyStrategy.SUPERVISED_F1:
            if prepared.supervised_threshold is None:
                raise RuntimeError("Supervised F1 threshold was not materialized")
            return prepared.supervised_threshold
        case PolicyStrategy.FEDCRG:
            return benign.evaluation.decision.threshold
        case _:
            raise RuntimeError(f"Unhandled threshold strategy: {strategy}")


class PolicyThresholdSelector:
    def select(
        self,
        benign_clients: tuple[BenignPolicyEvidence, ...],
        protocol: ProtocolConfig,
        statistics: StatisticsConfig,
        requested_policies: tuple[PolicyId, ...],
        supervised_clients: tuple[SupervisedDevelopmentEvidence, ...] | None = None,
    ) -> PolicyThresholdSet:
        if not benign_clients:
            raise ValueError("At least one client is required for policy selection")
        non_oracle = tuple(
            policy for policy in requested_policies if policy is not PolicyId.ORACLE_TEST
        )
        if not non_oracle and PolicyId.ORACLE_TEST not in requested_policies:
            raise ValueError("No policy was requested")

        client_ids = tuple(client.client_id for client in benign_clients)
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("Policy selection received duplicate client identities")

        supervised_by_client = self._resolve_supervised_evidence(
            non_oracle, client_ids, supervised_clients
        )
        need_method = any(POLICIES[policy].needs_method for policy in requested_policies)
        prepared = self._prepare_thresholds(
            benign_clients,
            protocol,
            statistics,
            requested_policies,
            non_oracle,
            supervised_by_client,
        )

        entries, undefined = self._build_threshold_entries(
            benign_clients,
            non_oracle,
            requested_policies,
            prepared,
            protocol,
            need_method,
        )

        expected = {(policy, client_id) for policy in non_oracle for client_id in client_ids}
        observed = {
            (entry.policy, entry.client_id) for entry in entries if entry.policy in non_oracle
        }
        if observed != expected:
            raise RuntimeError("Policy selection did not produce each requested client-policy cell")
        return PolicyThresholdSet(tuple(entries), tuple(undefined), prepared.shrinkage_n0)

    @staticmethod
    def _resolve_supervised_evidence(
        non_oracle: tuple[PolicyId, ...],
        client_ids: tuple[ClientId, ...],
        supervised_clients: tuple[SupervisedDevelopmentEvidence, ...] | None,
    ) -> dict[ClientId, SupervisedDevelopmentEvidence]:
        supervised_needed = any(
            POLICIES[policy].requires_supervised_development for policy in non_oracle
        )
        supervised_by_client: dict[ClientId, SupervisedDevelopmentEvidence] = {}
        if supervised_needed:
            if supervised_clients is None:
                raise ValueError("Requested supervised comparator lacks development evidence")
            supervised_by_client = {client.client_id: client for client in supervised_clients}
            if set(supervised_by_client) != set(client_ids):
                raise ValueError("Supervised development evidence does not cover the federation")
        elif supervised_clients is not None:
            raise ValueError(
                "Supervised development evidence was supplied although no supervised policy was requested"
            )
        return supervised_by_client

    @staticmethod
    def _prepare_thresholds(
        benign_clients: tuple[BenignPolicyEvidence, ...],
        protocol: ProtocolConfig,
        statistics: StatisticsConfig,
        requested_policies: tuple[PolicyId, ...],
        non_oracle: tuple[PolicyId, ...],
        supervised_by_client: dict[ClientId, SupervisedDevelopmentEvidence],
    ) -> _PreparedThresholds:
        need_global = any(POLICIES[policy].uses_global_budget for policy in requested_policies)
        need_local = any(POLICIES[policy].uses_local_budget for policy in requested_policies)

        global_q = global_quantile(benign_clients, protocol.alpha) if need_global else None
        local_q = (
            {client.client_id: local_quantile(client, protocol.alpha) for client in benign_clients}
            if need_local
            else {}
        )
        shrinkage_n0 = (
            tune_shrinkage(
                benign_clients,
                protocol.alpha,
                statistics.shrinkage_n0_candidates,
            )
            if any(POLICIES[policy].uses_shrinkage_tuning for policy in non_oracle)
            else None
        )
        three_sigma_threshold = (
            three_sigma(benign_clients)
            if any(POLICIES[policy].uses_three_sigma for policy in non_oracle)
            else None
        )
        summary_threshold = (
            summary_statistic_threshold(
                tuple(supervised_by_client.values()),
                statistics.supervised_threshold_candidates,
            )
            if any(POLICIES[policy].uses_summary_statistic for policy in non_oracle)
            else None
        )
        supervised_threshold = (
            supervised_global_f1(
                tuple(supervised_by_client.values()),
                statistics.supervised_threshold_candidates,
            )
            if any(POLICIES[policy].uses_supervised_f1 for policy in non_oracle)
            else None
        )
        return _PreparedThresholds(
            global_q=global_q,
            local_q=local_q,
            shrinkage_n0=shrinkage_n0,
            three_sigma_threshold=three_sigma_threshold,
            summary_threshold=summary_threshold,
            supervised_threshold=supervised_threshold,
            supervised_by_client=supervised_by_client,
        )

    @staticmethod
    def _build_threshold_entries(
        benign_clients: tuple[BenignPolicyEvidence, ...],
        non_oracle: tuple[PolicyId, ...],
        requested_policies: tuple[PolicyId, ...],
        prepared: _PreparedThresholds,
        protocol: ProtocolConfig,
        need_method: bool,
    ) -> tuple[list[ClientPolicyThreshold], list[UndefinedPolicyReason]]:
        entries: list[ClientPolicyThreshold] = []
        undefined: list[UndefinedPolicyReason] = []
        if PolicyId.SUMMARY_STATISTIC_SELECT in non_oracle and prepared.summary_threshold is None:
            undefined.append(
                UndefinedPolicyReason(
                    PolicyId.SUMMARY_STATISTIC_SELECT,
                    FailureCode.SUMMARY_STATISTIC_COMPARATOR_UNDEFINED,
                )
            )

        for benign in benign_clients:
            client_id = benign.client_id
            for policy in non_oracle:
                threshold = _evaluate(POLICIES[policy].strategy, benign, prepared, protocol.alpha)
                entries.append(ClientPolicyThreshold(policy, client_id, threshold))

            if PolicyId.ORACLE_TEST in requested_policies:
                PolicyThresholdSelector._append_oracle_candidates(
                    benign, entries, prepared, protocol, need_method
                )
        return entries, undefined

    @staticmethod
    def _append_oracle_candidates(
        benign: BenignPolicyEvidence,
        entries: list[ClientPolicyThreshold],
        prepared: _PreparedThresholds,
        protocol: ProtocolConfig,
        need_method: bool,
    ) -> None:
        client_id = benign.client_id
        if prepared.global_q is None or client_id not in prepared.local_q or not need_method:
            raise RuntimeError("Oracle candidate thresholds were not prepared")
        for policy in POLICIES[PolicyId.ORACLE_TEST].candidate_policies:
            if not any(item.policy is policy and item.client_id == client_id for item in entries):
                candidate = _evaluate(POLICIES[policy].strategy, benign, prepared, protocol.alpha)
                entries.append(ClientPolicyThreshold(policy, client_id, candidate))
