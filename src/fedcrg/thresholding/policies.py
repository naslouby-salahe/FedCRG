"""Threshold-comparator policies: evidence views, the twelve locked policy
rules, selection, and typed results.

All policies consume the same frozen score evidence. Supervised comparators
require explicit development evidence so attack labels cannot enter benign-only
code; the oracle requires final-test evidence and is never exposed as a
deployable threshold.
"""

from __future__ import annotations


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
    Fpr,
    Identifier,
    InformationRegime,
    Metric,
    NonNegativeInt,
    OperatingBand,
    PolicyId,
    PositiveCount,
    SampleCount,
    SupervisedClassLabel,
)

Frozen = ConfigDict(frozen=True)

_FLOAT64_BYTES = 8
_INT64_BYTES = 8


class PolicyTrafficLedgerRow(BaseModel):
    """Deterministic threshold-policy upload payload for one policy."""

    model_config = Frozen

    policy: PolicyId
    upload_bytes_per_client: ByteCount


def threshold_policy_communication(
    config: ExperimentConfig,
    client_count: PositiveCount,
) -> tuple[PolicyTrafficLedgerRow, ...]:
    """Threshold-policy upload payloads, separate from model-training traffic.

    The ledger covers the protocol-mandated uploads of the threshold-policy
    payload accounting section: the FedCRG reference scores, the naive
    full-budget score upload of the global and three-sigma comparators, and
    the summary-statistic/F1 candidate vectors. Every other policy constructs
    its threshold from local evidence or from reference scores already
    uploaded, so it adds no upload payload. All counts are read from the
    resolved experiment configuration.
    """
    if client_count <= 0:
        raise ValueError("Policy traffic accounting requires a positive client count")
    split = config.dataset.split
    full_policy_budget = (
        split.reference_benign + split.mismatch_benign + split.calibration_benign
    )
    reference_payload = split.reference_benign * _FLOAT64_BYTES
    full_budget_payload = full_policy_budget * _FLOAT64_BYTES
    candidates = config.statistics.supervised_threshold_candidates
    moment_payload = 2 * (_INT64_BYTES + 2 * _FLOAT64_BYTES)
    candidate_payload = candidates * _FLOAT64_BYTES

    rows = (
        PolicyTrafficLedgerRow(policy=PolicyId.REFERENCE_QUANTILE, upload_bytes_per_client=reference_payload),
        PolicyTrafficLedgerRow(policy=PolicyId.GLOBAL_QUANTILE, upload_bytes_per_client=full_budget_payload),
        PolicyTrafficLedgerRow(policy=PolicyId.LOCAL_QUANTILE, upload_bytes_per_client=0),
        PolicyTrafficLedgerRow(policy=PolicyId.READINESS_ONLY, upload_bytes_per_client=0),
        PolicyTrafficLedgerRow(policy=PolicyId.MISMATCH_ONLY, upload_bytes_per_client=0),
        PolicyTrafficLedgerRow(policy=PolicyId.SHRINKAGE, upload_bytes_per_client=0),
        PolicyTrafficLedgerRow(policy=PolicyId.THREE_SIGMA, upload_bytes_per_client=full_budget_payload),
        PolicyTrafficLedgerRow(policy=PolicyId.DEV_F1_SELECT, upload_bytes_per_client=0),
        PolicyTrafficLedgerRow(
            policy=PolicyId.SUMMARY_STATISTIC_SELECT,
            upload_bytes_per_client=moment_payload + candidate_payload,
        ),
        PolicyTrafficLedgerRow(policy=PolicyId.SUPERVISED_F1, upload_bytes_per_client=candidate_payload),
        PolicyTrafficLedgerRow(policy=PolicyId.ORACLE_TEST, upload_bytes_per_client=0),
        PolicyTrafficLedgerRow(policy=PolicyId.FEDCRG, upload_bytes_per_client=reference_payload),
    )
    if len(rows) != len(PolicyId):
        raise RuntimeError("Policy traffic ledger must contain exactly one row per policy")
    return rows


class BenignPolicyEvidence:
    """Benign-only score evidence for one client across the R/G/C roles."""

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
    """Balanced 500-benign + 500-malicious development evidence for supervised comparators."""

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
    """Final-label evidence opened only after non-oracle thresholds are frozen."""

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
    """Selected threshold for one client/policy."""
    def __init__(self, policy: PolicyId, client_id: ClientId, threshold: Threshold | None) -> None:
        self.policy = policy
        self.client_id = client_id
        self.threshold = threshold


class UndefinedPolicyReason:
    """Closed domain of undefined-threshold reasons."""
    def __init__(self, policy: PolicyId, reason: FailureCode) -> None:
        self.policy = policy
        self.reason = reason


class PolicyThresholdSet:
    """Thresholds frozen before final-test evidence is opened."""

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
        raise KeyError(f"No threshold for {policy.value}/{client_id}")


def _finite_vector(values: np.ndarray, name: Identifier) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite non-empty vector")
    return values


def empirical_quantile(scores: np.ndarray, alpha: Alpha) -> Threshold:
    """Empirical quantile of one finite vector."""
    values = np.sort(np.asarray(scores, dtype=np.float64), kind="stable")
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Quantile thresholds require a non-empty one-dimensional array")
    rank = min(len(values), int(np.ceil((len(values) + 1) * (1.0 - alpha))))
    return float(values[rank - 1])


def reference_quantile(client: BenignPolicyEvidence) -> Threshold:
    """Reference-quantile threshold comparator."""
    return client.evaluation.reference.value


def global_quantile(clients: tuple[BenignPolicyEvidence, ...], alpha: Alpha) -> Threshold:
    """Global-quantile threshold comparator."""
    counts = {len(client.full_policy_budget) for client in clients}
    if len(counts) != 1:
        raise ValueError("Global quantile requires equal per-client benign policy budgets")
    pooled = np.concatenate(tuple(client.full_policy_budget for client in clients))
    return empirical_quantile(pooled, alpha)


def local_quantile(client: BenignPolicyEvidence, alpha: Alpha) -> Threshold:
    """Local-quantile threshold comparator."""
    return empirical_quantile(client.full_policy_budget, alpha)


def readiness_only(client: BenignPolicyEvidence) -> Threshold:
    """Readiness-admission comparator."""
    readiness = client.evaluation.readiness
    if (
        readiness.plan.state is CalibrationReadinessState.READY
        and readiness.tie_count == 1
        and readiness.threshold is not None
    ):
        return readiness.threshold
    return client.evaluation.reference.value


def mismatch_only(client: BenignPolicyEvidence, alpha: Alpha) -> Threshold:
    """Mismatch-admission comparator."""
    if client.evaluation.mismatch.outcome in {MismatchOutcome.LOW, MismatchOutcome.HIGH}:
        return empirical_quantile(client.calibration_scores, alpha)
    return client.evaluation.reference.value


def _estimated_fpr(scores: np.ndarray, threshold: Threshold) -> Fpr:
    if len(scores) == 0:
        raise ValueError("Mismatch evidence cannot be empty when tuning shrinkage")
    return float(np.mean(np.asarray(scores, dtype=np.float64) > threshold))


def tune_shrinkage(
    clients: tuple[BenignPolicyEvidence, ...],
    alpha: Alpha,
    n0_candidates: tuple[SampleCount, ...],
) -> NonNegativeInt:
    """Grid-search shrinkage strength over candidates."""
    if not n0_candidates:
        raise ValueError("Shrinkage tuning requires at least one candidate n0")
    best_n0 = n0_candidates[0]
    best_error = float("inf")
    for n0 in n0_candidates:
        errors: list[float] = []
        for client in clients:
            local = empirical_quantile(client.calibration_scores, alpha)
            n_calibration = len(client.calibration_scores)
            weight = n_calibration / (n_calibration + n0)
            threshold = weight * local + (1.0 - weight) * client.evaluation.reference.value
            errors.append(abs(_estimated_fpr(client.mismatch_scores, threshold) - alpha))
        mean_error = float(np.mean(errors))
        if mean_error < best_error or (
            np.isclose(mean_error, best_error, rtol=0.0, atol=1e-15) and n0 > best_n0
        ):
            best_error = mean_error
            best_n0 = n0
    return best_n0


def shrinkage(client: BenignPolicyEvidence, alpha: Alpha, n0: NonNegativeInt) -> Threshold:
    """Shrinkage threshold comparator."""
    local = empirical_quantile(client.calibration_scores, alpha)
    n_calibration = len(client.calibration_scores)
    weight = n_calibration / (n_calibration + n0)
    return weight * local + (1.0 - weight) * client.evaluation.reference.value


def three_sigma(clients: tuple[BenignPolicyEvidence, ...]) -> Threshold:
    """Three-sigma threshold comparator."""
    pooled = np.concatenate(tuple(client.full_policy_budget for client in clients))
    mean = float(np.mean(pooled))
    population_std = float(np.sqrt(np.mean((pooled - mean) ** 2)))
    return mean + 3.0 * population_std


def f1_at_threshold(
    client: SupervisedDevelopmentEvidence, threshold: Threshold
) -> Metric:
    """F1 of one development evidence pair at a threshold."""
    value = f1(confusion_matrix(client.scores, client.labels, threshold))
    return -1.0 if value is None else value


def dev_local_global(
    client: SupervisedDevelopmentEvidence,
    global_threshold: Threshold,
    local_threshold: Threshold,
) -> Threshold:
    """Best-of development F1 comparator."""
    global_score = f1_at_threshold(client, global_threshold)
    local_score = f1_at_threshold(client, local_threshold)
    return local_threshold if local_score > global_score else global_threshold


def mean_client_f1_at_threshold(
    clients: tuple[SupervisedDevelopmentEvidence, ...], threshold: Threshold
) -> Metric:
    """Mean F1 across development clients at a threshold."""
    return float(np.mean([f1_at_threshold(client, threshold) for client in clients]))


def _pooled_moments(
    clients: tuple[SupervisedDevelopmentEvidence, ...],
    label: SupervisedClassLabel,
) -> ClassMoments:
    counts: list[int] = []
    means: list[float] = []
    variances: list[float] = []
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
    """Summary-statistic supervised comparator."""
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
    """Supervised global F1 comparator."""
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    minimum = min(float(np.min(client.scores)) for client in clients)
    maximum = max(float(np.max(client.scores)) for client in clients)
    candidates = np.linspace(minimum, maximum, candidate_count, dtype=np.float64)
    scores = np.asarray([mean_client_f1_at_threshold(clients, float(t)) for t in candidates])
    best = float(np.max(scores))
    return float(candidates[int(np.flatnonzero(scores == best)[0])])


def oracle_choice(
    client: FinalTestEvidence,
    candidates: tuple[Threshold, Threshold, Threshold],
    band: OperatingBand,
) -> Threshold:
    """Oracle threshold choice from final-test evidence."""
    ranked: list[tuple[float, float, int, float]] = []
    benign_labels = np.zeros(len(client.benign_test_scores), dtype=np.int64)
    attack_labels = np.ones(len(client.attack_test_scores), dtype=np.int64)
    for order, threshold in enumerate(candidates):
        benign_cm = confusion_matrix(client.benign_test_scores, benign_labels, threshold)
        attack_cm = confusion_matrix(client.attack_test_scores, attack_labels, threshold)
        client_fpr = benign_cm.fp / (benign_cm.fp + benign_cm.tn)
        client_tpr = tpr(attack_cm)
        tpr_rank = -1.0 if client_tpr is None else client_tpr
        ranked.append((band_error(client_fpr, band), -tpr_rank, order, threshold))
    return min(ranked)[3]


SUPERVISED_POLICIES = frozenset(
    {
        PolicyId.DEV_F1_SELECT,
        PolicyId.SUMMARY_STATISTIC_SELECT,
        PolicyId.SUPERVISED_F1,
    }
)


def information_regime(policy_id: PolicyId) -> InformationRegime:
    """Information regime of one policy."""
    if policy_id is PolicyId.ORACLE_TEST:
        return InformationRegime.FINAL_TEST_ORACLE
    if policy_id in SUPERVISED_POLICIES:
        return InformationRegime.SUPERVISED_DEVELOPMENT
    return InformationRegime.BENIGN_ONLY


def is_deployable(policy_id: PolicyId) -> bool:
    """Whether one policy may deploy client thresholds."""
    return policy_id not in SUPERVISED_POLICIES and policy_id is not PolicyId.ORACLE_TEST


class PolicyThresholdSelector:
    """Select only requested non-oracle policies from their permitted evidence."""

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

        supervised_needed = bool(set(non_oracle) & SUPERVISED_POLICIES)
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

        need_global = (
            bool(
                set(non_oracle)
                & {
                    PolicyId.GLOBAL_QUANTILE,
                    PolicyId.DEV_F1_SELECT,
                }
            )
            or PolicyId.ORACLE_TEST in requested_policies
        )
        need_local = (
            bool(
                set(non_oracle)
                & {
                    PolicyId.LOCAL_QUANTILE,
                    PolicyId.DEV_F1_SELECT,
                }
            )
            or PolicyId.ORACLE_TEST in requested_policies
        )
        need_method = PolicyId.FEDCRG in non_oracle or PolicyId.ORACLE_TEST in requested_policies

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
            if PolicyId.SHRINKAGE in non_oracle
            else None
        )
        three_sigma_threshold = (
            three_sigma(benign_clients) if PolicyId.THREE_SIGMA in non_oracle else None
        )
        summary_threshold = (
            summary_statistic_threshold(
                tuple(supervised_by_client.values()),
                statistics.supervised_threshold_candidates,
            )
            if PolicyId.SUMMARY_STATISTIC_SELECT in non_oracle
            else None
        )
        supervised_threshold = (
            supervised_global_f1(
                tuple(supervised_by_client.values()),
                statistics.supervised_threshold_candidates,
            )
            if PolicyId.SUPERVISED_F1 in non_oracle
            else None
        )

        entries: list[ClientPolicyThreshold] = []
        undefined: list[UndefinedPolicyReason] = []
        if PolicyId.SUMMARY_STATISTIC_SELECT in non_oracle and summary_threshold is None:
            undefined.append(
                UndefinedPolicyReason(
                    PolicyId.SUMMARY_STATISTIC_SELECT,
                    FailureCode.SUMMARY_STATISTIC_COMPARATOR_UNDEFINED,
                )
            )

        for benign in benign_clients:
            client_id = benign.client_id
            for policy in non_oracle:
                threshold: Threshold | None
                if policy is PolicyId.REFERENCE_QUANTILE:
                    threshold = reference_quantile(benign)
                elif policy is PolicyId.GLOBAL_QUANTILE:
                    threshold = global_q
                elif policy is PolicyId.LOCAL_QUANTILE:
                    threshold = local_q[client_id]
                elif policy is PolicyId.READINESS_ONLY:
                    threshold = readiness_only(benign)
                elif policy is PolicyId.MISMATCH_ONLY:
                    threshold = mismatch_only(benign, protocol.alpha)
                elif policy is PolicyId.SHRINKAGE:
                    if shrinkage_n0 is None:
                        raise RuntimeError("Shrinkage tuning was not materialized")
                    threshold = shrinkage(benign, protocol.alpha, shrinkage_n0)
                elif policy is PolicyId.THREE_SIGMA:
                    threshold = three_sigma_threshold
                elif policy is PolicyId.DEV_F1_SELECT:
                    if global_q is None:
                        raise RuntimeError("Global quantile was not materialized")
                    threshold = dev_local_global(
                        supervised_by_client[client_id],
                        global_q,
                        local_q[client_id],
                    )
                elif policy is PolicyId.SUMMARY_STATISTIC_SELECT:
                    threshold = summary_threshold
                elif policy is PolicyId.SUPERVISED_F1:
                    threshold = supervised_threshold
                elif policy is PolicyId.FEDCRG:
                    threshold = benign.evaluation.decision.threshold
                else:
                    raise RuntimeError(f"Unhandled non-oracle policy: {policy.value}")
                entries.append(ClientPolicyThreshold(policy, client_id, threshold))

            if PolicyId.ORACLE_TEST in requested_policies:
                if global_q is None or client_id not in local_q or not need_method:
                    raise RuntimeError("Oracle candidate thresholds were not prepared")
                for policy, threshold in (
                    (PolicyId.GLOBAL_QUANTILE, global_q),
                    (PolicyId.LOCAL_QUANTILE, local_q[client_id]),
                    (PolicyId.FEDCRG, benign.evaluation.decision.threshold),
                ):
                    if not any(
                        item.policy is policy and item.client_id == client_id for item in entries
                    ):
                        entries.append(ClientPolicyThreshold(policy, client_id, threshold))

        expected = {(policy, client_id) for policy in non_oracle for client_id in client_ids}
        observed = {
            (entry.policy, entry.client_id) for entry in entries if entry.policy in non_oracle
        }
        if observed != expected:
            raise RuntimeError("Policy selection did not produce each requested client-policy cell")
        return PolicyThresholdSet(tuple(entries), tuple(undefined), shrinkage_n0)
