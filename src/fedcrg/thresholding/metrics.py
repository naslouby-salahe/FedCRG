"""Evaluation metrics and per-client evidence composition.

Reliability metrics (band error, high excess, band violation, absolute FPR
error), utility metrics (attack-balanced macro TPR, ranking AUROC/AUPRC),
strict-threshold classification metrics with explicit undefined semantics,
equal-client federation aggregation, and the per-client governance
composition that produces one ``ClientEvaluationResult`` per client.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

import numpy as np
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import average_precision_score, roc_auc_score

from fedcrg.config import ProtocolConfig
from fedcrg.thresholding.readiness import (
    CalibrationReadinessEvaluator,
    ClientEvaluationResult,
    ReadinessPlan,
    ReadinessPlanCache,
    ReferenceMismatchEvaluator,
    ReferenceThreshold,
    DeploymentDecision,
    build_reference_threshold,
)
from fedcrg.types import (
    Alpha,
    CalibrationReadinessState,
    ClientId,
    ConfidenceInterval,
    DecisionState,
    ExceedanceCount,
    Fpr,
    Fraction,
    Metric,
    MismatchOutcome,
    NonNegativeCount,
    OperatingBand,
    PolicyEvaluationStatus,
    PolicyId,
    PositiveCount,
    SampleCount,
    Threshold,
    Tolerance,
    Tpr,
)

Frozen = ConfigDict(frozen=True)

ReferenceEstimator = Callable[[Mapping[ClientId, np.ndarray], ProtocolConfig], ReferenceThreshold]


class ConfusionMatrix(BaseModel):
    """Four-cell confusion matrix over one threshold."""

    model_config = Frozen

    tp: ExceedanceCount
    tn: ExceedanceCount
    fp: ExceedanceCount
    fn: ExceedanceCount


def confusion_matrix(
    scores: np.ndarray, labels: np.ndarray, threshold: Threshold
) -> ConfusionMatrix:
    """Build a confusion matrix for one threshold."""
    values = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if values.shape != targets.shape or values.ndim != 1:
        raise ValueError("scores and labels must be aligned one-dimensional arrays")
    if not np.isfinite(values).all():
        raise ValueError("NONFINITE_SCORE")
    predictions = values > threshold
    positives = targets == 1
    negatives = targets == 0
    if np.count_nonzero(~(positives | negatives)):
        raise ValueError("Labels must be binary 0/1")
    return ConfusionMatrix(
        tp=int(np.count_nonzero(predictions & positives)),
        tn=int(np.count_nonzero(~predictions & negatives)),
        fp=int(np.count_nonzero(predictions & negatives)),
        fn=int(np.count_nonzero(~predictions & positives)),
    )


def _ratio(numerator: NonNegativeCount, denominator: NonNegativeCount) -> Fraction | None:
    return float(numerator / denominator) if denominator else None


def fpr(cm: ConfusionMatrix) -> Fpr | None:
    """False-positive rate from a confusion matrix."""
    return _ratio(cm.fp, cm.fp + cm.tn)


def tpr(cm: ConfusionMatrix) -> Tpr | None:
    """True-positive rate from a confusion matrix."""
    return _ratio(cm.tp, cm.tp + cm.fn)


def precision(cm: ConfusionMatrix) -> Fpr | None:
    """Precision from a confusion matrix."""
    return _ratio(cm.tp, cm.tp + cm.fp)


def recall(cm: ConfusionMatrix) -> Tpr | None:
    """Recall from a confusion matrix."""
    return tpr(cm)


def f1(cm: ConfusionMatrix) -> Metric | None:
    """F1 score from a confusion matrix."""
    p = precision(cm)
    r = recall(cm)
    if p is None or r is None or p + r == 0.0:
        return None
    return 2.0 * p * r / (p + r)


def balanced_accuracy(cm: ConfusionMatrix) -> Fpr | None:
    """Balanced accuracy from a confusion matrix."""
    sensitivity = tpr(cm)
    specificity = _ratio(cm.tn, cm.tn + cm.fp)
    if sensitivity is None or specificity is None:
        return None
    return 0.5 * (sensitivity + specificity)


def band_error(fpr_value: Fpr, band: OperatingBand) -> Fraction:
    """Signed distance of one FPR to the operating band."""
    if fpr_value < band.lower:
        return band.lower - fpr_value
    if fpr_value > band.upper:
        return fpr_value - band.upper
    return 0.0


def high_excess(fpr_value: Fpr, band: OperatingBand) -> Fraction:
    """FPR excess above the operating-band upper edge."""
    return max(0.0, fpr_value - band.upper)


def band_violation(fpr_value: Fpr, band: OperatingBand) -> Fraction:
    """Whether one FPR falls outside the operating band."""
    return float(not band.contains(fpr_value))


def absolute_fpr_error(fpr_value: Fpr, alpha: Alpha) -> Fraction:
    """Absolute FPR deviation from the target alpha."""
    return abs(fpr_value - alpha)


def attack_balanced_tpr(
    scores: np.ndarray,
    labels: np.ndarray,
    attack_groups: np.ndarray,
    threshold: Threshold,
) -> Tpr | None:
    """Macro TPR over attack groups at one threshold."""
    values = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    groups_array = np.asarray(attack_groups, dtype=object)
    groups = sorted(set(groups_array[targets == 1].astype(str)))
    if not groups:
        return None
    per_group: list[Tpr] = []
    for group in groups:
        mask = (targets == 1) & (groups_array.astype(str) == group)
        if not np.any(mask):
            continue
        per_group.append(float(np.mean(values[mask] > threshold)))
    return float(np.mean(per_group)) if per_group else None


def auroc(scores: np.ndarray, labels: np.ndarray) -> Fpr:
    """Area under the ROC curve."""
    return float(roc_auc_score(labels, scores))


def auprc(scores: np.ndarray, labels: np.ndarray) -> Fpr:
    """Area under the precision-recall curve."""
    return float(average_precision_score(labels, scores))


class ClientMetrics(BaseModel):
    """Frozen evaluation metrics for one client/policy."""

    model_config = Frozen

    benign_n: PositiveCount
    attack_n: PositiveCount
    fp: ExceedanceCount
    tn: ExceedanceCount
    tp: ExceedanceCount
    fn: ExceedanceCount
    fpr: Fpr
    tpr: Tpr | None
    precision: Fpr | None
    recall: Tpr | None
    f1: Fpr | None
    balanced_accuracy: Fpr | None
    auroc: Fpr
    auprc: Fpr
    band_error: Fraction
    high_excess: Fraction
    band_violation: Fraction
    absolute_fpr_error: Fraction
    attack_balanced_tpr: Tpr | None
    fpr_reference_interval: ConfidenceInterval


class PolicyEvaluation(BaseModel):
    """One client/policy evaluation outcome."""

    model_config = Frozen

    client_id: ClientId
    policy: PolicyId
    threshold: Threshold | None
    status: PolicyEvaluationStatus
    metrics: ClientMetrics | None


class FederationMetrics(BaseModel):
    """Frozen federation-level reliability and utility metrics."""

    model_config = Frozen

    policy: PolicyId
    client_count: PositiveCount
    mebe: Fraction
    high_excess: Fraction
    band_violation_rate: Fraction
    mafe: Fraction
    max_fpr: Fpr
    fpr_iqr: Fpr
    attack_balanced_macro_tpr: Tpr | None
    macro_tpr: Tpr | None
    worst_client_tpr: Tpr | None
    worst_client_attack_balanced_tpr: Tpr | None
    mean_f1: Fpr | None


class EvaluationBundle(BaseModel):
    """All client, federation, and protocol results of one cell."""

    model_config = Frozen

    clients: tuple[PolicyEvaluation, ...]
    federations: tuple[FederationMetrics, ...]
    protocol_results: tuple[ClientEvaluationResult, ...]
    shrinkage_n0: PositiveCount | None


class AdmissionSummary(BaseModel):
    """Composition of reference, readiness, mismatch, and decision."""

    model_config = Frozen

    client_count: PositiveCount
    readiness_rate: Fraction
    low_mismatch_rate: Fraction
    high_mismatch_rate: Fraction
    admission_rate: Fraction
    calibration_deficit_rate: Fraction
    insufficient_evidence_rate: Fraction
    assumption_violation_rate: Fraction


def summarize_admission(results: tuple[ClientEvaluationResult, ...]) -> AdmissionSummary:
    """Summarize protocol results into an admission document."""
    if not results:
        raise ValueError("Admission summary requires clients")
    n = len(results)

    def rate(count: NonNegativeCount) -> Fraction:
        return count / n

    return AdmissionSummary(
        client_count=n,
        readiness_rate=rate(
            sum(item.readiness.plan.state is CalibrationReadinessState.READY for item in results)
        ),
        low_mismatch_rate=rate(
            sum(item.mismatch.outcome is MismatchOutcome.LOW for item in results)
        ),
        high_mismatch_rate=rate(
            sum(item.mismatch.outcome is MismatchOutcome.HIGH for item in results)
        ),
        admission_rate=rate(
            sum(item.decision.state is DecisionState.PERSONALIZED for item in results)
        ),
        calibration_deficit_rate=rate(
            sum(item.decision.state is DecisionState.CALIBRATION_DEFICIT for item in results)
        ),
        insufficient_evidence_rate=rate(
            sum(
                item.decision.state is DecisionState.MISMATCH_EVIDENCE_INSUFFICIENT
                for item in results
            )
        ),
        assumption_violation_rate=rate(
            sum(item.decision.state is DecisionState.ASSUMPTION_VIOLATION for item in results)
        ),
    )


def _mean_defined(values: Iterable[Metric | None]) -> Metric | None:
    defined = [value for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


def aggregate_policy(
    policy: PolicyId,
    evaluations: tuple[PolicyEvaluation, ...],
) -> FederationMetrics:
    """Federation-level metrics for one policy across clients."""
    rows = [
        row.metrics
        for row in evaluations
        if row.policy is policy
        and row.status is PolicyEvaluationStatus.EVALUATED
        and row.metrics is not None
    ]
    if not rows:
        raise ValueError(f"No evaluated client metrics for {policy.value}")
    fprs = np.asarray([row.fpr for row in rows], dtype=np.float64)
    return FederationMetrics(
        policy=policy,
        client_count=len(rows),
        mebe=float(np.mean([row.band_error for row in rows])),
        high_excess=float(np.max([row.high_excess for row in rows])),
        band_violation_rate=float(np.mean([row.band_violation for row in rows])),
        mafe=float(np.mean([row.absolute_fpr_error for row in rows])),
        max_fpr=float(np.max(fprs)),
        fpr_iqr=float(np.percentile(fprs, 75) - np.percentile(fprs, 25)),
        attack_balanced_macro_tpr=_mean_defined([row.attack_balanced_tpr for row in rows]),
        macro_tpr=_mean_defined([row.tpr for row in rows]),
        worst_client_tpr=min(
            (value for value in (row.tpr for row in rows) if value is not None),
            default=None,
        ),
        worst_client_attack_balanced_tpr=min(
            (value for value in (row.attack_balanced_tpr for row in rows) if value is not None),
            default=None,
        ),
        mean_f1=_mean_defined([row.f1 for row in rows]),
    )


def assert_ranking_metric_invariance(
    evaluations: tuple[PolicyEvaluation, ...], tolerance: Tolerance
) -> None:
    """Verify AUROC/AUPRC invariance across policies."""
    by_client: dict[ClientId, list[PolicyEvaluation]] = {}
    for row in evaluations:
        if row.metrics is not None:
            by_client.setdefault(row.client_id, []).append(row)
    for client_id, rows in by_client.items():
        auroc_values = [row.metrics.auroc for row in rows if row.metrics is not None]
        auprc_values = [row.metrics.auprc for row in rows if row.metrics is not None]
        if auroc_values and max(auroc_values) - min(auroc_values) > tolerance:
            raise RuntimeError(f"Ranking AUROC changed across policies for {client_id}")
        if auprc_values and max(auprc_values) - min(auprc_values) > tolerance:
            raise RuntimeError(f"Ranking AUPRC changed across policies for {client_id}")


class ClientEvaluation:
    """Compose reference estimation, readiness, mismatch evidence, and decision."""

    def __init__(
        self,
        reference_estimator: ReferenceEstimator | None = None, #TODO: looks like dead code. Check if needs to be wired or something
        readiness_evaluator: CalibrationReadinessEvaluator | None = None,
        mismatch_evaluator: ReferenceMismatchEvaluator | None = None,
        decision_engine: DeploymentDecision | None = None,
        readiness_cache: ReadinessPlanCache | None = None,
    ) -> None:
        self.readiness_evaluator = readiness_evaluator or CalibrationReadinessEvaluator()
        self.mismatch_evaluator = mismatch_evaluator or ReferenceMismatchEvaluator()
        self.decision_engine = decision_engine or DeploymentDecision()
        self.readiness_cache = readiness_cache or ReadinessPlanCache()

    def estimate_reference(
        self,
        reference_scores: Mapping[ClientId, np.ndarray],
        config: ProtocolConfig,
    ) -> ReferenceThreshold:
        return build_reference_threshold(reference_scores, config.alpha)

    def precompute_readiness(
        self,
        sample_count: SampleCount,
        config: ProtocolConfig,
    ) -> ReadinessPlan:
        return self.readiness_cache.precompute(
            sample_count=sample_count,
            band=config.band,
            assurance=config.readiness_assurance,
        )

    def require_readiness(
        self,
        sample_count: SampleCount,
        config: ProtocolConfig,
    ) -> ReadinessPlan:
        return self.readiness_cache.require(
            sample_count,
            config.band,
            config.readiness_assurance,
        )

    def evaluate_client(
        self,
        client_id: ClientId,
        reference: ReferenceThreshold,
        calibration_scores: np.ndarray,
        mismatch_scores: np.ndarray,
        config: ProtocolConfig,
        readiness_plan: ReadinessPlan | None = None,
    ) -> ClientEvaluationResult:
        plan = readiness_plan or self.require_readiness(
            len(calibration_scores),
            config,
        )
        readiness = self.readiness_evaluator.evaluate(calibration_scores, plan)
        mismatch = self.mismatch_evaluator.evaluate(
            scores=mismatch_scores,
            reference_threshold=reference.value,
            band=config.band,
            confidence=config.mismatch_confidence,
        )
        decision = self.decision_engine.decide(
            reference=reference,
            readiness=readiness,
            mismatch=mismatch,
            reject_calibration_ties=config.reject_calibration_ties,
        )
        return ClientEvaluationResult(
            client_id=client_id,
            reference=reference,
            readiness=readiness,
            mismatch=mismatch,
            decision=decision,
        )
