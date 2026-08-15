"""Classification and ranking metrics for threshold policies, plus aggregation across clients and policies."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import average_precision_score, roc_auc_score

from fedcrg.config import ProtocolConfig
from fedcrg.statistics import iqr
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
    Percentage,
    PolicyEvaluationStatus,
    PolicyId,
    PositiveCount,
    SampleCount,
    Threshold,
    Tolerance,
    Tpr,
)

Frozen = ConfigDict(frozen=True)


class ConfusionMatrix(BaseModel):
    """Counts of true/false positives and negatives for one threshold."""

    model_config = Frozen

    tp: ExceedanceCount
    tn: ExceedanceCount
    fp: ExceedanceCount
    fn: ExceedanceCount


def confusion_matrix(
    scores: np.ndarray, labels: np.ndarray, threshold: Threshold
) -> ConfusionMatrix:
    """Classify each score as positive iff it strictly exceeds `threshold` and tally against the labels."""
    values = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)

    if values.shape != targets.shape or values.ndim != 1:
        raise ValueError("scores and labels must be aligned one-dimensional arrays")
    if not np.isfinite(values).all():
        raise ValueError("NONFINITE_SCORE")
    if np.any((targets != 0) & (targets != 1)):
        raise ValueError("Labels must be binary 0/1")

    predictions = values > threshold
    idx = targets * 2 + predictions
    counts = np.bincount(idx, minlength=4)

    return ConfusionMatrix(
        tn=int(counts[0]),
        fp=int(counts[1]),
        fn=int(counts[2]),
        tp=int(counts[3]),
    )


def _ratio(numerator: NonNegativeCount, denominator: NonNegativeCount) -> Fraction | None:
    # zero denominator (e.g. no malicious records) is undefined, not 0
    return float(numerator / denominator) if denominator else None


def fpr(cm: ConfusionMatrix) -> Fpr | None:
    """False-positive rate among actual negatives, or None if there are none."""
    return _ratio(cm.fp, cm.fp + cm.tn)


def tpr(cm: ConfusionMatrix) -> Tpr | None:
    """True-positive rate (recall) among actual positives, or None if there are none."""
    return _ratio(cm.tp, cm.tp + cm.fn)


def precision(cm: ConfusionMatrix) -> Fpr | None:
    """Fraction of predicted positives that are true positives, or None if there are no predicted positives."""
    return _ratio(cm.tp, cm.tp + cm.fp)


def recall(cm: ConfusionMatrix) -> Tpr | None:
    """Alias for `tpr`."""
    return tpr(cm)


def f1(cm: ConfusionMatrix) -> Metric | None:
    """Harmonic mean of precision and recall, or None if either is undefined."""
    p = precision(cm)
    r = recall(cm)
    if p is None or r is None or not p + r:
        return None
    return 2.0 * p * r / (p + r)


def balanced_accuracy(cm: ConfusionMatrix) -> Fpr | None:
    """Mean of sensitivity and specificity, or None if either is undefined."""
    sensitivity = tpr(cm)
    specificity = _ratio(cm.tn, cm.tn + cm.fp)
    if sensitivity is None or specificity is None:
        return None
    return 0.5 * (sensitivity + specificity)


def band_error(fpr_value: Fpr, band: OperatingBand) -> Fraction:
    """Signed distance from `fpr_value` to the nearer edge of `band`, 0 if already inside."""
    if fpr_value < band.lower:
        return band.lower - fpr_value
    if fpr_value > band.upper:
        return fpr_value - band.upper
    return 0.0


def high_excess(fpr_value: Fpr, band: OperatingBand) -> Fraction:
    """Amount by which `fpr_value` exceeds the band's upper edge, 0 if not above it."""
    return max(0.0, fpr_value - band.upper)


def band_violation(fpr_value: Fpr, band: OperatingBand) -> Fraction:
    """1.0 if `fpr_value` falls outside `band`, else 0.0."""
    return float(not band.contains(fpr_value))


def absolute_fpr_error(fpr_value: Fpr, alpha: Alpha) -> Fraction:
    """Absolute distance between the observed FPR and the target rate `alpha`."""
    return abs(fpr_value - alpha)


def attack_balanced_tpr(
    scores: np.ndarray,
    labels: np.ndarray,
    attack_groups: np.ndarray,
    threshold: Threshold,
) -> Tpr | None:
    """Average recall over the attack subtypes present for this client, so a subtype the client never saw is excluded rather than scored as 0."""
    values = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)

    pos_mask = targets == 1
    if not np.any(pos_mask):
        return None

    pos_groups = np.asarray(attack_groups)[pos_mask].astype(str)
    pos_preds = values[pos_mask] > threshold

    _, inverse_indices = np.unique(pos_groups, return_inverse=True)
    group_counts = np.bincount(inverse_indices)
    tp_counts = np.bincount(inverse_indices, weights=pos_preds)

    return float(np.mean(tp_counts / group_counts))


def auroc(scores: np.ndarray, labels: np.ndarray) -> Fpr:
    """Area under the ROC curve; depends only on score ranking, not any chosen threshold."""
    return float(roc_auc_score(labels, scores))


def auprc(scores: np.ndarray, labels: np.ndarray) -> Fpr:
    """Area under the precision-recall curve; depends only on score ranking, not any chosen threshold."""
    return float(average_precision_score(labels, scores))


class ClientMetrics(BaseModel):
    """Full metric set for one client under one threshold policy."""

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
    """One client's threshold and resulting metrics under one policy, or the reason evaluation could not proceed."""

    model_config = Frozen

    client_id: ClientId
    policy: PolicyId
    threshold: Threshold | None
    status: PolicyEvaluationStatus
    metrics: ClientMetrics | None


class FederationMetrics(BaseModel):
    """Metrics for one policy aggregated across clients. mebe = mean band error; mafe = mean absolute FPR error from target alpha; high_excess is the worst-client high-side excess."""

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
    """Full evaluation output: per-client and per-federation metrics for every policy, plus the underlying readiness/mismatch decisions."""

    model_config = Frozen

    clients: tuple[PolicyEvaluation, ...]
    federations: tuple[FederationMetrics, ...]
    protocol_results: tuple[ClientEvaluationResult, ...]
    shrinkage_n0: PositiveCount | None


class AdmissionSummary(BaseModel):
    """Fleet-wide rates of each readiness/mismatch/decision outcome across clients."""

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
    """Compute fleet-wide readiness/mismatch/decision rates from per-client evaluation results."""
    if not results:
        raise ValueError("Admission summary requires clients")

    n = len(results)
    readiness_count = 0
    low_mismatch_count = 0
    high_mismatch_count = 0
    admission_count = 0
    cal_deficit_count = 0
    insuf_evid_count = 0
    assump_viol_count = 0

    for item in results:
        if item.readiness.plan.state is CalibrationReadinessState.READY:
            readiness_count += 1

        if item.mismatch.outcome is MismatchOutcome.LOW:
            low_mismatch_count += 1
        elif item.mismatch.outcome is MismatchOutcome.HIGH:
            high_mismatch_count += 1

        state = item.decision.state
        if state is DecisionState.PERSONALIZED:
            admission_count += 1
        elif state is DecisionState.CALIBRATION_DEFICIT:
            cal_deficit_count += 1
        elif state is DecisionState.MISMATCH_EVIDENCE_INSUFFICIENT:
            insuf_evid_count += 1
        elif state is DecisionState.ASSUMPTION_VIOLATION:
            assump_viol_count += 1

    return AdmissionSummary(
        client_count=n,
        readiness_rate=readiness_count / n,
        low_mismatch_rate=low_mismatch_count / n,
        high_mismatch_rate=high_mismatch_count / n,
        admission_rate=admission_count / n,
        calibration_deficit_rate=cal_deficit_count / n,
        insufficient_evidence_rate=insuf_evid_count / n,
        assumption_violation_rate=assump_viol_count / n,
    )


def _mean_defined(values: Iterable[Metric | None]) -> Metric | None:
    defined = [value for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


def aggregate_policy(
    policy: PolicyId,
    evaluations: tuple[PolicyEvaluation, ...],
    iqr_percentiles: tuple[Percentage, Percentage],
) -> FederationMetrics:
    """Aggregate per-client metrics for one policy into fleet-wide summary statistics."""
    rows = [
        row.metrics
        for row in evaluations
        if row.policy is policy
        and row.status is PolicyEvaluationStatus.EVALUATED
        and row.metrics is not None
    ]
    if not rows:
        raise ValueError(f"No evaluated client metrics for {policy}")

    fprs = np.fromiter((row.fpr for row in rows), dtype=np.float64, count=len(rows))

    return FederationMetrics(
        policy=policy,
        client_count=len(rows),
        mebe=float(np.mean([row.band_error for row in rows])),
        high_excess=float(np.max([row.high_excess for row in rows])),
        band_violation_rate=float(np.mean([row.band_violation for row in rows])),
        mafe=float(np.mean([row.absolute_fpr_error for row in rows])),
        max_fpr=float(np.max(fprs)),
        fpr_iqr=iqr(fprs, iqr_percentiles),
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
    """AUROC/AUPRC depend only on score ranking, not the chosen threshold, so a given client's values must match across policies."""
    by_client: dict[ClientId, tuple[list[Fpr], list[Fpr]]] = {}

    for row in evaluations:
        if row.metrics is not None:
            auroc_list, auprc_list = by_client.setdefault(row.client_id, ([], []))
            auroc_list.append(row.metrics.auroc)
            auprc_list.append(row.metrics.auprc)

    for client_id, (auroc_values, auprc_values) in by_client.items():
        if auroc_values and max(auroc_values) - min(auroc_values) > tolerance:
            raise RuntimeError(f"Ranking AUROC changed across policies for {client_id}")
        if auprc_values and max(auprc_values) - min(auprc_values) > tolerance:
            raise RuntimeError(f"Ranking AUPRC changed across policies for {client_id}")


class ClientEvaluation:
    """Runs the reference-threshold, readiness, mismatch, and decision steps for one client."""

    def __init__(
        self,
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
        """Build the pooled reference threshold from each client's reference scores."""
        return build_reference_threshold(reference_scores, config.alpha)

    def precompute_readiness(
        self,
        sample_count: SampleCount,
        config: ProtocolConfig,
    ) -> ReadinessPlan:
        """Compute and cache the readiness plan for a given calibration sample size."""
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
        """Look up the precomputed readiness plan for a given calibration sample size, failing if it was never precomputed."""
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
        """Run readiness, mismatch, and decision evaluation for one client and bundle the results."""
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
