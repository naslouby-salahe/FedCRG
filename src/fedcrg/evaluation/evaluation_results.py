"""Typed client and federation metric records."""

from __future__ import annotations

from dataclasses import dataclass

from fedcrg.domain.enums import PolicyEvaluationStatus, PolicyId
from fedcrg.domain.identifiers import ClientId
from fedcrg.domain.values import ConfidenceInterval
from fedcrg.decision.results import ClientEvaluationResult


@dataclass(frozen=True, slots=True)
class ClientMetrics:
    benign_n: int
    attack_n: int
    fp: int
    tn: int
    tp: int
    fn: int
    fpr: float
    tpr: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    balanced_accuracy: float | None
    auroc: float
    auprc: float
    band_error: float
    high_excess: float
    band_violation: float
    absolute_fpr_error: float
    attack_balanced_tpr: float | None
    fpr_reference_interval: ConfidenceInterval


@dataclass(frozen=True, slots=True)
class PolicyEvaluation:
    client_id: ClientId
    policy: PolicyId
    threshold: float | None
    status: PolicyEvaluationStatus
    metrics: ClientMetrics | None


@dataclass(frozen=True, slots=True)
class FederationMetrics:
    policy: PolicyId
    client_count: int
    mebe: float
    high_excess: float
    band_violation_rate: float
    mafe: float
    max_fpr: float
    fpr_iqr: float
    attack_balanced_macro_tpr: float | None
    macro_tpr: float | None
    worst_client_tpr: float | None
    worst_client_attack_balanced_tpr: float | None
    mean_f1: float | None


@dataclass(frozen=True, slots=True)
class EvaluationBundle:
    clients: tuple[PolicyEvaluation, ...]
    federations: tuple[FederationMetrics, ...]
    protocol_results: tuple[ClientEvaluationResult, ...]
    shrinkage_n0: int | None


@dataclass(frozen=True, slots=True)
class UtilityAssessment:
    anchor: float | None
    method_value: float | None
    difference: float | None
    margin: float
    preserved: bool | None
