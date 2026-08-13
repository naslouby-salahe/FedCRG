"""Equal-client federation aggregation."""

from __future__ import annotations

import numpy as np

from fedcrg.domain.enums import PolicyEvaluationStatus, PolicyId
from fedcrg.domain.identifiers import ClientId
from fedcrg.evaluation.evaluation_results import FederationMetrics, PolicyEvaluation


def _mean_defined(values: list[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


def aggregate_policy(
    policy: PolicyId,
    evaluations: tuple[PolicyEvaluation, ...],
) -> FederationMetrics:
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
    evaluations: tuple[PolicyEvaluation, ...], tolerance: float
) -> None:
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
