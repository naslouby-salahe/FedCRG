from __future__ import annotations

import numpy as np
import pytest

from fedcrg.data.splits import AttackGroupAllocation, AttackGroupCount, BaseSplitBuilder
from fedcrg.thresholding.metrics import (
    ClientMetrics,
    PolicyEvaluation,
    auroc,
    auprc,
    assert_ranking_metric_invariance,
)
from fedcrg.types import (
    AttackGroupId,
    ClientId,
    ConfidenceInterval,
    PolicyEvaluationStatus,
    PolicyId,
)
from pydantic import TypeAdapter

_ATTACK_GROUP_ADAPTER = TypeAdapter(AttackGroupId)


@pytest.mark.parametrize(
    ("capacities", "expected"),
    [
        ([200, 200, 200], [167, 167, 166]),
        ([0, 50, 900], [0, 50, 450]),
    ],
)
def test_diad_waterfill_normative_vectors(capacities: list[int], expected: list[int]) -> None:
    groups = tuple(_ATTACK_GROUP_ADAPTER.validate_python(value) for value in ("a", "b", "c"))
    development = BaseSplitBuilder.waterfill_allocation(
        groups=groups,
        capacities=AttackGroupAllocation(
            groups=tuple(
                AttackGroupCount(group=group, count=capacity)
                for group, capacity in zip(groups, capacities, strict=True)
            )
        ),
        development_budget=500,
    )
    assert [development.for_group(group) for group in groups] == expected
    assert sum(development.for_group(group) for group in groups) == 500


def test_ranking_metrics_are_invariant_across_policies() -> None:
    client_id = TypeAdapter(ClientId).validate_python("invariant_client")
    scores = np.array(
        [0.01, 0.02, 0.03, 0.04, 0.05, 0.6, 0.7, 0.8, 0.9, 0.95, 0.1, 0.2, 0.3, 0.4, 0.5]
    )
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    def evaluate(threshold: float) -> ClientMetrics:
        predictions = scores > threshold
        fp = int(np.sum((predictions == 1) & (labels == 0)))
        tn = int(np.sum((predictions == 0) & (labels == 0)))
        tp = int(np.sum((predictions == 1) & (labels == 1)))
        fn = int(np.sum((predictions == 0) & (labels == 1)))
        benign_n = fp + tn
        attack_n = tp + fn
        fpr = fp / benign_n
        tpr = tp / attack_n
        precision = tp / (tp + fp) if tp + fp > 0 else None
        recall = tpr
        f1_value = 2 * precision * recall / (precision + recall) if precision and recall else None
        return ClientMetrics(
            benign_n=benign_n,
            attack_n=attack_n,
            fp=fp,
            tn=tn,
            tp=tp,
            fn=fn,
            fpr=fpr,
            tpr=tpr,
            precision=precision,
            recall=recall,
            f1=f1_value,
            balanced_accuracy=(tpr + (tn / benign_n)) / 2,
            auroc=auroc(scores, labels),
            auprc=auprc(scores, labels),
            band_error=max(0.0, fpr - 0.015),
            high_excess=max(0.0, fpr - 0.015),
            band_violation=1.0 if fpr > 0.015 or fpr < 0.005 else 0.0,
            absolute_fpr_error=abs(fpr - 0.01),
            attack_balanced_tpr=tpr,
            fpr_reference_interval=ConfidenceInterval(lower=0.0, upper=1.0),
        )

    evaluations = (
        PolicyEvaluation(
            client_id=client_id,
            policy=PolicyId.GLOBAL_QUANTILE,
            threshold=0.5,
            status=PolicyEvaluationStatus.EVALUATED,
            metrics=evaluate(0.5),
        ),
        PolicyEvaluation(
            client_id=client_id,
            policy=PolicyId.LOCAL_QUANTILE,
            threshold=0.2,
            status=PolicyEvaluationStatus.EVALUATED,
            metrics=evaluate(0.2),
        ),
        PolicyEvaluation(
            client_id=client_id,
            policy=PolicyId.FEDCRG,
            threshold=0.9,
            status=PolicyEvaluationStatus.EVALUATED,
            metrics=evaluate(0.9),
        ),
    )
    records = [row.metrics for row in evaluations]
    assert all(record is not None for record in records)
    auroc_values = [record.auroc for record in records if record is not None]
    auprc_values = [record.auprc for record in records if record is not None]
    assert max(auroc_values) - min(auroc_values) <= 1e-12
    assert max(auprc_values) - min(auprc_values) <= 1e-12
    assert_ranking_metric_invariance(evaluations, tolerance=1e-12)
