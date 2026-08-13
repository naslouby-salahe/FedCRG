"""Locked supervised-development comparator choosing between global and local F1.

Requires an explicit supervised evidence view so attack development labels cannot
accidentally enter benign-only threshold-comparator code.
"""

from __future__ import annotations

from fedcrg.metrics.classification import confusion_matrix, f1
from fedcrg.thresholds.evidence import SupervisedDevelopmentEvidence


def f1_at_threshold(client: SupervisedDevelopmentEvidence, threshold: float) -> float:
    value = f1(confusion_matrix(client.scores, client.labels, threshold))
    return -1.0 if value is None else value


def dev_local_global(
    client: SupervisedDevelopmentEvidence,
    global_threshold: float,
    local_threshold: float,
) -> float:
    global_score = f1_at_threshold(client, global_threshold)
    local_score = f1_at_threshold(client, local_threshold)
    return local_threshold if local_score > global_score else global_threshold
