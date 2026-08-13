"""Comparator that reports the FedCRG federation reference threshold unmodified."""

from __future__ import annotations

from fedcrg.decision.evidence import BenignPolicyEvidence


def reference_quantile(client: BenignPolicyEvidence) -> float:
    return client.evaluation.reference.value
