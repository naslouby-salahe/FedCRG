"""Benign-only per-client local quantile comparator."""

from __future__ import annotations

from fedcrg.decision.evidence import BenignPolicyEvidence, empirical_quantile


def local_quantile(client: BenignPolicyEvidence, alpha: float) -> float:
    return empirical_quantile(client.full_policy_budget, alpha)
