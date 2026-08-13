"""Benign-only ablation retaining mismatch detection but not calibration readiness."""

from __future__ import annotations

from fedcrg.domain.enums import MismatchOutcome
from fedcrg.decision.evidence import BenignPolicyEvidence, empirical_quantile


def mismatch_only(client: BenignPolicyEvidence, alpha: float) -> float:
    if client.evaluation.mismatch.outcome in {MismatchOutcome.LOW, MismatchOutcome.HIGH}:
        return empirical_quantile(client.calibration_scores, alpha)
    return client.evaluation.reference.value
