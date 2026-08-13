"""Benign-only component ablations of the admission protocol."""

from __future__ import annotations

from fedcrg.core.enums import CalibrationReadinessState, MismatchOutcome
from fedcrg.policies.base import BenignPolicyEvidence, empirical_quantile


def readiness_only(client: BenignPolicyEvidence) -> float:
    readiness = client.protocol.readiness
    if (
        readiness.plan.state is CalibrationReadinessState.READY
        and readiness.tie_count == 1
        and readiness.threshold is not None
    ):
        return readiness.threshold
    return client.protocol.reference.value


def mismatch_only(client: BenignPolicyEvidence, alpha: float) -> float:
    if client.protocol.mismatch.outcome in {MismatchOutcome.LOW, MismatchOutcome.HIGH}:
        return empirical_quantile(client.calibration_scores, alpha)
    return client.protocol.reference.value
