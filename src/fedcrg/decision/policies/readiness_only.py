"""Benign-only ablation retaining calibration readiness but not mismatch detection."""

from __future__ import annotations

from fedcrg.domain.enums import CalibrationReadinessState
from fedcrg.decision.evidence import BenignPolicyEvidence


def readiness_only(client: BenignPolicyEvidence) -> float:
    readiness = client.evaluation.readiness
    if (
        readiness.plan.state is CalibrationReadinessState.READY
        and readiness.tie_count == 1
        and readiness.threshold is not None
    ):
        return readiness.threshold
    return client.evaluation.reference.value
