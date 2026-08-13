"""Federation summaries of operating-point admission states."""

from __future__ import annotations

from dataclasses import dataclass

from fedcrg.domain.enums import CalibrationReadinessState, DecisionState, MismatchOutcome
from fedcrg.decision.results import ClientEvaluationResult


@dataclass(frozen=True, slots=True)
class AdmissionSummary:
    client_count: int
    readiness_rate: float
    low_mismatch_rate: float
    high_mismatch_rate: float
    admission_rate: float
    calibration_deficit_rate: float
    insufficient_evidence_rate: float
    assumption_violation_rate: float


def summarize_admission(results: tuple[ClientEvaluationResult, ...]) -> AdmissionSummary:
    if not results:
        raise ValueError("Admission summary requires clients")
    n = len(results)

    def rate(count: int) -> float:
        return count / n

    return AdmissionSummary(
        client_count=n,
        readiness_rate=rate(
            sum(item.readiness.plan.state is CalibrationReadinessState.READY for item in results)
        ),
        low_mismatch_rate=rate(
            sum(item.mismatch.outcome is MismatchOutcome.LOW for item in results)
        ),
        high_mismatch_rate=rate(
            sum(item.mismatch.outcome is MismatchOutcome.HIGH for item in results)
        ),
        admission_rate=rate(
            sum(item.decision.state is DecisionState.PERSONALIZED for item in results)
        ),
        calibration_deficit_rate=rate(
            sum(item.decision.state is DecisionState.CALIBRATION_DEFICIT for item in results)
        ),
        insufficient_evidence_rate=rate(
            sum(
                item.decision.state is DecisionState.MISMATCH_EVIDENCE_INSUFFICIENT
                for item in results
            )
        ),
        assumption_violation_rate=rate(
            sum(item.decision.state is DecisionState.ASSUMPTION_VIOLATION for item in results)
        ),
    )
