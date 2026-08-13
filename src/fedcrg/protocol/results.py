"""Typed outputs for the operating-point governance protocol."""

from __future__ import annotations

from dataclasses import dataclass

from fedcrg.domain.enums import (
    CalibrationReadinessState,
    DecisionReason,
    DecisionState,
    MismatchOutcome,
    ThresholdSource,
)
from fedcrg.domain.identifiers import ClientId
from fedcrg.domain.values import ConfidenceInterval, OperatingBand


@dataclass(frozen=True, slots=True)
class ReferenceThreshold:
    value: float
    rank: int
    sample_count: int
    client_count: int
    samples_per_client: int


@dataclass(frozen=True, slots=True)
class ReadinessPlan:
    sample_count: int
    rank: int
    coverage_probability: float
    state: CalibrationReadinessState
    band: OperatingBand
    assurance: float


@dataclass(frozen=True, slots=True)
class ContinuityDiagnostics:
    unique_score_fraction: float
    duplicate_count: int
    selected_threshold_multiplicity: int
    minimum_positive_spacing: float | None


@dataclass(frozen=True, slots=True)
class CalibrationReadiness:
    plan: ReadinessPlan
    threshold: float | None
    diagnostics: ContinuityDiagnostics

    @property
    def tie_count(self) -> int:
        return self.diagnostics.selected_threshold_multiplicity


@dataclass(frozen=True, slots=True)
class MismatchEvidence:
    sample_count: int
    exceedance_count: int
    estimated_fpr: float
    interval: ConfidenceInterval
    outcome: MismatchOutcome
    minimum_sample_count: int | None
    p_low: float | None
    p_high: float
    high_side_only: bool = False


@dataclass(frozen=True, slots=True)
class ThresholdDecision:
    state: DecisionState
    threshold: float
    source: ThresholdSource
    reason: DecisionReason
    tie_count: int


@dataclass(frozen=True, slots=True)
class ClientProtocolResult:
    client_id: ClientId
    reference: ReferenceThreshold
    readiness: CalibrationReadiness
    mismatch: MismatchEvidence
    decision: ThresholdDecision
