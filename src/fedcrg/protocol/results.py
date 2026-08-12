"""Typed protocol results."""

from __future__ import annotations
from dataclasses import dataclass
from fedcrg.core.enums import CalibrationReadinessState, DecisionReason, DecisionState, MismatchOutcome, ThresholdSource
from fedcrg.core.types import ConfidenceInterval, OperatingBand

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
class CalibrationReadiness:
    plan: ReadinessPlan
    threshold: float | None
    tie_count: int

@dataclass(frozen=True, slots=True)
class MismatchEvidence:
    sample_count: int
    exceedance_count: int
    estimated_fpr: float
    interval: ConfidenceInterval | None
    outcome: MismatchOutcome
    minimum_sample_count: int
    p_low: float | None
    p_high: float | None

@dataclass(frozen=True, slots=True)
class ThresholdDecision:
    state: DecisionState
    threshold: float
    source: ThresholdSource
    reason: DecisionReason

@dataclass(frozen=True, slots=True)
class ClientProtocolResult:
    client_id: str
    reference: ReferenceThreshold
    readiness: CalibrationReadiness
    mismatch: MismatchEvidence
    decision: ThresholdDecision
