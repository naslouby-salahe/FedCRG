"""Composition service for the operating-point protocol."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from fedcrg.config.models import ProtocolConfig
from fedcrg.protocol.decision import ThresholdDecisionEngine
from fedcrg.protocol.mismatch import ReferenceMismatchEvaluator
from fedcrg.protocol.readiness import CalibrationReadinessEvaluator, CalibrationReadinessPlanner
from fedcrg.protocol.reference import ReferenceThresholdEstimator
from fedcrg.protocol.results import ClientProtocolResult, ReferenceThreshold


class FedCRGProtocol:
    def __init__(
        self,
        reference_estimator: ReferenceThresholdEstimator | None = None,
        readiness_planner: CalibrationReadinessPlanner | None = None,
        readiness_evaluator: CalibrationReadinessEvaluator | None = None,
        mismatch_evaluator: ReferenceMismatchEvaluator | None = None,
        decision_engine: ThresholdDecisionEngine | None = None,
    ) -> None:
        self.reference_estimator = reference_estimator or ReferenceThresholdEstimator()
        self.readiness_planner = readiness_planner or CalibrationReadinessPlanner()
        self.readiness_evaluator = readiness_evaluator or CalibrationReadinessEvaluator()
        self.mismatch_evaluator = mismatch_evaluator or ReferenceMismatchEvaluator()
        self.decision_engine = decision_engine or ThresholdDecisionEngine()

    def estimate_reference(self, reference_scores: Mapping[str, np.ndarray], config: ProtocolConfig) -> ReferenceThreshold:
        return self.reference_estimator.estimate(reference_scores, config.alpha)

    def evaluate_client(self, client_id: str, reference: ReferenceThreshold, calibration_scores: np.ndarray, mismatch_scores: np.ndarray, config: ProtocolConfig) -> ClientProtocolResult:
        plan = self.readiness_planner.plan(sample_count=len(calibration_scores), band=config.band, assurance=config.readiness_assurance)
        readiness = self.readiness_evaluator.evaluate(calibration_scores, plan)
        mismatch = self.mismatch_evaluator.evaluate(scores=mismatch_scores, reference_threshold=reference.value, band=config.band, confidence=config.mismatch_confidence)
        decision = self.decision_engine.decide(reference=reference, readiness=readiness, mismatch=mismatch, reject_calibration_ties=config.reject_calibration_ties)
        return ClientProtocolResult(client_id=client_id, reference=reference, readiness=readiness, mismatch=mismatch, decision=decision)
