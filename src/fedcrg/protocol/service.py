"""Composition service for reference estimation, evidence, readiness, and decision."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from fedcrg.config.models import ProtocolConfig
from fedcrg.core.ids import ClientId
from fedcrg.protocol.decision import ThresholdDecisionEngine
from fedcrg.protocol.mismatch import ReferenceMismatchEvaluator
from fedcrg.protocol.readiness import CalibrationReadinessEvaluator, ReadinessPlanCache
from fedcrg.protocol.reference import ReferenceThresholdEstimator
from fedcrg.protocol.results import ClientProtocolResult, ReadinessPlan, ReferenceThreshold


class FedCRGProtocol:
    def __init__(
        self,
        reference_estimator: ReferenceThresholdEstimator | None = None,
        readiness_evaluator: CalibrationReadinessEvaluator | None = None,
        mismatch_evaluator: ReferenceMismatchEvaluator | None = None,
        decision_engine: ThresholdDecisionEngine | None = None,
        readiness_cache: ReadinessPlanCache | None = None,
    ) -> None:
        self.reference_estimator = reference_estimator or ReferenceThresholdEstimator()
        self.readiness_evaluator = readiness_evaluator or CalibrationReadinessEvaluator()
        self.mismatch_evaluator = mismatch_evaluator or ReferenceMismatchEvaluator()
        self.decision_engine = decision_engine or ThresholdDecisionEngine()
        self.readiness_cache = readiness_cache or ReadinessPlanCache()

    @classmethod
    def with_persistent_readiness_cache(cls, path: Path) -> "FedCRGProtocol":
        return cls(readiness_cache=ReadinessPlanCache(path))

    def estimate_reference(
        self,
        reference_scores: Mapping[ClientId, np.ndarray],
        config: ProtocolConfig,
    ) -> ReferenceThreshold:
        return self.reference_estimator.estimate(reference_scores, config.alpha)

    def precompute_readiness(
        self,
        sample_count: int,
        config: ProtocolConfig,
    ) -> ReadinessPlan:
        """Populate a pre-data plan table entry before outcome evidence is inspected."""

        return self.readiness_cache.precompute(
            sample_count=sample_count,
            band=config.band,
            assurance=config.readiness_assurance,
        )

    def require_readiness(
        self,
        sample_count: int,
        config: ProtocolConfig,
    ) -> ReadinessPlan:
        """Read a frozen plan; never optimize a rank during real-data evaluation."""

        return self.readiness_cache.require(
            sample_count,
            config.band,
            config.readiness_assurance,
        )

    def evaluate_client(
        self,
        client_id: ClientId,
        reference: ReferenceThreshold,
        calibration_scores: np.ndarray,
        mismatch_scores: np.ndarray,
        config: ProtocolConfig,
        readiness_plan: ReadinessPlan | None = None,
    ) -> ClientProtocolResult:
        plan = readiness_plan or self.require_readiness(
            len(calibration_scores),
            config,
        )
        readiness = self.readiness_evaluator.evaluate(calibration_scores, plan)
        mismatch = self.mismatch_evaluator.evaluate(
            scores=mismatch_scores,
            reference_threshold=reference.value,
            band=config.band,
            confidence=config.mismatch_confidence,
        )
        decision = self.decision_engine.decide(
            reference=reference,
            readiness=readiness,
            mismatch=mismatch,
            reject_calibration_ties=config.reject_calibration_ties,
        )
        return ClientProtocolResult(
            client_id=client_id,
            reference=reference,
            readiness=readiness,
            mismatch=mismatch,
            decision=decision,
        )
