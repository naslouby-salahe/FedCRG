"""Validated construction of pre-registered experiment variants."""

from __future__ import annotations

from fedcrg.config.models import ExperimentConfig, ProtocolConfig
from fedcrg.config.validation import validate_experiment_config
from fedcrg.core.enums import PolicyId


class ExperimentVariantFactory:
    """Rebuild complete configs instead of bypassing validators with ``model_copy``."""

    def protocol_variant(
        self,
        config: ExperimentConfig,
        *,
        alpha: float | None = None,
        rho: float | None = None,
        readiness_assurance: float | None = None,
        mismatch_confidence: float | None = None,
    ) -> ExperimentConfig:
        protocol = ProtocolConfig(
            id=config.protocol.id,
            version=config.protocol.version,
            alpha=config.protocol.alpha if alpha is None else alpha,
            rho=config.protocol.rho if rho is None else rho,
            readiness_assurance=(
                config.protocol.readiness_assurance
                if readiness_assurance is None
                else readiness_assurance
            ),
            mismatch_confidence=(
                config.protocol.mismatch_confidence
                if mismatch_confidence is None
                else mismatch_confidence
            ),
            strict_exceedance=config.protocol.strict_exceedance,
            reject_calibration_ties=config.protocol.reject_calibration_ties,
        )
        variant = ExperimentConfig(
            id=config.id,
            protocol=protocol,
            dataset=config.dataset,
            detector=config.detector,
            training=config.training,
            randomness=config.randomness,
            policies=config.policies,
            outputs_root=config.outputs_root,
        )
        validate_experiment_config(variant)
        return variant

    def policy_subset(
        self,
        config: ExperimentConfig,
        policies: tuple[PolicyId, ...],
    ) -> ExperimentConfig:
        variant = ExperimentConfig(
            id=config.id,
            protocol=config.protocol,
            dataset=config.dataset,
            detector=config.detector,
            training=config.training,
            randomness=config.randomness,
            policies=policies,
            outputs_root=config.outputs_root,
        )
        validate_experiment_config(variant)
        return variant
