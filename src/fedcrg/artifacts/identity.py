"""Construction of immutable scientific run identities."""

from __future__ import annotations

from fedcrg.config.models import ExperimentConfig
from fedcrg.core.enums import DetectorId, PolicyId
from fedcrg.core.ids import CalibrationSeed, ModelSeed, RunId


class RunIdentityFactory:
    """Build path-safe run IDs from one fully resolved scientific configuration.

    The human-readable prefix follows the pre-registration convention. A short
    scientific-config digest is appended so distinct experiment specifications can
    never map to the same evidence directory.
    """

    @staticmethod
    def for_policy_cell(
        config: ExperimentConfig,
        model_seed: ModelSeed | int,
        calibration_seed: CalibrationSeed | int,
        policy: PolicyId,
    ) -> RunId:
        alpha_ppm = round(config.protocol.alpha * 1_000_000)
        rho_bp = round(config.protocol.rho * 10_000)
        assurance_bp = round(config.protocol.readiness_assurance * 10_000)
        confidence_bp = round(config.protocol.mismatch_confidence * 10_000)
        detector = (
            "ae" if config.detector.id is DetectorId.AUTOENCODER else config.detector.id.value
        )
        prefix = (
            f"{config.dataset.id.value}__{detector}__ms{int(model_seed)}__"
            f"cs{int(calibration_seed)}__a{alpha_ppm}__r{rho_bp}__"
            f"ga{assurance_bp}__gb{confidence_bp}__{policy.value.lower()}"
        )
        return RunId(f"{prefix}__cfg{config.config_hash[:12]}")
