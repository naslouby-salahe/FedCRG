"""Strong identifiers and the pre-registered run-identity convention."""

from __future__ import annotations

import re
from dataclasses import dataclass

from fedcrg.config.models import ExperimentConfig
from fedcrg.core.enums import PolicyId

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True, slots=True, order=True)
class ClientId:
    value: str

    def __post_init__(self) -> None:
        if not _ID_PATTERN.fullmatch(self.value):
            raise ValueError(f"Invalid client id: {self.value!r}")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True, order=True)
class Sha256:
    value: str

    def __post_init__(self) -> None:
        if len(self.value) != 64 or any(c not in "0123456789abcdef" for c in self.value):
            raise ValueError("SHA-256 values must be 64 lowercase hexadecimal characters")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class RunId:
    value: str

    @classmethod
    def for_policy_cell(
        cls,
        config: ExperimentConfig,
        model_seed: int,
        calibration_seed: int,
        policy: PolicyId,
    ) -> "RunId":
        alpha_ppm = round(config.protocol.alpha * 1_000_000)
        rho_bp = round(config.protocol.rho * 10_000)
        assurance_bp = round(config.protocol.readiness_assurance * 10_000)
        confidence_bp = round(config.protocol.mismatch_confidence * 10_000)
        detector = "ae" if config.detector.id.value == "autoencoder" else config.detector.id.value
        value = (
            f"{config.dataset.id.value}__{detector}__ms{model_seed}__cs{calibration_seed}__"
            f"a{alpha_ppm}__r{rho_bp}__ga{assurance_bp}__gb{confidence_bp}__"
            f"{policy.value.lower()}"
        )
        return cls(value)

    def __post_init__(self) -> None:
        if " " in self.value or not self.value:
            raise ValueError("Run IDs must be non-empty and contain no spaces")

    def __str__(self) -> str:
        return self.value
