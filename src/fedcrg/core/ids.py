"""Strong identifiers and deterministic run identity."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from fedcrg.core.enums import ExperimentId

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class ClientId:
    value: str

    def __post_init__(self) -> None:
        if not _ID_PATTERN.fullmatch(self.value):
            raise ValueError(f"Invalid client id: {self.value!r}")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
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
    def derive(
        cls,
        experiment_id: ExperimentId,
        config_hash: str,
        model_seed: int,
        calibration_seed: int,
    ) -> "RunId":
        payload = f"{experiment_id.value}|{config_hash}|{model_seed}|{calibration_seed}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return cls(f"{experiment_id.value}__m{model_seed}__c{calibration_seed}__{digest}")

    def __str__(self) -> str:
        return self.value
