"""Configuration for the FedCRG decision method itself."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from fedcrg.domain.enums import ProtocolId
from fedcrg.domain.values import OperatingBand


class ProtocolConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid", "use_enum_values": False}

    id: Literal[ProtocolId.FEDCRG] = ProtocolId.FEDCRG
    version: Literal["2.0"] = "2.0"
    alpha: float = Field(default=0.01, gt=0.0, lt=1.0)
    rho: float = Field(default=0.50, ge=0.0)
    readiness_assurance: float = Field(default=0.95, gt=0.0, lt=1.0)
    mismatch_confidence: float = Field(default=0.95, gt=0.0, lt=1.0)
    strict_exceedance: Literal[True] = True
    reject_calibration_ties: Literal[True] = True

    @property
    def band(self) -> OperatingBand:
        return OperatingBand(
            lower=max(0.0, self.alpha * (1.0 - self.rho)),
            upper=min(1.0, self.alpha * (1.0 + self.rho)),
        )
