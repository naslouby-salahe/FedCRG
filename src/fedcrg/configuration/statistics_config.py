"""Configuration for confirmatory statistical analysis choices.

Every field is a locked scientific choice that must be supplied by YAML. The model
declares no scientific defaults.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class StatisticsConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid", "use_enum_values": False}

    bootstrap_replicates: int = Field(gt=0)
    bootstrap_seed: int
    utility_margin: float = Field(gt=0.0)
    familywise_alpha: float = Field(gt=0.0, lt=1.0)
    ranking_invariance_tolerance: float = Field(gt=0.0)
    shrinkage_n0_candidates: tuple[int, ...]
    supervised_threshold_candidates: int = Field(gt=0)

    @property
    def utility_margin_allowance(self) -> float:
        """Signed allowance used by the locked non-inferiority comparison."""
        return -self.utility_margin
