"""Small immutable value objects shared by the scientific core."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OperatingBand:
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.lower < self.upper <= 1.0:
            raise ValueError("Operating band must satisfy 0 <= lower < upper <= 1")

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.lower <= self.upper <= 1.0:
            raise ValueError("Confidence interval must lie inside [0, 1]")
