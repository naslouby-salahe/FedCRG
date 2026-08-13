"""Strong identifiers used across the FedCRG domain.

This module is deliberately dependency-free: core identifiers must not know about
configuration, filesystem layout, CLI parsing, or experiment orchestration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import NewType

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

ModelSeed = NewType("ModelSeed", int)
CalibrationSeed = NewType("CalibrationSeed", int)


@dataclass(frozen=True, slots=True, order=True)
class ClientId:
    value: str

    def __post_init__(self) -> None:
        if not _ID_PATTERN.fullmatch(self.value):
            raise ValueError(f"Invalid client id: {self.value!r}")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True, order=True)
class RowId:
    """Stable SHA-256 identity of one source dataset row."""

    value: str

    def __post_init__(self) -> None:
        if not _SHA256_PATTERN.fullmatch(self.value):
            raise ValueError("Row IDs must be 64 lowercase hexadecimal characters")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True, order=True)
class AttackGroupId:
    value: str

    def __post_init__(self) -> None:
        normalized = self.value.strip()
        if not normalized:
            raise ValueError("Attack-group identifiers must be non-empty")
        if normalized != self.value:
            raise ValueError("Attack-group identifiers must be normalized before construction")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True, order=True)
class Sha256:
    value: str

    def __post_init__(self) -> None:
        if not _SHA256_PATTERN.fullmatch(self.value):
            raise ValueError("SHA-256 values must be 64 lowercase hexadecimal characters")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True, order=True)
class RunId:
    value: str

    def __post_init__(self) -> None:
        if not self.value or " " in self.value or "/" in self.value or "\\" in self.value:
            raise ValueError("Run IDs must be non-empty, path-safe, and contain no spaces")

    def __str__(self) -> str:
        return self.value
