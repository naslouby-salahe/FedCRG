"""Typed outputs of threshold-comparator selection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from fedcrg.domain.enums import FailureCode, PolicyId
from fedcrg.domain.identifiers import ClientId


class InformationRegime(StrEnum):
    BENIGN_ONLY = "benign_only"
    SUPERVISED_DEVELOPMENT = "supervised_development"
    FINAL_TEST_ORACLE = "final_test_oracle"


@dataclass(frozen=True, slots=True)
class ClientPolicyThreshold:
    policy: PolicyId
    client_id: ClientId
    threshold: float | None


@dataclass(frozen=True, slots=True)
class UndefinedPolicyReason:
    policy: PolicyId
    reason: FailureCode


@dataclass(frozen=True, slots=True)
class PolicyThresholdSet:
    """Thresholds frozen before final-test evidence is opened."""

    entries: tuple[ClientPolicyThreshold, ...]
    undefined_reasons: tuple[UndefinedPolicyReason, ...]
    shrinkage_n0: int | None

    def for_client(self, policy: PolicyId, client_id: ClientId) -> float | None:
        if policy is PolicyId.ORACLE_TEST:
            raise ValueError("oracle_test is not available before final-test evidence opens")
        for entry in self.entries:
            if entry.policy is policy and entry.client_id == client_id:
                return entry.threshold
        raise KeyError(f"No threshold for {policy.value}/{client_id.value}")
