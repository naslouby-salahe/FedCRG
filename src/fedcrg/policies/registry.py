"""One authoritative policy registry and information-regime-safe selector."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from fedcrg.config.models import ProtocolConfig
from fedcrg.core.enums import FailureCode, PolicyId
from fedcrg.core.ids import ClientId
from fedcrg.policies.attack_aware import (
    dev_local_global,
    summary_statistic_threshold,
    supervised_global_f1,
)
from fedcrg.policies.base import PolicySelectionInputs
from fedcrg.policies.personalized import mismatch_only, readiness_only
from fedcrg.policies.quantile import global_quantile, local_quantile, three_sigma
from fedcrg.policies.shrinkage import shrinkage, tune_shrinkage


class InformationRegime(StrEnum):
    BENIGN_ONLY = "benign_only"
    SUPERVISED_DEVELOPMENT = "supervised_development"
    FINAL_TEST_ORACLE = "final_test_oracle"


@dataclass(frozen=True, slots=True)
class PolicyDefinition:
    id: PolicyId
    information_regime: InformationRegime
    deployable: bool


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
    """Thresholds frozen before final-test evidence is opened.

    ``ORACLE-TEST`` is intentionally absent. Its diagnostic threshold can only be
    computed at the final-test evaluation boundary.
    """

    entries: tuple[ClientPolicyThreshold, ...]
    undefined_reasons: tuple[UndefinedPolicyReason, ...]
    shrinkage_n0: int

    def for_client(self, policy: PolicyId, client_id: ClientId) -> float | None:
        if policy is PolicyId.ORACLE_TEST:
            raise ValueError("ORACLE-TEST is not available before final-test evidence opens")
        for entry in self.entries:
            if entry.policy is policy and entry.client_id == client_id:
                return entry.threshold
        raise KeyError(f"No threshold for {policy.value}/{client_id.value}")


class PolicyRegistry:
    def __init__(self) -> None:
        supervised = {
            PolicyId.DEV_F1_SELECT,
            PolicyId.SUMMARY_STATISTIC_SELECT,
            PolicyId.SUPERVISED_F1,
        }
        self._definitions = {
            policy: PolicyDefinition(
                id=policy,
                information_regime=(
                    InformationRegime.FINAL_TEST_ORACLE
                    if policy is PolicyId.ORACLE_TEST
                    else InformationRegime.SUPERVISED_DEVELOPMENT
                    if policy in supervised
                    else InformationRegime.BENIGN_ONLY
                ),
                deployable=policy not in supervised | {PolicyId.ORACLE_TEST},
            )
            for policy in PolicyId
        }

    def get(self, policy_id: PolicyId) -> PolicyDefinition:
        return self._definitions[policy_id]

    def all_ids(self) -> tuple[PolicyId, ...]:
        return tuple(self._definitions)

    def assert_exact_protocol_registry(self) -> None:
        if set(self._definitions) != set(PolicyId) or len(self._definitions) != 12:
            raise RuntimeError("Policy registry must contain exactly 12 protocol policies")


class FederationPolicySelector:
    """Select every non-oracle threshold without receiving final-test evidence."""

    def select(
        self,
        clients: tuple[PolicySelectionInputs, ...],
        protocol: ProtocolConfig,
    ) -> PolicyThresholdSet:
        if not clients:
            raise ValueError("At least one client is required for policy selection")

        benign_clients = tuple(client.benign for client in clients)
        supervised_clients = tuple(client.supervised for client in clients)
        global_q = global_quantile(benign_clients, protocol.alpha)
        local_q = {
            client.client_id: local_quantile(client.benign, protocol.alpha)
            for client in clients
        }
        shrinkage_n0 = tune_shrinkage(benign_clients, protocol.alpha)
        three_sigma_threshold = three_sigma(benign_clients)
        summary_threshold = summary_statistic_threshold(supervised_clients)
        supervised_threshold = supervised_global_f1(supervised_clients)

        entries: list[ClientPolicyThreshold] = []
        undefined: list[UndefinedPolicyReason] = []
        if summary_threshold is None:
            undefined.append(
                UndefinedPolicyReason(
                    PolicyId.SUMMARY_STATISTIC_SELECT,
                    FailureCode.SUMMARY_STATISTIC_COMPARATOR_UNDEFINED,
                )
            )

        for client in clients:
            cid = client.client_id
            reference = client.benign.protocol.reference.value
            method_threshold = client.benign.protocol.decision.threshold
            thresholds = (
                (PolicyId.REFERENCE_QUANTILE, reference),
                (PolicyId.GLOBAL_QUANTILE, global_q),
                (PolicyId.LOCAL_QUANTILE, local_q[cid]),
                (PolicyId.READINESS_ONLY, readiness_only(client.benign)),
                (PolicyId.MISMATCH_ONLY, mismatch_only(client.benign, protocol.alpha)),
                (PolicyId.SHRINKAGE, shrinkage(client.benign, protocol.alpha, shrinkage_n0)),
                (PolicyId.THREE_SIGMA, three_sigma_threshold),
                (
                    PolicyId.DEV_F1_SELECT,
                    dev_local_global(client.supervised, global_q, local_q[cid]),
                ),
                (PolicyId.SUMMARY_STATISTIC_SELECT, summary_threshold),
                (PolicyId.SUPERVISED_F1, supervised_threshold),
                (PolicyId.FEDCRG, method_threshold),
            )
            entries.extend(
                ClientPolicyThreshold(policy, cid, threshold)
                for policy, threshold in thresholds
            )

        expected = {
            (policy, client.client_id)
            for policy in PolicyId
            if policy is not PolicyId.ORACLE_TEST
            for client in clients
        }
        observed = {(entry.policy, entry.client_id) for entry in entries}
        if observed != expected:
            raise RuntimeError("Policy selection did not produce one non-oracle cell per client")
        return PolicyThresholdSet(tuple(entries), tuple(undefined), shrinkage_n0)
