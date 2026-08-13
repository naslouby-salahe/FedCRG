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
from fedcrg.policies.base import BenignPolicyEvidence, SupervisedDevelopmentEvidence
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
    """Thresholds frozen before final-test evidence is opened."""

    entries: tuple[ClientPolicyThreshold, ...]
    undefined_reasons: tuple[UndefinedPolicyReason, ...]
    shrinkage_n0: int | None

    def for_client(self, policy: PolicyId, client_id: ClientId) -> float | None:
        if policy is PolicyId.ORACLE_TEST:
            raise ValueError("ORACLE-TEST is not available before final-test evidence opens")
        for entry in self.entries:
            if entry.policy is policy and entry.client_id == client_id:
                return entry.threshold
        raise KeyError(f"No threshold for {policy.value}/{client_id.value}")


class PolicyRegistry:
    _SUPERVISED = frozenset(
        {
            PolicyId.DEV_F1_SELECT,
            PolicyId.SUMMARY_STATISTIC_SELECT,
            PolicyId.SUPERVISED_F1,
        }
    )

    def __init__(self) -> None:
        self._definitions = {
            policy: PolicyDefinition(
                id=policy,
                information_regime=(
                    InformationRegime.FINAL_TEST_ORACLE
                    if policy is PolicyId.ORACLE_TEST
                    else InformationRegime.SUPERVISED_DEVELOPMENT
                    if policy in self._SUPERVISED
                    else InformationRegime.BENIGN_ONLY
                ),
                deployable=(
                    policy not in self._SUPERVISED
                    and policy is not PolicyId.ORACLE_TEST
                ),
            )
            for policy in PolicyId
        }

    def get(self, policy_id: PolicyId) -> PolicyDefinition:
        return self._definitions[policy_id]

    def all_ids(self) -> tuple[PolicyId, ...]:
        return tuple(self._definitions)

    def supervised_requested(self, policies: tuple[PolicyId, ...]) -> bool:
        return bool(set(policies) & self._SUPERVISED)

    def assert_exact_protocol_registry(self) -> None:
        if set(self._definitions) != set(PolicyId) or len(self._definitions) != 12:
            raise RuntimeError("Policy registry must contain exactly 12 protocol policies")


class FederationPolicySelector:
    """Select only requested non-oracle policies from their permitted evidence."""

    def __init__(self, registry: PolicyRegistry | None = None) -> None:
        self.registry = registry or PolicyRegistry()

    def select(
        self,
        benign_clients: tuple[BenignPolicyEvidence, ...],
        protocol: ProtocolConfig,
        requested_policies: tuple[PolicyId, ...],
        supervised_clients: tuple[SupervisedDevelopmentEvidence, ...] | None = None,
    ) -> PolicyThresholdSet:
        if not benign_clients:
            raise ValueError("At least one client is required for policy selection")
        non_oracle = tuple(
            policy for policy in requested_policies if policy is not PolicyId.ORACLE_TEST
        )
        if not non_oracle and PolicyId.ORACLE_TEST not in requested_policies:
            raise ValueError("No policy was requested")

        client_ids = tuple(client.client_id for client in benign_clients)
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("Policy selection received duplicate client identities")

        supervised_needed = self.registry.supervised_requested(non_oracle)
        supervised_by_client: dict[ClientId, SupervisedDevelopmentEvidence] = {}
        if supervised_needed:
            if supervised_clients is None:
                raise ValueError("Requested supervised comparator lacks development evidence")
            supervised_by_client = {
                client.client_id: client for client in supervised_clients
            }
            if set(supervised_by_client) != set(client_ids):
                raise ValueError("Supervised development evidence does not cover the federation")
        elif supervised_clients is not None:
            raise ValueError(
                "Supervised development evidence was supplied although no supervised policy was requested"
            )

        need_global = bool(
            set(non_oracle)
            & {
                PolicyId.GLOBAL_QUANTILE,
                PolicyId.DEV_F1_SELECT,
            }
        ) or PolicyId.ORACLE_TEST in requested_policies
        need_local = bool(
            set(non_oracle)
            & {
                PolicyId.LOCAL_QUANTILE,
                PolicyId.DEV_F1_SELECT,
            }
        ) or PolicyId.ORACLE_TEST in requested_policies
        need_method = PolicyId.FEDCRG in non_oracle or PolicyId.ORACLE_TEST in requested_policies

        global_q = global_quantile(benign_clients, protocol.alpha) if need_global else None
        local_q = (
            {
                client.client_id: local_quantile(client, protocol.alpha)
                for client in benign_clients
            }
            if need_local
            else {}
        )
        shrinkage_n0 = (
            tune_shrinkage(benign_clients, protocol.alpha)
            if PolicyId.SHRINKAGE in non_oracle
            else None
        )
        three_sigma_threshold = (
            three_sigma(benign_clients)
            if PolicyId.THREE_SIGMA in non_oracle
            else None
        )
        summary_threshold = (
            summary_statistic_threshold(tuple(supervised_by_client.values()))
            if PolicyId.SUMMARY_STATISTIC_SELECT in non_oracle
            else None
        )
        supervised_threshold = (
            supervised_global_f1(tuple(supervised_by_client.values()))
            if PolicyId.SUPERVISED_F1 in non_oracle
            else None
        )

        entries: list[ClientPolicyThreshold] = []
        undefined: list[UndefinedPolicyReason] = []
        if (
            PolicyId.SUMMARY_STATISTIC_SELECT in non_oracle
            and summary_threshold is None
        ):
            undefined.append(
                UndefinedPolicyReason(
                    PolicyId.SUMMARY_STATISTIC_SELECT,
                    FailureCode.SUMMARY_STATISTIC_COMPARATOR_UNDEFINED,
                )
            )

        for benign in benign_clients:
            client_id = benign.client_id
            for policy in non_oracle:
                threshold: float | None
                if policy is PolicyId.REFERENCE_QUANTILE:
                    threshold = benign.protocol.reference.value
                elif policy is PolicyId.GLOBAL_QUANTILE:
                    threshold = global_q
                elif policy is PolicyId.LOCAL_QUANTILE:
                    threshold = local_q[client_id]
                elif policy is PolicyId.READINESS_ONLY:
                    threshold = readiness_only(benign)
                elif policy is PolicyId.MISMATCH_ONLY:
                    threshold = mismatch_only(benign, protocol.alpha)
                elif policy is PolicyId.SHRINKAGE:
                    if shrinkage_n0 is None:
                        raise RuntimeError("Shrinkage tuning was not materialized")
                    threshold = shrinkage(benign, protocol.alpha, shrinkage_n0)
                elif policy is PolicyId.THREE_SIGMA:
                    threshold = three_sigma_threshold
                elif policy is PolicyId.DEV_F1_SELECT:
                    if global_q is None:
                        raise RuntimeError("Global quantile was not materialized")
                    threshold = dev_local_global(
                        supervised_by_client[client_id],
                        global_q,
                        local_q[client_id],
                    )
                elif policy is PolicyId.SUMMARY_STATISTIC_SELECT:
                    threshold = summary_threshold
                elif policy is PolicyId.SUPERVISED_F1:
                    threshold = supervised_threshold
                elif policy is PolicyId.FEDCRG:
                    threshold = benign.protocol.decision.threshold
                else:
                    raise RuntimeError(f"Unhandled non-oracle policy: {policy.value}")
                entries.append(
                    ClientPolicyThreshold(policy, client_id, threshold)
                )

            # Oracle candidates are computed from benign evidence but never exposed as
            # evaluated policies before final-test labels open.
            if PolicyId.ORACLE_TEST in requested_policies:
                if global_q is None or client_id not in local_q or not need_method:
                    raise RuntimeError("Oracle candidate thresholds were not prepared")
                for policy, threshold in (
                    (PolicyId.GLOBAL_QUANTILE, global_q),
                    (PolicyId.LOCAL_QUANTILE, local_q[client_id]),
                    (PolicyId.FEDCRG, benign.protocol.decision.threshold),
                ):
                    if not any(
                        item.policy is policy and item.client_id == client_id
                        for item in entries
                    ):
                        entries.append(
                            ClientPolicyThreshold(policy, client_id, threshold)
                        )

        expected = {
            (policy, client_id)
            for policy in non_oracle
            for client_id in client_ids
        }
        observed = {
            (entry.policy, entry.client_id)
            for entry in entries
            if entry.policy in non_oracle
        }
        if observed != expected:
            raise RuntimeError("Policy selection did not produce each requested client-policy cell")
        return PolicyThresholdSet(tuple(entries), tuple(undefined), shrinkage_n0)
