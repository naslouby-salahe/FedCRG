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
from fedcrg.policies.base import ClientPolicyInputs
from fedcrg.policies.oracle import oracle_choice
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
class PolicyThresholdSet:
    values: dict[PolicyId, dict[ClientId, float | None]]
    undefined_reasons: dict[PolicyId, FailureCode]
    shrinkage_n0: int

    def for_client(self, policy: PolicyId, client_id: ClientId) -> float | None:
        return self.values[policy][client_id]


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
    def select(
        self,
        clients: tuple[ClientPolicyInputs, ...],
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

        values: dict[PolicyId, dict[ClientId, float | None]] = {
            policy: {} for policy in PolicyId
        }
        undefined: dict[PolicyId, FailureCode] = {}
        if summary_threshold is None:
            undefined[PolicyId.SUMMARY_STATISTIC_SELECT] = (
                FailureCode.SUMMARY_STATISTIC_COMPARATOR_UNDEFINED
            )

        for client in clients:
            cid = client.client_id
            reference = client.benign.protocol.reference.value
            fedcrg = client.benign.protocol.decision.threshold
            values[PolicyId.REFERENCE_QUANTILE][cid] = reference
            values[PolicyId.GLOBAL_QUANTILE][cid] = global_q
            values[PolicyId.LOCAL_QUANTILE][cid] = local_q[cid]
            values[PolicyId.READINESS_ONLY][cid] = readiness_only(client.benign)
            values[PolicyId.MISMATCH_ONLY][cid] = mismatch_only(
                client.benign, protocol.alpha
            )
            values[PolicyId.SHRINKAGE][cid] = shrinkage(
                client.benign, protocol.alpha, shrinkage_n0
            )
            values[PolicyId.THREE_SIGMA][cid] = three_sigma_threshold
            values[PolicyId.DEV_F1_SELECT][cid] = dev_local_global(
                client.supervised, global_q, local_q[cid]
            )
            values[PolicyId.SUMMARY_STATISTIC_SELECT][cid] = summary_threshold
            values[PolicyId.SUPERVISED_F1][cid] = supervised_threshold
            values[PolicyId.ORACLE_TEST][cid] = oracle_choice(
                client.final_test,
                (global_q, local_q[cid], fedcrg),
                protocol.band,
            )
            values[PolicyId.FEDCRG][cid] = fedcrg

        ids = {client.client_id for client in clients}
        if not all(set(mapping) == ids for mapping in values.values()):
            raise RuntimeError("Policy selection did not produce one cell per client")
        return PolicyThresholdSet(values, undefined, shrinkage_n0)
