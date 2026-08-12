"""Typed policy registry and federation-level policy selection."""

from dataclasses import dataclass
from enum import StrEnum
from fedcrg.config.models import ProtocolConfig
from fedcrg.core.enums import PolicyId
from fedcrg.policies.attack_aware import dev_local_global, summary_statistic_threshold, supervised_global_f1
from fedcrg.policies.base import ClientPolicyData
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
@dataclass(frozen=True, slots=True)
class PolicyThresholdSet:
    values: dict[PolicyId, dict[str, float | None]]
    shrinkage_n0: int
    def for_client(self, policy: PolicyId, client_id: str) -> float | None:
        return self.values[policy][client_id]
class PolicyRegistry:
    def __init__(self) -> None:
        supervised = {PolicyId.DEV_F1_SELECT, PolicyId.SUMMARY_STATISTIC_SELECT, PolicyId.SUPERVISED_F1}
        self._definitions = {policy: PolicyDefinition(policy, InformationRegime.FINAL_TEST_ORACLE if policy is PolicyId.ORACLE_TEST else InformationRegime.SUPERVISED_DEVELOPMENT if policy in supervised else InformationRegime.BENIGN_ONLY) for policy in PolicyId}
    def get(self, policy_id: PolicyId) -> PolicyDefinition: return self._definitions[policy_id]
    def all_ids(self) -> tuple[PolicyId, ...]: return tuple(self._definitions)
class FederationPolicySelector:
    """Resolve every comparator once from only its permitted information regime."""
    def select(self, clients: tuple[ClientPolicyData, ...], protocol: ProtocolConfig) -> PolicyThresholdSet:
        if not clients: raise ValueError("At least one client is required for policy selection")
        ids = tuple(client.client_id for client in clients); global_q = global_quantile(clients, protocol.alpha); local_q = {client.client_id: local_quantile(client, protocol.alpha) for client in clients}; n0 = tune_shrinkage(clients, protocol.alpha); shared_three_sigma = three_sigma(clients); summary_threshold = summary_statistic_threshold(clients); supervised_threshold = supervised_global_f1(clients); values: dict[PolicyId, dict[str, float | None]] = {policy: {} for policy in PolicyId}
        for client in clients:
            cid = client.client_id; reference = client.protocol.reference.value; fedcrg = client.protocol.decision.threshold
            values[PolicyId.REFERENCE_QUANTILE][cid] = reference; values[PolicyId.GLOBAL_QUANTILE][cid] = global_q; values[PolicyId.LOCAL_QUANTILE][cid] = local_q[cid]; values[PolicyId.READINESS_ONLY][cid] = readiness_only(client); values[PolicyId.MISMATCH_ONLY][cid] = mismatch_only(client, protocol.alpha); values[PolicyId.SHRINKAGE][cid] = shrinkage(client, protocol.alpha, n0); values[PolicyId.THREE_SIGMA][cid] = shared_three_sigma; values[PolicyId.DEV_F1_SELECT][cid] = dev_local_global(client, global_q, local_q[cid]); values[PolicyId.SUMMARY_STATISTIC_SELECT][cid] = summary_threshold; values[PolicyId.SUPERVISED_F1][cid] = supervised_threshold; values[PolicyId.ORACLE_TEST][cid] = oracle_choice(client, (global_q, local_q[cid], fedcrg), protocol.band); values[PolicyId.FEDCRG][cid] = fedcrg
        if not all(set(mapping) == set(ids) for mapping in values.values()): raise RuntimeError("Policy selection did not produce one value per client")
        return PolicyThresholdSet(values=values, shrinkage_n0=n0)
