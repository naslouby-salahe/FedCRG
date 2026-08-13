"""Explicit, typed selection of threshold-comparator results per requested policy."""

from __future__ import annotations

from fedcrg.configuration.method_config import ProtocolConfig
from fedcrg.configuration.statistics_config import StatisticsConfig
from fedcrg.domain.enums import FailureCode, PolicyId
from fedcrg.domain.identifiers import ClientId
from fedcrg.decision.policies.development_f1 import dev_local_global
from fedcrg.decision.policies.global_quantile import global_quantile
from fedcrg.decision.policies.local_quantile import local_quantile
from fedcrg.decision.policies.mismatch_only import mismatch_only
from fedcrg.decision.policies.readiness_only import readiness_only
from fedcrg.decision.policies.reference_quantile import reference_quantile
from fedcrg.decision.policies.shrinkage import shrinkage, tune_shrinkage
from fedcrg.decision.policies.summary_statistic import summary_statistic_threshold
from fedcrg.decision.policies.supervised_f1 import supervised_global_f1
from fedcrg.decision.policies.three_sigma import three_sigma
from fedcrg.decision.evidence import BenignPolicyEvidence, SupervisedDevelopmentEvidence
from fedcrg.decision.policy_results import (
    ClientPolicyThreshold,
    InformationRegime,
    PolicyThresholdSet,
    UndefinedPolicyReason,
)

SUPERVISED_POLICIES = frozenset(
    {
        PolicyId.DEV_F1_SELECT,
        PolicyId.SUMMARY_STATISTIC_SELECT,
        PolicyId.SUPERVISED_F1,
    }
)


def information_regime(policy_id: PolicyId) -> InformationRegime:
    if policy_id is PolicyId.ORACLE_TEST:
        return InformationRegime.FINAL_TEST_ORACLE
    if policy_id in SUPERVISED_POLICIES:
        return InformationRegime.SUPERVISED_DEVELOPMENT
    return InformationRegime.BENIGN_ONLY


def is_deployable(policy_id: PolicyId) -> bool:
    return policy_id not in SUPERVISED_POLICIES and policy_id is not PolicyId.ORACLE_TEST


class PolicyThresholdSelector:
    """Select only requested non-oracle policies from their permitted evidence."""

    def select(
        self,
        benign_clients: tuple[BenignPolicyEvidence, ...],
        protocol: ProtocolConfig,
        statistics: StatisticsConfig,
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

        supervised_needed = bool(set(non_oracle) & SUPERVISED_POLICIES)
        supervised_by_client: dict[ClientId, SupervisedDevelopmentEvidence] = {}
        if supervised_needed:
            if supervised_clients is None:
                raise ValueError("Requested supervised comparator lacks development evidence")
            supervised_by_client = {client.client_id: client for client in supervised_clients}
            if set(supervised_by_client) != set(client_ids):
                raise ValueError("Supervised development evidence does not cover the federation")
        elif supervised_clients is not None:
            raise ValueError(
                "Supervised development evidence was supplied although no supervised policy was requested"
            )

        need_global = (
            bool(
                set(non_oracle)
                & {
                    PolicyId.GLOBAL_QUANTILE,
                    PolicyId.DEV_F1_SELECT,
                }
            )
            or PolicyId.ORACLE_TEST in requested_policies
        )
        need_local = (
            bool(
                set(non_oracle)
                & {
                    PolicyId.LOCAL_QUANTILE,
                    PolicyId.DEV_F1_SELECT,
                }
            )
            or PolicyId.ORACLE_TEST in requested_policies
        )
        need_method = PolicyId.FEDCRG in non_oracle or PolicyId.ORACLE_TEST in requested_policies

        global_q = global_quantile(benign_clients, protocol.alpha) if need_global else None
        local_q = (
            {client.client_id: local_quantile(client, protocol.alpha) for client in benign_clients}
            if need_local
            else {}
        )
        shrinkage_n0 = (
            tune_shrinkage(
                benign_clients,
                protocol.alpha,
                statistics.shrinkage_n0_candidates,
            )
            if PolicyId.SHRINKAGE in non_oracle
            else None
        )
        three_sigma_threshold = (
            three_sigma(benign_clients) if PolicyId.THREE_SIGMA in non_oracle else None
        )
        summary_threshold = (
            summary_statistic_threshold(
                tuple(supervised_by_client.values()),
                statistics.supervised_threshold_candidates,
            )
            if PolicyId.SUMMARY_STATISTIC_SELECT in non_oracle
            else None
        )
        supervised_threshold = (
            supervised_global_f1(
                tuple(supervised_by_client.values()),
                statistics.supervised_threshold_candidates,
            )
            if PolicyId.SUPERVISED_F1 in non_oracle
            else None
        )

        entries: list[ClientPolicyThreshold] = []
        undefined: list[UndefinedPolicyReason] = []
        if PolicyId.SUMMARY_STATISTIC_SELECT in non_oracle and summary_threshold is None:
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
                    threshold = reference_quantile(benign)
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
                    threshold = benign.evaluation.decision.threshold
                else:
                    raise RuntimeError(f"Unhandled non-oracle policy: {policy.value}")
                entries.append(ClientPolicyThreshold(policy, client_id, threshold))

            # Oracle candidates are computed from benign evidence but never exposed as
            # evaluated policies before final-test labels open.
            if PolicyId.ORACLE_TEST in requested_policies:
                if global_q is None or client_id not in local_q or not need_method:
                    raise RuntimeError("Oracle candidate thresholds were not prepared")
                for policy, threshold in (
                    (PolicyId.GLOBAL_QUANTILE, global_q),
                    (PolicyId.LOCAL_QUANTILE, local_q[client_id]),
                    (PolicyId.FEDCRG, benign.evaluation.decision.threshold),
                ):
                    if not any(
                        item.policy is policy and item.client_id == client_id for item in entries
                    ):
                        entries.append(ClientPolicyThreshold(policy, client_id, threshold))

        expected = {(policy, client_id) for policy in non_oracle for client_id in client_ids}
        observed = {
            (entry.policy, entry.client_id) for entry in entries if entry.policy in non_oracle
        }
        if observed != expected:
            raise RuntimeError("Policy selection did not produce each requested client-policy cell")
        return PolicyThresholdSet(tuple(entries), tuple(undefined), shrinkage_n0)
