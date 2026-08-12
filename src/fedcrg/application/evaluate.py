"""Protocol and comparator evaluation on one immutable score cache."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from fedcrg.config.models import ExperimentConfig
from fedcrg.core.enums import DataRole, PolicyEvaluationStatus
from fedcrg.metrics.attack_balanced import attack_balanced_tpr
from fedcrg.metrics.classification import confusion_matrix, f1, fpr, precision, recall, tpr
from fedcrg.metrics.operating_band import absolute_fpr_error, band_error, band_violation, high_excess
from fedcrg.metrics.ranking import auprc, auroc
from fedcrg.metrics.results import ClientMetrics, PolicyEvaluation
from fedcrg.policies.base import ClientPolicyData
from fedcrg.policies.registry import FederationPolicySelector
from fedcrg.protocol.service import FedCRGProtocol
from fedcrg.scoring.models import ScoreManifest


class EvaluatePolicies:
    def __init__(
        self,
        selector: FederationPolicySelector | None = None,
        protocol: FedCRGProtocol | None = None,
    ) -> None:
        self.selector = selector or FederationPolicySelector()
        self.protocol = protocol or FedCRGProtocol()

    def evaluate(
        self,
        config: ExperimentConfig,
        scores: ScoreManifest,
    ) -> tuple[PolicyEvaluation, ...]:
        reference_by_client = {
            client_id: client.scores[DataRole.REFERENCE].values
            for client_id, client in scores.clients.items()
        }
        reference = self.protocol.estimate_reference(reference_by_client, config.protocol)
        clients: list[ClientPolicyData] = []
        for client_id, score_set in sorted(scores.clients.items()):
            protocol_result = self.protocol.evaluate_client(
                client_id,
                reference,
                score_set.scores[DataRole.CALIBRATION].values,
                score_set.scores[DataRole.MISMATCH].values,
                config.protocol,
            )
            guard = score_set.scores[DataRole.BENIGN_GUARD].values
            attack_dev = score_set.scores[DataRole.ATTACK_DEV].values
            attack_test = score_set.scores[DataRole.ATTACK_TEST]
            groups = attack_test.attack_groups or tuple("attack" for _ in attack_test.values)
            clients.append(
                ClientPolicyData(
                    client_id=client_id,
                    reference_scores=score_set.scores[DataRole.REFERENCE].values,
                    mismatch_scores=score_set.scores[DataRole.MISMATCH].values,
                    calibration_scores=score_set.scores[DataRole.CALIBRATION].values,
                    benign_guard_scores=guard,
                    attack_dev_scores=attack_dev,
                    benign_test_scores=score_set.scores[DataRole.BENIGN_TEST].values,
                    attack_test_scores=attack_test.values,
                    attack_test_groups=groups,
                    protocol=protocol_result,
                )
            )
        client_tuple = tuple(clients)
        thresholds = self.selector.select(client_tuple, config.protocol)

        evaluations: list[PolicyEvaluation] = []
        for client in client_tuple:
            benign_test = client.benign_test_scores
            attack_test = client.attack_test_scores
            test_scores = np.concatenate((benign_test, attack_test))
            test_labels = np.concatenate(
                (np.zeros(len(benign_test), dtype=np.int64), np.ones(len(attack_test), dtype=np.int64))
            )
            test_groups = np.asarray(
                ("__benign__",) * len(benign_test) + client.attack_test_groups,
                dtype=object,
            )
            for policy_id in config.policies:
                threshold = thresholds.for_client(policy_id, client.client_id)
                if threshold is None:
                    evaluations.append(
                        PolicyEvaluation(
                            client.client_id,
                            policy_id,
                            None,
                            PolicyEvaluationStatus.UNDEFINED,
                            None,
                        )
                    )
                    continue
                cm = confusion_matrix(test_scores, test_labels, threshold)
                client_fpr = fpr(cm)
                client_metrics = ClientMetrics(
                    fpr=client_fpr,
                    tpr=tpr(cm),
                    precision=precision(cm),
                    recall=recall(cm),
                    f1=f1(cm),
                    auroc=auroc(test_scores, test_labels),
                    auprc=auprc(test_scores, test_labels),
                    band_error=band_error(client_fpr, config.protocol.band),
                    high_excess=high_excess(client_fpr, config.protocol.band),
                    band_violation=band_violation(client_fpr, config.protocol.band),
                    absolute_fpr_error=absolute_fpr_error(client_fpr, config.protocol.alpha),
                    attack_balanced_tpr=attack_balanced_tpr(
                        test_scores, test_labels, test_groups, threshold
                    ),
                )
                evaluations.append(
                    PolicyEvaluation(
                        client.client_id,
                        policy_id,
                        threshold,
                        PolicyEvaluationStatus.EVALUATED,
                        client_metrics,
                    )
                )
        return tuple(evaluations)

    @staticmethod
    def to_serializable(evaluations: tuple[PolicyEvaluation, ...]) -> list[dict[str, object]]:
        payload: list[dict[str, object]] = []
        for item in evaluations:
            record = asdict(item)
            record["policy"] = item.policy.value
            record["status"] = item.status.value
            payload.append(record)
        return payload
