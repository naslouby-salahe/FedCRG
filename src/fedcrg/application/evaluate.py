"""Evaluate the protocol and all locked comparators from one immutable score cache."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from fedcrg.artifacts.records import MetricRecord, ThresholdRecord, write_jsonl
from fedcrg.artifacts.serialization import atomic_write_json
from fedcrg.config.models import ExperimentConfig
from fedcrg.core.enums import (
    CalibrationAssignmentMode,
    DataRole,
    PolicyEvaluationStatus,
    PolicyId,
)
from fedcrg.core.ids import ClientId, Sha256
from fedcrg.metrics.attack_balanced import attack_balanced_tpr
from fedcrg.metrics.classification import (
    balanced_accuracy,
    confusion_matrix,
    f1,
    fpr,
    precision,
    recall,
    tpr,
)
from fedcrg.metrics.federation import aggregate_policy, assert_ranking_metric_invariance
from fedcrg.metrics.operating_band import (
    absolute_fpr_error,
    band_error,
    band_violation,
    high_excess,
)
from fedcrg.metrics.ranking import auprc, auroc
from fedcrg.metrics.results import ClientMetrics, EvaluationBundle, PolicyEvaluation
from fedcrg.policies.base import (
    BenignPolicyEvidence,
    ClientPolicyInputs,
    FinalTestEvidence,
    SupervisedDevelopmentEvidence,
)
from fedcrg.policies.registry import FederationPolicySelector
from fedcrg.protocol.mismatch import clopper_pearson_interval
from fedcrg.protocol.results import ClientProtocolResult
from fedcrg.protocol.service import FedCRGProtocol
from fedcrg.scoring.models import ScoreManifest
from fedcrg.scoring.views import CalibrationScoreViewBuilder, CalibrationScoreViews


class EvaluatePolicies:
    """Keep calibration assignment, policy fitting, and final-label evaluation distinct."""

    def __init__(
        self,
        selector: FederationPolicySelector | None = None,
        views: CalibrationScoreViewBuilder | None = None,
    ) -> None:
        self.selector = selector or FederationPolicySelector()
        self.views = views or CalibrationScoreViewBuilder()

    @staticmethod
    def _protocol(config: ExperimentConfig) -> FedCRGProtocol:
        cache_path = config.outputs_root / "cache" / "precomputed" / "readiness_plans.json"
        return FedCRGProtocol.with_persistent_readiness_cache(cache_path)

    @staticmethod
    def _validate_score_identity(config: ExperimentConfig, scores: ScoreManifest) -> None:
        if scores.dataset is not config.dataset.id:
            raise ValueError("Score cache dataset does not match experiment config")
        if scores.data_spec_hash != Sha256(config.data_spec_hash):
            raise ValueError("SCORE_CACHE_HASH_MISMATCH: data specification differs")
        if scores.training_spec_hash != Sha256(config.training_spec_hash):
            raise ValueError("SCORE_CACHE_HASH_MISMATCH: training specification differs")
        if scores.cache_sha256 is None:
            raise ValueError("SCORE_CACHE_HASH_MISMATCH: policy evaluation requires a finalized cache")

    def calibration_views(
        self,
        config: ExperimentConfig,
        scores: ScoreManifest,
        calibration_seed: int | None = None,
        mode: CalibrationAssignmentMode = CalibrationAssignmentMode.SEEDED_PERMUTATION,
        prepared_root: Path | None = None,
    ) -> CalibrationScoreViews:
        self._validate_score_identity(config, scores)
        seed = config.dataset.primary_calibration_seed if calibration_seed is None else calibration_seed
        return self.views.build(
            scores,
            config.dataset,
            seed,
            mode,
            prepared_root,
        )

    def protocol_results(
        self,
        config: ExperimentConfig,
        scores: ScoreManifest,
        calibration_seed: int | None = None,
        mode: CalibrationAssignmentMode = CalibrationAssignmentMode.SEEDED_PERMUTATION,
        prepared_root: Path | None = None,
        calibration_views: CalibrationScoreViews | None = None,
    ) -> dict[ClientId, ClientProtocolResult]:
        views = calibration_views or self.calibration_views(
            config, scores, calibration_seed, mode, prepared_root
        )
        protocol = self._protocol(config)
        reference_by_client = {
            client_id: views.get(client_id, DataRole.REFERENCE).values
            for client_id in views.clients
        }
        reference = protocol.estimate_reference(reference_by_client, config.protocol)
        calibration_sizes = {
            len(views.get(client_id, DataRole.CALIBRATION).values)
            for client_id in views.clients
        }
        if len(calibration_sizes) != 1:
            raise ValueError("Calibration evidence count must be identical across clients")
        plan = protocol.precompute_readiness(calibration_sizes.pop(), config.protocol)
        results: dict[ClientId, ClientProtocolResult] = {}
        for client_id in sorted(views.clients):
            results[client_id] = protocol.evaluate_client(
                client_id=client_id,
                reference=reference,
                calibration_scores=views.get(client_id, DataRole.CALIBRATION).values,
                mismatch_scores=views.get(client_id, DataRole.MISMATCH).values,
                config=config.protocol,
                readiness_plan=plan,
            )
        return results

    def _policy_inputs(
        self,
        scores: ScoreManifest,
        views: CalibrationScoreViews,
        protocol_results: dict[ClientId, ClientProtocolResult],
    ) -> tuple[ClientPolicyInputs, ...]:
        clients: list[ClientPolicyInputs] = []
        for client_id, score_set in sorted(scores.clients.items()):
            benign = BenignPolicyEvidence(
                client_id=client_id,
                reference_scores=views.get(client_id, DataRole.REFERENCE).values,
                mismatch_scores=views.get(client_id, DataRole.MISMATCH).values,
                calibration_scores=views.get(client_id, DataRole.CALIBRATION).values,
                protocol=protocol_results[client_id],
            )
            attack_test = score_set.scores[DataRole.ATTACK_TEST]
            attack_groups = attack_test.attack_groups or tuple(
                "attack" for _ in attack_test.values
            )
            supervised = SupervisedDevelopmentEvidence(
                benign=benign,
                benign_guard_scores=views.get(client_id, DataRole.BENIGN_GUARD).values,
                attack_dev_scores=score_set.scores[DataRole.ATTACK_DEV].values,
            )
            final_test = FinalTestEvidence(
                benign=benign,
                benign_test_scores=score_set.scores[DataRole.BENIGN_TEST].values,
                attack_test_scores=attack_test.values,
                attack_test_groups=attack_groups,
            )
            clients.append(ClientPolicyInputs(benign, supervised, final_test))
        return tuple(clients)

    def evaluate(
        self,
        config: ExperimentConfig,
        scores: ScoreManifest,
        calibration_seed: int | None = None,
        mode: CalibrationAssignmentMode = CalibrationAssignmentMode.SEEDED_PERMUTATION,
        prepared_root: Path | None = None,
        calibration_views: CalibrationScoreViews | None = None,
    ) -> EvaluationBundle:
        views = calibration_views or self.calibration_views(
            config, scores, calibration_seed, mode, prepared_root
        )
        protocol_results = self.protocol_results(
            config,
            scores,
            calibration_seed,
            mode,
            prepared_root,
            views,
        )
        clients = self._policy_inputs(scores, views, protocol_results)
        thresholds = self.selector.select(clients, config.protocol)

        evaluations: list[PolicyEvaluation] = []
        for client in clients:
            final = client.final_test
            benign_test = final.benign_test_scores
            attack_test = final.attack_test_scores
            test_scores = np.concatenate((benign_test, attack_test))
            test_labels = np.concatenate(
                (
                    np.zeros(len(benign_test), dtype=np.int64),
                    np.ones(len(attack_test), dtype=np.int64),
                )
            )
            test_groups = np.asarray(
                ("__benign__",) * len(benign_test) + final.attack_test_groups,
                dtype=object,
            )
            ranking_auroc = auroc(test_scores, test_labels)
            ranking_auprc = auprc(test_scores, test_labels)

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
                if client_fpr is None:
                    raise RuntimeError("Final benign test set is empty")
                client_metrics = ClientMetrics(
                    benign_n=cm.fp + cm.tn,
                    attack_n=cm.tp + cm.fn,
                    fp=cm.fp,
                    tn=cm.tn,
                    tp=cm.tp,
                    fn=cm.fn,
                    fpr=client_fpr,
                    tpr=tpr(cm),
                    precision=precision(cm),
                    recall=recall(cm),
                    f1=f1(cm),
                    balanced_accuracy=balanced_accuracy(cm),
                    auroc=ranking_auroc,
                    auprc=ranking_auprc,
                    band_error=band_error(client_fpr, config.protocol.band),
                    high_excess=high_excess(client_fpr, config.protocol.band),
                    band_violation=band_violation(client_fpr, config.protocol.band),
                    absolute_fpr_error=absolute_fpr_error(client_fpr, config.protocol.alpha),
                    attack_balanced_tpr=attack_balanced_tpr(
                        test_scores, test_labels, test_groups, threshold
                    ),
                    fpr_reference_interval=clopper_pearson_interval(
                        cm.fp, cm.fp + cm.tn, 0.95
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

        client_rows = tuple(evaluations)
        assert_ranking_metric_invariance(client_rows)
        federation_rows = tuple(
            aggregate_policy(policy, client_rows)
            for policy in config.policies
            if any(
                row.policy is policy and row.status is PolicyEvaluationStatus.EVALUATED
                for row in client_rows
            )
        )
        return EvaluationBundle(
            client_rows,
            federation_rows,
            tuple(protocol_results.values()),
            thresholds.shrinkage_n0,
        )

    def write_policy_artifacts(
        self,
        root: Path,
        run_id: str,
        policy: PolicyId,
        bundle: EvaluationBundle,
    ) -> tuple[Path, Path]:
        protocol_by_client = {item.client_id: item for item in bundle.protocol_results}
        threshold_records: list[ThresholdRecord] = []
        metric_records: list[MetricRecord] = []
        for row in bundle.clients:
            if row.policy is not policy:
                continue
            protocol = protocol_by_client[row.client_id]
            readiness = protocol.readiness
            mismatch = protocol.mismatch
            interval = mismatch.interval
            threshold_records.append(
                ThresholdRecord(
                    run_id=run_id,
                    policy_id=policy,
                    client_id=row.client_id.value,
                    tau_ref=protocol.reference.value,
                    tau_local=readiness.threshold,
                    selected_tau=row.threshold,
                    readiness_n=readiness.plan.sample_count,
                    readiness_rank=readiness.plan.rank,
                    readiness_probability=readiness.plan.coverage_probability,
                    mismatch_n=mismatch.sample_count,
                    mismatch_x=mismatch.exceedance_count,
                    cp_lower=None if interval is None else interval.lower,
                    cp_upper=None if interval is None else interval.upper,
                    p_low=mismatch.p_low,
                    p_high=mismatch.p_high,
                    state=protocol.decision.state.value,
                    tie_count=readiness.tie_count,
                    selected_source=protocol.decision.source.value,
                    reason_code=protocol.decision.reason.value,
                )
            )
            if row.metrics is not None:
                metric = row.metrics
                metric_records.append(
                    MetricRecord(
                        run_id=run_id,
                        policy_id=policy,
                        client_id=row.client_id.value,
                        benign_n=metric.benign_n,
                        attack_n=metric.attack_n,
                        fp=metric.fp,
                        tn=metric.tn,
                        tp=metric.tp,
                        fn=metric.fn,
                        fpr=metric.fpr,
                        tpr=metric.tpr,
                        precision=metric.precision,
                        f1=metric.f1,
                        balanced_accuracy=metric.balanced_accuracy,
                        auroc=metric.auroc,
                        auprc=metric.auprc,
                        band_error=metric.band_error,
                        attack_balanced_tpr=metric.attack_balanced_tpr,
                    )
                )
        decisions = root / "decisions" / "threshold_record.jsonl"
        metrics = root / "metrics" / "metric_record.jsonl"
        write_jsonl(decisions, tuple(threshold_records))
        write_jsonl(metrics, tuple(metric_records))
        federation = next(
            (item for item in bundle.federations if item.policy is policy), None
        )
        if federation is not None:
            payload = asdict(federation)
            payload["policy"] = federation.policy.value
            atomic_write_json(root / "metrics" / "federation.json", payload)
        return decisions, metrics

    @staticmethod
    def to_serializable(bundle: EvaluationBundle) -> dict[str, object]:
        clients: list[dict[str, object]] = []
        for item in bundle.clients:
            record = asdict(item)
            record["client_id"] = item.client_id.value
            record["policy"] = item.policy.value
            record["status"] = item.status.value
            clients.append(record)
        federations: list[dict[str, object]] = []
        for item in bundle.federations:
            record = asdict(item)
            record["policy"] = item.policy.value
            federations.append(record)
        return {
            "shrinkage_n0": bundle.shrinkage_n0,
            "clients": clients,
            "federations": federations,
        }
