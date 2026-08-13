"""Evaluate the protocol and locked comparators from one immutable score cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fedcrg.artifacts.records import MetricRecord, ThresholdRecord, write_jsonl
from fedcrg.artifacts.serialization import JsonValue, atomic_write_json, to_json_value
from fedcrg.config.models import ExperimentConfig
from fedcrg.core.enums import (
    CalibrationAssignmentMode,
    DataRole,
    PolicyEvaluationStatus,
    PolicyId,
)
from fedcrg.core.ids import AttackGroupId, CalibrationSeed, ClientId, RunId, Sha256
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
    FinalTestEvidence,
    PolicySelectionInputs,
    SupervisedDevelopmentEvidence,
)
from fedcrg.policies.oracle import oracle_choice
from fedcrg.policies.registry import FederationPolicySelector, PolicyThresholdSet
from fedcrg.protocol.mismatch import clopper_pearson_interval
from fedcrg.protocol.results import ClientProtocolResult
from fedcrg.protocol.service import FedCRGProtocol
from fedcrg.scoring.cache import ScoreCache, ScoreCacheDescriptor
from fedcrg.scoring.views import CalibrationScoreViewBuilder, CalibrationScoreViews


class EvaluatePolicies:
    """Freeze thresholds before opening final-test evidence.

    Reservoir and attack-development partitions are opened for threshold selection;
    final benign and attack partitions are opened one client at a time only after
    every non-oracle threshold has been frozen.
    """

    def __init__(
        self,
        selector: FederationPolicySelector | None = None,
        views: CalibrationScoreViewBuilder | None = None,
        score_cache: ScoreCache | None = None,
    ) -> None:
        self.selector = selector or FederationPolicySelector()
        self.views = views or CalibrationScoreViewBuilder()
        self.score_cache = score_cache or ScoreCache()

    @staticmethod
    def _protocol(config: ExperimentConfig) -> FedCRGProtocol:
        cache_path = (
            config.outputs_root
            / "cache"
            / "precomputed"
            / "readiness_plans.json"
        )
        return FedCRGProtocol.with_persistent_readiness_cache(cache_path)

    @staticmethod
    def _validate_score_identity(
        config: ExperimentConfig,
        descriptor: ScoreCacheDescriptor,
    ) -> None:
        identity = descriptor.identity
        if identity.dataset is not config.dataset.id:
            raise ValueError("Score cache dataset does not match experiment config")
        if identity.data_spec_hash != Sha256(config.data_spec_hash):
            raise ValueError("SCORE_CACHE_HASH_MISMATCH: data specification differs")
        if identity.training_spec_hash != Sha256(config.training_spec_hash):
            raise ValueError("SCORE_CACHE_HASH_MISMATCH: training specification differs")

    def calibration_views(
        self,
        config: ExperimentConfig,
        score_root: Path,
        calibration_seed: CalibrationSeed | int | None = None,
        mode: CalibrationAssignmentMode = CalibrationAssignmentMode.SEEDED_PERMUTATION,
        prepared_root: Path | None = None,
    ) -> CalibrationScoreViews:
        descriptor = self.score_cache.load_descriptor(score_root)
        self._validate_score_identity(config, descriptor)
        seed = CalibrationSeed(
            config.dataset.primary_calibration_seed
            if calibration_seed is None
            else int(calibration_seed)
        )
        return self.views.build_from_cache(
            self.score_cache,
            score_root,
            config.dataset,
            seed,
            mode,
            prepared_root,
        )

    def protocol_results(
        self,
        config: ExperimentConfig,
        calibration_views: CalibrationScoreViews,
    ) -> dict[ClientId, ClientProtocolResult]:
        protocol = self._protocol(config)
        reference_by_client = {
            client_id: calibration_views.get(client_id, DataRole.REFERENCE).values
            for client_id in calibration_views.client_ids
        }
        reference = protocol.estimate_reference(reference_by_client, config.protocol)
        calibration_sizes = {
            len(calibration_views.get(client_id, DataRole.CALIBRATION).values)
            for client_id in calibration_views.client_ids
        }
        if len(calibration_sizes) != 1:
            raise ValueError("Calibration evidence count must be identical across clients")
        plan = protocol.require_readiness(calibration_sizes.pop(), config.protocol)
        results: dict[ClientId, ClientProtocolResult] = {}
        for client_id in calibration_views.client_ids:
            results[client_id] = protocol.evaluate_client(
                client_id=client_id,
                reference=reference,
                calibration_scores=calibration_views.get(
                    client_id,
                    DataRole.CALIBRATION,
                ).values,
                mismatch_scores=calibration_views.get(
                    client_id,
                    DataRole.MISMATCH,
                ).values,
                config=config.protocol,
                readiness_plan=plan,
            )
        return results

    def _selection_inputs(
        self,
        score_root: Path,
        views: CalibrationScoreViews,
        protocol_results: dict[ClientId, ClientProtocolResult],
    ) -> tuple[PolicySelectionInputs, ...]:
        clients: list[PolicySelectionInputs] = []
        for client_id in views.client_ids:
            benign = BenignPolicyEvidence(
                client_id=client_id,
                reference_scores=views.get(client_id, DataRole.REFERENCE).values,
                mismatch_scores=views.get(client_id, DataRole.MISMATCH).values,
                calibration_scores=views.get(client_id, DataRole.CALIBRATION).values,
                protocol=protocol_results[client_id],
            )
            attack_dev = self.score_cache.read_role(
                score_root,
                client_id,
                DataRole.ATTACK_DEV,
            )
            supervised = SupervisedDevelopmentEvidence(
                benign=benign,
                benign_guard_scores=views.get(
                    client_id,
                    DataRole.BENIGN_GUARD,
                ).values,
                attack_dev_scores=attack_dev.values,
            )
            clients.append(PolicySelectionInputs(benign, supervised))
        return tuple(clients)

    @staticmethod
    def _oracle_threshold(
        final: FinalTestEvidence,
        thresholds: PolicyThresholdSet,
        config: ExperimentConfig,
    ) -> float:
        client_id = final.benign.client_id
        candidates = (
            thresholds.for_client(PolicyId.GLOBAL_QUANTILE, client_id),
            thresholds.for_client(PolicyId.LOCAL_QUANTILE, client_id),
            thresholds.for_client(PolicyId.FEDCRG, client_id),
        )
        if any(value is None for value in candidates):
            raise RuntimeError("Oracle candidates must all be defined")
        return oracle_choice(
            final,
            (
                float(candidates[0]),
                float(candidates[1]),
                float(candidates[2]),
            ),
            config.protocol.band,
        )

    def evaluate_from_cache(
        self,
        config: ExperimentConfig,
        score_root: Path,
        calibration_seed: CalibrationSeed | int | None = None,
        mode: CalibrationAssignmentMode = CalibrationAssignmentMode.SEEDED_PERMUTATION,
        prepared_root: Path | None = None,
        calibration_views: CalibrationScoreViews | None = None,
    ) -> EvaluationBundle:
        descriptor = self.score_cache.load_descriptor(score_root)
        self._validate_score_identity(config, descriptor)
        views = calibration_views or self.calibration_views(
            config,
            score_root,
            calibration_seed,
            mode,
            prepared_root,
        )
        protocol_results = self.protocol_results(config, views)
        selection_inputs = self._selection_inputs(
            score_root,
            views,
            protocol_results,
        )
        thresholds = self.selector.select(selection_inputs, config.protocol)
        selection_by_client = {
            item.client_id: item for item in selection_inputs
        }

        evaluations: list[PolicyEvaluation] = []
        for client_id in descriptor.client_ids:
            selection = selection_by_client[client_id]
            benign_test = self.score_cache.read_role(
                score_root,
                client_id,
                DataRole.BENIGN_TEST,
            )
            attack_test = self.score_cache.read_role(
                score_root,
                client_id,
                DataRole.ATTACK_TEST,
            )
            if attack_test.attack_groups is None:
                raise ValueError(
                    f"Final attack scores lack attack-group metadata for {client_id.value}"
                )
            attack_groups = attack_test.attack_groups
            final = FinalTestEvidence(
                benign=selection.benign,
                benign_test_scores=benign_test.values,
                attack_test_scores=attack_test.values,
                attack_test_groups=attack_groups,
            )
            test_scores = np.concatenate(
                (benign_test.values, attack_test.values)
            )
            test_labels = np.concatenate(
                (
                    np.zeros(len(benign_test.values), dtype=np.int64),
                    np.ones(len(attack_test.values), dtype=np.int64),
                )
            )
            test_groups = np.asarray(
                ("__benign__",) * len(benign_test.values)
                + tuple(group.value for group in attack_groups),
                dtype=object,
            )
            ranking_auroc = auroc(test_scores, test_labels)
            ranking_auprc = auprc(test_scores, test_labels)
            oracle_threshold = (
                self._oracle_threshold(final, thresholds, config)
                if PolicyId.ORACLE_TEST in config.policies
                else None
            )

            for policy_id in config.policies:
                threshold = (
                    oracle_threshold
                    if policy_id is PolicyId.ORACLE_TEST
                    else thresholds.for_client(policy_id, client_id)
                )
                if threshold is None:
                    evaluations.append(
                        PolicyEvaluation(
                            client_id,
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
                    band_violation=band_violation(
                        client_fpr,
                        config.protocol.band,
                    ),
                    absolute_fpr_error=absolute_fpr_error(
                        client_fpr,
                        config.protocol.alpha,
                    ),
                    attack_balanced_tpr=attack_balanced_tpr(
                        test_scores,
                        test_labels,
                        test_groups,
                        threshold,
                    ),
                    fpr_reference_interval=clopper_pearson_interval(
                        cm.fp,
                        cm.fp + cm.tn,
                        0.95,
                    ),
                )
                evaluations.append(
                    PolicyEvaluation(
                        client_id,
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
                row.policy is policy
                and row.status is PolicyEvaluationStatus.EVALUATED
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
        run_id: RunId,
        policy: PolicyId,
        bundle: EvaluationBundle,
    ) -> tuple[Path, Path]:
        protocol_by_client = {
            item.client_id: item for item in bundle.protocol_results
        }
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
                    client_id=row.client_id,
                    tau_ref=protocol.reference.value,
                    tau_local=readiness.threshold,
                    selected_tau=row.threshold,
                    readiness_n=readiness.plan.sample_count,
                    readiness_rank=readiness.plan.rank,
                    readiness_probability=readiness.plan.coverage_probability,
                    mismatch_n=mismatch.sample_count,
                    mismatch_x=mismatch.exceedance_count,
                    cp_lower=interval.lower,
                    cp_upper=interval.upper,
                    p_low=mismatch.p_low,
                    p_high=mismatch.p_high,
                    state=protocol.decision.state,
                    tie_count=readiness.tie_count,
                    selected_source=protocol.decision.source,
                    reason_code=protocol.decision.reason,
                )
            )
            if row.metrics is not None:
                metric = row.metrics
                metric_records.append(
                    MetricRecord(
                        run_id=run_id,
                        policy_id=policy,
                        client_id=row.client_id,
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
            (item for item in bundle.federations if item.policy is policy),
            None,
        )
        if federation is not None:
            atomic_write_json(
                root / "metrics" / "federation.json",
                federation,
            )
        return decisions, metrics

    @staticmethod
    def to_serializable(bundle: EvaluationBundle) -> JsonValue:
        return to_json_value(bundle)
