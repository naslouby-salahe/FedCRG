"""Execution spine: experiment plans, lifecycle transitions, dependency order,
frozen model training, immutable score caches, policy-cell materialization,
preflight audits, workload completion, and campaign orchestration.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from graphlib import TopologicalSorter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from pydantic import BaseModel, ConfigDict

from fedcrg.config import ExperimentConfig, ExperimentSpec, Study
from fedcrg.evidence.models import (
    ClientTrainingCount,
    MetricRecord,
    RunConfig,
    RunManifest,
    ThresholdRecord,
    TrainingManifest,
)
from fedcrg.evidence.store import (
    ArtifactVerifier,
    CacheReferenceStore,
    PreparedDatasetManifestStore,
    build_run_id,
    RunManifestStore,
    TrainingManifestStore,
    atomic_write_json,
    atomic_write_text,
    capture_environment,
    write_jsonl,
)
from fedcrg.hashing import sha256_file
from fedcrg.paths import (
    OutputsLayout,
    PreparedDatasetLayout,
    RunLayout,
    ScoreCacheLayout,
    campaign_results_root,
    campaign_status_path,
)
from fedcrg.learning.detectors import (
    DeepSvdd,
    DetectorModel,
    autoencoder_parameter_count,
    autoencoder_tensor_bytes,
    create_detector,
)
from fedcrg.learning.federated import FederatedTrainer
from fedcrg.learning.scores import (
    CalibrationScoreViewBuilder,
    CalibrationScoreViews,
    ScoreCache,
    ScoreCacheDescriptor,
    ScoreCacheIdentity,
    ScoreComputer,
    ScoreManifest,
)
from fedcrg.runtime import get_logger
from fedcrg.thresholding.metrics import (
    ClientEvaluation,
    ClientMetrics,
    EvaluationBundle,
    FederationMetrics,
    PolicyEvaluation,
    summarize_admission,
)
from fedcrg.thresholding.readiness import (
    BinomialCounts,
    ClientEvaluationResult,
    ReadinessPlanCache,
    clopper_pearson_interval,
)
from fedcrg.types import PolicyEvaluationStatus
from fedcrg.thresholding.policies import (
    SUPERVISED_POLICIES,
    BenignPolicyEvidence,
    FinalTestEvidence,
    PolicyThresholdSelector,
    SupervisedDevelopmentEvidence,
    oracle_choice,
)
from fedcrg.types import (
    CalibrationAssignmentMode,
    CalibrationSeed,
    CampaignId,
    CampaignStage,
    AttackGroupId,
    ClientId,
    DataRole,
    ExperimentId,
    ExperimentStatus,
    ExperimentType,
    FailureCode,
    FeatureName,
    Identifier,
    ModelSeed,
    PathString,
    PolicyId,
    PositiveCount,
    PreparedColumn,
    RunId,
    RowId,
    Sha256,
    Threshold,
    Timestamp,
    Duration,
)
from fedcrg.experiments.analyses import (
    ProtocolTablePrecomputer,
    RunBenchmark,
    RunSyntheticExperiments,
)

Frozen = ConfigDict(frozen=True)
_LOGGER = get_logger(__name__)

_ANALYSIS_CATEGORIES = frozenset({ExperimentType.SYNTHETIC, ExperimentType.BENCHMARK})


class PreflightReport(BaseModel):
    """Outcome of the research preflight audit."""

    model_config = Frozen

    valid: bool
    problems: tuple[Identifier, ...]


def _tensor_sha256(tensor: torch.Tensor) -> Sha256:
    import hashlib

    digest = hashlib.sha256()
    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class ExperimentPlan(BaseModel):
    """One validated execution plan for a policy cell."""

    model_config = Frozen

    definition: ExperimentSpec
    config_hash: Sha256
    model_seed: ModelSeed
    calibration_seed: CalibrationSeed


_ALLOWED_TRANSITIONS: dict[ExperimentStatus, tuple[ExperimentStatus, ...]] = {
    ExperimentStatus.PENDING: (ExperimentStatus.VALIDATING, ExperimentStatus.BLOCKED),
    ExperimentStatus.VALIDATING: (
        ExperimentStatus.READY,
        ExperimentStatus.INVALID,
        ExperimentStatus.FAILED,
    ),
    ExperimentStatus.READY: (ExperimentStatus.RUNNING, ExperimentStatus.BLOCKED),
    ExperimentStatus.RUNNING: (ExperimentStatus.VERIFYING, ExperimentStatus.FAILED),
    ExperimentStatus.VERIFYING: (ExperimentStatus.COMPLETE, ExperimentStatus.FAILED),
    ExperimentStatus.COMPLETE: (),
    ExperimentStatus.FAILED: (),
    ExperimentStatus.BLOCKED: (),
    ExperimentStatus.INVALID: (),
}


def assert_transition(current: ExperimentStatus, target: ExperimentStatus) -> None:
    """Reject a lifecycle transition outside the locked table."""
    if target not in _ALLOWED_TRANSITIONS[current]:
        raise ValueError(f"Invalid experiment transition: {current.value} -> {target.value}")


class DependencyResolver:
    """Evaluate dependencies and produce a deterministic topological order."""

    def __init__(self, study: Study | None = None) -> None:
        self.study = study or Study.load()

    def blockers(
        self,
        experiment_id: ExperimentId,
        statuses: Mapping[ExperimentId, ExperimentStatus],
    ) -> tuple[ExperimentId, ...]:
        return tuple(
            dependency
            for dependency in self.study.catalogue.spec(experiment_id).dependencies
            if statuses.get(dependency) is not ExperimentStatus.COMPLETE
        )

    def order(self, experiment_ids: tuple[ExperimentId, ...]) -> tuple[ExperimentId, ...]:
        requested = set(experiment_ids)
        expanded = set(requested)
        stack = list(requested)
        while stack:
            current = stack.pop()
            for dependency in self.study.catalogue.spec(current).dependencies:
                if dependency not in expanded:
                    expanded.add(dependency)
                    stack.append(dependency)
        sorter = TopologicalSorter(
            {
                experiment_id: self.study.catalogue.spec(experiment_id).dependencies
                for experiment_id in sorted(expanded, key=str)
            }
        )
        return tuple(item for item in sorter.static_order() if item in expanded)


class ExperimentPlanner:
    """Construct one execution plan from a validated configuration."""

    def __init__(self, study: Study | None = None) -> None:
        self.study = study or Study.load()

    def create(
        self,
        experiment_id: ExperimentId,
        config: ExperimentConfig,
        model_seed: ModelSeed,
        calibration_seed: CalibrationSeed,
    ) -> ExperimentPlan:
        if config.id is not experiment_id:
            raise ValueError(
                f"Experiment identity mismatch: plan={experiment_id.value}, "
                f"config={config.id.value}"
            )
        if int(model_seed) not in config.randomness.model_seeds:
            raise ValueError(f"Model seed {int(model_seed)} is not configured")
        if int(calibration_seed) not in config.dataset.calibration_seeds:
            raise ValueError(f"Calibration seed {int(calibration_seed)} is not configured")
        definition = self.study.catalogue.spec(experiment_id)
        if definition.policies and not set(config.policies).issubset(definition.policies):
            extra = set(config.policies) - set(definition.policies)
            raise ValueError(
                "Config contains policies outside the experiment catalogue: "
                + ", ".join(sorted(item.value for item in extra))
            )
        return ExperimentPlan(
            definition=definition,
            config_hash=config.config_hash,
            model_seed=model_seed,
            calibration_seed=calibration_seed,
        )


class RunExperiment:
    """Manage one policy-cell lifecycle without owning scientific computation."""

    def __init__(
        self,
        planner: ExperimentPlanner | None = None,
        manifests: RunManifestStore | None = None,
        verifier: ArtifactVerifier | None = None,
    ) -> None:
        self.planner = planner or ExperimentPlanner()
        self.manifests = manifests or RunManifestStore()
        self.verifier = verifier or ArtifactVerifier()

    @staticmethod
    def _manifest(
        layout: RunLayout,
        plan: ExperimentPlan,
        policy: PolicyId,
        status: ExperimentStatus,
    ) -> RunManifest:
        return RunManifest(
            run_id=layout.root.name,
            experiment_id=plan.definition.id,
            policy_id=policy,
            config_hash=plan.config_hash,
            model_seed=plan.model_seed,
            calibration_seed=plan.calibration_seed,
            status=status,
        )

    def _transition(
        self,
        layout: RunLayout,
        plan: ExperimentPlan,
        policy: PolicyId,
        target: ExperimentStatus,
    ) -> None:
        current = self.manifests.load(layout.manifest).status
        assert_transition(current, target)
        self.manifests.save(layout.manifest, self._manifest(layout, plan, policy, target))

    def prepare(
        self,
        experiment_id: ExperimentId,
        config: ExperimentConfig,
        model_seed: ModelSeed,
        calibration_seed: CalibrationSeed,
        policy: PolicyId,
        repository_root: Path,
    ) -> tuple[ExperimentPlan, RunLayout]:
        if policy not in config.policies:
            raise ValueError(f"Policy {policy.value} is not configured for this experiment")
        plan = self.planner.create(experiment_id, config, model_seed, calibration_seed)
        run_id = build_run_id(config, model_seed, calibration_seed, policy)
        layout = OutputsLayout(config.outputs_root).run(run_id)
        layout.create()
        self.manifests.save(
            layout.manifest,
            self._manifest(layout, plan, policy, ExperimentStatus.PENDING),
        )
        self._transition(layout, plan, policy, ExperimentStatus.VALIDATING)
        self._transition(layout, plan, policy, ExperimentStatus.READY)
        atomic_write_text(
            layout.resolved_config,
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        )
        environment = capture_environment(repository_root)
        atomic_write_json(layout.environment, environment)
        atomic_write_json(
            layout.run_config,
            RunConfig(
                run_id=run_id,
                experiment_id=experiment_id,
                policy_id=policy,
                parameters=config,
                model_seed=model_seed,
                calibration_seed=calibration_seed,
                config_hash=config.config_hash,
                data_spec_hash=config.data_spec_hash,
                training_spec_hash=config.training_spec_hash,
                git_commit=environment.git_commit,
                git_clean=environment.git_clean,
                git_patch_sha256=environment.git_patch_sha256,
                environment_pin_sha256=environment.environment_pin_sha256,
                environment_pin_kind=environment.environment_pin_kind,
            ),
        )
        return plan, layout

    def execute(
        self,
        experiment_id: ExperimentId,
        config: ExperimentConfig,
        model_seed: ModelSeed,
        calibration_seed: CalibrationSeed,
        policy: PolicyId,
        runner: Callable[[ExperimentPlan, RunLayout], FederationMetrics | None],
        repository_root: Path,
    ) -> tuple[FederationMetrics | None, RunLayout]:
        plan, layout = self.prepare(
            experiment_id,
            config,
            model_seed,
            calibration_seed,
            policy,
            repository_root,
        )
        self._transition(layout, plan, policy, ExperimentStatus.RUNNING)
        try:
            result = runner(plan, layout)
            self._transition(layout, plan, policy, ExperimentStatus.VERIFYING)
            verification = self.verifier.record(layout, plan.definition)
            if not verification.valid:
                raise RuntimeError(
                    "Run verification failed: "
                    + ", ".join(verification.missing + verification.mismatched)
                )
        except Exception:
            self._transition(layout, plan, policy, ExperimentStatus.FAILED)
            raise
        self._transition(layout, plan, policy, ExperimentStatus.COMPLETE)
        return result, layout


def _oracle_candidate_missing(client_id: ClientId) -> Threshold:
    """Raise when an oracle candidate threshold was not prepared by selection."""
    raise RuntimeError(f"Oracle candidate threshold is unavailable for {client_id}")


def feature_columns(frame: pd.DataFrame, expected_count: PositiveCount) -> tuple[FeatureName, ...]:
    """Resolve model-feature columns of a prepared frame."""
    metadata = {column.value for column in PreparedColumn}
    columns = tuple(
        column
        for column in frame.columns
        if column not in metadata
        and not column.startswith("_")
        and pd.api.types.is_numeric_dtype(frame[column])
    )
    if len(columns) != expected_count:
        raise ValueError(
            f"{FailureCode.FEATURE_SCHEMA_MISMATCH.value}: expected {expected_count} "
            f"numeric features, found {len(columns)}"
        )
    return columns


class TrainDetector:
    """Train one frozen detector per training spec/model seed and never policy-tune it."""

    def __init__(
        self,
        trainer: FederatedTrainer | None = None,
        manifests: TrainingManifestStore | None = None,
        dataset_manifests: PreparedDatasetManifestStore | None = None,
    ) -> None:
        self.trainer = trainer or FederatedTrainer()
        self.manifests = manifests or TrainingManifestStore()
        self.dataset_manifests = dataset_manifests or PreparedDatasetManifestStore()

    def create_model(self, config: ExperimentConfig) -> DetectorModel:
        if config.detector is None:
            raise ValueError("Training requires a detector profile")
        return create_detector(config.dataset.feature_count, config.detector)

    def _validate_architecture(self, config: ExperimentConfig, model: DetectorModel) -> None:
        if config.detector is None:
            return
        expected_parameters = autoencoder_parameter_count(
            config.dataset.feature_count, config.detector.hidden_dims
        )
        expected_bytes = autoencoder_tensor_bytes(
            config.dataset.feature_count, config.detector.hidden_dims
        )
        if model.trainable_parameter_count() != expected_parameters:
            raise RuntimeError(
                "Detector parameter-count contract failed: "
                f"{model.trainable_parameter_count()} != {expected_parameters}"
            )
        if model.trainable_tensor_bytes() != expected_bytes:
            raise RuntimeError(
                "Detector tensor-byte contract failed: "
                f"{model.trainable_tensor_bytes()} != {expected_bytes}"
            )

    def train_from_cache(
        self,
        config: ExperimentConfig,
        prepared_root: Path,
        model_seed: ModelSeed,
    ) -> tuple[Path, Path]:
        if int(model_seed) not in config.randomness.model_seeds:
            raise ValueError(f"Model seed {int(model_seed)} is not configured")

        prepared_layout = PreparedDatasetLayout(prepared_root)
        prepared_manifest_path = prepared_layout.manifest
        preprocessing_path = prepared_layout.preprocessing
        prepared_manifest = self.dataset_manifests.load_model(prepared_manifest_path)
        if prepared_manifest.data_spec_hash != config.data_spec_hash:
            raise ValueError("Prepared dataset data-spec hash does not match training config")
        if prepared_manifest.dataset_id is not config.dataset.id:
            raise ValueError("Prepared dataset identity does not match training config")
        prepared_manifest_hash = sha256_file(prepared_manifest_path)
        preprocessing_hash = sha256_file(preprocessing_path)

        datasets: dict[ClientId, torch.Tensor] = {}
        training_rows: list[ClientTrainingCount] = []
        for client_manifest in prepared_manifest.clients:
            client_id = client_manifest.client_id
            train_role = client_manifest.role(DataRole.TRAIN)
            path = prepared_root / train_role.relative_path
            if sha256_file(path) != train_role.file_sha256:
                raise ValueError(f"Prepared training file hash mismatch for {client_id}")
            frame = pd.read_csv(path)
            if len(frame) != config.dataset.split.train_benign:
                raise ValueError(
                    f"Training-row contract failed for {client_id}: {len(frame)} != "
                    f"{config.dataset.split.train_benign}"
                )
            columns = feature_columns(frame, config.dataset.feature_count)
            if tuple(columns) != prepared_manifest.feature_names:
                raise ValueError("Training feature order differs from frozen dataset manifest")
            tensor = torch.as_tensor(
                frame[list(columns)].to_numpy(dtype="float32"),
                dtype=torch.float32,
            )
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(FailureCode.TRAINING_NUMERICAL_FAILURE.value)
            datasets[client_id] = tensor
            training_rows.append(ClientTrainingCount(client_id=client_id, rows=len(frame)))

        if config.detector is None:
            raise ValueError("Training requires a detector profile")
        model_cache = OutputsLayout(config.outputs_root).model_cache(config, model_seed)
        model_root = model_cache.root
        model_path = model_cache.model
        manifest_path = model_cache.training_manifest
        if model_path.exists() or manifest_path.exists():
            self._validate_existing_cache(
                config,
                model_seed,
                model_path,
                manifest_path,
                prepared_manifest_hash,
                preprocessing_hash,
            )
            return model_path, manifest_path

        torch.manual_seed(int(model_seed))
        model = self.create_model(config)
        self._validate_architecture(config, model)
        center_hash: Sha256 | None = None
        if isinstance(model, DeepSvdd):
            model.initialize_center([dataset for dataset in datasets.values()])
            center_hash = _tensor_sha256(model.center)

        training = config.training
        if training is None:
            raise ValueError("Training requires a training profile")
        final_model, result = self.trainer.train(
            model,
            {
                client_id: torch.utils.data.TensorDataset(tensor)
                for client_id, tensor in datasets.items()
            },
            training,
            model_seed,
        )
        model_root.mkdir(parents=True, exist_ok=False)
        temp_model = model_root / ".model.pt.tmp"
        torch.save(final_model.state_dict(), temp_model)
        import os as _os

        _os.replace(temp_model, model_path)

        manifest = TrainingManifest(
            dataset_id=config.dataset.id,
            detector_id=config.detector.id,
            model_seed=model_seed,
            data_spec_hash=config.data_spec_hash,
            training_spec_hash=config.training_spec_hash,
            dataset_manifest_sha256=prepared_manifest_hash,
            preprocessing_sha256=preprocessing_hash,
            model_file_sha256=sha256_file(model_path),
            deep_svdd_center_sha256=center_hash,
            training_rows=tuple(training_rows),
            result=result,
        )
        self.manifests.save(manifest_path, manifest)
        return model_path, manifest_path

    def _validate_existing_cache(
        self,
        config: ExperimentConfig,
        model_seed: ModelSeed,
        model_path: Path,
        manifest_path: Path,
        prepared_manifest_hash: Sha256,
        preprocessing_hash: Sha256,
    ) -> None:
        if not model_path.is_file() or not manifest_path.is_file():
            raise FileExistsError("Model cache is partially materialized")
        manifest = self.manifests.load_model(manifest_path)
        if manifest.model_seed != model_seed:
            raise ValueError("Existing model cache has a different model seed")
        if manifest.data_spec_hash != config.data_spec_hash:
            raise ValueError("Existing model cache belongs to another data specification")
        if manifest.training_spec_hash != config.training_spec_hash:
            raise ValueError("Existing model cache belongs to another training specification")
        if manifest.dataset_manifest_sha256 != prepared_manifest_hash:
            raise ValueError("Existing model cache was trained from a different prepared manifest")
        if manifest.preprocessing_sha256 != preprocessing_hash:
            raise ValueError(
                "Existing model cache was trained with different preprocessing evidence"
            )
        if manifest.model_file_sha256 != sha256_file(model_path):
            raise ValueError("Existing frozen-model hash does not match its manifest")

    def load_model(self, config: ExperimentConfig, model_path: Path) -> DetectorModel:
        model = self.create_model(config)
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        self._validate_architecture(config, model)
        return model


class ComputeScores:
    """Generate one hash-finalized immutable score cache per frozen model seed."""

    def __init__(
        self,
        computer: ScoreComputer | None = None,
        trainer: TrainDetector | None = None,
        cache: ScoreCache | None = None,
        training_manifests: TrainingManifestStore | None = None,
        dataset_manifests: PreparedDatasetManifestStore | None = None,
    ) -> None:
        self.computer = computer or ScoreComputer()
        self.trainer = trainer or TrainDetector()
        self.cache = cache or ScoreCache()
        self.training_manifests = training_manifests or TrainingManifestStore()
        self.dataset_manifests = dataset_manifests or PreparedDatasetManifestStore()

    def score_from_cache(
        self,
        config: ExperimentConfig,
        prepared_root: Path,
        model_path: Path,
        model_seed: ModelSeed,
        training_manifest: Path,
    ) -> Path:
        prepared_layout = PreparedDatasetLayout(prepared_root)
        manifest_path = prepared_layout.manifest
        preprocessing_path = prepared_layout.preprocessing
        prepared_manifest = self.dataset_manifests.load_model(manifest_path)
        if prepared_manifest.data_spec_hash != config.data_spec_hash:
            raise ValueError("Prepared dataset data-spec hash does not match scoring config")
        if prepared_manifest.dataset_id is not config.dataset.id:
            raise ValueError("Prepared dataset identity does not match scoring config")
        prepared_manifest_hash = sha256_file(manifest_path)
        preprocessing_hash = sha256_file(preprocessing_path)

        training = self.training_manifests.load_model(training_manifest)
        if training.training_spec_hash != config.training_spec_hash:
            raise ValueError("Frozen model belongs to another training specification")
        if training.data_spec_hash != config.data_spec_hash:
            raise ValueError("Frozen model belongs to another data specification")
        if training.model_seed != model_seed:
            raise ValueError("Training manifest model seed does not match scoring request")
        if training.model_file_sha256 != sha256_file(model_path):
            raise ValueError("Frozen model file hash does not match training manifest")
        if training.dataset_manifest_sha256 != prepared_manifest_hash:
            raise ValueError("Training manifest does not reference this prepared dataset manifest")
        if training.preprocessing_sha256 != preprocessing_hash:
            raise ValueError("Training manifest does not reference this preprocessing artifact")

        model = self.trainer.load_model(config, model_path)
        model_hash = model.state_hash()
        if model_hash != training.result.final_model_hash:
            raise ValueError("Frozen model state does not match the training result hash")
        identity = ScoreCacheIdentity(
            dataset=config.dataset.id,
            model_seed=model_seed,
            model_hash=model_hash,
            data_spec_hash=config.data_spec_hash,
            training_spec_hash=config.training_spec_hash,
            dataset_manifest_hash=prepared_manifest_hash,
            preprocessing_hash=preprocessing_hash,
        )
        if config.detector is None:
            raise ValueError("Scoring requires a detector profile")
        score_root = OutputsLayout(config.outputs_root).score_cache(config, model_seed).root
        if score_root.exists():
            descriptor = self.cache.load_descriptor(score_root)
            if descriptor.identity != identity:
                raise ValueError("Existing score cache identity differs from scoring request")
            _LOGGER.info("score cache reused %s", score_root)
            return score_root

        clients: list[
            tuple[
                ClientId,
                list[
                    tuple[
                        DataRole,
                        np.ndarray,
                        tuple[AttackGroupId, ...],
                        tuple[RowId, ...],
                    ]
                ],
            ]
        ] = []
        for client_manifest in prepared_manifest.clients:
            client_id = client_manifest.client_id
            role_inputs: list[
                tuple[DataRole, np.ndarray, tuple[AttackGroupId, ...], tuple[RowId, ...]]
            ] = []
            for role in _BASE_SCORE_ROLES:
                role_manifest = client_manifest.role(role)
                path = prepared_root / role_manifest.relative_path
                if sha256_file(path) != role_manifest.file_sha256:
                    raise ValueError(f"Prepared role hash mismatch for {client_id}/{role.value}")
                frame = pd.read_csv(path)
                values = frame[
                    [column for column in frame.columns if column not in _METADATA_COLUMNS]
                ].to_numpy(dtype=np.float64)
                row_ids = tuple(
                    str(value) for value in frame[PreparedColumn.ROW_ID.value].astype(str)
                )
                if len(row_ids) != len(values):
                    raise ValueError(
                        f"Prepared role row ids do not align for {client_id}/{role.value}"
                    )
                if (
                    role is DataRole.ATTACK_TEST
                    and PreparedColumn.ATTACK_GROUP.value in frame.columns
                ):
                    groups = tuple(
                        str(value) for value in frame[PreparedColumn.ATTACK_GROUP.value].astype(str)
                    )
                else:
                    groups = ()
                role_inputs.append((role, values, groups, row_ids))
            clients.append((client_id, role_inputs))

        role_scores = []
        from fedcrg.learning.scores import ClientScoreSet, RoleScores

        for client_id, inputs in clients:
            training = config.training
            if training is None:
                raise ValueError("Scoring requires a training profile")
            scored = tuple(
                RoleScores(
                    role=role,
                    values=self.computer.compute(
                        model, values, training.device, training.batch_size
                    ),
                    client_id=client_id,
                    row_ids=row_ids,
                    attack_groups=groups if groups else None,
                )
                for role, values, groups, row_ids in inputs
            )
            role_scores.append(ClientScoreSet(client_id=client_id, scores=scored))
        manifest = ScoreManifest(
            dataset=config.dataset.id,
            model_seed=model_seed,
            model_hash=model_hash,
            data_spec_hash=config.data_spec_hash,
            training_spec_hash=config.training_spec_hash,
            dataset_manifest_hash=prepared_manifest_hash,
            preprocessing_hash=preprocessing_hash,
            clients=tuple(role_scores),
        )
        self.cache.save(manifest, score_root)
        _LOGGER.info("score cache materialized %s", score_root)
        return score_root


_METADATA_COLUMNS = {column.value for column in PreparedColumn}

_BASE_SCORE_ROLES = (
    DataRole.TRAIN,
    DataRole.RESERVOIR,
    DataRole.BENIGN_TEST,
    DataRole.ATTACK_DEV,
    DataRole.ATTACK_TEST,
)


class EvaluatePolicies:
    """Freeze thresholds before opening final-test evidence."""

    def __init__(
        self,
        selector: PolicyThresholdSelector | None = None,
        views: CalibrationScoreViewBuilder | None = None,
        score_cache: ScoreCache | None = None,
        protocol: ClientEvaluation | None = None,
    ) -> None:
        self.selector = selector or PolicyThresholdSelector()
        self.views = views or CalibrationScoreViewBuilder()
        self.score_cache = score_cache or ScoreCache()
        self.protocol = protocol or ClientEvaluation()

    @staticmethod
    def _validate_score_identity(
        config: ExperimentConfig,
        descriptor: ScoreCacheDescriptor,
    ) -> None:
        identity = descriptor.identity
        if identity.dataset is not config.dataset.id:
            raise ValueError("Score cache dataset does not match experiment config")
        if identity.data_spec_hash != config.data_spec_hash:
            raise ValueError("SCORE_CACHE_HASH_MISMATCH: data specification differs")
        if identity.training_spec_hash != config.training_spec_hash:
            raise ValueError("SCORE_CACHE_HASH_MISMATCH: training specification differs")

    def evaluate_from_cache(
        self,
        config: ExperimentConfig,
        score_root: Path,
        calibration_seed: CalibrationSeed,
        mode: CalibrationAssignmentMode = CalibrationAssignmentMode.SEEDED_PERMUTATION,
    ) -> EvaluationBundle:
        descriptor = self.score_cache.load_descriptor(score_root)
        self._validate_score_identity(config, descriptor)
        # Real-data evaluation consumes the frozen pre-data readiness-plan table;
        # a missing table is a missing precomputation step, not an on-the-fly gap.
        plans_path = OutputsLayout(config.outputs_root).readiness_plans_file
        self.protocol = ClientEvaluation(readiness_cache=ReadinessPlanCache(plans_path))
        manifest = self.score_cache.load(score_root)
        views = self.views.build(
            manifest,
            config.dataset,
            calibration_seed,
            mode,
        )
        protocol_results = self._protocol_results(config, views)
        benign_inputs = self._benign_inputs(views, protocol_results)
        supervised_requested = any(policy in SUPERVISED_POLICIES for policy in config.policies)
        supervised_inputs = (
            self._supervised_inputs(score_root, views, benign_inputs)
            if supervised_requested
            else None
        )
        thresholds = self.selector.select(
            benign_inputs,
            config.protocol,
            config.statistics,
            config.policies,
            supervised_inputs,
        )
        benign_by_client = {item.client_id: item for item in benign_inputs}

        from fedcrg.thresholding.metrics import (
            absolute_fpr_error,
            aggregate_policy,
            assert_ranking_metric_invariance,
            attack_balanced_tpr,
            auprc,
            auroc,
            balanced_accuracy,
            band_error,
            band_violation,
            confusion_matrix,
            f1,
            fpr,
            high_excess,
            precision,
            recall,
            tpr,
        )

        evaluations: list[PolicyEvaluation] = []
        oracle_requested = PolicyId.ORACLE_TEST in config.policies
        oracle_thresholds: dict[ClientId, Threshold] = {}
        for client_id in descriptor.client_ids:
            benign_test = self.score_cache.read_role(score_root, client_id, DataRole.BENIGN_TEST)
            attack_test = self.score_cache.read_role(score_root, client_id, DataRole.ATTACK_TEST)
            if attack_test.attack_groups is None:
                raise ValueError(f"Final attack scores lack attack-group metadata for {client_id}")
            attack_groups = attack_test.attack_groups
            final = FinalTestEvidence(
                benign=benign_by_client[client_id],
                benign_test_scores=benign_test.values,
                attack_test_scores=attack_test.values,
                attack_test_groups=attack_groups,
            )
            if oracle_requested:
                oracle_thresholds[client_id] = oracle_choice(
                    final,
                    (
                        thresholds.for_client(PolicyId.GLOBAL_QUANTILE, client_id)
                        or _oracle_candidate_missing(client_id),
                        thresholds.for_client(PolicyId.LOCAL_QUANTILE, client_id)
                        or _oracle_candidate_missing(client_id),
                        benign_by_client[client_id].evaluation.decision.threshold,
                    ),
                    config.protocol.band,
                )
            test_scores = np.concatenate((benign_test.values, attack_test.values))
            test_labels = np.concatenate(
                (
                    np.zeros(len(benign_test.values), dtype=np.int64),
                    np.ones(len(attack_test.values), dtype=np.int64),
                )
            )
            test_groups = np.asarray(
                ("__benign__",) * len(benign_test.values) + tuple(group for group in attack_groups),
                dtype=object,
            )
            ranking_auroc = auroc(test_scores, test_labels)
            ranking_auprc = auprc(test_scores, test_labels)

            for policy_id in config.policies:
                if policy_id is PolicyId.ORACLE_TEST:
                    threshold = oracle_thresholds[client_id]
                else:
                    threshold = thresholds.for_client(policy_id, client_id)
                if threshold is None:
                    evaluations.append(
                        PolicyEvaluation(
                            client_id=client_id,
                            policy=policy_id,
                            threshold=None,
                            status=PolicyEvaluationStatus.UNDEFINED,
                            metrics=None,
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
                        BinomialCounts(cm.fp, cm.fp + cm.tn),
                        config.protocol.mismatch_confidence,
                    ),
                )
                evaluations.append(
                    PolicyEvaluation(
                        client_id=client_id,
                        policy=policy_id,
                        threshold=threshold,
                        status=PolicyEvaluationStatus.EVALUATED,
                        metrics=client_metrics,
                    )
                )

        client_rows = tuple(evaluations)
        assert_ranking_metric_invariance(
            client_rows, tolerance=config.statistics.ranking_invariance_tolerance
        )
        federation_rows = tuple(aggregate_policy(policy, client_rows) for policy in config.policies)
        return EvaluationBundle(
            clients=client_rows,
            federations=federation_rows,
            protocol_results=tuple(protocol_results.values()),
            shrinkage_n0=thresholds.shrinkage_n0,
        )

    def _protocol_results(
        self,
        config: ExperimentConfig,
        views: CalibrationScoreViews,
    ) -> dict[ClientId, ClientEvaluationResult]:
        reference_by_client = {
            client_id: views.get(client_id, DataRole.REFERENCE).values
            for client_id in views.client_ids
        }
        reference = self.protocol.estimate_reference(reference_by_client, config.protocol)
        calibration_sizes = {
            len(views.get(client_id, DataRole.CALIBRATION).values) for client_id in views.client_ids
        }
        if len(calibration_sizes) != 1:
            raise ValueError("Calibration evidence count must be identical across clients")
        plan = self.protocol.require_readiness(calibration_sizes.pop(), config.protocol)
        return {
            client_id: self.protocol.evaluate_client(
                client_id=client_id,
                reference=reference,
                calibration_scores=views.get(client_id, DataRole.CALIBRATION).values,
                mismatch_scores=views.get(client_id, DataRole.MISMATCH).values,
                config=config.protocol,
                readiness_plan=plan,
            )
            for client_id in views.client_ids
        }

    @staticmethod
    def _benign_inputs(
        views: CalibrationScoreViews,
        protocol_results: Mapping[ClientId, ClientEvaluationResult],
    ) -> tuple[BenignPolicyEvidence, ...]:
        return tuple(
            BenignPolicyEvidence(
                client_id=client_id,
                reference_scores=views.get(client_id, DataRole.REFERENCE).values,
                mismatch_scores=views.get(client_id, DataRole.MISMATCH).values,
                calibration_scores=views.get(client_id, DataRole.CALIBRATION).values,
                evaluation=protocol_results[client_id],
            )
            for client_id in views.client_ids
        )

    def _supervised_inputs(
        self,
        score_root: Path,
        views: CalibrationScoreViews,
        benign_clients: tuple[BenignPolicyEvidence, ...],
    ) -> tuple[SupervisedDevelopmentEvidence, ...]:
        benign_by_client = {client.client_id: client for client in benign_clients}
        return tuple(
            SupervisedDevelopmentEvidence(
                benign=benign_by_client[client_id],
                benign_guard_scores=views.get(client_id, DataRole.BENIGN_GUARD).values,
                attack_dev_scores=self.score_cache.read_role(
                    score_root,
                    client_id,
                    DataRole.ATTACK_DEV,
                ).values,
            )
            for client_id in views.client_ids
        )

    def write_policy_artifacts(
        self,
        layout: RunLayout,
        run_id: RunId,
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
        decisions = layout.threshold_records
        metrics = layout.metric_records
        write_jsonl(decisions, tuple(threshold_records))
        write_jsonl(metrics, tuple(metric_records))
        federation = next(
            (item for item in bundle.federations if item.policy is policy),
            None,
        )
        if federation is not None:
            atomic_write_json(layout.federation_metrics, federation)
        atomic_write_json(
            layout.admission,
            summarize_admission(bundle.protocol_results),
        )
        return decisions, metrics


@dataclass(frozen=True, slots=True)
class FrozenCacheInputs:
    """Frozen upstream cache paths for one model seed."""

    prepared_root: Path
    model_path: Path
    training_manifest: Path
    score_root: Path


class PolicyCellMaterializer:
    """Create policy evidence while sharing one frozen federation evaluation."""

    def __init__(
        self,
        evaluator: EvaluatePolicies | None = None,
        score_cache: ScoreCache | None = None,
        references: CacheReferenceStore | None = None,
        training_manifests: TrainingManifestStore | None = None,
        dataset_manifests: PreparedDatasetManifestStore | None = None,
    ) -> None:
        self.evaluator = evaluator or EvaluatePolicies()
        self.score_cache = score_cache or ScoreCache()
        self.references = references or CacheReferenceStore()
        self.training_manifests = training_manifests or TrainingManifestStore()
        self.dataset_manifests = dataset_manifests or PreparedDatasetManifestStore()

    def evaluate_federation(
        self,
        config: ExperimentConfig,
        caches: FrozenCacheInputs,
        calibration_seed: CalibrationSeed,
        assignment_mode: CalibrationAssignmentMode = CalibrationAssignmentMode.SEEDED_PERMUTATION,
    ) -> EvaluationBundle:
        self.validate_upstream(config, caches)
        descriptor = self.score_cache.load_descriptor(caches.score_root)
        if descriptor.identity.data_spec_hash != config.data_spec_hash:
            raise ValueError("SCORE_CACHE_HASH_MISMATCH: data specification differs")
        if descriptor.identity.training_spec_hash != config.training_spec_hash:
            raise ValueError("SCORE_CACHE_HASH_MISMATCH: training specification differs")
        return self.evaluator.evaluate_from_cache(
            config,
            caches.score_root,
            calibration_seed=calibration_seed,
            mode=assignment_mode,
        )

    def materialize_analysis(
        self,
        config: ExperimentConfig,
        policy: PolicyId,
        layout: RunLayout,
        caches: FrozenCacheInputs,
        calibration_seed: CalibrationSeed,
        bundle: EvaluationBundle,
        assignment_mode: CalibrationAssignmentMode = CalibrationAssignmentMode.SEEDED_PERMUTATION,
    ) -> FederationMetrics | None:
        self._copy_manifests(layout, caches)
        self._write_cache_references(config, layout, caches)
        descriptor = self.score_cache.load_descriptor(caches.score_root)
        self.evaluator.write_policy_artifacts(
            layout,
            layout.root.name,
            policy,
            bundle,
        )
        atomic_write_json(
            layout.evaluation_summary,
            {
                "calibration_seed": int(calibration_seed),
                "calibration_assignment": assignment_mode.value,
                "score_cache_sha256": descriptor.cache_sha256,
                "evaluation": bundle.model_dump(mode="json"),
            },
        )
        return next((item for item in bundle.federations if item.policy is policy), None)

    def validate_upstream(
        self,
        config: ExperimentConfig,
        caches: FrozenCacheInputs,
    ) -> None:
        prepared_layout = PreparedDatasetLayout(caches.prepared_root)
        score_layout = ScoreCacheLayout(caches.score_root)
        prepared_manifest_path = prepared_layout.manifest
        preprocessing_path = prepared_layout.preprocessing
        required = (
            prepared_manifest_path,
            preprocessing_path,
            caches.model_path,
            caches.training_manifest,
            score_layout.score,
            score_layout.manifest,
        )
        missing = tuple(path for path in required if not path.is_file())
        if missing:
            raise FileNotFoundError(
                "Missing frozen upstream artifact(s): " + ", ".join(str(path) for path in missing)
            )

        prepared = self.dataset_manifests.load_model(prepared_manifest_path)
        if prepared.data_spec_hash != config.data_spec_hash:
            raise ValueError("Prepared dataset provenance does not match requested cell")
        if prepared.dataset_id is not config.dataset.id:
            raise ValueError("Prepared dataset identity does not match requested cell")

        training = self.training_manifests.load_model(caches.training_manifest)
        if training.data_spec_hash != config.data_spec_hash:
            raise ValueError("Training data specification does not match requested cell")
        if training.training_spec_hash != config.training_spec_hash:
            raise ValueError("Training specification does not match requested cell")
        if training.dataset_manifest_sha256 != sha256_file(prepared_manifest_path):
            raise ValueError("Training manifest references a different dataset manifest")
        if training.preprocessing_sha256 != sha256_file(preprocessing_path):
            raise ValueError("Training manifest references a different preprocessing artifact")
        if training.model_file_sha256 != sha256_file(caches.model_path):
            raise ValueError("Frozen model hash does not match its training manifest")

        descriptor = self.score_cache.load_descriptor(caches.score_root)
        if descriptor.identity.data_spec_hash != training.data_spec_hash:
            raise ValueError("Score cache references a different data specification")
        if descriptor.identity.training_spec_hash != training.training_spec_hash:
            raise ValueError("Score cache references a different training specification")
        if descriptor.identity.dataset_manifest_hash != training.dataset_manifest_sha256:
            raise ValueError("Score cache references a different dataset manifest")
        if descriptor.identity.preprocessing_hash != training.preprocessing_sha256:
            raise ValueError("Score cache references a different preprocessing artifact")
        if descriptor.identity.model_hash != training.result.final_model_hash:
            raise ValueError("Score-cache model state does not match training result")

    @staticmethod
    def _copy(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_name(f".{destination.name}.tmp")
        shutil.copyfile(source, temp)
        temp.replace(destination)

    def _copy_manifests(
        self,
        layout: RunLayout,
        caches: FrozenCacheInputs,
    ) -> None:
        prepared_layout = PreparedDatasetLayout(caches.prepared_root)
        self._copy(prepared_layout.manifest, layout.dataset_manifest)
        self._copy(prepared_layout.preprocessing, layout.preprocessing_evidence)
        self._copy(caches.training_manifest, layout.training_manifest)
        self._copy(ScoreCacheLayout(caches.score_root).manifest, layout.score_manifest)

    def _write_cache_references(
        self,
        config: ExperimentConfig,
        layout: RunLayout,
        caches: FrozenCacheInputs,
    ) -> None:
        self.references.save(
            layout.model_reference,
            self.references.build(caches.model_path, config.outputs_root),
        )
        self.references.save(
            layout.score_reference,
            self.references.build(
                ScoreCacheLayout(caches.score_root).score,
                config.outputs_root,
            ),
        )


@dataclass(frozen=True, slots=True)
class PolicyRunDirectory:
    """One policy run directory within a federation cell."""

    policy: PolicyId
    path: Path


class FederationCellResult(BaseModel):
    """Run directories produced for one federation cell."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    experiment_id: ExperimentId
    model_seed: ModelSeed
    calibration_seed: CalibrationSeed
    run_directories: tuple[PolicyRunDirectory, ...]

    def directory_for(self, policy: PolicyId) -> Path:
        for entry in self.run_directories:
            if entry.policy is policy:
                return entry.path
        raise KeyError(policy.value)


class FederationCellMaterializer:
    """Evaluate the federation once, then write every requested policy cell."""

    def __init__(
        self,
        policy_cells: PolicyCellMaterializer | None = None,
        run_experiment: RunExperiment | None = None,
    ) -> None:
        self.policy_cells = policy_cells or PolicyCellMaterializer()
        self.run_experiment = run_experiment or RunExperiment()

    def materialize(
        self,
        experiment_id: ExperimentId,
        config: ExperimentConfig,
        model_seed: ModelSeed,
        calibration_seed: CalibrationSeed,
        caches: FrozenCacheInputs,
        policies: tuple[PolicyId, ...] | None = None,
        assignment_mode: CalibrationAssignmentMode = CalibrationAssignmentMode.SEEDED_PERMUTATION,
    ) -> FederationCellResult:
        selected = config.policies if policies is None else policies
        if not selected:
            raise ValueError("At least one policy must be materialized")
        unknown = set(selected) - set(config.policies)
        if unknown:
            raise ValueError(
                "Requested policy is not configured: "
                + ", ".join(sorted(item.value for item in unknown))
            )

        bundle = self.policy_cells.evaluate_federation(
            config,
            caches,
            calibration_seed,
            assignment_mode,
        )
        run_dirs: list[PolicyRunDirectory] = []
        for policy in selected:

            def materialize_policy(
                _plan: ExperimentPlan, run_layout: RunLayout, policy: PolicyId = policy
            ) -> FederationMetrics | None:
                return self.policy_cells.materialize_analysis(
                    config,
                    policy,
                    run_layout,
                    caches,
                    calibration_seed,
                    bundle,
                    assignment_mode,
                )

            _, layout = self.run_experiment.execute(
                experiment_id=experiment_id,
                config=config,
                model_seed=model_seed,
                calibration_seed=calibration_seed,
                policy=policy,
                runner=materialize_policy,
                repository_root=Path("."),
            )
            run_dirs.append(PolicyRunDirectory(policy, layout.root))
        return FederationCellResult(
            experiment_id=experiment_id,
            model_seed=model_seed,
            calibration_seed=calibration_seed,
            run_directories=tuple(run_dirs),
        )


class FrozenModelEvidence(BaseModel):
    """Frozen model and score cache evidence for one seed."""

    model_config = Frozen

    model_seed: ModelSeed
    model_path: Path
    training_manifest: Path
    score_root: Path


class WorkloadExecution(BaseModel):
    """Executed model evidence and run directories for one experiment."""

    model_config = Frozen

    experiment_id: ExperimentId
    models: tuple[FrozenModelEvidence, ...]
    run_directories: tuple[Path, ...]


class ResearchExecution(BaseModel):
    """Preflight report and executed workload."""

    model_config = Frozen

    preflight: PreflightReport
    workload: WorkloadExecution


class RunAllExperiments:
    """Execute the single spine from prepared data to completed policy runs."""

    def __init__(
        self,
        trainer: TrainDetector | None = None,
        scorer: ComputeScores | None = None,
        federation_cells: FederationCellMaterializer | None = None,
    ) -> None:
        self.trainer = trainer or TrainDetector()
        self.scorer = scorer or ComputeScores()
        self.federation_cells = federation_cells or FederationCellMaterializer()

    def execute(
        self,
        experiment_id: ExperimentId,
        config: ExperimentConfig,
        prepared_root: Path,
        *,
        calibration_seeds: tuple[CalibrationSeed, ...] | None = None,
    ) -> WorkloadExecution:
        seed_values = calibration_seeds or tuple(
            int(seed) for seed in config.dataset.calibration_seeds
        )
        if not seed_values:
            raise ValueError("At least one calibration seed is required")
        configured = {int(seed) for seed in config.dataset.calibration_seeds}
        invalid = tuple(seed for seed in seed_values if int(seed) not in configured)
        if invalid:
            raise ValueError(
                f"Calibration seeds are outside the frozen dataset registry: {invalid}"
            )

        model_evidence: list[FrozenModelEvidence] = []
        run_directories: list[Path] = []
        for model_seed_value in config.randomness.model_seeds:
            model_seed = int(model_seed_value)
            model_path, training_manifest = self.trainer.train_from_cache(
                config,
                prepared_root,
                model_seed,
            )
            score_root = self.scorer.score_from_cache(
                config,
                prepared_root,
                model_path,
                model_seed,
                training_manifest,
            )
            model_evidence.append(
                FrozenModelEvidence(
                    model_seed=model_seed,
                    model_path=model_path,
                    training_manifest=training_manifest,
                    score_root=score_root,
                )
            )
            caches = FrozenCacheInputs(
                prepared_root=prepared_root,
                model_path=model_path,
                training_manifest=training_manifest,
                score_root=score_root,
            )
            for calibration_seed in seed_values:
                cell = self.federation_cells.materialize(
                    experiment_id=experiment_id,
                    config=config,
                    model_seed=model_seed,
                    calibration_seed=calibration_seed,
                    caches=caches,
                )
                run_directories.extend(entry.path for entry in cell.run_directories)

        return WorkloadExecution(
            experiment_id=experiment_id,
            models=tuple(model_evidence),
            run_directories=tuple(run_directories),
        )


class CampaignOutcomeRow(BaseModel):
    """One completed or failed campaign experiment outcome."""

    model_config = Frozen

    experiment_id: ExperimentId
    status: ExperimentStatus
    problem: Identifier | None = None
    finished_at: Timestamp | None = None

    @property
    def failed(self) -> bool:
        return self.status is ExperimentStatus.FAILED


class CampaignWorkItem(BaseModel):
    """One ordered campaign work item."""

    model_config = Frozen

    experiment_id: ExperimentId
    config_path: Path
    prepared_root: Path


class CampaignStatus(BaseModel):
    """Persistent restart-safe campaign status snapshot."""

    model_config = ConfigDict(frozen=True)

    campaign_id: CampaignId
    created_at: Timestamp
    updated_at: Timestamp
    current_experiment: ExperimentId | None = None
    current_stage: CampaignStage | None = None
    experiments: tuple[CampaignOutcomeRow, ...] = ()
    results_path: PathString | None = None
    elapsed_seconds: Duration = 0.0

    @property
    def completed_experiments(self) -> tuple[ExperimentId, ...]:
        return tuple(
            item.experiment_id
            for item in self.experiments
            if item.status is ExperimentStatus.COMPLETE
        )

    @property
    def pending_experiments(self) -> tuple[ExperimentId, ...]:
        return tuple(
            item.experiment_id
            for item in self.experiments
            if item.status is ExperimentStatus.PENDING
        )

    @property
    def failed_experiments(self) -> tuple[ExperimentId, ...]:
        return tuple(
            item.experiment_id
            for item in self.experiments
            if item.status is ExperimentStatus.FAILED
        )


class CampaignStatusStore:
    """Persist campaign status as an atomically written JSON snapshot."""

    def __init__(
        self,
        campaigns_root: Path | None = None,
        outputs_root: Path = Path("outputs"),
    ) -> None:
        self.campaigns_root = campaigns_root or OutputsLayout(outputs_root).campaigns

    def path_for(self, campaign_id: CampaignId) -> Path:
        return campaign_status_path(self.campaigns_root, campaign_id)

    def save(self, status: CampaignStatus) -> Path:
        path = self.path_for(status.campaign_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, status.model_dump(mode="json"))
        return path

    def load(self, campaign_id: CampaignId) -> CampaignStatus:
        path = self.path_for(campaign_id)
        if not path.is_file():
            raise FileNotFoundError(f"Campaign has no recorded status: {campaign_id}")
        return CampaignStatus.model_validate_json(path.read_text(encoding="utf-8"))


class CampaignExecutor:
    """Execute ordered work items with persistent restart-safe status."""

    def __init__(
        self,
        study: Study | None = None,
        status_store: CampaignStatusStore | None = None,
        runner: RunAllExperiments | None = None,
    ) -> None:
        self.study = study or Study.load()
        self.status_store = status_store or CampaignStatusStore()
        self.runner = runner or RunAllExperiments()

    def run(
        self,
        campaign_id: CampaignId,
        work_items: tuple[CampaignWorkItem, ...],
        *,
        outputs_root: Path,
        results_root: Path | None = None,
    ) -> CampaignStatus:
        from datetime import UTC, datetime

        now = datetime.now(UTC).isoformat()
        try:
            status = self.status_store.load(campaign_id)
        except FileNotFoundError:
            status = CampaignStatus(campaign_id=campaign_id, created_at=now, updated_at=now)

        completed: set[ExperimentId] = set(status.completed_experiments)
        rows: list[CampaignOutcomeRow] = list(status.experiments)
        started = time.monotonic()
        for item in work_items:
            if item.experiment_id in completed:
                continue
            spec = self.study.spec(item.experiment_id)
            config = self.study.resolve(item.experiment_id)
            prepared_root = item.prepared_root
            status = CampaignStatus(
                campaign_id=campaign_id,
                created_at=now,
                updated_at=datetime.now(UTC).isoformat(),
                current_experiment=item.experiment_id,
                current_stage=CampaignStage.RUNNING,
                experiments=tuple(rows),
                elapsed_seconds=time.monotonic() - started,
            )
            self.status_store.save(status)
            try:
                ProtocolTablePrecomputer().precompute(config, spec)
                if spec.category in _ANALYSIS_CATEGORIES:
                    output = OutputsLayout(outputs_root).analysis_result(item.experiment_id)
                    if item.experiment_id is ExperimentId.COMPUTATIONAL_BENCHMARK:
                        RunBenchmark(spec, config).run(output)
                    else:
                        RunSyntheticExperiments().run(item.experiment_id, spec, config, output)
                else:
                    self.runner.execute(
                        item.experiment_id,
                        config,
                        prepared_root,
                    )
                rows.append(_completed_row(item.experiment_id, now))
            except Exception as exc:
                rows.append(_failed_row(item.experiment_id, str(exc), now))
        results_path: Identifier | None = None
        if results_root is not None:
            results_path = self._build_results(campaign_id, outputs_root, results_root)
        final_status = CampaignStatus(
            campaign_id=campaign_id,
            created_at=now,
            updated_at=datetime.now(UTC).isoformat(),
            current_experiment=None,
            current_stage=(
                CampaignStage.DONE if not any(r.failed for r in rows) else CampaignStage.FAILED
            ),
            experiments=tuple(rows),
            results_path=results_path,
            elapsed_seconds=time.monotonic() - started,
        )
        self.status_store.save(final_status)
        return final_status

    @staticmethod
    def _build_results(
        campaign_id: CampaignId,
        outputs_root: Path,
        results_root: Path,
    ) -> Identifier:
        from fedcrg.reporting import build_results_bundle

        destination = campaign_results_root(results_root, campaign_id)
        if destination.exists():
            return destination.as_posix()
        return build_results_bundle(campaign_id, outputs_root, results_root).as_posix()


def _completed_row(experiment_id: ExperimentId, finished_at: Timestamp) -> CampaignOutcomeRow:
    return CampaignOutcomeRow(
        experiment_id=experiment_id,
        status=ExperimentStatus.COMPLETE,
        finished_at=finished_at,
    )


def _failed_row(
    experiment_id: ExperimentId, problem: Identifier, finished_at: Timestamp
) -> CampaignOutcomeRow:
    return CampaignOutcomeRow(
        experiment_id=experiment_id,
        status=ExperimentStatus.FAILED,
        problem=problem,
        finished_at=finished_at,
    )
