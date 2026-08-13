"""Generate one hash-finalized immutable score cache per frozen model seed."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from fedcrg.experiments.model_training import TrainDetector, feature_columns
from fedcrg.artifacts.manifests import PreparedDatasetManifestStore
from fedcrg.artifacts.integrity import sha256_file
from fedcrg.artifacts.manifests import TrainingManifestStore
from fedcrg.configuration.experiment_config import ExperimentConfig
from fedcrg.domain.enums import DataRole
from fedcrg.domain.identifiers import AttackGroupId, ModelSeed, RowId, Sha256
from fedcrg.runtime.logging import get_logger
from fedcrg.datasets.prepare import PreparedDatasetManifest
from fedcrg.detectors.detector import DetectorModel
from fedcrg.scoring.cache import ScoreCache, ScoreCacheIdentity
from fedcrg.scoring.compute import ScoreComputer
from fedcrg.scoring.calibration_scores import RoleScores

_LOGGER = get_logger(__name__)

_BASE_SCORE_ROLES = (
    DataRole.TRAIN,
    DataRole.RESERVOIR,
    DataRole.BENIGN_TEST,
    DataRole.ATTACK_DEV,
    DataRole.ATTACK_TEST,
)


class ComputeScores:
    """Score seed-independent base roles using bounded memory and frozen provenance."""

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
        seed = ModelSeed(int(model_seed))
        manifest_path = prepared_root / "manifest.json"
        preprocessing_path = prepared_root / "preprocessing.json"
        prepared_manifest = self.dataset_manifests.load(manifest_path)
        if prepared_manifest.data_spec_hash != Sha256(config.data_spec_hash):
            raise ValueError("Prepared dataset data-spec hash does not match scoring config")
        if prepared_manifest.dataset_id is not config.dataset.id:
            raise ValueError("Prepared dataset identity does not match scoring config")
        prepared_manifest_hash = Sha256(sha256_file(manifest_path))
        preprocessing_hash = Sha256(sha256_file(preprocessing_path))

        training = self.training_manifests.load(training_manifest)
        if training.training_spec_hash != Sha256(config.training_spec_hash):
            raise ValueError("Frozen model belongs to another training specification")
        if training.data_spec_hash != Sha256(config.data_spec_hash):
            raise ValueError("Frozen model belongs to another data specification")
        if training.model_seed != seed:
            raise ValueError("Training manifest model seed does not match scoring request")
        if training.model_file_sha256 != Sha256(sha256_file(model_path)):
            raise ValueError("Frozen model file hash does not match training manifest")
        if training.dataset_manifest_sha256 != prepared_manifest_hash:
            raise ValueError("Training manifest does not reference this prepared dataset manifest")
        if training.preprocessing_sha256 != preprocessing_hash:
            raise ValueError("Training manifest does not reference this preprocessing artifact")

        model = self.trainer.load_model(config, model_path)
        model_hash = Sha256(model.state_hash())
        if model_hash != training.result.final_model_hash:
            raise ValueError("Frozen model state does not match the training result hash")
        identity = ScoreCacheIdentity(
            dataset=config.dataset.id,
            model_seed=seed,
            model_hash=model_hash,
            data_spec_hash=Sha256(config.data_spec_hash),
            training_spec_hash=Sha256(config.training_spec_hash),
            dataset_manifest_hash=prepared_manifest_hash,
            preprocessing_hash=preprocessing_hash,
        )
        score_root = (
            config.outputs_root
            / "cache"
            / "scores"
            / config.dataset.id.value
            / config.detector.id.value
            / f"m{int(seed)}"
            / config.training_spec_hash[:16]
        )
        if score_root.exists():
            self._validate_existing(identity, score_root)
            return score_root

        self.cache.save_stream(
            identity,
            self._score_roles(config, prepared_root, prepared_manifest, model),
            score_root,
        )
        return score_root

    def _score_roles(
        self,
        config: ExperimentConfig,
        prepared_root: Path,
        prepared_manifest: PreparedDatasetManifest,
        model: DetectorModel,
    ) -> Iterator[RoleScores]:
        total_clients = len(prepared_manifest.clients)
        for client_index, client_manifest in enumerate(prepared_manifest.clients, start=1):
            client_id = client_manifest.client_id
            _LOGGER.info(
                "scoring client %d/%d %s (%d roles)",
                client_index,
                total_clients,
                client_id.value,
                len(_BASE_SCORE_ROLES),
            )
            for role in _BASE_SCORE_ROLES:
                role_manifest = client_manifest.role(role)
                path = prepared_root / Path(role_manifest.relative_path)
                if not path.is_file():
                    raise FileNotFoundError(f"Missing prepared base role: {path}")
                if Sha256(sha256_file(path)) != role_manifest.file_sha256:
                    raise ValueError(f"Prepared role hash mismatch: {client_id.value}/{role.value}")
                frame = pd.read_csv(path)
                columns = feature_columns(frame, config.dataset.feature_count)
                if tuple(columns) != prepared_manifest.feature_names:
                    raise ValueError("Score feature order differs from frozen dataset manifest")
                scores = self.computer.compute(
                    model,
                    frame[columns].to_numpy(dtype="float32"),
                    config.training.device,
                )
                groups = None
                if role in {DataRole.ATTACK_DEV, DataRole.ATTACK_TEST}:
                    if "attack_group" not in frame.columns:
                        raise ValueError(f"Attack role {role.value} is missing attack_group")
                    groups = tuple(
                        AttackGroupId(str(value)) for value in frame["attack_group"].astype(str)
                    )
                yield RoleScores(
                    role=role,
                    values=scores,
                    client_id=client_id,
                    row_ids=tuple(RowId(str(value)) for value in frame["row_id"].astype(str)),
                    attack_groups=groups,
                )
                del frame, scores

    def _validate_existing(
        self,
        expected: ScoreCacheIdentity,
        score_root: Path,
    ) -> None:
        descriptor = self.cache.load_descriptor(score_root)
        if descriptor.identity != expected:
            raise ValueError(
                "Existing score cache provenance does not exactly match the requested "
                "dataset, preprocessing, training specification, seed, and frozen model"
            )
