"""Generate one hash-finalized immutable score cache per frozen model seed."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from fedcrg.application.train import TrainDetector, feature_columns
from fedcrg.artifacts.hashing import sha256_file
from fedcrg.config.models import ExperimentConfig
from fedcrg.core.enums import DataRole
from fedcrg.core.ids import AttackGroupId, ClientId, ModelSeed, RowId, Sha256
from fedcrg.detectors.base import DetectorModel
from fedcrg.scoring.cache import ScoreCache, ScoreCacheIdentity
from fedcrg.scoring.computer import ScoreComputer
from fedcrg.scoring.models import RoleScores

_BASE_SCORE_ROLES = (
    DataRole.TRAIN,
    DataRole.RESERVOIR,
    DataRole.BENIGN_TEST,
    DataRole.ATTACK_DEV,
    DataRole.ATTACK_TEST,
)


class ComputeScores:
    """Score seed-independent base roles using bounded memory.

    A training manifest is mandatory: scoring an arbitrary model file without proving
    its training specification would break the frozen-detector provenance contract.
    """

    def __init__(
        self,
        computer: ScoreComputer | None = None,
        trainer: TrainDetector | None = None,
        cache: ScoreCache | None = None,
    ) -> None:
        self.computer = computer or ScoreComputer()
        self.trainer = trainer or TrainDetector()
        self.cache = cache or ScoreCache()

    def score_from_cache(
        self,
        config: ExperimentConfig,
        prepared_root: Path,
        model_path: Path,
        model_seed: ModelSeed | int,
        training_manifest: Path,
    ) -> Path:
        seed = ModelSeed(int(model_seed))
        manifest_path = prepared_root / "manifest.json"
        preprocessing_path = prepared_root / "preprocessing.json"
        prepared_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prepared_manifest.get("data_spec_hash") != config.data_spec_hash:
            raise ValueError("Prepared dataset data-spec hash does not match scoring config")

        training = json.loads(training_manifest.read_text(encoding="utf-8"))
        if training.get("training_spec_hash") != config.training_spec_hash:
            raise ValueError("Frozen model belongs to another training specification")
        if int(training.get("model_seed", -1)) != int(seed):
            raise ValueError("Training manifest model seed does not match scoring request")
        if training.get("model_file_sha256") != sha256_file(model_path):
            raise ValueError("Frozen model file hash does not match training manifest")
        if training.get("dataset_manifest_sha256") != sha256_file(manifest_path):
            raise ValueError("Training manifest does not reference this prepared dataset manifest")
        if training.get("preprocessing_sha256") != sha256_file(preprocessing_path):
            raise ValueError("Training manifest does not reference this preprocessing artifact")

        model = self.trainer.load_model(config, model_path)
        model_hash = Sha256(model.state_hash())
        identity = ScoreCacheIdentity(
            dataset=config.dataset.id,
            model_seed=seed,
            model_hash=model_hash,
            data_spec_hash=Sha256(config.data_spec_hash),
            training_spec_hash=Sha256(config.training_spec_hash),
            dataset_manifest_hash=Sha256(sha256_file(manifest_path)),
            preprocessing_hash=Sha256(sha256_file(preprocessing_path)),
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
            self._validate_existing(config, seed, model_hash, score_root)
            return score_root

        descriptor = self.cache.save_stream(
            identity,
            self._score_roles(config, prepared_root, prepared_manifest, model),
            score_root,
        )
        if not descriptor.cache_sha256.value:
            raise RuntimeError("Score cache was not hash-finalized")
        return score_root

    def _score_roles(
        self,
        config: ExperimentConfig,
        prepared_root: Path,
        prepared_manifest: dict[str, object],
        model: DetectorModel,
    ) -> Iterator[RoleScores]:
        clients = prepared_manifest.get("clients")
        if not isinstance(clients, dict):
            raise ValueError("Prepared dataset manifest has no client ledger")
        for client_value in sorted(clients):
            client_id = ClientId(str(client_value))
            for role in _BASE_SCORE_ROLES:
                path = prepared_root / "clients" / client_id.value / f"{role.value}.csv.gz"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing prepared base role: {path}")
                frame = pd.read_csv(path)
                columns = feature_columns(frame, config.dataset.feature_count)
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
                        AttackGroupId(str(value))
                        for value in frame["attack_group"].astype(str)
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
        config: ExperimentConfig,
        model_seed: ModelSeed,
        model_hash: Sha256,
        score_root: Path,
    ) -> None:
        descriptor = self.cache.load_descriptor(score_root)
        identity = descriptor.identity
        if identity.model_seed != model_seed:
            raise ValueError("Existing score cache has a different model seed")
        if identity.data_spec_hash != Sha256(config.data_spec_hash):
            raise ValueError("Existing score cache belongs to another data specification")
        if identity.training_spec_hash != Sha256(config.training_spec_hash):
            raise ValueError("Existing score cache belongs to another training specification")
        if identity.model_hash != model_hash:
            raise ValueError("Existing score cache belongs to another frozen model")
