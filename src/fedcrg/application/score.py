"""Generate one hash-finalized immutable score cache per frozen model seed."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fedcrg.application.train import TrainDetector, feature_columns
from fedcrg.artifacts.hashing import sha256_file
from fedcrg.config.models import ExperimentConfig
from fedcrg.core.enums import DataRole
from fedcrg.core.ids import ClientId, Sha256
from fedcrg.scoring.cache import ScoreCache
from fedcrg.scoring.computer import ScoreComputer
from fedcrg.scoring.models import ClientScoreInput, RoleScoreInput

_BASE_SCORE_ROLES = (
    DataRole.TRAIN,
    DataRole.RESERVOIR,
    DataRole.BENIGN_TEST,
    DataRole.ATTACK_DEV,
    DataRole.ATTACK_TEST,
)


class ComputeScores:
    """Score only seed-independent base roles; calibration roles are later views."""

    def __init__(self) -> None:
        self.computer = ScoreComputer()
        self.trainer = TrainDetector()
        self.cache = ScoreCache()

    def score_from_cache(
        self,
        config: ExperimentConfig,
        prepared_root: Path,
        model_path: Path,
        model_seed: int,
        training_manifest: Path | None = None,
    ) -> Path:
        manifest_path = prepared_root / "manifest.json"
        preprocessing_path = prepared_root / "preprocessing.json"
        prepared_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prepared_manifest.get("data_spec_hash") != config.data_spec_hash:
            raise ValueError("Prepared dataset data-spec hash does not match scoring config")

        if training_manifest is not None:
            training = json.loads(training_manifest.read_text(encoding="utf-8"))
            if training.get("training_spec_hash") != config.training_spec_hash:
                raise ValueError("Frozen model belongs to another training specification")
            if training.get("model_file_sha256") != sha256_file(model_path):
                raise ValueError("Frozen model hash does not match training manifest")

        clients: list[ClientScoreInput] = []
        for client_value in sorted(prepared_manifest["clients"]):
            client_id = ClientId(client_value)
            role_inputs: dict[DataRole, RoleScoreInput] = {}
            for role in _BASE_SCORE_ROLES:
                path = prepared_root / "clients" / client_value / f"{role.value}.csv.gz"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing prepared base role: {path}")
                frame = pd.read_csv(path)
                columns = feature_columns(frame, config.dataset.feature_count)
                groups = None
                if role in {DataRole.ATTACK_DEV, DataRole.ATTACK_TEST}:
                    if "attack_group" not in frame.columns:
                        raise ValueError(f"Attack role {role.value} is missing attack_group")
                    groups = tuple(frame["attack_group"].astype(str))
                role_inputs[role] = RoleScoreInput(
                    role=role,
                    values=frame[columns].to_numpy(dtype="float32"),
                    row_ids=tuple(frame["row_id"].astype(str)),
                    attack_groups=groups,
                )
            clients.append(ClientScoreInput(client_id, role_inputs))

        model = self.trainer.load_model(config, model_path)
        score_manifest = self.computer.compute_manifest(
            model=model,
            dataset=config.dataset.id,
            model_seed=model_seed,
            data_spec_hash=Sha256(config.data_spec_hash),
            training_spec_hash=Sha256(config.training_spec_hash),
            dataset_manifest_hash=Sha256(sha256_file(manifest_path)),
            preprocessing_hash=Sha256(sha256_file(preprocessing_path)),
            clients=tuple(clients),
            device=config.training.device.value,
        )
        score_root = (
            config.outputs_root
            / "cache"
            / "scores"
            / config.dataset.id.value
            / config.detector.id.value
            / f"m{model_seed}"
            / config.training_spec_hash[:16]
        )
        if score_root.exists():
            loaded = self.cache.load(score_root)
            self._validate_existing(config, model_seed, model_path, loaded)
            return score_root
        finalized = self.cache.save(score_manifest, score_root)
        if finalized.cache_sha256 is None:
            raise RuntimeError("Score cache was not hash-finalized")
        return score_root

    @staticmethod
    def _validate_existing(
        config: ExperimentConfig,
        model_seed: int,
        model_path: Path,
        manifest: object,
    ) -> None:
        from fedcrg.scoring.models import ScoreManifest

        if not isinstance(manifest, ScoreManifest):
            raise TypeError("Unexpected score-cache manifest")
        if manifest.model_seed != model_seed:
            raise ValueError("Existing score cache has a different model seed")
        if manifest.data_spec_hash != Sha256(config.data_spec_hash):
            raise ValueError("Existing score cache belongs to another data specification")
        if manifest.training_spec_hash != Sha256(config.training_spec_hash):
            raise ValueError("Existing score cache belongs to another training specification")
        if manifest.model_hash != Sha256(TrainDetector().load_model(config, model_path).state_hash()):
            raise ValueError("Existing score cache belongs to another frozen model")
