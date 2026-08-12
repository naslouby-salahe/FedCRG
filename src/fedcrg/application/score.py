"""Score-cache generation from prepared role data and a frozen model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fedcrg.application.train import TrainDetector, feature_columns
from fedcrg.config.models import ExperimentConfig
from fedcrg.core.enums import DataRole
from fedcrg.scoring.cache import ScoreCache
from fedcrg.scoring.computer import ScoreComputer


class ComputeScores:
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
        model_seed: int,
    ) -> Path:
        prepared_manifest = json.loads((prepared_root / "manifest.json").read_text(encoding="utf-8"))
        role_values: dict[str, dict[DataRole, np.ndarray]] = {}
        role_groups: dict[str, dict[DataRole, tuple[str, ...]]] = {}
        for client_id in sorted(prepared_manifest["clients"]):
            roles: dict[DataRole, np.ndarray] = {}
            groups: dict[DataRole, tuple[str, ...]] = {}
            for role in DataRole:
                path = prepared_root / client_id / f"{role.value}.csv.gz"
                if not path.exists():
                    continue
                frame = pd.read_csv(path)
                columns = feature_columns(frame, config.dataset.feature_count)
                roles[role] = frame[columns].to_numpy()
                if role in {DataRole.ATTACK_DEV, DataRole.ATTACK_TEST} and "attack_group" in frame.columns:
                    groups[role] = tuple(frame["attack_group"].astype(str))
            role_values[client_id] = roles
            role_groups[client_id] = groups
        model = self.trainer.load_model(config, model_path)
        manifest = self.computer.compute_manifest(
            model=model,
            dataset=config.dataset.id,
            model_seed=model_seed,
            role_values=role_values,
            role_groups=role_groups,
            device=config.training.device,
        )
        score_root = (
            config.outputs_root
            / "cache"
            / "scores"
            / config.dataset.id.value
            / config.detector.id.value
            / f"m{model_seed}"
            / config.config_hash[:16]
        )
        self.cache.save(manifest, score_root)
        return score_root
