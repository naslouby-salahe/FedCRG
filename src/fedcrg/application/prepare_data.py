"""Dataset preparation, eligibility freeze, preprocessing, and split caching."""

from __future__ import annotations

import json
import os
from pathlib import Path

from fedcrg.config.models import ExperimentConfig
from fedcrg.core.enums import DatasetId
from fedcrg.core.exceptions import DataIntegrityError
from fedcrg.data.adapter import DatasetAdapter
from fedcrg.data.datasets.diad import DiadAdapter
from fedcrg.data.datasets.nbaiot import NBaiotAdapter
from fedcrg.data.manifests import hash_row_ids
from fedcrg.data.models import ClientSplits
from fedcrg.data.preprocessing import FederatedPreprocessor
from fedcrg.data.splitting import DataSplitter


class PrepareData:
    def __init__(
        self,
        splitter: DataSplitter | None = None,
        preprocessor: FederatedPreprocessor | None = None,
    ) -> None:
        self.splitter = splitter or DataSplitter()
        self.preprocessor = preprocessor or FederatedPreprocessor()

    def adapter(self, dataset: DatasetId, root: Path) -> DatasetAdapter:
        if dataset is DatasetId.NBAIOT:
            return NBaiotAdapter(root)
        if dataset is DatasetId.DIAD:
            return DiadAdapter(root)
        raise ValueError(f"No filesystem adapter for {dataset.value}")

    def prepare(
        self,
        config: ExperimentConfig,
        data_root: Path,
        calibration_seed: int | None = None,
    ) -> Path:
        seed = calibration_seed if calibration_seed is not None else config.dataset.primary_calibration_seed
        if seed not in config.dataset.calibration_seeds:
            raise ValueError(f"Calibration seed {seed} is not configured")
        adapter = self.adapter(config.dataset.id, data_root)
        discovered = adapter.discover_clients()
        if config.dataset.expected_clients is not None and len(discovered) != config.dataset.expected_clients:
            raise DataIntegrityError(
                f"DATASET_COUNT_MISMATCH: found {len(discovered)} clients, "
                f"expected {config.dataset.expected_clients}"
            )

        splits_by_client: dict[str, ClientSplits] = {}
        exclusions: dict[str, str] = {}
        for client_id in discovered:
            try:
                splits = self.splitter.split(adapter.load_client(client_id), config.dataset, seed)
                self.preprocessor.validate_training_rows(
                    splits, config.dataset.id, config.dataset.feature_count
                )
                splits_by_client[client_id] = splits
            except DataIntegrityError as exc:
                if config.dataset.id is DatasetId.NBAIOT:
                    raise
                exclusions[client_id] = str(exc).split(":", 1)[0]
        if len(splits_by_client) < config.dataset.minimum_clients:
            raise DataIntegrityError(
                f"EXTERNAL_DATASET_INSUFFICIENT_CLIENTS: {len(splits_by_client)} eligible; "
                f"{config.dataset.minimum_clients} required"
            )

        preprocessing = self.preprocessor.fit(
            splits_by_client, config.dataset.id, config.dataset.feature_count
        )
        cache_root = (
            config.outputs_root
            / "cache"
            / "datasets"
            / config.dataset.id.value
            / f"c{seed}"
            / config.config_hash[:16]
        )
        cache_root.mkdir(parents=True, exist_ok=True)
        client_manifests: dict[str, dict[str, object]] = {}
        manifest: dict[str, object] = {
            "dataset": config.dataset.id.value,
            "calibration_seed": seed,
            "config_hash": config.config_hash,
            "feature_columns": list(preprocessing.feature_columns),
            "clients": client_manifests,
        }
        for client_id in sorted(splits_by_client):
            client_root = cache_root / client_id
            client_root.mkdir(parents=True, exist_ok=True)
            role_meta: dict[str, object] = {}
            for role, raw_frame in splits_by_client[client_id].roles.items():
                frame = preprocessing.transform(raw_frame, client_id)
                path = client_root / f"{role.value}.csv.gz"
                temp = path.with_suffix(".csv.gz.tmp")
                frame.to_csv(temp, index=False, compression="gzip")
                os.replace(temp, path)
                role_meta[role.value] = {
                    "rows": len(frame),
                    "row_id_hash": hash_row_ids(frame["row_id"].astype(str).tolist()),
                    "file": str(path.relative_to(cache_root)),
                }
            client_manifests[client_id] = role_meta
        self._write_json(cache_root / "preprocessing.json", preprocessing.to_dict())
        self._write_json(
            cache_root / "eligibility.json",
            {
                "discovered_clients": list(discovered),
                "eligible_clients": sorted(splits_by_client),
                "exclusions": exclusions,
            },
        )
        self._write_json(cache_root / "manifest.json", manifest)
        return cache_root

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temp, path)
