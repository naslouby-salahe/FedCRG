"""Freeze dataset eligibility, base partitions, preprocessing, and calibration assignments."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fedcrg.artifacts.hashing import sha256_file
from fedcrg.config.models import ExperimentConfig
from fedcrg.core.enums import CalibrationAssignmentMode, DatasetId, EligibilityStatus, FailureCode
from fedcrg.core.exceptions import DataIntegrityError
from fedcrg.core.ids import ClientId
from fedcrg.data.adapter import DatasetAdapter
from fedcrg.data.datasets.diad import DiadAdapter
from fedcrg.data.datasets.nbaiot import NBaiotAdapter
from fedcrg.data.eligibility import ClientEligibilityEvaluator
from fedcrg.data.manifests import (
    CalibrationAssignmentManifest,
    CalibrationRoleManifest,
    ClientCalibrationManifest,
    EligibilityManifest,
    hash_row_ids,
    source_file_manifest,
)
from fedcrg.data.models import ClientData, ClientSplits, EligibilityRecord
from fedcrg.data.preprocessing import FederatedPreprocessor
from fedcrg.data.splitting import DataSplitter


class PrepareData:
    """Materialize one seed-independent prepared-data cache per data specification."""

    def __init__(
        self,
        splitter: DataSplitter | None = None,
        preprocessor: FederatedPreprocessor | None = None,
        eligibility: ClientEligibilityEvaluator | None = None,
    ) -> None:
        self.splitter = splitter or DataSplitter()
        self.preprocessor = preprocessor or FederatedPreprocessor()
        self.eligibility = eligibility or ClientEligibilityEvaluator()

    @staticmethod
    def adapter(dataset: DatasetId, root: Path) -> DatasetAdapter:
        if dataset is DatasetId.NBAIOT:
            return NBaiotAdapter(root)
        if dataset is DatasetId.DIAD:
            return DiadAdapter(root)
        raise ValueError(f"No filesystem adapter for {dataset.value}")

    def prepare(
        self,
        config: ExperimentConfig,
        data_root: Path,
        adapter_override: DatasetAdapter | None = None,
        include_source_order_assignment: bool = True,
    ) -> Path:
        adapter = adapter_override or self.adapter(config.dataset.id, data_root)
        if adapter.dataset_id is not config.dataset.id:
            raise ValueError("Adapter dataset identity does not match experiment config")

        discovered = adapter.discover_clients()
        self._validate_source_identity_count(config, discovered)
        sources = tuple(source_file_manifest(path, adapter.root) for path in adapter.source_files())

        loaded: dict[ClientId, ClientData] = {}
        eligibility_records: list[EligibilityRecord] = []
        for client_id in discovered:
            data = adapter.load_client(client_id)
            loaded[client_id] = data
            if config.dataset.id is DatasetId.NBAIOT:
                self._validate_nbaiot_count(config, data)
            eligibility_records.append(self.eligibility.evaluate(data, config.dataset))

        eligible_ids = tuple(
            record.client_id
            for record in eligibility_records
            if record.status is EligibilityStatus.ELIGIBLE
        )
        eligibility_manifest = EligibilityManifest(
            dataset_id=config.dataset.id,
            discovered_clients=tuple(discovered),
            eligible_clients=eligible_ids,
            records=tuple(eligibility_records),
        )

        base_splits: dict[ClientId, ClientSplits] = {
            client_id: self.splitter.split_base(loaded[client_id], config.dataset)
            for client_id in eligible_ids
        }
        for splits in base_splits.values():
            self.preprocessor.validate_training_rows(
                splits,
                config.dataset.id,
                config.dataset.feature_count,
            )

        cache_root = (
            config.outputs_root
            / "cache"
            / "datasets"
            / config.dataset.id.value
            / config.data_spec_hash[:16]
        )
        if cache_root.exists():
            raise FileExistsError(f"Prepared dataset cache already exists and is immutable: {cache_root}")
        cache_root.mkdir(parents=True)
        (cache_root / "splits" / "seeded").mkdir(parents=True)

        eligibility_name = (
            "diad_eligibility.json"
            if config.dataset.id is DatasetId.DIAD
            else "eligibility.json"
        )
        self._write_json(cache_root / eligibility_name, eligibility_manifest.to_dict())

        if not base_splits:
            self._write_dataset_manifest(
                cache_root,
                config,
                sources,
                {},
                (),
                {},
            )
            return cache_root

        preprocessing = self.preprocessor.fit(
            base_splits,
            config.dataset.id,
            config.dataset.feature_count,
        )
        client_manifest: dict[str, dict[str, object]] = {}
        for client_id in sorted(base_splits):
            client_root = cache_root / "clients" / client_id.value
            client_root.mkdir(parents=True)
            role_metadata: dict[str, object] = {}
            for role, raw_frame in base_splits[client_id].roles.items():
                frame = preprocessing.transform(raw_frame, client_id)
                path = client_root / f"{role.value}.csv.gz"
                temp = path.with_name(f".{path.name}.tmp")
                frame.to_csv(temp, index=False, compression="gzip")
                os.replace(temp, path)
                role_metadata[role.value] = {
                    "rows": len(frame),
                    "row_id_hash": hash_row_ids(frame["row_id"].astype(str).tolist()),
                    "file": path.relative_to(cache_root).as_posix(),
                    "sha256": sha256_file(path),
                }
            client_manifest[client_id.value] = role_metadata

        self._write_json(cache_root / "preprocessing.json", preprocessing.to_dict())
        split_manifests = self._write_calibration_assignments(
            cache_root,
            config,
            base_splits,
            include_source_order_assignment,
        )
        self._write_dataset_manifest(
            cache_root,
            config,
            sources,
            client_manifest,
            preprocessing.feature_columns,
            split_manifests,
        )
        return cache_root

    def _write_calibration_assignments(
        self,
        cache_root: Path,
        config: ExperimentConfig,
        base_splits: dict[ClientId, ClientSplits],
        include_source_order: bool,
    ) -> dict[str, str]:
        manifests: dict[str, str] = {}
        for seed in config.dataset.calibration_seeds:
            manifest = self._assignment_manifest(
                config,
                base_splits,
                seed,
                CalibrationAssignmentMode.SEEDED_PERMUTATION,
            )
            path = cache_root / "splits" / "seeded" / f"c{seed}.json"
            self._write_json(path, manifest.to_dict())
            manifests[f"seeded:c{seed}"] = sha256_file(path)

        if include_source_order:
            manifest = self._assignment_manifest(
                config,
                base_splits,
                config.dataset.primary_calibration_seed,
                CalibrationAssignmentMode.SOURCE_ORDER,
            )
            path = cache_root / "splits" / "source_order.json"
            self._write_json(path, manifest.to_dict())
            manifests["source_order"] = sha256_file(path)
        return manifests

    def _assignment_manifest(
        self,
        config: ExperimentConfig,
        base_splits: dict[ClientId, ClientSplits],
        seed: int,
        mode: CalibrationAssignmentMode,
    ) -> CalibrationAssignmentManifest:
        clients: list[ClientCalibrationManifest] = []
        for client_id in sorted(base_splits):
            assignment = self.splitter.calibration_assignment(
                base_splits[client_id],
                config.dataset.id,
                config.dataset,
                seed,
                mode,
            )
            roles = {
                role: CalibrationRoleManifest(
                    row_count=len(assignment.positions_for(role)),
                    row_id_sha256=assignment.row_id_hashes[role],
                )
                for role in assignment.positions
            }
            clients.append(ClientCalibrationManifest(client_id, roles))
        return CalibrationAssignmentManifest(seed, mode.value, tuple(clients))

    @staticmethod
    def _validate_source_identity_count(
        config: ExperimentConfig,
        discovered: tuple[ClientId, ...],
    ) -> None:
        if config.dataset.id is DatasetId.DIAD:
            expected = config.dataset.expected_source_clients
            if expected is not None and len(discovered) != expected:
                raise DataIntegrityError(
                    f"{FailureCode.DIAD_DEVICE_COUNT_SOURCE_MISMATCH.value}: found "
                    f"{len(discovered)} source identities; expected {expected}"
                )
        elif config.dataset.expected_clients is not None and len(discovered) != config.dataset.expected_clients:
            raise DataIntegrityError(
                f"{FailureCode.DATASET_COUNT_MISMATCH.value}: found {len(discovered)} clients; "
                f"expected {config.dataset.expected_clients}"
            )

    @staticmethod
    def _validate_nbaiot_count(config: ExperimentConfig, data: ClientData) -> None:
        expected = config.dataset.expected_benign_counts.get(data.client_id.value)
        if expected is None or len(data.benign) != expected:
            raise DataIntegrityError(
                f"{FailureCode.DATASET_COUNT_MISMATCH.value}: {data.client_id} benign count "
                f"{len(data.benign)} != expected {expected}"
            )

    def _write_dataset_manifest(
        self,
        cache_root: Path,
        config: ExperimentConfig,
        source_manifests: tuple[object, ...],
        clients: dict[str, dict[str, object]],
        feature_columns: tuple[str, ...],
        split_manifests: dict[str, str],
    ) -> None:
        payload = {
            "dataset_id": config.dataset.id.value,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_version": config.dataset.source_version,
            "parser_version": config.dataset.parser_version,
            "data_spec_hash": config.data_spec_hash,
            "feature_names": list(feature_columns),
            "clients": clients,
            "source_files": [
                {
                    "relative_path": item.relative_path,
                    "sha256": item.sha256.value,
                    "size_bytes": item.size_bytes,
                }
                for item in source_manifests
            ],
            "calibration_assignments": dict(sorted(split_manifests.items())),
            "external_replication_supported": (
                True
                if config.dataset.id is not DatasetId.DIAD
                else len(clients) >= config.dataset.minimum_clients
            ),
            "dataset_level_code": (
                None
                if config.dataset.id is not DatasetId.DIAD or len(clients) >= config.dataset.minimum_clients
                else FailureCode.EXTERNAL_DATASET_INSUFFICIENT_CLIENTS.value
            ),
        }
        self._write_json(cache_root / "manifest.json", payload)

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.tmp")
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temp, path)
