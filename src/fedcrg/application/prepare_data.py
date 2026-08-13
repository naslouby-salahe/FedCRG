"""Freeze eligibility, base partitions, preprocessing, and calibration assignments."""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path, PurePosixPath

import pandas as pd

from fedcrg.artifacts.dataset import PreparedDatasetManifestStore
from fedcrg.artifacts.hashing import sha256_file
from fedcrg.artifacts.serialization import atomic_write_json
from fedcrg.config.models import ExperimentConfig
from fedcrg.core.enums import (
    CalibrationAssignmentMode,
    DataRole,
    DatasetId,
    EligibilityStatus,
    FailureCode,
)
from fedcrg.core.exceptions import DataIntegrityError
from fedcrg.core.ids import CalibrationSeed, ClientId, Sha256
from fedcrg.data.adapter import DatasetAdapter
from fedcrg.data.datasets.diad import DiadAdapter
from fedcrg.data.datasets.nbaiot import NBaiotAdapter
from fedcrg.data.eligibility import ClientEligibilityEvaluator
from fedcrg.data.manifests import (
    CalibrationAssignmentManifest,
    CalibrationAssignmentReference,
    CalibrationRoleManifest,
    ClientCalibrationManifest,
    ClientDatasetManifest,
    EligibilityManifest,
    RoleArtifactManifest,
    SourceFileManifest,
    hash_row_ids,
    source_file_manifest,
)
from fedcrg.data.models import ClientData, ClientSplits, EligibilityRecord, RoleFrame
from fedcrg.data.preprocessing import (
    ClientPreprocessingStatistics,
    FederatedPreprocessor,
    PreprocessingModel,
)
from fedcrg.data.splitting import DataSplitter


class PrepareData:
    """Materialize one immutable, seed-independent cache per data specification."""

    def __init__(
        self,
        splitter: DataSplitter | None = None,
        preprocessor: FederatedPreprocessor | None = None,
        eligibility: ClientEligibilityEvaluator | None = None,
        manifests: PreparedDatasetManifestStore | None = None,
    ) -> None:
        self.splitter = splitter or DataSplitter()
        self.preprocessor = preprocessor or FederatedPreprocessor()
        self.eligibility = eligibility or ClientEligibilityEvaluator()
        self.manifests = manifests or PreparedDatasetManifestStore()

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

        final_root = (
            config.outputs_root
            / "cache"
            / "datasets"
            / config.dataset.id.value
            / config.data_spec_hash[:16]
        )
        if final_root.exists():
            raise FileExistsError(
                f"Prepared dataset cache already exists and is immutable: {final_root}"
            )
        final_root.parent.mkdir(parents=True, exist_ok=True)
        staging_root = final_root.parent / (f".{final_root.name}.staging-{uuid.uuid4().hex}")
        staging_root.mkdir()

        try:
            eligibility_records, statistics = self._stage_clients(
                staging_root,
                config,
                adapter,
                discovered,
            )
            eligible_ids = tuple(
                record.client_id
                for record in eligibility_records
                if record.status is EligibilityStatus.ELIGIBLE
            )
            eligibility_manifest = EligibilityManifest(
                dataset_id=config.dataset.id,
                discovered_clients=tuple(discovered),
                eligible_clients=eligible_ids,
                records=eligibility_records,
            )
            self._write_eligibility(staging_root, config, eligibility_manifest)

            if not eligible_ids:
                self._write_dataset_manifest(
                    staging_root,
                    config,
                    sources,
                    (),
                    (),
                    (),
                )
            else:
                preprocessing = self.preprocessor.aggregate(
                    tuple(statistics[client_id] for client_id in eligible_ids),
                    config.dataset.id,
                )
                clients, assignments = self._finalize_staged_clients(
                    staging_root,
                    config,
                    eligible_ids,
                    preprocessing,
                    include_source_order_assignment,
                )
                atomic_write_json(
                    staging_root / "preprocessing.json",
                    preprocessing.to_dict(),
                )
                self._write_dataset_manifest(
                    staging_root,
                    config,
                    sources,
                    clients,
                    preprocessing.feature_columns,
                    assignments,
                )
            shutil.rmtree(staging_root / "_raw", ignore_errors=True)
            os.replace(staging_root, final_root)
        except Exception:
            shutil.rmtree(staging_root, ignore_errors=True)
            raise
        return final_root

    def _stage_clients(
        self,
        root: Path,
        config: ExperimentConfig,
        adapter: DatasetAdapter,
        discovered: tuple[ClientId, ...],
    ) -> tuple[
        tuple[EligibilityRecord, ...],
        dict[ClientId, ClientPreprocessingStatistics],
    ]:
        records: list[EligibilityRecord] = []
        statistics: dict[ClientId, ClientPreprocessingStatistics] = {}
        for client_id in discovered:
            data = adapter.load_client(client_id)
            if data.client_id != client_id:
                raise DataIntegrityError(
                    f"{FailureCode.NONDETERMINISTIC_PARITY_FAIL.value}: "
                    "adapter returned a client identity that does not match the request"
                )
            if config.dataset.id is DatasetId.NBAIOT:
                self._validate_nbaiot_count(config, data)
            record = self.eligibility.evaluate(data, config.dataset)
            records.append(record)
            if record.status is not EligibilityStatus.ELIGIBLE:
                continue
            splits = self.splitter.split_base(data, config.dataset)
            statistics[client_id] = self.preprocessor.client_statistics(
                splits,
                config.dataset.id,
                config.dataset.feature_count,
            )
            self._write_raw_splits(root, splits)
        return tuple(records), statistics

    @staticmethod
    def _write_raw_splits(root: Path, splits: ClientSplits) -> None:
        client_root = root / "_raw" / splits.client_id.value
        client_root.mkdir(parents=True, exist_ok=True)
        for role, frame in ((item.role, item.frame) for item in splits.roles):
            frame.to_parquet(client_root / f"{role.value}.parquet", index=False)

    @staticmethod
    def _load_raw_splits(root: Path, client_id: ClientId) -> ClientSplits:
        client_root = root / "_raw" / client_id.value
        roles = tuple(
            RoleFrame(role, pd.read_parquet(client_root / f"{role.value}.parquet"))
            for role in (
                DataRole.TRAIN,
                DataRole.RESERVOIR,
                DataRole.BENIGN_TEST,
                DataRole.ATTACK_DEV,
                DataRole.ATTACK_TEST,
            )
        )
        return ClientSplits(client_id, roles)

    def _finalize_staged_clients(
        self,
        root: Path,
        config: ExperimentConfig,
        eligible_ids: tuple[ClientId, ...],
        preprocessing: PreprocessingModel,
        include_source_order_assignment: bool,
    ) -> tuple[
        tuple[ClientDatasetManifest, ...],
        tuple[CalibrationAssignmentReference, ...],
    ]:
        client_manifests: list[ClientDatasetManifest] = []
        seeded_assignments: dict[CalibrationSeed, list[ClientCalibrationManifest]] = {
            CalibrationSeed(seed): [] for seed in config.dataset.calibration_seeds
        }
        source_order_clients: list[ClientCalibrationManifest] = []

        for client_id in sorted(eligible_ids):
            splits = self._load_raw_splits(root, client_id)
            client_manifests.append(self._write_client_roles(root, splits, preprocessing))
            for seed_value in config.dataset.calibration_seeds:
                seed = CalibrationSeed(seed_value)
                seeded_assignments[seed].append(
                    self._client_assignment_manifest(
                        config,
                        splits,
                        seed,
                        CalibrationAssignmentMode.SEEDED_PERMUTATION,
                    )
                )
            if include_source_order_assignment:
                source_order_clients.append(
                    self._client_assignment_manifest(
                        config,
                        splits,
                        CalibrationSeed(config.dataset.primary_calibration_seed),
                        CalibrationAssignmentMode.SOURCE_ORDER,
                    )
                )

        references = self._write_assignment_manifests(
            root,
            config,
            seeded_assignments,
            source_order_clients,
        )
        return tuple(client_manifests), references

    def _write_client_roles(
        self,
        root: Path,
        splits: ClientSplits,
        preprocessing: PreprocessingModel,
    ) -> ClientDatasetManifest:
        client_root = root / "clients" / splits.client_id.value
        client_root.mkdir(parents=True)
        roles: list[RoleArtifactManifest] = []
        for role, raw_frame in ((item.role, item.frame) for item in splits.roles):
            frame = preprocessing.transform(raw_frame, splits.client_id)
            path = client_root / f"{role.value}.csv.gz"
            temp = path.with_name(f".{path.name}.tmp")
            frame.to_csv(
                temp,
                index=False,
                compression={"method": "gzip", "mtime": 0},
            )
            os.replace(temp, path)
            roles.append(
                RoleArtifactManifest(
                    role=role,
                    rows=len(frame),
                    row_id_sha256=hash_row_ids(frame["row_id"].astype(str).tolist()),
                    relative_path=PurePosixPath(path.relative_to(root).as_posix()),
                    file_sha256=Sha256(sha256_file(path)),
                )
            )
        return ClientDatasetManifest(
            client_id=splits.client_id,
            roles=tuple(sorted(roles, key=lambda item: item.role.value)),
        )

    def _client_assignment_manifest(
        self,
        config: ExperimentConfig,
        splits: ClientSplits,
        seed: CalibrationSeed,
        mode: CalibrationAssignmentMode,
    ) -> ClientCalibrationManifest:
        assignment = self.splitter.calibration_assignment(
            splits,
            config.dataset.id,
            config.dataset,
            seed,
            mode,
        )
        roles = tuple(
            CalibrationRoleManifest(
                role=item.role,
                row_count=len(item.positions),
                row_id_sha256=item.row_id_hash,
            )
            for item in sorted(assignment.roles, key=lambda item: item.role.value)
        )
        return ClientCalibrationManifest(splits.client_id, roles)

    def _write_assignment_manifests(
        self,
        root: Path,
        config: ExperimentConfig,
        seeded: dict[CalibrationSeed, list[ClientCalibrationManifest]],
        source_order: list[ClientCalibrationManifest],
    ) -> tuple[CalibrationAssignmentReference, ...]:
        references: list[CalibrationAssignmentReference] = []
        seeded_root = root / "splits" / "seeded"
        seeded_root.mkdir(parents=True)
        for seed_value in config.dataset.calibration_seeds:
            seed = CalibrationSeed(seed_value)
            manifest = CalibrationAssignmentManifest(
                seed,
                CalibrationAssignmentMode.SEEDED_PERMUTATION,
                tuple(seeded[seed]),
            )
            path = seeded_root / f"c{int(seed)}.json"
            atomic_write_json(path, manifest)
            references.append(
                CalibrationAssignmentReference(
                    seed,
                    CalibrationAssignmentMode.SEEDED_PERMUTATION,
                    Sha256(sha256_file(path)),
                )
            )
        if source_order:
            seed = CalibrationSeed(config.dataset.primary_calibration_seed)
            manifest = CalibrationAssignmentManifest(
                seed,
                CalibrationAssignmentMode.SOURCE_ORDER,
                tuple(source_order),
            )
            path = root / "splits" / "source_order.json"
            atomic_write_json(path, manifest)
            references.append(
                CalibrationAssignmentReference(
                    seed,
                    CalibrationAssignmentMode.SOURCE_ORDER,
                    Sha256(sha256_file(path)),
                )
            )
        return tuple(references)

    @staticmethod
    def _write_eligibility(
        root: Path,
        config: ExperimentConfig,
        manifest: EligibilityManifest,
    ) -> None:
        name = (
            "diad_eligibility.json" if config.dataset.id is DatasetId.DIAD else "eligibility.json"
        )
        atomic_write_json(root / name, manifest)

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
                    f"{len(discovered)} source identities, expected {expected}"
                )
        elif (
            config.dataset.expected_clients is not None
            and len(discovered) != config.dataset.expected_clients
        ):
            raise DataIntegrityError(
                f"{FailureCode.DATASET_COUNT_MISMATCH.value}: found {len(discovered)} "
                f"clients, expected {config.dataset.expected_clients}"
            )

    @staticmethod
    def _validate_nbaiot_count(config: ExperimentConfig, data: ClientData) -> None:
        expected = config.dataset.expected_benign_counts.get(data.client_id.value)
        if expected is None or len(data.benign) != expected:
            raise DataIntegrityError(
                f"{FailureCode.DATASET_COUNT_MISMATCH.value}: {data.client_id} benign "
                f"count {len(data.benign)} != expected {expected}"
            )

    def _write_dataset_manifest(
        self,
        root: Path,
        config: ExperimentConfig,
        source_manifests: tuple[SourceFileManifest, ...],
        clients: tuple[ClientDatasetManifest, ...],
        feature_columns: tuple[str, ...],
        assignments: tuple[CalibrationAssignmentReference, ...],
    ) -> None:
        external_supported = (
            True
            if config.dataset.id is not DatasetId.DIAD
            else len(clients) >= config.dataset.minimum_clients
        )
        code = None if external_supported else FailureCode.EXTERNAL_DATASET_INSUFFICIENT_CLIENTS
        manifest = self.manifests.build(
            dataset_id=config.dataset.id,
            source_version=config.dataset.source_version,
            parser_version=config.dataset.parser_version,
            data_spec_hash=Sha256(config.data_spec_hash),
            feature_names=feature_columns,
            clients=clients,
            source_files=source_manifests,
            calibration_assignments=assignments,
            external_replication_supported=external_supported,
            dataset_level_code=code,
        )
        self.manifests.save(root / "manifest.json", manifest)
