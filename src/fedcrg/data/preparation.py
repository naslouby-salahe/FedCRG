from __future__ import annotations

import hashlib
import os
import shutil
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import pandas as pd

from fedcrg.config import DatasetConfig, ExperimentConfig
from fedcrg.data.datasets import (
    ClientData,
    ClientEligibilityEvaluator,
    DatasetAdapter,
    EligibilityManifest,
    EligibilityRecord,
    hash_row_ids,
)
from fedcrg.data.diad import DiadAdapter
from fedcrg.data.nbaiot import NBaiotAdapter
from fedcrg.data.preprocessing import (
    ClientPreprocessingStatistics,
    PreprocessingModel,
    TrainOnlyPreprocessing,
)
from fedcrg.data.splits import (
    BaseSplitBuilder,
    CalibrationAssignmentBuilder,
    ClientSplits,
    RoleFrame,
)
from fedcrg.evidence.models import (
    CalibrationAssignmentManifest,
    CalibrationAssignmentReference,
    CalibrationRoleManifest,
    ClientCalibrationManifest,
    ClientDatasetManifest,
    PreparedDatasetManifest,
    RoleArtifactManifest,
    SourceFileManifest,
)
from fedcrg.evidence.store import (
    CalibrationAssignmentManifestStore,
    PreparedDatasetManifestStore,
    atomic_write_json,
    sha256_file,
)
from fedcrg.paths import PreparedDatasetLayout, prepared_dataset_root
from fedcrg.runtime import get_logger
from fedcrg.types import (
    CalibrationAssignmentMode,
    CalibrationSeed,
    ClientId,
    DataIntegrityError,
    DataRole,
    DatasetId,
    EligibilityStatus,
    FailureCode,
    FeatureName,
    PreparedColumn,
    Sha256,
)

_LOGGER = get_logger(__name__)

_BASE_ROLES = (
    DataRole.TRAIN,
    DataRole.RESERVOIR,
    DataRole.BENIGN_TEST,
    DataRole.ATTACK_DEV,
    DataRole.ATTACK_TEST,
)
_CALIBRATION_ROLES = (
    DataRole.REFERENCE,
    DataRole.MISMATCH,
    DataRole.CALIBRATION,
    DataRole.BENIGN_GUARD,
)


class PrepareData:
    def __init__(
        self,
        splitter: CalibrationAssignmentBuilder | None = None,
        base_split_builder: BaseSplitBuilder | None = None,
        preprocessor: TrainOnlyPreprocessing | None = None,
        eligibility: ClientEligibilityEvaluator | None = None,
        manifests: PreparedDatasetManifestStore | None = None,
        calibration_assignment_manifests: CalibrationAssignmentManifestStore | None = None,
    ) -> None:
        self.splitter = splitter or CalibrationAssignmentBuilder()
        self.base_split_builder = base_split_builder or BaseSplitBuilder()
        self.preprocessor = preprocessor or TrainOnlyPreprocessing()
        self.eligibility = eligibility or ClientEligibilityEvaluator()
        self.manifests = manifests or PreparedDatasetManifestStore()
        self.calibration_assignment_manifests = (
            calibration_assignment_manifests or CalibrationAssignmentManifestStore()
        )

    @staticmethod
    def adapter(dataset: DatasetConfig, root: Path) -> DatasetAdapter:
        if dataset.id is DatasetId.NBAIOT:
            return NBaiotAdapter(root, dataset)
        if dataset.id is DatasetId.DIAD:
            return DiadAdapter(root, dataset)
        raise ValueError(f"No filesystem adapter for {dataset.id}")

    @staticmethod
    def _source_identity_hash(sources: tuple[SourceFileManifest, ...]) -> Sha256:
        payload = "\n".join(
            f"{item.relative_path.as_posix()}:{item.sha256}" for item in sorted(sources, key=str)
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def prepared_root(self, config: ExperimentConfig, source_identity_hash: Sha256) -> Path:
        return prepared_dataset_root(
            config.preprocessed_root,
            config.dataset.id,
            config.data_spec_hash,
            source_identity_hash,
        )

    def cache_root(self, config: ExperimentConfig, manifest: PreparedDatasetManifest) -> Path:
        return self.prepared_root(config, self._source_identity_hash(manifest.source_files))

    def ensure_prepared(
        self,
        config: ExperimentConfig,
        data_root: Path,
        adapter_override: DatasetAdapter | None = None,
        include_source_order_assignment: bool = True,
    ) -> PreparedDatasetManifest:
        adapter = adapter_override or self.adapter(
            config.dataset,
            data_root / config.dataset.source_directory,
        )
        if adapter.dataset_id is not config.dataset.id:
            raise ValueError("Adapter dataset identity does not match experiment config")

        sources = tuple(
            self._source_file_manifest(path, adapter.root) for path in adapter.source_files()
        )
        source_identity_hash = self._source_identity_hash(sources)
        final_root = self.prepared_root(config, source_identity_hash)

        if final_root.exists():
            try:
                return self._reuse_existing(
                    final_root, config, sources, data_root / config.dataset.source_directory
                )
            except DataIntegrityError as exc:
                # A cache that fails verification (hash mismatch, missing artifact) is untrusted;
                # rebuild from raw sources rather than risk serving stale or tampered splits.
                _LOGGER.warning("prepared cache invalid (%s); rebuilding %s", exc, final_root)
                shutil.rmtree(final_root, ignore_errors=True)

        return self._materialize(
            config,
            adapter,
            sources,
            include_source_order_assignment,
            final_root,
        )

    def _reuse_existing(
        self,
        final_root: Path,
        config: ExperimentConfig,
        sources: tuple[SourceFileManifest, ...],
        source_root: Path,
    ) -> PreparedDatasetManifest:
        layout = PreparedDatasetLayout(final_root)

        if not layout.manifest.is_file() or not layout.preprocessing.is_file():
            raise DataIntegrityError("Prepared cache is missing manifest or preprocessing evidence")

        manifest = self.manifests.load_model(layout.manifest)

        if manifest.data_spec_hash != config.data_spec_hash:
            raise DataIntegrityError("Prepared cache data-spec hash differs from configuration")
        if manifest.dataset_id is not config.dataset.id:
            raise DataIntegrityError("Prepared cache dataset differs from configuration")
        if manifest.source_files != sources:
            raise DataIntegrityError("Prepared cache source identity differs from the raw data")

        self._verify_source_files(sources, source_root)
        self._verify_role_artifacts(manifest, final_root)
        self._verify_calibration_assignments(manifest, final_root)

        _LOGGER.info("prepared cache reused %s", final_root)
        return manifest

    @staticmethod
    def _verify_source_files(sources: tuple[SourceFileManifest, ...], source_root: Path) -> None:
        for item in sources:
            source_path = source_root / item.relative_path
            if not source_path.is_file():
                raise DataIntegrityError(
                    f"Raw source file is missing for the prepared cache: {item.relative_path}"
                )
            if sha256_file(source_path) != item.sha256:
                raise DataIntegrityError(
                    f"Raw source file changed since preparation: {item.relative_path}"
                )

    @staticmethod
    def _verify_role_artifacts(manifest: PreparedDatasetManifest, final_root: Path) -> None:
        for client in manifest.clients:
            for role in client.roles:
                artifact = final_root / role.relative_path
                if not artifact.is_file():
                    raise DataIntegrityError(
                        f"Prepared role artifact is missing: {role.relative_path}"
                    )
                if sha256_file(artifact) != role.file_sha256:
                    raise DataIntegrityError(
                        f"Prepared role artifact hash mismatch: {role.relative_path}"
                    )

    @staticmethod
    def _verify_calibration_assignments(
        manifest: PreparedDatasetManifest, final_root: Path
    ) -> None:
        for reference in manifest.calibration_assignments:
            assignment = final_root / reference.relative_path
            if not assignment.is_file():
                raise DataIntegrityError(
                    f"Calibration-assignment manifest is missing: {reference.relative_path}"
                )
            if sha256_file(assignment) != reference.sha256:
                raise DataIntegrityError(
                    f"Calibration-assignment manifest hash changed: {reference.relative_path}"
                )

    def _materialize(
        self,
        config: ExperimentConfig,
        adapter: DatasetAdapter,
        sources: tuple[SourceFileManifest, ...],
        include_source_order_assignment: bool,
        final_root: Path,
    ) -> PreparedDatasetManifest:
        discovered = adapter.discover_clients()
        self._validate_source_identity_count(config, discovered)

        final_root.parent.mkdir(parents=True, exist_ok=True)
        # Build in a uniquely named staging directory and publish via a single atomic rename so a
        # crash mid-materialization can never leave a partially written cache at final_root.
        staging_root = final_root.parent / f".{final_root.name}.staging-{uuid.uuid4().hex}"
        staging_root.mkdir()
        staging_layout = PreparedDatasetLayout(staging_root)

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
            self._write_eligibility(staging_layout, config, eligibility_manifest)

            if not eligible_ids:
                self._write_dataset_manifest(staging_layout, config, sources, (), (), ())
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

                atomic_write_json(staging_layout.preprocessing, preprocessing)
                self._write_dataset_manifest(
                    staging_layout,
                    config,
                    sources,
                    clients,
                    preprocessing.feature_columns,
                    assignments,
                )

            shutil.rmtree(staging_layout.raw_staging, ignore_errors=True)
            os.replace(staging_root, final_root)
        except Exception:
            shutil.rmtree(staging_root, ignore_errors=True)
            raise

        manifest = self.manifests.load_model(PreparedDatasetLayout(final_root).manifest)
        _LOGGER.info("prepared cache materialized %s", final_root)
        return manifest

    @staticmethod
    def _source_file_manifest(path: Path, root: Path) -> SourceFileManifest:
        return SourceFileManifest(
            relative_path=PurePosixPath(path.relative_to(root).as_posix()),
            sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
        )

    def _validate_source_identity_count(
        self, config: ExperimentConfig, discovered: tuple[ClientId, ...]
    ) -> None:
        if config.dataset.id is DatasetId.NBAIOT:
            expected_devices = (device.client_id for device in config.dataset.device_directories)
            if sorted(discovered) != sorted(expected_devices):
                raise DataIntegrityError(
                    f"{FailureCode.DATASET_COUNT_MISMATCH}: expected nine N-BaIoT devices"
                )
        elif config.dataset.id is DatasetId.DIAD:
            expected_count = config.dataset.expected_source_clients
            if expected_count is not None and len(discovered) != expected_count:
                raise DataIntegrityError(
                    f"{FailureCode.DIAD_DEVICE_COUNT_SOURCE_MISMATCH}: "
                    f"expected {expected_count} DIAD devices, found {len(discovered)}"
                )

    def _stage_clients(
        self,
        root: Path,
        config: ExperimentConfig,
        adapter: DatasetAdapter,
        discovered: tuple[ClientId, ...],
    ) -> tuple[tuple[EligibilityRecord, ...], dict[ClientId, ClientPreprocessingStatistics]]:
        records: list[EligibilityRecord] = []
        statistics: dict[ClientId, ClientPreprocessingStatistics] = {}
        total = len(discovered)

        for index, client_id in enumerate(discovered, start=1):
            started = time.monotonic()
            _LOGGER.info("staging client %d/%d %s", index, total, client_id)

            data = adapter.load_client(client_id)
            if data.client_id != client_id:
                raise DataIntegrityError(
                    f"{FailureCode.NONDETERMINISTIC_PARITY_FAIL}: "
                    "adapter returned a client identity that does not match the request"
                )

            if config.dataset.id is DatasetId.NBAIOT:
                self._validate_nbaiot_count(config, data)

            record = self.eligibility.evaluate(data, config.dataset)
            records.append(record)

            if record.status is not EligibilityStatus.ELIGIBLE:
                _LOGGER.info(
                    "client %s excluded (%s) in %.1fs",
                    client_id,
                    record.status,
                    time.monotonic() - started,
                )
                continue

            splits = self.base_split_builder.build(
                data, config.dataset, config.randomness.attack_split_seed
            )
            statistics[client_id] = self.preprocessor.client_statistics(
                splits,
                config.dataset.id,
                config.dataset.feature_count,
                finite_rate_minimum=config.dataset.diad_finite_rate_minimum,
            )
            self._write_raw_splits(root, splits)
            _LOGGER.info("client %s staged in %.1fs", client_id, time.monotonic() - started)

        return tuple(records), statistics

    @staticmethod
    def _validate_nbaiot_count(config: ExperimentConfig, data: ClientData) -> None:
        """N-BaIoT has no fallback eligibility path, so an undersized benign pool must fail fast instead of silently truncating a role."""
        required_size = (
            config.dataset.split.reservoir_size
            + config.dataset.split.train_benign
            + config.dataset.split.min_benign_test
        )
        if len(data.benign) < required_size:
            raise DataIntegrityError(
                f"{FailureCode.NBAIOT_ATTACK_BUDGET_FAIL}: benign evidence is insufficient"
            )

    @staticmethod
    def _write_raw_splits(root: Path, splits: ClientSplits) -> None:
        client_root = PreparedDatasetLayout(root).raw_staging / splits.client_id
        client_root.mkdir(parents=True, exist_ok=True)
        for item in splits.roles:
            item.frame.to_parquet(client_root / f"{item.role}.parquet", index=False)

    @staticmethod
    def _load_raw_splits(root: Path, client_id: ClientId) -> ClientSplits:
        client_root = PreparedDatasetLayout(root).raw_staging / client_id
        roles = tuple(
            RoleFrame(role=role, frame=pd.read_parquet(client_root / f"{role}.parquet"))
            for role in _BASE_ROLES
        )
        return ClientSplits(client_id=client_id, roles=roles)

    def _finalize_staged_clients(
        self,
        root: Path,
        config: ExperimentConfig,
        eligible_ids: tuple[ClientId, ...],
        preprocessing: PreprocessingModel,
        include_source_order_assignment: bool,
    ) -> tuple[tuple[ClientDatasetManifest, ...], tuple[CalibrationAssignmentReference, ...]]:
        client_manifests: list[ClientDatasetManifest] = []
        parsed_seeds = tuple(int(seed) for seed in config.dataset.calibration_seeds)
        seeded_assignments: dict[CalibrationSeed, list[ClientCalibrationManifest]] = {
            seed: [] for seed in parsed_seeds
        }
        source_order_clients: list[ClientCalibrationManifest] = []

        for client_id in sorted(eligible_ids):
            splits = self._load_raw_splits(root, client_id)
            client_manifests.append(self._write_client_roles(root, splits, preprocessing))

            for seed in parsed_seeds:
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
                        int(config.dataset.primary_calibration_seed),
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
        client_root = root / str(splits.client_id)
        client_root.mkdir(parents=True, exist_ok=True)
        roles: list[RoleArtifactManifest] = []

        for item in splits.roles:
            frame = item.frame
            # Only TRAIN is scaled here; other roles are persisted raw and must be transformed by
            # callers via the same PreprocessingModel, since calibration/test scaling is
            # deliberately left unclipped and must not retroactively affect the fitted extrema.
            if item.role is DataRole.TRAIN:
                frame = preprocessing.transform(frame, splits.client_id)

            relative = f"{splits.client_id}/{item.role}.csv"
            path = root / relative
            frame.to_csv(path, index=False)

            roles.append(
                RoleArtifactManifest(
                    role=item.role,
                    rows=len(frame),
                    row_id_sha256=hash_row_ids(frame[PreparedColumn.ROW_ID].astype(str).tolist()),
                    relative_path=PurePosixPath(relative),
                    file_sha256=sha256_file(path),
                )
            )

        return ClientDatasetManifest(client_id=splits.client_id, roles=tuple(roles))

    def _client_assignment_manifest(
        self,
        config: ExperimentConfig,
        splits: ClientSplits,
        seed: CalibrationSeed,
        mode: CalibrationAssignmentMode,
    ) -> ClientCalibrationManifest:
        reservoir = splits.get(DataRole.RESERVOIR)
        assignment = self.splitter.build(
            reservoir,
            config.dataset.id,
            splits.client_id,
            config.dataset,
            seed,
            mode,
        )
        roles = tuple(
            CalibrationRoleManifest(
                role=role,
                row_count=len(assignment.positions_for(role)),
                row_id_sha256=assignment.row_id_hash_for(role),
            )
            for role in _CALIBRATION_ROLES
        )
        return ClientCalibrationManifest(client_id=splits.client_id, roles=roles)

    def _write_assignment_manifests(
        self,
        root: Path,
        config: ExperimentConfig,
        seeded: dict[CalibrationSeed, list[ClientCalibrationManifest]],
        source_order: list[ClientCalibrationManifest],
    ) -> tuple[CalibrationAssignmentReference, ...]:
        layout = PreparedDatasetLayout(root)
        references: list[CalibrationAssignmentReference] = []

        for seed, clients in sorted(seeded.items()):
            manifest = CalibrationAssignmentManifest(
                calibration_seed=seed,
                mode=CalibrationAssignmentMode.SEEDED_PERMUTATION,
                clients=tuple(clients),
            )
            path = layout.seeded_splits / f"c{seed}.json"
            relative = PurePosixPath(path.relative_to(root).as_posix())
            self.calibration_assignment_manifests.save(path, manifest)
            references.append(
                CalibrationAssignmentReference(
                    calibration_seed=seed,
                    mode=CalibrationAssignmentMode.SEEDED_PERMUTATION,
                    relative_path=relative,
                    sha256=sha256_file(path),
                )
            )

        if source_order:
            seed = int(config.dataset.primary_calibration_seed)
            manifest = CalibrationAssignmentManifest(
                calibration_seed=seed,
                mode=CalibrationAssignmentMode.SOURCE_ORDER,
                clients=tuple(source_order),
            )
            path = layout.source_order_split
            relative = PurePosixPath(path.relative_to(root).as_posix())
            self.calibration_assignment_manifests.save(path, manifest)
            references.append(
                CalibrationAssignmentReference(
                    calibration_seed=seed,
                    mode=CalibrationAssignmentMode.SOURCE_ORDER,
                    relative_path=relative,
                    sha256=sha256_file(path),
                )
            )

        return tuple(references)

    def _write_eligibility(
        self,
        layout: PreparedDatasetLayout,
        config: ExperimentConfig,
        manifest: EligibilityManifest,
    ) -> None:
        path = (
            layout.diad_eligibility if config.dataset.id is DatasetId.DIAD else layout.eligibility
        )
        atomic_write_json(path, manifest)

    def _write_dataset_manifest(
        self,
        layout: PreparedDatasetLayout,
        config: ExperimentConfig,
        sources: tuple[SourceFileManifest, ...],
        clients: tuple[ClientDatasetManifest, ...],
        feature_names: tuple[FeatureName, ...],
        assignments: tuple[CalibrationAssignmentReference, ...],
    ) -> None:
        manifest = PreparedDatasetManifest(
            dataset_id=config.dataset.id,
            source_version=config.dataset.source_version,
            parser_version=config.dataset.parser_version,
            data_spec_hash=config.data_spec_hash,
            feature_names=feature_names,
            clients=clients,
            source_files=sources,
            calibration_assignments=assignments,
            external_replication_supported=config.dataset.id is DatasetId.DIAD,
            created_at=datetime.now(UTC),
        )
        self.manifests.save(layout.manifest, manifest)
