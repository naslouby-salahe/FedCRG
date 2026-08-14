"""Frozen train-only preprocessing and immutable prepared-cache materialization.

The pipeline is reuse-first: an existing prepared cache for the same data
specification is validated and returned without rewriting any artifact, and a
valid cache is never reconstructed. Preparation stages eligibility, base
splits, per-client imputation statistics, and calibration assignments under
one immutable staging root, then atomically promotes it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, TypeAdapter

from fedcrg.config import DatasetConfig, ExperimentConfig
from fedcrg.data.datasets import (
    CalibrationAssignmentBuilder,
    ClientData,
    ClientEligibilityEvaluator,
    ClientSplits,
    DatasetAdapter,
    EligibilityManifest,
    EligibilityRecord,
    NBaiotAdapter,
    RoleFrame,
    hash_row_ids,
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
    PreparedLayout,
    atomic_write_json,
    sha256_file,
)
from fedcrg.runtime import get_logger
from fedcrg.types import (
    AttackGroupId,
    CalibrationAssignmentMode,
    CalibrationSeed,
    ClientId,
    DataIntegrityError,
    DataRole,
    DatasetId,
    EligibilityStatus,
    FailureCode,
    FeatureCount,
    FeatureName,
    NonNegativeCount,
    PositiveCount,
    Position,
    PreparedColumn,
    RngSeed,
    Score,
    Sha256,
)

Frozen = ConfigDict(frozen=True)
_LOGGER = get_logger(__name__)
_ATTACK_GROUP_ADAPTER = TypeAdapter(AttackGroupId)

_METADATA = {column.value for column in PreparedColumn}

_BASE_ROLES = (
    DataRole.TRAIN,
    DataRole.RESERVOIR,
    DataRole.BENIGN_TEST,
    DataRole.ATTACK_DEV,
    DataRole.ATTACK_TEST,
)


def model_feature_columns(frame: pd.DataFrame, expected: PositiveCount) -> tuple[FeatureName, ...]:
    """Resolve the model-feature column tuple for one dataset."""
    columns = tuple(column for column in frame.columns if column not in _METADATA)
    if len(columns) != expected:
        raise DataIntegrityError(f"Expected {expected} model features, found {len(columns)}")
    return columns


class ClientPreprocessingParameters(BaseModel):
    """Frozen per-client preprocessing parameterization."""

    model_config = Frozen

    client_id: ClientId
    training_row_sha256: Sha256
    medians: tuple[Score, ...] | None = None


class PreprocessingModel(BaseModel):
    """Frozen train-only preprocessing model."""

    model_config = Frozen

    dataset: DatasetId
    feature_columns: tuple[FeatureName, ...]
    clients: tuple[ClientPreprocessingParameters, ...]
    global_minima: tuple[Score, ...]
    global_maxima: tuple[Score, ...]

    @property
    def constant_features(self) -> tuple[bool, ...]:
        """Per-feature flag of zero global span (M == m), recorded for the audit."""
        return tuple(
            minimum == maximum
            for minimum, maximum in zip(self.global_minima, self.global_maxima, strict=True)
        )

    def parameters_for(self, client_id: ClientId) -> ClientPreprocessingParameters:
        for item in self.clients:
            if item.client_id == client_id:
                return item
        raise KeyError(client_id)

    def transform(self, frame: pd.DataFrame, client_id: ClientId) -> pd.DataFrame:
        values = frame.loc[:, list(self.feature_columns)].to_numpy(dtype=np.float64, copy=True)
        parameters = self.parameters_for(client_id)
        if parameters.medians is not None:
            medians = np.asarray(parameters.medians, dtype=np.float64)
            missing = ~np.isfinite(values)
            if missing.any():
                rows, columns = np.nonzero(missing)
                values[rows, columns] = medians[columns]
        if not np.isfinite(values).all():
            raise DataIntegrityError(
                f"Non-finite values remain after preprocessing for {client_id}"
            )
        minima = np.asarray(self.global_minima, dtype=np.float64)
        maxima = np.asarray(self.global_maxima, dtype=np.float64)
        span = maxima - minima
        scaled = np.zeros_like(values, dtype=np.float64)
        varying = span != 0.0
        scaled[:, varying] = (values[:, varying] - minima[varying]) / span[varying]
        result = frame.copy()
        result.loc[:, list(self.feature_columns)] = scaled
        return result


class ClientPreprocessingStatistics(BaseModel):
    """Frozen per-client preprocessing statistics."""

    model_config = Frozen

    client_id: ClientId
    feature_columns: tuple[FeatureName, ...]
    training_row_hash: Sha256
    local_minima: tuple[Score, ...]
    local_maxima: tuple[Score, ...]
    medians: tuple[Score, ...] | None = None


class TrainOnlyPreprocessing:
    """Compute per-client statistics locally, then aggregate only train-set extrema."""

    def validate_training_rows(
        self,
        splits: ClientSplits,
        dataset: DatasetId,
        expected_features: PositiveCount,
    ) -> tuple[FeatureName, ...]:
        train = splits.get(DataRole.TRAIN)
        columns = model_feature_columns(train, expected_features)
        values = train.loc[:, list(columns)].to_numpy(dtype=np.float64)
        if dataset is DatasetId.NBAIOT:
            if not np.isfinite(values).all():
                raise DataIntegrityError("N-BaIoT training features must all be finite")
        elif dataset is DatasetId.DIAD:
            finite_rate = np.isfinite(values).mean(axis=0)
            failing = [columns[index] for index, rate in enumerate(finite_rate) if rate < 0.99]
            if failing:
                raise DataIntegrityError("DIAD_FEATURE_FINITE_RATE_FAIL: " + ", ".join(failing[:5]))
        return columns

    def client_statistics(
        self,
        splits: ClientSplits,
        dataset: DatasetId,
        expected_features: PositiveCount,
    ) -> ClientPreprocessingStatistics:
        columns = self.validate_training_rows(splits, dataset, expected_features)
        train_frame = splits.get(DataRole.TRAIN)
        training_row_hash = hash_row_ids(
            train_frame[PreparedColumn.ROW_ID.value].astype(str).tolist()
        )
        train_values = train_frame.loc[:, list(columns)].to_numpy(dtype=np.float64, copy=True)
        if dataset is DatasetId.DIAD:
            medians = np.nanmedian(
                np.where(np.isfinite(train_values), train_values, np.nan), axis=0
            )
            if not np.isfinite(medians).all():
                raise DataIntegrityError(
                    f"DIAD imputation median is undefined for {splits.client_id}"
                )
            missing = ~np.isfinite(train_values)
            if missing.any():
                rows, indices = np.nonzero(missing)
                train_values[rows, indices] = medians[indices]
            median_tuple: tuple[float, ...] | None = tuple(float(value) for value in medians)
        else:
            median_tuple = None
        return ClientPreprocessingStatistics(
            client_id=splits.client_id,
            feature_columns=columns,
            training_row_hash=training_row_hash,
            local_minima=tuple(float(value) for value in np.min(train_values, axis=0)),
            local_maxima=tuple(float(value) for value in np.max(train_values, axis=0)),
            medians=median_tuple,
        )

    def aggregate(
        self,
        statistics: tuple[ClientPreprocessingStatistics, ...],
        dataset: DatasetId,
    ) -> PreprocessingModel:
        if not statistics:
            raise DataIntegrityError("Cannot fit preprocessing without clients")
        ordered = tuple(sorted(statistics, key=lambda item: item.client_id))
        feature_columns = ordered[0].feature_columns
        for item in ordered[1:]:
            if item.feature_columns != feature_columns:
                raise DataIntegrityError("Model feature order differs across clients")
        global_minima = np.min(np.stack([item.local_minima for item in ordered]), axis=0)
        global_maxima = np.max(np.stack([item.local_maxima for item in ordered]), axis=0)
        return PreprocessingModel(
            dataset=dataset,
            feature_columns=feature_columns,
            clients=tuple(
                ClientPreprocessingParameters(
                    client_id=item.client_id,
                    training_row_sha256=item.training_row_hash,
                    medians=item.medians,
                )
                for item in ordered
            ),
            global_minima=tuple(float(value) for value in global_minima),
            global_maxima=tuple(float(value) for value in global_maxima),
        )


class PrepareData:
    """Materialize one immutable, seed-independent cache per data specification."""

    def __init__(
        self,
        splitter: CalibrationAssignmentBuilder | None = None,
        preprocessor: TrainOnlyPreprocessing | None = None,
        eligibility: ClientEligibilityEvaluator | None = None,
        manifests: PreparedDatasetManifestStore | None = None,
        calibration_assignment_manifests: CalibrationAssignmentManifestStore | None = None,
    ) -> None:
        self.splitter = splitter or CalibrationAssignmentBuilder()
        self.preprocessor = preprocessor or TrainOnlyPreprocessing()
        self.eligibility = eligibility or ClientEligibilityEvaluator()
        self.manifests = manifests or PreparedDatasetManifestStore()
        self.calibration_assignment_manifests = (
            calibration_assignment_manifests or CalibrationAssignmentManifestStore()
        )

    @staticmethod
    def adapter(
        dataset: DatasetId, root: Path, expected_feature_count: FeatureCount
    ) -> DatasetAdapter:
        if dataset is DatasetId.NBAIOT:
            return NBaiotAdapter(root, expected_feature_count)
        raise ValueError(f"No filesystem adapter for {dataset.value}")

    @staticmethod
    def _source_identity_hash(sources: tuple[SourceFileManifest, ...]) -> Sha256:
        """Deterministic identity of the raw source files that produce one cache."""
        payload = "\n".join(
            f"{item.relative_path.as_posix()}:{item.sha256}" for item in sorted(sources, key=str)
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def prepared_root(self, config: ExperimentConfig, source_identity_hash: Sha256) -> Path:
        return (
            config.preprocessed_root
            / config.dataset.id.value
            / f"{config.data_spec_hash[:16]}-{source_identity_hash[:16]}"
        )

    def cache_root(self, config: ExperimentConfig, manifest: PreparedDatasetManifest) -> Path:
        """Resolve the cache directory that holds one prepared manifest."""
        return self.prepared_root(config, self._source_identity_hash(manifest.source_files))

    def ensure_prepared(
        self,
        config: ExperimentConfig,
        data_root: Path,
        adapter_override: DatasetAdapter | None = None,
        include_source_order_assignment: bool = True,
    ) -> PreparedDatasetManifest:
        """Return the frozen prepared cache, reusing a valid existing one.

        The cache identity covers both the data specification and the raw
        source-file identities, so a changed source file produces a new cache
        rather than silently reusing stale artifacts. A cache whose identity
        matches but whose artifacts fail validation is rebuilt.
        """
        adapter = adapter_override or self.adapter(
            config.dataset.id, data_root, config.dataset.feature_count
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
                return self._reuse_existing(final_root, config, sources, data_root)
            except DataIntegrityError as exc:
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
        data_root: Path,
    ) -> PreparedDatasetManifest:
        manifest_path = final_root / PreparedLayout.manifest_filename
        preprocessing_path = final_root / PreparedLayout.preprocessing_filename
        if not manifest_path.is_file() or not preprocessing_path.is_file():
            raise DataIntegrityError("Prepared cache is missing manifest or preprocessing evidence")
        manifest = self.manifests.load_model(manifest_path)
        if manifest.data_spec_hash != config.data_spec_hash:
            raise DataIntegrityError("Prepared cache data-spec hash differs from configuration")
        if manifest.dataset_id is not config.dataset.id:
            raise DataIntegrityError("Prepared cache dataset differs from configuration")
        if manifest.source_files != sources:
            raise DataIntegrityError("Prepared cache source identity differs from the raw data")
        for item in sources:
            source_path = data_root / item.relative_path
            if not source_path.is_file():
                raise DataIntegrityError(
                    f"Raw source file is missing for the prepared cache: {item.relative_path}"
                )
            if sha256_file(source_path) != item.sha256:
                raise DataIntegrityError(
                    f"Raw source file changed since preparation: {item.relative_path}"
                )
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
        _LOGGER.info("prepared cache reused %s", final_root)
        return manifest

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
                self._write_dataset_manifest(staging_root, config, sources, (), (), ())
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
                    staging_root / PreparedLayout.preprocessing_filename, preprocessing
                )
                self._write_dataset_manifest(
                    staging_root,
                    config,
                    sources,
                    clients,
                    preprocessing.feature_columns,
                    assignments,
                )
            shutil.rmtree(staging_root / PreparedLayout.raw_staging_directory, ignore_errors=True)
            os.replace(staging_root, final_root)
        except Exception:
            shutil.rmtree(staging_root, ignore_errors=True)
            raise
        manifest = self.manifests.load_model(final_root / PreparedLayout.manifest_filename)
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
            expected = tuple(f"nb{i:02d}" for i in range(1, 10))
            if tuple(sorted(discovered)) != tuple(sorted(expected)):
                raise DataIntegrityError(
                    f"{FailureCode.DATASET_COUNT_MISMATCH.value}: expected nine N-BaIoT devices"
                )
        elif config.dataset.id is DatasetId.DIAD:
            expected_count = config.dataset.expected_source_clients
            if expected_count is not None and len(discovered) != expected_count:
                raise DataIntegrityError(
                    f"{FailureCode.DIAD_DEVICE_COUNT_SOURCE_MISMATCH.value}: "
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
                    f"{FailureCode.NONDETERMINISTIC_PARITY_FAIL.value}: "
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
                    record.status.value,
                    time.monotonic() - started,
                )
                continue
            splits = self.split_base(data, config.dataset, config.randomness.attack_split_seed)
            statistics[client_id] = self.preprocessor.client_statistics(
                splits,
                config.dataset.id,
                config.dataset.feature_count,
            )
            self._write_raw_splits(root, splits)
            _LOGGER.info("client %s staged in %.1fs", client_id, time.monotonic() - started)
        return tuple(records), statistics

    @staticmethod
    def _validate_nbaiot_count(config: ExperimentConfig, data: ClientData) -> None:
        if (
            len(data.benign)
            < config.dataset.split.reservoir_size
            + config.dataset.split.train_benign
            + config.dataset.split.min_benign_test
        ):
            raise DataIntegrityError(
                f"{FailureCode.NBAIOT_ATTACK_BUDGET_FAIL.value}: benign evidence is insufficient"
            )

    def split_base(
        self,
        data: ClientData,
        config: DatasetConfig,
        attack_split_seed: RngSeed,
    ) -> ClientSplits:
        """Cut deterministic base roles from one client's benign/attack frames."""
        from fedcrg.data.datasets import attack_rng, stable_row_id, validate_split_disjointness

        split = config.split
        benign = data.benign.reset_index(drop=True)
        if len(benign) < split.train_benign + split.reservoir_size + split.min_benign_test:
            raise DataIntegrityError(f"Benign evidence is insufficient for {data.client_id}")
        train = benign.iloc[: split.train_benign].copy()
        reservoir = benign.iloc[
            split.train_benign : split.train_benign + split.reservoir_size
        ].copy()
        benign_test = benign.iloc[
            split.train_benign + split.reservoir_size : split.train_benign
            + split.reservoir_size
            + split.min_benign_test
        ].copy()
        attack = data.attack.reset_index(drop=True)
        if PreparedColumn.ATTACK_GROUP.value not in attack.columns:
            raise DataIntegrityError(f"Attack frame lacks attack_group for {data.client_id}")
        group_values = attack[PreparedColumn.ATTACK_GROUP.value].astype(str)
        groups = tuple(
            sorted(_ATTACK_GROUP_ADAPTER.validate_python(value) for value in set(group_values))
        )
        group_counts: dict[AttackGroupId, NonNegativeCount] = {
            _ATTACK_GROUP_ADAPTER.validate_python(str(group)): int(count)
            for group, count in group_values.value_counts().items()
        }
        development = self._development_allocation(
            data.dataset,
            groups,
            group_counts,
            split.attack_dev,
            split.min_attack_test_per_group,
        )
        dev_rows: list[Position] = []
        for group in groups:
            members = sorted(
                index for index in range(len(attack)) if group_values.iloc[index] == group
            )
            rng = attack_rng(data.dataset, data.client_id, group, attack_split_seed)
            chosen = tuple(
                int(index)
                for index in rng.choice(len(members), size=development[group], replace=False)
            )
            dev_rows.extend(members[index] for index in chosen)
        dev_index = set(dev_rows)
        test_rows = [index for index in range(len(attack)) if index not in dev_index]
        attack_dev = attack.iloc[dev_rows].copy()
        attack_test = attack.iloc[test_rows].copy()

        for frame, role in (
            (train, DataRole.TRAIN),
            (reservoir, DataRole.RESERVOIR),
            (benign_test, DataRole.BENIGN_TEST),
            (attack_dev, DataRole.ATTACK_DEV),
            (attack_test, DataRole.ATTACK_TEST),
        ):
            frame[PreparedColumn.ROLE.value] = role.value
            frame[PreparedColumn.LABEL.value] = (
                0 if role in {DataRole.TRAIN, DataRole.RESERVOIR, DataRole.BENIGN_TEST} else 1
            )
            if PreparedColumn.ROW_ID.value not in frame.columns:
                frame[PreparedColumn.ROW_ID.value] = [
                    stable_row_id(data.dataset, data.client_id, role.value, int(index))
                    for index in range(len(frame))
                ]
        splits = ClientSplits(
            client_id=data.client_id,
            roles=tuple(
                RoleFrame(role=role, frame=frame)
                for role, frame in (
                    (DataRole.TRAIN, train),
                    (DataRole.RESERVOIR, reservoir),
                    (DataRole.BENIGN_TEST, benign_test),
                    (DataRole.ATTACK_DEV, attack_dev),
                    (DataRole.ATTACK_TEST, attack_test),
                )
            ),
        )
        validate_split_disjointness(splits)
        return splits

    @staticmethod
    def _development_allocation(
        dataset: DatasetId,
        groups: tuple[AttackGroupId, ...],
        group_counts: dict[AttackGroupId, NonNegativeCount],
        development_budget: PositiveCount,
        reserve_per_group: PositiveCount,
    ) -> dict[AttackGroupId, NonNegativeCount]:
        """Per-group development counts under the dataset's locked allocation rule."""
        if dataset is DatasetId.DIAD:
            capacities = {
                group: max(0, int(count) - min(reserve_per_group, int(count)))
                for group, count in group_counts.items()
            }
            return PrepareData.waterfill_allocation(groups, capacities, development_budget)
        return PrepareData._nbaiot_balanced_allocation(
            groups, group_counts, development_budget, reserve_per_group
        )

    @staticmethod
    def _nbaiot_balanced_allocation(
        groups: tuple[AttackGroupId, ...],
        group_counts: dict[AttackGroupId, NonNegativeCount],
        development_budget: PositiveCount,
        reserve_per_group: PositiveCount,
    ) -> dict[AttackGroupId, NonNegativeCount]:
        """Equal floor allocation with the remainder distributed lexicographically.

        Every present attack group receives ``floor(budget / m)`` development
        records, then the remainder is distributed one record at a time in
        lexicographic group order. A group that cannot retain the minimum
        final-test evidence after that allocation blocks the run.
        """
        if not groups:
            raise DataIntegrityError("Attack frame contains no attack groups")
        per_group, remainder = divmod(int(development_budget), len(groups))
        allocation: dict[AttackGroupId, NonNegativeCount] = {
            group: int(per_group) for group in groups
        }
        for group in groups[:remainder]:
            allocation[group] = int(allocation[group]) + 1
        for group in groups:
            total = int(group_counts.get(group, 0))
            capacity = total - min(reserve_per_group, total)
            if int(allocation[group]) > capacity:
                raise DataIntegrityError(
                    f"{FailureCode.NBAIOT_ATTACK_BUDGET_FAIL.value}: attack group {group} "
                    f"cannot retain {reserve_per_group} final-test rows"
                )
        if sum(int(value) for value in allocation.values()) != int(development_budget):
            raise RuntimeError("Attack development allocation is unbalanced")
        return allocation

    @staticmethod
    def waterfill_allocation(
        groups: tuple[AttackGroupId, ...],
        capacities: dict[AttackGroupId, NonNegativeCount],
        development_budget: PositiveCount,
    ) -> dict[AttackGroupId, NonNegativeCount]:
        """Most-even per-group allocation bounded by each group's capacity.

        At every step the lexicographically first group with the current
        minimum development count below its capacity receives one record. The
        result is the most even deterministic allocation possible subject to
        the per-group capacity constraints.
        """
        if not groups:
            raise DataIntegrityError("Attack development allocation requires groups")
        development: dict[AttackGroupId, NonNegativeCount] = {group: 0 for group in groups}
        for _ in range(int(development_budget)):
            eligible = [
                group for group in groups if int(development[group]) < int(capacities[group])
            ]
            if not eligible:
                raise DataIntegrityError(
                    f"{FailureCode.ATTACK_DEV_CAPACITY_LT_500.value}: "
                    "attack development capacity is exhausted before the budget is met"
                )
            minimum = min(int(development[group]) for group in eligible)
            chosen = next(group for group in eligible if int(development[group]) == minimum)
            development[chosen] = int(development[chosen]) + 1
        for group in groups:
            if not 0 <= int(development[group]) <= int(capacities[group]):
                raise RuntimeError("Attack development allocation violates group capacity")
        return development

    @staticmethod
    def _write_raw_splits(root: Path, splits: ClientSplits) -> None:
        client_root = root / PreparedLayout.raw_staging_directory / splits.client_id
        client_root.mkdir(parents=True, exist_ok=True)
        for item in splits.roles:
            item.frame.to_parquet(client_root / f"{item.role.value}.parquet", index=False)

    @staticmethod
    def _load_raw_splits(root: Path, client_id: ClientId) -> ClientSplits:
        client_root = root / PreparedLayout.raw_staging_directory / client_id
        roles = tuple(
            RoleFrame(role=role, frame=pd.read_parquet(client_root / f"{role.value}.parquet"))
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
        seeded_assignments: dict[CalibrationSeed, list[ClientCalibrationManifest]] = {
            int(seed): [] for seed in config.dataset.calibration_seeds
        }
        source_order_clients: list[ClientCalibrationManifest] = []

        for client_id in sorted(eligible_ids):
            splits = self._load_raw_splits(root, client_id)
            client_manifests.append(self._write_client_roles(root, splits, preprocessing))
            for seed_value in config.dataset.calibration_seeds:
                seed = int(seed_value)
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
            if item.role is DataRole.TRAIN:
                frame = preprocessing.transform(frame, splits.client_id)
            relative = f"{splits.client_id}/{item.role.value}.csv"
            path = root / relative
            frame.to_csv(path, index=False)
            roles.append(
                RoleArtifactManifest(
                    role=item.role,
                    rows=len(frame),
                    row_id_sha256=hash_row_ids(
                        frame[PreparedColumn.ROW_ID.value].astype(str).tolist()
                    ),
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
            for role in (
                DataRole.REFERENCE,
                DataRole.MISMATCH,
                DataRole.CALIBRATION,
                DataRole.BENIGN_GUARD,
            )
        )
        return ClientCalibrationManifest(client_id=splits.client_id, roles=roles)

    def _write_assignment_manifests(
        self,
        root: Path,
        config: ExperimentConfig,
        seeded: dict[CalibrationSeed, list[ClientCalibrationManifest]],
        source_order: list[ClientCalibrationManifest],
    ) -> tuple[CalibrationAssignmentReference, ...]:
        references: list[CalibrationAssignmentReference] = []
        for seed, clients in sorted(seeded.items()):
            manifest = CalibrationAssignmentManifest(
                calibration_seed=seed,
                mode=CalibrationAssignmentMode.SEEDED_PERMUTATION,
                clients=tuple(clients),
            )
            relative = PurePosixPath(
                f"{PreparedLayout.calibration_split_directory}/c{int(seed)}.json"
            )
            path = root / relative
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
            manifest = CalibrationAssignmentManifest(
                calibration_seed=int(config.dataset.primary_calibration_seed),
                mode=CalibrationAssignmentMode.SOURCE_ORDER,
                clients=tuple(source_order),
            )
            relative = PurePosixPath(PreparedLayout.source_order_split_filename)
            path = root / relative
            self.calibration_assignment_manifests.save(path, manifest)
            references.append(
                CalibrationAssignmentReference(
                    calibration_seed=int(config.dataset.primary_calibration_seed),
                    mode=CalibrationAssignmentMode.SOURCE_ORDER,
                    relative_path=relative,
                    sha256=sha256_file(path),
                )
            )
        return tuple(references)

    def _write_eligibility(
        self,
        root: Path,
        config: ExperimentConfig,
        manifest: EligibilityManifest,
    ) -> None:
        name = (
            PreparedLayout.diad_eligibility_filename
            if config.dataset.id is DatasetId.DIAD
            else PreparedLayout.eligibility_filename
        )
        atomic_write_json(root / name, manifest)

    def _write_dataset_manifest(
        self,
        root: Path,
        config: ExperimentConfig,
        sources: tuple[SourceFileManifest, ...],
        clients: tuple[ClientDatasetManifest, ...],
        feature_names: tuple[FeatureName, ...],
        assignments: tuple[CalibrationAssignmentReference, ...],
    ) -> None:
        payload = {
            "dataset_id": config.dataset.id.value,
            "source_version": config.dataset.source_version,
            "parser_version": config.dataset.parser_version,
            "data_spec_hash": config.data_spec_hash,
            "feature_names": list(feature_names),
            "clients": [
                {
                    "client_id": client.client_id,
                    "roles": [
                        {
                            PreparedColumn.ROLE.value: role.role.value,
                            "rows": role.rows,
                            "row_id_sha256": role.row_id_sha256,
                            "relative_path": str(role.relative_path),
                            "file_sha256": role.file_sha256,
                        }
                        for role in client.roles
                    ],
                }
                for client in clients
            ],
            "source_files": [
                {
                    "relative_path": str(item.relative_path),
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                }
                for item in sources
            ],
            "calibration_assignments": [
                {
                    "calibration_seed": int(item.calibration_seed),
                    "mode": item.mode.value,
                    "relative_path": str(item.relative_path),
                    "sha256": item.sha256,
                }
                for item in assignments
            ],
            "external_replication_supported": config.dataset.id is DatasetId.DIAD,
        }
        payload["created_at"] = datetime.now(UTC).isoformat()
        payload["deterministic_payload_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        atomic_write_json(root / PreparedLayout.manifest_filename, payload)
