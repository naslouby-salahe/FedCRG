"""Persistence and deterministic hashing for prepared-dataset evidence."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, UTC
from pathlib import Path, PurePosixPath

from fedcrg.artifacts.serialization import atomic_write_json, to_json_value
from fedcrg.core.enums import (
    CalibrationAssignmentMode,
    ChronologyStatus,
    DataRole,
    DatasetId,
    EligibilityStatus,
    FailureCode,
)
from fedcrg.core.ids import CalibrationSeed, ClientId, Sha256
from fedcrg.data.manifests import (
    CalibrationAssignmentManifest,
    CalibrationAssignmentReference,
    CalibrationRoleManifest,
    ClientCalibrationManifest,
    ClientDatasetManifest,
    EligibilityManifest,
    PreparedDatasetManifest,
    RoleArtifactManifest,
    SourceFileManifest,
)
from fedcrg.data.models import EligibilityRecord


class PreparedDatasetManifestStore:
    def build(
        self,
        *,
        dataset_id: DatasetId,
        source_version: str,
        parser_version: str,
        data_spec_hash: Sha256,
        feature_names: tuple[str, ...],
        clients: tuple[ClientDatasetManifest, ...],
        source_files: tuple[SourceFileManifest, ...],
        calibration_assignments: tuple[CalibrationAssignmentReference, ...],
        external_replication_supported: bool,
        dataset_level_code: FailureCode | None,
    ) -> PreparedDatasetManifest:
        deterministic = {
            "dataset_id": dataset_id,
            "source_version": source_version,
            "parser_version": parser_version,
            "data_spec_hash": data_spec_hash,
            "feature_names": feature_names,
            "clients": clients,
            "source_files": source_files,
            "calibration_assignments": calibration_assignments,
            "external_replication_supported": external_replication_supported,
            "dataset_level_code": dataset_level_code,
        }
        canonical = json.dumps(
            to_json_value(deterministic),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return PreparedDatasetManifest(
            dataset_id=dataset_id,
            source_version=source_version,
            parser_version=parser_version,
            data_spec_hash=data_spec_hash,
            feature_names=feature_names,
            clients=clients,
            source_files=source_files,
            calibration_assignments=calibration_assignments,
            external_replication_supported=external_replication_supported,
            dataset_level_code=dataset_level_code,
            created_at=datetime.now(UTC),
            deterministic_payload_sha256=Sha256(hashlib.sha256(canonical).hexdigest()),
        )

    def save(self, path: Path, manifest: PreparedDatasetManifest) -> None:
        atomic_write_json(path, manifest)

    def load(self, path: Path) -> PreparedDatasetManifest:
        raw = json.loads(path.read_text(encoding="utf-8"))
        clients = tuple(
            ClientDatasetManifest(
                client_id=ClientId(str(client["client_id"])),
                roles=tuple(
                    RoleArtifactManifest(
                        role=DataRole(str(role["role"])),
                        rows=int(role["rows"]),
                        row_id_sha256=Sha256(str(role["row_id_sha256"])),
                        relative_path=PurePosixPath(str(role["relative_path"])),
                        file_sha256=Sha256(str(role["file_sha256"])),
                    )
                    for role in client["roles"]
                ),
            )
            for client in raw["clients"]
        )
        sources = tuple(
            SourceFileManifest(
                relative_path=PurePosixPath(str(item["relative_path"])),
                sha256=Sha256(str(item["sha256"])),
                size_bytes=int(item["size_bytes"]),
            )
            for item in raw["source_files"]
        )
        assignments = tuple(
            CalibrationAssignmentReference(
                calibration_seed=CalibrationSeed(int(item["calibration_seed"])),
                mode=CalibrationAssignmentMode(str(item["mode"])),
                sha256=Sha256(str(item["sha256"])),
            )
            for item in raw["calibration_assignments"]
        )
        code = raw["dataset_level_code"]
        manifest = PreparedDatasetManifest(
            dataset_id=DatasetId(str(raw["dataset_id"])),
            source_version=str(raw["source_version"]),
            parser_version=str(raw["parser_version"]),
            data_spec_hash=Sha256(str(raw["data_spec_hash"])),
            feature_names=tuple(str(value) for value in raw["feature_names"]),
            clients=clients,
            source_files=sources,
            calibration_assignments=assignments,
            external_replication_supported=bool(raw["external_replication_supported"]),
            dataset_level_code=None if code is None else FailureCode(str(code)),
            created_at=datetime.fromisoformat(str(raw["created_at"])),
            deterministic_payload_sha256=Sha256(str(raw["deterministic_payload_sha256"])),
        )
        rebuilt = self.build(
            dataset_id=manifest.dataset_id,
            source_version=manifest.source_version,
            parser_version=manifest.parser_version,
            data_spec_hash=manifest.data_spec_hash,
            feature_names=manifest.feature_names,
            clients=manifest.clients,
            source_files=manifest.source_files,
            calibration_assignments=manifest.calibration_assignments,
            external_replication_supported=manifest.external_replication_supported,
            dataset_level_code=manifest.dataset_level_code,
        )
        if rebuilt.deterministic_payload_sha256 != manifest.deterministic_payload_sha256:
            raise ValueError("Prepared dataset deterministic payload hash mismatch")
        return manifest


class EligibilityManifestStore:
    def save(self, path: Path, manifest: EligibilityManifest) -> None:
        atomic_write_json(path, manifest)

    def load(self, path: Path) -> EligibilityManifest:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return EligibilityManifest(
            dataset_id=DatasetId(str(raw["dataset_id"])),
            discovered_clients=tuple(ClientId(str(value)) for value in raw["discovered_clients"]),
            eligible_clients=tuple(ClientId(str(value)) for value in raw["eligible_clients"]),
            records=tuple(
                EligibilityRecord(
                    client_id=ClientId(str(item["client_id"])),
                    status=EligibilityStatus(str(item["status"])),
                    benign_count=int(item["benign_count"]),
                    malicious_count=int(item["malicious_count"]),
                    attack_development_capacity=int(item["attack_development_capacity"]),
                    primary_code=(
                        None
                        if item["primary_code"] is None
                        else FailureCode(str(item["primary_code"]))
                    ),
                    secondary_codes=tuple(
                        FailureCode(str(code)) for code in item["secondary_codes"]
                    ),
                    chronology=ChronologyStatus(str(item["chronology"])),
                )
                for item in raw["records"]
            ),
        )


class CalibrationAssignmentManifestStore:
    def save(self, path: Path, manifest: CalibrationAssignmentManifest) -> None:
        atomic_write_json(path, manifest)

    def load(self, path: Path) -> CalibrationAssignmentManifest:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return CalibrationAssignmentManifest(
            calibration_seed=CalibrationSeed(int(raw["calibration_seed"])),
            mode=CalibrationAssignmentMode(str(raw["mode"])),
            clients=tuple(
                ClientCalibrationManifest(
                    client_id=ClientId(str(client["client_id"])),
                    roles=tuple(
                        CalibrationRoleManifest(
                            role=DataRole(str(role["role"])),
                            row_count=int(role["row_count"]),
                            row_id_sha256=Sha256(str(role["row_id_sha256"])),
                        )
                        for role in client["roles"]
                    ),
                )
                for client in raw["clients"]
            ),
        )
