"""Require independently audited prepared data and frozen statistical tables before training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from fedcrg.artifacts.json_io import as_json_int
from fedcrg.config.experiment_config import ExperimentConfig
from fedcrg.domain.constants import DIAD_EXPECTED_SOURCE_CLIENTS
from fedcrg.domain.enums import CalibrationAssignmentMode, DataRole, DatasetId
from fedcrg.domain.identifiers import ClientId
from fedcrg.pipeline.select_thresholds import ProtocolTablePrecomputer


@dataclass(frozen=True, slots=True)
class PreparedDataAudit:
    valid: bool
    problems: tuple[str, ...]
    client_count: int
    calibration_manifest_count: int


class PreparedDatasetAuditor:
    """Reconcile a prepared cache with the immutable data specification."""

    def audit(self, prepared_root: Path, config: ExperimentConfig) -> PreparedDataAudit:
        problems: list[str] = []
        manifest = self._load_object(prepared_root / "manifest.json", problems)
        preprocessing = self._load_object(prepared_root / "preprocessing.json", problems)
        eligibility_name = (
            "diad_eligibility.json" if config.dataset.id is DatasetId.DIAD else "eligibility.json"
        )
        eligibility = self._load_object(prepared_root / eligibility_name, problems)

        if manifest:
            if manifest.get("dataset_id") != config.dataset.id.value:
                problems.append("dataset manifest id differs from config")
            if manifest.get("data_spec_hash") != config.data_spec_hash:
                problems.append("dataset manifest data_spec_hash differs from config")
            features = manifest.get("feature_names")
            if not isinstance(features, list) or len(features) != config.dataset.feature_count:
                problems.append("dataset feature count differs from frozen config")

        eligible_clients = self._eligible_clients(eligibility)
        if config.dataset.id is DatasetId.NBAIOT:
            expected = {ClientId(f"nb{i:02d}") for i in range(1, 10)}
            if eligible_clients != expected:
                problems.append("N-BaIoT prepared cache does not contain exactly nb01-nb09")
        elif config.dataset.id is DatasetId.DIAD:
            discovered = eligibility.get("discovered_clients") if eligibility else None
            if not isinstance(discovered, list) or len(discovered) != DIAD_EXPECTED_SOURCE_CLIENTS:
                problems.append(
                    f"DIAD source identity ledger does not contain {DIAD_EXPECTED_SOURCE_CLIENTS} devices"
                )

        if preprocessing:
            if preprocessing.get("dataset") != config.dataset.id.value:
                problems.append("preprocessing dataset id differs from config")
            feature_columns = preprocessing.get("feature_columns")
            if (
                not isinstance(feature_columns, list)
                or len(feature_columns) != config.dataset.feature_count
            ):
                problems.append("preprocessing feature count differs from frozen config")
            clients_raw = preprocessing.get("clients")
            covered = (
                {str(item.get("client_id")) for item in clients_raw if isinstance(item, dict)}
                if isinstance(clients_raw, list)
                else set()
            )
            if covered != {client.value for client in eligible_clients}:
                problems.append("preprocessing fit-row hashes do not cover the eligible clients")

        calibration_count = 0
        for seed in config.dataset.calibration_seeds:
            path = prepared_root / "splits" / "seeded" / f"c{seed}.json"
            payload = self._load_object(path, problems)
            if not payload:
                continue
            calibration_count += 1
            self._audit_assignment(
                payload,
                seed,
                CalibrationAssignmentMode.SEEDED_PERMUTATION,
                eligible_clients,
                config,
                problems,
            )

        source_order = self._load_object(prepared_root / "splits" / "source_order.json", problems)
        if source_order:
            self._audit_assignment(
                source_order,
                config.dataset.primary_calibration_seed,
                CalibrationAssignmentMode.SOURCE_ORDER,
                eligible_clients,
                config,
                problems,
            )

        return PreparedDataAudit(
            not problems, tuple(problems), len(eligible_clients), calibration_count
        )

    @staticmethod
    def _load_object(path: Path, problems: list[str]) -> dict[str, object]:
        if not path.is_file():
            problems.append(f"missing prepared artifact: {path.name}")
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            problems.append(f"invalid JSON artifact {path}: {exc}")
            return {}
        if not isinstance(payload, dict):
            problems.append(f"prepared artifact is not a JSON object: {path}")
            return {}
        return payload

    @staticmethod
    def _eligible_clients(payload: dict[str, object]) -> set[ClientId]:
        values = payload.get("eligible_clients")
        if not isinstance(values, list):
            return set()
        return {ClientId(str(value)) for value in values}

    @staticmethod
    def _audit_assignment(
        payload: dict[str, object],
        expected_seed: int,
        expected_mode: CalibrationAssignmentMode,
        eligible_clients: set[ClientId],
        config: ExperimentConfig,
        problems: list[str],
    ) -> None:
        if as_json_int(payload.get("calibration_seed", -1)) != expected_seed:
            problems.append(f"calibration assignment seed mismatch: expected {expected_seed}")
        if payload.get("mode") != expected_mode.value:
            problems.append(f"calibration assignment mode mismatch: expected {expected_mode.value}")
        rows = payload.get("clients")
        if not isinstance(rows, list):
            problems.append("calibration assignment lacks client records")
            return
        observed: set[ClientId] = set()
        expected_counts = {
            DataRole.REFERENCE: config.dataset.split.reference_benign,
            DataRole.MISMATCH: config.dataset.split.mismatch_benign,
            DataRole.CALIBRATION: config.dataset.split.calibration_benign,
            DataRole.BENIGN_GUARD: config.dataset.split.benign_guard,
        }
        for row in rows:
            if not isinstance(row, dict):
                problems.append("malformed calibration client record")
                continue
            client = ClientId(str(row.get("client_id")))
            observed.add(client)
            roles = row.get("roles")
            if not isinstance(roles, dict):
                problems.append(f"{client}: calibration roles missing")
                continue
            hashes: set[str] = set()
            total = 0
            for role, expected_count in expected_counts.items():
                record = roles.get(role.value)
                if not isinstance(record, dict):
                    problems.append(f"{client}: missing {role.value} assignment")
                    continue
                count = int(record.get("row_count", -1))
                digest = str(record.get("row_id_sha256", ""))
                if count != expected_count:
                    problems.append(f"{client}/{role.value}: {count} rows != {expected_count}")
                if len(digest) != 64:
                    problems.append(f"{client}/{role.value}: invalid row-id hash")
                if digest in hashes:
                    problems.append(
                        f"{client}: duplicate calibration role hash suggests role reuse"
                    )
                hashes.add(digest)
                total += max(count, 0)
            if total != config.dataset.split.reservoir_size:
                problems.append(f"{client}: calibration roles do not cover the full reservoir")
        if observed != eligible_clients:
            problems.append("calibration assignment client set differs from eligibility")


@dataclass(frozen=True, slots=True)
class PreflightResult:
    prepared_data: PreparedDataAudit
    readiness_table: Path
    mismatch_table: Path

    @property
    def valid(self) -> bool:
        return self.prepared_data.valid


class ResearchPreflight:
    """Require independently audited data and frozen statistical tables before training."""

    def __init__(
        self,
        data_auditor: PreparedDatasetAuditor | None = None,
        table_precomputer: ProtocolTablePrecomputer | None = None,
    ) -> None:
        self.data_auditor = data_auditor or PreparedDatasetAuditor()
        self.table_precomputer = table_precomputer or ProtocolTablePrecomputer()

    def run(self, config: ExperimentConfig, prepared_root: Path) -> PreflightResult:
        audit = self.data_auditor.audit(prepared_root, config)
        if not audit.valid:
            raise RuntimeError("Prepared-data preflight failed: " + ", ".join(audit.problems))
        readiness, mismatch = self.table_precomputer.precompute(config)
        return PreflightResult(
            prepared_data=audit,
            readiness_table=readiness,
            mismatch_table=mismatch,
        )
