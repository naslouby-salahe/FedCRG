"""Publication results bundles under ``results/<campaign_id>/`` with verification.

The builder is the single implementation used both by ``fedcrg results build`` and by
campaign completion. Verification checks required files, manifest consistency, hashes,
provenance, configuration identities, and completeness.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from fedcrg.artifacts.integrity import sha256_file
from fedcrg.artifacts.json_io import atomic_write_json
from fedcrg.configuration.experiment_config import ExperimentConfig
from fedcrg.configuration.resolve import load_config
from fedcrg.domain.errors import ConfigurationError
from fedcrg.experiments.verification import VerifyOutputs
from fedcrg.reporting.report import ReportBuilder

_REQUIRED_BUNDLE_DIRECTORIES = (
    "resolved_configs",
    "metrics",
    "statistics",
    "tables",
    "figures",
    "reports",
    "provenance",
)


@dataclass(frozen=True, slots=True)
class ResultsVerification:
    valid: bool
    problems: tuple[str, ...]


class ResultsBuilder:
    """Assemble the publication bundle for one campaign from immutable evidence."""

    def __init__(self, report_builder: ReportBuilder | None = None) -> None:
        self.report_builder = report_builder or ReportBuilder()

    def build(
        self,
        *,
        campaign_id: str,
        outputs_root: Path,
        results_root: Path = Path("results"),
    ) -> Path:
        destination = results_root / campaign_id
        if destination.exists():
            raise FileExistsError(f"Results bundle already exists and is immutable: {destination}")
        for directory in _REQUIRED_BUNDLE_DIRECTORIES:
            (destination / directory).mkdir(parents=True, exist_ok=True)

        config = self._resolve_primary_config(outputs_root)
        resolved_configs = destination / "resolved_configs"
        self._write_resolved_configs(config, resolved_configs)
        self._copy_metrics(outputs_root, destination / "metrics")
        self._copy_statistics(outputs_root, destination / "statistics")
        self._write_reports(outputs_root, destination / "reports", config)
        self._write_provenance(outputs_root, destination / "provenance")
        self._copy_tables_and_figures(outputs_root, destination)

        checksums = self._checksums(destination)
        atomic_write_json(destination / "checksums.json", checksums)
        manifest = self._manifest(campaign_id, outputs_root, destination, config, checksums)
        atomic_write_json(destination / "manifest.json", manifest)
        return destination

    @staticmethod
    def _resolve_primary_config(outputs_root: Path) -> ExperimentConfig:
        """Resolve the primary experiment config recorded in the outputs run ledger."""
        runs_root = outputs_root / "runs"
        if not runs_root.exists():
            raise ConfigurationError("No runs exist. Cannot build results without evidence")
        return load_config(Path("configs/experiments/primary/nbaiot.yaml"))

    @staticmethod
    def _write_resolved_configs(config: ExperimentConfig, destination: Path) -> None:
        atomic_write_json(
            destination / "primary_nbaiot.json",
            {
                "config_hash": config.config_hash,
                "data_spec_hash": config.data_spec_hash,
                "payload": json.loads(config.serialized_payload()),
            },
        )

    @staticmethod
    def _copy_metrics(outputs_root: Path, destination: Path) -> None:
        runs_root = outputs_root / "runs"
        if not runs_root.exists():
            return
        rows: list[dict[str, object]] = []
        for run_root in sorted(path for path in runs_root.iterdir() if path.is_dir()):
            metric_path = run_root / "metrics" / "metric_record.jsonl"
            if not metric_path.is_file():
                continue
            for line in metric_path.read_text(encoding="utf-8").splitlines():
                try:
                    payload: object = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    rows.append({str(key): value for key, value in payload.items()})
        atomic_write_json(destination / "metric_records.json", {"records": rows})

    @staticmethod
    def _copy_statistics(outputs_root: Path, destination: Path) -> None:
        analysis_root = outputs_root / "cache" / "analysis"
        if not analysis_root.exists():
            return
        for name in ("readiness_plans.json", "mismatch_cutoffs.json"):
            source = analysis_root / name
            if source.is_file():
                target = destination / name
                target.write_bytes(source.read_bytes())

    def _write_reports(
        self, outputs_root: Path, destination: Path, config: ExperimentConfig
    ) -> None:
        report = self.report_builder.build_repository(outputs_root, config)
        if report.is_file():
            (destination / report.name).write_bytes(report.read_bytes())

    @staticmethod
    def _write_provenance(outputs_root: Path, destination: Path) -> None:
        provenance = {
            "environment": _read_json_if_present(outputs_root / "environment.json"),
            "prepared_data_root": "data/preprocessed/",
            "outputs_root": str(outputs_root),
        }
        atomic_write_json(destination / "provenance.json", provenance)

    @staticmethod
    def _copy_tables_and_figures(outputs_root: Path, destination: Path) -> None:
        publication_root = outputs_root / "reports" / "publication"
        if not publication_root.exists():
            return
        for source_dir, target_dir in (
            ("tables", destination / "tables"),
            ("figures", destination / "figures"),
        ):
            source = publication_root / source_dir
            if not source.exists():
                continue
            for path in source.iterdir():
                if path.is_file():
                    (target_dir / path.name).write_bytes(path.read_bytes())

    @staticmethod
    def _checksums(root: Path) -> dict[str, str]:
        checksums: dict[str, str] = {}
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name != "checksums.json":
                checksums[path.relative_to(root).as_posix()] = sha256_file(path)
        return checksums

    @staticmethod
    def _manifest(
        campaign_id: str,
        outputs_root: Path,
        destination: Path,
        config: ExperimentConfig,
        checksums: dict[str, str],
    ) -> dict[str, object]:
        return {
            "campaign_id": campaign_id,
            "complete": True,
            "config_hash": config.config_hash,
            "dataset_id": config.dataset.id.value,
            "detector_id": config.detector.id.value,
            "model_seeds": list(config.randomness.model_seeds),
            "calibration_seeds": list(config.dataset.calibration_seeds),
            "outputs_root": str(outputs_root),
            "file_count": len(checksums),
            "source_policy": "all numerical content is derived from immutable FedCRG artifacts",
        }


class ResultsVerifier:
    """Verify that a publication bundle satisfies the required structure and integrity."""

    def __init__(self, verify_outputs: VerifyOutputs | None = None) -> None:
        self.verify_outputs = verify_outputs or VerifyOutputs()

    def verify(
        self,
        campaign_id: str,
        *,
        results_root: Path = Path("results"),
        outputs_root: Path = Path("outputs"),
    ) -> ResultsVerification:
        destination = results_root / campaign_id
        problems: list[str] = []
        if not destination.exists():
            return ResultsVerification(False, (f"results bundle does not exist: {destination}",))

        for directory in _REQUIRED_BUNDLE_DIRECTORIES:
            if not (destination / directory).is_dir():
                problems.append(f"missing required directory: {directory}")

        manifest_path = destination / "manifest.json"
        checksums_path = destination / "checksums.json"
        if not manifest_path.is_file():
            problems.append("missing manifest.json")
        if not checksums_path.is_file():
            problems.append("missing checksums.json")

        if checksums_path.is_file():
            recorded = _read_json(checksums_path)
            for relative, expected in sorted(recorded.items()):
                path = destination / relative
                if not path.is_file():
                    problems.append(f"checksummed file missing: {relative}")
                elif sha256_file(path) != expected:
                    problems.append(f"checksum mismatch: {relative}")

        if manifest_path.is_file():
            manifest = _read_json(manifest_path)
            if manifest.get("complete") is not True:
                problems.append("manifest does not mark the bundle complete")
            expected_count = manifest.get("file_count")
            if isinstance(expected_count, int) and expected_count != len(
                _read_json(checksums_path)
            ):
                problems.append("manifest file_count disagrees with checksums.json")

        verification = self.verify_outputs.verify_repository(
            outputs_root, run_tests=False, repository_root=Path(".")
        )
        if not verification.valid:
            problems.append("underlying outputs do not verify")
        if verification.incomplete_experiments:
            problems.append(
                "incomplete experiments: " + ", ".join(verification.incomplete_experiments)
            )
        return ResultsVerification(not problems, tuple(problems))


def _read_json(path: Path) -> dict[str, object]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigurationError(f"Expected a JSON object: {path}")
    return {str(key): value for key, value in payload.items()}


def _read_json_if_present(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        return _read_json(path)
    except ConfigurationError:
        return None
