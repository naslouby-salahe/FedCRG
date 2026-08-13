"""Deterministic manuscript-table builders from frozen configuration and artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from fedcrg.analysis.primary import confirmatory_contrasts, load_federation_results
from fedcrg.artifacts.layout import RunLayout
from fedcrg.config.models import ExperimentConfig
from fedcrg.core.enums import PolicyId


class PublicationTableBuilder:
    def literature_boundary(self, output: Path) -> Path:
        """Registered comparator identities. Published-source numeric values are
        entered by hand at submission time and are not reconstructed from evidence."""
        rows = [
            {"policy_id": policy.value, "information_regime": "benign_only"}
            for policy in (
                PolicyId.REFERENCE_QUANTILE,
                PolicyId.GLOBAL_QUANTILE,
                PolicyId.LOCAL_QUANTILE,
                PolicyId.SHRINKAGE,
                PolicyId.THREE_SIGMA,
                PolicyId.FEDCRG,
            )
        ] + [
            {"policy_id": policy.value, "information_regime": "supervised_development"}
            for policy in (
                PolicyId.DEV_F1_SELECT,
                PolicyId.SUMMARY_STATISTIC_SELECT,
                PolicyId.SUPERVISED_F1,
            )
        ]
        return self._write(pd.DataFrame.from_records(rows), output)

    def primary_policy_results(self, run_dirs: tuple[Path, ...], output: Path) -> Path:
        records = load_federation_results(run_dirs)
        return self._write(pd.DataFrame.from_records([asdict(row) for row in records]), output)

    def external_replication(self, run_dirs: tuple[Path, ...], output: Path) -> Path:
        return self.primary_policy_results(run_dirs, output)

    def admission_states_from_runs(self, run_dirs: tuple[Path, ...], output: Path) -> Path:
        records: list[dict[str, object]] = []
        for run_dir in run_dirs:
            path = RunLayout(run_dir).threshold_records
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if line:
                    row = json.loads(line)
                    row["run_id"] = run_dir.name
                    records.append(row)
        columns = [
            "run_id",
            "client_id",
            "mismatch_x",
            "mismatch_n",
            "cp_lower",
            "cp_upper",
            "readiness_n",
            "readiness_rank",
            "readiness_probability",
            "state",
            "tau_ref",
            "tau_local",
            "selected_tau",
            "selected_source",
            "reason_code",
            "tie_count",
        ]
        return self._write(pd.DataFrame.from_records(records).reindex(columns=columns), output)

    def ablations(self, run_dirs: tuple[Path, ...], output: Path) -> Path:
        records = load_federation_results(run_dirs)
        primary = tuple(row for row in records if row.experiment_id == "primary_nbaiot")
        contrasts = confirmatory_contrasts(primary, named_calibration_seed=1000)
        rows = [
            {
                "comparator": result.comparator.value,
                "metric": metric.metric,
                "method_mean": metric.method_summary.mean,
                "comparator_mean": metric.comparator_summary.mean,
                "observed_difference": metric.paired_difference.observed_difference,
                "bootstrap_lower": metric.paired_difference.lower,
                "bootstrap_upper": metric.paired_difference.upper,
                "relative_difference": metric.relative_difference,
            }
            for result in contrasts
            for metric in result.metrics
        ]
        return self._write(pd.DataFrame.from_records(rows), output)

    def sensitivity(self, experiments_root: Path, output: Path) -> Path:
        """R2-R7 each write one SensitivityEnvelope/MultiplicityEnvelope per cells/*.json file."""
        rows: list[dict[str, object]] = []
        for code in ("R2", "R3", "R4", "R5", "R6", "R7"):
            cells_root = experiments_root / code / "cells"
            if not cells_root.is_dir():
                continue
            for path in sorted(cells_root.glob("*.json")):
                payload = json.loads(path.read_text(encoding="utf-8"))
                for cell in payload.get("cells", []):
                    if isinstance(cell, dict):
                        rows.append(
                            {
                                "protocol_code": code,
                                "model_seed": payload.get("model_seed"),
                                "calibration_seed": payload.get("calibration_seed"),
                                **cell,
                            }
                        )
        return self._write(pd.DataFrame.from_records(rows), output)

    def protocol_constants(self, config: ExperimentConfig, output: Path) -> Path:
        rows = [
            ("alpha", config.protocol.alpha),
            ("rho", config.protocol.rho),
            ("band_lower", config.protocol.band.lower),
            ("band_upper", config.protocol.band.upper),
            ("readiness_assurance", config.protocol.readiness_assurance),
            ("mismatch_confidence", config.protocol.mismatch_confidence),
            ("rounds", config.training.rounds),
            ("local_epochs", config.training.local_epochs),
            ("batch_size", config.training.batch_size),
            ("learning_rate_initial", config.training.learning_rate_initial),
            ("learning_rate_final", config.training.learning_rate_final),
            ("client_fraction", config.training.client_fraction),
        ]
        return self._write(pd.DataFrame(rows, columns=["constant", "value"]), output)

    def dataset_inventory(self, prepared_manifest: Path, output: Path) -> Path:
        payload = json.loads(prepared_manifest.read_text(encoding="utf-8"))
        feature_count = len(payload.get("feature_names", []))
        rows: list[dict[str, object]] = []
        for client in sorted(payload.get("clients", []), key=lambda item: item["client_id"]):
            row: dict[str, object] = {
                "client_id": client["client_id"],
                "feature_count": feature_count,
            }
            for role_entry in client.get("roles", []):
                role = role_entry["role"]
                row[f"{role}_rows"] = role_entry["rows"]
                row[f"{role}_sha256"] = role_entry.get("row_id_sha256")
            rows.append(row)
        return self._write(pd.DataFrame.from_records(rows), output)

    def federation_results(self, run_dirs: tuple[Path, ...], output: Path) -> Path:
        rows: list[dict[str, object]] = []
        for run_dir in run_dirs:
            metrics = run_dir / "metrics" / "federation.json"
            if metrics.exists():
                row = json.loads(metrics.read_text(encoding="utf-8"))
                row["run_id"] = run_dir.name
                rows.append(row)
        return self._write(pd.DataFrame.from_records(rows), output)

    def admission_states(self, threshold_jsonl: Path, output: Path) -> Path:
        records = [
            json.loads(line)
            for line in threshold_jsonl.read_text(encoding="utf-8").splitlines()
            if line
        ]
        columns = [
            "run_id",
            "client_id",
            "mismatch_x",
            "mismatch_n",
            "cp_lower",
            "cp_upper",
            "readiness_n",
            "readiness_rank",
            "readiness_probability",
            "state",
            "tau_ref",
            "tau_local",
            "selected_tau",
            "selected_source",
            "reason_code",
            "tie_count",
        ]
        return self._write(pd.DataFrame.from_records(records).reindex(columns=columns), output)

    @staticmethod
    def _write(frame: pd.DataFrame, output: Path) -> Path:
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        return output
