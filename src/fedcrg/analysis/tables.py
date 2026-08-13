"""Deterministic manuscript-table builders from frozen configuration and artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fedcrg.config.models import ExperimentConfig


class PublicationTableBuilder:
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
        rows: list[dict[str, object]] = []
        for client_id, roles in sorted(payload.get("clients", {}).items()):
            row: dict[str, object] = {
                "client_id": client_id,
                "feature_count": len(payload.get("feature_names", [])),
            }
            for role, detail in roles.items():
                row[f"{role}_rows"] = detail["rows"]
                row[f"{role}_sha256"] = detail.get("sha256")
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
