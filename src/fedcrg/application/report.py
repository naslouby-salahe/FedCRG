"""Build reproducible run and repository reports only from immutable evidence."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from fedcrg.analysis.primary import (
    confirmatory_contrasts,
    load_federation_results,
    split_sensitivity,
)
from fedcrg.application.verify import VerifyOutputs
from fedcrg.artifacts.layout import RunLayout
from fedcrg.artifacts.verification import ArtifactVerifier
from fedcrg.domain.enums import ExperimentCode


class ReportBuilder:
    """Never load a detector: reports are projections of frozen evidence only."""

    def build(self, run_dir: Path) -> Path:
        layout = RunLayout(run_dir)
        manifest = json.loads(layout.manifest.read_text(encoding="utf-8"))
        verification = ArtifactVerifier().verify(layout)
        federation = (
            json.loads(layout.federation_metrics.read_text(encoding="utf-8"))
            if layout.federation_metrics.exists()
            else None
        )
        client_count = self._jsonl_count(layout.metric_records)
        lines = [
            f"# FedCRG Run {manifest['run_id']}",
            "",
            f"- Experiment: `{manifest['experiment_id']}`",
            f"- Policy: `{manifest['policy_id']}`",
            f"- Status: `{manifest['status']}`",
            f"- Config hash: `{manifest['config_hash']}`",
            f"- Verified artifact hashes: `{verification.valid}`",
            f"- Evaluated clients: `{client_count}`",
        ]
        if federation is not None:
            lines.extend(
                [
                    "",
                    "## Federation endpoints",
                    "",
                    f"- MEBE: `{federation.get('mebe')}`",
                    f"- HighExcess: `{federation.get('high_excess')}`",
                    f"- BandViolationRate: `{federation.get('band_violation_rate')}`",
                    f"- MAFE: `{federation.get('mafe')}`",
                    f"- ABMacroTPR: `{federation.get('attack_balanced_macro_tpr')}`",
                    f"- MacroTPR: `{federation.get('macro_tpr')}`",
                    f"- Worst-client TPR: `{federation.get('worst_client_tpr')}`",
                ]
            )
        lines.extend(
            [
                "",
                "This report is generated only from immutable run artifacts, it does not load a detector or retrain a model.",
            ]
        )
        output = layout.reports / "summary.md"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output

    def build_repository(self, outputs_root: Path) -> Path:
        """Build a publication-oriented evidence index from every completed run."""
        reports_root = outputs_root / "reports" / "latest"
        reports_root.mkdir(parents=True, exist_ok=True)
        run_dirs = tuple(self._completed_run_dirs(outputs_root / "runs"))
        policy_table = self.build_publication_table(
            run_dirs,
            reports_root / "policy_federation_results.csv",
        )
        federation_records = load_federation_results(run_dirs)
        primary_records = tuple(
            row for row in federation_records if row.experiment_id == "primary_nbaiot"
        )
        verification = VerifyOutputs().verify_repository(
            outputs_root,
            run_tests=False,
        )
        r1 = next(
            (
                item
                for item in verification.experiment_completion
                if item.protocol_code is ExperimentCode.R1
            ),
            None,
        )
        contrasts_path = reports_root / "primary_contrasts.json"
        if r1 is not None and r1.complete:
            contrasts = confirmatory_contrasts(
                primary_records,
                named_calibration_seed=1000,
            )
            contrasts_payload: object = [
                {
                    "comparator": item.comparator.value,
                    "metrics": [asdict(metric) for metric in item.metrics],
                }
                for item in contrasts
            ]
        else:
            contrasts_payload = {
                "status": "incomplete",
                "reason": "R1 workload has not reconciled, confirmatory contrasts are withheld",
            }
        contrasts_path.write_text(
            json.dumps(contrasts_payload, indent=2) + "\n",
            encoding="utf-8",
        )
        sensitivity_path = reports_root / "split_sensitivity.csv"
        pd.DataFrame.from_records(split_sensitivity(primary_records)).to_csv(
            sensitivity_path,
            index=False,
        )
        experiment_rows = [
            {
                "experiment": item.protocol_code,
                "complete": item.complete,
                "expected_cells": item.expected_cells,
                "observed_cells": item.observed_cells,
                "problems": ", ".join(item.problems),
            }
            for item in verification.experiment_completion
        ]
        experiment_table = reports_root / "experiment_completion.csv"
        pd.DataFrame.from_records(experiment_rows).to_csv(experiment_table, index=False)

        lines = [
            "# FedCRG Reproducibility Report",
            "",
            f"- Repository evidence complete: `{verification.valid}`",
            f"- Completed run directories: `{len(run_dirs)}`",
            f"- Incomplete experiment codes: `{', '.join(verification.incomplete_experiments) or 'none'}`",
            f"- Federation-results table: `{policy_table.name}`",
            f"- Experiment-completion table: `{experiment_table.name}`",
            f"- Confirmatory contrast analysis: `{contrasts_path.name}`",
            f"- Split-sensitivity summary: `{sensitivity_path.name}`",
            "",
            "## Experiment ledger",
            "",
        ]
        for item in verification.experiment_completion:
            status = "COMPLETE" if item.complete else "INCOMPLETE"
            detail = ", ".join(item.problems) if item.problems else "ledger reconciled"
            lines.append(
                f"- **{item.protocol_code}** — {status}, observed={item.observed_cells}, {detail}"
            )
        lines.extend(
            [
                "",
                "The repository report is generated from immutable artifacts and workload ledgers. Missing experiments remain explicitly incomplete, report generation never fills missing evidence with inferred values.",
            ]
        )
        output = reports_root / "README.md"
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output

    def build_publication_table(
        self,
        run_dirs: tuple[Path, ...],
        output: Path,
    ) -> Path:
        rows: list[dict[str, object]] = []
        for run_dir in run_dirs:
            layout = RunLayout(run_dir)
            if not layout.federation_metrics.exists():
                continue
            row = json.loads(layout.federation_metrics.read_text(encoding="utf-8"))
            manifest = json.loads(layout.manifest.read_text(encoding="utf-8"))
            row.update(
                {
                    "run_id": run_dir.name,
                    "experiment_id": manifest["experiment_id"],
                    "model_seed": manifest["model_seed"],
                    "calibration_seed": manifest["calibration_seed"],
                    "policy_id": manifest["policy_id"],
                }
            )
            rows.append(row)
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame.from_records(rows).to_csv(output, index=False)
        return output

    @staticmethod
    def _completed_run_dirs(runs_root: Path):
        if not runs_root.exists():
            return
        for path in sorted(item for item in runs_root.iterdir() if item.is_dir()):
            manifest_path = path / "manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") == "complete":
                yield path

    @staticmethod
    def _jsonl_count(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)
