"""Build reproducible run reports from immutable threshold and metric artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fedcrg.artifacts.layout import RunLayout
from fedcrg.artifacts.verification import ArtifactVerifier


class ReportBuilder:
    def build(self, run_dir: Path) -> Path:
        layout = RunLayout(run_dir)
        manifest = json.loads(layout.manifest.read_text(encoding="utf-8"))
        verification = ArtifactVerifier().verify(layout)
        federation = None
        if layout.federation_metrics.exists():
            federation = json.loads(layout.federation_metrics.read_text(encoding="utf-8"))
        client_count = 0
        if layout.metric_records.exists():
            client_count = sum(1 for line in layout.metric_records.read_text(encoding="utf-8").splitlines() if line)
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
            lines.extend([
                "",
                "## Federation endpoints",
                "",
                f"- MEBE: `{federation.get('mebe')}`",
                f"- HighExcess: `{federation.get('high_excess')}`",
                f"- BandViolationRate: `{federation.get('band_violation_rate')}`",
                f"- MAFE: `{federation.get('mafe')}`",
                f"- ABMacroTPR: `{federation.get('attack_balanced_macro_tpr')}`",
            ])
        lines.extend([
            "",
            "This report is generated only from immutable run artifacts; it does not load a detector or retrain a model.",
        ])
        output = layout.reports / "summary.md"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output

    def build_publication_table(self, run_dirs: tuple[Path, ...], output: Path) -> Path:
        rows: list[dict[str, object]] = []
        for run_dir in run_dirs:
            layout = RunLayout(run_dir)
            if layout.federation_metrics.exists():
                row = json.loads(layout.federation_metrics.read_text(encoding="utf-8"))
                row["run_id"] = run_dir.name
                rows.append(row)
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame.from_records(rows).to_csv(output, index=False)
        return output
