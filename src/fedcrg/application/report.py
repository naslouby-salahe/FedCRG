"""Build a concise run report from immutable artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from fedcrg.artifacts.layout import RunLayout
from fedcrg.artifacts.verification import ArtifactVerifier


class ReportBuilder:
    def build(self, run_dir: Path) -> Path:
        layout = RunLayout(run_dir)
        manifest = json.loads(layout.manifest.read_text(encoding="utf-8"))
        verification = ArtifactVerifier().verify(layout)
        lines = [
            f"# FedCRG Run {manifest['run_id']}",
            "",
            f"- Experiment: `{manifest['experiment_id']}`",
            f"- Status: `{manifest['status']}`",
            f"- Config hash: `{manifest['config_hash']}`",
            f"- Verification valid: `{verification.valid}`",
            "",
            "Generated from immutable run metadata; scientific tables and figures remain separate artifacts.",
        ]
        output = layout.reports / "summary.md"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output
