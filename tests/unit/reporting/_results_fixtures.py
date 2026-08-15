"""Shared fixture for writing a minimal fake outputs tree used by results-bundle tests."""

from __future__ import annotations

from pathlib import Path

from fedcrg.paths import OutputsLayout


def write_fake_evidence(outputs_root: Path) -> None:
    layout = OutputsLayout(outputs_root)
    run_layout = layout.run("run_1")
    run_layout.metrics.mkdir(parents=True)
    run_layout.metric_records.write_text('{"run_id": "run_1", "fpr": 0.01}\n', encoding="utf-8")
    layout.cache_analysis.mkdir(parents=True)
    layout.readiness_plans_file.write_text('{"plans": []}\n', encoding="utf-8")
    publication = layout.publication
    publication.tables.mkdir(parents=True)
    publication.figures.mkdir()
    (publication.tables / "table_1.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (publication.figures / "figure_1.png").write_bytes(b"png")
