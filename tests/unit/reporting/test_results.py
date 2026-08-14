"""Unit tests for the publication results builder and verifier."""

from __future__ import annotations

import json
from pathlib import Path

from fedcrg.reporting import ResultsBuilder, ResultsVerifier


def _write_fake_evidence(outputs_root: Path) -> None:
    run_root = outputs_root / "runs" / "run_1"
    (run_root / "metrics").mkdir(parents=True)
    (run_root / "metrics" / "metric_record.jsonl").write_text(
        '{"run_id": "run_1", "fpr": 0.01}\n', encoding="utf-8"
    )
    analysis_root = outputs_root / "cache" / "analysis"
    analysis_root.mkdir(parents=True)
    (analysis_root / "readiness_plans.json").write_text('{"plans": []}\n', encoding="utf-8")
    publication = outputs_root / "reports" / "publication"
    (publication / "tables").mkdir(parents=True)
    (publication / "figures").mkdir()
    (publication / "tables" / "table_1.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (publication / "figures" / "figure_1.png").write_bytes(b"png")


def test_results_builder_creates_bundle_and_marks_partial_evidence_incomplete(
    tmp_path: Path,
) -> None:
    outputs_root = tmp_path / "outputs"
    _write_fake_evidence(outputs_root)
    destination = ResultsBuilder().build(
        campaign_id="c1",
        outputs_root=outputs_root,
        results_root=tmp_path / "results",
    )
    assert destination == tmp_path / "results" / "c1"
    assert (destination / "manifest.json").is_file()
    assert (destination / "checksums.json").is_file()
    assert (destination / "metrics" / "metric_records.json").is_file()
    assert (destination / "statistics" / "readiness_plans.json").is_file()
    assert (destination / "tables" / "table_1.csv").is_file()
    assert (destination / "figures" / "figure_1.png").is_file()
    assert (destination / "provenance" / "provenance.json").is_file()

    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["campaign_id"] == "c1"
    # The fake evidence has no completed run cells, so the bundle must honestly
    # report itself as incomplete rather than claiming completeness.
    assert manifest["complete"] is False
    checksums = json.loads((destination / "checksums.json").read_text(encoding="utf-8"))
    assert manifest["file_count"] == len(checksums)


def test_results_builder_refuses_to_overwrite(tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    _write_fake_evidence(outputs_root)
    builder = ResultsBuilder()
    builder.build(campaign_id="c1", outputs_root=outputs_root, results_root=tmp_path / "results")
    import pytest

    with pytest.raises(FileExistsError):
        builder.build(
            campaign_id="c1", outputs_root=outputs_root, results_root=tmp_path / "results"
        )


def test_results_verifier_detects_missing_bundle(tmp_path: Path) -> None:
    result = ResultsVerifier().verify(
        "missing",
        results_root=tmp_path / "results",
        outputs_root=tmp_path / "outputs",
    )
    assert not result.valid
    assert any("does not exist" in problem for problem in result.problems)
