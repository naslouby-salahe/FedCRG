from __future__ import annotations

import json
from pathlib import Path

from fedcrg.evidence.models import GitEnvironment
from fedcrg.evidence.store import atomic_write_json
from fedcrg.paths import OutputsLayout, ResultsBundleLayout
from fedcrg.reporting import (
    ResultsBuilder,
    ResultsVerifier,
    build_results_bundle,
    verify_results_bundle,
)
from fedcrg.types import DataIntegrityError, ExperimentId, PolicyId

import pytest

from tests._fixtures import primary_experiment_config
from tests.unit.reporting._results_fixtures import write_fake_evidence
from tests.unit.reporting._run_fixtures import write_completed_run


def test_results_builder_with_empty_outputs_root(tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    outputs_root.mkdir()
    destination = ResultsBuilder().build(
        campaign_id="c2",
        outputs_root=outputs_root,
        results_root=tmp_path / "results",
    )
    assert (destination / "manifest.json").is_file()
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is False
    assert not (destination / "metrics" / "metric_records.json").exists()
    assert list((destination / "tables").iterdir()) == []


def test_results_builder_copies_git_environment_into_provenance(tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    write_fake_evidence(outputs_root)
    environment = GitEnvironment(
        git_commit="deadbeef",
        git_clean=True,
        git_patch_sha256=None,
        environment_pin_sha256="e" * 64,
        environment_pin_kind="conda",
    )
    atomic_write_json(
        OutputsLayout(outputs_root).environment_file, environment.model_dump(mode="json")
    )
    destination = ResultsBuilder().build(
        campaign_id="c3",
        outputs_root=outputs_root,
        results_root=tmp_path / "results",
    )
    provenance = json.loads((destination / "provenance" / "provenance.json").read_text())
    assert provenance["environment"]["git_commit"] == "deadbeef"


def test_build_results_bundle_and_verify_wrappers_round_trip(tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    write_fake_evidence(outputs_root)
    destination = build_results_bundle("c4", outputs_root, tmp_path / "results")
    assert destination.is_dir()
    verification = verify_results_bundle("c4", tmp_path / "results")
    assert verification.valid
    assert verification.problems == ()


def test_results_verifier_detects_checksum_mismatch(tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    write_fake_evidence(outputs_root)
    destination = ResultsBuilder().build(
        campaign_id="c5",
        outputs_root=outputs_root,
        results_root=tmp_path / "results",
    )
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    result = ResultsVerifier().verify("c5", results_root=tmp_path / "results")
    assert not result.valid
    assert any("hash mismatch" in problem for problem in result.problems)


def test_results_verifier_detects_unchecksummed_file(tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    write_fake_evidence(outputs_root)
    destination = ResultsBuilder().build(
        campaign_id="c6",
        outputs_root=outputs_root,
        results_root=tmp_path / "results",
    )
    (destination / "tables" / "extra_untracked.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    result = ResultsVerifier().verify("c6", results_root=tmp_path / "results")
    assert not result.valid
    assert any("unchecksummed" in problem for problem in result.problems)


def test_results_verifier_detects_missing_required_directory(tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    write_fake_evidence(outputs_root)
    destination = ResultsBuilder().build(
        campaign_id="c7",
        outputs_root=outputs_root,
        results_root=tmp_path / "results",
    )
    layout = ResultsBundleLayout(destination)
    import shutil

    shutil.rmtree(layout.tables)
    result = ResultsVerifier().verify("c7", results_root=tmp_path / "results")
    assert not result.valid
    assert any("missing required bundle directory" in problem for problem in result.problems)


def test_copy_statistics_raises_when_target_escapes_statistics_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs_root = tmp_path / "outputs"
    write_fake_evidence(outputs_root)
    monkeypatch.setattr(
        ResultsBundleLayout,
        "readiness_plans",
        property(lambda self: self.root.parent / "escaped_readiness_plans.json"),
    )
    builder = ResultsBuilder()
    with pytest.raises(DataIntegrityError, match="escapes the statistics directory"):
        builder.build(
            campaign_id="c8",
            outputs_root=outputs_root,
            results_root=tmp_path / "results",
        )


def test_write_reports_raises_when_report_escapes_reports_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs_root = tmp_path / "outputs"
    write_fake_evidence(outputs_root)
    decoy = tmp_path / "decoy_reports"
    call_count = {"value": 0}

    def _flaky_reports(self: ResultsBundleLayout) -> Path:
        call_count["value"] += 1
        if call_count["value"] == 2:
            return decoy
        return self.root / "reports"

    monkeypatch.setattr(ResultsBundleLayout, "reports", property(_flaky_reports))
    builder = ResultsBuilder()
    with pytest.raises(DataIntegrityError, match="escapes the reports directory"):
        builder.build(
            campaign_id="c9",
            outputs_root=outputs_root,
            results_root=tmp_path / "results",
        )


def test_results_builder_skips_malformed_metric_record_lines(tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    config = primary_experiment_config(outputs_root)
    run_dir = write_completed_run(
        outputs_root,
        run_id="run-1",
        experiment_id=ExperimentId.PRIMARY_NBAIOT,
        policy_id=PolicyId.FEDCRG,
        config=config,
        model_seed=11,
        calibration_seed=1000,
        include_run_config=False,
    )
    from fedcrg.paths import RunLayout

    valid_line = json.dumps(
        {
            "run_id": "run-1",
            "policy_id": "fedcrg",
            "client_id": "nb01",
            "benign_n": 10,
            "attack_n": 5,
            "fp": 1,
            "tn": 9,
            "tp": 4,
            "fn": 1,
            "fpr": 0.1,
            "tpr": 0.8,
            "precision": 0.8,
            "f1": 0.8,
            "balanced_accuracy": 0.85,
            "auroc": 0.9,
            "auprc": 0.85,
            "band_error": 0.01,
            "attack_balanced_tpr": 0.8,
        }
    )
    RunLayout(run_dir).metric_records.write_text(
        valid_line + "\n" + "not valid json\n", encoding="utf-8"
    )
    write_fake_evidence(outputs_root)
    destination = ResultsBuilder().build(
        campaign_id="c10",
        outputs_root=outputs_root,
        results_root=tmp_path / "results",
    )
    payload = json.loads((destination / "metrics" / "metric_records.json").read_text())
    run_ids = {record["run_id"] for record in payload["records"]}
    assert "run-1" in run_ids


def test_copy_tables_and_figures_handles_missing_figures_directory(tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    layout = OutputsLayout(outputs_root)
    run_layout = layout.run("run_1")
    run_layout.metrics.mkdir(parents=True)
    run_layout.metric_records.write_text('{"run_id": "run_1"}\n', encoding="utf-8")
    layout.cache_analysis.mkdir(parents=True)
    layout.readiness_plans_file.write_text('{"plans": []}\n', encoding="utf-8")
    publication = layout.publication
    publication.tables.mkdir(parents=True)
    (publication.tables / "table_1.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    destination = ResultsBuilder().build(
        campaign_id="c11",
        outputs_root=outputs_root,
        results_root=tmp_path / "results",
    )
    assert (destination / "tables" / "table_1.csv").is_file()
    assert list((destination / "figures").iterdir()) == []


def test_results_verifier_detects_missing_manifest_and_checksums(tmp_path: Path) -> None:
    destination = tmp_path / "results" / "c12"
    layout = ResultsBundleLayout(destination)
    for directory in layout.required_directories:
        directory.mkdir(parents=True, exist_ok=True)
    result = ResultsVerifier().verify("c12", results_root=tmp_path / "results")
    assert not result.valid
    assert "missing bundle manifest.json" in result.problems
    assert "missing bundle checksums.json" in result.problems
