from __future__ import annotations

import json
import shutil
from pathlib import Path

from fedcrg.paths import ExperimentResultsBundleLayout, OutputsLayout, RunLayout
from fedcrg.reporting import (
    ExperimentPublisher,
    ResultsBuilder,
    ResultsVerifier,
    build_experiment_results_bundle,
    verify_results_bundle,
)
from fedcrg.types import ExperimentId, PolicyId, ResultsGenerationStatus

from fedcrg.experiments.analyses import MismatchPowerResult, SyntheticExperimentEnvelope
from tests._fixtures import primary_experiment_config
from tests.unit.reporting._run_fixtures import write_completed_run


def _publish(experiment_id: ExperimentId, outputs_root: Path) -> None:
    config = primary_experiment_config(outputs_root, experiment_id)
    ExperimentPublisher().publish(experiment_id, outputs_root, config)


def test_experiment_bundle_contains_run_artifacts_and_round_trips(tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    results_root = tmp_path / "results"
    config = primary_experiment_config(outputs_root)
    write_completed_run(
        outputs_root,
        run_id="run-1",
        experiment_id=ExperimentId.PRIMARY_NBAIOT,
        policy_id=PolicyId.FEDCRG,
        config=config,
        model_seed=11,
        calibration_seed=1000,
    )
    _publish(ExperimentId.PRIMARY_NBAIOT, outputs_root)
    destination = ResultsBuilder().build(
        outputs_root=outputs_root,
        results_root=results_root,
        experiment_id=ExperimentId.PRIMARY_NBAIOT,
    )
    assert destination == results_root / "experiments" / "primary_nbaiot"
    assert (destination / "manifest.json").is_file()
    assert (destination / "checksums.json").is_file()
    assert (destination / "runs" / "run-1" / "manifest.json").is_file()
    assert (destination / "json" / "metric_records.json").is_file()
    assert (destination / "provenance" / "resolved_configs" / "primary_nbaiot.json").is_file()

    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment_id"] == "primary_nbaiot"
    verification = verify_results_bundle(results_root, experiment_id=ExperimentId.PRIMARY_NBAIOT)
    assert verification.valid
    assert verification.problems == ()


def test_experiment_bundle_is_idempotent(tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    config = primary_experiment_config(outputs_root)
    write_completed_run(
        outputs_root,
        run_id="run-1",
        experiment_id=ExperimentId.PRIMARY_NBAIOT,
        policy_id=PolicyId.FEDCRG,
        config=config,
        model_seed=11,
        calibration_seed=1000,
    )
    _publish(ExperimentId.PRIMARY_NBAIOT, outputs_root)
    builder = ResultsBuilder()
    first = builder.build_with_status(
        outputs_root=outputs_root,
        results_root=tmp_path / "results",
        experiment_id=ExperimentId.PRIMARY_NBAIOT,
    )
    second = builder.build_with_status(
        outputs_root=outputs_root,
        results_root=tmp_path / "results",
        experiment_id=ExperimentId.PRIMARY_NBAIOT,
    )
    assert first.status is ResultsGenerationStatus.BUILT
    assert second.status is ResultsGenerationStatus.ALREADY_GENERATED


def test_experiment_bundle_for_synthetic_copies_analysis(tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    analysis = OutputsLayout(outputs_root).analysis_result(ExperimentId.MISMATCH_POWER)
    analysis.parent.mkdir(parents=True)
    envelope = SyntheticExperimentEnvelope(
        experiment_id=ExperimentId.MISMATCH_POWER,
        expected_monte_carlo_trials=0,
        expected_exact_cells=1,
        actual_monte_carlo_trials=0,
        actual_cells=1,
        cells=(MismatchPowerResult(sample_count=736, true_fpr=0.02, declaration_probability=0.8),),
    )
    analysis.write_text(envelope.model_dump_json(), encoding="utf-8")
    _publish(ExperimentId.MISMATCH_POWER, outputs_root)
    destination = build_experiment_results_bundle(
        ExperimentId.MISMATCH_POWER, outputs_root, tmp_path / "results"
    )
    assert (destination / "json" / "results.json").is_file()
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment_id"] == "mismatch_power"
    assert manifest["complete"] is True
    assert verify_results_bundle(
        tmp_path / "results", experiment_id=ExperimentId.MISMATCH_POWER
    ).valid


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
    _publish(ExperimentId.PRIMARY_NBAIOT, outputs_root)
    destination = ResultsBuilder().build(
        outputs_root=outputs_root,
        results_root=tmp_path / "results",
        experiment_id=ExperimentId.PRIMARY_NBAIOT,
    )
    payload = json.loads((destination / "json" / "metric_records.json").read_text())
    run_ids = {record["run_id"] for record in payload["records"]}
    assert "run-1" in run_ids


def test_results_verifier_detects_checksum_mismatch(tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    config = primary_experiment_config(outputs_root)
    write_completed_run(
        outputs_root,
        run_id="run-1",
        experiment_id=ExperimentId.PRIMARY_NBAIOT,
        policy_id=PolicyId.FEDCRG,
        config=config,
        model_seed=11,
        calibration_seed=1000,
    )
    _publish(ExperimentId.PRIMARY_NBAIOT, outputs_root)
    destination = ResultsBuilder().build(
        outputs_root=outputs_root,
        results_root=tmp_path / "results",
        experiment_id=ExperimentId.PRIMARY_NBAIOT,
    )
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    result = ResultsVerifier().verify(
        tmp_path / "results", experiment_id=ExperimentId.PRIMARY_NBAIOT
    )
    assert not result.valid
    assert any("hash mismatch" in problem for problem in result.problems)


def test_results_verifier_detects_unchecksummed_file(tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    config = primary_experiment_config(outputs_root)
    write_completed_run(
        outputs_root,
        run_id="run-1",
        experiment_id=ExperimentId.PRIMARY_NBAIOT,
        policy_id=PolicyId.FEDCRG,
        config=config,
        model_seed=11,
        calibration_seed=1000,
    )
    _publish(ExperimentId.PRIMARY_NBAIOT, outputs_root)
    destination = ResultsBuilder().build(
        outputs_root=outputs_root,
        results_root=tmp_path / "results",
        experiment_id=ExperimentId.PRIMARY_NBAIOT,
    )
    (destination / "csv" / "extra_untracked.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    result = ResultsVerifier().verify(
        tmp_path / "results", experiment_id=ExperimentId.PRIMARY_NBAIOT
    )
    assert not result.valid
    assert any("unchecksummed" in problem for problem in result.problems)


def test_results_verifier_detects_missing_required_directory(tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    config = primary_experiment_config(outputs_root)
    write_completed_run(
        outputs_root,
        run_id="run-1",
        experiment_id=ExperimentId.PRIMARY_NBAIOT,
        policy_id=PolicyId.FEDCRG,
        config=config,
        model_seed=11,
        calibration_seed=1000,
    )
    _publish(ExperimentId.PRIMARY_NBAIOT, outputs_root)
    destination = ResultsBuilder().build(
        outputs_root=outputs_root,
        results_root=tmp_path / "results",
        experiment_id=ExperimentId.PRIMARY_NBAIOT,
    )
    layout = ExperimentResultsBundleLayout(destination)
    shutil.rmtree(layout.csv_dir)
    result = ResultsVerifier().verify(
        tmp_path / "results", experiment_id=ExperimentId.PRIMARY_NBAIOT
    )
    assert not result.valid
    assert any("missing required bundle directory" in problem for problem in result.problems)


def test_results_verifier_detects_missing_manifest_and_checksums(tmp_path: Path) -> None:
    destination = tmp_path / "results" / "experiments" / "primary_nbaiot"
    layout = ExperimentResultsBundleLayout(destination)
    for directory in layout.required_directories:
        directory.mkdir(parents=True, exist_ok=True)
    result = ResultsVerifier().verify(
        tmp_path / "results", experiment_id=ExperimentId.PRIMARY_NBAIOT
    )
    assert not result.valid
    assert "missing bundle manifest.json" in result.problems
    assert "missing bundle checksums.json" in result.problems


def test_results_verifier_detects_missing_bundle(tmp_path: Path) -> None:
    result = ResultsVerifier().verify(
        tmp_path / "results", experiment_id=ExperimentId.PRIMARY_NBAIOT
    )
    assert not result.valid
    assert any("does not exist" in problem for problem in result.problems)


def test_verify_missing_experiment_bundle(tmp_path: Path) -> None:
    verification = verify_results_bundle(
        tmp_path / "results", experiment_id=ExperimentId.EXTERNAL_DIAD
    )
    assert not verification.valid
    assert any("does not exist" in problem for problem in verification.problems)
