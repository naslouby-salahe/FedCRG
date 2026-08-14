from __future__ import annotations

from pathlib import Path

from fedcrg.paths import RunLayout
from fedcrg.reporting import build_run_report
from fedcrg.types import ExperimentId, PolicyId

from tests.unit.reporting._run_fixtures import write_completed_run


def test_build_run_report_without_federation_metrics(tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    run_dir = write_completed_run(
        outputs_root,
        run_id="run-a",
        experiment_id=ExperimentId.PRIMARY_NBAIOT,
        policy_id=PolicyId.FEDCRG,
        config=None,
        model_seed=11,
        calibration_seed=1000,
        include_run_config=False,
        include_federation_metrics=False,
    )
    output = build_run_report(run_dir)
    assert output.is_file()
    text = output.read_text(encoding="utf-8")
    assert "run-a" in text
    assert "Federation endpoints" not in text
    assert "Verified artifact hashes: `False`" in text
    assert "Evaluated clients: `0`" in text


def test_build_run_report_with_federation_metrics(tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    run_dir = write_completed_run(
        outputs_root,
        run_id="run-b",
        experiment_id=ExperimentId.PRIMARY_NBAIOT,
        policy_id=PolicyId.FEDCRG,
        config=None,
        model_seed=22,
        calibration_seed=1000,
        include_run_config=False,
    )
    RunLayout(run_dir).metric_records.write_text(
        '{"a": 1}\n{"b": 2}\n\n',
        encoding="utf-8",
    )
    output = build_run_report(run_dir)
    text = output.read_text(encoding="utf-8")
    assert "Federation endpoints" in text
    assert "MEBE: `0.02`" in text
    assert "Evaluated clients: `2`" in text
