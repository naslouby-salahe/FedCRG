from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fedcrg.config import Study
from fedcrg.reporting import RepositoryReportFilename, build_repository_report
from fedcrg.types import ExperimentId, PolicyId

from tests.unit.reporting._run_fixtures import write_primary_policy_grid


def test_build_repository_report_without_primary_runs_is_incomplete(tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    config = Study.load().resolve(ExperimentId.PRIMARY_NBAIOT)
    output = build_repository_report(outputs_root, config)
    assert output.is_file()
    reports_root = output.parent
    contrasts_path = reports_root / RepositoryReportFilename.PRIMARY_CONTRASTS
    payload = json.loads(contrasts_path.read_text(encoding="utf-8"))
    assert payload["status"] == "incomplete"
    text = output.read_text(encoding="utf-8")
    assert "Completed run directories: `0`" in text
    sensitivity_path = reports_root / RepositoryReportFilename.SPLIT_SENSITIVITY
    assert sensitivity_path.is_file()


def test_build_repository_report_with_primary_runs_computes_contrasts(tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    config = Study.load().resolve(ExperimentId.PRIMARY_NBAIOT)
    write_primary_policy_grid(outputs_root, config, policies=config.policies)
    output = build_repository_report(outputs_root, config)
    reports_root = output.parent
    contrasts_path = reports_root / RepositoryReportFilename.PRIMARY_CONTRASTS
    payload = json.loads(contrasts_path.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    comparators = {item["comparator"] for item in payload}
    assert comparators == {
        PolicyId.GLOBAL_QUANTILE,
        PolicyId.LOCAL_QUANTILE,
        PolicyId.READINESS_ONLY,
        PolicyId.SHRINKAGE,
    }
    policy_table_path = reports_root / RepositoryReportFilename.POLICY_FEDERATION_RESULTS
    frame = pd.read_csv(policy_table_path)
    assert len(frame) == len(config.policies) * len(config.randomness.model_seeds)
    text = output.read_text(encoding="utf-8")
    assert f"Completed run directories: `{len(frame)}`" in text
