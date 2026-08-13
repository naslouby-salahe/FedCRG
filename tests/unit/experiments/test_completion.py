import json
from pathlib import Path

from fedcrg.core.enums import ExperimentCode
from fedcrg.experiments.completion import ExperimentCompletionAuditor


def _write_cell(root: Path, code: ExperimentCode, name: str, payload: dict[str, object]) -> None:
    cells_root = root / "experiments" / code.value / "cells"
    cells_root.mkdir(parents=True, exist_ok=True)
    (cells_root / name).write_text(json.dumps(payload), encoding="utf-8")


def test_real_sensitivity_workload_reads_sensitivity_envelope_schema(tmp_path: Path) -> None:
    """R2-R6/R9 write SensitivityEnvelope(protocol_code, model_seed, calibration_seed, cells)."""
    for model_seed in (11, 22, 33, 44, 55):
        _write_cell(
            tmp_path,
            ExperimentCode.R2,
            f"{model_seed}.json",
            {
                "experiment_id": "readiness_sample_size",
                "protocol_code": "R2",
                "model_seed": model_seed,
                "calibration_seed": 1000,
                "cells": [{"settings": [], "config_hash": "x", "evaluation": {}}],
            },
        )
    result = ExperimentCompletionAuditor._real_sensitivity_workload(
        tmp_path,
        ExperimentCode.R2,
        expected_model_seeds=(11, 22, 33, 44, 55),
        expected_calibration_seed=1000,
    )
    assert result.complete
    assert result.observed_cells == 5
    assert not result.problems


def test_real_sensitivity_workload_detects_missing_model_seed(tmp_path: Path) -> None:
    _write_cell(
        tmp_path,
        ExperimentCode.R2,
        "11.json",
        {
            "protocol_code": "R2",
            "model_seed": 11,
            "calibration_seed": 1000,
            "cells": [{"settings": []}],
        },
    )
    result = ExperimentCompletionAuditor._real_sensitivity_workload(
        tmp_path,
        ExperimentCode.R2,
        expected_model_seeds=(11, 22, 33, 44, 55),
        expected_calibration_seed=1000,
    )
    assert not result.complete
    assert any("missing" in problem for problem in result.problems)


def test_single_seed_sensitivity_workload_reads_multiplicity_envelope_schema(
    tmp_path: Path,
) -> None:
    """R7/R8 write MultiplicityEnvelope/SourceOrderEnvelope with no model-seed axis."""
    _write_cell(
        tmp_path,
        ExperimentCode.R7,
        "1000.json",
        {
            "experiment_id": "multiplicity_sensitivity",
            "protocol_code": "R7",
            "calibration_seed": 1000,
            "cells": [{"procedure": "bonferroni_readiness"}],
        },
    )
    result = ExperimentCompletionAuditor._single_seed_sensitivity_workload(
        tmp_path,
        ExperimentCode.R7,
        expected_calibration_seed=1000,
    )
    assert result.complete
    assert result.observed_cells == 1


def test_source_order_workload_reads_run_source_order_calibration_schema(tmp_path: Path) -> None:
    """R12's real producer (application/source_order.py) writes this exact schema."""
    for dataset, model_seed in [("nbaiot", model_seed) for model_seed in (11, 22, 33, 44, 55)] + [
        ("diad", model_seed) for model_seed in (11, 22, 33, 44, 55)
    ]:
        calibration_seed = 1000 if dataset == "nbaiot" else 2000
        _write_cell(
            tmp_path,
            ExperimentCode.R12,
            f"{dataset}_{model_seed}.json",
            {
                "experiment": "R12",
                "complete": True,
                "dataset_id": dataset,
                "model_seed": model_seed,
                "calibration_seed": calibration_seed,
                "calibration_assignment": "source_order",
                "score_cache_sha256": "a" * 64,
                "evaluation": {},
            },
        )
    result = ExperimentCompletionAuditor._source_order_workload(tmp_path)
    assert result.complete
    assert result.observed_cells == 10
    assert not result.problems
