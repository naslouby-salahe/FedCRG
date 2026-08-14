from __future__ import annotations

from pathlib import Path

from fedcrg.config import DatasetConfig, ExperimentConfig, Study
from fedcrg.evidence.store import build_run_id
from fedcrg.types import (
    DatasetFeatureContractId,
    ExperimentId,
    PolicyId,
)


def _external_config() -> ExperimentConfig:
    return Study.load().resolve(ExperimentId.EXTERNAL_DIAD)


def _r14_derived_config() -> ExperimentConfig:
    """Derive the R14 training-schema-only config after the feature freeze."""
    external = _external_config()
    derived_dataset = DatasetConfig.model_validate(
        external.dataset.model_dump(mode="python")
        | {
            "feature_contract": DatasetFeatureContractId.DIAD_TRAINING_NUMERIC_SAFE,
            "feature_count": 60,
            "feature_names": tuple(f"f{index}" for index in range(1, 61)),
            "calibration_seeds": (external.dataset.primary_calibration_seed,),
        }
    )
    derived = ExperimentConfig.model_validate(
        external.model_dump(mode="python")
        | {"id": ExperimentId.DIAD_FEATURE_SENSITIVITY, "dataset": derived_dataset}
    )
    return derived


def test_run_id_maps_distinct_configs_to_distinct_evidence_directories() -> None:
    external = _external_config()
    r14 = _r14_derived_config()

    confirmatory_id = build_run_id(external, 11, 2000, PolicyId.FEDCRG)
    sensitivity_id = build_run_id(r14, 11, 2000, PolicyId.FEDCRG)

    assert external.config_hash != r14.config_hash
    assert confirmatory_id != sensitivity_id


def test_config_hash_does_not_depend_on_output_directory(tmp_path: Path) -> None:
    external = _external_config()
    relocated = external.model_copy(update={"outputs_root": tmp_path / "elsewhere"})

    assert external.config_hash == relocated.config_hash
