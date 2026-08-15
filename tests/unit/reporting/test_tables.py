from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from fedcrg.evidence.models import (
    ClientDatasetManifest,
    PreparedDatasetManifest,
    RoleArtifactManifest,
)
from fedcrg.evidence.models import ThresholdRecord
from fedcrg.evidence.store import atomic_write_json
from fedcrg.paths import RunLayout
from fedcrg.reporting import PublicationTableBuilder
from fedcrg.types import (
    DataRole,
    DatasetId,
    DecisionReason,
    DecisionState,
    ExperimentId,
    PolicyId,
    ThresholdSource,
)

from tests._fixtures import primary_experiment_config
from tests.unit.reporting._run_fixtures import write_completed_run, write_primary_policy_grid


def test_primary_policy_results_table_reflects_federation_metrics(tmp_path: Path) -> None:
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
        mebe=0.05,
    )
    output = PublicationTableBuilder().primary_policy_results((run_dir,), tmp_path / "t3.csv")
    frame = pd.read_csv(output)
    assert len(frame) == 1
    assert frame.loc[0, "run_id"] == "run-1"
    assert frame.loc[0, "mebe"] == pytest.approx(0.05)


def test_primary_policy_results_table_empty_when_no_runs(tmp_path: Path) -> None:
    output = PublicationTableBuilder().primary_policy_results((), tmp_path / "t3.csv")
    assert output.is_file()
    assert output.read_text(encoding="utf-8").strip() == ""


def _threshold_record(run_id: str, client_id: str) -> ThresholdRecord:
    return ThresholdRecord(
        run_id=run_id,
        policy_id=PolicyId.FEDCRG,
        client_id=client_id,
        tau_ref=0.5,
        tau_local=0.4,
        selected_tau=0.4,
        readiness_n=100,
        readiness_rank=95,
        readiness_probability=0.96,
        mismatch_n=50,
        mismatch_x=2,
        cp_lower=0.01,
        cp_upper=0.05,
        p_low=0.2,
        p_high=0.8,
        state=DecisionState.PERSONALIZED,
        tie_count=1,
        selected_source=ThresholdSource.LOCAL_CALIBRATION,
        reason_code=DecisionReason.LOCAL_PERSONALIZATION_ADMITTED,
    )


def test_admission_states_from_runs_reads_threshold_records(tmp_path: Path) -> None:
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
    )
    records = [_threshold_record("run-1", "nb01"), _threshold_record("run-1", "nb02")]
    RunLayout(run_dir).threshold_records.write_text(
        "\n".join(record.model_dump_json() for record in records) + "\n",
        encoding="utf-8",
    )
    output = PublicationTableBuilder().admission_states_from_runs((run_dir,), tmp_path / "t4.csv")
    frame = pd.read_csv(output)
    assert len(frame) == 2
    assert set(frame["client_id"]) == {"nb01", "nb02"}


def test_admission_states_from_runs_skips_runs_without_threshold_file(tmp_path: Path) -> None:
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
    )
    output = PublicationTableBuilder().admission_states_from_runs((run_dir,), tmp_path / "t4.csv")
    assert output.is_file()
    assert output.read_text(encoding="utf-8").strip() == ""


def test_protocol_constants_table_includes_training_rows_when_training_present(
    tmp_path: Path,
) -> None:
    config = primary_experiment_config(tmp_path / "outputs")
    assert config.training is not None
    output = PublicationTableBuilder().protocol_constants(config, tmp_path / "t1.csv")
    frame = pd.read_csv(output)
    assert "rounds" in set(frame["constant"])
    assert "learning_rate_initial" in set(frame["constant"])


def test_protocol_constants_table_omits_training_rows_when_training_absent(
    tmp_path: Path,
) -> None:
    config = primary_experiment_config(tmp_path / "outputs").model_copy(update={"training": None})
    output = PublicationTableBuilder().protocol_constants(config, tmp_path / "t1.csv")
    frame = pd.read_csv(output)
    assert "rounds" not in set(frame["constant"])
    assert "alpha" in set(frame["constant"])


def _prepared_manifest(root: Path) -> Path:
    role = RoleArtifactManifest(
        role=DataRole.TRAIN,
        rows=10,
        row_id_sha256="a" * 64,
        relative_path="clients/nb01/train.csv",
        file_sha256="b" * 64,
    )
    client = ClientDatasetManifest(client_id="nb01", roles=(role,))
    manifest = PreparedDatasetManifest(
        dataset_id=DatasetId.NBAIOT,
        source_version="1",
        parser_version="1",
        data_spec_hash="c" * 64,
        feature_names=("f1", "f2"),
        clients=(client,),
        source_files=(),
        calibration_assignments=(),
        external_replication_supported=False,
        dataset_level_code=None,
        created_at=datetime.now(UTC),
    )
    path = root / "manifest.json"
    atomic_write_json(path, manifest.model_dump(mode="json"))
    return path


def test_dataset_inventory_table_expands_role_counts(tmp_path: Path) -> None:
    manifest_path = _prepared_manifest(tmp_path)
    output = PublicationTableBuilder().dataset_inventory(manifest_path, tmp_path / "t2.csv")
    frame = pd.read_csv(output)
    assert len(frame) == 1
    assert frame.loc[0, "client_id"] == "nb01"
    assert frame.loc[0, "feature_count"] == 2


def test_federation_results_table_reads_available_metrics_only(tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    config = primary_experiment_config(outputs_root)
    with_metrics = write_completed_run(
        outputs_root,
        run_id="run-with",
        experiment_id=ExperimentId.PRIMARY_NBAIOT,
        policy_id=PolicyId.FEDCRG,
        config=config,
        model_seed=11,
        calibration_seed=1000,
    )
    without_metrics = write_completed_run(
        outputs_root,
        run_id="run-without",
        experiment_id=ExperimentId.PRIMARY_NBAIOT,
        policy_id=PolicyId.FEDCRG,
        config=config,
        model_seed=22,
        calibration_seed=1000,
        include_federation_metrics=False,
    )
    output = PublicationTableBuilder().federation_results(
        (with_metrics, without_metrics), tmp_path / "policy.csv"
    )
    frame = pd.read_csv(output)
    assert len(frame) == 1
    assert frame.loc[0, "run_id"] == "run-with"


def test_ablations_table_computes_contrasts_for_all_comparators(tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    config = primary_experiment_config(outputs_root).model_copy(
        update={
            "randomness": primary_experiment_config(outputs_root).randomness.model_copy(
                update={"model_seeds": (11, 22)}
            ),
            "statistics": primary_experiment_config(outputs_root).statistics.model_copy(
                update={"bootstrap_replicates": 200}
            ),
            "policies": (
                PolicyId.FEDCRG,
                PolicyId.GLOBAL_QUANTILE,
                PolicyId.LOCAL_QUANTILE,
                PolicyId.READINESS_ONLY,
                PolicyId.SHRINKAGE,
            ),
        }
    )
    write_primary_policy_grid(outputs_root, config, policies=config.policies)
    run_dirs = tuple(sorted((outputs_root / "runs").iterdir()))
    output = PublicationTableBuilder().ablations(run_dirs, tmp_path / "t5.csv", config)
    frame = pd.read_csv(output)
    assert not frame.empty
    assert set(frame["comparator"]) == {
        PolicyId.GLOBAL_QUANTILE,
        PolicyId.LOCAL_QUANTILE,
        PolicyId.READINESS_ONLY,
        PolicyId.SHRINKAGE,
    }
    assert set(frame["metric"]) == {"mebe", "high_excess", "attack_balanced_macro_tpr"}


def test_protocol_constants_table_is_deterministic(tmp_path: Path) -> None:
    config = primary_experiment_config(tmp_path / "outputs")
    first = PublicationTableBuilder().protocol_constants(config, tmp_path / "first.csv")
    second = PublicationTableBuilder().protocol_constants(config, tmp_path / "second.csv")
    assert first.read_bytes() == second.read_bytes()
    frame = pd.read_csv(first)
    assert not frame.empty
    assert "constant" in frame.columns
