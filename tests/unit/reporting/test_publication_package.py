from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from fedcrg.evidence.models import (
    ClientDatasetManifest,
    PreparedDatasetManifest,
    RoleArtifactManifest,
)
from fedcrg.evidence.store import atomic_write_json
from fedcrg.paths import RunLayout
from fedcrg.reporting import PublicationPackageBuilder, build_publication
from fedcrg.types import (
    DataRole,
    DatasetId,
    DecisionReason,
    DecisionState,
    PolicyId,
    ThresholdSource,
)

from tests._fixtures import primary_experiment_config
from tests.unit.reporting._run_fixtures import write_completed_run
from fedcrg.evidence.models import ThresholdRecord
from fedcrg.types import ExperimentId


def test_publication_package_marks_optional_tables_unavailable_without_evidence(
    tmp_path: Path,
) -> None:
    outputs_root = tmp_path / "outputs"
    config = primary_experiment_config(outputs_root)
    package = PublicationPackageBuilder().build(
        config=config,
        outputs_root=outputs_root,
        prepared_manifest=None,
        destination=tmp_path / "publication",
    )
    assert not package.complete
    table_names = {table.name: table for table in package.tables}
    assert table_names["Table 1 - Protocol constants"].available
    assert not table_names["Table 2 - Dataset inventory"].available
    assert table_names["Table 2 - Dataset inventory"].reason == (
        "prepared dataset manifest is unavailable"
    )
    assert not table_names["Table 3 - Primary policy results"].available
    assert not table_names["Table 4 - Admission states"].available
    assert not table_names["Table 5 - Ablations"].available
    assert package.manifest.is_file()
    manifest_payload = json.loads(package.manifest.read_text(encoding="utf-8"))
    assert manifest_payload["complete"] is False
    assert len(manifest_payload["tables"]) == 5
    assert len(manifest_payload["figures"]) == 8


def test_publication_package_builds_catalogue_driven_figures(tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    config = primary_experiment_config(outputs_root)
    package = PublicationPackageBuilder().build(
        config=config,
        outputs_root=outputs_root,
        destination=tmp_path / "publication",
    )
    figure_names = {figure.name: figure for figure in package.figures}
    assert figure_names["Figure 1 - Decision architecture"].available
    assert figure_names["Figure 2 - Finite-sample readiness frontier"].available
    assert figure_names["Figure 3 - Reference-mismatch evidence/power map"].available
    assert figure_names["Figure 6 - Calibration-size phase transition"].available
    assert not figure_names["Figure 4 - Per-client operating points"].available
    assert not figure_names["Figure 5 - Reliability-utility frontier"].available
    assert not figure_names["Figure 7 - Assumption stress"].available
    assert not figure_names["Figure 8 - External replication"].available
    for figure in package.figures:
        if figure.available:
            assert figure.path is not None
            assert figure.path.is_file()
            assert figure.path.stat().st_size > 0


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


def test_publication_package_marks_optional_tables_available_with_full_evidence(
    tmp_path: Path,
) -> None:
    outputs_root = tmp_path / "outputs"
    config = primary_experiment_config(outputs_root)
    run_dir = write_completed_run(
        outputs_root,
        run_id="run-1",
        experiment_id=ExperimentId.PRIMARY_NBAIOT,
        policy_id=PolicyId.FEDCRG,
        config=config,
        model_seed=11,
        calibration_seed=config.dataset.primary_calibration_seed,
    )
    record = ThresholdRecord(
        run_id="run-1",
        policy_id=PolicyId.FEDCRG,
        client_id="nb01",
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
    RunLayout(run_dir).threshold_records.write_text(
        record.model_dump_json() + "\n", encoding="utf-8"
    )
    manifest_path = _prepared_manifest(tmp_path)
    package = PublicationPackageBuilder().build(
        config=config,
        outputs_root=outputs_root,
        prepared_manifest=manifest_path,
        destination=tmp_path / "publication",
    )
    table_names = {table.name: table for table in package.tables}
    assert table_names["Table 2 - Dataset inventory"].available
    assert table_names["Table 3 - Primary policy results"].available
    assert table_names["Table 4 - Admission states"].available


def test_build_publication_returns_manifest_path(tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    config = primary_experiment_config(outputs_root)
    manifest_path = build_publication(config, outputs_root, destination=tmp_path / "publication")
    assert manifest_path.is_file()
    assert manifest_path.name == "manifest.json"
