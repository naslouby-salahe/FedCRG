from __future__ import annotations

from pathlib import Path

import pytest

from fedcrg.paths import OutputsLayout
from fedcrg.reporting import ResultsBuilder, ResultsVerifier, build_results_bundle


def _write_fake_evidence(outputs_root: Path) -> None:
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


@pytest.fixture()
def bundle(tmp_path: Path) -> Path:
    outputs_root = tmp_path / "outputs"
    _write_fake_evidence(outputs_root)
    return build_results_bundle("c1", outputs_root=outputs_root, results_root=tmp_path / "results")


def _verify(campaign_id: str, tmp_path: Path):
    return ResultsVerifier().verify(
        campaign_id,
        results_root=tmp_path / "results",
    )


def test_built_bundle_verifies_clean(bundle: Path, tmp_path: Path) -> None:
    result = _verify("c1", tmp_path)
    assert result.valid
    assert result.problems == ()


def test_bundle_detects_tampered_file(bundle: Path, tmp_path: Path) -> None:
    target = bundle / "tables" / "table_1.csv"
    target.write_text("a,b\n9,9\n", encoding="utf-8")
    result = _verify("c1", tmp_path)
    assert not result.valid
    assert any("hash mismatch" in problem for problem in result.problems)


def test_bundle_detects_extra_unchecksummed_file(bundle: Path, tmp_path: Path) -> None:
    (bundle / "tables" / "smuggled.csv").write_text("x\n1\n", encoding="utf-8")
    result = _verify("c1", tmp_path)
    assert not result.valid
    assert any("unchecksummed" in problem for problem in result.problems)


def test_bundle_detects_missing_required_directory(bundle: Path, tmp_path: Path) -> None:
    import shutil

    shutil.rmtree(bundle / "provenance")
    result = _verify("c1", tmp_path)
    assert not result.valid
    assert any("missing required bundle directory" in problem for problem in result.problems)


def test_bundle_build_refuses_overwrite(bundle: Path, tmp_path: Path) -> None:
    with pytest.raises(FileExistsError):
        ResultsBuilder().build(
            campaign_id="c1",
            outputs_root=tmp_path / "outputs",
            results_root=tmp_path / "results",
        )
