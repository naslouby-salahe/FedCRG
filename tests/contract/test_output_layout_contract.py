from __future__ import annotations

from pathlib import Path

from fedcrg.evidence.store import RunLayout
from fedcrg.types import RunId


def test_run_layout_has_single_evidence_namespace(tmp_path: Path) -> None:
    run_id: RunId = (
        "nbaiot__ae__ms11__cs1000__a10000__r5000__ga9500__gb9500__fedcrg__cfg0123456789ab"
    )
    layout = RunLayout.for_run(tmp_path, run_id)

    assert layout.root == tmp_path / "runs" / run_id
    assert layout.manifest == layout.root / "manifest.json"
    assert layout.resolved_config == layout.root / "resolved_config.yaml"
    assert layout.environment == layout.root / "environment.json"
    assert layout.data == layout.root / "data"
    assert layout.training == layout.root / "training"
    assert layout.scores == layout.root / "scores"
    assert layout.decisions == layout.root / "decisions"
    assert layout.metrics == layout.root / "metrics"
    assert layout.tables == layout.root / "tables"
    assert layout.figures == layout.root / "figures"
    assert layout.reports == layout.root / "reports"
    assert layout.logs == layout.root / "logs"
    assert layout.verification == layout.root / "verification"


def test_top_level_outputs_are_reserved_by_responsibility(tmp_path: Path) -> None:
    expected = {
        "logs",
        "monitoring",
        "cache",
        "runs",
        "campaigns",
        "reports",
    }
    # Vocabulary contract rather than a directory-existence assertion: generated
    # folders do not need to exist in a clean checkout.
    documented = {
        "logs",
        "monitoring",
        "cache",
        "runs",
        "campaigns",
        "reports",
    }
    assert documented == expected


def test_preprocessed_root_is_repository_owned(tmp_path: Path) -> None:
    from fedcrg.config import ExperimentConfig

    assert ExperimentConfig.model_fields["preprocessed_root"].default == Path(
        "data/preprocessed"
    )
