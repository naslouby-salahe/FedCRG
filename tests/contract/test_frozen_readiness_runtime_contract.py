from __future__ import annotations

from pathlib import Path

import pytest

from fedcrg.domain.values import OperatingBand
from fedcrg.decision.calibration_readiness import ReadinessPlanCache


def test_runtime_lookup_never_materializes_missing_readiness_plan(tmp_path: Path) -> None:
    path = tmp_path / "readiness_plans.json"
    cache = ReadinessPlanCache(path)
    band = OperatingBand(0.005, 0.015)

    with pytest.raises(FileNotFoundError):
        cache.require(2000, band, 0.95)
    assert not path.exists()


def test_precompute_persists_then_runtime_requires_exact_contract(tmp_path: Path) -> None:
    path = tmp_path / "readiness_plans.json"
    band = OperatingBand(0.005, 0.015)
    writer = ReadinessPlanCache(path)
    plan = writer.precompute(2000, band, 0.95)

    reader = ReadinessPlanCache(path)
    assert reader.require(2000, band, 0.95) == plan
    with pytest.raises(FileNotFoundError):
        reader.require(2000, band, 0.99)
