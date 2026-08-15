from __future__ import annotations

import json
from pathlib import Path

import pytest

from fedcrg.config import (
    ExperimentSpec,
    ParameterAxis,
    ParameterCell,
    Study,
)
from fedcrg.experiments.analyses import ProtocolTablePrecomputer
from fedcrg.thresholding.readiness import ReadinessPlanCache, minimum_bidirectional_sample_count
from fedcrg.types import (
    DatasetId,
    ExperimentAxisId,
    ExperimentId,
    ExperimentType,
    OperatingBand,
)


@pytest.fixture(scope="module")
def real_config():
    return Study.load().resolve(ExperimentId.READINESS_THEOREM)


def test_has_axis_reports_axis_presence() -> None:
    spec = ExperimentSpec(
        id=ExperimentId.READINESS_THEOREM,
        category=ExperimentType.SYNTHETIC,
        dataset=DatasetId.SYNTHETIC,
        axes=[ParameterAxis(id=ExperimentAxisId.CALIBRATION_N, values=[10, 20])],
    )
    assert ProtocolTablePrecomputer._has_axis(spec, ExperimentAxisId.CALIBRATION_N)
    assert not ProtocolTablePrecomputer._has_axis(spec, ExperimentAxisId.MISMATCH_N)


def test_precompute_writes_readiness_plans_and_mismatch_cutoffs(
    tmp_path: Path, real_config
) -> None:
    config = real_config.model_copy(update={"outputs_root": tmp_path})
    spec = ExperimentSpec(
        id=ExperimentId.READINESS_THEOREM,
        category=ExperimentType.SYNTHETIC,
        dataset=DatasetId.SYNTHETIC,
        axes=[
            ParameterAxis(id=ExperimentAxisId.CALIBRATION_N, values=[10, 20]),
            ParameterAxis(id=ExperimentAxisId.READINESS_ASSURANCE, values=[0.8]),
            ParameterAxis(id=ExperimentAxisId.MISMATCH_N, values=[5, 8]),
        ],
        coupled_cells=[ParameterCell.model_validate({"alpha": 0.05, "calibration_n": 15})],
    )
    readiness_path, mismatch_path = ProtocolTablePrecomputer().precompute(config, spec)

    assert readiness_path.is_file()
    assert mismatch_path.is_file()

    cache = ReadinessPlanCache(readiness_path)
    plan = cache.require(10, config.protocol.band, config.protocol.readiness_assurance)
    assert plan.sample_count == 10

    payload = json.loads(mismatch_path.read_text())
    assert payload["minimum_bidirectional_sample_count"] == minimum_bidirectional_sample_count(
        config.protocol.band.lower, config.protocol.mismatch_confidence
    )
    cells = {cell["sample_count"]: cell for cell in payload["cells"]}
    assert set(cells) == {5, 8}


def test_precompute_skips_mismatch_cutoffs_without_mismatch_axis(
    tmp_path: Path, real_config
) -> None:
    config = real_config.model_copy(update={"outputs_root": tmp_path})
    spec = ExperimentSpec(
        id=ExperimentId.READINESS_THEOREM,
        category=ExperimentType.SYNTHETIC,
        dataset=DatasetId.SYNTHETIC,
        axes=[ParameterAxis(id=ExperimentAxisId.CALIBRATION_N, values=[10])],
    )
    _, mismatch_path = ProtocolTablePrecomputer().precompute(config, spec)
    assert not mismatch_path.exists()


def test_mismatch_cutoffs_reports_none_when_no_exceedance_declares(real_config) -> None:
    band = OperatingBand(lower=0.0001, upper=0.9999)
    cell = ProtocolTablePrecomputer._mismatch_cutoffs(10, band, 0.95)
    assert cell.sample_count == 10
    assert cell.low_max_exceedances is None
    assert cell.high_min_exceedances is None


def test_mismatch_cutoffs_identifies_low_and_high_cutoffs(real_config) -> None:
    band = OperatingBand(lower=0.1, upper=0.2)
    cell = ProtocolTablePrecomputer._mismatch_cutoffs(50, band, 0.95)
    assert cell.sample_count == 50
    assert cell.low_max_exceedances is not None
    assert cell.high_min_exceedances is not None
    assert cell.low_max_exceedances < cell.high_min_exceedances
