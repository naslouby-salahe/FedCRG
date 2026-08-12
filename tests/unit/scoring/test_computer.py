import numpy as np

from fedcrg.config.models import AutoencoderConfig
from fedcrg.core.enums import DataRole, DatasetId
from fedcrg.detectors.autoencoder import Autoencoder
from fedcrg.scoring.computer import ScoreComputer


def test_score_computer_emits_float64_manifest() -> None:
    model = Autoencoder(2, AutoencoderConfig(hidden_dims=(1,)))
    manifest = ScoreComputer().compute_manifest(model, DatasetId.NBAIOT, 11, {"c1": {DataRole.REFERENCE: np.zeros((3, 2), dtype=np.float32)}})
    scores = manifest.clients["c1"].scores[DataRole.REFERENCE]
    assert scores.values.dtype == np.float64
    assert len(scores.sha256) == 64
