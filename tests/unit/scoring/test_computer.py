import hashlib

import numpy as np

from fedcrg.config.training_config import AutoencoderConfig
from fedcrg.domain.enums import DataRole, DatasetId
from fedcrg.domain.identifiers import ClientId, ModelSeed, RowId, Sha256
from fedcrg.detectors.autoencoder import Autoencoder
from fedcrg.scoring.compute import ScoreComputer
from fedcrg.scoring.calibration_scores import ClientScoreInput, RoleScoreInput

_SOME_HASH = Sha256("a" * 64)


def test_score_computer_emits_float64_manifest() -> None:
    model = Autoencoder(2, AutoencoderConfig(hidden_dims=(1,)))
    client_id = ClientId("c1")
    row_ids = tuple(
        RowId(hashlib.sha256(f"row-{index}".encode()).hexdigest()) for index in range(3)
    )
    client = ClientScoreInput(
        client_id=client_id,
        roles=(
            RoleScoreInput(
                role=DataRole.TRAIN,
                values=np.zeros((3, 2), dtype=np.float32),
                row_ids=row_ids,
            ),
        ),
    )
    manifest = ScoreComputer().compute_manifest(
        model,
        DatasetId.NBAIOT,
        ModelSeed(11),
        _SOME_HASH,
        _SOME_HASH,
        _SOME_HASH,
        _SOME_HASH,
        (client,),
    )
    scores = manifest.client(client_id).get(DataRole.TRAIN)
    assert scores.values.dtype == np.float64
    assert len(scores.sha256.value) == 64
