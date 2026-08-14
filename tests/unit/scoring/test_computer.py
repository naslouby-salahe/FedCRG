from __future__ import annotations

import hashlib

import numpy as np
from pydantic import TypeAdapter

from fedcrg.config import AutoencoderConfig
from fedcrg.learning.detectors import Autoencoder
from fedcrg.learning.scores import ClientScoreInput, RoleScoreInput, ScoreComputer
from fedcrg.types import ClientId, ComputeDeviceId, DataRole, DatasetId

_CLIENT_ID_ADAPTER = TypeAdapter(ClientId)
_SOME_HASH = "a" * 64


def test_score_computer_emits_float64_manifest() -> None:
    model = Autoencoder(2, AutoencoderConfig(hidden_dims=(1,), xavier_tanh_gain=5.0 / 3.0))
    client_id = _CLIENT_ID_ADAPTER.validate_python("c1")
    row_ids = tuple(hashlib.sha256(f"row-{index}".encode()).hexdigest() for index in range(3))
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
        model_seed=11,
        data_spec_hash=_SOME_HASH,
        training_spec_hash=_SOME_HASH,
        dataset_manifest_hash=_SOME_HASH,
        preprocessing_hash=_SOME_HASH,
        clients=(client,),
        device=ComputeDeviceId.CPU,
        batch_size=4,
    )
    scores = manifest.client(client_id).get(DataRole.TRAIN)
    assert scores.values.dtype == np.float64
    assert len(scores.sha256) == 64
