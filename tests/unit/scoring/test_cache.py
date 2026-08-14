from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from pydantic import TypeAdapter

from fedcrg.learning.scores import ClientScoreSet, RoleScores, ScoreCache, ScoreManifest
from fedcrg.paths import ScoreCacheLayout
from fedcrg.types import ClientId, DataRole, DatasetId

_CLIENT_ID_ADAPTER = TypeAdapter(ClientId)
_CLIENT = _CLIENT_ID_ADAPTER.validate_python("c1")
_SOME_HASH = "a" * 64
_ROLES = (
    DataRole.TRAIN,
    DataRole.RESERVOIR,
    DataRole.BENIGN_TEST,
    DataRole.ATTACK_DEV,
    DataRole.ATTACK_TEST,
)


def _row_ids(role: DataRole, count: int) -> tuple[str, ...]:
    return tuple(
        hashlib.sha256(f"{role.value}-{index}".encode()).hexdigest() for index in range(count)
    )


def _role_scores(role: DataRole, count: int) -> RoleScores:
    return RoleScores(
        role=role,
        values=np.linspace(0.0, 1.0, count),
        client_id=_CLIENT,
        row_ids=_row_ids(role, count),
    )


def _manifest() -> ScoreManifest:
    scores = tuple(_role_scores(role, 5) for role in _ROLES)
    return ScoreManifest(
        dataset=DatasetId.NBAIOT,
        model_seed=11,
        model_hash=_SOME_HASH,
        data_spec_hash=_SOME_HASH,
        training_spec_hash=_SOME_HASH,
        dataset_manifest_hash=_SOME_HASH,
        preprocessing_hash=_SOME_HASH,
        clients=(ClientScoreSet(client_id=_CLIENT, scores=scores),),
    )


def test_score_cache_round_trip_verifies_hashes(tmp_path: Path) -> None:
    manifest = _manifest()
    cache = ScoreCache()
    root = tmp_path / "cache"
    cache.save(manifest, root)
    loaded = cache.load(root)
    loaded_by_role = {
        item.client_id: {role.role: role.sha256 for role in item.roles}
        for item in loaded.role_hashes()
    }
    saved_by_role = {
        item.client_id: {role.role: role.sha256 for role in item.roles}
        for item in manifest.role_hashes()
    }
    assert loaded_by_role == saved_by_role


def test_score_cache_detects_tampering(tmp_path: Path) -> None:
    manifest = _manifest()
    cache = ScoreCache()
    root = tmp_path / "cache"
    cache.save(manifest, root)
    parquet_path = ScoreCacheLayout(root).score
    with open(parquet_path, "r+b") as handle:
        handle.seek(0)
        handle.write(b"\x00" * min(16, parquet_path.stat().st_size))
    with pytest.raises(ValueError, match="(?i)hash.mismatch"):
        cache.load(root)
