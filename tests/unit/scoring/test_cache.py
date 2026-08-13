import hashlib
from pathlib import Path

import numpy as np
import pytest

from fedcrg.core.enums import DataRole, DatasetId
from fedcrg.core.ids import ClientId, ModelSeed, RowId, Sha256
from fedcrg.scoring.cache import ScoreCache
from fedcrg.scoring.models import ClientScoreSet, RoleScores, ScoreManifest

_CLIENT = ClientId("c1")
_SOME_HASH = Sha256("a" * 64)


def _row_ids(role: DataRole, count: int) -> tuple[RowId, ...]:
    return tuple(
        RowId(hashlib.sha256(f"{role.value}-{index}".encode()).hexdigest())
        for index in range(count)
    )


def _role_scores(role: DataRole, count: int) -> RoleScores:
    return RoleScores(
        role=role,
        values=np.linspace(0.0, 1.0, count),
        client_id=_CLIENT,
        row_ids=_row_ids(role, count),
    )


def _manifest() -> ScoreManifest:
    scores = {
        role: _role_scores(role, 5)
        for role in (
            DataRole.TRAIN,
            DataRole.RESERVOIR,
            DataRole.BENIGN_TEST,
            DataRole.ATTACK_DEV,
            DataRole.ATTACK_TEST,
        )
    }
    return ScoreManifest(
        dataset=DatasetId.NBAIOT,
        model_seed=ModelSeed(11),
        model_hash=_SOME_HASH,
        data_spec_hash=_SOME_HASH,
        training_spec_hash=_SOME_HASH,
        dataset_manifest_hash=_SOME_HASH,
        preprocessing_hash=_SOME_HASH,
        clients={_CLIENT: ClientScoreSet(_CLIENT, scores)},
    )


def test_score_cache_round_trip_verifies_hashes(tmp_path: Path) -> None:
    manifest = _manifest()
    cache = ScoreCache()
    root = tmp_path / "cache"
    cache.save(manifest, root)
    loaded = cache.load(root)
    assert loaded.role_hashes() == manifest.role_hashes()


def test_score_cache_detects_tampering(tmp_path: Path) -> None:
    manifest = _manifest()
    cache = ScoreCache()
    root = tmp_path / "cache"
    cache.save(manifest, root)
    parquet_path = root / cache.filename
    with open(parquet_path, "r+b") as handle:
        handle.seek(0)
        handle.write(b"\x00" * min(16, parquet_path.stat().st_size))
    with pytest.raises(ValueError, match="(?i)hash.mismatch"):
        cache.load(root)
