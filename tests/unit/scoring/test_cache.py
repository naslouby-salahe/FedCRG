from pathlib import Path

import numpy as np

from fedcrg.core.enums import DataRole, DatasetId
from fedcrg.scoring.cache import ScoreCache
from fedcrg.scoring.models import ClientScoreSet, RoleScores, ScoreManifest


def test_score_cache_round_trip_verifies_hashes(tmp_path: Path) -> None:
    role = RoleScores(DataRole.REFERENCE, np.array([1.0, 2.0]), "c1")
    manifest = ScoreManifest(DatasetId.NBAIOT, 11, "model-hash", {"c1": ClientScoreSet("c1", {DataRole.REFERENCE: role})})
    cache = ScoreCache()
    cache.save(manifest, tmp_path)
    loaded = cache.load(tmp_path)
    assert loaded.role_hashes() == manifest.role_hashes()


def test_score_cache_detects_tampering(tmp_path: Path) -> None:
    role = RoleScores(DataRole.REFERENCE, np.array([1.0, 2.0]), "c1")
    manifest = ScoreManifest(DatasetId.NBAIOT, 11, "model-hash", {"c1": ClientScoreSet("c1", {DataRole.REFERENCE: role})})
    cache = ScoreCache()
    cache.save(manifest, tmp_path)
    np.save(tmp_path / "c1__reference.npy", np.array([9.0, 9.0]))
    try:
        cache.load(tmp_path)
    except ValueError as exc:
        assert "hash mismatch" in str(exc).lower()
    else:
        raise AssertionError("Tampered score cache was accepted")
