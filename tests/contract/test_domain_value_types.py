from __future__ import annotations

import typing

from fedcrg.evidence.models import CacheReference, RunManifest
from fedcrg.learning.scores import RoleScores, ScoreCacheIdentity, ScoreRoleCacheRecord
from fedcrg.types import (
    CalibrationSeed,
    ClientId,
    Identifier,
    ModelSeed,
    RunId,
    Sha256,
)


def _field_type(cls: type[object], name: str) -> object:
    hints = typing.get_type_hints(cls, include_extras=True)
    return hints[name]


def test_run_manifest_uses_domain_ids_not_primitives() -> None:
    assert _field_type(RunManifest, "run_id") == RunId
    assert _field_type(RunManifest, "config_hash") == Sha256
    assert _field_type(RunManifest, "model_seed") == ModelSeed
    assert _field_type(RunManifest, "calibration_seed") == CalibrationSeed


def test_cache_references_use_path_and_hash_value_types() -> None:
    assert _field_type(CacheReference, "relative_path") == Identifier
    assert _field_type(CacheReference, "sha256") == Sha256


def test_score_evidence_uses_typed_identity_fields() -> None:
    assert _field_type(RoleScores, "client_id") == ClientId
    assert _field_type(ScoreCacheIdentity, "model_seed") == ModelSeed
    assert _field_type(ScoreCacheIdentity, "model_hash") == Sha256
    assert _field_type(ScoreRoleCacheRecord, "client_id") == ClientId
    assert _field_type(ScoreRoleCacheRecord, "score_array_sha256") == Sha256
