from __future__ import annotations

from dataclasses import fields
from pathlib import PurePosixPath

from fedcrg.artifacts.manifest import RunManifest
from fedcrg.artifacts.references import CacheReference
from fedcrg.core.ids import (
    CalibrationSeed,
    ClientId,
    ModelSeed,
    RowId,
    RunId,
    Sha256,
)
from fedcrg.scoring.cache import ScoreCacheIdentity, ScoreRoleCacheRecord
from fedcrg.scoring.models import RoleScores


def _field_type(cls: type[object], name: str) -> object:
    return next(field.type for field in fields(cls) if field.name == name)


def test_run_manifest_uses_domain_ids_not_primitives() -> None:
    assert _field_type(RunManifest, "run_id") is RunId
    assert _field_type(RunManifest, "config_hash") is Sha256
    assert _field_type(RunManifest, "model_seed") is ModelSeed
    assert _field_type(RunManifest, "calibration_seed") is CalibrationSeed


def test_cache_references_use_path_and_hash_value_types() -> None:
    assert _field_type(CacheReference, "relative_path") is PurePosixPath
    assert _field_type(CacheReference, "sha256") is Sha256


def test_score_evidence_uses_typed_identity_fields() -> None:
    assert _field_type(RoleScores, "client_id") is ClientId
    assert _field_type(ScoreCacheIdentity, "model_seed") is ModelSeed
    assert _field_type(ScoreCacheIdentity, "model_hash") is Sha256
    assert _field_type(ScoreRoleCacheRecord, "client_id") is ClientId
    assert _field_type(ScoreRoleCacheRecord, "score_array_sha256") is Sha256
