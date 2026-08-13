"""Score-cache integrity and detector-invariance checks."""

from __future__ import annotations

import numpy as np

from fedcrg.core.enums import DataRole
from fedcrg.core.ids import RowId
from fedcrg.scoring.models import ScoreManifest

_REQUIRED_BASE_ROLES = frozenset(
    {
        DataRole.TRAIN,
        DataRole.RESERVOIR,
        DataRole.BENIGN_TEST,
        DataRole.ATTACK_DEV,
        DataRole.ATTACK_TEST,
    }
)
_FORBIDDEN_DERIVED_ROLES = frozenset(
    {
        DataRole.REFERENCE,
        DataRole.MISMATCH,
        DataRole.CALIBRATION,
        DataRole.BENIGN_GUARD,
    }
)


def validate_score_manifest(manifest: ScoreManifest) -> None:
    if not manifest.clients:
        raise ValueError("Score manifest has no clients")
    for client in manifest.clients:
        client_id = client.client_id
        present = {item.role for item in client.scores}
        missing = _REQUIRED_BASE_ROLES - present
        if missing:
            names = ", ".join(sorted(role.value for role in missing))
            raise ValueError(f"Score manifest {client_id} is missing base roles: {names}")
        duplicated = _FORBIDDEN_DERIVED_ROLES & present
        if duplicated:
            names = ", ".join(sorted(role.value for role in duplicated))
            raise ValueError(
                f"Score cache must not materialize calibration-assignment roles: {names}"
            )
        row_ids: set[RowId] = set()
        for role_scores in client.scores:
            role = role_scores.role
            if role_scores.values.dtype != np.float64:
                raise ValueError("Cached scores must be float64")
            if not np.isfinite(role_scores.values).all():
                raise ValueError("NONFINITE_SCORE")
            overlap = row_ids.intersection(role_scores.row_ids)
            if overlap:
                raise ValueError(f"ROLE_OVERLAP: {client_id}/{role.value}")
            row_ids.update(role_scores.row_ids)
            if len(role_scores.sha256.value) != 64:
                raise ValueError("Invalid role-score hash")
        if len(client.get(DataRole.BENIGN_TEST).values) == 0:
            raise ValueError(f"Final benign test is empty for {client_id}")
        if len(client.get(DataRole.ATTACK_TEST).values) == 0:
            raise ValueError(f"Final attack test is empty for {client_id}")


def assert_same_cache_hash(*manifests: ScoreManifest) -> None:
    hashes = {manifest.cache_sha256 for manifest in manifests}
    if None in hashes or len(hashes) != 1:
        raise ValueError("SCORE_CACHE_HASH_MISMATCH")
