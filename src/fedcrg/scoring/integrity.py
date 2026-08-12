"""Score integrity checks."""

import numpy as np
from fedcrg.scoring.models import ScoreManifest


def validate_score_manifest(manifest: ScoreManifest) -> None:
    if not manifest.clients:
        raise ValueError("Score manifest has no clients")
    for client in manifest.clients.values():
        for role_scores in client.scores.values():
            if role_scores.values.dtype != np.float64:
                raise ValueError("Cached scores must be float64")
            if not np.isfinite(role_scores.values).all():
                raise ValueError("Cached scores contain non-finite values")
            if len(role_scores.sha256) != 64:
                raise ValueError("Invalid score hash")
