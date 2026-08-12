"""
FedCRG Scoring Module

Implements score computation and caching per Section 8.2.

Normative reference: Section 8.2
"""

from fedcrg.scoring.cache import (
    ScoreCache,
    ScoreCacheConfig,
    load_score_cache,
    save_score_cache,
)
from fedcrg.scoring.computer import (
    ScoreComputer,
    ScoreComputerConfig,
)
from fedcrg.scoring.schemas import (
    ClientScores,
    RoleScores,
    ScoreManifest,
)

__all__ = [
    # Cache
    "ScoreCache",
    "ScoreCacheConfig",
    "load_score_cache",
    "save_score_cache",
    # Computer
    "ScoreComputer",
    "ScoreComputerConfig",
    # Schemas
    "ClientScores",
    "RoleScores",
    "ScoreManifest",
]
