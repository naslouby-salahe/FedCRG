"""
Score Schemas

Defines the data structures for score caching per Section 8.2.

Normative reference: Section 8.2
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class RoleScores:
    """
    Scores for a specific role (R, G, C, B, A_test).
    
    Contains the anomaly scores and metadata for a single role.
    
    Normative reference: Section 8.2
    """
    role: str  # e.g., "R", "G", "C", "B", "A_test"
    scores: np.ndarray  # float64 scores, shape (n_samples,)
    client_id: Optional[str] = None  # Client ID if per-client
    hash: str = field(default="")  # SHA-256 hash of scores
    
    def __post_init__(self):
        """Compute hash if not provided."""
        # Convert to object to allow modification for hash computation
        object.__setattr__(self, 'scores', np.asarray(self.scores, dtype=np.float64))
        
        if not self.hash:
            hash_val = self._compute_hash()
            object.__setattr__(self, 'hash', hash_val)
    
    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of the scores."""
        scores_bytes = self.scores.tobytes()
        return hashlib.sha256(scores_bytes).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "role": self.role,
            "scores": self.scores.tolist(),
            "client_id": self.client_id,
            "hash": self.hash,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoleScores":
        """Create from dictionary."""
        return cls(
            role=data["role"],
            scores=np.array(data["scores"], dtype=np.float64),
            client_id=data.get("client_id"),
            hash=data.get("hash", ""),
        )
    
    def to_tensor(self) -> torch.Tensor:
        """Convert scores to PyTorch tensor."""
        return torch.from_numpy(self.scores).double()
    
    def verify_hash(self) -> bool:
        """Verify the hash matches the scores."""
        return self.hash == self._compute_hash()


@dataclass(frozen=True, slots=True)
class ClientScores:
    """
    All scores for a single client.
    
    Contains scores for all roles for one client.
    
    Normative reference: Section 8.2
    """
    client_id: str
    role_scores: Dict[str, RoleScores]  # role -> RoleScores
    model_hash: str  # Hash of the model used
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def get_role_scores(self, role: str) -> Optional[RoleScores]:
        """Get scores for a specific role."""
        return self.role_scores.get(role)
    
    def get_all_scores(self) -> Dict[str, np.ndarray]:
        """Get all scores as a dictionary."""
        return {role: rs.scores for role, rs in self.role_scores.items()}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "client_id": self.client_id,
            "role_scores": {role: rs.to_dict() for role, rs in self.role_scores.items()},
            "model_hash": self.model_hash,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClientScores":
        """Create from dictionary."""
        role_scores = {
            role: RoleScores.from_dict(rs_data)
            for role, rs_data in data["role_scores"].items()
        }
        return cls(
            client_id=data["client_id"],
            role_scores=role_scores,
            model_hash=data["model_hash"],
            timestamp=data.get("timestamp", ""),
        )
    
    def compute_hash(self) -> str:
        """Compute SHA-256 hash of all scores."""
        all_data = [
            self.client_id.encode(),
            self.model_hash.encode(),
            json.dumps({role: rs.hash for role, rs in self.role_scores.items()}, sort_keys=True).encode(),
        ]
        combined = b"".join(all_data)
        return hashlib.sha256(combined).hexdigest()


@dataclass(frozen=True, slots=True)
class ScoreManifest:
    """
    Manifest for a complete score cache.
    
    Contains all metadata needed to verify and reproduce the score cache.
    
    Normative reference: Section 8.2
    """
    model_seed: int
    dataset: str  # "nbaiot" or "diad"
    client_ids: List[str]
    model_hash: str  # Hash of the final global model
    client_scores: Dict[str, ClientScores]  # client_id -> ClientScores
    score_dtype: str = "float64"  # Always float64 per Section 8.2
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_seed": self.model_seed,
            "dataset": self.dataset,
            "client_ids": self.client_ids,
            "model_hash": self.model_hash,
            "score_dtype": self.score_dtype,
            "timestamp": self.timestamp,
            "client_scores": {
                cid: cs.to_dict() for cid, cs in self.client_scores.items()
            },
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScoreManifest":
        """Create from dictionary."""
        client_scores = {
            cid: ClientScores.from_dict(cs_data)
            for cid, cs_data in data["client_scores"].items()
        }
        return cls(
            model_seed=data["model_seed"],
            dataset=data["dataset"],
            client_ids=data["client_ids"],
            model_hash=data["model_hash"],
            client_scores=client_scores,
            score_dtype=data.get("score_dtype", "float64"),
            timestamp=data.get("timestamp", ""),
        )
    
    def to_json(self, path: Path) -> None:
        """Save manifest to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, path: Path) -> "ScoreManifest":
        """Load manifest from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def compute_hash(self) -> str:
        """Compute SHA-256 hash of the entire manifest."""
        manifest_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(manifest_str.encode()).hexdigest()
    
    def verify_all_hashes(self) -> bool:
        """Verify all score hashes in the manifest."""
        for cid, cs in self.client_scores.items():
            if not cs.compute_hash():
                return False
            for role, rs in cs.role_scores.items():
                if not rs.verify_hash():
                    return False
        return True


def verify_schemas() -> None:
    """Verify schema implementations."""
    # Test RoleScores
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    rs = RoleScores(role="R", scores=scores, client_id="nb01")
    
    assert rs.role == "R"
    assert rs.client_id == "nb01"
    assert rs.hash != ""
    assert rs.verify_hash()
    assert len(rs.hash) == 64  # SHA-256 hex
    
    # Test serialization
    rs_dict = rs.to_dict()
    rs_restored = RoleScores.from_dict(rs_dict)
    assert rs_restored == rs
    
    # Test ClientScores
    client_scores = {
        "R": RoleScores(role="R", scores=np.array([1.0, 2.0], dtype=np.float64)),
        "G": RoleScores(role="G", scores=np.array([3.0, 4.0], dtype=np.float64)),
    }
    cs = ClientScores(
        client_id="nb01",
        role_scores=client_scores,
        model_hash="test_hash",
    )
    
    assert cs.get_role_scores("R") is not None
    assert cs.get_role_scores("X") is None
    
    # Test ScoreManifest
    manifest = ScoreManifest(
        model_seed=42,
        dataset="nbaiot",
        client_ids=["nb01", "nb02"],
        model_hash="global_hash",
        client_scores={
            "nb01": ClientScores(
                client_id="nb01",
                role_scores={
                    "R": RoleScores(role="R", scores=np.array([1.0, 2.0], dtype=np.float64)),
                },
                model_hash="global_hash",
            ),
        },
    )
    
    assert manifest.score_dtype == "float64"
    assert manifest.verify_all_hashes()
    
    print("Schema verification passed.")


if __name__ == "__main__":
    verify_schemas()
