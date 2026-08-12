"""
Score Cache

Implements score cache serialization per Section 8.2.

Normative reference: Section 8.2
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from fedcrg.scoring.schemas import ClientScores, RoleScores, ScoreManifest


@dataclass(frozen=True, slots=True)
class ScoreCacheConfig:
    """
    Configuration for the score cache.
    
    Normative reference: Section 8.2
    """
    cache_dir: Path
    dataset: str  # "nbaiot" or "diad"
    model_seed: int
    compress: bool = True  # Use gzip compression for scores
    format: str = "numpy"  # "numpy" or "torch"


class ScoreCache:
    """
    Handles caching of anomaly scores.
    
    Scores are stored as float64 per Section 8.2.
    The cache is immutable once finalized.
    
    Normative reference: Section 8.2
    """
    
    def __init__(self, config: ScoreCacheConfig):
        """
        Initialize the score cache.
        
        Args:
            config: Score cache configuration
        """
        self.config = config
        self.cache_dir = config.cache_dir
        self.dataset = config.dataset
        self.model_seed = config.model_seed
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # State
        self._finalized: bool = False
        self._manifest: Optional[ScoreManifest] = None
        self._client_scores: Dict[str, ClientScores] = {}
    
    def is_finalized(self) -> bool:
        """Check if cache is finalized."""
        return self._finalized
    
    def get_manifest_path(self) -> Path:
        """Get path to manifest file."""
        return self.cache_dir / f"{self.dataset}_seed{self.model_seed}_manifest.json"
    
    def get_client_scores_path(self, client_id: str) -> Path:
        """Get path to client scores directory."""
        return self.cache_dir / f"{self.dataset}_seed{self.model_seed}" / client_id
    
    def get_role_scores_path(self, client_id: str, role: str) -> Path:
        """Get path to role scores file."""
        client_dir = self.get_client_scores_path(client_id)
        client_dir.mkdir(parents=True, exist_ok=True)
        
        if self.config.compress:
            return client_dir / f"{role}.npz"
        else:
            return client_dir / f"{role}.npy"
    
    def add_client_scores(self, client_scores: ClientScores) -> None:
        """
        Add scores for a client to the cache.
        
        Args:
            client_scores: ClientScores to add
            
        Raises:
            ValueError: If cache is already finalized
        """
        if self._finalized:
            raise ValueError("Cannot add scores to finalized cache")
        
        self._client_scores[client_scores.client_id] = client_scores
    
    def save_client_scores(self, client_scores: ClientScores) -> None:
        """
        Save client scores to disk.
        
        Args:
            client_scores: ClientScores to save
        """
        client_id = client_scores.client_id
        
        # Save each role separately
        for role, role_scores in client_scores.role_scores.items():
            path = self.get_role_scores_path(client_id, role)
            self._save_role_scores(role_scores, path)
        
        # Save client metadata
        client_meta_path = self.get_client_scores_path(client_id) / "_metadata.json"
        client_meta = {
            "client_id": client_id,
            "model_hash": client_scores.model_hash,
            "timestamp": client_scores.timestamp,
            "roles": list(client_scores.role_scores.keys()),
        }
        with open(client_meta_path, "w") as f:
            json.dump(client_meta, f, indent=2)
    
    def _save_role_scores(self, role_scores: RoleScores, path: Path) -> None:
        """Save role scores to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.config.format == "numpy":
            if self.config.compress:
                np.savez_compressed(path, scores=role_scores.scores)
            else:
                np.save(path, role_scores.scores)
        elif self.config.format == "torch":
            torch.save({"scores": role_scores.scores}, path)
        else:
            raise ValueError(f"Unknown format: {self.config.format}")
    
    def load_client_scores(self, client_id: str) -> Optional[ClientScores]:
        """
        Load client scores from disk.
        
        Args:
            client_id: Client identifier
            
        Returns:
            ClientScores or None if not found
        """
        client_dir = self.get_client_scores_path(client_id)
        client_meta_path = client_dir / "_metadata.json"
        
        if not client_meta_path.exists():
            return None
        
        with open(client_meta_path, "r") as f:
            client_meta = json.load(f)
        
        role_scores = {}
        for role in client_meta["roles"]:
            role_path = self.get_role_scores_path(client_id, role)
            rs = self._load_role_scores(role_path, role, client_id, client_meta["model_hash"])
            if rs:
                role_scores[role] = rs
        
        return ClientScores(
            client_id=client_id,
            role_scores=role_scores,
            model_hash=client_meta["model_hash"],
            timestamp=client_meta.get("timestamp", ""),
        )
    
    def _load_role_scores(
        self,
        path: Path,
        role: str,
        client_id: Optional[str],
        model_hash: str,
    ) -> Optional[RoleScores]:
        """Load role scores from file."""
        if not path.exists():
            return None
        
        try:
            if self.config.format == "numpy":
                if self.config.compress:
                    data = np.load(path)
                    scores = data["scores"]
                else:
                    scores = np.load(path)
            elif self.config.format == "torch":
                data = torch.load(path, map_location="cpu")
                scores = data["scores"]
                if isinstance(scores, torch.Tensor):
                    scores = scores.numpy()
            else:
                raise ValueError(f"Unknown format: {self.config.format}")
            
            return RoleScores(
                role=role,
                scores=scores,
                client_id=client_id,
            )
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None
    
    def finalize(self, manifest: ScoreManifest) -> None:
        """
        Finalize the cache.
        
        Once finalized, no more scores can be added.
        Saves the manifest and computes the cache hash.
        
        Args:
            manifest: Complete score manifest
            
        Normative reference: Section 8.2
        """
        if self._finalized:
            return
        
        # Save all client scores
        for client_scores in self._client_scores.values():
            self.save_client_scores(client_scores)
        
        # Save manifest
        manifest_path = self.get_manifest_path()
        manifest.to_json(manifest_path)
        
        # Compute cache hash
        cache_hash = self.compute_hash()
        
        # Save hash
        hash_path = self.cache_dir / f"{self.dataset}_seed{self.model_seed}_hash.txt"
        with open(hash_path, "w") as f:
            f.write(cache_hash)
        
        self._manifest = manifest
        self._finalized = True
    
    def compute_hash(self) -> str:
        """
        Compute SHA-256 hash of the entire cache.
        
        Returns:
            Hash string
            
        Normative reference: Section 8.2
        """
        # Hash all role score files
        hasher = hashlib.sha256()
        
        for client_id, client_scores in self._client_scores.items():
            for role in sorted(client_scores.role_scores.keys()):
                path = self.get_role_scores_path(client_id, role)
                if path.exists():
                    with open(path, "rb") as f:
                        hasher.update(path.name.encode())
                        hasher.update(f.read())
        
        # Hash manifest
        manifest_path = self.get_manifest_path()
        if manifest_path.exists():
            with open(manifest_path, "rb") as f:
                hasher.update(f.read())
        
        return hasher.hexdigest()
    
    def get_manifest(self) -> Optional[ScoreManifest]:
        """Get the cache manifest."""
        return self._manifest
    
    def load_manifest(self) -> Optional[ScoreManifest]:
        """Load manifest from disk."""
        manifest_path = self.get_manifest_path()
        if manifest_path.exists():
            return ScoreManifest.from_json(manifest_path)
        return None
    
    def verify(self) -> bool:
        """
        Verify the cache is complete and consistent.
        
        Returns:
            True if cache is valid
            
        Normative reference: Section 8.2
        """
        if not self._finalized:
            return False
        
        manifest = self.get_manifest()
        if manifest is None:
            return False
        
        # Check all expected files exist
        for client_id in manifest.client_ids:
            client_scores = self.load_client_scores(client_id)
            if client_scores is None:
                return False
            
            for role in manifest.client_scores[client_id].role_scores.keys():
                path = self.get_role_scores_path(client_id, role)
                if not path.exists():
                    return False
        
        # Verify hashes
        return manifest.verify_all_hashes()


def save_score_cache(
    cache: ScoreCache,
    manifest: ScoreManifest,
) -> ScoreCache:
    """
    Save a complete score cache.
    
    Args:
        cache: ScoreCache to finalize
        manifest: ScoreManifest with all scores
        
    Returns:
        Finalized ScoreCache
        
    Normative reference: Section 8.2
    """
    # Add all client scores to cache
    for client_id, client_scores in manifest.client_scores.items():
        cache.add_client_scores(client_scores)
    
    # Finalize
    cache.finalize(manifest)
    
    return cache


def load_score_cache(
    cache_dir: Path,
    dataset: str,
    model_seed: int,
    format: str = "numpy",
    compress: bool = True,
) -> Optional[ScoreCache]:
    """
    Load an existing score cache.
    
    Args:
        cache_dir: Cache directory
        dataset: Dataset identifier
        model_seed: Model seed
        format: Storage format
        compress: Whether compressed
        
    Returns:
        ScoreCache or None if not found
        
    Normative reference: Section 8.2
    """
    config = ScoreCacheConfig(
        cache_dir=cache_dir,
        dataset=dataset,
        model_seed=model_seed,
        format=format,
        compress=compress,
    )
    
    cache = ScoreCache(config)
    
    # Load manifest
    manifest = cache.load_manifest()
    if manifest is None:
        return None
    
    # Load all client scores
    for client_id in manifest.client_ids:
        client_scores = cache.load_client_scores(client_id)
        if client_scores:
            cache.add_client_scores(client_scores)
    
    # Mark as finalized
    cache._finalized = True
    cache._manifest = manifest
    
    return cache


def verify_cache() -> None:
    """Verify cache implementation."""
    from fedcrg.models import Autoencoder, create_nbaiot_ae_config
    from fedcrg.scoring.computer import ScoreComputer, ScoreComputerConfig
    import tempfile
    import torch
    import numpy as np
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        
        # Create model and compute some scores
        config = create_nbaiot_ae_config()
        model = Autoencoder(config)
        
        computer = ScoreComputer(
            ScoreComputerConfig(use_float64=True, batch_size=32),
            model,
        )
        
        # Create role data for one client
        role_data = {
            "R": torch.randn(50, 115),
            "G": torch.randn(60, 115),
            "C": torch.randn(70, 115),
        }
        
        client_scores = computer.compute_client_scores(role_data, client_id="nb01")
        
        # Create cache
        cache_config = ScoreCacheConfig(
            cache_dir=cache_dir,
            dataset="nbaiot",
            model_seed=42,
            compress=True,
        )
        cache = ScoreCache(cache_config)
        
        # Add and save
        cache.add_client_scores(client_scores)
        cache.save_client_scores(client_scores)
        
        # Load back
        loaded_scores = cache.load_client_scores("nb01")
        assert loaded_scores is not None
        assert "R" in loaded_scores.role_scores
        
        # Verify scores match
        for role in ["R", "G", "C"]:
            orig = client_scores.role_scores[role].scores
            loaded = loaded_scores.role_scores[role].scores
            assert np.allclose(orig, loaded)
        
        print("Score cache verification passed.")


if __name__ == "__main__":
    verify_cache()
