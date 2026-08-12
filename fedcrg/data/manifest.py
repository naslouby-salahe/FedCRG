"""
Manifest Module

Provides manifest generation and hash verification for datasets.

Normative reference: Section 7 (Dataset and Data-Partition Protocol)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from fedcrg.data.base import (
    DatasetRole,
    RowIDComponents,
    generate_row_id,
    compute_file_hash,
)


@dataclass(frozen=True, slots=True)
class FileEntry:
    """Entry for a single file in the manifest."""
    relative_path: str
    sha256: str
    row_count: int
    column_count: int
    
    @classmethod
    def from_file(cls, filepath: Path, relative_root: Path | None = None) -> "FileEntry":
        """
        Create a FileEntry from a CSV file.
        
        Args:
            filepath: Absolute path to the file
            relative_root: Root path for computing relative path.
                          If None, uses filepath.parent.
        
        Returns:
            FileEntry with computed hash and metadata
        """
        if relative_root is None:
            relative_root = filepath.parent
        
        rel_path = str(filepath.relative_to(relative_root))
        
        # Read file to get metadata
        df = pd.read_csv(filepath)
        row_count = len(df)
        column_count = len(df.columns)
        
        # Compute hash
        sha256 = compute_file_hash(filepath)
        
        return cls(
            relative_path=rel_path,
            sha256=sha256,
            row_count=row_count,
            column_count=column_count,
        )


@dataclass(frozen=True, slots=True)
class SplitInfo:
    """Information about a single split/role for a client."""
    role: DatasetRole
    row_count: int
    file_hash: Optional[str] = None
    row_ids: Optional[List[str]] = None  # Sample of row IDs for verification
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, omitting None values."""
        d = {"role": self.role.value, "row_count": self.row_count}
        if self.file_hash is not None:
            d["file_hash"] = self.file_hash
        if self.row_ids is not None:
            d["row_ids"] = self.row_ids
        return d


@dataclass(frozen=True, slots=True)
class ClientManifest:
    """
    Manifest for a single client.
    
    Contains all metadata needed to verify client data integrity.
    """
    client_id: str
    dataset_id: str
    benign_files: Tuple[FileEntry, ...]
    malicious_files: Tuple[FileEntry, ...]
    total_benign_rows: int
    total_malicious_rows: int
    feature_names: List[str]
    feature_count: int
    splits: Tuple[SplitInfo, ...]
    calibration_seeds: List[int]
    row_id_samples: Dict[str, str]  # role -> sample row_id
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "client_id": self.client_id,
            "dataset_id": self.dataset_id,
            "benign_files": [asdict(f) for f in self.benign_files],
            "malicious_files": [asdict(f) for f in self.malicious_files],
            "total_benign_rows": self.total_benign_rows,
            "total_malicious_rows": self.total_malicious_rows,
            "feature_names": self.feature_names,
            "feature_count": self.feature_count,
            "splits": [s.to_dict() for s in self.splits],
            "calibration_seeds": self.calibration_seeds,
            "row_id_samples": self.row_id_samples,
            "created_at": self.created_at,
        }
    
    def save(self, path: Path | str) -> None:
        """Save manifest to JSON file."""
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
    
    @classmethod
    def load(cls, path: Path | str) -> "ClientManifest":
        """Load manifest from JSON file."""
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)
        
        # Convert back to objects
        benign_files = tuple(
            FileEntry(**f) for f in data["benign_files"]
        )
        malicious_files = tuple(
            FileEntry(**f) for f in data["malicious_files"]
        )
        splits = tuple(
            SplitInfo(
                role=DatasetRole(s["role"]),
                row_count=s["row_count"],
                file_hash=s.get("file_hash"),
                row_ids=s.get("row_ids"),
            )
            for s in data["splits"]
        )
        
        return cls(
            client_id=data["client_id"],
            dataset_id=data["dataset_id"],
            benign_files=benign_files,
            malicious_files=malicious_files,
            total_benign_rows=data["total_benign_rows"],
            total_malicious_rows=data["total_malicious_rows"],
            feature_names=data["feature_names"],
            feature_count=data["feature_count"],
            splits=splits,
            calibration_seeds=data["calibration_seeds"],
            row_id_samples=data["row_id_samples"],
            created_at=data.get("created_at", ""),
        )
    
    def get_split_row_counts(self) -> Dict[DatasetRole, int]:
        """Get row counts by role."""
        return {s.role: s.row_count for s in self.splits}


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """
    Manifest for an entire dataset.
    
    Contains metadata for all clients and dataset-level information.
    """
    dataset_id: str
    dataset_version: str
    data_root: str
    clients: Dict[str, ClientManifest]  # client_id -> ClientManifest
    eligibility: Dict[str, Any]  # Eligibility information (for DIAD)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "data_root": self.data_root,
            "clients": {cid: cm.to_dict() for cid, cm in self.clients.items()},
            "eligibility": self.eligibility,
            "created_at": self.created_at,
        }
    
    def save(self, path: Path | str) -> None:
        """Save manifest to JSON file."""
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
    
    @classmethod
    def load(cls, path: Path | str) -> "DatasetManifest":
        """Load manifest from JSON file."""
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)
        
        clients = {
            cid: ClientManifest.load(cm_data)
            for cid, cm_data in data["clients"].items()
        }
        
        return cls(
            dataset_id=data["dataset_id"],
            dataset_version=data["dataset_version"],
            data_root=data["data_root"],
            clients=clients,
            eligibility=data.get("eligibility", {}),
            created_at=data.get("created_at", ""),
        )
    
    def get_client_ids(self) -> List[str]:
        """Get list of client IDs."""
        return list(self.clients.keys())
    
    def get_total_row_counts(self) -> Dict[str, int]:
        """Get total row counts across all clients."""
        total_benign = sum(
            cm.total_benign_rows for cm in self.clients.values()
        )
        total_malicious = sum(
            cm.total_malicious_rows for cm in self.clients.values()
        )
        return {
            "total_benign": total_benign,
            "total_malicious": total_malicious,
            "num_clients": len(self.clients),
        }


def generate_manifest_hash(manifest: DatasetManifest) -> str:
    """
    Generate SHA-256 hash of a manifest for verification.
    
    Args:
        manifest: DatasetManifest to hash
        
    Returns:
        SHA-256 hex string of the JSON representation
    """
    manifest_dict = manifest.to_dict()
    # Sort for deterministic hashing
    manifest_json = json.dumps(manifest_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(manifest_json.encode()).hexdigest()


def verify_manifest_hash(manifest: DatasetManifest, expected_hash: str) -> bool:
    """
    Verify manifest hash matches expected value.
    
    Args:
        manifest: DatasetManifest to verify
        expected_hash: Expected SHA-256 hash
        
    Returns:
        True if hash matches, False otherwise
    """
    actual_hash = generate_manifest_hash(manifest)
    return actual_hash == expected_hash
