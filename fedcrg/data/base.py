"""
Base Dataset Adapter Module

Provides the abstract base class and common utilities for dataset adapters.

Normative reference: Section 7 (Dataset and Data-Partition Protocol)
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd


class DatasetRole(str, Enum):
    """
    Role identifiers for data partitioning per Section 7.
    
    R: Reference - used for federation reference threshold
    G: Gate - used for Gate B (reference mismatch detection)
    C: Calibration - used for Gate A (local readiness) and local threshold
    GUARD: Comparator benign guard - used only by attack-aware baselines
    TRAIN: Benign training - used for model/scaler fitting
    TEST_BENIGN: Final benign test - used for final evaluation
    TEST_ATTACK: Final attack test - used for final evaluation
    DEV_ATTACK: Development attack - used only by attack-aware comparators
    """
    R = "R"
    G = "G"
    C = "C"
    GUARD = "GUARD"
    TRAIN = "TRAIN"
    TEST_BENIGN = "TEST_BENIGN"
    TEST_ATTACK = "TEST_ATTACK"
    DEV_ATTACK = "DEV_ATTACK"


# All benign roles (exclude attack roles)
BENIGN_ROLES = {
    DatasetRole.R,
    DatasetRole.G,
    DatasetRole.C,
    DatasetRole.GUARD,
    DatasetRole.TRAIN,
    DatasetRole.TEST_BENIGN,
}

# All attack roles
ATTACK_ROLES = {
    DatasetRole.DEV_ATTACK,
    DatasetRole.TEST_ATTACK,
}

# Roles that FedCRG directly uses (no attack labels)
FEDCRG_ROLES = {
    DatasetRole.R,
    DatasetRole.G,
    DatasetRole.C,
    DatasetRole.TRAIN,
    DatasetRole.TEST_BENIGN,
}

# Roles used only by comparators (attack-aware)
COMPARATOR_ROLES = {
    DatasetRole.GUARD,
    DatasetRole.DEV_ATTACK,
}


@dataclass(frozen=True, slots=True)
class RowIDComponents:
    """Components for generating a stable row ID."""
    dataset_id: str
    client_id: str
    source_file: str
    source_row_index: int

    def to_string(self) -> str:
        """Convert components to pipe-separated string for hashing."""
        return f"{self.dataset_id}|{self.client_id}|{self.source_file}|{self.source_row_index}"


def generate_row_id(components: RowIDComponents) -> str:
    """
    Generate a stable SHA-256 row ID from components.
    
    Per Section 7.1.4: row_id = SHA256(dataset_id || client_id ||
    source_file_relative_path || source_row_index)
    
    Args:
        components: RowIDComponents containing the four required fields
        
    Returns:
        SHA-256 hex string (64 characters)
    """
    return hashlib.sha256(components.to_string().encode()).hexdigest()


def generate_row_id_from_parts(
    dataset_id: str,
    client_id: str,
    source_file: str,
    source_row_index: int,
) -> str:
    """
    Convenience function to generate row ID directly from parts.
    
    Args:
        dataset_id: Dataset identifier (e.g., "nbaiot", "diad")
        client_id: Client identifier (e.g., "nb01", "diad_<hash>")
        source_file: Relative path to source file
        source_row_index: 0-indexed row position in source file
        
    Returns:
        SHA-256 hex string
    """
    return generate_row_id(RowIDComponents(
        dataset_id=dataset_id,
        client_id=client_id,
        source_file=source_file,
        source_row_index=source_row_index,
    ))


def compute_file_hash(filepath: Path | str, chunk_size: int = 8192) -> str:
    """
    Compute SHA-256 hash of a file.
    
    Args:
        filepath: Path to the file
        chunk_size: Size of chunks to read (default 8KB)
        
    Returns:
        SHA-256 hex string
    """
    filepath = Path(filepath)
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


@dataclass(frozen=True, slots=True)
class DataIntegrityCheck:
    """Result of a data integrity check."""
    check_name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass(frozen=True, slots=True)
class DatasetIntegrityReport:
    """Complete integrity report for a dataset."""
    dataset_id: str
    checks: Tuple[DataIntegrityCheck, ...]
    client_reports: Dict[str, "ClientIntegrityReport"]
    
    @property
    def all_passed(self) -> bool:
        """Check if all integrity checks passed."""
        return all(c.passed for c in self.checks) and all(
            cr.all_passed for cr in self.client_reports.values()
        )
    
    @property
    def failed_checks(self) -> List[DataIntegrityCheck]:
        """Get all failed checks."""
        failed = [c for c in self.checks if not c.passed]
        for cr in self.client_reports.values():
            failed.extend(cr.failed_checks)
        return failed


@dataclass(frozen=True, slots=True)
class ClientIntegrityReport:
    """Integrity report for a single client."""
    client_id: str
    checks: Tuple[DataIntegrityCheck, ...]
    row_counts: Dict[str, int]
    
    @property
    def all_passed(self) -> bool:
        """Check if all client integrity checks passed."""
        return all(c.passed for c in self.checks)
    
    @property
    def failed_checks(self) -> List[DataIntegrityCheck]:
        """Get all failed checks for this client."""
        return [c for c in self.checks if not c.passed]


class BaseDatasetAdapter(ABC):
    """
    Abstract base class for dataset adapters.
    
    Provides common interface for loading, partitioning, and validating
    datasets per Section 7 requirements.
    """
    
    dataset_id: str
    data_root: Path
    
    def __init__(self, data_root: Path | str | None = None):
        """
        Initialize the adapter.
        
        Args:
            data_root: Root path to data directory. Defaults to data/raw
        """
        if data_root is None:
            data_root = Path("/home/naslouby/Projects/FedCRG/data/raw")
        self.data_root = Path(data_root)
        self._validate_data_root()
    
    @property
    @abstractmethod
    def dataset_id(self) -> str:
        """Dataset identifier (e.g., 'nbaiot', 'diad')."""
        pass
    
    @property
    @abstractmethod
    def expected_features(self) -> int:
        """Expected number of features."""
        pass
    
    @property
    @abstractmethod
    def expected_clients(self) -> Optional[int]:
        """Expected number of clients, if fixed."""
        pass
    
    def _validate_data_root(self) -> None:
        """Validate that data root exists or raise informative error."""
        if not self.data_root.exists():
            raise FileNotFoundError(
                f"Data root {self.data_root} does not exist. "
                f"Please ensure {self.data_root}/ exists and contains the dataset files. "
                f"Per Section 5 of prompt.md, the canonical entrypoint is "
                f"/home/naslouby/Projects/FedCRG/data/raw. "
                f"If the raw data is at an external location, create a symlink."
            )
    
    @abstractmethod
    def discover_clients(self) -> List[str]:
        """
        Discover available client identifiers.
        
        Returns:
            List of client IDs found in the dataset
        """
        pass
    
    @abstractmethod
    def load_client_benign(
        self,
        client_id: str,
        features_only: bool = False,
    ) -> pd.DataFrame:
        """
        Load benign data for a client.
        
        Args:
            client_id: Client identifier
            features_only: If True, return only feature columns
            
        Returns:
            DataFrame with benign data
        """
        pass
    
    @abstractmethod
    def load_client_malicious(
        self,
        client_id: str,
        features_only: bool = False,
    ) -> pd.DataFrame:
        """
        Load malicious data for a client.
        
        Args:
            client_id: Client identifier
            features_only: If True, return only feature columns
            
        Returns:
            DataFrame with malicious data
        """
        pass
    
    @abstractmethod
    def generate_splits(
        self,
        client_id: str,
        calibration_seed: int,
    ) -> Dict[DatasetRole, pd.DataFrame]:
        """
        Generate role-based splits for a client.
        
        Per Section 7, this creates R, G, C, GUARD, TRAIN, TEST_BENIGN,
        DEV_ATTACK, TEST_ATTACK partitions.
        
        Args:
            client_id: Client identifier
            calibration_seed: Random seed for calibration split permutation
            
        Returns:
            Dictionary mapping DatasetRole to DataFrame
        """
        pass
    
    @abstractmethod
    def verify_integrity(self) -> DatasetIntegrityReport:
        """
        Verify dataset integrity per Section 7 requirements.
        
        Checks include:
        - Expected feature count
        - No NaN/inf values
        - Row count feasibility
        - Disjointness of partitions
        
        Returns:
            DatasetIntegrityReport with all check results
        """
        pass
    
    @abstractmethod
    def get_client_manifest(self, client_id: str) -> "ClientManifest":
        """
        Generate a manifest for a client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            ClientManifest with SHA-256 hashes and row counts
        """
        pass
