"""
FedCRG Data Infrastructure Module

This module provides data loading, partitioning, and integrity verification for
N-BaIoT and CIC IoT-DIAD datasets per Section 7 of the FedCRG roadmap.

Key responsibilities:
- Dataset discovery and integrity checking
- Role-based partitioning (R, G, C, guard, train, test)
- Row ID generation and disjointness verification
- Manifest generation and SHA-256 hashing
- Attack/begnign separation and firewall

Normative reference: Section 7 (Dataset and Data-Partition Protocol)
"""

# Import base types first to avoid circular imports
from fedcrg.data.base import (
    BaseDatasetAdapter,
    DatasetRole,
    RowIDComponents,
    DataIntegrityCheck,
    DatasetIntegrityReport,
    ClientIntegrityReport,
    generate_row_id,
    generate_row_id_from_parts,
    compute_file_hash,
    BENIGN_ROLES,
    ATTACK_ROLES,
    FEDCRG_ROLES,
    COMPARATOR_ROLES,
)

# Import manifest types
from fedcrg.data.manifest import (
    FileEntry,
    SplitInfo,
    ClientManifest,
    DatasetManifest,
    generate_manifest_hash,
    verify_manifest_hash,
)

# Import splitting utilities
from fedcrg.data.splitting import (
    create_hash_seeded_generator,
    generate_calibration_permutation,
    generate_nbaiot_splits,
    generate_diad_splits,
    generate_attack_splits,
    verify_disjointness,
    verify_benign_attack_separation,
)

# Import dataset adapters
from fedcrg.data.nbaiot import NBaiotAdapter
from fedcrg.data.diad import DiadAdapter


__all__ = [
    # Base classes and types
    "BaseDatasetAdapter",
    "DatasetRole",
    "RowIDComponents",
    "DataIntegrityCheck",
    "DatasetIntegrityReport",
    "ClientIntegrityReport",
    # Utility functions
    "generate_row_id",
    "generate_row_id_from_parts",
    "compute_file_hash",
    # Role sets
    "BENIGN_ROLES",
    "ATTACK_ROLES",
    "FEDCRG_ROLES",
    "COMPARATOR_ROLES",
    # Manifest types and functions
    "FileEntry",
    "SplitInfo",
    "ClientManifest",
    "DatasetManifest",
    "generate_manifest_hash",
    "verify_manifest_hash",
    # Splitting utilities
    "create_hash_seeded_generator",
    "generate_calibration_permutation",
    "generate_nbaiot_splits",
    "generate_diad_splits",
    "generate_attack_splits",
    "verify_disjointness",
    "verify_benign_attack_separation",
    # Dataset adapters
    "NBaiotAdapter",
    "DiadAdapter",
]
