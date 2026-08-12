"""
Splitting Module

Provides role-based data splitting utilities per Section 7.

Normative reference: Section 7 (Dataset and Data-Partition Protocol)
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from fedcrg.data.base import (
    DatasetRole,
    RowIDComponents,
    generate_row_id,
)
from fedcrg.config import DatasetID


def create_hash_seeded_generator(seed_str: str) -> np.random.Generator:
    """
    Create a NumPy PCG64 generator seeded from a hash.
    
    Per Section 7.2.2: seed derived from SHA256 hash reduced to unsigned 64-bit integer.
    
    Args:
        seed_str: String to hash for seeding
        
    Returns:
        NumPy Generator with PCG64 bit generator
    """
    seed_bytes = hashlib.sha256(seed_str.encode()).digest()
    seed_int = int.from_bytes(seed_bytes, byteorder="big", signed=False)
    # Use first 64 bits
    seed_int = seed_int & 0xFFFFFFFFFFFFFFFF
    return np.random.Generator(np.random.PCG64(seed_int))


def generate_calibration_permutation(
    n: int,
    client_id: str,
    calibration_seed: int,
    dataset_id: str = "nbaiot",
) -> np.ndarray:
    """
    Generate a deterministic permutation of indices for calibration splitting.
    
    Per Section 7.1.2 for N-BaIoT:
    - Reservoir of 6000 rows is permuted using calibration_seed
    - R: positions 1-500
    - G: positions 501-3500
    - C: positions 3501-5500
    - GUARD: positions 5501-6000
    
    For DIAD (Section 7.2.2):
    - Reservoir of 3800 rows is permuted
    - R: positions 1-300
    - G: positions 301-1800
    - C: positions 1801-3300
    - GUARD: positions 3301-3800
    
    Args:
        n: Size of array to permute
        client_id: Client identifier
        calibration_seed: Calibration random seed
        dataset_id: Dataset identifier
        
    Returns:
        Permutation indices (0 to n-1)
    """
    # Create seed string per roadmap
    if dataset_id == DatasetID.NBAIOT.value:
        seed_str = f"fedcrg|{dataset_id}|calibration|{calibration_seed}|{client_id}"
    else:  # DIAD
        seed_str = f"fedcrg|diad|calibration|{calibration_seed}|{client_id}"
    
    rng = create_hash_seeded_generator(seed_str)
    return rng.permutation(n)


def generate_nbaiot_splits(
    benign_df: pd.DataFrame,
    malicious_df: pd.DataFrame,
    client_id: str,
    calibration_seed: int,
    dataset_id: str = DatasetID.NBAIOT.value,
) -> Dict[DatasetRole, pd.DataFrame]:
    """
    Generate N-BaIoT role-based splits per Section 7.1.2.
    
    Benign partition:
    - T_k (TRAIN): First 4000 benign rows in source-file order
    - Reservoir: Next 6000 benign rows
      - R: First 500 of permuted reservoir
      - G: Next 3000 of permuted reservoir (positions 501-3500)
      - C: Next 2000 of permuted reservoir (positions 3501-5500)
      - GUARD: Last 500 of permuted reservoir (positions 5501-6000)
    - B_k (TEST_BENIGN): All remaining benign rows (never subsampled)
    
    Malicious partition (Section 7.1.3):
    - A_dev: Exactly 500 records, category-balanced
    - A_test: All remaining malicious records
    
    Args:
        benign_df: DataFrame with all benign data (source-file order)
        malicious_df: DataFrame with all malicious data
        client_id: Client identifier (e.g., "nb01")
        calibration_seed: Calibration random seed
        dataset_id: Dataset identifier
        
    Returns:
        Dictionary mapping DatasetRole to DataFrame
    """
    splits = {}
    
    # Benign splitting
    train_end = 4000
    reservoir_start = train_end
    reservoir_end = reservoir_start + 6000
    
    # TRAIN: First 4000 benign rows
    if len(benign_df) >= train_end:
        train_df = benign_df.iloc[:train_end].copy()
        train_df["_role"] = DatasetRole.TRAIN.value
        splits[DatasetRole.TRAIN] = train_df
    else:
        raise ValueError(
            f"Client {client_id} has only {len(benign_df)} benign rows, "
            f"but {train_end} required for TRAIN"
        )
    
    # Reservoir: Next 6000 benign rows
    if len(benign_df) >= reservoir_end:
        reservoir_df = benign_df.iloc[reservoir_start:reservoir_end].copy()
    else:
        raise ValueError(
            f"Client {client_id} has only {len(benign_df)} benign rows, "
            f"but {reservoir_end} required (4000 train + 6000 reservoir)"
        )
    
    # Permute reservoir
    permutation = generate_calibration_permutation(
        6000, client_id, calibration_seed, dataset_id
    )
    reservoir_permuted = reservoir_df.iloc[permutation].copy()
    
    # Split reservoir into R, G, C, GUARD
    splits[DatasetRole.R] = reservoir_permuted.iloc[:500].copy()
    splits[DatasetRole.R]["_role"] = DatasetRole.R.value
    
    splits[DatasetRole.G] = reservoir_permuted.iloc[500:3500].copy()
    splits[DatasetRole.G]["_role"] = DatasetRole.G.value
    
    splits[DatasetRole.C] = reservoir_permuted.iloc[3500:5500].copy()
    splits[DatasetRole.C]["_role"] = DatasetRole.C.value
    
    splits[DatasetRole.GUARD] = reservoir_permuted.iloc[5500:6000].copy()
    splits[DatasetRole.GUARD]["_role"] = DatasetRole.GUARD.value
    
    # TEST_BENIGN: All remaining benign rows
    test_benign_df = benign_df.iloc[reservoir_end:].copy()
    if len(test_benign_df) < 3000:
        raise ValueError(
            f"Client {client_id} has only {len(test_benign_df)} final benign rows, "
            f"but minimum 3000 required"
        )
    test_benign_df["_role"] = DatasetRole.TEST_BENIGN.value
    splits[DatasetRole.TEST_BENIGN] = test_benign_df
    
    # Malicious splitting per Section 7.1.3
    # Get attack subtypes from file paths or labels
    # For N-BaIoT, we need to identify subtypes from the malicious files
    # This is a simplified version - full implementation needs file parsing
    attack_splits = generate_attack_splits(malicious_df, client_id, dataset_id)
    splits[DatasetRole.DEV_ATTACK] = attack_splits[DatasetRole.DEV_ATTACK]
    splits[DatasetRole.TEST_ATTACK] = attack_splits[DatasetRole.TEST_ATTACK]
    
    return splits


def generate_diad_splits(
    benign_df: pd.DataFrame,
    malicious_df: pd.DataFrame,
    client_id: str,
    calibration_seed: int,
) -> Dict[DatasetRole, pd.DataFrame]:
    """
    Generate DIAD role-based splits per Section 7.2.2.
    
    Benign partition:
    - T_k (TRAIN): First 2000 benign rows in verified order
    - Reservoir: Next 3800 benign rows
      - R: First 300 of permuted reservoir
      - G: Next 1500 of permuted reservoir (positions 301-1800)
      - C: Next 1500 of permuted reservoir (positions 1801-3300)
      - GUARD: Last 500 of permuted reservoir (positions 3301-3800)
    - B_k (TEST_BENIGN): All remaining benign rows (>=2000 required)
    
    Malicious partition (Section 7.2.3):
    - A_dev: Exactly 500 records, category-balanced
    - A_test: All remaining malicious records (>=500 required)
    
    Args:
        benign_df: DataFrame with all benign data
        malicious_df: DataFrame with all malicious data
        client_id: Client identifier
        calibration_seed: Calibration random seed
        
    Returns:
        Dictionary mapping DatasetRole to DataFrame
    """
    splits = {}
    
    # Benign splitting
    train_end = 2000
    reservoir_start = train_end
    reservoir_end = reservoir_start + 3800
    
    # TRAIN: First 2000 benign rows
    if len(benign_df) >= train_end:
        train_df = benign_df.iloc[:train_end].copy()
        train_df["_role"] = DatasetRole.TRAIN.value
        splits[DatasetRole.TRAIN] = train_df
    else:
        raise ValueError(
            f"Client {client_id} has only {len(benign_df)} benign rows, "
            f"but {train_end} required for TRAIN"
        )
    
    # Reservoir: Next 3800 benign rows
    if len(benign_df) >= reservoir_end:
        reservoir_df = benign_df.iloc[reservoir_start:reservoir_end].copy()
    else:
        raise ValueError(
            f"Client {client_id} has only {len(benign_df)} benign rows, "
            f"but {reservoir_end} required (2000 train + 3800 reservoir)"
        )
    
    # Permute reservoir
    permutation = generate_calibration_permutation(
        3800, client_id, calibration_seed, "diad"
    )
    reservoir_permuted = reservoir_df.iloc[permutation].copy()
    
    # Split reservoir into R, G, C, GUARD
    splits[DatasetRole.R] = reservoir_permuted.iloc[:300].copy()
    splits[DatasetRole.R]["_role"] = DatasetRole.R.value
    
    splits[DatasetRole.G] = reservoir_permuted.iloc[300:1800].copy()
    splits[DatasetRole.G]["_role"] = DatasetRole.G.value
    
    splits[DatasetRole.C] = reservoir_permuted.iloc[1800:3300].copy()
    splits[DatasetRole.C]["_role"] = DatasetRole.C.value
    
    splits[DatasetRole.GUARD] = reservoir_permuted.iloc[3300:3800].copy()
    splits[DatasetRole.GUARD]["_role"] = DatasetRole.GUARD.value
    
    # TEST_BENIGN: All remaining benign rows
    test_benign_df = benign_df.iloc[reservoir_end:].copy()
    if len(test_benign_df) < 2000:
        raise ValueError(
            f"Client {client_id} has only {len(test_benign_df)} final benign rows, "
            f"but minimum 2000 required"
        )
    test_benign_df["_role"] = DatasetRole.TEST_BENIGN.value
    splits[DatasetRole.TEST_BENIGN] = test_benign_df
    
    # Malicious splitting per Section 7.2.3
    attack_splits = generate_attack_splits(malicious_df, client_id, "diad")
    splits[DatasetRole.DEV_ATTACK] = attack_splits[DatasetRole.DEV_ATTACK]
    splits[DatasetRole.TEST_ATTACK] = attack_splits[DatasetRole.TEST_ATTACK]
    
    return splits


def generate_attack_splits(
    malicious_df: pd.DataFrame,
    client_id: str,
    dataset_id: str,
) -> Dict[DatasetRole, pd.DataFrame]:
    """
    Generate attack development and test splits.
    
    For N-BaIoT (Section 7.1.3):
    - Identify attack subtypes from files
    - Allocate exactly 500 records to A_dev, category-balanced
    - Remaining to A_test
    
    For DIAD (Section 7.2.3):
    - Category-balanced allocation with capacity constraints
    - Reserve final-test evidence before development sampling
    
    This is a placeholder implementation. Full implementation requires
    parsing attack category/label columns from the dataset.
    
    Args:
        malicious_df: DataFrame with all malicious data
        client_id: Client identifier
        dataset_id: Dataset identifier
        
    Returns:
        Dictionary with DEV_ATTACK and TEST_ATTACK DataFrames
    """
    # Placeholder: split malicious data into dev and test
    # Full implementation needs to:
    # 1. Identify attack categories/subtypes
    # 2. For N-BaIoT: m_k subtypes, allocate floor(500/m_k) each, then remainder
    # 3. For DIAD: reserve final-test, then allocate with water-filling
    
    # For now, simple split (this will be replaced with proper implementation)
    total_malicious = len(malicious_df)
    
    if total_malicious < 500:
        raise ValueError(
            f"Client {client_id} has only {total_malicious} malicious rows, "
            f"but 500 required for A_dev"
        )
    
    # Take first 500 for development, rest for test
    # Note: This is NOT the correct implementation per roadmap
    # The correct implementation needs category-balanced sampling
    dev_df = malicious_df.iloc[:500].copy()
    dev_df["_role"] = DatasetRole.DEV_ATTACK.value
    
    test_df = malicious_df.iloc[500:].copy()
    test_df["_role"] = DatasetRole.TEST_ATTACK.value
    
    return {
        DatasetRole.DEV_ATTACK: dev_df,
        DatasetRole.TEST_ATTACK: test_df,
    }


def verify_disjointness(
    splits: Dict[DatasetRole, pd.DataFrame],
    row_id_column: str = "row_id",
) -> Dict[DatasetRole, Set[str]]:
    """
    Verify that all splits have disjoint row IDs.
    
    Per Section 7.1.4: T_k, reservoir, B_k, A_dev,k, and A_test,k are pairwise
    disjoint by row_id.
    
    Args:
        splits: Dictionary mapping roles to DataFrames
        row_id_column: Name of the row ID column
        
    Returns:
        Dictionary mapping roles to their row ID sets
        
    Raises:
        ValueError: If any row IDs overlap between splits
    """
    role_row_ids = {}
    all_row_ids = set()
    
    for role, df in splits.items():
        if role_id_column not in df.columns:
            raise ValueError(f"Row ID column '{row_id_column}' not found in {role} split")
        
        row_ids = set(df[row_id_column].unique())
        role_row_ids[role] = row_ids
        
        # Check for overlap with previously seen roles
        overlap = all_row_ids & row_ids
        if overlap:
            raise ValueError(
                f"Row ID overlap detected: {len(overlap)} rows in {role} "
                f"also appear in other splits. Overlapping IDs: {list(overlap)[:10]}..."
            )
        
        all_row_ids.update(row_ids)
    
    return role_row_ids


def verify_benign_attack_separation(
    benign_splits: Dict[DatasetRole, pd.DataFrame],
    attack_splits: Dict[DatasetRole, pd.DataFrame],
    row_id_column: str = "row_id",
) -> None:
    """
    Verify that benign and attack splits have no overlapping row IDs.
    
    Per Section 7.1.4: no attack row is present in T, reservoir, or B.
    
    Args:
        benign_splits: Dictionary of benign role splits
        attack_splits: Dictionary of attack role splits
        row_id_column: Name of the row ID column
        
    Raises:
        ValueError: If any attack row IDs appear in benign splits
    """
    benign_ids = set()
    for role, df in benign_splits.items():
        if row_id_column in df.columns:
            benign_ids.update(df[row_id_column].unique())
    
    attack_ids = set()
    for role, df in attack_splits.items():
        if row_id_column in df.columns:
            attack_ids.update(df[row_id_column].unique())
    
    overlap = benign_ids & attack_ids
    if overlap:
        raise ValueError(
            f"Benign-attack overlap detected: {len(overlap)} attack row IDs "
            f"appear in benign splits. This violates Section 7.1.4."
        )
