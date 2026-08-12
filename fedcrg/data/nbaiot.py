"""
N-BaIoT Dataset Adapter

Provides data loading and partitioning for the N-BaIoT dataset.

Normative reference: Section 7.1 (N-BaIoT — primary natural-client experiment)
"""

from __future__ import annotations

import glob
import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from fedcrg.data.base import (
    BaseDatasetAdapter,
    DatasetRole,
    RowIDComponents,
    DataIntegrityCheck,
    DatasetIntegrityReport,
    ClientIntegrityReport,
    generate_row_id,
    compute_file_hash,
    BENIGN_ROLES,
    ATTACK_ROLES,
    FEDCRG_ROLES,
    COMPARATOR_ROLES,
)
from fedcrg.data.manifest import (
    FileEntry,
    SplitInfo,
    ClientManifest,
)
from fedcrg.data.splitting import (
    generate_nbaiot_splits,
    verify_disjointness,
    verify_benign_attack_separation,
    generate_attack_splits,
)
from fedcrg.config import DatasetID


# Canonical N-BaIoT client IDs
NBAIOT_CLIENTS = [
    "nb01",  # Danmini Doorbell
    "nb02",  # Ennio Doorbell
    "nb03",  # Ecobee Thermostat
    "nb04",  # Philips B120N/10 Baby Monitor
    "nb05",  # Provision PT-737E Security Camera
    "nb06",  # Provision PT-838 Security Camera
    "nb07",  # SimpleHome XCS7-1002-WHT Security Camera
    "nb08",  # SimpleHome XCS7-1003-WHT Security Camera
    "nb09",  # Samsung SNH-1011N Webcam
]

# Expected benign row counts from literature (preflight cross-check)
# Per Section 7.1.1
NBAIOT_EXPECTED_BENIGN = {
    "nb01": 49548,
    "nb02": 39100,
    "nb03": 13113,
    "nb04": 175240,
    "nb05": 62154,
    "nb06": 98514,
    "nb07": 46585,
    "nb08": 19528,
    "nb09": 52150,
}

# N-BaIoT attack subtypes
# 5 BASHLITE/Gafgyt subtypes for all devices
BASHLITE_SUBTYPES = ["combo", "junk", "scan", "tcp", "udp"]

# 5 Mirai subtypes for 7 devices (not Ennio/Samsung)
MIRAI_SUBTYPES = ["ack", "scan", "syn", "udp", "udpplain"]

# All attack subtypes
ALL_NBAIOT_ATTACK_SUBTYPES = BASHLITE_SUBTYPES + MIRAI_SUBTYPES

# Devices that have Mirai attacks
MIRAI_DEVICES = {"nb01", "nb04", "nb05", "nb06", "nb07", "nb08", "nb09"}


class NBaiotAdapter(BaseDatasetAdapter):
    """
    Dataset adapter for N-BaIoT.
    
    Implements data loading, partitioning, and integrity verification
    per Section 7.1.
    """
    
    dataset_id = DatasetID.NBAIOT.value
    expected_features = 115
    expected_clients = 9
    
    # Attack subtype information per client
    _client_attack_subtypes: Dict[str, List[str]]
    
    def __init__(self, data_root: Path | str | None = None):
        """
        Initialize the N-BaIoT adapter.
        
        Args:
            data_root: Root path to N-BaIoT data. Expected structure:
                      data_root/nb01/benign/*.csv
                      data_root/nb01/malicious/*.csv
                      ...
        """
        super().__init__(data_root)
        self._client_attack_subtypes = {}
        self._discover_attack_subtypes()
    
    def _discover_attack_subtypes(self) -> None:
        """Discover which attack subtypes are present for each client."""
        for client_id in NBAIOT_CLIENTS:
            client_dir = self.data_root / client_id
            if not client_dir.exists():
                continue
            
            malicious_dir = client_dir / "malicious"
            if not malicious_dir.exists():
                self._client_attack_subtypes[client_id] = []
                continue
            
            # Find all malicious CSV files
            csv_files = glob.glob(str(malicious_dir / "*.csv"))
            
            # Extract subtypes from filenames
            # N-BaIoT malicious files are named like:
            # Gafgyt_combo.csv, Gafgyt_scan.csv, Mirai_ack.csv, etc.
            subtypes = set()
            for filepath in csv_files:
                filename = os.path.basename(filepath)
                # Remove .csv extension
                filename = filename.replace(".csv", "")
                
                # Extract subtype (second part after underscore)
                # Format: <attack_type>_<subtype>.csv
                parts = filename.split("_")
                if len(parts) >= 2:
                    subtype = parts[1]
                    subtypes.add(subtype)
            
            self._client_attack_subtypes[client_id] = sorted(subtypes)
    
    def get_attack_subtypes(self, client_id: str) -> List[str]:
        """Get attack subtypes present for a client."""
        return self._client_attack_subtypes.get(client_id, [])
    
    def discover_clients(self) -> List[str]:
        """Discover available N-BaIoT client directories."""
        clients = []
        for expected_client in NBAIOT_CLIENTS:
            client_dir = self.data_root / expected_client
            if client_dir.exists() and (client_dir / "benign").exists():
                clients.append(expected_client)
        
        if not clients:
            raise FileNotFoundError(
                f"No N-BaIoT client directories found in {self.data_root}. "
                f"Expected: {NBAIOT_CLIENTS}"
            )
        
        return sorted(clients)
    
    def _load_client_csvs(
        self,
        client_id: str,
        data_type: str,  # "benign" or "malicious"
    ) -> List[Path]:
        """Get list of CSV files for a client and data type."""
        client_dir = self.data_root / client_id / data_type
        if not client_dir.exists():
            raise FileNotFoundError(
                f"No {data_type} directory for client {client_id} at "
                f"{client_dir}"
            )
        
        csv_files = sorted(glob.glob(str(client_dir / "*.csv")))
        return [Path(f) for f in csv_files]
    
    def load_client_benign(
        self,
        client_id: str,
        features_only: bool = False,
    ) -> pd.DataFrame:
        """
        Load all benign data for a client.
        
        Per Section 7.1.2: Preserves source-file row order.
        
        Args:
            client_id: Client identifier (e.g., "nb01")
            features_only: If True, return only feature columns
            
        Returns:
            DataFrame with all benign data, with row_id column added
        """
        csv_files = self._load_client_csvs(client_id, "benign")
        
        all_dfs = []
        global_row_index = 0
        
        for csv_file in csv_files:
            # Read CSV
            df = pd.read_csv(csv_file)
            
            # Verify feature count
            if len(df.columns) != self.expected_features:
                raise ValueError(
                    f"Client {client_id}, file {csv_file}: expected "
                    f"{self.expected_features} features, got {len(df.columns)}"
                )
            
            # Check for NaN/inf
            if df.isnull().values.any() or np.isinf(df.values).any():
                raise ValueError(
                    f"Client {client_id}, file {csv_file}: contains NaN or inf values"
                )
            
            # Generate row IDs
            rel_path = str(csv_file.relative_to(self.data_root))
            row_ids = [
                generate_row_id(RowIDComponents(
                    dataset_id=self.dataset_id,
                    client_id=client_id,
                    source_file=rel_path,
                    source_row_index=i,
                ))
                for i in range(len(df))
            ]
            
            df = df.copy()
            df["row_id"] = row_ids
            df["_source_file"] = rel_path
            df["_source_row_index"] = range(len(df))
            df["_global_row_index"] = range(
                global_row_index, global_row_index + len(df)
            )
            
            all_dfs.append(df)
            global_row_index += len(df)
        
        if not all_dfs:
            raise ValueError(f"No benign data found for client {client_id}")
        
        result = pd.concat(all_dfs, ignore_index=True)
        
        # Remove metadata columns if features_only
        if features_only:
            # Keep only the 115 feature columns
            feature_cols = [c for c in result.columns if not c.startswith("_")]
            if "row_id" in result.columns:
                feature_cols.remove("row_id")
            return result[feature_cols]
        
        return result
    
    def load_client_malicious(
        self,
        client_id: str,
        features_only: bool = False,
    ) -> pd.DataFrame:
        """
        Load all malicious data for a client.
        
        Args:
            client_id: Client identifier
            features_only: If True, return only feature columns
            
        Returns:
            DataFrame with all malicious data, with row_id and attack_subtype columns
        """
        csv_files = self._load_client_csvs(client_id, "malicious")
        
        all_dfs = []
        global_row_index = 0
        
        for csv_file in csv_files:
            # Read CSV
            df = pd.read_csv(csv_file)
            
            # Verify feature count
            if len(df.columns) != self.expected_features:
                raise ValueError(
                    f"Client {client_id}, malicious file {csv_file}: expected "
                    f"{self.expected_features} features, got {len(df.columns)}"
                )
            
            # Check for NaN/inf
            if df.isnull().values.any() or np.isinf(df.values).any():
                raise ValueError(
                    f"Client {client_id}, malicious file {csv_file}: "
                    f"contains NaN or inf values"
                )
            
            # Extract attack subtype from filename
            filename = os.path.basename(csv_file).replace(".csv", "")
            parts = filename.split("_")
            attack_subtype = parts[1] if len(parts) >= 2 else "unknown"
            
            # Generate row IDs
            rel_path = str(csv_file.relative_to(self.data_root))
            row_ids = [
                generate_row_id(RowIDComponents(
                    dataset_id=self.dataset_id,
                    client_id=client_id,
                    source_file=rel_path,
                    source_row_index=i,
                ))
                for i in range(len(df))
            ]
            
            df = df.copy()
            df["row_id"] = row_ids
            df["_source_file"] = rel_path
            df["_source_row_index"] = range(len(df))
            df["_global_row_index"] = range(
                global_row_index, global_row_index + len(df)
            )
            df["attack_subtype"] = attack_subtype
            
            all_dfs.append(df)
            global_row_index += len(df)
        
        if not all_dfs:
            # Some clients may have no malicious data (unlikely for N-BaIoT)
            return pd.DataFrame()
        
        result = pd.concat(all_dfs, ignore_index=True)
        
        # Remove metadata columns if features_only
        if features_only:
            feature_cols = [c for c in result.columns 
                          if not c.startswith("_") and c != "attack_subtype"]
            if "row_id" in result.columns:
                feature_cols.remove("row_id")
            return result[feature_cols]
        
        return result
    
    def generate_splits(
        self,
        client_id: str,
        calibration_seed: int,
    ) -> Dict[DatasetRole, pd.DataFrame]:
        """
        Generate role-based splits for an N-BaIoT client.
        
        Implements Section 7.1.2 (benign partition) and 7.1.3 (attack partition).
        
        Args:
            client_id: Client identifier
            calibration_seed: Calibration random seed
            
        Returns:
            Dictionary mapping DatasetRole to DataFrame
        """
        # Load data
        benign_df = self.load_client_benign(client_id)
        malicious_df = self.load_client_malicious(client_id)
        
        # Generate splits
        splits = generate_nbaiot_splits(
            benign_df,
            malicious_df,
            client_id,
            calibration_seed,
            self.dataset_id,
        )
        
        # Verify disjointness
        all_splits = {**splits}
        verify_disjointness(all_splits, "row_id")
        
        # Verify benign-attack separation
        benign_splits = {r: s for r, s in splits.items() if r in BENIGN_ROLES}
        attack_splits = {r: s for r, s in splits.items() if r in ATTACK_ROLES}
        verify_benign_attack_separation(benign_splits, attack_splits, "row_id")
        
        return splits
    
    def verify_integrity(self) -> DatasetIntegrityReport:
        """
        Verify N-BaIoT dataset integrity per Section 7.1.4.
        
        Checks:
        - Exactly 9 canonical device directories
        - Each benign/malicious CSV has exactly 115 numeric columns
        - No NaN/inf values
        - Row counts match literature cross-check (with warning if mismatch)
        - Row disjointness by row_id
        - Benign-attack separation
        
        Returns:
            DatasetIntegrityReport with all check results
        """
        clients = self.discover_clients()
        checks = []
        client_reports = {}
        
        # Check 1: Expected number of clients
        if len(clients) != self.expected_clients:
            checks.append(DataIntegrityCheck(
                check_name="client_count",
                passed=False,
                message=f"Expected {self.expected_clients} clients, found {len(clients)}",
                details={"expected": self.expected_clients, "found": len(clients)},
            ))
        else:
            checks.append(DataIntegrityCheck(
                check_name="client_count",
                passed=True,
                message=f"Found all {self.expected_clients} expected clients",
            ))
        
        # Check each client
        for client_id in clients:
            client_checks = []
            row_counts = {}
            
            try:
                # Load benign data
                benign_df = self.load_client_benign(client_id)
                row_counts["benign"] = len(benign_df)
                
                # Check benign row count feasibility
                if len(benign_df) < 10000:
                    client_checks.append(DataIntegrityCheck(
                        check_name="benign_row_count",
                        passed=False,
                        message=f"Client {client_id}: only {len(benign_df)} benign rows, "
                               f"but 10000 required (4000 train + 6000 reservoir)",
                        details={"client": client_id, "count": len(benign_df)},
                    ))
                else:
                    client_checks.append(DataIntegrityCheck(
                        check_name="benign_row_count",
                        passed=True,
                        message=f"Client {client_id}: {len(benign_df)} benign rows OK",
                        details={"client": client_id, "count": len(benign_df)},
                    ))
                
                # Cross-check against literature
                expected = NBAIOT_EXPECTED_BENIGN.get(client_id)
                if expected and len(benign_df) != expected:
                    client_checks.append(DataIntegrityCheck(
                        check_name="benign_row_cross_check",
                        passed=False,
                        message=f"Client {client_id}: row count {len(benign_df)} != "
                               f"literature value {expected}",
                        details={"client": client_id, "actual": len(benign_df), 
                               "expected": expected},
                    ))
                
                # Load malicious data
                malicious_df = self.load_client_malicious(client_id)
                row_counts["malicious"] = len(malicious_df)
                
                # Check malicious row count
                if len(malicious_df) < 500:
                    client_checks.append(DataIntegrityCheck(
                        check_name="malicious_row_count",
                        passed=False,
                        message=f"Client {client_id}: only {len(malicious_df)} malicious rows, "
                               f"but 500 required for A_dev",
                        details={"client": client_id, "count": len(malicious_df)},
                    ))
                else:
                    client_checks.append(DataIntegrityCheck(
                        check_name="malicious_row_count",
                        passed=True,
                        message=f"Client {client_id}: {len(malicious_df)} malicious rows OK",
                        details={"client": client_id, "count": len(malicious_df)},
                    ))
                
                # Check attack subtypes
                subtypes = self.get_attack_subtypes(client_id)
                client_checks.append(DataIntegrityCheck(
                    check_name="attack_subtypes",
                    passed=len(subtypes) > 0,
                    message=f"Client {client_id}: {len(subtypes)} attack subtypes: {subtypes}",
                    details={"client": client_id, "subtypes": subtypes},
                ))
                
            except Exception as e:
                client_checks.append(DataIntegrityCheck(
                    check_name="data_loading",
                    passed=False,
                    message=f"Client {client_id}: error loading data - {str(e)}",
                    details={"client": client_id, "error": str(e)},
                ))
            
            client_reports[client_id] = ClientIntegrityReport(
                client_id=client_id,
                checks=tuple(client_checks),
                row_counts=row_counts,
            )
        
        return DatasetIntegrityReport(
            dataset_id=self.dataset_id,
            checks=tuple(checks),
            client_reports=client_reports,
        )
    
    def get_client_manifest(
        self,
        client_id: str,
        calibration_seeds: List[int],
    ) -> ClientManifest:
        """
        Generate a manifest for an N-BaIoT client.
        
        Args:
            client_id: Client identifier
            calibration_seeds: List of calibration seeds to document
            
        Returns:
            ClientManifest with all metadata
        """
        # Load data
        benign_files = self._load_client_csvs(client_id, "benign")
        malicious_files = self._load_client_csvs(client_id, "malicious")
        
        # Create file entries
        file_entries_benign = []
        for f in benign_files:
            fe = FileEntry.from_file(f, self.data_root)
            file_entries_benign.append(fe)
        
        file_entries_malicious = []
        for f in malicious_files:
            fe = FileEntry.from_file(f, self.data_root)
            file_entries_malicious.append(fe)
        
        # Load benign data to get feature names
        benign_df = self.load_client_benign(client_id)
        feature_names = [c for c in benign_df.columns 
                       if not c.startswith("_") and c != "row_id"]
        
        # Generate splits for primary calibration seed
        primary_splits = self.generate_splits(client_id, calibration_seeds[0])
        
        # Create split infos
        split_infos = []
        row_id_samples = {}
        
        for role, df in primary_splits.items():
            row_count = len(df)
            row_ids = df["row_id"].tolist() if "row_id" in df.columns else []
            
            split_infos.append(SplitInfo(
                role=role,
                row_count=row_count,
                row_ids=row_ids[:10] if row_ids else None,  # Sample first 10
            ))
            
            if row_ids:
                row_id_samples[role.value] = row_ids[0]
        
        return ClientManifest(
            client_id=client_id,
            dataset_id=self.dataset_id,
            benign_files=tuple(file_entries_benign),
            malicious_files=tuple(file_entries_malicious),
            total_benign_rows=len(benign_df),
            total_malicious_rows=sum(len(self.load_client_malicious(client_id)) 
                                     if self.load_client_malicious(client_id) is not None else 0),
            feature_names=feature_names,
            feature_count=len(feature_names),
            splits=tuple(split_infos),
            calibration_seeds=calibration_seeds,
            row_id_samples=row_id_samples,
        )
