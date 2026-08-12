"""
DIAD Dataset Adapter

Provides data loading and partitioning for the CIC IoT-DIAD dataset.

Normative reference: Section 7.2 (CIC IoT-DIAD 2024 — external validation)
"""

from __future__ import annotations

import glob
import hashlib
import os
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
    BENIGN_ROLES,
    ATTACK_ROLES,
)
from fedcrg.data.manifest import (
    FileEntry,
    SplitInfo,
    ClientManifest,
)
from fedcrg.data.splitting import (
    generate_diad_splits,
    verify_disjointness,
    verify_benign_attack_separation,
)
from fedcrg.config import DatasetID


# DIAD feature contract: exactly 86 numeric features
DIAD_FEATURES = [
    # 11 base features
    "inter_arrival_time",
    "time_since_previously_displayed_frame",
    "l4_tcp",
    "l4_udp",
    "ttl",
    "eth_size",
    "tcp_window_size",
    "payload_entropy",
    "payload_length",
    "l3_ip_dst_count",
    "jitter",
    # 75 windowed features (15 per window: 1, 5, 10, 30, 60)
    # This is a subset - full list in Section 7.3
]

# Full 86-feature list would be:
# 11 base + 75 windowed = 86
# For now, we use the base features and note that the full list
# is specified in Section 7.3

# DIAD attack categories (7 categories from official schema)
DIAD_ATTACK_CATEGORIES = [
    "ddos",
    "dos",
    "infiltration",
    "mitm",
    "password",
    "scanning",
    "vulnerability",
]


class DiadAdapter(BaseDatasetAdapter):
    """
    Dataset adapter for CIC IoT-DIAD 2024.
    
    Implements data loading, partitioning, and integrity verification
    per Section 7.2.
    """
    
    dataset_id = DatasetID.DIAD.value
    expected_features = 86
    expected_clients = None  # Variable - determined by eligibility
    
    # Eligibility thresholds
    min_benign_rows = 7800
    min_malicious_rows = 1000
    min_final_attack_rows = 500
    min_attack_test_rows_per_category = 100
    min_clients = 10
    
    def __init__(self, data_root: Path | str | None = None):
        """
        Initialize the DIAD adapter.
        
        Args:
            data_root: Root path to DIAD data
        """
        super().__init__(data_root)
    
    def discover_clients(self) -> List[str]:
        """
        Discover available DIAD device directories.
        
        Per Section 7.2.1: Each device is mapped to a stable ID.
        
        Returns:
            List of device client IDs
        """
        # Look for device directories in data root
        device_dirs = []
        for entry in self.data_root.iterdir():
            if entry.is_dir():
                # Check if it contains benign data
                benign_dir = entry / "benign"
                if benign_dir.exists():
                    device_dirs.append(entry.name)
        
        if not device_dirs:
            raise FileNotFoundError(
                f"No DIAD device directories found in {self.data_root}"
            )
        
        return sorted(device_dirs)
    
    def _normalize_device_mac(self, device_mac: str) -> str:
        """
        Normalize device MAC address per Section 7.2.1.
        
        Args:
            device_mac: Raw device MAC address
            
        Returns:
            Normalized MAC address (lowercase, trimmed)
        """
        return device_mac.strip().lower()
    
    def _generate_device_id(self, device_mac: str) -> str:
        """
        Generate stable device ID per Section 7.2.1.
        
        diad_<sha256(normalized_device_mac)[:12]>
        
        Args:
            device_mac: Device MAC address
            
        Returns:
            Stable device ID
        """
        normalized = self._normalize_device_mac(device_mac)
        hash_hex = hashlib.sha256(normalized.encode()).hexdigest()
        return f"diad_{hash_hex[:12]}"
    
    def get_eligibility(self) -> Dict[str, Any]:
        """
        Determine client eligibility per Section 7.2.4.
        
        Returns:
            Dictionary with eligibility information:
            - eligible_clients: List of eligible client IDs
            - all_clients: List of all discovered client IDs
            - exclusion_reasons: Dict mapping excluded client to reason code
        """
        all_clients = self.discover_clients()
        eligible_clients = []
        exclusion_reasons = {}
        
        for client_id in all_clients:
            try:
                benign_df = self.load_client_benign(client_id)
                malicious_df = self.load_client_malicious(client_id)
                
                benign_count = len(benign_df)
                malicious_count = len(malicious_df)
                
                # Check eligibility criteria
                if benign_count < self.min_benign_rows:
                    exclusion_reasons[client_id] = "BENIGN_COUNT_LT_7800"
                    continue
                
                if malicious_count < self.min_malicious_rows:
                    exclusion_reasons[client_id] = "MALICIOUS_COUNT_LT_1000"
                    continue
                
                # Check per-category development capacity
                # This requires parsing attack categories
                # For now, assume eligible if enough total malicious
                # Full implementation needs category parsing
                eligible_clients.append(client_id)
                
            except Exception as e:
                exclusion_reasons[client_id] = f"ERROR:{str(e)}"
        
        return {
            "eligible_clients": sorted(eligible_clients),
            "all_clients": sorted(all_clients),
            "exclusion_reasons": exclusion_reasons,
            "min_benign_rows": self.min_benign_rows,
            "min_malicious_rows": self.min_malicious_rows,
        }
    
    def load_client_benign(
        self,
        client_id: str,
        features_only: bool = False,
    ) -> pd.DataFrame:
        """
        Load benign data for a DIAD client.
        
        Per Section 7.2.1-7.2.2: Establishes stable benign ordering.
        
        Args:
            client_id: Client identifier
            features_only: If True, return only feature columns
            
        Returns:
            DataFrame with benign data, with row_id column added
        """
        client_dir = self.data_root / client_id / "benign"
        
        if not client_dir.exists():
            raise FileNotFoundError(
                f"No benign directory for DIAD client {client_id} at {client_dir}"
            )
        
        # Find all CSV files
        csv_files = sorted(glob.glob(str(client_dir / "*.csv")))
        
        if not csv_files:
            raise FileNotFoundError(
                f"No benign CSV files found for DIAD client {client_id}"
            )
        
        # Load and concatenate all benign files
        all_dfs = []
        global_row_index = 0
        
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            
            # Select only the 86 features (placeholder - full implementation
            # needs to select specific columns per Section 7.3)
            # For now, use all numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < self.expected_features:
                raise ValueError(
                    f"Client {client_id}, file {csv_file}: only "
                    f"{len(numeric_cols)} numeric columns, need {self.expected_features}"
                )
            
            # For now, use first 86 numeric columns
            if len(numeric_cols) > self.expected_features:
                numeric_cols = numeric_cols[:self.expected_features]
            
            df = df[numeric_cols].copy()
            
            # Check for NaN/inf
            if df.isnull().values.any() or np.isinf(df.values).any():
                # Check finite rate per Section 7.2.4
                finite_rate = df.notna().all(axis=1).sum() / len(df)
                if finite_rate < 0.99:
                    raise ValueError(
                        f"Client {client_id}: finite rate {finite_rate:.2%} < 99% "
                        f"(FINITE_RATE_FAIL)"
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
            
            df["row_id"] = row_ids
            df["_source_file"] = rel_path
            df["_source_row_index"] = range(len(df))
            
            all_dfs.append(df)
            global_row_index += len(df)
        
        result = pd.concat(all_dfs, ignore_index=True)
        
        # Sort by capture time if available, otherwise by source order
        # Per Section 7.2.1: use verified_chronology if capture time available
        if "capture_time" in result.columns:
            result = result.sort_values("capture_time").reset_index(drop=True)
            result["_verified_chronology"] = True
        else:
            result["_verified_chronology"] = False
        
        if features_only:
            feature_cols = [c for c in result.columns 
                          if not c.startswith("_") and c != "row_id"]
            return result[feature_cols]
        
        return result
    
    def load_client_malicious(
        self,
        client_id: str,
        features_only: bool = False,
    ) -> pd.DataFrame:
        """
        Load malicious data for a DIAD client.
        
        Args:
            client_id: Client identifier
            features_only: If True, return only feature columns
            
        Returns:
            DataFrame with malicious data
        """
        client_dir = self.data_root / client_id / "malicious"
        
        if not client_dir.exists():
            raise FileNotFoundError(
                f"No malicious directory for DIAD client {client_id} at {client_dir}"
            )
        
        # Find all CSV files
        csv_files = sorted(glob.glob(str(client_dir / "*.csv")))
        
        if not csv_files:
            return pd.DataFrame()
        
        # Load and concatenate all malicious files
        all_dfs = []
        global_row_index = 0
        
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            
            # Select only the 86 features
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < self.expected_features:
                raise ValueError(
                    f"Client {client_id}, malicious file {csv_file}: only "
                    f"{len(numeric_cols)} numeric columns, need {self.expected_features}"
                )
            
            if len(numeric_cols) > self.expected_features:
                numeric_cols = numeric_cols[:self.expected_features]
            
            df = df[numeric_cols].copy()
            
            # Check for NaN/inf
            if df.isnull().values.any() or np.isinf(df.values).any():
                finite_rate = df.notna().all(axis=1).sum() / len(df)
                if finite_rate < 0.99:
                    raise ValueError(
                        f"Client {client_id}: finite rate {finite_rate:.2%} < 99% "
                        f"(FINITE_RATE_FAIL)"
                    )
            
            # Extract attack category from filename or label column
            # This is a placeholder - full implementation needs proper parsing
            filename = os.path.basename(csv_file).replace(".csv", "")
            
            # Try to extract category from filename
            attack_category = "unknown"
            for cat in DIAD_ATTACK_CATEGORIES:
                if cat in filename.lower():
                    attack_category = cat
                    break
            
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
            
            df["row_id"] = row_ids
            df["_source_file"] = rel_path
            df["_source_row_index"] = range(len(df))
            df["attack_category"] = attack_category
            
            all_dfs.append(df)
            global_row_index += len(df)
        
        if not all_dfs:
            return pd.DataFrame()
        
        result = pd.concat(all_dfs, ignore_index=True)
        
        if features_only:
            feature_cols = [c for c in result.columns 
                          if not c.startswith("_") and c != "row_id"]
            return result[feature_cols]
        
        return result
    
    def generate_splits(
        self,
        client_id: str,
        calibration_seed: int,
    ) -> Dict[DatasetRole, pd.DataFrame]:
        """
        Generate role-based splits for a DIAD client.
        
        Implements Section 7.2.2 (benign partition) and 7.2.3 (malicious partition).
        
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
        splits = generate_diad_splits(
            benign_df,
            malicious_df,
            client_id,
            calibration_seed,
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
        Verify DIAD dataset integrity per Section 7.2.4.
        
        Returns:
            DatasetIntegrityReport with all check results
        """
        # Check eligibility
        eligibility = self.get_eligibility()
        
        clients = eligibility["eligible_clients"]
        all_clients = eligibility["all_clients"]
        exclusion_reasons = eligibility["exclusion_reasons"]
        
        checks = []
        client_reports = {}
        
        # Check: At least 10 eligible clients
        if len(clients) < self.min_clients:
            checks.append(DataIntegrityCheck(
                check_name="min_eligible_clients",
                passed=False,
                message=f"Only {len(clients)} eligible clients, need {self.min_clients}",
                details={"eligible": len(clients), "required": self.min_clients},
            ))
        else:
            checks.append(DataIntegrityCheck(
                check_name="min_eligible_clients",
                passed=True,
                message=f"Found {len(clients)} eligible clients (>= {self.min_clients})",
                details={"eligible": len(clients), "required": self.min_clients},
            ))
        
        # Check each client
        for client_id in all_clients:
            client_checks = []
            row_counts = {}
            
            try:
                benign_df = self.load_client_benign(client_id)
                row_counts["benign"] = len(benign_df)
                
                malicious_df = self.load_client_malicious(client_id)
                row_counts["malicious"] = len(malicious_df)
                
                # Check benign count
                if len(benign_df) < self.min_benign_rows:
                    client_checks.append(DataIntegrityCheck(
                        check_name="benign_count",
                        passed=False,
                        message=f"Client {client_id}: {len(benign_df)} benign rows < {self.min_benign_rows}",
                        details={"actual": len(benign_df), "required": self.min_benign_rows},
                    ))
                else:
                    client_checks.append(DataIntegrityCheck(
                        check_name="benign_count",
                        passed=True,
                        message=f"Client {client_id}: {len(benign_df)} benign rows OK",
                        details={"actual": len(benign_df), "required": self.min_benign_rows},
                    ))
                
                # Check malicious count
                if len(malicious_df) < self.min_malicious_rows:
                    client_checks.append(DataIntegrityCheck(
                        check_name="malicious_count",
                        passed=False,
                        message=f"Client {client_id}: {len(malicious_df)} malicious rows < {self.min_malicious_rows}",
                        details={"actual": len(malicious_df), "required": self.min_malicious_rows},
                    ))
                else:
                    client_checks.append(DataIntegrityCheck(
                        check_name="malicious_count",
                        passed=True,
                        message=f"Client {client_id}: {len(malicious_df)} malicious rows OK",
                        details={"actual": len(malicious_df), "required": self.min_malicious_rows},
                    ))
                
                # Check feature count
                feature_cols = [c for c in benign_df.columns if not c.startswith("_")]
                if len(feature_cols) != self.expected_features:
                    client_checks.append(DataIntegrityCheck(
                        check_name="feature_count",
                        passed=False,
                        message=f"Client {client_id}: {len(feature_cols)} features != {self.expected_features}",
                        details={"actual": len(feature_cols), "required": self.expected_features},
                    ))
                else:
                    client_checks.append(DataIntegrityCheck(
                        check_name="feature_count",
                        passed=True,
                        message=f"Client {client_id}: {len(feature_cols)} features OK",
                        details={"actual": len(feature_cols), "required": self.expected_features},
                    ))
                
            except Exception as e:
                client_checks.append(DataIntegrityCheck(
                    check_name="data_loading",
                    passed=False,
                    message=f"Client {client_id}: error - {str(e)}",
                    details={"error": str(e)},
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
        Generate a manifest for a DIAD client.
        
        Args:
            client_id: Client identifier
            calibration_seeds: List of calibration seeds to document
            
        Returns:
            ClientManifest with all metadata
        """
        # Load data
        client_dir = self.data_root / client_id
        
        benign_files = sorted(glob.glob(str(client_dir / "benign" / "*.csv")))
        malicious_files = sorted(glob.glob(str(client_dir / "malicious" / "*.csv")))
        
        # Create file entries
        file_entries_benign = []
        for f in benign_files:
            fe = FileEntry.from_file(Path(f), self.data_root)
            file_entries_benign.append(fe)
        
        file_entries_malicious = []
        for f in malicious_files:
            fe = FileEntry.from_file(Path(f), self.data_root)
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
                row_ids=row_ids[:10] if row_ids else None,
            ))
            
            if row_ids:
                row_id_samples[role.value] = row_ids[0]
        
        return ClientManifest(
            client_id=client_id,
            dataset_id=self.dataset_id,
            benign_files=tuple(file_entries_benign),
            malicious_files=tuple(file_entries_malicious),
            total_benign_rows=len(benign_df),
            total_malicious_rows=len(self.load_client_malicious(client_id)) if self.load_client_malicious(client_id) is not None else 0,
            feature_names=feature_names,
            feature_count=len(feature_names),
            splits=tuple(split_infos),
            calibration_seeds=calibration_seeds,
            row_id_samples=row_id_samples,
        )
