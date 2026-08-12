"""N-BaIoT natural-device adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fedcrg.core.constants import NBAIOT_CLIENT_IDS, NBAIOT_EXPECTED_FEATURES
from fedcrg.core.enums import DatasetId
from fedcrg.core.exceptions import DataIntegrityError
from fedcrg.data.adapter import DatasetAdapter
from fedcrg.data.discovery import DatasetDiscovery
from fedcrg.data.models import ClientData
from fedcrg.data.splitting import stable_row_id


class NBaiotAdapter(DatasetAdapter):
    """Load the nine official N-BaIoT device clients in deterministic source order."""

    @property
    def dataset_id(self) -> DatasetId:
        return DatasetId.NBAIOT

    def discover_clients(self) -> tuple[str, ...]:
        directories = DatasetDiscovery.directories(self.root)
        if len(directories) != len(NBAIOT_CLIENT_IDS):
            raise DataIntegrityError(f"DATASET_COUNT_MISMATCH: expected {len(NBAIOT_CLIENT_IDS)} device directories, found {len(directories)}")
        return NBAIOT_CLIENT_IDS

    def load_client(self, client_id: str) -> ClientData:
        if client_id not in NBAIOT_CLIENT_IDS:
            raise DataIntegrityError(f"Unknown N-BaIoT client id: {client_id}")
        directories = DatasetDiscovery.directories(self.root)
        directory = directories[NBAIOT_CLIENT_IDS.index(client_id)]
        files = DatasetDiscovery.csv_files(directory)
        benign_files = tuple(path for path in files if "benign" in path.name.lower())
        attack_files = tuple(path for path in files if path not in benign_files)
        if not benign_files or not attack_files:
            raise DataIntegrityError(f"Client {client_id} must expose benign and attack CSV files")
        benign = self._load_files(benign_files, client_id, attack_group=None)
        attacks = [self._load_files((path,), client_id, attack_group=self._attack_group(path)) for path in attack_files]
        return ClientData(dataset=self.dataset_id, client_id=client_id, benign=benign, attack=pd.concat(attacks, ignore_index=True))

    def _load_files(self, files: tuple[Path, ...], client_id: str, attack_group: str | None) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for path in files:
            frame = pd.read_csv(path)
            if frame.shape[1] != NBAIOT_EXPECTED_FEATURES:
                raise DataIntegrityError(f"{path}: expected {NBAIOT_EXPECTED_FEATURES} model columns, found {frame.shape[1]}")
            numeric = frame.apply(pd.to_numeric, errors="raise")
            values = numeric.to_numpy(dtype=np.float64, copy=False)
            if not np.isfinite(values).all():
                raise DataIntegrityError(f"{path}: N-BaIoT model features must all be finite")
            source = path.relative_to(self.root).as_posix()
            numeric["row_id"] = [stable_row_id(self.dataset_id.value, client_id, source, index) for index in range(len(numeric))]
            numeric["source_file"] = source
            numeric["source_row_index"] = np.arange(len(numeric), dtype=np.int64)
            if attack_group is not None:
                numeric["attack_group"] = attack_group
            frames.append(numeric)
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _attack_group(path: Path) -> str:
        lower = path.stem.lower()
        family = "mirai" if "mirai" in path.as_posix().lower() else "gafgyt"
        subtype = next((name for name in ("udpplain", "combo", "junk", "scan", "tcp", "udp", "ack", "syn") if name in lower), lower)
        return f"{family}:{subtype}"
