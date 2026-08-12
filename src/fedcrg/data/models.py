"""Typed dataset records and split containers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fedcrg.core.enums import DataRole, DatasetId


@dataclass(frozen=True, slots=True)
class ClientData:
    dataset: DatasetId
    client_id: str
    benign: pd.DataFrame
    attack: pd.DataFrame


@dataclass(frozen=True, slots=True)
class ClientSplits:
    client_id: str
    roles: dict[DataRole, pd.DataFrame]

    def get(self, role: DataRole) -> pd.DataFrame:
        return self.roles[role]
