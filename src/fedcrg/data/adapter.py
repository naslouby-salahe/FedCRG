"""Natural-client dataset adapter boundary."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from fedcrg.core.enums import DatasetId
from fedcrg.core.ids import ClientId
from fedcrg.data.models import ClientData


class DatasetAdapter(ABC):
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()

    @property
    @abstractmethod
    def dataset_id(self) -> DatasetId:
        raise NotImplementedError

    @abstractmethod
    def discover_clients(self) -> tuple[ClientId, ...]:
        raise NotImplementedError

    @abstractmethod
    def load_client(self, client_id: ClientId) -> ClientData:
        raise NotImplementedError

    @abstractmethod
    def source_files(self) -> tuple[Path, ...]:
        raise NotImplementedError
