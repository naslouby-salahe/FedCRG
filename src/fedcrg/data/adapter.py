"""Dataset adapter contract."""
from abc import ABC,abstractmethod
from pathlib import Path
from fedcrg.core.enums import DatasetId
from fedcrg.data.models import ClientData
class DatasetAdapter(ABC):
    def __init__(self,root:Path|str)->None:self.root=Path(root)
    @property
    @abstractmethod
    def dataset_id(self)->DatasetId:raise NotImplementedError
    @abstractmethod
    def discover_clients(self)->tuple[str,...]:raise NotImplementedError
    @abstractmethod
    def load_client(self,client_id:str)->ClientData:raise NotImplementedError
