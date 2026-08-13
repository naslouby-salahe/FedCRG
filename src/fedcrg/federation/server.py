"""Federated server state and aggregation."""

from collections.abc import Sequence

from fedcrg.detectors.detector import DetectorModel
from fedcrg.federation.aggregation import EqualMeanAggregator


class FederatedServer:
    def __init__(
        self,
        initial_model: DetectorModel,
        aggregator: EqualMeanAggregator | None = None,
    ) -> None:
        self.model = initial_model.clone()
        self.aggregator = aggregator or EqualMeanAggregator()

    def broadcast(self) -> DetectorModel:
        return self.model.clone()

    def aggregate(self, client_models: Sequence[DetectorModel]) -> str:
        self.aggregator.aggregate_into(self.model, client_models)
        return self.model.state_hash()
