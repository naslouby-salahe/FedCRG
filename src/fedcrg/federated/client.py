"""Federated client local training with protocol-defined deterministic shuffling."""

from __future__ import annotations

import hashlib

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from fedcrg.config.models import TrainingConfig
from fedcrg.detectors.autoencoder import Autoencoder
from fedcrg.detectors.base import DetectorModel
from fedcrg.detectors.deep_svdd import DeepSvdd
from fedcrg.federated.models import ClientRoundResult


def _epoch_seed(model_seed: int, client_id: str, round_index: int, epoch: int) -> int:
    text = f"fedcrg|training|{model_seed}|{client_id}|{round_index}|{epoch}"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) & 0x7FFFFFFFFFFFFFFF


class FederatedClient:
    def __init__(self, client_id: str, dataset: Dataset[Tensor], device: torch.device) -> None:
        self.client_id = client_id
        self.dataset = dataset
        self.device = device

    def train(self, global_model: DetectorModel, config: TrainingConfig, learning_rate: float, model_seed: int, round_index: int) -> tuple[DetectorModel, ClientRoundResult]:
        model = global_model.clone().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=config.adam_betas, eps=config.adam_epsilon, weight_decay=config.weight_decay)
        total_loss = 0.0
        total_records = 0
        model.train()
        for epoch in range(config.local_epochs):
            generator = torch.Generator()
            generator.manual_seed(_epoch_seed(model_seed, self.client_id, round_index, epoch))
            loader = DataLoader(self.dataset, batch_size=config.batch_size, shuffle=True, generator=generator, drop_last=False)
            for batch in loader:
                values = batch[0] if isinstance(batch, (tuple, list)) else batch
                values = values.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                if isinstance(model, Autoencoder):
                    loss = torch.mean((model(values) - values) ** 2)
                elif isinstance(model, DeepSvdd):
                    loss = model.anomaly_score(values).mean()
                else:
                    loss = model.anomaly_score(values).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"TRAINING_NUMERICAL_FAILURE: {self.client_id}")
                loss.backward()
                optimizer.step()
                batch_size = int(values.shape[0])
                total_loss += float(loss.detach().cpu()) * batch_size
                total_records += batch_size
        mean_loss = total_loss / total_records if total_records else 0.0
        result = ClientRoundResult(client_id=self.client_id, mean_loss=mean_loss, model_hash=model.state_hash())
        return model.cpu(), result
