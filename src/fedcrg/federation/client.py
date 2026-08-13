"""Federated client local training with deterministic per-epoch shuffling."""

from __future__ import annotations

import hashlib

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from fedcrg.configuration.training_config import TrainingConfig
from fedcrg.domain.identifiers import ClientId, ModelSeed, Sha256
from fedcrg.detectors.autoencoder import Autoencoder
from fedcrg.detectors.detector import DetectorModel
from fedcrg.detectors.deep_svdd import DeepSvdd
from fedcrg.federation.training_results import ClientRoundResult


def epoch_seed(model_seed: ModelSeed, client_id: ClientId, round_index: int, epoch: int) -> int:
    text = f"fedcrg|training|{model_seed}|{client_id.value}|{round_index}|{epoch}"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) & 0x7FFFFFFFFFFFFFFF


class FederatedClient:
    """Own one local benign-training dataset and execute one round at a time."""

    def __init__(self, client_id: ClientId, dataset: TensorDataset, device: torch.device) -> None:
        self.client_id = client_id
        self.dataset = dataset
        self.device = device

    def train(
        self,
        global_model: DetectorModel,
        config: TrainingConfig,
        learning_rate: float,
        model_seed: ModelSeed,
        round_index: int,
    ) -> tuple[DetectorModel, ClientRoundResult]:
        model = global_model.clone().to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=config.adam_betas,
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay,
        )

        weighted_loss_sum = 0.0
        record_presentations = 0
        optimizer_steps = 0
        model.train()

        for local_epoch in range(config.local_epochs):
            generator = torch.Generator()
            generator.manual_seed(epoch_seed(model_seed, self.client_id, round_index, local_epoch))
            loader = DataLoader(
                self.dataset,
                batch_size=config.batch_size,
                shuffle=True,
                generator=generator,
                drop_last=False,
            )
            for batch in loader:
                values = batch[0] if isinstance(batch, (tuple, list)) else batch
                values = values.to(self.device, dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                loss = self._loss(model, values)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"TRAINING_NUMERICAL_FAILURE: non-finite loss for {self.client_id}"
                    )
                loss.backward()
                optimizer.step()
                self._assert_finite_parameters(model)

                batch_records = int(values.shape[0])
                weighted_loss_sum += float(loss.detach().cpu()) * batch_records
                record_presentations += batch_records
                optimizer_steps += 1

        if record_presentations == 0:
            raise RuntimeError(f"Client {self.client_id} produced no training records")
        mean_loss = weighted_loss_sum / record_presentations
        result = ClientRoundResult(
            client_id=self.client_id,
            mean_loss=mean_loss,
            record_presentations=record_presentations,
            optimizer_steps=optimizer_steps,
            model_hash=Sha256(model.state_hash()),
        )
        return model.cpu(), result

    @staticmethod
    def _loss(model: DetectorModel, values: Tensor) -> Tensor:
        if isinstance(model, Autoencoder):
            return torch.mean((model(values) - values) ** 2)
        if isinstance(model, DeepSvdd):
            return model.anomaly_score(values).mean()
        return model.anomaly_score(values).mean()

    @staticmethod
    def _assert_finite_parameters(model: DetectorModel) -> None:
        for name, parameter in model.named_parameters():
            if not torch.isfinite(parameter).all():
                raise FloatingPointError(f"TRAINING_NUMERICAL_FAILURE: non-finite parameter {name}")
