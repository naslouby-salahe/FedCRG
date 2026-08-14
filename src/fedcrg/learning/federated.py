"""Deterministic federated training: local epochs, equal-client aggregation,
cosine learning-rate schedule, participation, and protocol diagnostics."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from fedcrg.config import TrainingConfig
from fedcrg.learning.detectors import Autoencoder, DeepSvdd, DetectorModel
from fedcrg.runtime import resolve_compute_device
from fedcrg.types import (
    ByteCount,
    ClientFraction,
    ClientId,
    Correlation,
    EpochCount,
    FeatureCount,
    LearningRate,
    Loss,
    MetricDifference,
    ModelSeed,
    ParameterCount,
    PositiveCount,
    RngSeed,
    RoundCount,
    RoundIndex,
    Sha256,
)

Frozen = ConfigDict(frozen=True)

_FLOAT32_BYTES = 4
_FLOAT64_BYTES = 8
_INT64_BYTES = 8


class ModelCommunicationLedger(BaseModel):
    """Deterministic federated model-tensor communication accounting."""

    model_config = Frozen

    client_count: PositiveCount
    rounds: RoundCount
    trainable_parameters: ParameterCount
    model_payload_bytes: ByteCount
    federation_bytes: ByteCount


def model_communication(
    client_count: PositiveCount,
    rounds: RoundCount,
    trainable_parameters: ParameterCount,
) -> ModelCommunicationLedger:
    """Model tensor payload and 30-round full-federation exchange.

    Each round exchanges one server-to-client broadcast copy and one
    client-to-server upload per client, so the federation total is
    ``2 * client_count * rounds * payload`` with a float32 payload of
    ``4 * trainable_parameters`` bytes.
    """
    if client_count <= 0 or rounds <= 0 or trainable_parameters <= 0:
        raise ValueError("Communication accounting requires positive inputs")
    payload = trainable_parameters * _FLOAT32_BYTES
    return ModelCommunicationLedger(
        client_count=client_count,
        rounds=rounds,
        trainable_parameters=trainable_parameters,
        model_payload_bytes=payload,
        federation_bytes=2 * client_count * rounds * payload,
    )


class PreprocessingCommunicationLedger(BaseModel):
    """Deterministic federated min/max extrema exchange accounting."""

    model_config = Frozen

    client_count: PositiveCount
    feature_count: FeatureCount
    bytes_per_client: ByteCount
    federation_upload_bytes: ByteCount


def preprocessing_communication(
    client_count: PositiveCount,
    feature_count: FeatureCount,
) -> PreprocessingCommunicationLedger:
    """Per-feature extrema exchange: 2 float64 values per feature."""
    if client_count <= 0 or feature_count <= 0:
        raise ValueError("Communication accounting requires positive inputs")
    bytes_per_client = 2 * feature_count * _FLOAT64_BYTES
    return PreprocessingCommunicationLedger(
        client_count=client_count,
        feature_count=feature_count,
        bytes_per_client=bytes_per_client,
        federation_upload_bytes=client_count * bytes_per_client,
    )


def cosine_learning_rate(
    round_index: RoundIndex,
    rounds: RoundCount,
    initial: LearningRate,
    final: LearningRate,
) -> LearningRate:
    """Cosine learning-rate schedule between locked endpoints."""
    if rounds <= 0 or not 0 <= round_index < rounds:
        raise ValueError("round_index must be inside the configured training horizon")
    if rounds == 1:
        return final
    progress = round_index / (rounds - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return final + (initial - final) * cosine


def epoch_seed(
    model_seed: ModelSeed,
    client_id: ClientId,
    round_index: RoundIndex,
    epoch: EpochCount,
) -> RngSeed:
    """Deterministic per-epoch RNG seed."""
    text = f"fedcrg|training|{model_seed}|{client_id}|{round_index}|{epoch}"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) & 0x7FFFFFFFFFFFFFFF


StateDict = dict[str, torch.Tensor]


def equal_client_mean(models: Sequence[DetectorModel]) -> StateDict:
    """Equal-weight mean of client model parameters."""
    if not models:
        raise ValueError("At least one client model is required")
    states = [model.state_dict() for model in models]
    keys = tuple(states[0].keys())
    if any(tuple(state.keys()) != keys for state in states[1:]):
        raise ValueError("Client model state dictionaries do not match")
    result: StateDict = {}
    for key in keys:
        tensors = [state[key].detach() for state in states]
        shape = tensors[0].shape
        dtype = tensors[0].dtype
        if any(tensor.shape != shape for tensor in tensors[1:]):
            raise ValueError(f"Client tensor shapes differ for {key}")
        if any(tensor.dtype != dtype for tensor in tensors[1:]):
            raise ValueError(f"Client tensor dtypes differ for {key}")
        if dtype.is_floating_point:
            if any(not torch.isfinite(tensor).all() for tensor in tensors):
                raise FloatingPointError(f"TRAINING_NUMERICAL_FAILURE: non-finite tensor {key}")
            result[key] = torch.stack(tensors, dim=0).mean(dim=0)
        else:
            if any(not torch.equal(tensors[0], tensor) for tensor in tensors[1:]):
                raise ValueError(f"Non-floating state differs across clients: {key}")
            result[key] = tensors[0].clone()
    return result


class ClientRoundResult(BaseModel):
    """One client's local-round training outcome."""

    model_config = Frozen

    client_id: ClientId
    mean_loss: Loss
    record_presentations: PositiveCount
    optimizer_steps: PositiveCount
    model_hash: Sha256


class RoundResult(BaseModel):
    """One federated round's aggregate outcome."""

    model_config = Frozen

    round_index: RoundIndex
    learning_rate: LearningRate
    selected_clients: tuple[ClientId, ...]
    client_results: tuple[ClientRoundResult, ...]
    mean_client_loss: Loss
    minimum_client_loss: Loss
    maximum_client_loss: Loss
    parameter_update_norm: MetricDifference
    model_payload_bytes: ByteCount
    round_communication_bytes: ByteCount
    global_model_hash: Sha256


class TrainingResult(BaseModel):
    """Frozen federated-training evidence for one model seed."""

    model_config = Frozen

    model_seed: ModelSeed
    rounds: tuple[RoundResult, ...]
    final_model_hash: Sha256
    trainable_parameter_count: PositiveCount
    model_payload_bytes: ByteCount
    total_model_communication_bytes: ByteCount
    round20_training_score_correlation: Correlation | None = None


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
        learning_rate: LearningRate,
        model_seed: ModelSeed,
        round_index: RoundIndex,
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
            model_hash=model.state_hash(),
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


class ClientSampler:
    """Select configured client participation without dependence on input ordering."""

    def __init__(
        self, client_ids: tuple[ClientId, ...], fraction: ClientFraction, seed: ModelSeed
    ) -> None:
        if not client_ids:
            raise ValueError("At least one client is required")
        if not 0.0 < fraction <= 1.0:
            raise ValueError("Client fraction must be in (0, 1]")
        self.client_ids = tuple(sorted(client_ids))
        self.count = max(1, math.ceil(len(self.client_ids) * fraction))
        self.rng = np.random.default_rng(int(seed))

    def select(self) -> tuple[ClientId, ...]:
        if self.count == len(self.client_ids):
            return self.client_ids
        indices = self.rng.choice(len(self.client_ids), size=self.count, replace=False)
        return tuple(sorted(self.client_ids[int(index)] for index in indices))


class FederatedServer:
    """Deterministic federated training orchestration."""

    def __init__(self, initial_model: DetectorModel) -> None:
        self.model = initial_model.clone()

    def broadcast(self) -> DetectorModel:
        return self.model.clone()

    def aggregate(self, client_models: Sequence[DetectorModel]) -> Sha256:
        self.model.load_state_dict(equal_client_mean(client_models), strict=True)
        return self.model.state_hash()


class FederatedTrainer:
    """Execute the frozen full-participation equal-client-mean training protocol."""

    diagnostic_round_index = 19

    def train(
        self,
        model: DetectorModel,
        datasets: Mapping[ClientId, TensorDataset],
        config: TrainingConfig,
        model_seed: ModelSeed,
    ) -> tuple[DetectorModel, TrainingResult]:
        if not datasets:
            raise ValueError("At least one federated client is required")

        self._configure_determinism(model_seed, config)
        device = resolve_compute_device(config.device)
        client_ids = tuple(sorted(datasets))
        clients = {
            client_id: FederatedClient(client_id, datasets[client_id], device)
            for client_id in client_ids
        }
        sampler = ClientSampler(client_ids, config.client_fraction, model_seed)
        server = FederatedServer(model)

        round_results: list[RoundResult] = []
        round20_model: DetectorModel | None = None
        model_payload_bytes = model.trainable_tensor_bytes()

        for round_index in range(config.rounds):
            round_result = self._train_round(
                server=server,
                clients=clients,
                sampler=sampler,
                config=config,
                model_seed=model_seed,
                round_index=round_index,
                model_payload_bytes=model_payload_bytes,
            )
            round_results.append(round_result)
            if (
                config.record_round20_score_correlation
                and round_index == self.diagnostic_round_index
            ):
                round20_model = server.broadcast().cpu()

        final_model = server.broadcast().cpu()
        score_correlation = self._round20_score_correlation(
            round20_model,
            final_model,
            datasets,
        )
        total_communication = sum(item.round_communication_bytes for item in round_results)
        result = TrainingResult(
            model_seed=model_seed,
            rounds=tuple(round_results),
            final_model_hash=final_model.state_hash(),
            trainable_parameter_count=final_model.trainable_parameter_count(),
            model_payload_bytes=model_payload_bytes,
            total_model_communication_bytes=total_communication,
            round20_training_score_correlation=score_correlation,
        )
        return final_model, result

    def _train_round(
        self,
        server: FederatedServer,
        clients: dict[ClientId, FederatedClient],
        sampler: ClientSampler,
        config: TrainingConfig,
        model_seed: ModelSeed,
        round_index: RoundIndex,
        model_payload_bytes: ByteCount,
    ) -> RoundResult:
        learning_rate = cosine_learning_rate(
            round_index,
            config.rounds,
            config.learning_rate_initial,
            config.learning_rate_final,
        )
        selected_clients = sampler.select()
        before = server.broadcast().cpu()
        global_model = server.broadcast()
        trained_models: list[DetectorModel] = []
        client_results: list[ClientRoundResult] = []

        for client_id in selected_clients:
            trained_model, client_result = clients[client_id].train(
                global_model=global_model,
                config=config,
                learning_rate=learning_rate,
                model_seed=model_seed,
                round_index=round_index,
            )
            trained_models.append(trained_model)
            client_results.append(client_result)

        global_hash = server.aggregate(trained_models)
        after = server.broadcast().cpu()
        losses = np.asarray([item.mean_loss for item in client_results], dtype=np.float64)
        update_norm = self._parameter_update_norm(before, after)
        communication_bytes = 2 * len(selected_clients) * model_payload_bytes

        return RoundResult(
            round_index=round_index,
            learning_rate=learning_rate,
            selected_clients=selected_clients,
            client_results=tuple(client_results),
            mean_client_loss=float(np.mean(losses)),
            minimum_client_loss=float(np.min(losses)),
            maximum_client_loss=float(np.max(losses)),
            parameter_update_norm=update_norm,
            model_payload_bytes=model_payload_bytes,
            round_communication_bytes=communication_bytes,
            global_model_hash=global_hash,
        )

    @staticmethod
    def _configure_determinism(model_seed: ModelSeed, config: TrainingConfig) -> None:
        seed = int(model_seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if config.deterministic_algorithms:
            torch.use_deterministic_algorithms(True, warn_only=False)
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

    @staticmethod
    def _parameter_update_norm(before: DetectorModel, after: DetectorModel) -> MetricDifference:
        squared_norm = 0.0
        before_state = before.state_dict()
        for name, tensor in after.state_dict().items():
            if not tensor.dtype.is_floating_point:
                continue
            delta = tensor.detach().cpu().to(torch.float64) - before_state[name].detach().cpu().to(
                torch.float64
            )
            squared_norm += float(torch.sum(delta * delta))
        return float(np.sqrt(squared_norm))

    @staticmethod
    def _round20_score_correlation(
        round20_model: DetectorModel | None,
        final_model: DetectorModel,
        datasets: Mapping[ClientId, TensorDataset],
    ) -> Correlation | None:
        if round20_model is None:
            return None
        round20_scores: list[np.ndarray] = []
        final_scores: list[np.ndarray] = []
        round20_model.eval()
        final_model.eval()
        with torch.no_grad():
            for client_id in sorted(datasets):
                dataset = datasets[client_id]
                values = dataset.tensors[0]
                values = values.to(dtype=torch.float32)
                round20_scores.append(
                    round20_model.anomaly_score(values).cpu().numpy().astype(np.float64)
                )
                final_scores.append(
                    final_model.anomaly_score(values).cpu().numpy().astype(np.float64)
                )
        left = np.concatenate(round20_scores)
        right = np.concatenate(final_scores)
        if np.std(left) == 0.0 or np.std(right) == 0.0:
            return None
        return float(np.corrcoef(left, right)[0, 1])
