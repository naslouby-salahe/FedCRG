"""Deterministic federated training orchestration and protocol diagnostics."""

from __future__ import annotations

import random
import time
from collections.abc import Mapping

import numpy as np
import torch
from torch.utils.data import TensorDataset

from fedcrg.config.training_config import TrainingConfig
from fedcrg.domain.identifiers import ClientId, ModelSeed, Sha256
from fedcrg.runtime import get_logger
from fedcrg.detectors.base import DetectorModel
from fedcrg.federated.client import FederatedClient
from fedcrg.federated.models import RoundResult, TrainingResult
from fedcrg.federated.sampling import ClientSampler
from fedcrg.federated.scheduling import cosine_learning_rate
from fedcrg.federated.server import FederatedServer

_LOGGER = get_logger(__name__)


class FederatedTrainer:
    """Execute the frozen full-participation equal-client-mean training protocol."""

    diagnostic_round_index = 19

    def train(
        self,
        model: DetectorModel,
        datasets: Mapping[ClientId, TensorDataset],
        config: TrainingConfig,
        model_seed: ModelSeed | int,
    ) -> tuple[DetectorModel, TrainingResult]:
        if not datasets:
            raise ValueError("At least one federated client is required")

        seed = ModelSeed(int(model_seed))
        self._configure_determinism(seed, config)
        device = torch.device(config.device.value)
        client_ids = tuple(sorted(datasets))
        clients = {
            client_id: FederatedClient(client_id, datasets[client_id], device)
            for client_id in client_ids
        }
        sampler = ClientSampler(client_ids, config.client_fraction, int(seed))
        server = FederatedServer(model)

        round_results: list[RoundResult] = []
        round20_model: DetectorModel | None = None
        model_payload_bytes = model.trainable_tensor_bytes()

        _LOGGER.info(
            "training start model_seed=%d device=%s clients=%d rounds=%d local_epochs=%d",
            int(seed),
            device,
            len(client_ids),
            config.rounds,
            config.local_epochs,
        )
        for round_index in range(config.rounds):
            started = time.monotonic()
            round_result = self._train_round(
                server=server,
                clients=clients,
                sampler=sampler,
                config=config,
                model_seed=seed,
                round_index=round_index,
                model_payload_bytes=model_payload_bytes,
            )
            round_results.append(round_result)
            _LOGGER.info(
                "model_seed=%d round %d/%d loss=%.6f lr=%.2e elapsed=%.1fs",
                int(seed),
                round_index + 1,
                config.rounds,
                round_result.mean_client_loss,
                round_result.learning_rate,
                time.monotonic() - started,
            )
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
            model_seed=seed,
            rounds=tuple(round_results),
            final_model_hash=Sha256(final_model.state_hash()),
            trainable_parameter_count=final_model.trainable_parameter_count(),
            model_payload_bytes=model_payload_bytes,
            total_model_communication_bytes=total_communication,
            round20_training_score_correlation=score_correlation,
        )
        _LOGGER.info(
            "training done model_seed=%d final_loss=%.6f hash=%s",
            int(seed),
            round_results[-1].mean_client_loss,
            result.final_model_hash.value[:12],
        )
        return final_model, result

    def _train_round(
        self,
        server: FederatedServer,
        clients: dict[ClientId, FederatedClient],
        sampler: ClientSampler,
        config: TrainingConfig,
        model_seed: ModelSeed,
        round_index: int,
        model_payload_bytes: int,
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
        client_results = []

        for client_id in selected_clients:
            trained_model, client_result = clients[client_id].train(
                global_model=global_model,
                config=config,
                learning_rate=learning_rate,
                model_seed=int(model_seed),
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
            global_model_hash=Sha256(global_hash),
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
    def _parameter_update_norm(before: DetectorModel, after: DetectorModel) -> float:
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
    ) -> float | None:
        if round20_model is None:
            return None
        round20_scores: list[np.ndarray] = []
        final_scores: list[np.ndarray] = []
        round20_model.eval()
        final_model.eval()
        with torch.no_grad():
            for client_id in sorted(datasets):
                dataset = datasets[client_id]
                values = dataset.tensors[0] if isinstance(dataset, TensorDataset) else None
                if values is None:
                    raise TypeError("Round-20 diagnostic requires tensor-backed training datasets")
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
