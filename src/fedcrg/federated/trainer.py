"""Deterministic federated training orchestration."""

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from fedcrg.config.models import TrainingConfig
from fedcrg.detectors.base import DetectorModel
from fedcrg.federated.client import FederatedClient
from fedcrg.federated.models import RoundResult, TrainingResult
from fedcrg.federated.sampling import ClientSampler
from fedcrg.federated.scheduling import cosine_learning_rate
from fedcrg.federated.server import FederatedServer

class FederatedTrainer:
    def train(self, model: DetectorModel, datasets: dict[str, Dataset[Tensor]], config: TrainingConfig, model_seed: int) -> tuple[DetectorModel, TrainingResult]:
        if not datasets: raise ValueError("At least one federated client is required")
        torch.manual_seed(model_seed); np.random.seed(model_seed); device = torch.device(config.device); client_ids = tuple(sorted(datasets)); clients = {cid: FederatedClient(cid, datasets[cid], device) for cid in client_ids}; server = FederatedServer(model); round_results=[]; sampler=ClientSampler(client_ids, config.client_fraction, model_seed)
        for round_index in range(config.rounds):
            learning_rate = cosine_learning_rate(round_index, config.rounds, config.learning_rate_initial, config.learning_rate_final); selected = sampler.select(); client_models=[]; client_results=[]
            for client_id in selected:
                trained, result = clients[client_id].train(server.broadcast(), config, learning_rate, model_seed=model_seed, round_index=round_index); client_models.append(trained); client_results.append(result)
            global_hash = server.aggregate(client_models); round_results.append(RoundResult(round_index, learning_rate, selected, tuple(client_results), global_hash))
        final_model = server.broadcast().cpu(); return final_model, TrainingResult(model_seed, tuple(round_results), final_model.state_hash())
