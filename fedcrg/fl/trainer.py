"""
Federated Learning Trainer

Implements the complete federated training loop per Section 8.2.

Normative reference: Section 8.2
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from fedcrg.fl.client import FederatedClient, FederatedClientConfig
from fedcrg.fl.lr_schedule import get_lr_for_round
from fedcrg.fl.server import FederatedServer, FederatedServerConfig, TrainingManifest
from fedcrg.models.base import BaseDetectorModel


@dataclass(frozen=True, slots=True)
class FederatedTrainerConfig:
    """
    Configuration for the federated trainer.
    
    Combines server and client configurations.
    
    Normative reference: Section 8.2
    """
    # Server config
    num_rounds: int = 30
    num_clients: int = 9
    model_seed: int = 42
    
    # Client config (same for all clients)
    batch_size: int = 64
    num_local_epochs: int = 120  # N-BaIoT; 20 for DIAD
    drop_last: bool = False
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    
    # Training config
    use_fp16: bool = False
    device: str = "cpu"
    
    # Checkpointing
    checkpoint_dir: Optional[Path] = None
    checkpoint_frequency: int = 5
    
    # Client IDs
    client_ids: Optional[List[str]] = None


class FederatedTrainer:
    """
    Federated learning trainer.
    
    Orchestrates the complete federated training process including:
    - Server initialization
    - Client training
    - Model aggregation
    - Checkpointing
    - Manifest generation
    
    Normative reference: Section 8.2
    """
    
    def __init__(
        self,
        config: FederatedTrainerConfig,
        model_template: BaseDetectorModel,
        train_datasets: Dict[str, Dataset],
    ):
        """
        Initialize the federated trainer.
        
        Args:
            config: Trainer configuration
            model_template: Template model to clone
            train_datasets: Dictionary mapping client_id to training dataset
        """
        self.config = config
        self.model_template = model_template
        self.train_datasets = train_datasets
        
        # Determine client IDs
        if config.client_ids:
            self.client_ids = config.client_ids
        else:
            self.client_ids = list(train_datasets.keys())
        
        if len(self.client_ids) != len(train_datasets):
            raise ValueError(
                f"Number of client IDs ({len(self.client_ids)}) does not match "
                f"number of datasets ({len(train_datasets)})"
            )
        
        # Create server
        server_config = FederatedServerConfig(
            num_rounds=config.num_rounds,
            num_clients=len(self.client_ids),
            model_seed=config.model_seed,
            checkpoint_dir=config.checkpoint_dir,
            checkpoint_frequency=config.checkpoint_frequency,
        )
        self.server = FederatedServer(server_config, model_template)
        
        # Create clients
        self.clients: Dict[str, FederatedClient] = {}
        for cid in self.client_ids:
            client_config = FederatedClientConfig(
                client_id=cid,
                model_seed=config.model_seed,
                batch_size=config.batch_size,
                num_local_epochs=config.num_local_epochs,
                drop_last=config.drop_last,
                adam_betas=config.adam_betas,
                adam_eps=config.adam_eps,
                weight_decay=config.weight_decay,
                use_fp16=config.use_fp16,
            )
            client = FederatedClient(
                config=client_config,
                model=model_template.clone(),
                train_dataset=train_datasets[cid],
                device=torch.device(config.device),
            )
            self.clients[cid] = client
        
        self.device = torch.device(config.device)
        self.manifest: Optional[TrainingManifest] = None
    
    def train_round(self, round: int) -> Dict[str, Any]:
        """
        Train for one federated round.
        
        Args:
            round: Round index (0 to num_rounds-1)
            
        Returns:
            Dictionary with round information
            
        Normative reference: Section 8.2
        """
        if round >= self.config.num_rounds:
            raise ValueError(f"Round {round} exceeds num_rounds {self.config.num_rounds}")
        
        # Get learning rate for this round
        lr = get_lr_for_round(
            round,
            eta_0=1e-3,
            eta_min=1e-5,
            num_rounds=self.config.num_rounds,
        )
        
        # Broadcast global model
        global_model = self.server.broadcast_global_model()
        
        # Each client trains
        client_models: Dict[str, BaseDetectorModel] = {}
        client_losses: Dict[str, float] = {}
        
        round_start = time.time()
        
        for cid in self.client_ids:
            client = self.clients[cid]
            
            # Load global model into client
            client.model = global_model.clone()
            
            # Train
            trained_model, loss = client.train_round(
                round=round,
                lr=lr,
                return_loss=True,
            )
            
            client_models[cid] = trained_model
            client_losses[cid] = loss or 0.0
        
        # Aggregate
        round_state = self.server.run_round(round, client_models)
        
        round_end = time.time()
        
        return {
            "round": round,
            "lr": lr,
            "client_losses": client_losses,
            "round_state": round_state,
            "duration_seconds": round_end - round_start,
        }
    
    def train(self) -> TrainingManifest:
        """
        Run complete federated training.
        
        Returns:
            TrainingManifest with complete training information
            
        Normative reference: Section 8.2
        """
        total_start = time.time()
        
        round_results = []
        
        for round_idx in range(self.config.num_rounds):
            result = self.train_round(round_idx)
            round_results.append(result)
            
            # Log progress
            avg_loss = sum(result["client_losses"].values()) / len(result["client_losses"])
            print(f"Round {round_idx:2d}: LR={result['lr']:.2e}, "
                  f"Avg Loss={avg_loss:.6f}, "
                  f"Duration={result['duration_seconds']:.2f}s")
        
        # Get final manifest from server
        manifest = self.server.training_manifest
        if manifest is None:
            # Build manifest manually
            manifest = self._build_manifest(round_results)
        
        self.manifest = manifest
        
        total_end = time.time()
        total_duration = total_end - total_start
        
        print(f"\nTraining complete. Total time: {total_duration:.2f}s")
        
        return manifest
    
    def _build_manifest(self, round_results: List[Dict[str, Any]]) -> TrainingManifest:
        """Build training manifest from round results."""
        from datetime import datetime
        
        start_time = datetime.fromtimestamp(time.time() - self._estimate_total_duration(round_results))
        end_time = datetime.now()
        
        round_states = []
        for result in round_results:
            rs = result["round_state"]
            round_states.append(rs)
        
        manifest = TrainingManifest(
            model_seed=self.config.model_seed,
            num_rounds=self.config.num_rounds,
            num_clients=len(self.client_ids),
            client_ids=self.client_ids,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            total_duration_seconds=self._estimate_total_duration(round_results),
            round_states=round_states,
            final_model_hash=self.server.global_model.state_dict_hash(),
        )
        
        return manifest
    
    def _estimate_total_duration(self, round_results: List[Dict[str, Any]]) -> float:
        """Estimate total duration from round results."""
        return sum(r["duration_seconds"] for r in round_results)
    
    def get_global_model(self) -> BaseDetectorModel:
        """
        Get the current global model.
        
        Returns:
            Clone of the global model
        """
        return self.server.get_global_model()
    
    def compute_all_scores(
        self,
        data: Dict[str, torch.Tensor],
        use_float64: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute scores for all clients.
        
        Args:
            data: Dictionary mapping client_id to data tensor
            use_float64: Whether to convert scores to float64
            
        Returns:
            Dictionary mapping client_id to scores
        """
        scores = {}
        for cid, client_data in data.items():
            client = self.clients[cid]
            scores[cid] = client.compute_scores(client_data, use_float64=use_float64)
        return scores
    
    def compute_global_scores(
        self,
        data: torch.Tensor,
        use_float64: bool = True,
    ) -> torch.Tensor:
        """
        Compute scores using the global model.
        
        Args:
            data: Input tensor of shape (n_samples, input_dim)
            use_float64: Whether to convert scores to float64
            
        Returns:
            Score tensor of shape (n_samples,)
        """
        global_model = self.get_global_model()
        global_model.eval()
        global_model.to(self.device)
        
        with torch.no_grad():
            data = data.to(self.device)
            scores = global_model.compute_score(data)
            
            if use_float64:
                scores = scores.double()
            
            scores = scores.to("cpu")
        
        return scores


def verify_trainer() -> None:
    """
    Verify trainer works correctly with a small test.
    """
    from fedcrg.models import Autoencoder, create_nbaiot_ae_config
    import torch
    
    # Create model template
    config = create_nbaiot_ae_config()
    model_template = Autoencoder(config)
    
    # Create dummy datasets for 2 clients
    class DummyDataset:
        def __init__(self, size=100):
            self.size = size
        def __len__(self):
            return self.size
        def __getitem__(self, idx):
            return torch.randn(115)
    
    datasets = {
        "nb01": DummyDataset(100),
        "nb02": DummyDataset(100),
    }
    
    # Create trainer with short config
    trainer_config = FederatedTrainerConfig(
        num_rounds=2,
        num_clients=2,
        num_local_epochs=2,  # Very short for testing
        client_ids=["nb01", "nb02"],
    )
    
    trainer = FederatedTrainer(
        config=trainer_config,
        model_template=model_template,
        train_datasets=datasets,
    )
    
    # Train
    manifest = trainer.train()
    
    assert manifest is not None
    assert manifest.num_rounds == 2
    assert len(manifest.round_states) == 2
    
    print(f"Trainer verification passed. Completed {manifest.num_rounds} rounds.")


if __name__ == "__main__":
    verify_trainer()
