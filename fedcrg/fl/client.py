"""
Federated Learning Client

Implements the client-side training per Section 8.2.

Normative reference: Section 8.2
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from fedcrg.fl.lr_schedule import get_lr_for_round
from fedcrg.fl.sampling import DeterministicSampler, create_deterministic_sampler
from fedcrg.models.base import BaseDetectorModel


@dataclass(frozen=True, slots=True)
class FederatedClientConfig:
    """
    Configuration for a federated learning client.
    
    Normative reference: Section 8.2
    """
    client_id: str
    model_seed: int
    batch_size: int = 64
    num_local_epochs: int = 120  # N-BaIoT; 20 for DIAD
    drop_last: bool = False
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    use_fp16: bool = False


class FederatedClient:
    """
    Federated learning client.
    
    Handles local training for a single client in a federated round.
    
    Normative reference: Section 8.2
    """
    
    def __init__(
        self,
        config: FederatedClientConfig,
        model: BaseDetectorModel,
        train_dataset: Dataset,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the federated client.
        
        Args:
            config: Client configuration
            model: Detector model to train
            train_dataset: Training dataset (T_k for confirmatory run)
            device: Device to use for training
        """
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.device = device or torch.device("cpu")
        
        # Create deterministic sampler
        self.sampler = create_deterministic_sampler(
            dataset=train_dataset,
            model_seed=config.model_seed,
            client_id=config.client_id,
            num_rounds=30,  # Standard
            num_local_epochs=config.num_local_epochs,
            batch_size=config.batch_size,
            drop_last=config.drop_last,
        )
    
    def train_round(
        self,
        round: int,
        lr: float,
        return_loss: bool = True,
    ) -> Tuple[BaseDetectorModel, Optional[float]]:
        """
        Perform one round of local training.
        
        This implements the exact training procedure from Section 8.2:
        - Create fresh Adam optimizer at lr
        - Deterministic shuffle per (model_seed, client_id, round, local_epoch)
        - Train exactly num_local_epochs
        - Use drop_last=false
        - Return trained model and optionally the mean epoch loss
        
        Args:
            round: Current federated round (0 to 29)
            lr: Learning rate for this round
            return_loss: Whether to return the mean epoch loss
            
        Returns:
            Tuple of (trained model, mean epoch loss) if return_loss is True,
            otherwise (trained model, None)
            
        Normative reference: Section 8.2
        """
        model = self.model.clone()
        model.to(self.device)
        model.train()
        
        # Create Adam optimizer with fresh state
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            betas=self.config.adam_betas,
            eps=self.config.adam_eps,
            weight_decay=self.config.weight_decay,
        )
        
        # Create data loader
        # Note: In FL, we typically iterate through epochs manually
        # to allow deterministic shuffling per epoch
        n_samples = len(self.train_dataset)
        num_batches = (n_samples + self.config.batch_size - 1) // self.config.batch_size
        
        total_loss = 0.0
        total_samples = 0
        
        for local_epoch in range(self.config.num_local_epochs):
            # Get deterministic shuffle for this round and epoch
            shuffle_indices = self.sampler.get_shuffle_indices(
                n=n_samples,
                round=round,
                local_epoch=local_epoch,
            )
            
            epoch_loss = 0.0
            epoch_samples = 0
            
            # Process in batches
            for batch_start in range(0, n_samples, self.config.batch_size):
                batch_end = min(batch_start + self.config.batch_size, n_samples)
                batch_indices = shuffle_indices[batch_start:batch_end]
                
                # Get batch data
                batch_data = []
                for idx in batch_indices:
                    batch_data.append(self.train_dataset[idx])
                
                # Stack into tensor
                # Assuming dataset returns tensors
                if isinstance(batch_data[0], torch.Tensor):
                    x = torch.stack(batch_data)
                else:
                    # Handle tuple (data, label) or dict
                    x = torch.stack([bd[0] if isinstance(bd, (tuple, list)) else bd for bd in batch_data])
                
                x = x.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                x_recon = model(x)
                
                # Compute MSE loss
                loss = nn.functional.mse_loss(x_recon, x)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                batch_loss = loss.item() * (batch_end - batch_start)
                epoch_loss += batch_loss
                epoch_samples += (batch_end - batch_start)
            
            # Record epoch loss
            epoch_mean_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0.0
            total_loss += epoch_loss
            total_samples += epoch_samples
        
        # Return cloned model on CPU
        model.to("cpu")
        
        mean_loss = total_loss / total_samples if total_samples > 0 and return_loss else None
        
        return model, mean_loss
    
    def compute_scores(
        self,
        data: torch.Tensor,
        use_float64: bool = True,
    ) -> torch.Tensor:
        """
        Compute anomaly scores for data.
        
        Args:
            data: Input tensor of shape (n_samples, input_dim)
            use_float64: Whether to convert scores to float64
            
        Returns:
            Score tensor of shape (n_samples,)
        """
        self.model.eval()
        self.model.to(self.device)
        
        with torch.no_grad():
            data = data.to(self.device)
            scores = self.model.compute_score(data)
            
            if use_float64:
                scores = scores.double()
            
            scores = scores.to("cpu")
        
        return scores


def verify_client() -> None:
    """
    Verify client training works correctly.
    """
    from fedcrg.models import Autoencoder, create_nbaiot_ae_config
    import torch
    
    # Create model and dummy dataset
    config = create_nbaiot_ae_config()
    model = Autoencoder(config)
    
    class DummyDataset:
        def __len__(self):
            return 100
        def __getitem__(self, idx):
            return torch.randn(115)
    
    dataset = DummyDataset()
    
    client_config = FederatedClientConfig(
        client_id="nb01",
        model_seed=42,
        num_local_epochs=2,  # Short for testing
    )
    
    client = FederatedClient(
        config=client_config,
        model=model,
        train_dataset=dataset,
    )
    
    # Test training round
    lr = get_lr_for_round(0)
    trained_model, loss = client.train_round(round=0, lr=lr, return_loss=True)
    
    assert loss is not None
    assert loss > 0
    print(f"Client training verification passed. Loss: {loss:.6f}")


if __name__ == "__main__":
    verify_client()
