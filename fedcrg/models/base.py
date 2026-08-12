"""
Base Detector Model

Provides the abstract base class for detector models.

Normative reference: Section 8 (Frozen Detector and Federated Training)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True, slots=True)
class ModelState:
    """State of a detector model."""
    weights: Dict[str, torch.Tensor]
    epoch: int
    round: int
    loss: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "weights": {k: v.numpy().tolist() for k, v in self.weights.items()},
            "epoch": self.epoch,
            "round": self.round,
            "loss": self.loss,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelState":
        """Create from dictionary."""
        weights = {
            k: torch.tensor(v, dtype=torch.float32) for k, v in data["weights"].items()
        }
        return cls(
            weights=weights,
            epoch=data["epoch"],
            round=data["round"],
            loss=data["loss"],
        )


class ModelConfig(ABC):
    """Base configuration for a detector model."""
    
    @abstractmethod
    def get_architecture(self) -> List[int]:
        """Get the layer dimensions."""
        pass
    
    @abstractmethod
    def get_param_count(self) -> int:
        """Compute the total number of parameters."""
        pass


class BaseDetectorModel(ABC, nn.Module):
    """
    Abstract base class for detector models.
    
    All detector models must:
    - Be PyTorch nn.Module subclasses
    - Implement forward() for inference
    - Compute anomaly scores from inputs
    - Support federated training
    """
    
    config: ModelConfig
    device: torch.device
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the detector model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.device = torch.device("cpu")
    
    def to(self, device: torch.device) -> "BaseDetectorModel":
        """Move model to device."""
        self.device = device
        return super().to(device)
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        pass
    
    @abstractmethod
    def compute_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly score for input.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Anomaly score tensor of shape (batch_size,)
        """
        pass
    
    def get_param_count(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_weight_norm(self) -> float:
        """Get L2 norm of all weights."""
        total_norm = 0.0
        for p in self.parameters():
            total_norm += torch.norm(p).item() ** 2
        return total_norm ** 0.5
    
    def get_grad_norm(self) -> Optional[float]:
        """Get L2 norm of all gradients."""
        if not any(p.requires_grad for p in self.parameters()):
            return None
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                total_norm += torch.norm(p.grad).item() ** 2
        return total_norm ** 0.5
    
    def state_dict_hash(self) -> str:
        """
        Compute SHA-256 hash of model state dict for reproducibility.
        
        Returns:
            SHA-256 hex string
        """
        import hashlib
        state = self.state_dict()
        # Convert to deterministic string representation
        state_str = str(sorted(state.items()))
        return hashlib.sha256(state_str.encode()).hexdigest()
    
    def save(self, path: Path | str) -> None:
        """Save model and configuration."""
        path = Path(path)
        torch.save({
            "config": self.config,
            "state_dict": self.state_dict(),
            "hash": self.state_dict_hash(),
        }, path)
    
    @classmethod
    def load(cls, path: Path | str) -> "BaseDetectorModel":
        """Load model from checkpoint."""
        path = Path(path)
        checkpoint = torch.load(path, map_location="cpu")
        config = checkpoint["config"]
        model = cls(config)
        model.load_state_dict(checkpoint["state_dict"])
        model.to("cpu")
        return model
    
    def clone(self) -> "BaseDetectorModel":
        """Create a copy of the model."""
        clone = type(self)(self.config)
        clone.load_state_dict(self.state_dict())
        return clone
