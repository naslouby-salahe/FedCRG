"""
Deep-SVDD Model

Implements the Deep-SVDD detector for robustness check per Section 8.4.

Normative reference: Section 8.4 (Mandatory second score generator)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fedcrg.models.base import BaseDetectorModel, ModelConfig
from fedcrg.config import DatasetID


# Deep-SVDD encoder architecture for N-BaIoT: 115-64-32
DEEP_SVDD_ENCODER = [115, 64, 32]

# Xavier uniform initialization gain for tanh
TANH_GAIN = 5 / 3


@dataclass(frozen=True, slots=True)
class DeepSVDDConfig(ModelConfig):
    """
    Configuration for the Deep-SVDD model.
    
    Matches Section 8.4 specifications exactly.
    """
    input_dim: int
    output_dim: int
    encoder: List[int]
    embedding_dim: int
    param_count: int
    hidden_activation: str = "tanh"
    output_activation: str = "linear"
    use_bias: bool = False  # Per Section 8.4: bias disabled
    center_mode: str = "equal_mean_of_client_initial_embeddings"
    
    def get_architecture(self) -> List[int]:
        """Get the encoder architecture."""
        return self.encoder
    
    def get_param_count(self) -> int:
        """Get precomputed parameter count."""
        return self.param_count


def compute_deep_svdd_param_count(encoder: List[int], use_bias: bool = False) -> int:
    """
    Compute parameter count for Deep-SVDD encoder.
    
    Args:
        encoder: List of encoder layer dimensions
        use_bias: Whether layers use bias
        
    Returns:
        Total parameter count
    """
    total = 0
    for i in range(len(encoder) - 1):
        in_dim = encoder[i]
        out_dim = encoder[i + 1]
        # Weight parameters
        total += in_dim * out_dim
        # Bias parameters
        if use_bias:
            total += out_dim
    return total


# Precompute parameter counts for verification
# Deep-SVDD: 115-64-32 encoder, no biases, plus center parameter
# Encoder params: 115*64 + 64*32 = 7360 + 2048 = 9408
# Center params: 32 (embedding dimension)
# Total: 9408 + 32 = 9440
DEEP_SVDD_PARAM_COUNT = compute_deep_svdd_param_count(DEEP_SVDD_ENCODER, use_bias=False) + DEEP_SVDD_ENCODER[-1]


class EncoderBlock(nn.Module):
    """Encoder block for Deep-SVDD."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "tanh",
        use_bias: bool = False,
        gain: float = TANH_GAIN,
    ):
        """
        Initialize an encoder layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            activation: Activation function name
            use_bias: Whether to use bias
            gain: Xavier initialization gain factor
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=use_bias)
        self.activation = activation
        
        # Initialize with Xavier uniform, scaled by gain
        nn.init.xavier_uniform_(self.linear.weight, gain=gain)
        # Note: No bias initialization since use_bias=False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with activation."""
        x = self.linear(x)
        if self.activation == "tanh":
            return torch.tanh(x)
        elif self.activation == "linear":
            return x
        elif self.activation == "relu":
            return F.relu(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")


class DeepSVDD(BaseDetectorModel):
    """
    Deep-SVDD detector model.
    
    Implements the Deep-SVDD architecture per Section 8.4.
    The center is computed once before training from the seed-initialized
    encoder and is frozen during training.
    
    Architecture: 115-64-32 (encoder), tanh, no biases
    Embedding dimension: 32
    Center: mean of client embeddings from T_k, averaged across clients
    Loss: MSE of distance to center
    Score: squared L2 distance to center
    
    Normative reference: Section 8.4
    """
    
    config: DeepSVDDConfig
    encoder: nn.Sequential
    center: torch.Tensor
    
    def __init__(self, config: DeepSVDDConfig, center: Optional[torch.Tensor] = None):
        """
        Initialize Deep-SVDD.
        
        Args:
            config: Deep-SVDD configuration
            center: Precomputed center tensor. If None, will be computed later.
        """
        super().__init__(config)
        
        # Build encoder
        encoder_layers = []
        for i in range(len(config.encoder) - 1):
            in_dim = config.encoder[i]
            out_dim = config.encoder[i + 1]
            layer = EncoderBlock(
                in_features=in_dim,
                out_features=out_dim,
                activation=config.hidden_activation,
                use_bias=config.use_bias,
                gain=TANH_GAIN,
            )
            encoder_layers.append(layer)
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Center (will be computed or provided)
        embedding_dim = config.encoder[-1]
        if center is not None:
            self.center = nn.Parameter(center, requires_grad=False)
        else:
            # Initialize to zeros, will be set later
            self.center = nn.Parameter(
                torch.zeros(embedding_dim), requires_grad=False
            )
        
        # Verify parameter count
        actual_count = self.get_param_count()
        expected_count = config.get_param_count()
        assert actual_count == expected_count, \
            f"Parameter count mismatch: {actual_count} != {expected_count}"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Embedding tensor of shape (batch_size, embedding_dim)
        """
        return self.encoder(x)
    
    def compute_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly score as squared L2 distance to center.
        
        Per Section 8.4: score = ||f_theta(x) - c||_2^2
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Anomaly score tensor of shape (batch_size,)
        """
        # Get embedding
        embedding = self.forward(x)
        
        # Compute squared L2 distance to center
        distance = torch.norm(embedding - self.center, dim=1)
        score = distance ** 2
        
        return score
    
    def compute_center(
        self,
        data_loader: torch.utils.data.DataLoader,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Compute center as mean of embeddings from training data.
        
        Per Section 8.4: initialize encoder from model seed; each client
        computes mean embedding on T_k; server equal-averages nine client
        means; center then frozen.
        
        This is a simplified version that computes the center from a
        single data loader. The full federated version would compute
        per-client means and then average them.
        
        Args:
            data_loader: DataLoader with training data
            device: Device to use for computation
            
        Returns:
            Center tensor of shape (embedding_dim,)
        """
        if device is None:
            device = self.device
        
        self.eval()
        self.to(device)
        
        embeddings = []
        with torch.no_grad():
            for batch in data_loader:
                x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                emb = self.forward(x)
                embeddings.append(emb)
        
        all_embeddings = torch.cat(embeddings, dim=0)
        center = torch.mean(all_embeddings, dim=0)
        
        return center
    
    def set_center(self, center: torch.Tensor) -> None:
        """
        Set the center and freeze it.
        
        Args:
            center: Center tensor of shape (embedding_dim,)
        """
        self.center.data = center.clone()
        self.center.requires_grad = False
    
    def freeze_center(self) -> None:
        """Freeze the center (disable gradient)."""
        self.center.requires_grad = False
    
    def train_with_frozen_center(self, mode: bool = True) -> "DeepSVDD":
        """
        Set training mode with frozen center.
        
        Args:
            mode: Whether to set training mode
            
        Returns:
            self for chaining
        """
        self.train(mode)
        self.freeze_center()
        return self


# Predefined configurations

def create_nbaiot_deep_svdd_config() -> DeepSVDDConfig:
    """Create N-BaIoT Deep-SVDD configuration."""
    return DeepSVDDConfig(
        input_dim=115,
        output_dim=32,
        encoder=DEEP_SVDD_ENCODER,
        embedding_dim=32,
        hidden_activation="tanh",
        output_activation="linear",
        use_bias=False,
        center_mode="equal_mean_of_client_initial_embeddings",
        param_count=DEEP_SVDD_PARAM_COUNT,
    )


# Factory function
def create_deep_svdd(dataset: str = "nbaiot") -> DeepSVDD:
    """
    Create a Deep-SVDD model for the specified dataset.
    
    Args:
        dataset: Dataset identifier (currently only "nbaiot" supported)
        
    Returns:
        Configured DeepSVDD instance
    """
    if dataset == "nbaiot":
        config = create_nbaiot_deep_svdd_config()
    else:
        raise ValueError(f"Deep-SVDD only implemented for N-BaIoT, got {dataset}")
    
    return DeepSVDD(config)
