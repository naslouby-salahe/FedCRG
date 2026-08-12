"""
Autoencoder Model

Implements the federated autoencoder detector per Section 8.1.

Normative reference: Section 8.1 (Primary federated autoencoder)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fedcrg.models.base import BaseDetectorModel, ModelConfig


# N-BaIoT architecture: 115-86-57-38-29-38-57-86-115
# This is a symmetric autoencoder
NBAIOT_ARCHITECTURE = [115, 86, 57, 38, 29, 38, 57, 86, 115]

# DIAD architecture: 86-64-43-28-21-28-43-64-86
DIAD_ARCHITECTURE = [86, 64, 43, 28, 21, 28, 43, 64, 86]

# Xavier uniform initialization gain for tanh
TANH_GAIN = 5 / 3


@dataclass(frozen=True, slots=True)
class AutoencoderConfig(ModelConfig):
    """
    Configuration for the autoencoder model.
    
    Matches Section 8.1 specifications exactly.
    """
    input_dim: int
    output_dim: int
    architecture: List[int]
    param_count: int
    hidden_activation: str = "tanh"
    output_activation: str = "linear"
    use_bias: bool = True
    
    def get_architecture(self) -> List[int]:
        """Get the layer dimensions."""
        return self.architecture
    
    def get_param_count(self) -> int:
        """Get precomputed parameter count."""
        return self.param_count


class AutoencoderLayer(nn.Module):
    """Single linear layer with configurable activation."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "tanh",
        use_bias: bool = True,
        gain: float = TANH_GAIN,
    ):
        """
        Initialize a layer with Xavier uniform initialization.
        
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
        if use_bias and self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with activation."""
        x = self.linear(x)
        if self.activation == "tanh":
            return torch.tanh(x)
        elif self.activation == "linear":
            return x
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)
        elif self.activation == "relu":
            return F.relu(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")


def compute_ae_param_count(architecture: List[int], use_bias: bool = True) -> int:
    """
    Compute parameter count for an autoencoder with given architecture.
    
    Uses the formula from Section 8.1.1:
    P = sum over l of (d_l * d_{l+1} + d_{l+1}) for biased-linear layers
    
    Args:
        architecture: List of layer dimensions
        use_bias: Whether layers use bias
        
    Returns:
        Total parameter count
    """
    total = 0
    for i in range(len(architecture) - 1):
        in_dim = architecture[i]
        out_dim = architecture[i + 1]
        # Weight parameters
        total += in_dim * out_dim
        # Bias parameters
        if use_bias:
            total += out_dim
    return total


# Precompute parameter counts for verification
# N-BaIoT: 115-86-57-38-29-38-57-86-115
NBAIOT_PARAM_COUNT = compute_ae_param_count(NBAIOT_ARCHITECTURE, use_bias=True)
assert NBAIOT_PARAM_COUNT == 36626, f"N-BaIoT param count mismatch: {NBAIOT_PARAM_COUNT}"

# DIAD: 86-64-43-28-21-28-43-64-86
DIAD_PARAM_COUNT = compute_ae_param_count(DIAD_ARCHITECTURE, use_bias=True)
assert DIAD_PARAM_COUNT == 20473, f"DIAD param count mismatch: {DIAD_PARAM_COUNT}"


class Autoencoder(BaseDetectorModel):
    """
    Autoencoder detector model.
    
    Implements the symmetric autoencoder architecture per Section 8.1.
    The anomaly score is the mean feature-wise reconstruction MSE.
    
    Architecture options:
    - N-BaIoT: 115-86-57-38-29-38-57-86-115 (36,626 params)
    - DIAD: 86-64-43-28-21-28-43-64-86 (20,473 params)
    
    Normative reference: Section 8.1
    """
    
    config: AutoencoderConfig
    layers: nn.ModuleList
    
    def __init__(self, config: AutoencoderConfig):
        """
        Initialize the autoencoder.
        
        Args:
            config: Autoencoder configuration
        """
        super().__init__(config)
        
        # Build layers
        self.layers = nn.ModuleList()
        architecture = config.architecture
        
        for i in range(len(architecture) - 1):
            in_dim = architecture[i]
            out_dim = architecture[i + 1]
            
            # Determine activation
            # Last layer uses output_activation, others use hidden_activation
            if i == len(architecture) - 2:
                activation = config.output_activation
            else:
                activation = config.hidden_activation
            
            layer = AutoencoderLayer(
                in_features=in_dim,
                out_features=out_dim,
                activation=activation,
                use_bias=config.use_bias,
                gain=TANH_GAIN,
            )
            self.layers.append(layer)
        
        # Verify parameter count
        actual_count = self.get_param_count()
        expected_count = config.get_param_count()
        assert actual_count == expected_count, \
            f"Parameter count mismatch: {actual_count} != {expected_count}"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def compute_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly score as mean feature-wise reconstruction MSE.
        
        Per Section 8.1: score = (1/d) * sum_{j=1}^d (x_j - x_hat_j)^2
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Anomaly score tensor of shape (batch_size,)
        """
        # Get reconstruction
        x_recon = self.forward(x)
        
        # Compute per-feature squared error
        squared_error = (x - x_recon) ** 2
        
        # Mean over features
        score = torch.mean(squared_error, dim=1)
        
        return score
    
    def get_bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get bottleneck layer representation.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Bottleneck tensor of shape (batch_size, bottleneck_dim)
        """
        # Find the bottleneck layer (smallest layer)
        architecture = self.config.architecture
        bottleneck_idx = architecture.index(min(architecture))
        
        x = self.layers[0](x)
        for i in range(1, bottleneck_idx + 1):
            x = self.layers[i](x)
        
        return x


# Predefined configurations

def create_nbaiot_ae_config() -> AutoencoderConfig:
    """Create N-BaIoT autoencoder configuration."""
    return AutoencoderConfig(
        input_dim=115,
        output_dim=115,
        architecture=NBAIOT_ARCHITECTURE,
        hidden_activation="tanh",
        output_activation="linear",
        use_bias=True,
        param_count=NBAIOT_PARAM_COUNT,
    )


def create_diad_ae_config() -> AutoencoderConfig:
    """Create DIAD autoencoder configuration."""
    return AutoencoderConfig(
        input_dim=86,
        output_dim=86,
        architecture=DIAD_ARCHITECTURE,
        hidden_activation="tanh",
        output_activation="linear",
        use_bias=True,
        param_count=DIAD_PARAM_COUNT,
    )


# Factory function
def create_autoencoder(dataset: str = "nbaiot") -> Autoencoder:
    """
    Create an autoencoder for the specified dataset.
    
    Args:
        dataset: Dataset identifier ("nbaiot" or "diad")
        
    Returns:
        Configured Autoencoder instance
    """
    if dataset == "nbaiot":
        config = create_nbaiot_ae_config()
    elif dataset == "diad":
        config = create_diad_ae_config()
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Must be 'nbaiot' or 'diad'.")
    
    return Autoencoder(config)
