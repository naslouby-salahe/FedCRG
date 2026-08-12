"""
Model Aggregation

Implements equal arithmetic mean aggregation per Section 8.2.

Normative reference: Section 8.2
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from fedcrg.models.base import BaseDetectorModel


def aggregate_models_equal_mean(
    models: List[BaseDetectorModel],
    weights: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Aggregate models using equal arithmetic mean.
    
    All models are assumed to have the same parameter names and shapes.
    The aggregation is performed as an equal arithmetic mean of all client
    parameter tensors.
    
    Args:
        models: List of models to aggregate
        weights: Optional list of weights for weighted aggregation.
                 If None, equal weights (1/len(models)) are used.
                 If provided, must sum to 1.0.
        
    Returns:
        Dictionary mapping parameter names to aggregated tensors
        
    Normative reference: Section 8.2 (Aggregation: equal arithmetic mean
    of client parameter tensors)
    """
    if not models:
        raise ValueError("No models provided for aggregation")
    
    # Get first model to determine parameter structure
    reference_model = models[0]
    param_names = [name for name, _ in reference_model.named_parameters()]
    
    if weights is None:
        # Equal weights
        n = len(models)
        weights = [1.0 / n] * n
    else:
        # Verify weights sum to 1
        weight_sum = sum(weights)
        if abs(weight_sum - 1.0) > 1e-10:
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")
        if len(weights) != len(models):
            raise ValueError(f"Length of weights ({len(weights)}) must match "
                           f"number of models ({len(models)})")
    
    # Aggregate each parameter
    aggregated_params = {}
    for param_name in param_names:
        # Collect parameter tensors from all models
        param_tensors = []
        for model in models:
            param = dict(model.named_parameters())[param_name]
            param_tensors.append(param.detach().clone())
        
        # Weighted sum
        aggregated = torch.zeros_like(param_tensors[0])
        for tensor, weight in zip(param_tensors, weights):
            aggregated += weight * tensor
        
        aggregated_params[param_name] = aggregated
    
    return aggregated_params


def aggregate_models_in_place(
    target_model: BaseDetectorModel,
    source_models: List[BaseDetectorModel],
    weights: Optional[List[float]] = None,
) -> None:
    """
    Aggregate models into a target model in place.
    
    This modifies the target_model's parameters directly.
    
    Args:
        target_model: Model to store aggregated parameters
        source_models: List of models to aggregate
        weights: Optional list of weights for weighted aggregation
        
    Normative reference: Section 8.2
    """
    aggregated_params = aggregate_models_equal_mean(source_models, weights)
    
    # Load aggregated parameters into target model
    target_state = target_model.state_dict()
    for param_name, param_value in aggregated_params.items():
        if param_name in target_state:
            target_state[param_name].copy_(param_value)
        else:
            raise KeyError(f"Parameter {param_name} not found in target model")


class ModelAggregator:
    """
    Handles model aggregation for federated learning.
    
    Provides methods for equal arithmetic mean aggregation of client models.
    
    Normative reference: Section 8.2
    """
    
    def __init__(
        self,
        use_equal_weights: bool = True,
    ):
        """
        Initialize the model aggregator.
        
        Args:
            use_equal_weights: If True, use equal weights for all clients.
                             If False, weights can be provided per aggregation call.
        """
        self.use_equal_weights = use_equal_weights
    
    def aggregate(
        self,
        models: List[BaseDetectorModel],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate models.
        
        Args:
            models: List of models to aggregate
            weights: Optional weights (ignored if use_equal_weights is True)
            
        Returns:
            Aggregated parameters as dictionary
        """
        if self.use_equal_weights:
            return aggregate_models_equal_mean(models, weights=None)
        else:
            return aggregate_models_equal_mean(models, weights=weights)
    
    def aggregate_and_update(
        self,
        target_model: BaseDetectorModel,
        source_models: List[BaseDetectorModel],
        weights: Optional[List[float]] = None,
    ) -> None:
        """
        Aggregate models and update target model in place.
        
        Args:
            target_model: Model to update
            source_models: Models to aggregate
            weights: Optional weights
        """
        aggregated_params = self.aggregate(source_models, weights)
        
        target_state = target_model.state_dict()
        for param_name, param_value in aggregated_params.items():
            if param_name in target_state:
                target_state[param_name].copy_(param_value)


def verify_aggregation() -> None:
    """
    Verify model aggregation works correctly.
    """
    from fedcrg.models import Autoencoder, create_nbaiot_ae_config
    import torch
    
    # Create two models with same architecture
    config = create_nbaiot_ae_config()
    model1 = Autoencoder(config)
    model2 = Autoencoder(config)
    
    # Set different parameter values
    with torch.no_grad():
        for param in model1.parameters():
            param.fill_(1.0)
        for param in model2.parameters():
            param.fill_(3.0)
    
    # Aggregate
    aggregated = aggregate_models_equal_mean([model1, model2])
    
    # Check that aggregation produces correct mean (2.0)
    for param_name, param_value in aggregated.items():
        expected = torch.full_like(param_value, 2.0)
        assert torch.allclose(param_value, expected, rtol=1e-5), \
            f"Parameter {param_name} not correctly aggregated"
    
    print("Model aggregation verification passed.")


if __name__ == "__main__":
    verify_aggregation()
