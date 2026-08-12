"""
FedCRG Federated Learning Module

Implements federated training per Section 8.2 (Federated training state machine).

Normative reference: Section 8.2
"""

from fedcrg.fl.trainer import (
    FederatedTrainer,
    FederatedTrainerConfig,
)
from fedcrg.fl.server import (
    FederatedServer,
    FederatedServerConfig,
)
from fedcrg.fl.client import (
    FederatedClient,
    FederatedClientConfig,
)
from fedcrg.fl.sampling import (
    DeterministicSampler,
    create_deterministic_sampler,
)
from fedcrg.fl.aggregation import (
    aggregate_models_equal_mean,
)
from fedcrg.fl.lr_schedule import (
    CosineLearningRateSchedule,
    get_lr_for_round,
)

__all__ = [
    # Trainer
    "FederatedTrainer",
    "FederatedTrainerConfig",
    # Server
    "FederatedServer",
    "FederatedServerConfig",
    # Client
    "FederatedClient",
    "FederatedClientConfig",
    # Sampling
    "DeterministicSampler",
    "create_deterministic_sampler",
    # Aggregation
    "aggregate_models_equal_mean",
    # Learning rate schedule
    "CosineLearningRateSchedule",
    "get_lr_for_round",
]
