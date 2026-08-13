"""Typed federated-training diagnostics."""

from dataclasses import dataclass

from fedcrg.domain.identifiers import ClientId, ModelSeed, Sha256


@dataclass(frozen=True, slots=True)
class ClientRoundResult:
    client_id: ClientId
    mean_loss: float
    record_presentations: int
    optimizer_steps: int
    model_hash: Sha256


@dataclass(frozen=True, slots=True)
class RoundResult:
    round_index: int
    learning_rate: float
    selected_clients: tuple[ClientId, ...]
    client_results: tuple[ClientRoundResult, ...]
    mean_client_loss: float
    minimum_client_loss: float
    maximum_client_loss: float
    parameter_update_norm: float
    model_payload_bytes: int
    round_communication_bytes: int
    global_model_hash: Sha256


@dataclass(frozen=True, slots=True)
class TrainingResult:
    model_seed: ModelSeed
    rounds: tuple[RoundResult, ...]
    final_model_hash: Sha256
    trainable_parameter_count: int
    model_payload_bytes: int
    total_model_communication_bytes: int
    round20_training_score_correlation: float | None
