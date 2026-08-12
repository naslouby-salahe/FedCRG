"""Typed federated-training results."""

from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class ClientRoundResult:
    client_id: str
    mean_loss: float
    model_hash: str

@dataclass(frozen=True, slots=True)
class RoundResult:
    round_index: int
    learning_rate: float
    selected_clients: tuple[str, ...]
    client_results: tuple[ClientRoundResult, ...]
    global_model_hash: str

@dataclass(frozen=True, slots=True)
class TrainingResult:
    model_seed: int
    rounds: tuple[RoundResult, ...]
    final_model_hash: str
