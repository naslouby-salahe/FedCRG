"""
Federated Learning Server

Implements the server-side aggregation and broadcast per Section 8.2.

Normative reference: Section 8.2
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from fedcrg.fl.aggregation import aggregate_models_equal_mean, aggregate_models_in_place
from fedcrg.fl.lr_schedule import get_lr_for_round
from fedcrg.models.base import BaseDetectorModel, ModelState


@dataclass(frozen=True, slots=True)
class FederatedServerConfig:
    """
    Configuration for the federated learning server.
    
    Normative reference: Section 8.2
    """
    num_rounds: int = 30
    num_clients: int = 9  # For N-BaIoT
    aggregation_mode: str = "equal_mean"  # Only mode supported
    model_seed: int = 42
    
    # Communication tracking
    track_communication: bool = True
    
    # Checkpointing
    checkpoint_dir: Optional[Path] = None
    checkpoint_frequency: int = 5  # Save every N rounds


@dataclass(frozen=True, slots=True)
class RoundState:
    """
    State information for a training round.
    
    Captures the hash, metrics, and timing for each round.
    """
    round: int
    global_model_hash: str
    client_hashes: Dict[str, str]
    client_losses: Dict[str, float]
    timestamp: str
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class TrainingManifest:
    """
    Manifest of a complete training run.
    
    Contains all metadata needed to reproduce and verify the training.
    """
    model_seed: int
    num_rounds: int
    num_clients: int
    client_ids: List[str]
    start_time: str
    end_time: str
    total_duration_seconds: float
    round_states: List[RoundState]
    final_model_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_seed": self.model_seed,
            "num_rounds": self.num_rounds,
            "num_clients": self.num_clients,
            "client_ids": self.client_ids,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration_seconds": self.total_duration_seconds,
            "round_states": [
                {
                    "round": rs.round,
                    "global_model_hash": rs.global_model_hash,
                    "client_hashes": rs.client_hashes,
                    "client_losses": rs.client_losses,
                    "timestamp": rs.timestamp,
                    "duration_seconds": rs.duration_seconds,
                }
                for rs in self.round_states
            ],
            "final_model_hash": self.final_model_hash,
        }
    
    def to_json(self, path: Path) -> None:
        """Save manifest to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingManifest":
        """Create from dictionary."""
        round_states = [
            RoundState(
                round=rs["round"],
                global_model_hash=rs["global_model_hash"],
                client_hashes=rs["client_hashes"],
                client_losses=rs["client_losses"],
                timestamp=rs["timestamp"],
                duration_seconds=rs["duration_seconds"],
            )
            for rs in data["round_states"]
        ]
        return cls(
            model_seed=data["model_seed"],
            num_rounds=data["num_rounds"],
            num_clients=data["num_clients"],
            client_ids=data["client_ids"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            total_duration_seconds=data["total_duration_seconds"],
            round_states=round_states,
            final_model_hash=data["final_model_hash"],
        )
    
    @classmethod
    def from_json(cls, path: Path) -> "TrainingManifest":
        """Load manifest from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def hash(self) -> str:
        """Compute SHA-256 hash of the manifest for verification."""
        manifest_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(manifest_str.encode()).hexdigest()


class FederatedServer:
    """
    Federated learning server.
    
    Handles model aggregation, broadcast, and training state management.
    
    Normative reference: Section 8.2
    """
    
    def __init__(
        self,
        config: FederatedServerConfig,
        model_template: BaseDetectorModel,
    ):
        """
        Initialize the federated server.
        
        Args:
            config: Server configuration
            model_template: Template model to clone for aggregation
        """
        self.config = config
        self.model_template = model_template
        
        # Initialize global model
        self.global_model = model_template.clone()
        
        # Training state
        self.current_round: int = 0
        self.round_states: List[RoundState] = []
        self.start_time: Optional[datetime] = None
        self.training_manifest: Optional[TrainingManifest] = None
        
        # Checkpointing
        self.checkpoint_dir = config.checkpoint_dir
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def get_global_model(self) -> BaseDetectorModel:
        """
        Get a clone of the current global model.
        
        Returns:
            Clone of the global model
        """
        return self.global_model.clone()
    
    def broadcast_global_model(self) -> BaseDetectorModel:
        """
        Broadcast the global model to clients.
        
        Returns:
            Clone of the global model for client use
            
        Normative reference: Section 8.2 (broadcast the current global
        parameter tensors)
        """
        return self.get_global_model()
    
    def aggregate_client_models(
        self,
        client_models: Dict[str, BaseDetectorModel],
        round: int,
    ) -> RoundState:
        """
        Aggregate client models and update the global model.
        
        Implements equal arithmetic mean aggregation per Section 8.2.
        
        Args:
            client_models: Dictionary mapping client_id to trained model
            round: Current round index
            
        Returns:
            RoundState with information about this round
            
        Normative reference: Section 8.2
        """
        round_start = datetime.now()
        
        # Collect models in consistent order
        client_ids = sorted(client_models.keys())
        models = [client_models[cid] for cid in client_ids]
        
        # Compute global model hash before aggregation
        global_model_hash = self.global_model.state_dict_hash()
        
        # Aggregate models
        aggregate_models_in_place(self.global_model, models)
        
        # Compute new global model hash
        new_global_model_hash = self.global_model.state_dict_hash()
        
        # Collect client hashes and losses
        client_hashes = {}
        client_losses = {}
        for cid, model in client_models.items():
            client_hashes[cid] = model.state_dict_hash()
            # Loss is not available from model state; would need to be tracked separately
        
        round_end = datetime.now()
        duration = (round_end - round_start).total_seconds()
        
        round_state = RoundState(
            round=round,
            global_model_hash=global_model_hash,
            client_hashes=client_hashes,
            client_losses=client_losses,
            timestamp=round_start.isoformat(),
            duration_seconds=duration,
        )
        
        self.round_states.append(round_state)
        self.current_round = round + 1
        
        # Checkpoint if configured
        if self.checkpoint_dir and round % self.config.checkpoint_frequency == 0:
            self._save_checkpoint(round)
        
        return round_state
    
    def run_round(
        self,
        round: int,
        client_models: Dict[str, BaseDetectorModel],
    ) -> RoundState:
        """
        Run a complete federated round.
        
        Args:
            round: Current round index
            client_models: Dictionary of client_id -> trained model
            
        Returns:
            RoundState for this round
        """
        if round != self.current_round:
            raise ValueError(f"Round mismatch: expected {self.current_round}, got {round}")
        
        return self.aggregate_client_models(client_models, round)
    
    def run_training(
        self,
        client_ids: List[str],
        get_client_model: callable,
    ) -> TrainingManifest:
        """
        Run complete federated training.
        
        Args:
            client_ids: List of client identifiers
            get_client_model: Callable that takes (client_id, round, model)
                             and returns trained model and loss
        
        Returns:
            TrainingManifest with complete training information
        """
        self.start_time = datetime.now()
        
        for round_idx in range(self.config.num_rounds):
            # Broadcast global model to all clients
            global_model = self.broadcast_global_model()
            
            # Each client trains
            client_models = {}
            client_losses = {}
            
            for cid in client_ids:
                trained_model, loss = get_client_model(cid, round_idx, global_model)
                client_models[cid] = trained_model
                client_losses[cid] = loss
            
            # Aggregate
            round_state = self.aggregate_client_models(client_models, round_idx)
            
            # Update losses in round state
            round_state = RoundState(
                round=round_state.round,
                global_model_hash=round_state.global_model_hash,
                client_hashes=round_state.client_hashes,
                client_losses=client_losses,
                timestamp=round_state.timestamp,
                duration_seconds=round_state.duration_seconds,
            )
            self.round_states[-1] = round_state
        
        # Build manifest
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        manifest = TrainingManifest(
            model_seed=self.config.model_seed,
            num_rounds=self.config.num_rounds,
            num_clients=len(client_ids),
            client_ids=client_ids,
            start_time=self.start_time.isoformat(),
            end_time=end_time.isoformat(),
            total_duration_seconds=total_duration,
            round_states=self.round_states,
            final_model_hash=self.global_model.state_dict_hash(),
        )
        
        self.training_manifest = manifest
        
        # Save final checkpoint
        if self.checkpoint_dir:
            self._save_final_checkpoint()
        
        return manifest
    
    def _save_checkpoint(self, round: int) -> None:
        """Save checkpoint for a specific round."""
        if not self.checkpoint_dir:
            return
        
        checkpoint_path = self.checkpoint_dir / f"round_{round:03d}.pt"
        
        checkpoint = {
            "round": round,
            "model_state_dict": self.global_model.state_dict(),
            "model_config": self.global_model.config,
            "hash": self.global_model.state_dict_hash(),
        }
        
        torch.save(checkpoint, checkpoint_path)
    
    def _save_final_checkpoint(self) -> None:
        """Save final checkpoint."""
        if not self.checkpoint_dir:
            return
        
        checkpoint_path = self.checkpoint_dir / "final.pt"
        
        checkpoint = {
            "round": self.current_round - 1,
            "model_state_dict": self.global_model.state_dict(),
            "model_config": self.global_model.config,
            "hash": self.global_model.state_dict_hash(),
            "manifest": self.training_manifest.to_dict() if self.training_manifest else None,
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        # Also save manifest separately
        if self.training_manifest:
            manifest_path = self.checkpoint_dir / "manifest.json"
            self.training_manifest.to_json(manifest_path)


def verify_server() -> None:
    """
    Verify server aggregation works correctly.
    """
    from fedcrg.models import Autoencoder, create_nbaiot_ae_config
    import torch
    
    # Create model template
    config = create_nbaiot_ae_config()
    model_template = Autoencoder(config)
    
    # Create server
    server_config = FederatedServerConfig(
        num_rounds=3,
        num_clients=2,
    )
    server = FederatedServer(server_config, model_template)
    
    # Create two different models (simulating client training)
    model1 = model_template.clone()
    model2 = model_template.clone()
    
    with torch.no_grad():
        for param in model1.parameters():
            param.fill_(1.0)
        for param in model2.parameters():
            param.fill_(3.0)
    
    # Aggregate
    round_state = server.aggregate_client_models(
        {"client1": model1, "client2": model2},
        round=0,
    )
    
    # Check that global model has been updated to mean
    global_model = server.get_global_model()
    for param in global_model.parameters():
        assert torch.allclose(param, torch.full_like(param, 2.0), rtol=1e-5)
    
    print("Server aggregation verification passed.")


if __name__ == "__main__":
    verify_server()
