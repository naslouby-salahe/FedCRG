"""
Score Computer

Implements score computation per Section 8.2.

Normative reference: Section 8.2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from fedcrg.data.base import DatasetRole
from fedcrg.models.base import BaseDetectorModel
from fedcrg.scoring.schemas import ClientScores, RoleScores, ScoreManifest


@dataclass(frozen=True, slots=True)
class ScoreComputerConfig:
    """
    Configuration for the score computer.
    
    Normative reference: Section 8.2
    """
    use_float64: bool = True  # Per Section 8.2: Score storage is float64
    batch_size: int = 256
    device: str = "cpu"


class ScoreComputer:
    """
    Computes anomaly scores for a detector model.
    
    Scores are always computed as float64 after float32 forward pass
    per Section 8.2.
    
    Normative reference: Section 8.2
    """
    
    def __init__(
        self,
        config: ScoreComputerConfig,
        model: BaseDetectorModel,
    ):
        """
        Initialize the score computer.
        
        Args:
            config: Score computer configuration
            model: Detector model to use for scoring
        """
        self.config = config
        self.model = model
        self.device = torch.device(config.device)
    
    def compute_scores(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly scores for data.
        
        Args:
            data: Input tensor of shape (n_samples, input_dim)
            
        Returns:
            Score tensor of shape (n_samples,) in float64
            
        Normative reference: Section 8.2
        """
        self.model.eval()
        self.model.to(self.device)
        
        with torch.no_grad():
            # Process in batches if needed
            if data.shape[0] <= self.config.batch_size:
                data = data.to(self.device)
                scores = self.model.compute_score(data)
            else:
                scores_list = []
                for i in range(0, data.shape[0], self.config.batch_size):
                    batch = data[i:i + self.config.batch_size].to(self.device)
                    batch_scores = self.model.compute_score(batch)
                    scores_list.append(batch_scores)
                scores = torch.cat(scores_list, dim=0)
            
            # Convert to float64
            if self.config.use_float64:
                scores = scores.double()
            
            scores = scores.to("cpu")
        
        return scores
    
    def compute_client_scores(
        self,
        role_data: Dict[str, torch.Tensor],
        client_id: Optional[str] = None,
    ) -> ClientScores:
        """
        Compute scores for all roles for a client.
        
        Args:
            role_data: Dictionary mapping role to data tensor
            client_id: Optional client identifier
            
        Returns:
            ClientScores with scores for all roles
            
        Normative reference: Section 8.2
        """
        role_scores = {}
        
        for role, data in role_data.items():
            scores = self.compute_scores(data)
            role_scores[role] = RoleScores(
                role=role,
                scores=scores.numpy(),
                client_id=client_id,
            )
        
        return ClientScores(
            client_id=client_id or "",
            role_scores=role_scores,
            model_hash=self.model.state_dict_hash(),
        )
    
    def compute_all_client_scores(
        self,
        client_role_data: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, ClientScores]:
        """
        Compute scores for all clients and roles.
        
        Args:
            client_role_data: Nested dictionary mapping client_id -> role -> data
            
        Returns:
            Dictionary mapping client_id to ClientScores
            
        Normative reference: Section 8.2
        """
        client_scores = {}
        
        for client_id, role_data in client_role_data.items():
            client_scores[client_id] = self.compute_client_scores(
                role_data, client_id=client_id
            )
        
        return client_scores
    
    def compute_manifest(
        self,
        client_role_data: Dict[str, Dict[str, torch.Tensor]],
        dataset: str,
        model_seed: int,
    ) -> ScoreManifest:
        """
        Compute complete score manifest.
        
        Args:
            client_role_data: Nested dictionary mapping client_id -> role -> data
            dataset: Dataset identifier ("nbaiot" or "diad")
            model_seed: Model seed used
            
        Returns:
            Complete ScoreManifest
            
        Normative reference: Section 8.2
        """
        client_scores = self.compute_all_client_scores(client_role_data)
        
        return ScoreManifest(
            model_seed=model_seed,
            dataset=dataset,
            client_ids=list(client_scores.keys()),
            model_hash=self.model.state_dict_hash(),
            client_scores=client_scores,
            score_dtype="float64",
        )


def verify_computer() -> None:
    """Verify score computer works correctly."""
    from fedcrg.models import Autoencoder, create_nbaiot_ae_config
    import torch
    
    # Create model
    config = create_nbaiot_ae_config()
    model = Autoencoder(config)
    
    # Create computer
    computer_config = ScoreComputerConfig(
        use_float64=True,
        batch_size=32,
        device="cpu",
    )
    computer = ScoreComputer(computer_config, model)
    
    # Test score computation
    data = torch.randn(100, 115)
    scores = computer.compute_scores(data)
    
    assert scores.shape == (100,)
    assert scores.dtype == torch.float64
    assert scores.dim() == 1
    
    # Test client scores
    role_data = {
        "R": torch.randn(50, 115),
        "G": torch.randn(60, 115),
        "C": torch.randn(70, 115),
    }
    client_scores = computer.compute_client_scores(role_data, client_id="nb01")
    
    assert client_scores.client_id == "nb01"
    assert "R" in client_scores.role_scores
    assert "G" in client_scores.role_scores
    assert "C" in client_scores.role_scores
    
    # Verify float64
    for role, rs in client_scores.role_scores.items():
        assert rs.scores.dtype == np.float64
    
    print("Score computer verification passed.")


if __name__ == "__main__":
    import numpy as np
    verify_computer()
