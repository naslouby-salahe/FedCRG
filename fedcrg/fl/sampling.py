"""
Deterministic Sampling

Implements deterministic shuffling per Section 8.2.1.

Normative reference: Section 8.2.1 (Exact local batch semantics)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


@dataclass(frozen=True, slots=True)
class DeterministicSampler(Sampler[int]):
    """
    Deterministic sampler for federated training.
    
    Implements deterministic shuffling per Section 8.2.1 using a hash-seeded
    PCG64 generator. The shuffle seed is derived from:
    SHA256("fedcrg|shuffle|" || model_seed || client_id || round || local_epoch)
    
    This ensures that the shuffle order is deterministic and reproducible
    across different runs with the same parameters.
    
    Normative reference: Section 8.2.1
    """
    data_source: Dataset
    model_seed: int
    client_id: str
    num_rounds: int = 30
    num_local_epochs: int = 120
    batch_size: int = 64
    drop_last: bool = False
    
    def __post_init__(self):
        """Validate parameters."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.num_rounds <= 0:
            raise ValueError(f"num_rounds must be positive, got {self.num_rounds}")
        if self.num_local_epochs <= 0:
            raise ValueError(f"num_local_epochs must be positive, got {self.num_local_epochs}")
    
    def _get_seed(
        self,
        round: int,
        local_epoch: int,
    ) -> int:
        """
        Get deterministic seed for a specific round and epoch.
        
        The seed is derived from a hash of the concatenated parameters.
        
        Args:
            round: Current round index (0 to num_rounds-1)
            local_epoch: Current local epoch index (0 to num_local_epochs-1)
            
        Returns:
            64-bit unsigned integer seed
        """
        # Create seed string
        seed_str = f"fedcrg|shuffle|{self.model_seed}|{self.client_id}|{round}|{local_epoch}"
        
        # Hash and convert to 64-bit unsigned integer
        seed_bytes = hashlib.sha256(seed_str.encode()).digest()
        seed = int.from_bytes(seed_bytes[:8], byteorder='big', signed=False)
        
        return seed
    
    def _create_generator(self, seed: int) -> np.random.Generator:
        """
        Create a PCG64 generator with the given seed.
        
        Args:
            seed: 64-bit unsigned integer seed
            
        Returns:
            NumPy random generator
        """
        return np.random.default_rng(seed)
    
    def get_shuffle_indices(
        self,
        n: int,
        round: int,
        local_epoch: int,
    ) -> np.ndarray:
        """
        Get deterministic shuffle indices for a given round and epoch.
        
        Args:
            n: Number of samples to shuffle
            round: Current round index
            local_epoch: Current local epoch index
            
        Returns:
            Array of shuffled indices
        """
        seed = self._get_seed(round, local_epoch)
        rng = self._create_generator(seed)
        indices = rng.permutation(n)
        return indices
    
    def get_batch_indices(
        self,
        n: int,
        round: int,
        local_epoch: int,
    ) -> list[list[int]]:
        """
        Get deterministic batch indices for a given round and epoch.
        
        Args:
            n: Number of samples
            round: Current round index
            local_epoch: Current local epoch index
            
        Returns:
            List of batch index lists
        """
        shuffle_indices = self.get_shuffle_indices(n, round, local_epoch)
        
        num_batches = (n + self.batch_size - 1) // self.batch_size
        if self.drop_last and n % self.batch_size != 0:
            num_batches -= 1
        
        batches = []
        for i in range(num_batches):
            start = i * self.batch_size
            end = start + self.batch_size
            if not self.drop_last or end <= n:
                batch = shuffle_indices[start:end].tolist()
                batches.append(batch)
        
        return batches
    
    def __iter__(self):
        """
        Iterate over samples in deterministic order.
        
        This sampler is designed to be used per epoch, not for the entire
        training loop. For FL training, use get_shuffle_indices or get_batch_indices
        directly.
        """
        # For compatibility with Sampler interface
        # But in FL we typically use get_shuffle_indices per epoch
        n = len(self.data_source)
        seed = self._get_seed(0, 0)  # Default to round 0, epoch 0
        rng = self._create_generator(seed)
        indices = rng.permutation(n)
        return iter(indices.tolist())
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.data_source)


def create_deterministic_sampler(
    dataset: Dataset,
    model_seed: int,
    client_id: str,
    num_rounds: int = 30,
    num_local_epochs: int = 120,
    batch_size: int = 64,
    drop_last: bool = False,
) -> DeterministicSampler:
    """
    Create a deterministic sampler for federated training.
    
    Args:
        dataset: PyTorch Dataset to sample from
        model_seed: Global model seed
        client_id: Client identifier
        num_rounds: Number of federated rounds
        num_local_epochs: Number of local epochs per round
        batch_size: Batch size
        drop_last: Whether to drop the last incomplete batch
        
    Returns:
        DeterministicSampler instance
        
    Normative reference: Section 8.2.1
    """
    return DeterministicSampler(
        data_source=dataset,
        model_seed=model_seed,
        client_id=client_id,
        num_rounds=num_rounds,
        num_local_epochs=num_local_epochs,
        batch_size=batch_size,
        drop_last=drop_last,
    )


def verify_determinism() -> None:
    """
    Verify that the sampler produces deterministic results.
    """
    # Create a simple dummy dataset
    class DummyDataset:
        def __len__(self):
            return 100
        def __getitem__(self, idx):
            return idx
    
    dataset = DummyDataset()
    
    # Create sampler with specific parameters
    sampler = DeterministicSampler(
        data_source=dataset,
        model_seed=42,
        client_id="nb01",
        num_rounds=30,
        num_local_epochs=120,
        batch_size=64,
        drop_last=False,
    )
    
    # Get indices multiple times and verify they are the same
    indices1 = sampler.get_shuffle_indices(100, round=0, local_epoch=0)
    indices2 = sampler.get_shuffle_indices(100, round=0, local_epoch=0)
    
    assert np.array_equal(indices1, indices2), "Indices are not deterministic!"
    
    # Verify that different rounds/epochs produce different shuffles
    indices_r1 = sampler.get_shuffle_indices(100, round=1, local_epoch=0)
    assert not np.array_equal(indices1, indices_r1), "Different rounds should produce different shuffles"
    
    # Verify batch computation
    batches = sampler.get_batch_indices(100, round=0, local_epoch=0)
    assert len(batches) == 2  # ceil(100/64) = 2
    assert len(batches[0]) == 64
    assert len(batches[1]) == 36  # 100 - 64 = 36
    
    # Verify all indices are unique and cover the full range
    all_indices = [idx for batch in batches for idx in batch]
    assert sorted(all_indices) == list(range(100)), "Indices don't cover full range"
    
    print("Deterministic sampler verification passed.")


if __name__ == "__main__":
    verify_determinism()
