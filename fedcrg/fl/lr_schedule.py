"""
Learning Rate Schedule

Implements the cosine learning rate schedule per Section 8.1.1.

Normative reference: Section 8.1.1
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass(frozen=True, slots=True)
class CosineLearningRateSchedule:
    """
    Cosine learning rate schedule.
    
    Implements the exact formula from Section 8.1.1:
    eta_t = eta_min + 0.5 * (eta_0 - eta_min) * (1 + cos(pi * t / 29))
    
    where t ranges from 0 to 29 (30 rounds total).
    
    Normative reference: Section 8.1.1
    """
    eta_0: float = 1e-3  # Initial learning rate
    eta_min: float = 1e-5  # Final learning rate
    num_rounds: int = 30
    
    def get_lr(self, round: int) -> float:
        """
        Get learning rate for a given round.
        
        Args:
            round: Round index (0 to 29)
            
        Returns:
            Learning rate for the round
        """
        # Clamp round to valid range
        t = max(0, min(round, self.num_rounds - 1))
        
        # Cosine schedule formula
        lr = self.eta_min + 0.5 * (self.eta_0 - self.eta_min) * (
            1 + math.cos(math.pi * t / (self.num_rounds - 1))
        )
        return lr
    
    def get_all_lrs(self) -> list[float]:
        """
        Get learning rates for all rounds.
        
        Returns:
            List of learning rates for rounds 0 to 29
        """
        return [self.get_lr(t) for t in range(self.num_rounds)]


def get_lr_for_round(
    round: int,
    eta_0: float = 1e-3,
    eta_min: float = 1e-5,
    num_rounds: int = 30,
) -> float:
    """
    Get learning rate for a given round using the cosine schedule.
    
    This is a convenience function that implements the exact formula from
    Section 8.1.1 without creating a schedule object.
    
    Args:
        round: Round index (0 to 29)
        eta_0: Initial learning rate (default 1e-3)
        eta_min: Final learning rate (default 1e-5)
        num_rounds: Total number of rounds (default 30)
        
    Returns:
        Learning rate for the round
        
    Normative reference: Section 8.1.1
    """
    t = max(0, min(round, num_rounds - 1))
    lr = eta_min + 0.5 * (eta_0 - eta_min) * (
        1 + math.cos(math.pi * t / (num_rounds - 1))
    )
    return lr


# Precompute the standard schedule values for verification
STANDARD_SCHEDULE = CosineLearningRateSchedule()

# Verify against known values
# At t=0: eta_0 = 1e-3
# At t=29: eta_min = 1e-5
# At t=14 or 15: should be approximately the midpoint


def verify_schedule() -> None:
    """
    Verify the learning rate schedule against known values.
    """
    schedule = CosineLearningRateSchedule()
    
    # Check endpoints
    assert abs(schedule.get_lr(0) - 1e-3) < 1e-12, f"LR at round 0: {schedule.get_lr(0)}"
    assert abs(schedule.get_lr(29) - 1e-5) < 1e-12, f"LR at round 29: {schedule.get_lr(29)}"
    
    # Check symmetry (cosine is symmetric around pi/2)
    for t in range(15):
        lr_t = schedule.get_lr(t)
        lr_29_minus_t = schedule.get_lr(29 - t)
        # Due to floating point, allow small tolerance
        assert abs(lr_t - lr_29_minus_t) < 1e-10, \
            f"Symmetry check failed: t={t}, 29-t={29-t}, " \
            f"lr_t={lr_t}, lr_29_t={lr_29_minus_t}"
    
    print("Learning rate schedule verification passed.")


if __name__ == "__main__":
    verify_schedule()
    
    # Print schedule for reference
    schedule = CosineLearningRateSchedule()
    print("\nCosine Learning Rate Schedule (eta_0=1e-3, eta_min=1e-5, 30 rounds):")
    for t in range(30):
        print(f"  Round {t:2d}: {schedule.get_lr(t):.2e}")
