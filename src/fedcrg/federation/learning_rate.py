"""Federated learning-rate scheduling."""

import math


def cosine_learning_rate(round_index: int, rounds: int, initial: float, final: float) -> float:
    if rounds <= 0 or not 0 <= round_index < rounds:
        raise ValueError("round_index must be inside the configured training horizon")
    if rounds == 1:
        return final
    progress = round_index / (rounds - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return final + (initial - final) * cosine
