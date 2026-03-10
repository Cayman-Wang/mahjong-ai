from __future__ import annotations


def score_delta_to_rewards(score_delta: list[int]) -> dict[int, float]:
    """Convert engine score deltas to per-agent rewards."""
    return {i: float(score_delta[i]) for i in range(4)}

