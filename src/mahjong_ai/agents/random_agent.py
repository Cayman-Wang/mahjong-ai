from __future__ import annotations

import random

from mahjong_ai.agents.base import Agent, AgentContext
from mahjong_ai.core.state import GameState


class RandomAgent(Agent):
    name = "random"

    def __init__(self, *, seed: int | None = None):
        self._rng = random.Random(seed)

    def act(self, ctx: AgentContext, state: GameState, action_mask: list[int]) -> int:
        legal = [i for i, m in enumerate(action_mask) if m == 1]
        if not legal:
            raise RuntimeError(f"no legal actions for player {ctx.seat} in phase {state.phase}")
        return self._rng.choice(legal)

