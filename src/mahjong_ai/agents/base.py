from __future__ import annotations

from dataclasses import dataclass

from mahjong_ai.core.state import GameState


@dataclass(slots=True)
class AgentContext:
    seat: int


class Agent:
    """Minimal agent interface used by the CLI simulator.

    The engine is multi-agent and has simultaneous decision phases, so `act()`
    is called for every player that is required to act in the current phase.
    """

    name: str = "agent"

    def act(self, ctx: AgentContext, state: GameState, action_mask: list[int]) -> int:
        raise NotImplementedError

