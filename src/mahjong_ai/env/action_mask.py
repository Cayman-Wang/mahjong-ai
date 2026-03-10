from __future__ import annotations

from mahjong_ai.core.engine import GameEngine
from mahjong_ai.core.state import GameState


def get_action_mask(engine: GameEngine, state: GameState, pid: int) -> list[int]:
    return engine.legal_action_mask(state, pid)

