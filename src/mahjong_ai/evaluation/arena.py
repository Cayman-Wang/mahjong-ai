from __future__ import annotations

from mahjong_ai.agents.base import AgentContext
from mahjong_ai.agents.registry import make_agent
from mahjong_ai.core.engine import GameEngine
from mahjong_ai.evaluation.metrics import MatchStats


def play_match(
    engine: GameEngine,
    *,
    seed: int,
    games: int,
    agents: list[str],
) -> MatchStats:
    if len(agents) != 4:
        raise ValueError("agents must have length 4")

    total = [0, 0, 0, 0]

    for g in range(games):
        s = seed + g
        agent_objs = [make_agent(agents[i], seed=s + i * 9973) for i in range(4)]
        ctxs = [AgentContext(seat=i) for i in range(4)]

        state = engine.reset(seed=s)
        steps = 0
        while True:
            if state.phase.value == "round_end":
                break
            required = engine.required_players(state)
            actions = {}
            for pid in required:
                mask = engine.legal_action_mask(state, pid)
                actions[pid] = agent_objs[pid].act(ctxs[pid], state, mask)
            res = engine.step(state, actions)
            steps += 1
            if res.done:
                break
            if steps > 5000:
                raise RuntimeError("arena exceeded step limit; possible engine bug")

        for i in range(4):
            total[i] += state.scores[i]

    return MatchStats(games=games, total_scores=total)

