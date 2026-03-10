from __future__ import annotations

from mahjong_ai.agents.base import Agent
from mahjong_ai.agents.heuristic_agent import HeuristicAgent
from mahjong_ai.agents.random_agent import RandomAgent


def make_agent(name: str, *, seed: int | None = None) -> Agent:
    name = name.strip().lower()
    if name in {"random", "rand"}:
        return RandomAgent(seed=seed)
    if name in {"heuristic", "rule"}:
        return HeuristicAgent(seed=seed)
    raise ValueError(f"unknown agent: {name}")

