from __future__ import annotations

from dataclasses import dataclass

from mahjong_ai.core.engine import GameEngine
from mahjong_ai.env.obs_encoder import encode_observation


@dataclass(slots=True)
class EnvStep:
    obs: dict[int, dict[str, object]]
    rewards: dict[int, float]
    terminateds: dict[object, bool]
    truncateds: dict[object, bool]
    infos: dict[int, dict[str, object]]


class SimpleMultiAgentEnv:
    """A lightweight multi-agent env wrapper around the engine.

    This is *not* tied to any RL framework. It returns RLlib-like dicts to keep
    integration straightforward later, but it does not require Ray/RLlib.
    """

    def __init__(self, engine: GameEngine):
        self.engine = engine
        self.state = None

    def reset(self, *, seed: int) -> dict[int, dict[str, object]]:
        self.state = self.engine.reset(seed=seed)
        return self._collect_obs()

    def step(self, action_dict: dict[int, int]) -> EnvStep:
        if self.state is None:
            raise RuntimeError("env.reset() must be called before step()")

        res = self.engine.step(self.state, action_dict)
        rewards = {i: float(res.score_delta[i]) for i in range(4)}

        terminateds: dict[object, bool] = {i: bool(res.done or self.state.players[i].won) for i in range(4)}
        terminateds["__all__"] = bool(res.done)
        truncateds: dict[object, bool] = {i: False for i in range(4)}
        truncateds["__all__"] = False

        obs = self._collect_obs()
        infos = {i: {} for i in range(4)}

        return EnvStep(obs=obs, rewards=rewards, terminateds=terminateds, truncateds=truncateds, infos=infos)

    def _collect_obs(self) -> dict[int, dict[str, object]]:
        assert self.state is not None
        obs: dict[int, dict[str, object]] = {}
        for pid in self.engine.required_players(self.state):
            obs[pid] = {
                "obs": encode_observation(self.state, pid),
                "action_mask": self.engine.legal_action_mask(self.state, pid),
            }
        return obs
