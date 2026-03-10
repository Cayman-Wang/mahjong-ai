from __future__ import annotations

import secrets

from typing import Any

from mahjong_ai.core.actions import N_ACTIONS
from mahjong_ai.core.engine import GameEngine
from mahjong_ai.env.obs_vector_encoder import OBS_VECTOR_DIM, encode_observation_vector

try:  # pragma: no cover
    import numpy as np
    from gymnasium import spaces
    from ray.rllib.env.multi_agent_env import MultiAgentEnv  # type: ignore
except Exception as e:  # pragma: no cover
    np = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]
    MultiAgentEnv = object  # type: ignore[assignment]
    _RLLIB_IMPORT_ERROR = e
else:  # pragma: no cover
    _RLLIB_IMPORT_ERROR = None


class RllibMahjongEnv(MultiAgentEnv):  # type: ignore[misc]
    """RLlib MultiAgentEnv wrapper with fixed vector observations + action mask."""

    def __init__(self, engine: GameEngine):
        if _RLLIB_IMPORT_ERROR is not None:  # pragma: no cover
            raise RuntimeError(
                "Ray RLlib env deps are not available. Install optional deps and use a supported Python version."
            ) from _RLLIB_IMPORT_ERROR
        super().__init__()
        self.engine = engine
        self.state = None
        self._terminated_agents: set[int] = set()

        single_observation_space = spaces.Dict(
            {
                "obs": spaces.Box(
                    low=-1e6,
                    high=1e6,
                    shape=(OBS_VECTOR_DIM,),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(low=0.0, high=1.0, shape=(N_ACTIONS,), dtype=np.float32),
            }
        )
        single_action_space = spaces.Discrete(N_ACTIONS)

        # Keep single-agent style spaces for backward compatibility.
        self.observation_space = single_observation_space
        self.action_space = single_action_space

        # RLlib new API stack reads per-agent spaces from these mappings.
        self.possible_agents = [0, 1, 2, 3]
        self.agents = self.possible_agents.copy()
        self.observation_spaces = {pid: single_observation_space for pid in self.possible_agents}
        self.action_spaces = {pid: single_action_space for pid in self.possible_agents}
        self._pending_rewards = {pid: 0.0 for pid in self.possible_agents}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        if seed is None:
            # RLlib may call reset() without a seed; avoid constant-seed episodes by default.
            seed = secrets.randbits(31)
        self.state = self.engine.reset(seed=int(seed))
        self._terminated_agents = set()
        self.agents = self.possible_agents.copy()
        self._pending_rewards = {pid: 0.0 for pid in self.possible_agents}
        return self._collect_obs(), {}

    def step(self, action_dict):
        assert self.state is not None
        res = self.engine.step(self.state, action_dict)
        obs = self._collect_obs()

        done_now = {pid for pid in self.possible_agents if (res.done or self.state.players[pid].won)}
        newly_done = done_now - self._terminated_agents

        # Accumulate score deltas for alive agents and emit rewards only for
        # agents that currently have observations or just terminated.
        reward_sources = set(self.possible_agents) - self._terminated_agents
        for pid in reward_sources:
            self._pending_rewards[pid] += float(res.score_delta[pid])

        reward_keys = set(obs.keys()) | newly_done
        rewards = {pid: self._pending_rewards.get(pid, 0.0) for pid in reward_keys}
        for pid in reward_keys:
            self._pending_rewards[pid] = 0.0

        terminateds = {pid: False for pid in obs.keys()}
        terminateds.update({pid: True for pid in newly_done})
        terminateds["__all__"] = bool(res.done)

        truncateds = {pid: False for pid in terminateds.keys() if pid != "__all__"}
        truncateds["__all__"] = False

        info_keys = set(obs.keys()) | newly_done
        infos = {pid: {} for pid in info_keys}

        self._terminated_agents |= newly_done
        self.agents = [pid for pid in self.possible_agents if pid not in self._terminated_agents]
        return obs, rewards, terminateds, truncateds, infos

    def _collect_obs(self):
        assert self.state is not None
        obs = {}
        for pid in self.engine.required_players(self.state):
            if pid in self._terminated_agents:
                continue
            obs[pid] = {
                "obs": encode_observation_vector(self.state, pid),
                "action_mask": np.asarray(self.engine.legal_action_mask(self.state, pid), dtype=np.float32),
            }
        return obs
