from __future__ import annotations

import importlib.util
import unittest


def _has_mod(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


HAS_RLLIB_ENV = _has_mod("ray") and _has_mod("gymnasium") and _has_mod("numpy")


@unittest.skipUnless(HAS_RLLIB_ENV, "ray/gymnasium/numpy are required for RLlib env tests")
class TestRllibEnvContract(unittest.TestCase):
    def test_spaces_and_obs_contract(self) -> None:
        from mahjong_ai.core.actions import N_ACTIONS
        from mahjong_ai.core.engine import GameEngine
        from mahjong_ai.env.rllib_multiagent_env import RllibMahjongEnv
        from mahjong_ai.rules.schema import RulesConfig

        env = RllibMahjongEnv(engine=GameEngine(rules=RulesConfig(), enable_events=False))
        self.assertEqual(int(env.action_space.n), N_ACTIONS)
        self.assertEqual(sorted(env.action_spaces.keys()), [0, 1, 2, 3])
        self.assertEqual(sorted(env.observation_spaces.keys()), [0, 1, 2, 3])

        obs, info = env.reset(seed=1)
        self.assertEqual(info, {})
        self.assertTrue(obs)

        pid, sample = next(iter(obs.items()))
        self.assertIsInstance(pid, int)
        self.assertTrue(env.observation_space.contains(sample))
        self.assertEqual(sample["action_mask"].shape[0], N_ACTIONS)

    def test_reward_keys_subset_of_obs_or_newly_done(self) -> None:
        from mahjong_ai.core.engine import GameEngine
        from mahjong_ai.env.rllib_multiagent_env import RllibMahjongEnv
        from mahjong_ai.rules.schema import RulesConfig

        env = RllibMahjongEnv(engine=GameEngine(rules=RulesConfig(), enable_events=False))
        obs, _ = env.reset(seed=1)
        steps = 0

        while True:
            action_dict = {}
            assert env.state is not None
            for pid in env.engine.required_players(env.state):
                mask = env.engine.legal_action_mask(env.state, pid)
                action_dict[pid] = next(i for i, legal in enumerate(mask) if legal == 1)

            obs, rewards, terminateds, truncateds, infos = env.step(action_dict)
            del truncateds

            newly_done = {pid for pid, done in terminateds.items() if pid != "__all__" and bool(done)}
            expected_keys = set(obs.keys()) | newly_done
            self.assertTrue(set(rewards.keys()).issubset(expected_keys))
            self.assertTrue(set(infos.keys()).issubset(expected_keys))

            steps += 1
            if terminateds.get("__all__"):
                break
            if steps > 3000:
                self.fail("environment did not terminate in expected number of steps")


if __name__ == "__main__":
    unittest.main()
