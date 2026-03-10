from __future__ import annotations

import importlib.util
import unittest


def _has_mod(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


HAS_MODEL_DEPS = _has_mod("ray") and _has_mod("torch") and _has_mod("gymnasium") and _has_mod("numpy")


@unittest.skipUnless(HAS_MODEL_DEPS, "ray/torch/gymnasium/numpy are required for RLModule tests")
class TestRllibActionMaskRLModule(unittest.TestCase):
    def test_illegal_actions_get_low_logits(self) -> None:
        import numpy as np
        import torch
        from gymnasium import spaces
        from ray.rllib.core.columns import Columns

        from mahjong_ai.core.actions import N_ACTIONS
        from mahjong_ai.env.obs_vector_encoder import OBS_VECTOR_DIM
        from mahjong_ai.training.rllib_action_mask_rl_module import MahjongActionMaskTorchRLModule

        obs_space = spaces.Dict(
            {
                "obs": spaces.Box(low=-1.0, high=1.0, shape=(OBS_VECTOR_DIM,), dtype=np.float32),
                "action_mask": spaces.Box(low=0.0, high=1.0, shape=(N_ACTIONS,), dtype=np.float32),
            }
        )
        action_space = spaces.Discrete(N_ACTIONS)

        module = MahjongActionMaskTorchRLModule(
            observation_space=obs_space,
            action_space=action_space,
            model_config={
                "fcnet_hiddens": [32],
                "head_fcnet_hiddens": [],
                "vf_share_layers": True,
            },
        )

        obs = torch.zeros((1, OBS_VECTOR_DIM), dtype=torch.float32)
        action_mask = torch.zeros((1, N_ACTIONS), dtype=torch.float32)
        action_mask[0, 3] = 1.0

        infer_batch = {
            Columns.OBS: {
                "obs": obs.clone(),
                "action_mask": action_mask.clone(),
            }
        }
        outs = module.forward_inference(infer_batch)
        logits = outs[Columns.ACTION_DIST_INPUTS]

        self.assertTrue(torch.isfinite(logits[0, 3]))
        self.assertTrue(torch.isfinite(logits[0, 0]))
        self.assertLess(float(logits[0, 0]), -1e30)

        values = module.compute_values(
            {
                Columns.OBS: {
                    "obs": obs.clone(),
                    "action_mask": action_mask.clone(),
                }
            }
        )
        self.assertEqual(tuple(values.shape), (1,))

    def test_forward_inference_does_not_mutate_input_batch(self) -> None:
        import numpy as np
        import torch
        from gymnasium import spaces
        from ray.rllib.core.columns import Columns

        from mahjong_ai.core.actions import N_ACTIONS
        from mahjong_ai.env.obs_vector_encoder import OBS_VECTOR_DIM
        from mahjong_ai.training.rllib_action_mask_rl_module import MahjongActionMaskTorchRLModule

        obs_space = spaces.Dict(
            {
                "obs": spaces.Box(low=-1.0, high=1.0, shape=(OBS_VECTOR_DIM,), dtype=np.float32),
                "action_mask": spaces.Box(low=0.0, high=1.0, shape=(N_ACTIONS,), dtype=np.float32),
            }
        )
        action_space = spaces.Discrete(N_ACTIONS)

        module = MahjongActionMaskTorchRLModule(
            observation_space=obs_space,
            action_space=action_space,
            model_config={
                "fcnet_hiddens": [16],
                "head_fcnet_hiddens": [],
                "vf_share_layers": True,
            },
        )

        batch = {
            Columns.OBS: {
                "obs": torch.zeros((1, OBS_VECTOR_DIM), dtype=torch.float32),
                "action_mask": torch.ones((1, N_ACTIONS), dtype=torch.float32),
            }
        }
        original_obs = batch[Columns.OBS]["obs"]
        original_mask = batch[Columns.OBS]["action_mask"]

        _ = module.forward_inference(batch)

        self.assertIsInstance(batch[Columns.OBS], dict)
        self.assertIn("obs", batch[Columns.OBS])
        self.assertIn("action_mask", batch[Columns.OBS])
        self.assertIs(batch[Columns.OBS]["obs"], original_obs)
        self.assertIs(batch[Columns.OBS]["action_mask"], original_mask)
        self.assertNotIn("action_mask", batch)


if __name__ == "__main__":
    unittest.main()
