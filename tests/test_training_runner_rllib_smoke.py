from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from mahjong_ai.core.engine import GameEngine
from mahjong_ai.rules.schema import RulesConfig
from mahjong_ai.training.rllib_runner import train_with_rllib


def _has_mod(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


HAS_RL_STACK = _has_mod("ray") and _has_mod("torch") and _has_mod("gymnasium") and _has_mod("numpy")


@unittest.skipUnless(HAS_RL_STACK, "ray/torch/gymnasium/numpy are required for RL training smoke test")
class TestTrainingRunnerRllibSmoke(unittest.TestCase):
    def test_train_and_resume_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)

            cfg = {
                "experiment_name": "smoke_train",
                "seed": 7,
                "run_dir": str(td_path / "run_a"),
                "num_iterations": 1,
                "checkpoint_every": 1,
                "rollout_workers": 0,
                "num_envs_per_worker": 1,
                "train_batch_size": 512,
                "sgd_minibatch_size": 64,
                "num_sgd_iter": 1,
                "lr": 3e-4,
                "clip_param": 0.2,
                "gamma": 0.99,
                "lambda": 0.95,
                "num_gpus": 0,
                "evaluation": {"eval_every": 0, "eval_games": 1},
                "model": {"hidden_sizes": [64]},
            }

            engine = GameEngine(rules=RulesConfig(), enable_events=False)
            train_with_rllib(engine=engine, config=cfg)

            checkpoint_dir = td_path / "run_a" / "smoke_train"
            self.assertTrue((checkpoint_dir / "rllib_checkpoint.json").exists())

            cfg_resume = dict(cfg)
            cfg_resume["experiment_name"] = "smoke_resume"
            cfg_resume["run_dir"] = str(td_path / "run_b")
            cfg_resume["seed"] = 11

            engine_resume = GameEngine(rules=RulesConfig(), enable_events=False)
            train_with_rllib(
                engine=engine_resume,
                config=cfg_resume,
                resume_from=str(checkpoint_dir),
            )

            resumed_ckpt = td_path / "run_b" / "smoke_resume" / "rllib_checkpoint.json"
            self.assertTrue(resumed_ckpt.exists())


if __name__ == "__main__":
    unittest.main()
