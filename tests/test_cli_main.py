from __future__ import annotations

import argparse
import sys
import unittest
from unittest import mock

from mahjong_ai.cli.main import _parse_seed_list, build_parser, cmd_eval_benchmark
from mahjong_ai.evaluation.benchmark_config import EvalBenchmarkConfig


class TestCliMain(unittest.TestCase):
    def test_eval_rllib_parser_accepts_checkpoint(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["eval-rllib", "--checkpoint", "runs/ppo_selfplay"])
        self.assertEqual(args.cmd, "eval-rllib")
        self.assertEqual(args.checkpoint, "runs/ppo_selfplay")
        self.assertEqual(args.baselines, "heuristic,random")

    def test_train_rllib_parser_accepts_warning_toggles(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "train-rllib",
            "--quiet-ray-future-warning",
            "--no-quiet-new-api-stack-warning",
            "--quiet-ray-deprecation-warning",
            "--no-strict-illegal-action",
        ])
        self.assertEqual(args.cmd, "train-rllib")
        self.assertTrue(args.quiet_ray_future_warning)
        self.assertFalse(args.quiet_new_api_stack_warning)
        self.assertTrue(args.quiet_ray_deprecation_warning)
        self.assertFalse(args.strict_illegal_action)

    def test_eval_rllib_parser_accepts_warning_toggles(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "eval-rllib",
            "--checkpoint",
            "runs/ppo_selfplay",
            "--quiet-new-api-stack-warning",
            "--quiet-ray-deprecation-warning",
            "--no-strict-illegal-action",
        ])
        self.assertEqual(args.cmd, "eval-rllib")
        self.assertTrue(args.quiet_new_api_stack_warning)
        self.assertFalse(args.quiet_ray_future_warning)
        self.assertTrue(args.quiet_ray_deprecation_warning)
        self.assertFalse(args.strict_illegal_action)

    def test_eval_benchmark_parser_accepts_config_and_checkpoint(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "eval-benchmark",
            "--config",
            "configs/eval/standard.yaml",
            "--checkpoint",
            "runs/ppo_selfplay",
        ])
        self.assertEqual(args.cmd, "eval-benchmark")
        self.assertEqual(args.config, "configs/eval/standard.yaml")
        self.assertEqual(args.checkpoint, "runs/ppo_selfplay")

    def test_cmd_eval_benchmark_uses_loaded_protocol(self) -> None:
        args = argparse.Namespace(
            config="configs/eval/standard.yaml",
            checkpoint="runs/ppo_selfplay",
            output=None,
        )
        cfg = EvalBenchmarkConfig(
            benchmark_name="standard",
            rules_path="configs/rules/sichuan_xuezhan_default.yaml",
            baselines=["heuristic"],
            games=4,
            seed=101,
            seed_list=[101, 102, 103, 104],
            strict_illegal_action=False,
        )

        with mock.patch("mahjong_ai.cli.main.load_eval_benchmark_config", return_value=cfg), mock.patch(
            "mahjong_ai.training.rllib_runner.run_evaluation_entry"
        ) as mocked_eval:
            rc = cmd_eval_benchmark(args)

        self.assertEqual(rc, 0)
        mocked_eval.assert_called_once()
        kwargs = mocked_eval.call_args.kwargs
        self.assertEqual(kwargs["benchmark_name"], "standard")
        self.assertEqual(kwargs["games"], 4)
        self.assertEqual(kwargs["seed_list"], [101, 102, 103, 104])

    def test_grid_rllib_parser_accepts_params(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "grid-rllib",
            "--pool-sizes",
            "2,4",
            "--snapshot-intervals",
            "1",
            "--main-probs",
            "0.2,0.4",
            "--benchmark-config",
            "configs/eval/smoke.yaml",
            "--python-bin",
            "/usr/bin/python3",
        ])
        self.assertEqual(args.cmd, "grid-rllib")
        self.assertEqual(args.pool_sizes, "2,4")
        self.assertEqual(args.snapshot_intervals, "1")
        self.assertEqual(args.main_probs, "0.2,0.4")
        self.assertEqual(args.benchmark_config, "configs/eval/smoke.yaml")
        self.assertEqual(args.python_bin, "/usr/bin/python3")

    def test_grid_rllib_parser_warning_toggles_default_keep(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["grid-rllib"])
        self.assertEqual(args.cmd, "grid-rllib")
        self.assertEqual(args.benchmark_config, "")
        self.assertFalse(args.quiet_ray_future_warning)
        self.assertFalse(args.quiet_new_api_stack_warning)
        self.assertFalse(args.quiet_ray_deprecation_warning)
        self.assertEqual(args.python_bin, sys.executable)

    def test_parse_seed_list_empty(self) -> None:
        self.assertEqual(_parse_seed_list(""), [])

    def test_parse_seed_list_values(self) -> None:
        self.assertEqual(_parse_seed_list("1, 2,3"), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
