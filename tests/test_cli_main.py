from __future__ import annotations

import sys
import unittest

from mahjong_ai.cli.main import _parse_seed_list, build_parser


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
            "--python-bin",
            "/usr/bin/python3",
        ])
        self.assertEqual(args.cmd, "grid-rllib")
        self.assertEqual(args.pool_sizes, "2,4")
        self.assertEqual(args.snapshot_intervals, "1")
        self.assertEqual(args.main_probs, "0.2,0.4")
        self.assertEqual(args.python_bin, "/usr/bin/python3")

    def test_grid_rllib_parser_warning_toggles_default_keep(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["grid-rllib"])
        self.assertEqual(args.cmd, "grid-rllib")
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
