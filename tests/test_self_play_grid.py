from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mahjong_ai.evaluation.benchmark_config import EvalBenchmarkConfig
from mahjong_ai.training import self_play_grid as grid


class TestSelfPlayGrid(unittest.TestCase):
    def test_build_grid_combinations_dedup_and_validation(self) -> None:
        combos = grid.build_grid_combinations(
            pool_sizes=[2, 2],
            snapshot_intervals=[1],
            main_probs=[0.2, 0.2],
        )
        self.assertEqual(len(combos), 1)
        self.assertEqual(combos[0].pool_size, 2)
        self.assertEqual(combos[0].snapshot_interval, 1)
        self.assertAlmostEqual(combos[0].main_prob, 0.2)

        with self.assertRaisesRegex(ValueError, "main_prob"):
            grid.build_grid_combinations(
                pool_sizes=[2],
                snapshot_intervals=[1],
                main_probs=[1.2],
            )

    def test_build_experiment_name(self) -> None:
        combo = grid.GridCombo(pool_size=4, snapshot_interval=5, main_prob=0.25)
        self.assertEqual(
            grid.build_experiment_name(prefix="grid", combo=combo),
            "grid_pool4_snap5_main0p25",
        )

    def test_render_markdown_has_sorted_table_rows(self) -> None:
        rows = [
            {
                "experiment_name": "exp_b",
                "pool_size": 2,
                "snapshot_interval": 1,
                "main_prob": 0.2,
                "primary_baseline": "heuristic",
                "primary_win_rate": 0.4,
                "primary_avg_score": 3.0,
                "report_path": "runs/exp_b/eval/iter_000001.json",
            },
            {
                "experiment_name": "exp_a",
                "pool_size": 4,
                "snapshot_interval": 5,
                "main_prob": 0.1,
                "primary_baseline": "heuristic",
                "primary_win_rate": 0.5,
                "primary_avg_score": 1.0,
                "report_path": "runs/exp_a/eval/iter_000001.json",
            },
        ]
        md = grid._render_markdown(rows)
        self.assertIn("| rank | experiment |", md)
        self.assertIn("exp_a", md)
        self.assertIn("exp_b", md)

    def test_apply_benchmark_to_grid_config_uses_effective_seed_list_and_rules_fallback(self) -> None:
        cfg = {
            "seed": 7,
            "evaluation": {
                "eval_games": 1,
                "baselines": ["random"],
                "seed_list": [],
                "strict_illegal_action": False,
            }
        }
        benchmark_cfg = EvalBenchmarkConfig(
            benchmark_name="smoke",
            rules_path="configs/rules/sichuan_xuezhan_default.yaml",
            baselines=["heuristic", "random"],
            games=3,
            seed=101,
            seed_list=[],
            strict_illegal_action=True,
        )

        merged_cfg, rules_path, metadata = grid._apply_benchmark_to_grid_config(
            train_cfg=cfg,
            rules_path="",
            benchmark_cfg=benchmark_cfg,
            benchmark_config_path="configs/eval/smoke.yaml",
        )

        self.assertEqual(merged_cfg["evaluation"]["eval_games"], 3)
        self.assertEqual(merged_cfg["evaluation"]["baselines"], ["heuristic", "random"])
        self.assertEqual(merged_cfg["evaluation"]["seed_list"], [101, 102, 103])
        self.assertTrue(merged_cfg["evaluation"]["strict_illegal_action"])
        self.assertEqual(rules_path, "configs/rules/sichuan_xuezhan_default.yaml")
        self.assertEqual(metadata["benchmark_name"], "smoke")
        self.assertEqual(metadata["effective_seed_list"], [101, 102, 103])
        self.assertTrue(metadata["benchmark_config_path"].endswith("configs/eval/smoke.yaml"))

    def test_run_self_play_grid_applies_benchmark_protocol_to_config_command_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = root / "train.json"
            config_path.write_text(
                json.dumps(
                    {
                        "experiment_name": "ppo_selfplay_base",
                        "evaluation": {
                            "eval_every": 1,
                            "eval_games": 1,
                            "baselines": ["random"],
                            "seed_list": [],
                            "strict_illegal_action": False,
                        },
                    }
                ),
                encoding="utf-8",
            )
            run_dir = root / "runs"
            benchmark_cfg = EvalBenchmarkConfig(
                benchmark_name="smoke",
                rules_path="configs/rules/sichuan_xuezhan_default.yaml",
                baselines=["heuristic", "random"],
                games=4,
                seed=201,
                seed_list=[],
                strict_illegal_action=True,
            )

            with mock.patch(
                "mahjong_ai.training.self_play_grid._load_benchmark_config_with_python_fallback",
                return_value=benchmark_cfg,
            ), mock.patch("mahjong_ai.training.self_play_grid.subprocess.run") as mocked_run:
                result = grid.run_self_play_grid(
                    config_path=str(config_path),
                    rules_path="",
                    benchmark_config_path="configs/eval/smoke.yaml",
                    run_dir=str(run_dir),
                    experiment_prefix="grid",
                    pool_sizes=[2],
                    snapshot_intervals=[1],
                    main_probs=[0.2],
                    num_iterations=1,
                    checkpoint_every=1,
                    eval_every=1,
                    eval_games=1,
                    seed=7,
                    python_bin=sys.executable,
                    quiet_ray_future_warning=False,
                    quiet_new_api_stack_warning=False,
                    quiet_ray_deprecation_warning=False,
                )

            mocked_run.assert_called_once()
            cmd = mocked_run.call_args.args[0]
            self.assertIn("--rules", cmd)
            self.assertIn("configs/rules/sichuan_xuezhan_default.yaml", cmd)
            self.assertIn("--eval-games", cmd)
            self.assertEqual(cmd[cmd.index("--eval-games") + 1], "4")

            cfg_file = run_dir / "grid_configs" / "grid_pool2_snap1_main0p2.json"
            payload = json.loads(cfg_file.read_text(encoding="utf-8"))
            self.assertEqual(payload["evaluation"]["baselines"], ["heuristic", "random"])
            self.assertEqual(payload["evaluation"]["eval_games"], 4)
            self.assertEqual(payload["evaluation"]["seed_list"], [201, 202, 203, 204])
            self.assertTrue(payload["evaluation"]["strict_illegal_action"])

            report_payload = json.loads(Path(result["report_json"]).read_text(encoding="utf-8"))
            self.assertEqual(report_payload["grid"]["eval_games"], 4)
            self.assertEqual(report_payload["benchmark"]["benchmark_name"], "smoke")
            self.assertEqual(report_payload["benchmark"]["effective_seed_list"], [201, 202, 203, 204])


if __name__ == "__main__":
    unittest.main()
