from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
