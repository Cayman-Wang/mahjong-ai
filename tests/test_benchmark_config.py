from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mahjong_ai.evaluation.benchmark_config import load_eval_benchmark_config


class TestBenchmarkConfig(unittest.TestCase):
    def test_load_config_resolves_seed_file_relative_to_config(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            seeds_dir = root / "seeds"
            seeds_dir.mkdir()
            (seeds_dir / "standard.json").write_text(
                json.dumps({"seeds": [101, 102, 103, 104]}, ensure_ascii=False),
                encoding="utf-8",
            )
            config_path = root / "standard.json"
            config_path.write_text(
                json.dumps(
                    {
                        "benchmark_name": "standard",
                        "baselines": ["heuristic", "random"],
                        "games": 4,
                        "seed_file": "seeds/standard.json",
                        "strict_illegal_action": True,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            cfg = load_eval_benchmark_config(config_path)

        self.assertEqual(cfg.benchmark_name, "standard")
        self.assertEqual(cfg.seed_list, [101, 102, 103, 104])
        self.assertEqual(cfg.games, 4)

    def test_seed_file_and_seed_list_cannot_both_be_set(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "seeds.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
            config_path = root / "bad.json"
            config_path.write_text(
                json.dumps(
                    {
                        "benchmark_name": "bad",
                        "games": 3,
                        "seed_list": [1, 2, 3],
                        "seed_file": "seeds.json",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "cannot both be set"):
                load_eval_benchmark_config(config_path)

    def test_seed_list_length_must_match_games(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "bad.json"
            config_path.write_text(
                json.dumps(
                    {
                        "benchmark_name": "bad",
                        "games": 2,
                        "seed_list": [1],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "seed_list length"):
                load_eval_benchmark_config(config_path)


if __name__ == "__main__":
    unittest.main()
