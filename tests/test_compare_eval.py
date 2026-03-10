from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def _load_compare_eval_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "compare_eval.py"
    spec = importlib.util.spec_from_file_location("compare_eval_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load compare_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCompareEvalScript(unittest.TestCase):
    def test_iter_rows_reads_metadata_and_metrics(self) -> None:
        compare_eval = _load_compare_eval_module()
        with tempfile.TemporaryDirectory() as td:
            report = Path(td) / "report.json"
            report.write_text(
                json.dumps(
                    {
                        "mode": "checkpoint_eval",
                        "checkpoint_path": "runs/ppo_selfplay",
                        "evaluation": {"benchmark_name": "standard"},
                        "metrics": {
                            "heuristic": {
                                "avg_score": 1.5,
                                "score_std": 0.2,
                                "win_rate": 0.6,
                                "avg_steps": 120,
                                "illegal_action_rate": 0.0,
                            }
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            rows = list(compare_eval._iter_rows(report))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["mode"], "checkpoint_eval")
        self.assertEqual(rows[0]["benchmark"], "standard")
        self.assertEqual(rows[0]["target"], "runs/ppo_selfplay")

    def test_fmt_row_includes_mode_and_benchmark(self) -> None:
        compare_eval = _load_compare_eval_module()
        line = compare_eval._fmt_row(
            {
                "file": "report.json",
                "mode": "training_eval",
                "benchmark": "smoke",
                "target": "ppo_selfplay",
                "baseline": "heuristic",
                "avg_score": 1.0,
                "score_std": 0.1,
                "win_rate": 0.5,
                "avg_steps": 100.0,
                "illegal_action_rate": 0.0,
            }
        )
        self.assertIn("training_eval", line)
        self.assertIn("smoke", line)
        self.assertIn("ppo_selfplay", line)


if __name__ == "__main__":
    unittest.main()
