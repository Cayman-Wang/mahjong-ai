#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _iter_rows(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    evaluation = payload.get("evaluation", {})
    benchmark_name = ""
    if isinstance(evaluation, dict):
        benchmark_name = str(evaluation.get("benchmark_name", ""))
    target = payload.get("experiment_name") or payload.get("checkpoint_path") or ""
    metrics = payload.get("metrics", {})
    for baseline, stats in metrics.items():
        yield {
            "file": str(path),
            "mode": str(payload.get("mode", "")),
            "benchmark": benchmark_name,
            "target": str(target),
            "baseline": str(baseline),
            "avg_score": float(stats.get("avg_score", 0.0)),
            "score_std": float(stats.get("score_std", 0.0)),
            "win_rate": float(stats.get("win_rate", 0.0)),
            "avg_steps": float(stats.get("avg_steps", 0.0)),
            "illegal_action_rate": float(stats.get("illegal_action_rate", 0.0)),
        }


def _fmt_row(row: dict[str, object]) -> str:
    return (
        f"{row['file']} | {row['mode']} | {row['benchmark']} | {row['target']} | {row['baseline']} | "
        f"avg_score={row['avg_score']:.3f} "
        f"score_std={row['score_std']:.3f} "
        f"win_rate={row['win_rate']:.3f} "
        f"avg_steps={row['avg_steps']:.3f} "
        f"illegal_action_rate={row['illegal_action_rate']:.3f}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="compare_eval")
    p.add_argument("reports", nargs="+", help="eval report json files")
    args = p.parse_args(argv)

    rows: list[dict[str, object]] = []
    for report in args.reports:
        path = Path(report).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(str(path))
        rows.extend(_iter_rows(path))

    rows.sort(
        key=lambda x: (
            str(x["baseline"]),
            str(x["benchmark"]),
            -float(x["win_rate"]),
            -float(x["avg_score"]),
        )
    )

    print("file | mode | benchmark | target | baseline | metrics")
    for row in rows:
        print(_fmt_row(row))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
