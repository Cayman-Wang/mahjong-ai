from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Hashable

from mahjong_ai.evaluation.benchmark_config import EvalBenchmarkConfig, load_eval_benchmark_config
from mahjong_ai.training.rllib_runner import load_train_config


@dataclass(frozen=True, slots=True)
class GridCombo:
    pool_size: int
    snapshot_interval: int
    main_prob: float


def _parse_int_csv(raw: str) -> list[int]:
    out: list[int] = []
    for chunk in raw.split(","):
        value = chunk.strip()
        if value:
            out.append(int(value))
    if not out:
        raise ValueError("expected at least one integer value")
    return out


def _parse_float_csv(raw: str) -> list[float]:
    out: list[float] = []
    for chunk in raw.split(","):
        value = chunk.strip()
        if value:
            out.append(float(value))
    if not out:
        raise ValueError("expected at least one float value")
    return out


def _dedupe_keep_order(values: list[Hashable]) -> list[Hashable]:
    seen: set[Hashable] = set()
    out: list[Hashable] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def build_grid_combinations(
    *,
    pool_sizes: list[int],
    snapshot_intervals: list[int],
    main_probs: list[float],
) -> list[GridCombo]:
    combos: list[GridCombo] = []
    for pool_size, snapshot_interval, main_prob in itertools.product(pool_sizes, snapshot_intervals, main_probs):
        if int(pool_size) < 0:
            raise ValueError("pool_size must be >= 0")
        if int(snapshot_interval) <= 0:
            raise ValueError("snapshot_interval must be positive")
        if not 0.0 <= float(main_prob) <= 1.0:
            raise ValueError("main_prob must be in [0, 1]")
        combos.append(
            GridCombo(
                pool_size=int(pool_size),
                snapshot_interval=int(snapshot_interval),
                main_prob=float(main_prob),
            )
        )
    return [c for c in _dedupe_keep_order(combos)]


def _float_token(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def build_experiment_name(*, prefix: str, combo: GridCombo) -> str:
    return f"{prefix}_pool{combo.pool_size}_snap{combo.snapshot_interval}_main{_float_token(combo.main_prob)}"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _find_latest_eval_report(exp_dir: Path) -> Path | None:
    eval_dir = exp_dir / "eval"
    if not eval_dir.exists():
        return None
    reports = sorted(eval_dir.glob("iter_*.json"))
    if not reports:
        return None
    return reports[-1]


def _primary_baseline(metrics: dict[str, dict[str, float]]) -> str:
    if "heuristic" in metrics:
        return "heuristic"
    if "random" in metrics:
        return "random"
    if not metrics:
        return ""
    return sorted(metrics.keys())[0]


def _baseline_value(metrics: dict[str, dict[str, float]], baseline: str, key: str) -> float | None:
    stats = metrics.get(baseline)
    if not stats:
        return None
    if key not in stats:
        return None
    return float(stats[key])


def _score_tuple(row: dict[str, Any]) -> tuple[float, float]:
    win_rate = row.get("primary_win_rate")
    avg_score = row.get("primary_avg_score")
    return (
        float(win_rate) if isinstance(win_rate, (int, float)) else float("-inf"),
        float(avg_score) if isinstance(avg_score, (int, float)) else float("-inf"),
    )


def _render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Self-play Grid Report",
        "",
        "| rank | experiment | pool_size | snapshot_interval | main_prob | primary_baseline | win_rate | avg_score | report |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]

    for rank, row in enumerate(rows, start=1):
        win_rate = row.get("primary_win_rate")
        avg_score = row.get("primary_avg_score")
        main_prob = row.get("main_prob")

        experiment_name = row.get("experiment_name") or "-"
        pool_size = row.get("pool_size", "-")
        snapshot_interval = row.get("snapshot_interval", "-")
        main_prob_txt = "-" if not isinstance(main_prob, (int, float)) else f"{float(main_prob):.3f}"
        baseline = row.get("primary_baseline") or "-"
        win_rate_txt = "-" if win_rate is None else f"{float(win_rate):.4f}"
        avg_score_txt = "-" if avg_score is None else f"{float(avg_score):.4f}"
        report_path = row.get("report_path") or "-"

        lines.append(
            f"| {rank} | {experiment_name} | {pool_size} | {snapshot_interval} | {main_prob_txt} | "
            f"{baseline} | {win_rate_txt} | {avg_score_txt} | {report_path} |"
        )
    lines.append("")
    return "\n".join(lines)


def _load_config_with_python_fallback(
    *,
    config_path: str,
    python_bin: str,
    cwd: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    try:
        cfg = load_train_config(config_path)
    except RuntimeError as e:
        # Allow invoking grid from a lightweight interpreter (without PyYAML)
        # while running training in another interpreter via --python-bin.
        suffix = Path(config_path).suffix.lower()
        if "PyYAML is required" not in str(e) or suffix not in {".yaml", ".yml"}:
            raise

        code = """
import json
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
text = p.read_text(encoding="utf-8")
if p.suffix.lower() in {".yaml", ".yml"}:
    import yaml

    data = yaml.safe_load(text)
elif p.suffix.lower() == ".json":
    data = json.loads(text)
else:
    raise ValueError(f"unsupported config file extension: {p.suffix}")

print(json.dumps(data, ensure_ascii=False))
"""
        try:
            proc = subprocess.run(
                [python_bin, "-c", code, str(Path(config_path).expanduser().resolve())],
                check=True,
                cwd=str(cwd),
                env=env,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as sub_err:
            stderr = (sub_err.stderr or "").strip()
            raise RuntimeError(
                "failed to parse training config via --python-bin; ensure that interpreter has PyYAML installed"
                + (f": {stderr}" if stderr else "")
            ) from e

        cfg = json.loads(proc.stdout)

    if not isinstance(cfg, dict):
        raise ValueError("training config must be a mapping")
    return cfg


def _load_benchmark_config_with_python_fallback(
    *,
    benchmark_config_path: str,
    python_bin: str,
    cwd: Path,
    env: dict[str, str],
) -> EvalBenchmarkConfig:
    try:
        cfg = load_eval_benchmark_config(benchmark_config_path)
    except RuntimeError as e:
        suffix = Path(benchmark_config_path).suffix.lower()
        if "PyYAML is required" not in str(e) or suffix not in {".yaml", ".yml"}:
            raise

        code = """
import json
import sys
from dataclasses import asdict

from mahjong_ai.evaluation.benchmark_config import load_eval_benchmark_config

cfg = load_eval_benchmark_config(sys.argv[1])
print(json.dumps(asdict(cfg), ensure_ascii=False))
"""
        try:
            proc = subprocess.run(
                [python_bin, "-c", code, str(Path(benchmark_config_path).expanduser().resolve())],
                check=True,
                cwd=str(cwd),
                env=env,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as sub_err:
            stderr = (sub_err.stderr or "").strip()
            raise RuntimeError(
                "failed to parse benchmark config via --python-bin; ensure that interpreter has PyYAML installed"
                + (f": {stderr}" if stderr else "")
            ) from e

        cfg = EvalBenchmarkConfig(**json.loads(proc.stdout))
        cfg.validate()

    return cfg


def _resolve_benchmark_seed_list(benchmark_cfg: EvalBenchmarkConfig) -> list[int]:
    if benchmark_cfg.seed_list:
        return [int(seed) for seed in benchmark_cfg.seed_list]
    return [int(benchmark_cfg.seed + offset) for offset in range(int(benchmark_cfg.games))]


def _apply_benchmark_to_grid_config(
    *,
    train_cfg: dict[str, Any],
    rules_path: str,
    benchmark_cfg: EvalBenchmarkConfig,
    benchmark_config_path: str,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    cfg = copy.deepcopy(train_cfg)
    eval_cfg = cfg.setdefault("evaluation", {})
    eval_cfg["baselines"] = [str(name) for name in benchmark_cfg.baselines]
    eval_cfg["eval_games"] = int(benchmark_cfg.games)
    eval_cfg["seed_list"] = _resolve_benchmark_seed_list(benchmark_cfg)
    eval_cfg["strict_illegal_action"] = bool(benchmark_cfg.strict_illegal_action)

    effective_rules_path = str(rules_path or benchmark_cfg.rules_path or "")
    metadata = asdict(benchmark_cfg)
    metadata["benchmark_config_path"] = str(Path(benchmark_config_path).expanduser().resolve())
    metadata["effective_seed_list"] = list(eval_cfg["seed_list"])
    metadata["effective_rules_path"] = effective_rules_path
    return cfg, effective_rules_path, metadata


def run_self_play_grid(
    *,
    config_path: str,
    rules_path: str,
    run_dir: str,
    experiment_prefix: str,
    pool_sizes: list[int],
    snapshot_intervals: list[int],
    main_probs: list[float],
    num_iterations: int,
    checkpoint_every: int,
    eval_every: int,
    eval_games: int,
    seed: int,
    python_bin: str,
    quiet_ray_future_warning: bool,
    quiet_new_api_stack_warning: bool,
    quiet_ray_deprecation_warning: bool,
    benchmark_config_path: str = "",
) -> dict[str, Any]:
    combos = build_grid_combinations(
        pool_sizes=pool_sizes,
        snapshot_intervals=snapshot_intervals,
        main_probs=main_probs,
    )
    if not combos:
        raise ValueError("grid combinations are empty")

    root = _repo_root()

    env = os.environ.copy()
    src_path = str(root / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_path if not existing else src_path + os.pathsep + existing

    base_cfg = _load_config_with_python_fallback(
        config_path=config_path,
        python_bin=python_bin,
        cwd=root,
        env=env,
    )
    effective_rules_path = str(rules_path or "")
    effective_eval_games = int(eval_games)
    benchmark_metadata: dict[str, Any] = {}
    if benchmark_config_path:
        benchmark_cfg = _load_benchmark_config_with_python_fallback(
            benchmark_config_path=benchmark_config_path,
            python_bin=python_bin,
            cwd=root,
            env=env,
        )
        base_cfg, effective_rules_path, benchmark_metadata = _apply_benchmark_to_grid_config(
            train_cfg=base_cfg,
            rules_path=effective_rules_path,
            benchmark_cfg=benchmark_cfg,
            benchmark_config_path=benchmark_config_path,
        )
        effective_eval_games = int(base_cfg["evaluation"]["eval_games"])

    run_path = Path(run_dir).expanduser().resolve()
    configs_path = run_path / "grid_configs"
    run_path.mkdir(parents=True, exist_ok=True)
    configs_path.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for idx, combo in enumerate(combos, start=1):
        exp_name = build_experiment_name(prefix=experiment_prefix, combo=combo)
        cfg = copy.deepcopy(base_cfg)
        cfg["experiment_name"] = exp_name
        cfg.setdefault("self_play", {})
        cfg["self_play"]["enabled"] = combo.pool_size > 0
        cfg["self_play"]["opponent_pool_size"] = combo.pool_size
        cfg["self_play"]["snapshot_interval"] = combo.snapshot_interval
        cfg["self_play"]["main_policy_opponent_prob"] = combo.main_prob
        cfg["self_play"]["seat0_always_main"] = True

        cfg_file = configs_path / f"{exp_name}.json"
        cfg_file.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

        cmd = [
            python_bin,
            "-m",
            "mahjong_ai.cli.main",
            "train-rllib",
            "--config",
            str(cfg_file),
            "--run-dir",
            str(run_path),
            "--num-iterations",
            str(num_iterations),
            "--checkpoint-every",
            str(checkpoint_every),
            "--eval-every",
            str(eval_every),
            "--eval-games",
            str(effective_eval_games),
            "--seed",
            str(seed),
        ]
        if effective_rules_path:
            cmd.extend(["--rules", effective_rules_path])
        if quiet_ray_future_warning:
            cmd.append("--quiet-ray-future-warning")
        if quiet_new_api_stack_warning:
            cmd.append("--quiet-new-api-stack-warning")
        if quiet_ray_deprecation_warning:
            cmd.append("--quiet-ray-deprecation-warning")

        print(
            f"[grid] ({idx}/{len(combos)}) pool={combo.pool_size} snap={combo.snapshot_interval} "
            f"main_prob={combo.main_prob:.3f}"
        )
        subprocess.run(cmd, check=True, cwd=str(root), env=env)

        exp_dir = run_path / exp_name
        report_path = _find_latest_eval_report(exp_dir)
        metrics: dict[str, dict[str, float]] = {}
        if report_path is not None:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            raw_metrics = payload.get("metrics", {})
            if isinstance(raw_metrics, dict):
                metrics = {
                    str(name): {str(k): float(v) for k, v in stats.items()}
                    for name, stats in raw_metrics.items()
                    if isinstance(stats, dict)
                }

        primary = _primary_baseline(metrics)
        row = {
            "experiment_name": exp_name,
            "pool_size": combo.pool_size,
            "snapshot_interval": combo.snapshot_interval,
            "main_prob": combo.main_prob,
            "primary_baseline": primary,
            "primary_win_rate": _baseline_value(metrics, primary, "win_rate") if primary else None,
            "primary_avg_score": _baseline_value(metrics, primary, "avg_score") if primary else None,
            "metrics": metrics,
            "report_path": str(report_path) if report_path is not None else "",
        }
        rows.append(row)

    rows.sort(key=_score_tuple, reverse=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_json = run_path / f"grid_report_{timestamp}.json"
    report_md = run_path / f"grid_report_{timestamp}.md"

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(Path(config_path).expanduser().resolve()),
        "rules_path": str(Path(effective_rules_path).expanduser().resolve()) if effective_rules_path else "",
        "run_dir": str(run_path),
        "experiment_prefix": experiment_prefix,
        "benchmark": benchmark_metadata,
        "grid": {
            "pool_sizes": pool_sizes,
            "snapshot_intervals": snapshot_intervals,
            "main_probs": main_probs,
            "num_iterations": num_iterations,
            "checkpoint_every": checkpoint_every,
            "eval_every": eval_every,
            "eval_games": effective_eval_games,
            "seed": seed,
        },
        "rows": rows,
        "sort": "primary_win_rate desc, primary_avg_score desc",
    }

    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md.write_text(_render_markdown(rows), encoding="utf-8")

    print(f"[grid] report_json={report_json}")
    print(f"[grid] report_md={report_md}")

    return {
        "report_json": str(report_json),
        "report_md": str(report_md),
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m mahjong_ai.training.self_play_grid")
    parser.add_argument("--config", type=str, default="configs/train/ppo_selfplay_rllib.yaml")
    parser.add_argument("--benchmark-config", type=str, default="")
    parser.add_argument("--rules", type=str, default="")
    parser.add_argument("--run-dir", type=str, default="runs/self_play_grid")
    parser.add_argument("--experiment-prefix", type=str, default="grid_sp")
    parser.add_argument("--pool-sizes", type=str, default="2,4")
    parser.add_argument("--snapshot-intervals", type=str, default="1,5")
    parser.add_argument("--main-probs", type=str, default="0.1,0.2,0.4")
    parser.add_argument("--num-iterations", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--eval-games", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument(
        "--quiet-ray-future-warning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="mute or keep Ray accelerator override FutureWarning while running the grid",
    )
    parser.add_argument(
        "--quiet-new-api-stack-warning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="mute or keep RLlib new API stack warning while running the grid",
    )
    parser.add_argument(
        "--quiet-ray-deprecation-warning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="mute or keep Ray deprecation warnings while running the grid",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_self_play_grid(
        config_path=args.config,
        rules_path=args.rules,
        run_dir=args.run_dir,
        experiment_prefix=args.experiment_prefix,
        pool_sizes=_parse_int_csv(args.pool_sizes),
        snapshot_intervals=_parse_int_csv(args.snapshot_intervals),
        main_probs=_parse_float_csv(args.main_probs),
        num_iterations=int(args.num_iterations),
        checkpoint_every=int(args.checkpoint_every),
        eval_every=int(args.eval_every),
        eval_games=int(args.eval_games),
        seed=int(args.seed),
        python_bin=str(args.python_bin),
        quiet_ray_future_warning=bool(args.quiet_ray_future_warning),
        quiet_new_api_stack_warning=bool(args.quiet_new_api_stack_warning),
        quiet_ray_deprecation_warning=bool(args.quiet_ray_deprecation_warning),
        benchmark_config_path=str(args.benchmark_config),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
