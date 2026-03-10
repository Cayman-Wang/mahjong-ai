from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

from mahjong_ai.agents.base import AgentContext
from mahjong_ai.agents.registry import make_agent
from mahjong_ai.core.engine import GameEngine
from mahjong_ai.core.tiles import pretty_tile
from mahjong_ai.rules.loader import load_rules
from mahjong_ai.rules.schema import RulesConfig


@dataclass(slots=True)
class SimResult:
    scores: list[int]
    steps: int


def _parse_agents(s: str) -> list[str]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return ["random", "random", "random", "random"]
    if len(parts) == 1:
        return [parts[0]] * 4
    if len(parts) == 2:
        return [parts[0], parts[1], parts[0], parts[1]]
    if len(parts) == 3:
        return [parts[0], parts[1], parts[2], parts[0]]
    return parts[:4]


def _parse_csv_list(s: str) -> list[str]:
    return [p.strip() for p in s.split(",") if p.strip()]


def _parse_seed_list(s: str) -> list[int]:
    if not s.strip():
        return []
    out: list[int] = []
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    return out


def _parse_int_csv(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        p = part.strip()
        if p:
            out.append(int(p))
    if not out:
        raise ValueError("expected at least one integer value")
    return out


def _parse_float_csv(s: str) -> list[float]:
    out: list[float] = []
    for part in s.split(","):
        p = part.strip()
        if p:
            out.append(float(p))
    if not out:
        raise ValueError("expected at least one float value")
    return out


def run_one_game(engine: GameEngine, *, seed: int, agent_names: list[str], verbose: bool, max_steps: int) -> SimResult:
    agents = [make_agent(agent_names[i], seed=seed + i * 9973) for i in range(4)]
    ctxs = [AgentContext(seat=i) for i in range(4)]

    state = engine.reset(seed=seed)
    steps = 0

    if verbose:
        print(f"seed={seed} dealer={state.dealer}")

    while True:
        if state.phase.value == "round_end":
            break
        if steps >= max_steps:
            raise RuntimeError(f"max_steps exceeded ({max_steps}); possible engine bug")

        required = engine.required_players(state)
        actions: dict[int, int] = {}
        for pid in required:
            mask = engine.legal_action_mask(state, pid)
            a = agents[pid].act(ctxs[pid], state, mask)
            if a < 0 or a >= len(mask) or mask[a] != 1:
                raise RuntimeError(f"agent {agents[pid].name} chose illegal action {a} in phase {state.phase}")
            actions[pid] = a

        res = engine.step(state, actions)
        steps += 1

        if verbose:
            for ev in res.events:
                if ev.tile is None:
                    print(f"  {ev.type}: actor={ev.actor} meta={ev.meta}")
                else:
                    print(f"  {ev.type}: actor={ev.actor} tile={pretty_tile(ev.tile)} meta={ev.meta}")

        if res.done:
            break

    if verbose:
        print(f"final scores={state.scores} steps={steps}")

    return SimResult(scores=state.scores, steps=steps)


def cmd_sim(args: argparse.Namespace) -> int:
    rules: RulesConfig
    if args.rules:
        rules = load_rules(args.rules)
    else:
        rules = RulesConfig()

    engine = GameEngine(rules=rules, enable_events=args.verbose)
    agent_names = _parse_agents(args.agents)

    total = [0, 0, 0, 0]
    total_steps = 0
    for i in range(args.games):
        r = run_one_game(
            engine,
            seed=args.seed + i,
            agent_names=agent_names,
            verbose=args.verbose,
            max_steps=args.max_steps,
        )
        for j in range(4):
            total[j] += r.scores[j]
        total_steps += r.steps

    avg = [x / args.games for x in total]
    print(f"games={args.games} avg_scores={avg} total_steps={total_steps}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    rules = RulesConfig()
    engine = GameEngine(rules=rules, enable_events=False)
    agent_names = ["random", "random", "random", "random"]

    t0 = time.perf_counter()
    steps = 0
    for i in range(args.games):
        r = run_one_game(engine, seed=args.seed + i, agent_names=agent_names, verbose=False, max_steps=args.max_steps)
        steps += r.steps
    dt = time.perf_counter() - t0

    print(f"games={args.games} steps={steps} seconds={dt:.3f} games_per_s={args.games/dt:.2f} steps_per_s={steps/dt:.2f}")
    return 0


def cmd_train_rllib(args: argparse.Namespace) -> int:
    from mahjong_ai.training.rllib_runner import run_training_entry

    run_training_entry(
        config_path=args.config,
        rules_path=args.rules,
        seed=args.seed,
        num_iterations=args.num_iterations,
        checkpoint_every=args.checkpoint_every,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        run_dir=args.run_dir,
        resume_from=args.resume_from,
        quiet_ray_future_warning=args.quiet_ray_future_warning,
        quiet_new_api_stack_warning=args.quiet_new_api_stack_warning,
        quiet_ray_deprecation_warning=args.quiet_ray_deprecation_warning,
        strict_illegal_action=args.strict_illegal_action,
    )
    return 0


def cmd_eval_rllib(args: argparse.Namespace) -> int:
    from mahjong_ai.training.rllib_runner import run_evaluation_entry

    baselines = _parse_csv_list(args.baselines)
    if not baselines:
        raise ValueError("--baselines must provide at least one baseline agent")

    seed_list = _parse_seed_list(args.seed_list)
    run_evaluation_entry(
        checkpoint_path=args.checkpoint,
        rules_path=args.rules,
        baselines=baselines,
        seed=args.seed,
        games=args.games,
        seed_list=seed_list,
        output_path=args.output,
        quiet_ray_future_warning=bool(args.quiet_ray_future_warning),
        quiet_new_api_stack_warning=bool(args.quiet_new_api_stack_warning),
        quiet_ray_deprecation_warning=bool(args.quiet_ray_deprecation_warning),
        strict_illegal_action=bool(args.strict_illegal_action),
    )
    return 0


def cmd_grid_rllib(args: argparse.Namespace) -> int:
    from mahjong_ai.training.self_play_grid import run_self_play_grid

    run_self_play_grid(
        config_path=args.config,
        rules_path=args.rules,
        run_dir=args.run_dir,
        experiment_prefix=args.experiment_prefix,
        pool_sizes=_parse_int_csv(args.pool_sizes),
        snapshot_intervals=_parse_int_csv(args.snapshot_intervals),
        main_probs=_parse_float_csv(args.main_probs),
        num_iterations=args.num_iterations,
        checkpoint_every=args.checkpoint_every,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        seed=args.seed,
        python_bin=args.python_bin,
        quiet_ray_future_warning=bool(args.quiet_ray_future_warning),
        quiet_new_api_stack_warning=bool(args.quiet_new_api_stack_warning),
        quiet_ray_deprecation_warning=bool(args.quiet_ray_deprecation_warning),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mahjong-ai")
    sub = p.add_subparsers(dest="cmd", required=True)

    sim = sub.add_parser("sim", help="simulate games with baseline agents")
    sim.add_argument("--rules", type=str, default="", help="path to rules yaml/json (requires pyyaml for yaml)")
    sim.add_argument("--seed", type=int, default=1)
    sim.add_argument("--games", type=int, default=1)
    sim.add_argument("--agents", type=str, default="heuristic,random,random,random", help="comma-separated agent names for 4 seats")
    sim.add_argument("--verbose", action="store_true")
    sim.add_argument("--max-steps", type=int, default=2000)
    sim.set_defaults(func=cmd_sim)

    bench = sub.add_parser("bench", help="micro-benchmark random self-play")
    bench.add_argument("--seed", type=int, default=1)
    bench.add_argument("--games", type=int, default=100)
    bench.add_argument("--max-steps", type=int, default=2000)
    bench.set_defaults(func=cmd_bench)

    train = sub.add_parser("train-rllib", help="run RLlib PPO shared-policy self-play training")
    train.add_argument("--config", type=str, default="configs/train/ppo_selfplay_rllib.yaml")
    train.add_argument("--rules", type=str, default="", help="path to rules yaml/json (optional)")
    train.add_argument("--seed", type=int, default=None)
    train.add_argument("--num-iterations", type=int, default=None)
    train.add_argument("--checkpoint-every", type=int, default=None)
    train.add_argument("--eval-every", type=int, default=None)
    train.add_argument("--eval-games", type=int, default=None)
    train.add_argument("--run-dir", type=str, default=None)
    train.add_argument("--resume-from", type=str, default=None)
    train.add_argument(
        "--quiet-ray-future-warning",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="mute or keep Ray accelerator override FutureWarning (defaults to config)",
    )
    train.add_argument(
        "--quiet-new-api-stack-warning",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="mute or keep RLlib new API stack startup warning (defaults to config)",
    )
    train.add_argument(
        "--quiet-ray-deprecation-warning",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="mute or keep Ray deprecation warnings (defaults to config)",
    )
    train.add_argument(
        "--strict-illegal-action",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="raise immediately on illegal eval actions during training eval (defaults to config)",
    )
    train.set_defaults(func=cmd_train_rllib)

    eval_cmd = sub.add_parser("eval-rllib", help="evaluate a RLlib checkpoint against baseline agents")
    eval_cmd.add_argument("--checkpoint", type=str, required=True, help="checkpoint path/URI to evaluate")
    eval_cmd.add_argument("--rules", type=str, default="", help="path to rules yaml/json (optional)")
    eval_cmd.add_argument("--baselines", type=str, default="heuristic,random", help="comma-separated baseline agents")
    eval_cmd.add_argument("--games", type=int, default=20)
    eval_cmd.add_argument("--seed", type=int, default=1)
    eval_cmd.add_argument("--seed-list", type=str, default="", help="comma-separated fixed seeds (length must equal --games)")
    eval_cmd.add_argument("--output", type=str, default=None, help="optional output json file (or directory)")
    eval_cmd.add_argument(
        "--quiet-ray-future-warning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="mute or keep Ray accelerator override FutureWarning",
    )
    eval_cmd.add_argument(
        "--quiet-new-api-stack-warning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="mute or keep RLlib new API stack startup warning",
    )
    eval_cmd.add_argument(
        "--quiet-ray-deprecation-warning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="mute or keep Ray deprecation warnings",
    )
    eval_cmd.add_argument(
        "--strict-illegal-action",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="raise immediately on illegal actions during evaluation",
    )
    eval_cmd.set_defaults(func=cmd_eval_rllib)

    grid_cmd = sub.add_parser("grid-rllib", help="run self-play stability grid and output ranked report")
    grid_cmd.add_argument("--config", type=str, default="configs/train/ppo_selfplay_rllib.yaml")
    grid_cmd.add_argument("--rules", type=str, default="")
    grid_cmd.add_argument("--run-dir", type=str, default="runs/self_play_grid")
    grid_cmd.add_argument("--experiment-prefix", type=str, default="grid_sp")
    grid_cmd.add_argument("--pool-sizes", type=str, default="2,4")
    grid_cmd.add_argument("--snapshot-intervals", type=str, default="1,5")
    grid_cmd.add_argument("--main-probs", type=str, default="0.1,0.2,0.4")
    grid_cmd.add_argument("--num-iterations", type=int, default=1)
    grid_cmd.add_argument("--checkpoint-every", type=int, default=1)
    grid_cmd.add_argument("--eval-every", type=int, default=1)
    grid_cmd.add_argument("--eval-games", type=int, default=2)
    grid_cmd.add_argument("--seed", type=int, default=1)
    grid_cmd.add_argument("--python-bin", type=str, default=sys.executable)
    grid_cmd.add_argument(
        "--quiet-ray-future-warning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="mute or keep Ray accelerator override FutureWarning",
    )
    grid_cmd.add_argument(
        "--quiet-new-api-stack-warning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="mute or keep RLlib new API stack startup warning",
    )
    grid_cmd.add_argument(
        "--quiet-ray-deprecation-warning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="mute or keep Ray deprecation warnings",
    )
    grid_cmd.set_defaults(func=cmd_grid_rllib)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
