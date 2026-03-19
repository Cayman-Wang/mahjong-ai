# RL 自对弈训练环境建议

## 推荐版本矩阵

- Python: `3.10.x`（已验证）或 `3.11.x`
- Ray RLlib: `2.53.0`
- Torch: `>=2.2`
- Gymnasium: `>=0.29`
- NumPy: `>=1.26`
- PyYAML: `>=6`

> 说明：核心引擎不依赖上述 RL 组件；仅训练/评测链路需要。

## 安装（示例）

```bash
python -m venv .venv310
source .venv310/bin/activate
python -m pip install -U pip
pip install -e ".[rl,extras,dev]"
```

## 最小回归流程（稳定性 smoke）

```bash
bash scripts/train_smoke.sh
```

该脚本会依次执行：

1. `unittest` 全量回归
2. `sim --games 5 --seed 1` 非 RL 回归
3. RL 最小训练（1 iter + checkpoint + eval）
4. 基于 checkpoint 的续训验证

## 手动训练与评测

训练：

```bash
PYTHONPATH=src python -m mahjong_ai.cli.main train-rllib \
  --config configs/train/ppo_selfplay_rllib_smoke.yaml \
  --num-iterations 1 --checkpoint-every 1 --eval-every 1 --eval-games 1
```

推荐模板：
- smoke：`configs/train/ppo_selfplay_rllib_smoke.yaml`
- standard：`configs/train/ppo_selfplay_rllib_standard.yaml`
- standard-2gpu：`configs/train/ppo_selfplay_rllib_standard_2gpu.yaml`
- long-run：`configs/train/ppo_selfplay_rllib_long_run.yaml`
- long-run-2gpu：`configs/train/ppo_selfplay_rllib_long_run_2gpu.yaml`

双卡建议：

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src python -m mahjong_ai.cli.main train-rllib \
  --config configs/train/ppo_selfplay_rllib_standard_2gpu.yaml
```

- `standard` 模板保留单机轻量设置，采样侧只有本地 env runner
- `standard_2gpu` 模板使用 `num_learners: 2`、`num_gpus_per_learner: 1`，并把采样提升为 `rollout_workers: 6`、`num_envs_per_worker: 2`

双卡长训建议：

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src python -m mahjong_ai.cli.main train-rllib \
  --config configs/train/ppo_selfplay_rllib_long_run_2gpu.yaml
```

- `long_run_2gpu` 使用 `num_iterations: 200`、`train_batch_size: 16384`、`eval_games: 64`
- `long_run_2gpu` 把 `opponent_pool_size` 提到 `8`，并把 `main_policy_opponent_prob` 降到 `0.1`
- `long_run_2gpu` 每次评测额外导出 `3` 局 replay，更适合做策略回放对比

静音可选告警（默认保留告警输出）：

```bash
PYTHONPATH=src python -m mahjong_ai.cli.main train-rllib \
  --config configs/train/ppo_selfplay_rllib_standard.yaml \
  --quiet-ray-future-warning --quiet-new-api-stack-warning
```

关键训练配置：
- `self_play.enabled`：是否启用对手池自对弈。
- `self_play.opponent_pool_size`：历史对手策略槽位数。
- `self_play.snapshot_interval`：每 N 轮把主策略快照复制到对手池。
- `self_play.main_policy_opponent_prob`：座位 1~3 直接采样主策略的概率。
- `evaluation.replay.enabled`：是否在周期评测时额外导出自对弈回放。
- `evaluation.replay.games_per_eval`：每次评测抽样多少局自对弈做回放。
- `evaluation.replay.include_omniscient` / `evaluation.replay.seat_views`：控制全知视角和单座位视角输出。
- `evaluation.replay.output_dir`：回放根目录；空串表示 `runs/<experiment>/replays/iter_<iteration>/`。
- `evaluation.replay.max_steps`：单局回放的最大步数保护。
- `warnings.quiet_ray_future_warning` / `warnings.quiet_new_api_stack_warning`：按需静音 Ray 提示。

默认回放产物：
- `runs/<experiment>/replays/iter_<iteration>/seed_<seed>_omniscient.txt`
- `runs/<experiment>/replays/iter_<iteration>/seed_<seed>_seat0.txt`

参数网格对比（pool_size / snapshot_interval / main_prob）：

```bash
PYTHONPATH=src python -m mahjong_ai.cli.main grid-rllib \
  --pool-sizes 2,4 --snapshot-intervals 1,5 --main-probs 0.2,0.4 \
  --num-iterations 1 --checkpoint-every 1 --eval-every 1 --eval-games 2 \
  --python-bin /usr/bin/python3 --run-dir runs/self_play_grid
```

输出报告：
- `runs/self_play_grid/grid_report_<timestamp>.json`
- `runs/self_play_grid/grid_report_<timestamp>.md`

> 如果你当前 `python` 环境没有 PyYAML，也可以照常运行 `grid-rllib`，它会使用 `--python-bin` 指定解释器加载 YAML 配置。

评测 checkpoint：

```bash
PYTHONPATH=src python -m mahjong_ai.cli.main eval-benchmark \
  --config configs/eval/standard.yaml \
  --checkpoint runs/ppo_selfplay
```

自定义评测：

```bash
PYTHONPATH=src python -m mahjong_ai.cli.main eval-rllib \
  --checkpoint runs/ppo_selfplay \
  --baselines heuristic,random \
  --games 20 --seed 1 \
  --output runs/ppo_selfplay/eval
```

训练完成后单独回放 checkpoint：

```bash
PYTHONPATH=src python -m mahjong_ai.cli.main replay-rllib \
  --checkpoint runs/ppo_selfplay \
  --games 1 --seed 1001 \
  --seat-views 0 \
  --include-omniscient
```

说明：
- `replay-rllib` 当前只支持本地 checkpoint 目录，因为需要读取其中的 `resolved_config.json` 恢复 self-play 策略映射。
- 若不传 `--output`，默认写到 `runs/<experiment>/replays_manual/<timestamp>/`。
- CLI 显式参数优先于 checkpoint 内保存的 `evaluation.replay` / `evaluation.strict_illegal_action` / `warnings.*`。

正式 benchmark 配置：
- `configs/eval/smoke.yaml`
- `configs/eval/standard.yaml`
- `configs/eval/long_run.yaml`
