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
  --config configs/train/ppo_selfplay_rllib.yaml \
  --num-iterations 1 --checkpoint-every 1 --eval-every 1 --eval-games 1
```

静音可选告警（默认保留告警输出）：

```bash
PYTHONPATH=src python -m mahjong_ai.cli.main train-rllib \
  --config configs/train/ppo_selfplay_rllib.yaml \
  --quiet-ray-future-warning --quiet-new-api-stack-warning
```

关键训练配置：
- `self_play.enabled`：是否启用对手池自对弈。
- `self_play.opponent_pool_size`：历史对手策略槽位数。
- `self_play.snapshot_interval`：每 N 轮把主策略快照复制到对手池。
- `self_play.main_policy_opponent_prob`：座位 1~3 直接采样主策略的概率。
- `warnings.quiet_ray_future_warning` / `warnings.quiet_new_api_stack_warning`：按需静音 Ray 提示。

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
PYTHONPATH=src python -m mahjong_ai.cli.main eval-rllib \
  --checkpoint runs/ppo_selfplay \
  --baselines heuristic,random \
  --games 20 --seed 1 \
  --output runs/ppo_selfplay/eval
```
