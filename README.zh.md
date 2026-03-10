# mahjong-ai

四川麻将（血战到底）规则引擎 + 强化学习环境。

本仓库目标：
- 可复现的血战到底对局模拟（换三张、定缺、碰/杠/胡、一炮多响、血战出局、流局结算）。
- 可配置的计分/番型框架（封顶、杠收分、查叫、花猪等）。
- 自对弈训练与评测（可选：RLlib）。

## 计划与研究文档

- 项目总览：`research/guides/project_overview_zh.md`
- 当前主计划入口：`research/plans/ACTIVE_PLAN.md`
- 主计划：`research/plans/mahjong-ai/master_plan_zh.md`
- 历史交接与审查：`research/handoffs/`、`research/reviews/`

## 快速开始（仅引擎 + 本地模拟）

方式 A：不安装（开发调试）

```bash
PYTHONPATH=src python -m mahjong_ai.cli.main sim --games 10 --agents heuristic,random,random,random --seed 1
```

> 说明：核心引擎尽量保持“无第三方依赖”。RLlib/torch 等训练依赖在 `pyproject.toml` 的 optional extras 里。

## 推荐：创建虚拟环境（venv）

方式 B：推荐（venv + 可编辑安装）

```bash
cd /path/to/mahjong-ai
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev,extras]"
```

安装后运行：

```bash
mahjong-ai sim --games 10 --agents heuristic,random,random,random --seed 1
python -m unittest -v
```

训练依赖（可选，建议单独 Python 3.10/3.11 环境）：

```bash
# 例：conda
conda create -n mahjong-ai-rl python=3.11 -y
conda activate mahjong-ai-rl
cd /path/to/mahjong-ai
pip install -e ".[rl,extras,dev]"
```

## RL 训练与评测

- 推荐环境与 smoke 流程：`docs/training_env.md`
- 正式训练模板：`configs/train/ppo_selfplay_rllib_smoke.yaml`、`configs/train/ppo_selfplay_rllib_standard.yaml`、`configs/train/ppo_selfplay_rllib_long_run.yaml`
- 正式 benchmark 协议：`configs/eval/smoke.yaml`、`configs/eval/standard.yaml`、`configs/eval/long_run.yaml`

最小训练：

```bash
PYTHONPATH=src python -m mahjong_ai.cli.main train-rllib \
  --config configs/train/ppo_selfplay_rllib_smoke.yaml \
  --num-iterations 1 --checkpoint-every 1 --eval-every 1 --eval-games 1
```

日常训练建议使用：

```bash
PYTHONPATH=src python -m mahjong_ai.cli.main train-rllib \
  --config configs/train/ppo_selfplay_rllib_standard.yaml
```

训练默认启用共享主策略 + 对手池自对弈，可在训练模板中调整：
- `self_play.opponent_pool_size`
- `self_play.snapshot_interval`
- `self_play.main_policy_opponent_prob`
- `self_play.seat0_always_main`

如需静音 Ray FutureWarning / new API stack 提示（默认保留），可加：

```bash
--quiet-ray-future-warning --quiet-new-api-stack-warning
```

对手池稳定化参数网格（输出对比报告）：

```bash
PYTHONPATH=src python -m mahjong_ai.cli.main grid-rllib \
  --pool-sizes 2,4 --snapshot-intervals 1,5 --main-probs 0.2,0.4 \
  --num-iterations 1 --checkpoint-every 1 --eval-every 1 --eval-games 2 \
  --python-bin /usr/bin/python3 --run-dir runs/self_play_grid
```

生成结果：
- `runs/self_play_grid/grid_report_<timestamp>.json`
- `runs/self_play_grid/grid_report_<timestamp>.md`

按冻结 benchmark 评测 checkpoint：

```bash
PYTHONPATH=src python -m mahjong_ai.cli.main eval-benchmark \
  --config configs/eval/standard.yaml \
  --checkpoint runs/ppo_selfplay
```

手动评测 checkpoint（自定义基线/样本量）：

```bash
PYTHONPATH=src python -m mahjong_ai.cli.main eval-rllib \
  --checkpoint runs/ppo_selfplay \
  --baselines heuristic,random \
  --games 20 --seed 1 \
  --output runs/ppo_selfplay/eval_manual
```

对比多份评测报告：

```bash
python scripts/compare_eval.py runs/benchmarks/standard/*.json
```

一键 smoke（单测 + sim + train + resume）：

```bash
bash scripts/train_smoke.sh
```

## 目录结构

- `src/mahjong_ai/core/`：对局状态机
- `src/mahjong_ai/scoring/`：胡牌判定、番型、结算
- `src/mahjong_ai/env/`：RL 环境封装（action mask、观测编码）
- `src/mahjong_ai/agents/`：基线智能体
- `src/mahjong_ai/training/`：训练入口（可选）
- `src/mahjong_ai/evaluation/`：评测与指标
- `src/mahjong_ai/integrations/vision_bot/`：未来预留接口，当前未实现
