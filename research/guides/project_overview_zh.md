# mahjong-ai 项目总览

## 1. 项目定位

`mahjong-ai` 是一个面向四川麻将（血战到底）的研究型代码库，目标是把以下三层能力收口到同一仓库中：

- 规则正确、可复现的本地对局引擎
- 可用于强化学习的数值化观测与动作约束环境
- 可复现、可比较的训练与评测工作流

当前主线不是外部平台接入，而是把项目打造成稳定的研究基座。

## 2. 当前状态

### 已落地
- 对局引擎：换三张、定缺、碰/杠/胡、一炮多响、血战出局、流局结算等主流程已实现。
- 规则与计分：庄闲倍数、点炮配置、查叫、花猪、番型等关键结算项已配置化。
- 训练链路：RLlib 训练入口、action mask RLModule、checkpoint 保存、续训、周期评测、self-play 对手池、参数网格对比已实现。
- 回归体系：核心引擎、观测编码、训练 runner、CLI、grid 报告等均有测试覆盖。
- 文档治理：`research/` 已成为计划、审查、交接、复盘的主入口。

### 正在推进
- benchmark 协议固定化
- smoke / standard / long-run 模板化
- 训练复现与实验产物规范化
- 策略质量的系统性筛选

### 明确后置
- `vision_bot` 整条链路仍是占位模块，近期不进入主线。
- CI 分层回归、长期实验面板、可视化报表尚未完全落地。

## 3. 模块地图

- `src/mahjong_ai/core/`
  - 牌、状态、事件、状态机，是整个项目的规则真相来源。
- `src/mahjong_ai/scoring/`
  - 胡牌判定、番型、结算逻辑。
- `src/mahjong_ai/env/`
  - 观测编码、action mask、多智能体环境封装。
- `src/mahjong_ai/training/`
  - RLlib 训练、评测、checkpoint、grid search。
- `src/mahjong_ai/agents/`
  - `random` / `heuristic` 基线智能体。
- `src/mahjong_ai/evaluation/`
  - 基础评测与 benchmark 配置加载。
- `src/mahjong_ai/integrations/vision_bot/`
  - 未来预留接口，当前不应视为已实现能力。

## 4. 当前正式入口

- 项目使用说明：`README.zh.md`
- 当前活动计划：`research/plans/ACTIVE_PLAN.md`
- 主计划：`research/plans/mahjong-ai/master_plan_zh.md`
- 当前状态交接：`research/handoffs/2026-03-10_project_status_refresh.md`
- 训练环境建议：`docs/training_env.md`

## 5. 正式执行模板

### 训练模板
- `configs/train/ppo_selfplay_rllib_smoke.yaml`
- `configs/train/ppo_selfplay_rllib_standard.yaml`
- `configs/train/ppo_selfplay_rllib_long_run.yaml`

### Benchmark 协议
- `configs/eval/smoke.yaml`
- `configs/eval/standard.yaml`
- `configs/eval/long_run.yaml`

### 常用命令

纯引擎回归：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m mahjong_ai.cli.main sim --games 5 --seed 1
```

训练 smoke：

```bash
bash scripts/train_smoke.sh
```

按冻结 benchmark 评测 checkpoint：

```bash
PYTHONPATH=src python -m mahjong_ai.cli.main eval-benchmark \
  --config configs/eval/standard.yaml \
  --checkpoint runs/ppo_selfplay
```

对比多个评测报告：

```bash
python scripts/compare_eval.py runs/benchmarks/standard/*.json
```

## 6. 当前主要风险

- RLlib / Ray API 演进快，训练相关依赖需要持续锁版本验证。
- 训练“能跑”不代表策略“有效”；必须在冻结 benchmark 下比较 checkpoint。
- 历史 handoff / review 文档中有一部分结论已经过时，新会话必须优先读当前总览和主计划。

## 7. 近期默认原则

- 优先研究基座，不优先做外部平台集成。
- 优先固定 benchmark，再做策略优化。
- 训练产物放在 `runs/` 等运行目录，不放进 `research/`。
