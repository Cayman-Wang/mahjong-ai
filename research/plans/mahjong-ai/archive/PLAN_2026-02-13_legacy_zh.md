# mahjong-ai 项目计划（PLAN）

> 更新时间：2026-02-13
> 当前阶段：训练 MVP（P0）完成 + P1 第二阶段完成（近期动作历史 + 对手池参数网格）

## 0. 目标与边界

### 目标
- 在本地稳定复现四川麻将（血战到底）完整流程，并保证关键不变量（牌守恒、结算零和、合法动作约束）。
- 提供可配置规则框架，支持不同平台细则差异。
- 提供可直接启动的自对弈训练入口（RLlib），并可周期评测对 heuristic/random。

### 明确不做/不建议
- 不做任何内存注入、协议破解、绕过反作弊。
- 未获授权场景不建议自动化代打。

## 1. 当前完成情况（2026-02-13）

### 1.1 对局引擎与规则（已完成）
- 血战流程完整：换三张、定缺、摸打、响应（碰/杠/胡/过）、抢杠胡、一炮多响、流局结算。
- 关键规则问题已修复并落地测试：
  - 一炮多响点炮赔付逻辑
  - 庄闲倍数（配置项 + 结算应用）
  - 抢杠胡阶段动作掩码校验与牌信息一致性
- RLlib reset 默认随机种子已修复，避免固定种子 episode。

### 1.2 训练 MVP（P0 已完成）
- 数值化观测编码已落地：`src/mahjong_ai/env/obs_vector_encoder.py`（固定维度、纯数值、无字符串 phase）。
- RLlib 多智能体环境 spaces 已补齐：`src/mahjong_ai/env/rllib_multiagent_env.py`。
- 动作掩码 RLModule 已实现：`src/mahjong_ai/training/rllib_action_mask_rl_module.py`（非法动作 logits 置极小值）。
- RLlib 训练入口已实现：`src/mahjong_ai/training/rllib_runner.py`
  - 共享策略 self-play
  - 对手池快照轮转
  - checkpoint 到 `runs/`
  - 定期评测 vs `heuristic` / `random`
- CLI 已接入：`train-rllib` / `eval-rllib`。
- 告警静音可配置（默认保留）：
  - `warnings.quiet_ray_future_warning`
  - `warnings.quiet_new_api_stack_warning`
  - CLI 覆盖参数同名。

### 1.3 P1 第二阶段（已完成）
- 新增近期动作历史状态：
  - `src/mahjong_ai/core/state.py` 增加 `ActionTrace`、`recent_actions`、历史上限常量。
  - `src/mahjong_ai/core/engine.py` 在主要决策阶段记录最近动作。
- 观测新增“近期动作历史”特征：
  - `src/mahjong_ai/env/obs_vector_encoder.py` 新增 `RECENT_ACTION_*` 特征块（固定 slot、actor one-hot、action kind one-hot、action id 归一化）。
- 对手池稳定化参数网格与报告：
  - 新增 `src/mahjong_ai/training/self_play_grid.py`
  - CLI 新增 `grid-rllib`：`src/mahjong_ai/cli/main.py`
  - 自动输出：`grid_report_<timestamp>.json/.md`
  - 支持 `pool_size/snapshot_interval/main_prob` 组合对比。

### 1.4 回归与可运行性验证（已完成）
- `PYTHONPATH=src python -m unittest discover -s tests -v`：84 通过，12 跳过（当前解释器缺少 RL 可选依赖）。
- `PYTHONPATH=src python -m mahjong_ai.cli.main sim --games 5 --seed 1`：通过。
- 近期动作历史相关测试通过：
  - `tests/test_core_action_history.py`
  - `tests/test_obs_vector_encoder.py`（在 `/usr/bin/python3` 环境验证）
- 网格脚本与 CLI 已实跑：
  - `python -m mahjong_ai.training.self_play_grid ...`（单组合 + 多组合）
  - `python -m mahjong_ai.cli.main grid-rllib ...`（可从无 PyYAML 的轻环境启动，使用 `--python-bin` 解析 YAML）
- 对比报告样例：
  - `runs/self_play_grid_compare/grid_report_20260213T101817Z.md`

## 2. 下一阶段工作（按优先级）

### P1：策略质量提升（进行中）
- 扩展更强的序列/时序观测（如分阶段动作统计、局内事件摘要）。
- 用更大评测样本进行网格二轮筛选（提升 `eval_games`，固定更长 seed 列表）。
- 评估奖励设计（得分/胜率混合目标）。

### P2：评测体系完善（进行中）
- 固化长期 benchmark：统一训练步数、固定对手、统计区间与阈值。
- 增加自动汇总脚本（多次 grid 报告聚合 + 可视化导出）。

### P3：工程化与文档（进行中）
- 补充更多训练配置模板（smoke / standard / long-run）。
- 增加 CI 分层回归（core-only 与 RL-extended 两套环境）。

## 3. 关键风险与注意事项
- RLlib 版本迭代较快，旧 API 可能在后续版本移除。
- 本仓库不是 git 仓库，修改追踪依赖文档与命令记录。
- YAML 配置加载依赖 `pyyaml`；`grid-rllib` 已支持通过 `--python-bin` 解释器回退加载。

## 4. 常用命令
- 回归测试：
  - `PYTHONPATH=src python -m unittest discover -s tests -v`
- 快速模拟：
  - `PYTHONPATH=src python -m mahjong_ai.cli.main sim --games 5 --seed 1`
- 最小训练：
  - `PYTHONPATH=src /usr/bin/python3 -m mahjong_ai.cli.main train-rllib --config configs/train/ppo_selfplay_rllib.yaml --num-iterations 1 --checkpoint-every 1 --eval-every 1 --eval-games 1 --run-dir runs`
- checkpoint 评测：
  - `PYTHONPATH=src /usr/bin/python3 -m mahjong_ai.cli.main eval-rllib --checkpoint runs/ppo_selfplay --baselines heuristic,random --games 20 --seed 1 --output runs/ppo_selfplay/eval`
- 稳定化参数网格：
  - `PYTHONPATH=src python -m mahjong_ai.cli.main grid-rllib --pool-sizes 2,4 --snapshot-intervals 1,5 --main-probs 0.2,0.4 --num-iterations 1 --eval-every 1 --eval-games 2 --python-bin /usr/bin/python3 --run-dir runs/self_play_grid`
- 一键 smoke：
  - `bash scripts/train_smoke.sh`
