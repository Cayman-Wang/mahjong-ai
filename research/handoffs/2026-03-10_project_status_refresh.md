# Handoff (mahjong-ai) - Project Status Refresh

Date: 2026-03-10
Repo root: /home/wangyumu/wym-project/mahjong-ai

## Current Truth

- 核心对局引擎已可运行，关键规则与不变量已有测试覆盖。
- RLlib 训练 MVP 已实现，不再是占位符。
- `vision_bot` 仍是预留模块，不在近期主线。
- 当前主目标是把项目收口为稳定研究基座：benchmark、训练模板、评测口径、实验复现。

## Current Priorities

1. 固定 benchmark 协议与 seed 集。
2. 固定 smoke / standard / long-run 训练模板。
3. 在冻结评测口径下推进策略质量提升。
4. 为未来 CI 和长跑实验做工程化准备。

## Recommended Entry Points

- `README.zh.md`
- `research/guides/project_overview_zh.md`
- `research/plans/ACTIVE_PLAN.md`
- `research/plans/mahjong-ai/master_plan_zh.md`
- `docs/training_env.md`

## Deferred Items

- `src/mahjong_ai/integrations/vision_bot/` 仅保留边界定义，不作为当前交付目标。
- 不做任何未授权自动化、内存读取、协议破解或反作弊绕过。

## Verification Snapshot

- 核心测试与 CLI 模拟可在当前仓库直接运行。
- RL 训练与评测依赖可选环境；无 RL 依赖时，相关测试会按设计跳过。
- 当前 benchmark 配置入口位于 `configs/eval/`，训练模板位于 `configs/train/`。
