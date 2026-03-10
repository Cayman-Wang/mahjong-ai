# ACTIVE_PLAN

goal: 将 mahjong-ai 收口为稳定研究基座，优先固定 benchmark、训练模板、评测口径与实验复现方式。
current_milestone: M1 Benchmark 与训练复现基线建立
must_read:
  - research/guides/project_overview_zh.md
  - research/plans/mahjong-ai/master_plan_zh.md
  - research/handoffs/2026-03-10_project_status_refresh.md
  - docs/training_env.md
  - research/reviews/README.md
  - research/reviews/2026-02-13_第三轮审查-1.md
  - README.zh.md
locked_decisions:
  - 近期主目标是研究基座，不是 vision_bot 或外部平台接入。
  - `vision_bot` 明确后置，只保留接口边界与约束说明。
  - `smoke / standard / long_run` 是正式的训练与评测分级入口。
  - benchmark 优先使用固定 seed 集，评测报告必须可横向比较。
  - 训练产物、checkpoint 和评测输出继续保留在 research/ 之外。
next_action: 使用 `configs/eval/` 的冻结 benchmark 对现有 checkpoint 产物建立第一版对照样例，并补齐标准运行指引。
out_of_scope:
  - 不在近期主线推进 `vision_bot` 实现。
  - 不引入任何未授权自动化、内存读取或协议破解能力。
  - 不把运行产物迁入 `research/`。
latest_retrospective: none
last_updated: 2026-03-10
