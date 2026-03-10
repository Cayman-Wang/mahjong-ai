# ACTIVE_PLAN

goal: 将 research/ 建成 mahjong-ai 的主计划入口，统一管理训练路线、评测标准、审查、交接与复盘文档。
current_milestone: M1 历史文档迁移与主计划收口
must_read:
  - research/plans/mahjong-ai/master_plan_zh.md
  - research/reviews/README.md
  - research/reviews/2026-02-13_第三轮审查-1.md
  - research/plans/mahjong-ai/archive/PLAN_2026-02-13_legacy_zh.md
  - research/reviews/2026-02-13_第二轮审查-1.md
  - research/handoffs/2026-02-12_session_handoff_mahjong-ai.md
  - README.zh.md
locked_decisions:
  - research/ 是新的主计划入口，主目录不再保留散落的计划、交接和审查文档。
  - README.zh.md 继续保留在仓库根目录，作为项目使用说明和包说明入口。
  - 原 `debug/` 下的正式审查文档已归拢到 `research/reviews/`，历史草稿进入 `research/reviews/archive/`。
  - 训练产物、checkpoint 和评测输出继续保留在 research/ 之外。
next_action: 基于 legacy plan、handoff 与 review，补齐主计划中的 benchmark、评测口径和近期执行里程碑。
out_of_scope:
  - 不迁移 README.zh.md、pyproject.toml 等项目入口或构建元数据。
  - 不移动 runs/、docs/ 等运行产物或非研究型目录。
latest_retrospective: none
last_updated: 2026-03-10
