# mahjong-ai 主计划（中文）

- 创建日期：2026-03-10
- 最后更新：2026-03-10
- 负责人：[待填写]

## 一、目标
- 在本地稳定复现四川麻将（血战到底）完整流程，并保持牌守恒、零和结算、合法动作约束等关键不变量。
- 提供可配置规则框架，支持不同平台规则差异。
- 提供可运行的自对弈训练与评测入口，并逐步建立可比较的 benchmark。

## 二、主计划入口说明
- `research/plans/ACTIVE_PLAN.md` 是当前唯一活动入口。
- 历史根目录计划已迁移至 `research/plans/mahjong-ai/archive/PLAN_2026-02-13_legacy_zh.md`。
- 历史交接已迁移至 `research/handoffs/2026-02-12_session_handoff_mahjong-ai.md`。
- 历史审查已迁移至 `research/reviews/2026-02-13_第二轮审查-1.md`。
- 根目录保留 `README.zh.md` 作为项目使用说明与包说明入口。

## 三、当前状态
- 对局引擎与关键规则流程已落地，已有单元测试覆盖主要不变量。
- RLlib 训练 MVP、checkpoint 保存、周期评测和 self-play 对手池已接入。
- 当前重点从“能跑”转向“策略质量提升 + 评测口径收敛 + 工程化收口”。

## 四、方案总览
- 用 `research/` 统一沉淀计划、审查、交接和复盘，避免根目录继续累积散落文档。
- 训练与评测以 `runs/` 等运行目录承载产物，`research/` 只保留决策和方法文档。
- 后续新增 milestone、review、handoff 均优先落在 `research/` 对应子目录。

## 五、里程碑
- M1：文档与入口收口
  - 完成 `research/` 骨架初始化。
  - 迁移根目录计划、交接、审查文档。
  - 用 `ACTIVE_PLAN` 固定当前必读入口。
- M2：策略质量提升
  - 固化 benchmark、eval 样本量和固定 seed 集。
  - 继续筛选 reward 与 self-play 参数。
  - 补齐训练效果对比报告。
- M3：评测与工程化
  - 固化 smoke / standard / long-run 配置模板。
  - 增加分层回归与 CI。
  - 规范化里程碑复盘与跨会话交接。

## 六、验收标准
- 研究与计划类文档不再散落于仓库根目录。
- 新会话可仅通过 `ACTIVE_PLAN.md` 和 `must_read` 清楚进入当前上下文。
- benchmark、训练配置和评测口径有明确文档化入口。
- 运行产物与决策文档保持分离。

## 七、风险与对策
- RLlib / Ray 版本变化较快，需要持续约束依赖与 API 兼容性。
- 历史计划与现状可能出现偏差，需要继续整合 legacy plan 与 review 结论。
- 审查报告中的相对链接在迁移后必须保持可跳转，后续新增报告也要遵守同一规则。

## 八、默认假设
- 默认中文文档。
- 默认 `research/` 作为计划、审查、交接和复盘的唯一主入口。
- 默认不在 `research/` 存放训练产物、日志输出或 checkpoint。
