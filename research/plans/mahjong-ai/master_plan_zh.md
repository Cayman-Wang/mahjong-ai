# mahjong-ai 主计划（中文）

- 创建日期：2026-03-10
- 最后更新：2026-03-10
- 负责人：[待填写]

## 一、目标

- 在本地稳定复现四川麻将（血战到底）完整流程，并保持牌守恒、零和结算、合法动作约束等关键不变量。
- 提供可配置规则框架，支持不同平台规则差异。
- 提供可运行、可复现、可比较的自对弈训练与评测入口。

## 二、当前判断

- 对局引擎、规则、计分、数值观测、RLlib 训练 MVP、checkpoint、续训和周期评测已经落地。
- `vision_bot` 仍是占位模块，当前不能作为项目已实现能力对外描述。
- 研发主线已从“补核心功能”切换到“固定 benchmark、固定训练模板、提升实验复现、再推进策略质量”。

## 三、固定决策

- `research/` 是计划、审查、交接和复盘的唯一主入口。
- `README.zh.md` 继续保留在仓库根目录，作为使用说明入口。
- 训练产物、checkpoint 和评测输出放在 `runs/` 等运行目录，不放在 `research/`。
- 近期主目标是研究基座，不优先做外部平台集成。
- `vision_bot` 后置，只有在 benchmark 与训练主线稳定后才重新进入 roadmap。

## 四、阶段计划

### Phase 0：项目基线重建（已完成）
- 迁移并收口历史计划、审查、交接文档。
- 确认核心引擎、训练 MVP 和测试现状。
- 建立 `ACTIVE_PLAN` 作为当前活动入口。

完成定义：
- 新会话可以通过 `README.zh.md`、`ACTIVE_PLAN` 和项目总览快速恢复上下文。

### Phase 1：Benchmark 与评测口径固化（当前阶段）
- 固定 `smoke / standard / long_run` 三档 benchmark 协议。
- 固定 benchmark 对手组合、seed 集、指标集和报告格式。
- 建立正式 benchmark 入口：`configs/eval/` + `eval-benchmark`。
- 明确 checkpoint 对比的标准产物与命名约定。

完成定义：
- 相同 checkpoint 在相同 benchmark 配置下可重复评测并得到可比较报告。
- 评测报告至少包含 `avg_score`、`win_rate`、`score_std`、`avg_steps`、`illegal_action_rate`。

### Phase 2：训练复现与实验工程化
- 固定 `smoke / standard / long_run` 训练模板。
- 固定训练产物目录结构、配置快照、checkpoint 输出与续训流程。
- 把 `scripts/train_smoke.sh` 定义为最低回归门槛。
- 形成“核心引擎回归”和“RL 扩展回归”两层执行约定。

完成定义：
- 任意一次训练实验都能复现配置、恢复训练、重新评测。
- 新成员可以直接判断应该跑哪套模板和哪套回归。

### Phase 3：策略质量提升
- 在冻结 benchmark 之后，再系统筛选 self-play 参数、奖励设计和新增观测。
- 所有候选策略必须以同一套 benchmark 和 seed 协议做对比。
- 网格搜索、单次实验和默认 checkpoint 的升级依据必须文档化。

完成定义：
- 至少形成一个被明确认可的默认训练配置和默认 checkpoint。
- 所有“更优”结论都能追溯到具体评测报告。

### Phase 4：后置集成准备
- 保留 `vision_bot` 的接口边界、合规约束和 profile 轮廓。
- 仅在 Phase 1 到 Phase 3 收口后，再进入视觉识别、状态跟踪、执行链路设计。

完成定义：
- 外部集成不再反向影响核心引擎、训练协议和 benchmark 主线。

## 五、正式入口与模板

- 项目总览：`research/guides/project_overview_zh.md`
- 当前活动计划：`research/plans/ACTIVE_PLAN.md`
- 当前状态交接：`research/handoffs/2026-03-10_project_status_refresh.md`
- 训练模板：`configs/train/ppo_selfplay_rllib_smoke.yaml`、`configs/train/ppo_selfplay_rllib_standard.yaml`、`configs/train/ppo_selfplay_rllib_long_run.yaml`
- Benchmark 配置：`configs/eval/smoke.yaml`、`configs/eval/standard.yaml`、`configs/eval/long_run.yaml`

## 六、阶段验收口径

- 所有正式阶段都必须定义入口配置、回归命令和输出产物。
- 没有 benchmark 报告支撑的“效果更好”不作为默认路线依据。
- 没有模板、没有输出路径约定、没有回归门槛的流程不视为已收口。

## 七、风险与对策

- RLlib / Ray 版本变化较快，需要持续约束依赖与 API 兼容性。
- 历史 handoff / review 中有部分结论已过时，必须优先读取当前总览与当前 handoff。
- 训练“能跑”与策略“有效”不是一回事，所有对外结论必须回到冻结 benchmark。

## 八、默认假设

- 默认中文文档。
- 默认单仓库内维护训练、评测和研究文档。
- 默认近期不在主线推进 `vision_bot` 实现。
