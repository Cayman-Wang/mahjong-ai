# 自对弈过程可视化专题方案（中文）

- 创建日期：2026-03-18
- 最后更新：2026-03-18
- 负责人：[待填写]

## 一、背景与问题

当前仓库已经具备：
- 可运行的四川麻将引擎
- RLlib 自对弈训练入口
- checkpoint、续训与周期评测链路

但训练期只能看到聚合指标与 checkpoint 产物，缺少“逐局、逐步、可人工审阅”的自对弈回放。结果是：
- 难以判断自对弈行为是否符合预期
- 很难排查策略是否在异常信息、非法动作或不合理决策上收敛
- 训练日志与 benchmark 报告之间缺少中间层可解释证据

因此需要增加一个轻量、可复现、默认可用的训练期自对弈回放能力。

## 二、目标与非目标

### 目标

- 在 `train-rllib` 的周期评测阶段自动抽样少量自对弈局，输出可直接阅读的文本回放。
- 同时支持两种视角：
  - 全知视角：方便直接审查四家真实状态与动作是否合理
  - `seat0` 视角：方便检查主策略在其信息集下的可见信息与决策
- 回放使用当前 iteration 的策略参数、对手池快照和评测 seed，保证可复现、可对比。
- 保持运行产物继续落在 `runs/`，不进入 `research/`。

### 非目标

- 本阶段不做浏览器 HTML 回放器。
- 本阶段不做 RLlib rollout 实时直播面板。
- 本阶段不新增独立 replay CLI；先绑定在训练期评测节点。
- 本阶段不把结构化 trace 作为正式外部产物暴露；对外仍以文本 artifact 为主。

## 三、方案设计

### 1. 触发时机

- 仅在 `evaluation.eval_every` 命中的 iteration 上触发。
- 回放采样使用该次评测已经解析出的 `eval_seeds` 前缀，不引入额外随机源。
- 默认每次评测抽样 1 局；后续通过配置扩展到多局。

### 2. 配置归属

回放配置挂在 `evaluation.replay`，与周期评测语义保持一致：

```yaml
evaluation:
  replay:
    enabled: true
    games_per_eval: 1
    output_dir: ""
    include_omniscient: true
    seat_views: [0]
    max_steps: 5000
```

字段解释：
- `enabled`：是否生成回放
- `games_per_eval`：每次评测抽样的回放局数
- `output_dir`：回放根目录；空串表示 `runs/<experiment>/replays/iter_<iteration>/`
- `include_omniscient`：是否导出全知视角
- `seat_views`：需要导出的单座位视角列表，v1 默认 `[0]`
- `max_steps`：单局保护上限，避免异常死循环

### 3. 产物约定

每次命中评测时，输出目录固定为：

- `runs/<experiment>/replays/iter_<iteration>/`

每局固定文件名：
- `seed_<seed>_omniscient.txt`
- `seed_<seed>_seat0.txt`

训练日志打印每个生成文件路径；训练评测 JSON 报告附带 replay 配置摘要与文件列表。

### 4. 回放内容

#### 全知视角

必须展示：
- iteration / seed / 生成时间 / rules_path
- 各 seat 的 policy assignment
- 初始状态与每一步的 phase、current player、required players、剩余牌墙、分数
- 四家完整手牌、定缺、副露、弃牌、是否已胡
- 各 required players 的合法动作列表与实际动作
- engine 返回的事件流
- 每步结算增量、终局原因与最终状态

#### `seat0` 视角

必须展示：
- `seat0` 完整手牌与合法动作
- 其他三家公开信息：定缺、副露、弃牌、已胡状态
- 各 seat 的 policy assignment
- 公开事件流、分数变化、终局原因

必须打码：
- 其他三家暗手
- 其他三家在换三张阶段的私有选牌
- 其他三家基于私有手牌推导出的合法动作集合

### 5. 采集方式

- 在训练进程内新增 replay trace 采集层。
- 使用当前 iteration 的 `algo` 和 `policy_mapping_fn`，对 4 个座位做一次完整自对弈。
- replay 采集不复用 baseline 评测逻辑；它是“当前主策略 + 当前对手池”的纯自对弈留样。
- 先记录结构化 trace，再由同一 trace 渲染多个文本视角，避免双份采集逻辑分叉。

## 四、实现拆解与测试口径

### 实现拆解

1. 在训练配置默认值与训练模板中加入 `evaluation.replay`
2. 在 `train_with_rllib()` 的周期评测分支中挂接 replay 生成
3. 新增 replay trace / renderer 模块，负责：
   - 单局 trace 采样
   - 状态快照标准化
   - 全知视角文本渲染
   - 单座位视角文本渲染
   - artifact 写盘
4. 在训练评测 JSON 里补充 replay 摘要信息
5. 更新 `README.zh.md`、`docs/training_env.md` 和 `research/README.md`

### 测试口径

至少覆盖：
- `evaluation.replay` 默认值和参数校验
- 全知视角包含完整四家手牌
- `seat0` 视角不泄露其他三家暗手与私有换张
- 评测 iteration 命中时会生成预期 replay 文件
- `enabled=false` 或 `eval_every=0` 时不生成 replay 文件
- 有固定 `seed_list` 时，回放使用其前缀；无固定 `seed_list` 时，回放使用派生 seed 前缀

## 五、风险、限制与后续演进

### 风险与限制

- 文本回放可读性强，但不适合大规模人工浏览，因此默认只抽样少量局。
- 若策略在某一时刻产生非法动作，回放生成会直接暴露该问题；这有利于尽早发现契约错误。
- `seat0` 视角的“保密边界”必须长期与观测编码保持一致，避免调试工具反向泄露信息集。

### 后续演进

- 二期可增加 HTML 回放器或 markdown 汇总页。
- 二期可增加 checkpoint 事后独立回放命令。
- 二期可扩展到多座位视角、关键帧摘要、非法动作高亮与对局筛选。

## 六、验收标准

满足以下条件即视为本专题收口：
- 默认训练模板在周期评测时能稳定生成 replay 文本文件
- 生成路径、文件名、视角语义固定且可复现
- `seat0` 视角不泄露其他三家暗手与私有换张信息
- 文档已经明确配置入口、产物路径和查看方式
- `research/` 中已有本专题方案入口，且不破坏现有主计划定位
