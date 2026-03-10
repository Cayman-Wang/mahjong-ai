# 新对话启动模板（跨平台通用）

> 用法：新建任意 AI 对话后，把本文件内容整体粘贴为第一条消息，再补你的具体任务。

```text
请先读取以下计划文件，再回答我的问题：

workspace_path=<workspace_path>
main_plan=<main_plan>
active_plan=<active_plan>

执行顺序要求：
1) 先读取 active_plan。
2) 再读取 main_plan。
3) 再按 active_plan 中 must_read 列表读取其余必读文件。

在你的第一条回复中，先输出 Intent Digest（三行）：
Goal: <从计划中提炼的一句话目标>
Current Milestone: <当前里程碑>
Next Action: <下一步可执行动作>

如果任何必读文件缺失、路径无效或计划冲突，请先明确报错并给出修复建议，不要直接开始实现。
```
