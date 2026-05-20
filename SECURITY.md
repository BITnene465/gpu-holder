# 安全策略

`gpu-holder` 的核心安全边界是：外部 GPU 进程只能作为只读调度信号。

如果你发现任何会终止、暂停、调整优先级、释放显存，或以其他方式操控非
`gpu-holder` worker 进程的行为，请把它视为安全缺陷。

当前仓库发布到 GitHub 后，私密报告优先通过 GitHub Security Advisory 提交。
