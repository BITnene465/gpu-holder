# 安全策略

`gpu-holder` 面向共享 GPU 机器设计。它的安全边界故意保持很窄：

- 只允许启动和停止自己创建的 guard 与 worker 进程。
- 外部 CUDA 进程必须只作为只读调度信号。
- 不得杀死、挂起、renice、ptrace 或以其他方式控制外部任务。
- 如果存在干扰训练任务的风险，应优先释放自己的 holder。

## 报告方式

请通过本仓库的 GitHub private vulnerability reporting 私下报告安全或 safety 问题：

https://github.com/BITnene465/gpu-holder/security/advisories/new

如果 private vulnerability reporting 不可用，请创建一个不包含敏感日志的最小 issue，并请求
私下联系方式。

## 有用上下文

报告行为问题时，请尽量包含：

- `gpu-holder doctor --json`
- `gpu-holder status --json`
- 启动 guard 的完整命令。
- 相关 `~/.gpu-holder/gpu-holder.log` 片段。
- GPU 型号、driver 版本，以及是否在容器内运行。

不要包含 secret、access token、私有数据集路径或专有训练日志。
