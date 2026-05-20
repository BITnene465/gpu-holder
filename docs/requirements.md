# 需求说明

## 背景

共享 GPU 机器常见的回收规则是：如果某个小时的平均 GPU 利用率低于阈值，例如 `70%`，机器会被回收。单纯占显存没有意义，因为显存占用不会提高 GPU 利用率；单纯写死计算负载又容易影响真实任务。

`gpu-holder` 要解决的是：在真实任务不足以撑起整机平均利用率时补计算负载；当真实任务出现时尽量让道。

## 功能需求

- 监控 NVIDIA GPU 利用率、显存、温度和 per-process 显存。
- 优先使用 NVML；缺少 Python NVML 绑定时，使用 `nvidia-smi` fallback。
- 支持选择 GPU：全部、指定列表、配置文件。
- 启动每张卡独立 worker。
- 默认显存占用为 `20%`，并支持：
  - 百分比：`20%`
  - GiB：`10GiB`
  - MiB：`12000MiB`
- 显存申请必须受当前空闲显存和 reserve 限制。
- 支持多种 CUDA 计算程序：
  - matrix multiplication
  - convolution
  - FFT
  - elementwise
  - mixed rotation
  - random
  - 自定义逗号列表
- 默认目标整机平均利用率为 `75%`。
- 如果整机平均已经高于目标，不应继续启动新的 holder，除非单卡进入低利用率紧急介入。
- 如果某张卡连续 `60s` 低于 `50%`，必须介入。
- 如果某张卡存在超过 `10GiB` 显存的非 holder 进程，通常让道。
- 低利用率紧急介入优先级高于大进程让道。
- 如果在忙卡上紧急介入，使用 assist 模式降低显存占用。
- 支持 protected process pattern，匹配后强制让道。
- 支持温度释放和恢复 hysteresis。
- 支持 tmux 前台运行。
- 支持轻量后台 daemon：`start`、`stop`、`status`。
- 支持非交互 CLI 状态快照。
- 支持 runtime pause / resume。
- 支持单卡临时禁用和自动恢复。
- 支持 systemd user service unit 生成，但不自动安装。
- 支持状态 stale 检测和脚本友好的健康检查。
- 支持 Prometheus metrics、告警规则、Grafana dashboard。
- 支持 read-only support report。
- 支持离线策略模拟和调参建议。

## 非功能需求

- 基础包零三方依赖。
- CUDA/PyTorch/NVML 都应位于 optional extras 或运行时导入路径。
- CLI 使用 `pyproject.toml` 管理，后续可直接打包发布。
- 代码必须可测试，策略逻辑不依赖真实 GPU。
- 外部 GPU 进程只能作为只读调度信号。
- 任何命令都不得 kill、suspend、renice 非 holder 进程。

## 验收标准

- `python -m pytest -q` 通过。
- `python -m ruff check --no-cache src tests` 通过。
- `compileall` 通过。
- `gpu-holder doctor` 能解释当前机器缺少或具备哪些运行时能力。
- `gpu-holder plan --fake`、`simulate`、`tune` 不触碰 GPU worker。
- 在明确选择的 GPU 上真实运行时，能产生计算利用率，而不是只占显存。
- `gpu-holder stop` 只停止自己的 daemon。
