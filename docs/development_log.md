# 开发日志

本文记录后续可能再次影响项目的决策和回归问题。它不是 changelog；面向 release 的变化应写入
`README.md` 或 `ROADMAP.md`。

## 2026-05-21 - Driver backend 兼容性边界

### 现象

项目方向倾向使用 `ctypes + libcuda.so.1 + embedded PTX` 作为轻量 worker 路径，但这容易被误读
为“所有机器都能跑”。

### 根因

Driver API 路径移除了 PyTorch 和 CUDA Toolkit 依赖，但仍依赖已安装的 NVIDIA driver、可访问的
NVIDIA GPU device，以及足够新、可以 JIT 内置 PTX 的 driver。

### 影响范围

没有 NVIDIA GPU、driver 损坏、容器限制 GPU device 或 driver library、macOS、AMD/Intel GPU、
以及未单独支持的 Windows 环境，都不应预期 driver backend 可用。

### 修复方式

把准确边界记录为“在有 NVIDIA driver 且 CUDA 设备可访问的 Linux 机器上工作”，保留 PyTorch
作为显式回退后端，并把 `gpu-holder doctor --backend driver` 暴露为第一诊断入口。

### 回归测试

- `gpu-holder doctor --backend driver --json`
- 短生命周期 `WorkerProcess(..., backend="driver")` 启停 smoke
- `gpu-holder guard --gpus 0 --dry-run --backend driver`
- 覆盖 backend strategy 文档和共享 worker controls 的单元测试

### 后续防线

backend-neutral timing 和 hold-mode 语义保持在 `worker_controls.py`。未来兼容性表述必须写入
`docs/backend_strategy.md`，并绑定具体 doctor 与 worker smoke test。
