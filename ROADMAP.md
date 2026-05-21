# 路线图

`gpu-holder` 面向使用共享 NVIDIA 训练机器的工程师和科研人员，目标是提供一个小型、
可审计的 GPU 空闲利用率 guard。

## 当前范围

- 前台和后台 guard CLI。
- 基于利用率、显存和外部 CUDA 进程的单卡调度策略。
- 面向大型外部训练或推理任务的启动 grace window。
- 用于真实 CUDA 利用率的 PyTorch worker backend。
- 面向无 PyTorch 环境的实验性 NVIDIA Driver API worker backend。
- 通过 `gpu-holder doctor --backend driver` 执行带内置 PTX smoke 的 NVIDIA Driver API 诊断。
- 基于 `ctypes + libcuda.so.1 + embedded conservative PTX` 的 Driver API 实现。
- 只读 `nvidia-smi` telemetry。
- status、policy 和诊断没有第三方运行时依赖。

## 近期优先级

1. 在真实共享机器负载下加固 Driver API worker backend。
2. 保留 `--backend torch`，作为已依赖 PyTorch 环境的回退后端。
3. 在把 Driver API backend 设为默认前，改进 `CUDA_VISIBLE_DEVICES`、物理 GPU index 和
   GPU UUID 处理。
4. 默认测试保持 GPU-free，同时文档化明确的手动 CUDA smoke 检查。

## 兼容性现实

Driver API 方向是为了让 `gpu-holder` 更容易在共享 NVIDIA 训练机器上运行。它不是“所有机器
都能跑”的承诺。

预期 baseline：

- Linux。
- NVIDIA driver 已安装且健康。
- 进程能看到 `libcuda.so.1`。
- 目标 CUDA 设备通过 `/dev/nvidia*` 暴露。
- driver 足够新，可以 JIT 内置 PTX。

仍不覆盖：

- 没有 NVIDIA GPU 的机器。
- 损坏或缺失的 driver 安装。
- 没有挂载 driver library 或 GPU device 的容器。
- 非 NVIDIA GPU。
- macOS。
- Windows，直到单独实现 `nvcuda.dll` 路径。

## 非目标

- 管理、杀死、挂起、renice 或以其他方式控制外部 GPU 任务。
- 替代集群调度器、任务队列或 quota 系统。
- 保证兼容非 NVIDIA GPU。
- 宣称支持每一台机器或每一种操作系统。
- 在核心包中加入重型 monitoring stack 或 dashboard。

## 默认后端毕业标准

只有满足以下条件后，Driver API backend 才能成为默认 backend：

- `gpu-holder doctor --backend driver` 验证 driver load、device count、PTX JIT 和极小
  kernel launch。
- `gpu-holder guard --backend driver --once` 能干净地启动并停止一个 worker。
- 前台关闭后没有残留 holder worker。
- `CUDA_VISIBLE_DEVICES` remapping 由测试覆盖，或有文档化的手动检查。
- PyTorch backend 仍然可用，并作为回退路径写入文档。
