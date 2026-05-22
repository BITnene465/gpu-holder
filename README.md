# gpu-holder

[![CI](https://github.com/BITnene465/gpu-holder/actions/workflows/ci.yml/badge.svg)](https://github.com/BITnene465/gpu-holder/actions/workflows/ci.yml)

<p align="center"><img src="assets/logo.png" alt="gpu-holder logo" width="160"></p>

[English](README.en.md)

## 一分钟选择

```bash
gpu-holder doctor --backend driver
gpu-holder guard --gpus 0-7 --dry-run --backend driver --explain
```

- 不确定机器环境：先运行 `gpu-holder doctor --backend driver`。
- 不想安装 PyTorch：优先试 `--backend driver`，doctor 失败时根据 hint 排查 driver/device。
- 已有 CUDA 版 PyTorch：可以直接用默认的 `--backend torch`。
- 在共享机器上首次试用：先加 `--dry-run --explain`，确认每张 GPU 的决策理由。
- 真正接管前台：去掉 `--dry-run`；后台运行则使用 `gpu-holder start ...`。

`gpu-holder` 是一个面向共享 NVIDIA 训练机器的小型 GPU guard。它会在指定 GPU
低于单卡利用率下限时启动轻量 CUDA holder worker，并在真实训练或推理任务出现时主动让出资源。
它不会杀死、挂起、降优先级或修改外部进程。

默认 worker backend 使用可选的 PyTorch，因为很多训练环境已经安装了 PyTorch。
如果不想安装 PyTorch，也可以使用实验性的 NVIDIA Driver API backend。调度策略与 backend
解耦，因此两种 backend 都遵守同一套单卡安全规则。

### 为什么需要

共享 GPU 机器常在数据加载、启动、checkpoint 或训练卡住时出现低利用率，从而被误判为空闲。
`gpu-holder` 的目标是在保持机器活跃的同时守住很窄的安全边界：

- 只启动和停止自己创建的 holder worker。
- 外部 CUDA 进程只作为只读调度信号。
- 不杀死、不挂起、不 renice，也不以其他方式控制用户训练任务。
- 按单张 GPU 决策，不使用整机平均利用率替代单卡状态。
- 大型训练进程启动后会获得 grace window，期间不会启动 assist holder 干扰初始化。

### 快速开始

```bash
git clone https://github.com/BITnene465/gpu-holder.git
cd gpu-holder
python -m pip install -e .
```

如果环境里已有 CUDA 版 PyTorch，可以直接使用默认 worker。否则按需安装可选依赖：

```bash
python -m pip install -e ".[torch]"
```

使用 PyTorch backend 在 8 张 GPU 上前台运行：

```bash
gpu-holder guard --gpus 0-7 --risk-util 0.6 --target-util 0.9 --mem 0.05 --backend torch
```

或者避开 PyTorch，使用实验性的 Driver API backend：

```bash
gpu-holder doctor --backend driver
gpu-holder guard --gpus 0-7 --risk-util 0.6 --target-util 0.9 --mem 0.05 --backend driver
```

前台 guard 可用 `Ctrl+C` 停止。

### 常用命令

```bash
gpu-holder doctor
gpu-holder doctor --backend torch
gpu-holder doctor --backend driver
gpu-holder guard --gpus 0-7 --risk-util 0.6 --target-util 0.9 --mem 0.05 --backend torch
gpu-holder guard --gpus 0-7 --risk-util 0.6 --target-util 0.9 --mem 0.05 --backend driver
gpu-holder start --gpus 0-7 --risk-util 0.6 --target-util 0.9 --mem 0.05 --backend torch
gpu-holder status
gpu-holder dashboard
gpu-holder stop
```

只做一次 dry-run 调度决策，不启动 worker、不写状态：

```bash
gpu-holder guard --gpus 0-7 --risk-util 0.6 --target-util 0.9 --mem 0.05 --dry-run
```

真实运行一轮 guard 后退出：

```bash
gpu-holder guard --gpus 0-7 --risk-util 0.6 --target-util 0.9 --mem 0.05 --once
```

### 策略

所有利用率和显存比例都使用 `0..1` 浮点数。兼容旧写法，例如 `60`、`60%`、`0.6`
都会归一化为 `0.6`。

重要默认值：

- `--risk-util 0.5`：GPU 利用率低于 50% 时启动 holder。
- `--target-util 0.75`：holder 的计算 duty target。
- `--mem 0.2`：holder 最多保留 20% 单卡显存。
- `--reserve 2GiB`：至少保留这部分空闲显存。
- `--busy-process-mem-threshold 10GiB`：把大型外部 CUDA 进程识别为训练/推理任务。
- `--process-grace-window 120`：大型外部进程初始化后的 120 秒内不启动 assist holder。
- `--sample-interval 2`：每 2 秒重新评估一次。

单卡调度规则：

1. 没有外部 CUDA 进程且利用率低于 `risk-util`：启动或保留普通 holder。
2. 普通 holder 已运行：直到外部 CUDA 进程出现前都保持运行。
3. 大型外部 CUDA 进程出现：停止该 GPU 上的 holder，进入 `busy_process_grace`。
4. grace window 内：继续让出资源，即使外部进程利用率较低。
5. grace window 结束且利用率仍低于 `risk-util`：使用 `--assist-mem` 启动 assist holder。
6. 外部进程利用率回到 `risk-util` 以上：再次释放 assist holder。
7. 外部进程退出且 GPU 低于 `risk-util`：下一次采样重新接管。

### 安全模型

`gpu-holder` 不管理外部任务。它唯一执行的进程控制操作是：

- 停止自己前台/后台 guard。
- 停止该 guard 创建的 worker 子进程。

如果 pidfile 指向的不是 `gpu-holder` guard 进程，它会拒绝停止。

### 架构

```text
src/gpu_holder/
  cli.py       # CLI、daemon 生命周期、状态文件、guard 编排
  backends.py  # backend 选择与健康检查
  driver_backend.py # NVIDIA Driver API 诊断与 worker 实现
  models.py    # GPU snapshot、进程、调度决策 dataclass
  policy.py    # 单卡调度策略与显存 sizing
  telemetry.py # 只读 nvidia-smi 采集与 snapshot 解析
  worker.py    # worker 进程生命周期与 backend dispatch
  worker_controls.py # backend-neutral duty cycle、jitter、hold-mode 控制
  torch_backend.py # PyTorch CUDA worker 实现
  __main__.py
```

`policy.py` 不依赖 subprocess 或 PyTorch。GPU telemetry 与 CLI/scheduler 隔离，worker
生命周期也与 backend 实现分离，便于测试交接规则和继续添加更轻的 backend。

### 依赖策略

基础安装没有第三方运行时依赖，只使用系统 `nvidia-smi` 做只读 GPU 状态采集。

默认计算 worker 需要 CUDA 版 PyTorch：

```bash
python -m pip install -e ".[torch]"
```

backend selector 暴露 `--backend torch` 和 `--backend driver`。当前默认仍是 `torch`。
`driver` backend 使用 `ctypes`、`libcuda.so.1` 和内置 PTX，不需要 PyTorch、CUDA Toolkit、
`nvcc`、CuPy、Numba 或本地编译。
`gpu-holder doctor --backend driver` 会检查 `libcuda.so.1` 与 CUDA 设备是否可见，并 JIT
和启动一个很小的内置 PTX kernel。

后续 backend 方向：

- 默认：通过 `libcuda.so.1` 使用 NVIDIA Driver API worker，不依赖 PyTorch 或 CUDA Toolkit。
- fallback：保留 PyTorch worker，服务已安装 PyTorch 的环境。
- monitor-only：只做状态与策略检查，不启动 CUDA worker。

兼容性表述必须是“在有 NVIDIA driver 且 CUDA 设备可访问的机器上工作”，不是“所有机器都能跑”。

参见 [docs/backend_strategy.md](docs/backend_strategy.md) 了解兼容性边界和切换默认 backend
前的验收标准。

参见 [ROADMAP.md](ROADMAP.md) 了解当前范围、近期优先级和明确的 non-goals。
参见 [docs/development_log.md](docs/development_log.md) 了解反复出现的重要决策与回归记录。

### 开发

参见 [CONTRIBUTING.md](CONTRIBUTING.md) 了解架构规则、安全不变量和 pull request 要求。

```bash
python -m pip install -e ".[dev]"
python -m ruff check --no-cache src tests
python -m pytest -q
PYTHONPYCACHEPREFIX=/tmp/gpu-holder-pycache python -m compileall -q src tests
python -m build
```

真实 CUDA smoke test 只应在你拥有或确认空闲的 GPU 上运行。
