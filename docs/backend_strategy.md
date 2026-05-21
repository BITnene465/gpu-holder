# 后端策略

`gpu-holder` 应该容易运行在共享 NVIDIA 训练机器上。长期默认 backend 应避免重型 Python
CUDA stack，但兼容性表述必须精确。

## 当前后端

当前 worker backend 在启动 holder worker 时使用 CUDA 版 PyTorch。这对已经使用 PyTorch
训练的开发机器简单且稳定，但对于一个小型 guard 工具来说太重，不适合作为长期默认依赖。

基础包没有第三方运行时依赖。PyTorch 是可选依赖，仅当前 compute worker 需要：

```bash
python -m pip install -e ".[torch]"
```

公开 backend selector 当前支持：

```bash
gpu-holder guard --backend torch
gpu-holder guard --backend driver
gpu-holder doctor --backend torch
gpu-holder doctor --backend driver
```

保持显式 selector 可以让后续默认 backend 切换不需要重新设计 CLI。

`driver` backend 仍是实验性能力，但已经可作为 worker 使用。诊断路径会检查
`libcuda.so.1` 是否可加载、`cuInit` 是否成功、driver version 是否可查询、至少一个 CUDA
device 是否可见，以及一个很小的内置 PTX kernel 是否能 JIT、launch 并 synchronize。worker
路径会创建 Driver API context，按需用 `cuMemAlloc` 保留显存，并根据与 PyTorch backend
相同的 duty-cycle 控制启动内置 spin kernel。Driver API worker 故意只使用一个保守的 spin
kernel；`matmul`、`conv`、`fft`、`elementwise` 等 `--program` 选择仍是 PyTorch backend 的
工作负载变体。

当前代码边界：

- `backends.py`：backend 名称、校验和健康检查。
- `driver_backend.py`：NVIDIA Driver API 诊断和 worker 代码。
- `telemetry.py`：只读 `nvidia-smi` 采集和 snapshot 解析。
- `worker.py`：进程生命周期、startup readiness、shutdown 和 backend dispatch。
- `worker_controls.py`：backend-neutral duty-cycle、burst jitter 和 hold-mode 语义。
- `torch_backend.py`：PyTorch 专属 CUDA 显存和计算工作负载。
- `policy.py`：backend-neutral 单卡调度决策。

## 倾向的默认后端

最终倾向的默认 backend 是：

```text
ctypes + libcuda.so.1 + embedded conservative PTX
```

这意味着：

- 通过 `ctypes` 直接使用 NVIDIA Driver API。
- 加载由已安装 NVIDIA driver 提供的 `libcuda.so.1`。
- 在 package 内内置一个小 PTX kernel。
- 运行时通过 driver JIT PTX。
- 避免 PyTorch、CUDA Toolkit、`nvcc`、CuPy、Numba 和本地编译。

这最适合小型 CLI，因为它依赖的是运行 CUDA 工作负载本来就需要的同一套 driver stack。

## 兼容性边界

Driver API backend 应表述为：

> 在有 NVIDIA driver 且 CUDA 设备可访问的 Linux 机器上工作。

不应写成“所有机器都能跑”。

预期可运行：

- 带 NVIDIA GPU 的 Linux 训练主机。
- 正确挂载 `/dev/nvidia*` device 和 driver library 的容器。
- 未安装 PyTorch 或 CUDA Toolkit 的环境。
- `nvidia-smi` 和 `libcuda.so.1` 能看到目标 device 的机器。

不预期可运行：

- 没有 NVIDIA GPU 的机器。
- driver 损坏或缺失的机器。
- 未暴露 GPU device 或 driver library 的容器。
- macOS。
- AMD、Intel 或其他非 NVIDIA GPU。
- Windows，直到单独实现 `nvcuda.dll` 路径。

driver 版本也重要。内置 PTX 必须使用保守 PTX version 和简单 kernel，确保仍常见的旧数据中心
driver 可以 JIT。

## 实现规则

Driver API backend 应遵守以下规则：

1. 优先加载 `libcuda.so.1`，不依赖开发环境 symlink `libcuda.so`。
2. 谨慎处理 `CUDA_VISIBLE_DEVICES`、物理 index 和 GPU UUID。不要假设 logical index `0`
   一定等于物理 GPU `0`。
3. 声称 backend 可用前，startup smoke test 必须通过：
   - 初始化 driver。
   - 枚举 device。
   - 在选中的 GPU 上创建 context。
   - 加载内置 PTX。
   - launch 一个极小 kernel。
4. `policy.py` 保持 backend-neutral。backend 代码只实现 worker execution。
5. Driver API backend 成熟前，保留 PyTorch backend 作为回退路径。
6. 在 `gpu-holder doctor` 中暴露清晰诊断，不要等到 worker 内部才失败。

## 验收标准

把 Driver API backend 设为默认前，必须验证：

- `gpu-holder doctor --backend driver` 报告 driver library、GPU count 和 PTX smoke-test 状态。
- `gpu-holder guard --backend driver --once` 能干净启动和停止一个 worker。
- 前台 shutdown 能及时退出且没有残留 worker。
- `CUDA_VISIBLE_DEVICES` remapping 有测试覆盖，或有文档化的手动检查。
- PyTorch 回退路径对已有用户仍可用。
