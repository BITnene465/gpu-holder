# 贡献指南

`gpu-holder` 是面向共享 NVIDIA 训练机器的小型运维工具。任何变更都应保持项目易安装、
易审计，并且对其他用户的任务保持保守。

## 开发环境

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[torch]"  # 可选，仅真实 CUDA worker 测试需要
```

提交 pull request 前请先运行常规本地检查：

```bash
python -m ruff check --no-cache src tests
python -m pytest -q
PYTHONPYCACHEPREFIX=/tmp/gpu-holder-pycache python -m compileall -q src tests
python -m build
```

## 安全规则

以下不变量属于项目契约：

- 外部 CUDA 进程只是只读调度信号。
- `gpu-holder` 只能停止自己创建的 guard 和 worker 进程。
- 不得增加杀死、挂起、renice、ptrace 或以其他方式控制外部任务的行为。
- 如果存在干扰训练或推理进程的风险，优先释放自己的 holder。
- 真实 CUDA smoke test 只允许在你拥有或确认空闲的 GPU 上运行。

## 架构规则

- 单卡调度决策放在 `policy.py`。
- CLI、daemon 生命周期和状态文件放在 `cli.py`。
- 只读 `nvidia-smi` 采集和 GPU snapshot 解析放在 `telemetry.py`。
- worker 进程启动、关闭和 backend dispatch 放在 `worker.py`。
- backend-neutral worker timing 和 hold-mode 控制放在 `worker_controls.py`。
- PyTorch 专属计算代码放在 `torch_backend.py`。
- NVIDIA Driver API 诊断和 worker 代码放在 `driver_backend.py`。
- backend 名称、校验和健康检查 dispatch 放在 `backends.py`。

`policy.py` 必须保持 backend-neutral，不应导入 PyTorch、`ctypes` 或 subprocess worker
实现。

扩展范围前先阅读 [ROADMAP.md](ROADMAP.md)。新增功能必须保留项目 non-goals 和安全边界。

## 测试指南

policy、解析、状态 payload、backend 校验和 worker 生命周期使用单元测试覆盖。默认测试套件
不应依赖真实 GPU。

只有变更影响真实 CUDA 行为时，才运行手动 GPU 检查：

```bash
gpu-holder doctor --json
gpu-holder doctor --backend driver --json
gpu-holder guard --gpus 0 --risk-util 0.6 --target-util 0.9 --mem 0 --backend driver --once
gpu-holder guard --gpus 0 --risk-util 0.6 --target-util 0.9 --mem 0.05 --backend torch --once
```

请在 pull request 中写明你运行过的命令。
