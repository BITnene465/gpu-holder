# 贡献指南

感谢改进 `gpu-holder`。这个项目优先保持边界清楚、依赖极少、行为可预测。

## 开发环境

推荐使用项目自己的虚拟环境：

```bash
cd /root/workspace/nene/gpu-holder
virtualenv --python /usr/bin/python3.10 --system-site-packages --clear .venv
.venv/bin/python -m pip install --no-build-isolation -e ".[dev]"
```

常用检查：

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check --no-cache src tests
PYTHONPYCACHEPREFIX=/tmp/gpu-holder-pycache .venv/bin/python -m compileall -q src tests
```

真实 CUDA smoke 只能在空闲 GPU 或你明确拥有的 GPU 上运行。不要在繁忙共享机器上随意启动真实 holder worker。

## 安全不变量

外部 GPU 进程是只读调度信号。贡献代码不得 kill、suspend、renice 或以其他方式控制非 holder PID。

允许：

- 停止 `gpu-holder` 自己创建的 worker。
- 读取 NVML 或 `nvidia-smi` 的进程元数据。
- 写状态文件、事件日志和监控指标。

不允许：

- kill 训练、推理或服务进程。
- 释放其他进程的显存。
- shell out 到破坏性进程管理命令。

## PR 检查清单

- 策略变更需要补测试。
- 默认依赖保持最小化。
- CUDA/PyTorch/NVML 导入必须放在运行时路径或 optional extras 后面。
- 行为变化需要更新 `README.md`、`docs/` 和 `CHANGELOG.md`。
- 发布前检查 `docs/release.md`。
- 确认 pytest、ruff、compileall 通过。
