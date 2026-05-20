# 测试说明

## 无 GPU 测试

这些检查不会启动真实 worker，适合在忙机器上运行：

```bash
python -m pip install --no-build-isolation -e .
python -m pytest -q
python -m ruff check --no-cache src tests
PYTHONPYCACHEPREFIX=/tmp/gpu-holder-pycache python -m compileall -q src tests
gpu-holder --version
gpu-holder plan --fake
gpu-holder plan --fake --json
gpu-holder preflight --fake --no-diagnostics
gpu-holder preflight --fake --no-diagnostics --json
gpu-holder simulate --scenario all --json
gpu-holder tune --strict --json
```

`pytest` 已配置从项目根目录导入 `src/`。这些测试不导入 PyTorch、不申请 CUDA 显存、不启动 holder worker。

## 离线策略模拟

`simulate` 用时间序列 trace 测试策略行为：

```bash
gpu-holder simulate
gpu-holder simulate --scenario thermal --json
gpu-holder simulate --scenario all --json
gpu-holder simulate --trace trace.json --json
```

内置场景包括 `mixed`、`idle`、`busy`、`thermal`、`quota-risk`。发布前建议跑 `--scenario all`。

`tune` 在模拟基础上给出配置建议：

```bash
gpu-holder tune
gpu-holder tune --profile quota --json
gpu-holder tune --strict --json
gpu-holder tune --compare-profiles
gpu-holder tune --target-util 60 --program matmul
```

## 只读 GPU 测试

这些命令读取真实状态，但不启动或停止 worker：

```bash
gpu-holder doctor --json
gpu-holder status --check
gpu-holder metrics
gpu-holder dashboard --once
gpu-holder explain
```

`doctor` 在没有 NVML Python 绑定但 `nvidia-smi` 可用时仍可通过监控后端检查。PyTorch CUDA 不可用时，真实 worker 不能运行。

## 真实 GPU smoke

只在空闲 GPU 或你明确拥有的 GPU 上运行。示例：

```bash
gpu-holder guard \
  --gpus 7 \
  --target-util 90 \
  --mem 2GiB \
  --program matmul \
  --min-duty-cycle 1 \
  --max-duty-cycle 1 \
  --sample-interval 2
```

验证点：

- `nvidia-smi` 能看到利用率上升。
- `status` 中 worker alive。
- `pause` 会释放自有 worker。
- `disable-gpu` 会释放并跳过指定 GPU。
- `stop` 不会停止非 holder PID。

不要在 release 验证中 kill 现有用户 GPU 任务。
