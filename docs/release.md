# 发布检查清单

发布 `gpu-holder` 前按此清单确认。

## 发布前检查

- `CHANGELOG.md` 已记录本次用户可见变化。
- `README.md`、`docs/`、`examples/` 与当前 CLI 行为一致。
- 默认依赖仍然最小化，基础包不强依赖 PyTorch 或 NVML。
- CUDA/PyTorch/NVML 导入都在运行时路径或 optional extras 后面。
- 外部 GPU 进程仍然只是只读调度信号。
- 不要 kill 现有用户 GPU 任务。
- `gpu-holder start` 跟随当前 CLI 解释器，不主动注入源码树 `PYTHONPATH`。

## 本地验证

建议在独立虚拟环境中执行：

```bash
virtualenv --python /usr/bin/python3.10 --system-site-packages --clear .venv
.venv/bin/python -m pip install --no-build-isolation -e ".[dev]"
uv lock --check
.venv/bin/python -m ruff check --no-cache src tests
.venv/bin/python -m pytest -q
PYTHONPYCACHEPREFIX=/tmp/gpu-holder-pycache .venv/bin/python -m compileall -q src tests
.venv/bin/gpu-holder --version
.venv/bin/gpu-holder init-config --stdout
.venv/bin/gpu-holder recipes
.venv/bin/gpu-holder plan --fake --json
.venv/bin/gpu-holder preflight --fake --no-diagnostics --json
.venv/bin/gpu-holder simulate --scenario all --json
.venv/bin/gpu-holder tune --strict --json
.venv/bin/gpu-holder config-reference --json
.venv/bin/gpu-holder profile-reference --json
.venv/bin/gpu-holder reason-reference --json
.venv/bin/gpu-holder completion bash
.venv/bin/gpu-holder manual --format man
.venv/bin/gpu-holder alerts --json
.venv/bin/gpu-holder grafana-dashboard
.venv/bin/gpu-holder monitoring-bundle --output-dir /tmp/gpu-holder-monitoring --json
.venv/bin/python -m build
```

只读运行时检查：

```bash
.venv/bin/gpu-holder doctor --json
.venv/bin/gpu-holder status --check
.venv/bin/gpu-holder metrics
.venv/bin/gpu-holder dashboard --once
.venv/bin/gpu-holder explain
```

`status --check` 在没有 daemon 或状态 stale 时返回非零是正常的；关键是输出要准确。

## 真实 GPU smoke

只在空闲 GPU 或你明确拥有的 GPU 上运行：

```bash
.venv/bin/gpu-holder guard --gpus 7 --mem 2GiB --program matmul --min-duty-cycle 1 --max-duty-cycle 1
```

验证：

- status 按 sample interval 更新。
- `pause` 释放自有 worker。
- `pause --for 1m` 自动过期。
- `disable-gpu 7 --for 1m` 自动恢复。
- `history --since 5m` 不触碰 worker。
- `explain` 给出可读建议。
- `stop` 拒绝非 holder PID。

## 发布

- 更新 `pyproject.toml` 版本号。
- 把 `CHANGELOG.md` 的 `[未发布]` 移到正式版本段。
- 确认项目 URL 指向真实 GitHub 仓库。
- 打 tag。
- 确认 CI 通过 Python 3.10。
- 发布说明必须提到安全边界、支持的 Python 版本、optional CUDA 依赖。
