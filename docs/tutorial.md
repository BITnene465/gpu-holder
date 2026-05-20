# gpu-holder 详细教程

这份教程从零开始说明如何安装、前台运行、后台运行、观察状态、临时让道、接入监控和排查问题。

## 1. 准备环境

进入项目目录：

```bash
cd /root/workspace/nene/gpu-holder
```

项目已经固定 Python 版本：

- `.python-version`: `3.10.11`
- `pyproject.toml`: `requires-python = ">=3.10,<3.11"`
- `uv.lock`: 锁定依赖解析结果

使用 `uv`：

```bash
uv sync --python 3.10.11 --extra dev
uv run gpu-holder doctor --json
```

创建独立虚拟环境：

```bash
virtualenv --python /usr/bin/python3.10 --system-site-packages --clear .venv
```

安装 CLI：

```bash
.venv/bin/python -m pip install --no-build-isolation -e .
```

验证：

```bash
.venv/bin/gpu-holder --version
.venv/bin/gpu-holder doctor --json
```

当前机器建议保留 `--system-site-packages`，因为宿主 Python 3.10 已经有可用的 CUDA PyTorch。这样环境独立于 Shaft，但不重复下载庞大的 torch wheel。

## 2. 前台占卡

前台运行适合 tmux：

```bash
cd /root/workspace/nene/gpu-holder

env -u PYTHONPATH -u VIRTUAL_ENV .venv/bin/gpu-holder guard \
  --gpus 0,1,2,3,4,5,6,7 \
  --target-util 95 \
  --mem 20% \
  --program matmul \
  --min-duty-cycle 1 \
  --max-duty-cycle 1 \
  --compute-burst-seconds 2 \
  --compute-burst-jitter 0 \
  --process-grace-window 0 \
  --sample-interval 2
```

这个配置的含义：

- 使用 8 张卡。
- 每张卡申请约 `20%` 显存。
- 使用矩阵乘法拉高利用率。
- duty cycle 固定为 `1`，尽量保持高负载。
- 关闭 process grace，避免高负载测试中被自有 worker PID 差异干扰。
- 每 `2s` 采样一次。

前台运行会定期输出一行摘要，同时写入 `/root/.gpu-holder/gpu-holder.log`，适合在 tmux 中直接观察。

## 3. 后台 daemon

启动：

```bash
env -u PYTHONPATH -u VIRTUAL_ENV .venv/bin/gpu-holder start \
  --gpus 0,1,2,3,4,5,6,7 \
  --target-util 95 \
  --mem 20% \
  --program matmul \
  --min-duty-cycle 1 \
  --max-duty-cycle 1 \
  --compute-burst-seconds 2 \
  --compute-burst-jitter 0 \
  --process-grace-window 0 \
  --sample-interval 2
```

查看：

```bash
.venv/bin/gpu-holder status
.venv/bin/gpu-holder dashboard
.venv/bin/gpu-holder events --limit 30
```

停止：

```bash
.venv/bin/gpu-holder stop
```

`stop` 只会停止 pidfile 指向的 holder guard。如果 pidfile 指向其他进程，它会拒绝。

## 4. 配置文件

生成默认配置：

```bash
.venv/bin/gpu-holder init-config --stdout
.venv/bin/gpu-holder init-config --path gpu-holder.toml
```

生成配方：

```bash
.venv/bin/gpu-holder recipes
.venv/bin/gpu-holder recipes --name strict-quota
```

使用配置文件：

```bash
.venv/bin/gpu-holder preflight --config gpu-holder.toml
.venv/bin/gpu-holder guard --config gpu-holder.toml
```

配置文件是严格校验的。未知 key 会报错，不会静默忽略。

## 5. 临时让道

暂停所有 holder：

```bash
.venv/bin/gpu-holder pause --for 10m
```

恢复：

```bash
.venv/bin/gpu-holder resume
```

临时禁用某张卡：

```bash
.venv/bin/gpu-holder disable-gpu 7 --for 30m
```

重新启用：

```bash
.venv/bin/gpu-holder enable-gpu 7
```

这些命令只影响 holder 自己的 worker，不会处理外部训练或服务进程。

## 6. 观测与监控

单次状态：

```bash
.venv/bin/gpu-holder status --json
```

Prometheus text：

```bash
.venv/bin/gpu-holder metrics
.venv/bin/gpu-holder metrics --output /tmp/gpu-holder.prom
```

告警和 Grafana：

```bash
.venv/bin/gpu-holder alerts > gpu-holder-alerts.yml
.venv/bin/gpu-holder grafana-dashboard > gpu-holder-dashboard.json
.venv/bin/gpu-holder monitoring-bundle --output-dir /tmp/gpu-holder-monitoring
```

## 7. 常见问题

### 只有显存，没有利用率

使用 `--program matmul`，并把 duty cycle 固定到 `1`：

```bash
--program matmul --min-duty-cycle 1 --max-duty-cycle 1
```

### `doctor` 提示没有 pynvml

如果 `monitor_backend` 显示 `nvidia-smi fallback` 可用，可以先不装 `pynvml`。需要更完整 NVML 信息时再安装：

```bash
.venv/bin/python -m pip install -e ".[monitor]"
```

### `doctor` 提示没有 torch

真实 worker 需要 CUDA PyTorch。当前机器推荐用 Python 3.10 的 system site packages：

```bash
virtualenv --python /usr/bin/python3.10 --system-site-packages --clear .venv
```

如果完全隔离，需要按机器驱动版本安装 CUDA PyTorch wheel。

### `gpu-holder already running`

先看状态：

```bash
.venv/bin/gpu-holder status
```

如果确认是 holder 自己的 daemon：

```bash
.venv/bin/gpu-holder stop
```

不要手动 kill 其他 GPU 进程。

### 想确认 daemon 用的是哪个环境

```bash
pid=$(cat /root/.gpu-holder/gpu-holder.pid)
ps -fp "$pid"
readlink -f /proc/$pid/exe
tr '\0' '\n' </proc/$pid/environ | grep -E '^(PYTHONPATH|VIRTUAL_ENV)=' || true
```

用 `.venv/bin/gpu-holder` 启动时，解释器应来自 `/root/workspace/nene/gpu-holder/.venv/`，并且不需要 Shaft 的 `VIRTUAL_ENV`。
