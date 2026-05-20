# gpu-holder

`gpu-holder` 是一个面向共享 NVIDIA 机器的“礼貌占卡”工具。它会在机器空闲时用 CUDA 计算把 GPU 平均利用率维持在目标线以上，同时在真实训练、推理或服务进程出现时主动让道。

这个项目的目标很明确：

- 默认保持整机平均 GPU 利用率不低于回收阈值，例如 `70%`。
- 默认只占 `20%` 显存，避免把整张卡锁死。
- 当某张卡出现大显存真实进程时，默认释放该卡。
- 当某张卡持续低利用率超过窗口时，优先介入补负载。
- 不 kill、不 suspend、不 renice 任何外部 GPU 进程。
- 基础包零三方依赖，后续可以直接发布为 CLI 工具。

## 当前默认策略

- 目标整机平均利用率：`75%`
- 低利用率阈值：`50%`
- 低利用率判定窗口：`60s`
- 统计窗口：`3600s`
- 默认显存占用：`20%`
- 大进程让道阈值：`10GiB`
- 显存预留：`2GiB`
- 温度释放阈值：`85C`
- 温度恢复阈值：`80C`

低利用率介入优先级高于大进程让道：如果某张卡连续 `60s` 低于 `50%`，即使卡上已有大进程，`gpu-holder` 也会以更小显存的 assist 模式补负载，尽量帮助整机平均利用率达标。

## 安装

基础安装不引入任何三方依赖。监控可以使用系统自带的 `nvidia-smi`，真实 CUDA worker 会使用当前环境中已有的 PyTorch。

项目 Python 版本固定为 `3.10.11`，包运行约束为 `>=3.10,<3.11`，依赖解析结果记录在 `uv.lock`。

如果使用 `uv`：

```bash
cd /root/workspace/nene/gpu-holder
uv sync --python 3.10.11 --extra dev
uv run gpu-holder doctor
```

如果要复用宿主机已经装好的 CUDA PyTorch，推荐在项目内创建独立 virtualenv，避免混入其他项目环境：

```bash
cd /root/workspace/nene/gpu-holder
virtualenv --python /usr/bin/python3.10 --system-site-packages --clear .venv
.venv/bin/python -m pip install --no-build-isolation -e .
.venv/bin/gpu-holder doctor
```

说明：

- `--system-site-packages` 用来复用宿主机已经可用的 CUDA PyTorch。
- 如果希望完全隔离，可以去掉 `--system-site-packages`，然后单独安装适配本机驱动的 PyTorch CUDA wheel。
- 当前机器上已验证：独立 `.venv` 使用 Python `3.10.11`，可复用 `torch=2.5.1+cu121`。

可选 extras：

```bash
python -m pip install -e ".[monitor]"  # 安装 nvidia-ml-py
python -m pip install -e ".[torch]"    # 从当前 pip 源安装 PyTorch
python -m pip install -e ".[cuda]"     # monitor + torch
python -m pip install -e ".[dev]"      # 测试、lint、构建工具
```

## 快速使用

前台运行，适合挂在 tmux 中：

```bash
gpu-holder guard
```

后台守护进程：

```bash
gpu-holder start
gpu-holder status
gpu-holder dashboard
gpu-holder stop
```

强制使用矩阵乘法把 8 张卡都拉高负载：

```bash
gpu-holder start \
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

只看策略，不启动 worker：

```bash
gpu-holder plan --fake
gpu-holder preflight --fake --no-diagnostics
gpu-holder tune --strict
```

生成配置：

```bash
gpu-holder init-config --stdout
gpu-holder init-config --path gpu-holder.toml
gpu-holder recipes
gpu-holder recipes --name strict-quota
```

暂停和恢复：

```bash
gpu-holder pause --for 10m
gpu-holder resume
gpu-holder disable-gpu 7 --for 30m
gpu-holder enable-gpu 7
```

导出监控：

```bash
gpu-holder metrics
gpu-holder alerts > gpu-holder-alerts.yml
gpu-holder grafana-dashboard > gpu-holder-dashboard.json
gpu-holder monitoring-bundle --output-dir /tmp/gpu-holder-monitoring
```

## CLI 状态快照

当前版本移除了交互式 TUI，只保留 CLI。`dashboard` 会打印一次文本快照，适合 tmux、日志和远程 SSH：

```bash
gpu-holder dashboard
```

```bash
gpu-holder dashboard --once
gpu-holder dashboard --once --history --explain --advice
```

## 安全边界

`gpu-holder` 只会管理自己启动的 worker 进程。外部 GPU 进程只作为调度信号读取：

- 可以读取外部 PID 的显存、进程名、GPU 归属。
- 可以因为外部进程出现而释放自己的 worker。
- 不可以 kill 外部训练或服务进程。
- 不可以释放别的进程显存。
- 不可以通过 shell 调用破坏性进程管理命令。

`gpu-holder stop` 也会检查 pidfile 指向的是否是自己的 guard 进程；如果 pidfile 属于非 holder 进程，会拒绝操作。

## 项目结构

```text
src/gpu_holder/
  cli.py              # CLI 入口
  config.py           # 配置、profile、recipe
  policy.py           # 纯策略决策
  controller.py       # 进程与运行态编排
  worker.py           # CUDA worker
  monitor.py          # NVML / nvidia-smi 监控
  dashboard.py        # CLI 状态快照格式化
  diagnostics.py      # doctor / preflight 诊断
docs/
  tutorial.md         # 详细教程
  requirements.md     # 需求说明
  architecture.md     # 架构说明
  testing.md          # 测试说明
  release.md          # 发布检查清单
```

## 开发与测试

```bash
uv lock --check
python -m pip install --no-build-isolation -e .
python -m pytest -q
python -m ruff check --no-cache src tests
PYTHONPYCACHEPREFIX=/tmp/gpu-holder-pycache python -m compileall -q src tests
```

打包检查：

```bash
python -m build
```

## 详细教程

从安装、配置、前台日志、后台运行、监控接入到故障排查，见：

- [docs/tutorial.md](docs/tutorial.md)
