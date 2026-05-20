# gpu-holder

`gpu-holder` 是一个很小的 NVIDIA GPU 占卡工具：机器空闲时按单卡启动 CUDA worker，把每张卡的利用率维持到目标强度附近；真实训练、推理或服务进程出现时释放或降级为 assist。

它的安全边界很窄：

- 只管理自己启动的 guard / worker。
- 外部 GPU 进程只作为只读调度信号。
- 不 kill、不 suspend、不 renice 外部任务。
- 不维护配置系统、监控导出、外部面板、策略解释器或 TUI。

## 安装

基础包没有运行时三方依赖。状态读取依赖系统里的 `nvidia-smi`；真实 CUDA worker 依赖当前环境可用的 PyTorch。

```bash
cd /root/workspace/nene/gpu-holder
uv sync --python 3.10.11 --extra dev
uv run gpu-holder doctor
```

复用宿主机已有 CUDA PyTorch 时，可以创建带 system site packages 的环境：

```bash
cd /root/workspace/nene/gpu-holder
virtualenv --python /usr/bin/python3.10 --system-site-packages --clear .venv
.venv/bin/python -m pip install --no-build-isolation -e .
.venv/bin/gpu-holder doctor
```

可选依赖：

```bash
python -m pip install -e ".[torch]"  # 从当前 pip 源安装 PyTorch
python -m pip install -e ".[dev]"    # 测试、lint、构建工具
```

## 使用

前台指定 8 张卡运行：

```bash
gpu-holder guard --gpus 0-7 --risk-util 0.5 --target-util 0.75 --mem 0.2
```

后台运行：

```bash
gpu-holder start --gpus 0-7 --risk-util 0.5 --target-util 0.75 --mem 0.2
gpu-holder status
gpu-holder dashboard
gpu-holder stop
```

只看本轮决策，不启动 worker、不写状态：

```bash
gpu-holder guard --gpus 0-7 --risk-util 0.5 --target-util 0.75 --mem 0.2 --dry-run
```

跑一轮真实 guard 循环后退出，适合 smoke check：

```bash
gpu-holder guard --gpus 0-7 --risk-util 0.5 --target-util 0.75 --mem 0.2 --once
```

`--risk-util`、`--target-util` 和 `--mem` 都使用 `0~1` 浮点数：

- `--gpus 0-7` 表示使用 0 到 7 号卡；也支持 `all` 和 `0-3,6,7`。
- `--risk-util 0.5` 表示单卡利用率低于 50% 时开始占卡。
- `--target-util 0.75` 表示每张卡上的 holder 目标利用率，worker 会用短周期矩阵乘法 duty 控制接近该目标。
- `--mem 0.2` 表示最多使用单卡显存的 20%。
- 兼容旧写法：`20%`、`75%`、`75` 仍会被解析成 `0.2`、`0.75`、`0.75`。

默认策略：

- 单卡风险阈值：`0.5`
- 单卡 holder 目标利用率：`0.75`
- 默认最小 duty：`0.0`
- 默认计算周期：`0.2s`，用于减少 `nvidia-smi` 采样抖动
- 默认显存占用：`0.2`
- 显存预留：`2GiB`
- 大进程让道阈值：`10GiB`
- 大进程启动观察窗口：`120s`
- 温度释放阈值：`85C`

占卡启动依据是单卡利用率，不是整机平均利用率。某张卡低于 `--risk-util` 时会启动 holder；holder 启动后不会因为达到 `--target-util` 自动释放，避免 start/stop 抖动。

外部 CUDA 进程优先级最高。占用显存超过 `--busy-process-mem-threshold` 的训练进程出现在某张卡上时，`gpu-holder` 会先释放该卡并观察 `--process-grace-window` 秒，默认 120 秒；窗口结束后，如果这张卡仍低于 `--risk-util`，才会用 assist 模式补利用率。训练进程消失后，如果这张卡低于 `--risk-util`，下一轮调度会立刻重新占卡。

## 命令

当前只保留 6 个命令：

```bash
gpu-holder guard      # 前台运行
gpu-holder start      # 后台运行
gpu-holder stop       # 快速停止后台 guard
gpu-holder status     # 打印最近状态
gpu-holder dashboard  # 打印一次紧凑状态快照
gpu-holder doctor     # 检查 nvidia-smi 和 PyTorch CUDA
```

`stop` 会先发 `SIGTERM`，短时间未退出则发 `SIGKILL`。pidfile 指向非 holder 进程时会拒绝操作。

## 项目结构

```text
src/gpu_holder/
  cli.py       # CLI、nvidia-smi 读取、调度循环
  worker.py    # CUDA worker
  __init__.py
  __main__.py
```

## 开发与测试

```bash
python -m pytest -q
python -m ruff check --no-cache src tests
PYTHONPYCACHEPREFIX=/tmp/gpu-holder-pycache python -m compileall -q src tests
python -m build
```
