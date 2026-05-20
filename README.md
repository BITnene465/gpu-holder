# gpu-holder

`gpu-holder` 是一个很小的 NVIDIA GPU 占卡工具：机器空闲时启动自己的 CUDA worker，把平均利用率维持到目标线附近；真实训练、推理或服务进程出现时释放自己的 worker。

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
gpu-holder guard --gpus 0-7 --mem 0.2 --target-util 0.75
```

后台运行：

```bash
gpu-holder start --gpus 0-7 --mem 0.2 --target-util 0.75
gpu-holder status
gpu-holder dashboard
gpu-holder stop
```

`--mem` 和 `--target-util` 都使用 `0~1` 浮点数：

- `--gpus 0-7` 表示使用 0 到 7 号卡；也支持 `all` 和 `0-3,6,7`。
- `--mem 0.2` 表示最多使用单卡显存的 20%。
- `--target-util 0.75` 表示目标整机平均利用率 75%。
- 兼容旧写法：`20%`、`75%`、`75` 仍会被解析成 `0.2`、`0.75`、`0.75`。

默认策略：

- 目标整机平均利用率：`0.75`
- 释放缓冲：目标线以上 `0.03`
- 默认显存占用：`0.2`
- 显存预留：`2GiB`
- 大进程让道阈值：`10GiB`
- 低利用率阈值：`50%`
- 温度释放阈值：`85C`

释放缓冲的作用是避免刚到目标线就释放 worker，导致最终平均利用率总是略低于 `--target-util`。已有 holder worker 时，工具会等平均利用率超过 `target + 0.03` 后再释放。

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
