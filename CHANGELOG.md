# 变更日志

`gpu-holder` 的用户可见变化记录在这里。项目仍处于 alpha 阶段，版本号遵循语义化版本。

## [未发布]

### 新增

- 智能策略引擎：按整机目标利用率调度，并支持单卡低利用率紧急介入。
- Holder 模式：`balanced`、`memory-only`、`compute-only`，以及内部 `assist` 模式。
- CUDA worker 程序：`matmul`、`conv`、`fft`、`elementwise`、`mixed`、`random`。
- `--program` 支持逗号分隔的自定义轮转，例如 `matmul,conv,fft`。
- 可配置计算 burst：`--compute-burst-seconds` 和 `--compute-burst-jitter`。
- CLI 状态快照：展示状态、事件、历史、解释和调参建议；交互式 TUI 已移除。
- 后台生命周期命令：`start`、`stop`、`status`、`events`、`pause`、`resume`、`disable-gpu`、`enable-gpu`。
- TOML 配置、严格 key 校验、`init-config`、`config-reference`、profile 和 recipe。
- `preflight` 启动前检查：配置、诊断、状态目录、pidfile、可见 GPU、策略决策和风险摘要。
- `plan`、`simulate`、`tune` 离线策略调试能力，不触碰真实 GPU worker。
- Prometheus metrics、告警规则、Grafana dashboard 和 monitoring bundle 生成。
- `doctor` 诊断 Python、监控后端、NVML、`nvidia-smi` 和 PyTorch CUDA。
- Shell completion、命令手册、支持报告、事件日志轮转和状态健康检查。
- `nvidia-smi` 监控 fallback，在缺少 Python NVML 绑定时仍可保持低依赖运行。
- `recipes` 提供 `first-run`、`strict-quota`、`busy-shared`、`compute-only` 等可复制部署模板。
- 打包 extras 拆分为：
  - `monitor`：仅安装 `nvidia-ml-py`
  - `torch`：安装 PyTorch
  - `cuda`：`monitor + torch`
- 基础包保持零三方依赖，并暴露 `gpu-holder` CLI entrypoint。
- `docs/tutorial.md` 提供中文详细教程。

### 安全

- 外部 GPU 进程是只读调度信号。`gpu-holder` 只启动和停止自己拥有的 worker。
- 前台 `guard` 和后台 `start` 使用一致的 pidfile 保护，避免重复控制器或覆盖非 holder 进程 pidfile。
- 控制器退出时只在 pidfile 仍指向当前进程时清理 pidfile。
- `start` 遇到 spawn 后 pidfile 冲突时，只终止刚创建的子进程。
- `start` 现在跟随当前 CLI 解释器启动后台进程，不再主动注入源码树 `PYTHONPATH`，避免继承无关项目环境。

### 修复

- `nvidia-smi` fallback 现在可以根据 GPU index 和显存占用识别命名空间隐藏的 holder worker PID，避免把自有 worker 误判为外部进程。
- 当滚动策略窗口仍低于目标利用率时，即使瞬时利用率高于目标，策略也会保持 holder，减少 matmul 高负载时的启停震荡。
- package metadata 使用兼容 setuptools 的 license table，使最小 virtualenv 中的 editable install 可以正常工作。
- Python 版本固定到 3.10 系列，并新增 `.python-version` 与 `uv.lock`。

## [0.1.0] - 未发布

- 初始 alpha 发布目标。
