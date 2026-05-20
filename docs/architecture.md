# 架构说明

`gpu-holder` 的架构目标是：策略可测试，运行态可控，依赖尽量少，安全边界清楚。

## 分层

```text
CLI
  解析参数、加载配置、调用 controller 或只读命令

config
  默认值、profile、recipe、TOML 加载、严格校验

monitor
  NVML / nvidia-smi 监控后端，产出统一 GPU snapshot

policy
  纯决策逻辑：hold / release / standby / assist

controller
  主循环、worker 生命周期、pidfile、状态文件、事件日志

worker
  CUDA 显存申请和计算负载

dashboard / metrics / diagnostics
  只读展示、监控导出和诊断
```

## 真源

- 配置真源：`GuardConfig`。
- 策略真源：`decide_for_fleet`。
- 运行状态真源：`status.json`。
- 事件真源：`events.jsonl`。
- worker 所有权真源：controller 内部 worker registry 和 pidfile。

## 监控后端

优先级：

1. NVML Python 绑定：可读字段更多，进程信息更完整。
2. `nvidia-smi` fallback：依赖少，适合受控机器。

fallback 后端需要处理 PID 命名空间差异。当前实现会用 GPU index 和显存占用辅助识别自有 worker，避免把 holder worker 误判为外部大进程。

## 策略模型

每轮采样后，controller 把 GPU snapshot、历史窗口、pause/disable 状态、worker 状态交给 policy。policy 返回每张卡的决策：

- `hold`：启动或保持 holder。
- `release`：释放自有 holder。
- `standby`：不启动 holder。
- `assist`：低显存介入，用于低利用率紧急场景。

核心优先级：

1. 手动 pause 或 disable 优先。
2. 温度限制优先。
3. protected process 强制让道。
4. 单卡低利用率持续超过窗口时介入。
5. 大显存外部进程通常让道。
6. 整机平均低于目标时补负载。

## worker

worker 是独立子进程。每个 worker：

- 绑定一张 GPU。
- 按配置申请显存。
- 按配置运行 CUDA 计算程序。
- 根据 duty cycle 控制忙/闲节奏。

真实高负载推荐使用 `matmul`，因为它最直接拉高 GPU 利用率。

## daemon 启动

`gpu-holder start` 使用当前 CLI 的 `sys.executable` 启动后台 `guard`，不主动注入源码树 `PYTHONPATH`。因此推荐通过独立 `.venv/bin/gpu-holder` 启动，daemon 就会跟随该虚拟环境。

## 安全边界

controller 只管理自己创建的 worker。外部 PID 只用于：

- 判断是否让道。
- 生成状态、事件、报告和监控指标。

禁止任何外部进程控制逻辑进入代码路径。
