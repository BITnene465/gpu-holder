# gpu-holder 开发日志

这里记录实现决策、验证结果和线上排障经验，避免只留在聊天记录里。

## 2026-05-20

### 初始产品范围

- 目标：构建一个共享 NVIDIA 机器上的礼貌占卡工具，在空闲时维持整机小时平均利用率，真实业务出现时让道。
- 默认策略：
  - 目标整机利用率：`75%`
  - 低利用率阈值：`50%`
  - 低利用率窗口：`60s`
  - 默认显存占用：`20%`
  - 大进程让道阈值：`10GiB`
- 安全不变量：外部 GPU 进程只读。工具只能停止自己创建的 worker。

### 架构决策

- 策略逻辑放在 `policy`，保持纯函数、易测试。
- 运行态进程控制放在 `controller`。
- CUDA 显存和计算负载隔离到每张卡的 worker 子进程。
- 配置使用严格 TOML 校验，拼写错误在启动 worker 前失败。
- completion、manual、reference 尽量从共享元信息生成，减少文档漂移。
- 增加 recipe，因为真实部署更需要可复制模板，而不是抽象 profile 名称。

### GPU smoke 结果

- 释放旧的手动 `hold.py` tmux 任务时，只向该 tmux 窗口发送 `Ctrl-C`，没有 kill 无关进程。
- 在 cuda:7 做短时真实 worker smoke：
  - 观察到约 `1650 MiB` 显存占用。
  - 观察到约 `93%` GPU 利用率。
  - 观察到约 `333W` 功耗。
  - 停止时只停止 smoke 自己启动的 worker。
- smoke 后确认 cuda:7 回到空闲。

### 依赖经验

- 现象：机器有 CUDA PyTorch，但缺少 `pynvml` / `nvidia-ml-py`，原始 `guard`/`doctor` 路径过于依赖 NVML Python 绑定。
- 修复：增加 `nvidia-smi` fallback，并让 `doctor` 区分必须可用的监控后端和可选 NVML 绑定。
- 后续防线：PyTorch 和 NVML 都保持 optional extras，因为 CUDA/PyTorch 版本通常由宿主机器决定。

### 高负载 matmul 修复

- 现象：用户观察到 holder worker 存在，但 GPU 利用率经常掉到 `0%`。
- 根因：
  - 当前宿主上 `nvidia-smi` 返回的 GPU PID 和控制器命名空间 PID 不一致，fallback 监控误把自有 worker 当作外部进程，从而触发释放。
  - `target_util = 75` 时，matmul 瞬时利用率很快超过目标，旧策略会释放 holder，导致震荡。
- 修复：
  - fallback 监控用 GPU index + 显存占用匹配自有 worker。
  - 滚动策略窗口低于目标时保持 holder。
  - 强制高负载场景建议设置 `target_util = 95`、`min_duty_cycle = max_duty_cycle = 1`、`process_grace_window = 0`。
- 验证：
  - 重启 matmul-only full-duty 配置。
  - 8 张卡约 `85-87%` 利用率，每张约 `16.5GiB` 显存。

### 独立 CLI 打包

- 现象：从源码树启动 daemon 时继承了 Shaft 虚拟环境，并且 `start` 主动注入源码树 `PYTHONPATH`。
- 根因：`start` 虽然使用 `sys.executable -m gpu_holder.cli`，但又把本地 `src` prepend 到 `PYTHONPATH`。
- 修复：
  - 基础包依赖保持空。
  - extras 拆成 `monitor`、`torch`、`cuda`。
  - `start` 跟随当前 CLI 解释器，不再主动改 `PYTHONPATH`。
- 验证：
  - 新增单测检查 `start` 使用当前解释器且不修改 `PYTHONPATH`。
  - 使用独立 `.venv` 安装 CLI。
  - Python `3.10.11` 环境可复用宿主 `torch=2.5.1+cu121`。
