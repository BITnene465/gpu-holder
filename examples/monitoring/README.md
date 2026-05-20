# gpu-holder 监控示例

这些示例把 `gpu-holder metrics` 接入标准 Prometheus 体系。它们只读取状态并导出指标，不会启动或停止 holder worker。

## node_exporter textfile collector

1. 启用 node_exporter textfile collector，并指定目录，例如：

   ```bash
   node_exporter --collector.textfile.directory=$HOME/.gpu-holder/node_exporter
   ```

2. 安装示例 user timer：

   ```bash
   mkdir -p ~/.config/systemd/user ~/.gpu-holder/node_exporter
   cp examples/monitoring/gpu-holder-metrics.service ~/.config/systemd/user/
   cp examples/monitoring/gpu-holder-metrics.timer ~/.config/systemd/user/
   systemctl --user daemon-reload
   systemctl --user enable --now gpu-holder-metrics.timer
   ```

3. 使用 `prometheus.yml` 中的 scrape 示例，或把其中的 `scrape_configs` 合并到已有 Prometheus 配置。

## 告警和 dashboard

从当前 CLI 生成告警规则和 Grafana dashboard，避免手写配置和指标漂移：

```bash
gpu-holder alerts > gpu-holder-alerts.yml
gpu-holder grafana-dashboard > gpu-holder-dashboard.json
```

告警覆盖：状态缺失或 stale、quota forecast 不可恢复、目标利用率缺口、worker 启动 backoff、温度让道。
