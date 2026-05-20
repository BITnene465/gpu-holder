# gpu-holder monitoring examples

These examples wire `gpu-holder metrics` into a standard Prometheus stack
without starting or stopping holder workers.

## Node exporter textfile collector

1. Enable the node_exporter textfile collector and point it at a directory, for
   example:

   ```bash
   node_exporter --collector.textfile.directory=$HOME/.gpu-holder/node_exporter
   ```

2. Install the example user timer:

   ```bash
   mkdir -p ~/.config/systemd/user ~/.gpu-holder/node_exporter
   cp examples/monitoring/gpu-holder-metrics.service ~/.config/systemd/user/
   cp examples/monitoring/gpu-holder-metrics.timer ~/.config/systemd/user/
   systemctl --user daemon-reload
   systemctl --user enable --now gpu-holder-metrics.timer
   ```

3. Import the Prometheus scrape example in `prometheus.yml`, or merge the
   `scrape_configs` into your existing Prometheus config.

## Alerts and dashboard

Generate alert rules and dashboard JSON from the current CLI so they stay in
sync with exported metrics:

```bash
gpu-holder alerts > gpu-holder-alerts.yml
gpu-holder grafana-dashboard > gpu-holder-dashboard.json
```

The alert rules cover missing/stale status, unrecoverable quota forecast,
target gap, worker startup backoff, and thermal yielding.
