# gpu-holder

`gpu-holder` is a polite GPU guard for shared NVIDIA machines. It keeps machine
GPU utilization above a target level when the machine is idle, while yielding to
real workloads when they start using significant memory.

The default policy is conservative:

- target average GPU utilization: `75%`
- emergency low-util threshold: `50%`
- emergency window: `60s`
- machine utilization window: `3600s`
- default memory hold: `20%`
- busy process threshold: `10GiB`
- reserve memory: `2GiB`
- thermal release limit: `85C`
- thermal resume limit: `80C`

If a GPU stays below `50%` utilization for the full `60s` window, `gpu-holder`
intervenes even when a large process exists on that GPU. In that case it uses
assist mode with smaller memory allocation to reduce OOM risk.

## Install

Development install:

```bash
python -m pip install -e ".[cuda,dev]"
```

Runtime-only install:

```bash
python -m pip install -e ".[cuda]"
```

`torch` and `nvidia-ml-py` are optional package extras because CUDA/PyTorch
versions are machine-specific. The controller prefers NVML bindings when they
are installed, but can fall back to `nvidia-smi` for dependency-light
monitoring.

Shell completion:

```bash
gpu-holder completion bash > ~/.local/share/bash-completion/completions/gpu-holder
gpu-holder completion zsh > "${fpath[1]}/_gpu-holder"
gpu-holder completion fish > ~/.config/fish/completions/gpu-holder.fish
```

## Quick Start

Run in a tmux session:

```bash
gpu-holder guard
```

Use a config file for long-running machines:

```bash
gpu-holder init-config --stdout
gpu-holder init-config --path gpu-holder.toml
gpu-holder init-config --profile quota --path gpu-holder-quota.toml
gpu-holder init-config --minimal --stdout
gpu-holder recipes
gpu-holder recipes --name strict-quota
gpu-holder config --config examples/gpu-holder.toml
gpu-holder config --config examples/gpu-holder.toml --json
gpu-holder config --config examples/gpu-holder.toml --explain
gpu-holder tune --config examples/gpu-holder.toml
gpu-holder tune --config examples/gpu-holder.toml --json
gpu-holder preflight --config examples/gpu-holder.toml
gpu-holder preflight --config examples/gpu-holder.toml --json
gpu-holder preflight --fake --no-diagnostics --protected-process train.py --json
gpu-holder config-reference
gpu-holder config-reference --json
gpu-holder profile-reference
gpu-holder profile-reference --json
gpu-holder reason-reference
gpu-holder reason-reference --json
gpu-holder completion bash
gpu-holder manual
gpu-holder guard --config examples/gpu-holder.toml
gpu-holder plan --config examples/gpu-holder.toml --fake
```

`init-config --stdout` prints the default TOML without writing files.
`init-config` refuses to overwrite an existing file unless `--force` is passed.
Config files are strict: unknown keys are treated as errors so misspelled
settings do not silently fall back to defaults.

Profiles provide safer starting points. Profile values are defaults; explicit
TOML fields and CLI flags still win:

```bash
gpu-holder guard --profile conservative
gpu-holder guard --profile quota
gpu-holder guard --profile compute-only --program random
```

`init-config --profile <name>` writes a compact profile-friendly TOML so the
selected profile is not accidentally overridden by a full explicit template.
Use `init-config --minimal` when you want the same compact style for the
balanced profile.
Use `config --explain` to see which profile defaults are active and which ones
are overridden by TOML or CLI flags.
Use `tune` before a long run to score the resolved config against offline
policy scenarios and get concrete tuning recommendations without touching GPUs.
Use `recipes` when you want an opinionated TOML starting point for a real
deployment scenario such as `first-run`, `strict-quota`, `busy-shared`, or
`compute-only`.

```bash
gpu-holder recipes
gpu-holder recipes --json
gpu-holder recipes --name busy-shared > gpu-holder.toml
```

Equivalent defaults:

```bash
gpu-holder guard \
  --gpus all \
  --target-util 75 \
  --idle-util 50 \
  --idle-window 60 \
  --mem 20% \
  --reserve 2GiB \
  --busy-process-mem-threshold 10GiB \
  --protected-process "python*train.py" \
  --max-gpu-temp 85 \
  --thermal-resume-temp 80 \
  --sample-interval 2
```

Use more memory on idle GPUs:

```bash
gpu-holder guard --mem 70%
```

Run a terminal dashboard:

```bash
gpu-holder dashboard
gpu-holder dashboard --once
gpu-holder dashboard --once --events --event-gpu 0
gpu-holder dashboard --once --history
gpu-holder dashboard --once --explain
gpu-holder dashboard --once --advice
gpu-holder dashboard --once --config
gpu-holder dashboard --temporary-pause-duration 20m
gpu-holder dashboard --temporary-disable-duration 1h
```

Dashboard controls:

- `p`: pause holder workers by creating the pause file
- `P`: temporarily pause holder workers for the configured dashboard duration
- `r`: resume by removing the pause file
- `e`: show/hide recent events
- `y`: show/hide recent history summary
- `x`: show/hide current explanation and guidance
- `a`: show/hide offline tuning advice for the resolved config
- `f`: filter recent events/history to the selected GPU
- `c`: show/hide the resolved config from `status.json`
- `j` / `k`: move the selected GPU row
- `i` / `Enter`: show/hide details for the selected GPU
- `d`: toggle runtime disable for the selected GPU
- `D`: temporarily disable the selected GPU for the configured dashboard duration
- `h` / `?`: show/hide the dashboard help
- `q`: quit the dashboard

Or render live status while running the foreground guard:

```bash
gpu-holder guard --tui
```

Run as a lightweight daemon:

```bash
gpu-holder start
gpu-holder status
gpu-holder status --check
gpu-holder status --check --require-target
gpu-holder status --check --require-forecast
gpu-holder status --watch
gpu-holder status --stale-after 30
gpu-holder status --json
gpu-holder metrics
gpu-holder metrics --output /var/lib/node_exporter/textfile_collector/gpu-holder.prom
gpu-holder alerts
gpu-holder alerts --json
gpu-holder grafana-dashboard > gpu-holder-dashboard.json
gpu-holder monitoring-bundle --output-dir gpu-holder-monitoring
gpu-holder events
gpu-holder events --event-type decision --gpu 0
gpu-holder events --follow
gpu-holder events --json
gpu-holder history
gpu-holder history --since 1h
gpu-holder history --gpu 0 --json
gpu-holder explain
gpu-holder explain --json
gpu-holder disable-gpu 3
gpu-holder enable-gpu 3
gpu-holder stop
```

Generate a systemd user service unit without installing it:

```bash
gpu-holder service --config /path/to/gpu-holder.toml
gpu-holder service --config /path/to/gpu-holder.toml -- --gpus 0,1
```

Review the generated unit before copying it into
`~/.config/systemd/user/gpu-holder.service`.

For machine readers, `status --watch --json` and `events --follow --json`
emit one compact JSON object per line.
`metrics` exports the latest status as Prometheus text for scrape jobs or
node-local collectors. Use `metrics --output <path>` to atomically write a
Prometheus textfile for node_exporter textfile collection.
`alerts` prints Prometheus alerting rules for missing/stale status, quota
forecast risk, target gaps, worker startup backoff, and thermal yielding:

```bash
gpu-holder alerts > gpu-holder-alerts.yml
gpu-holder alerts --group-name shared-a100 --json
```

`grafana-dashboard` prints an importable Grafana dashboard JSON wired to the
same Prometheus metrics:

```bash
gpu-holder grafana-dashboard > gpu-holder-dashboard.json
gpu-holder grafana-dashboard --title "Shared A100s" --datasource Prometheus
```

See [examples/monitoring](examples/monitoring/) for a node_exporter textfile
collector timer and Prometheus scrape example.
Use `monitoring-bundle` when you want alert rules, dashboard JSON, and a short
import README written into one directory:

```bash
gpu-holder monitoring-bundle --output-dir gpu-holder-monitoring
```

`events` can filter long logs by event type and GPU:

```bash
gpu-holder events --event-type decision --gpu 0
gpu-holder events --event-type worker_start,worker_stop --json
gpu-holder events --follow --event-type thermal_limit_start --gpu 3
```

`history` summarizes recent event logs without touching workers. Use it when a
machine misses a quota window or keeps yielding to real workloads and you need a
quick reason/action breakdown:

```bash
gpu-holder history --since 1h
gpu-holder history --gpu 0 --json
```

`explain` combines the latest status and recent event history into operator
findings with guidance:

```bash
gpu-holder explain
gpu-holder explain --json
```

`status` marks status files as stale when their timestamp is older than
`--stale-after` seconds, defaulting to `15s`. Use `--stale-after 0` to disable
the stale flag. The dashboard shows the same age/stale state and utilization
target health in its header.
The status payload also reports policy-window sample count and coverage, so
operators can tell whether the rolling machine average represents a warmed-up
hourly window or just the first few samples after startup. The CLI and
dashboard show this as `window=<status> coverage=<percent>`, and Prometheus
exports matching `gpu_holder_machine_policy_window_*` gauges.
It also estimates the average utilization required over the remaining window to
hit the configured target. CLI, dashboard, and explain output show this as
`quota=<status> required=<percent>`; Prometheus exports
`gpu_holder_target_required_remaining_average_percent` and related forecast
gauges.
Use `status --check` in scripts or cron checks: it exits `0` for a fresh status
file, `1` when no status file exists, and `2` when the status file is stale or
invalid.
Add `--require-target` to also fail when the policy/window average utilization
is below the configured `target_util`, which is useful for quota/reclaim
watchdogs.
Add `--require-forecast` to fail only when the remaining-window forecast is
unrecoverable, which is useful for earlier warning without failing every
temporary dip.
When `--json` is used, missing or invalid status files are reported as JSON with
`ok=false` and `status_error`; `--check --json` also includes `status_check`.
For non-interactive SSH logs, cron checks, and issue reports,
`dashboard --once` renders the dashboard as plain text and exits without
entering curses or changing runtime state.

`guard` and `start` handle stale pid files conservatively. If the pid file
points to a dead process, it is removed before the controller starts. If it
points to a live process that is not `gpu-holder guard`, the command refuses to
overwrite it. Foreground `guard` pre-reserves the pid file with its own PID
before the controller loop starts, reducing duplicate-controller races when two
tmux commands are launched close together. On shutdown, the controller removes
the pid file only if it still points to its own PID. Daemon `start` also refuses
to overwrite a pid file that appears after spawning its child process, and it
terminates that just-created child on conflict.

Dry-run the policy without starting workers:

```bash
gpu-holder plan
gpu-holder plan --json
gpu-holder plan --fake --json
```

Simulate policy behavior across a time series without touching GPUs:

```bash
gpu-holder simulate
gpu-holder simulate --scenario thermal --json
gpu-holder simulate --scenario all --json
gpu-holder simulate --json
gpu-holder simulate --trace trace.json --json
```

Invalid or missing trace files return exit code `2`. With `--json`, the error
is reported as `{"ok": false, "error": "trace error: ..."}`.
Built-in scenarios are `mixed`, `idle`, `busy`, `thermal`, and `quota-risk`.
Use `--scenario all` to run every built-in scenario as an offline policy
regression before touching real GPUs.

Score a config against those scenarios and common shared-machine safety checks:

```bash
gpu-holder tune
gpu-holder tune --profile quota --json
gpu-holder tune --strict --json
gpu-holder tune --compare-profiles
gpu-holder tune --target-util 60 --program matmul
```

`tune` reports a `pass`, `warn`, or `fail` grade, a numeric score, per-check
results, recommendations, and the underlying simulation summary. It is fully
offline and never starts holder workers. Use `--strict` in CI when warnings
should block a rollout. Use `--compare-profiles` to score the built-in
profiles and get a recommended starting point.

Validate the resolved config before a long tmux or daemon run:

```bash
gpu-holder config --config examples/gpu-holder.toml
gpu-holder config --config examples/gpu-holder.toml --json
gpu-holder preflight --config examples/gpu-holder.toml
gpu-holder preflight --config examples/gpu-holder.toml --json
gpu-holder config-reference
gpu-holder config-reference --json
```

`guard`, `start`, and `plan` also run the same config validation before doing
work. Unknown TOML keys are rejected, including common misspellings, so a long
tmux or daemon run does not silently use an unintended default.

`preflight` is the safest full startup preview. It validates the resolved
config, optionally runs `doctor` diagnostics, reads current GPU snapshots, and
prints the policy decisions that would be requested without starting workers.
It checks whether `state_dir` is a usable runtime directory, or whether a
missing directory can be created from an accessible parent.
It also inspects the pid file in read-only mode, reporting existing
`gpu-holder` controllers, live non-holder pidfile conflicts, and stale pidfiles
before `guard` or `start` is attempted.
Use `--fake` for a completely offline scheduler check, and `--no-diagnostics`
when you only want config/snapshot/policy output. Preflight includes action and
reason counts plus a `protected_process_matches` risk when protected process
patterns would make GPUs yield.

`config-reference` prints the documented configuration schema with defaults,
CLI flags, value types, examples, and field descriptions. The JSON form is
intended for generated docs or shell integrations.

`reason-reference` explains every scheduler decision reason, including likely
actions and operator guidance.
The dashboard detail view reuses the same reference data, so pressing `i` on a
selected GPU shows both the reason meaning and the recommended operator action.

`manual` prints a generated command reference in Markdown by default, or a roff
manpage draft with `gpu-holder manual --format man`.

Diagnose runtime readiness:

```bash
gpu-holder doctor
gpu-holder doctor --json
```

`doctor` checks Python, the active monitoring backend, optional NVML bindings,
optional `nvidia-smi` fallback, and PyTorch CUDA readiness. Failed checks
include a `suggestion` field in JSON output and an actionable hint in text
output.

Build a read-only support bundle for bug reports:

```bash
gpu-holder report
gpu-holder report --json
gpu-holder report --redact --json
gpu-holder report --no-diagnostics --json
```

`report` reads `status.json`, `events.jsonl`, the pid file, and optional
diagnostics. It includes a compact `summary` with target health, policy-window
health, quota forecast, action counts, decision reason counts, event type
counts, worker count, and external process count. Use `--redact` before
attaching output to public issues; it hides local paths, PIDs, process names,
and PID lists while preserving the diagnostic shape. It does not start workers
or stop any process.

## Load Programs

The compute worker can rotate through multiple CUDA programs:

- `matmul`: dense matrix multiplication
- `conv`: convolution workload
- `fft`: FFT workload
- `elementwise`: elementwise arithmetic workload
- `mixed`: rotates through all of the above
- `random`: randomly picks one of the above each compute burst
- comma-list, for example `matmul,conv,fft`: rotates through that explicit
  program sequence

Example:

```bash
gpu-holder guard --program mixed --target-util 75
gpu-holder guard --program random --target-util 75
gpu-holder guard --program matmul,conv,fft --target-util 75
```

Compute bursts default to `0.20s` with `20%` symmetric jitter. Tune
`--compute-burst-seconds` and `--compute-burst-jitter` if you want smoother or
more varied load shaping while preserving the adaptive duty-cycle target.
The controller also avoids restarting an owned worker for tiny duty changes
below `--worker-update-duty-threshold` (default `0.05`) to reduce churn on
noisy utilization samples.

## Hold Modes

`gpu-holder` supports several hold modes:

- `balanced`: allocate memory and run compute load
- `memory-only`: reserve memory without compute load
- `compute-only`: run compute load without a large memory hold
- `assist`: internal mode used when a large process exists but the GPU has been
  below the low-util threshold for the full emergency window

Memory allocations are capped by current free memory after `--reserve` is
honored. In `memory-only` mode, if that cap would allocate zero bytes, the
scheduler releases or stands by instead of starting a no-op worker.

Examples:

```bash
gpu-holder guard --hold-mode balanced
gpu-holder guard --hold-mode compute-only --program fft
gpu-holder guard --hold-mode memory-only --mem 20%
gpu-holder guard --max-held-gpus 2
```

## Safety Model

`gpu-holder` never kills other processes. It only starts and stops its own
workers. This is a hard project invariant: external process PIDs are read-only
signals for scheduling decisions, never control targets.

Event log:

- `~/.gpu-holder/events.jsonl` records controller lifecycle, policy decision
  changes, and worker start/stop events for workers owned by `gpu-holder`.
- Failed worker startups are recorded as `worker_start_failed` events instead of
  crashing the controller.
- Repeated worker startup failures are throttled by `worker_restart_backoff`
  seconds per GPU to avoid tight retry loops.
- Worker startup must report CUDA readiness within `worker_start_timeout`
  seconds. A worker that stays alive but never reports readiness is stopped,
  recorded as a startup failure, and then governed by the restart backoff.
- New non-holder GPU processes trigger a `process_grace_start` event and a
  short `process_grace_window` yield period so user jobs can initialize before
  holder workers compete for resources.
- Process names matching `protected_process_patterns` / `--protected-process`
  are explicit protection signals; the holder releases that GPU even if the
  process has not crossed the busy memory threshold.
- If NVML reports a GPU temperature at or above `max_gpu_temp`, the holder
  releases that GPU and will not start a new worker there until it cools below
  `thermal_resume_temp`. Set `max_gpu_temp = 0` or `--max-gpu-temp 0` to disable
  this guard; set `thermal_resume_temp = 0` to resume as soon as the GPU falls
  below `max_gpu_temp`.
- Thermal blocks emit `thermal_limit_start` and `thermal_limit_end` events.
- `status.json`, `gpu-holder status`, and the dashboard show owned worker
  details, active per-GPU startup backoff timers, process grace windows, and
  runtime-disabled GPUs, including per-GPU temperature when NVML provides it.
- `status.json`, dashboard GPU details, and Prometheus metrics mark external
  processes that matched `protected_process_patterns`.
- Unexpected exits from owned workers are recorded as `worker_exit` events and
  the controller can start replacements on later policy decisions.
- The log is append-only JSONL so long-running tmux/daemon sessions can explain
  why a GPU was held, assisted, released, or left in standby.
- Event readers skip corrupt or partial JSONL lines so one bad log line does
  not break `events`, `dashboard --once`, or `report`.
- `status.json` is written through an atomic same-directory replace, preserving
  the previous status if a write is interrupted or replacement fails.
- Status readers report corrupt or partial `status.json` files as invalid
  status instead of raising a traceback; `report` includes the status read error.
- Event logs rotate by default at `10MiB` with 3 backups:
  `events.jsonl.1`, `events.jsonl.2`, `events.jsonl.3`.

Normal yielding:

- If a non-holder process uses more than `10GiB` on a GPU, the holder releases
  that GPU.
- If a non-holder process name matches `protected_process_patterns`, the holder
  releases that GPU. Patterns are case-insensitive shell-style globs, e.g.
  `python*train.py` or `vllm*`.
- NVML process rows are deduplicated by PID before policy sees them, so the
  same process reported by both compute and graphics APIs is counted once.
- If a GPU is disabled at runtime, the holder releases that GPU and skips new
  workers there until it is enabled again.
- If a new non-holder process appears on a GPU, the holder yields for
  `process_grace_window` seconds. Set it to `0` to disable this startup grace.
- If a GPU reaches `max_gpu_temp`, the holder releases that GPU before applying
  low-util emergency or busy-process rules. The thermal block stays active
  until the GPU cools to `thermal_resume_temp`, which prevents start/stop
  oscillation around the high-temperature boundary.
- If `max_held_gpus` is set, the scheduler caps concurrent holder workers and
  prioritizes emergency/low-util GPUs before starting extra holders.
- The scheduler uses `machine_window` as the rolling machine-average window for
  target decisions. This is intended for hourly utilization quotas.

Emergency intervention:

- If the same GPU remains below `50%` utilization for a full `60s` window,
  holder starts assist mode on that GPU even if a large process exists.
- Assist mode defaults to smaller memory allocation and mainly adds compute
  load.

Pause file:

```bash
gpu-holder pause
gpu-holder pause --for 10m
gpu-holder resume
```

When the pause file exists, `gpu-holder` releases its own workers and waits.
`pause --for` writes an auto-expiring pause state, so a temporary manual job or
debugging session does not leave the guard paused forever.
The same pause/resume actions are available from the TUI dashboard. In the
dashboard, `P` writes a temporary pause using `--temporary-pause-duration`,
defaulting to `10m`.

Per-GPU runtime disable:

```bash
gpu-holder disable-gpu 3
gpu-holder disable-gpu 3 --for 30m
gpu-holder enable-gpu 3
```

Use `disable-gpu --for <duration>` to temporarily reserve a GPU for a manual
job; the guard automatically drops the disable state after the duration expires.
`status`, the TUI dashboard, and Prometheus metrics show the expiration time for
temporary per-GPU disables.
The dashboard `d` key toggles the selected GPU using the same
`disabled_gpus.json` state file, and `D` temporarily disables it for the
`--temporary-disable-duration` window, defaulting to `30m`. Updates to that
file are written through an atomic same-directory replace, so an interrupted
toggle keeps the previous runtime disable state instead of leaving a partial
JSON file.

## Adaptive Load Control

`gpu-holder` does not simply run a fixed busy loop. The policy combines:

- current GPU utilization
- rolling per-GPU utilization history
- current machine-average utilization

to choose a duty cycle. This helps the guard raise utilization when the machine
is at risk, while backing off when the fleet is already above target.

The aggressiveness is configurable:

```bash
gpu-holder guard \
  --min-duty-cycle 0.10 \
  --max-duty-cycle 0.80 \
  --current-gap-weight 0.45 \
  --history-gap-weight 0.25 \
  --machine-gap-weight 0.30
```

The three weights must sum to `1.0`. Lower `max-duty-cycle` values make the
holder more conservative; higher `machine-gap-weight` values make it react more
to hourly fleet-average deficits.

## Safe Testing on Busy Machines

When other GPU jobs are already running, do not start real workers. Use:

```bash
gpu-holder plan --fake
gpu-holder plan --fake --json
gpu-holder preflight --fake --no-diagnostics
gpu-holder preflight --fake --no-diagnostics --json
gpu-holder doctor
python -m pytest
```

These tests exercise the scheduler without touching CUDA or existing GPU
processes.

For local source-tree testing without installation:

```bash
PYTHONPATH=src python -m gpu_holder --version
PYTHONPATH=src python -m gpu_holder plan --fake
PYTHONPATH=src python -m gpu_holder plan --fake --json
PYTHONPATH=src python -m gpu_holder preflight --fake --no-diagnostics
python -m pytest -q
```

See [docs/testing.md](docs/testing.md) for GPU smoke-test guidance.

## Project Maintenance

- See [CHANGELOG.md](CHANGELOG.md) for notable changes.
- See [docs/release.md](docs/release.md) for the release checklist.
- Use the GitHub issue templates for bug reports, feature requests, and runtime
  support so diagnostics and safety context are included.

## Project Status

This is an alpha implementation intended for real shared training machines. The
policy engine is unit-tested without GPUs; CUDA execution requires NVIDIA NVML
and PyTorch.
