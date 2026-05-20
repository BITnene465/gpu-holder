# Requirements

## Problem

Some shared GPU machines are reclaimed when their hourly average GPU utilization
falls below a threshold such as 70%. A simple memory holder is not enough:

- memory-only holders do not increase utilization
- compute-only holders can interfere with real jobs
- fixed-load holders waste resources when real jobs already keep the machine busy

`gpu-holder` should keep machine utilization above a safe target while yielding
to real workloads.

## Functional Requirements

- Monitor NVIDIA GPU utilization, memory, and per-process memory with NVML
  when bindings are installed.
- Fall back to `nvidia-smi` monitoring when NVML Python bindings are missing,
  so the runtime can stay dependency-light on managed machines.
- Deduplicate NVML process rows by PID before busy-process policy decisions.
- Start holder workers on selected GPUs.
- Allocate GPU memory with a configurable target:
  - default: `20%`
  - examples: `70%`, `10GiB`, `12000MiB`
  - cap allocations by current free memory after reserve
- Generate compute utilization with multiple CUDA programs:
  - matrix multiplication
  - convolution
  - FFT
  - elementwise workloads
  - mixed rotation
  - random selection
  - user-defined comma-list rotation
- Default target machine utilization: `75%`.
- Fleet-level policy should avoid starting new holders when the current machine
  average is already above target, unless a GPU is in low-util emergency.
- Duty cycle should adapt to current utilization, per-GPU history, and machine
  average instead of using a fixed busy/sleep ratio.
- If a GPU stays below `50%` utilization for a full `60s` window, intervene.
- If a non-holder process uses more than `10GiB`, normally yield that GPU.
- If a non-holder process name matches `protected_process_patterns`, always
  yield that GPU.
- Low-util emergency intervention has higher priority than busy-process yield.
- When emergency intervention happens on a busy GPU, use assist mode with
  smaller memory allocation.
- Read GPU temperature from NVML when available, and release/standby at or above
  a configurable thermal limit.
- Keep a thermal block active until a lower configurable resume temperature is
  reached, avoiding worker oscillation near the limit.
- Support foreground `tmux` usage.
- Support lightweight daemon commands: `start`, `stop`, `status`.
- Support runtime per-GPU disable and temporary auto-expiring per-GPU disable
  for manual jobs.
- Generate a systemd user service unit without installing or enabling it.
- Report status age and stale state so users can detect a stopped or stuck
  controller from `status` and the TUI dashboard.
- Provide a script-friendly status health check with non-zero exit codes for
  missing or stale status files.
- Provide an optional utilization-target health check so scripts can fail when
  the policy/window average is below the configured reclaim target.
- Expose policy-window sample count and coverage so operators can distinguish
  a warmed-up utilization window from startup data.
- Forecast the remaining-window average utilization required to hit the target
  and flag windows that are no longer recoverable.
- Export the latest status as Prometheus text for external monitoring stacks,
  including atomic node_exporter textfile output.
- Generate Prometheus alerting rules for stale status, quota forecast risk,
  target gaps, worker startup backoff, and thermal yielding.
- Generate an importable Grafana dashboard JSON for the exported Prometheus
  metrics.
- Generate a monitoring bundle directory containing alert rules, dashboard JSON,
  and import notes.
- Show utilization-target health in the dashboard header so quota risk is
  visible during terminal monitoring.
- Provide a TUI dashboard for terminal environments.
- Provide a non-interactive dashboard snapshot for SSH logs, issue reports, and
  terminals where curses is not appropriate.
- Let the TUI temporarily pause all holders for a bounded manual-work window.
- Let the TUI temporarily disable the selected GPU for a bounded manual-work
  window.
- Let the TUI show the resolved runtime config without leaving the dashboard.
- Let the TUI filter recent events to the selected GPU for focused debugging.
- Let the TUI show a recent history summary without leaving the dashboard.
- Let the TUI show current explanation findings and guidance without leaving
  the dashboard.
- Let the TUI show offline tuning advice for the resolved runtime config
  without leaving the dashboard.
- Let the TUI detail view explain the selected GPU decision reason and show
  operator guidance from the shared reason reference.
- Provide `plan` / dry-run output for policy debugging.
- Provide a full startup preflight command that validates config, diagnostics,
  runtime state directory readiness, pidfile readiness, visible GPU selection,
  and would-be policy decisions without starting workers.
- Provide a reason reference so policy decision strings are explainable without
  reading source code.
- Provide offline policy simulation over time-series traces.
- Provide read-only runtime diagnostics with actionable suggestions for missing
  NVML bindings, NVML runtime failures, and PyTorch CUDA issues.
- Include a compact support report summary with target health, action counts,
  decision reasons, event types, workers, and external process counts.
- Provide a redacted support report mode for public issue reports that hides
  local paths and process identity fields.
- Require worker processes to report readiness within a configurable startup
  timeout before they are considered running.
- Allow event log inspection to filter by event type and GPU index for long
  tmux/daemon runs.
- Provide a read-only history summary command for recent event action/reason
  counts, per-GPU worker churn, and thermal event debugging.
- Provide a read-only explain command that turns current status and recent
  events into operator findings with guidance.
- Event log readers should tolerate corrupt or partial JSONL lines.
- Support TOML config files for long-running deployments, with CLI values
  overriding file values.
- Support documented runtime profiles as default layers for common deployment
  modes: balanced, conservative, quota, and compute-only.
- Support scenario-oriented config recipes that print copyable TOML for common
  deployments such as first rollout, strict quota, busy shared machine, and
  compute-only usage.
- Print the default TOML config without writing files for previews and docs.
- Generate compact profile-friendly TOML files so profile defaults are not
  accidentally overridden by explicit default fields.
- Explain resolved config precedence, including which profile defaults are
  applied or overridden by TOML and CLI flags.
- Provide a config validation/preview command before starting long-running
  guard processes.
- Reject unknown TOML config keys so misspelled settings cannot silently fall
  back to defaults.
- Provide a documented config reference command with JSON output for generated
  docs and integrations.
- Provide generated shell completion scripts for common terminal shells.
- Provide a generated command manual suitable for Markdown docs and manpage
  packaging.
- Support hold modes:
  - `balanced`
  - `memory-only`
  - `compute-only`
  - internal `assist`

## Non-Goals

- Do not kill user processes.
- Treat external process PIDs as read-only scheduling signals.
- Do not require systemd.
- Do not hide GPU usage.
- Do not support non-NVIDIA GPUs in the first release.
- Do not promise exact per-second utilization; use rolling windows and duty
  cycle control.

## Defaults

| Setting | Default |
| --- | --- |
| target utilization | `75%` |
| low-util emergency threshold | `50%` |
| low-util window | `60s` |
| memory hold | `20%` |
| busy process threshold | `10GiB` |
| protected process patterns | `[]` |
| reserve memory | `2GiB` |
| assist memory | `512MiB` |
| max GPU temperature | `85C` |
| thermal resume temperature | `80C` |
| sample interval | `2s` |
| program | `mixed` |
| compute burst | `0.20s` |
| compute burst jitter | `0.20` |
| worker duty update threshold | `0.05` |
| min duty cycle | `0.10` |
| max duty cycle | `1.0` |
| duty weights | `0.45 / 0.25 / 0.30` |
