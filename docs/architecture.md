# Architecture

`gpu-holder` is split into small layers so the policy can be tested without GPU
hardware.

## Layers

- `monitor`
  - NVML integration when `pynvml` / `nvidia-ml-py` is installed.
  - `nvidia-smi` fallback for dependency-light runtime monitoring.
  - Produces `GpuSnapshot` objects.
  - Deduplicates process rows by PID across NVML compute/graphics process APIs
    before policy consumes them.
- `history`
  - Maintains rolling utilization samples per GPU.
- `policy`
  - Pure scheduling logic.
  - Converts snapshots and history into `HolderDecision` objects.
  - Decision reason strings are documented by `reason-reference` for operator
    troubleshooting.
- `config`
  - Defines strict TOML/CLI config fields and runtime profiles.
  - Applies profiles as default layers before explicit config and CLI overrides.
  - Provides scenario-oriented recipes that generate copyable TOML without
    duplicating config schema rules.
  - Explains config precedence so profile defaults are auditable.
- `simulator`
  - Offline policy simulation from built-in scenarios or JSON traces.
  - Provides built-in `mixed`, `idle`, `busy`, `thermal`, and `quota-risk`
    scenarios so operators can exercise policy branches without touching GPUs.
  - Reuses the same history and fleet decision logic as the live controller.
  - Reports missing or invalid trace files as CLI errors instead of tracebacks;
    JSON mode returns an `ok=false` error payload.
  - Includes frame/GPU/process paths in trace schema errors.
- `advisor`
  - Scores a resolved config against the built-in offline scenarios.
  - Produces pass/warn/fail checks and tuning recommendations for target
    utilization, emergency intervention, busy-process yielding, thermal
    yielding, memory hold size, compute program diversity, and sample cadence.
  - Never reads live GPU state or starts workers.
- `worker`
  - PyTorch CUDA workloads and memory allocation.
  - Runs in one process per GPU.
- `controller`
  - Main loop.
  - Starts/stops workers, writes status JSON, handles pause and shutdown.
  - Logs worker startup failures and continues the control loop.
  - Reaps exited owned workers and can restart them on later policy decisions.
- `tui`
  - Curses-based terminal dashboard.
  - Provides a plain-text one-shot dashboard renderer for non-interactive
    terminals and support reports.
  - Provides keyboard controls for pause, resume, recent event visibility,
    recent history summary, current explanation/guidance, offline tuning
    advice, per-GPU disable, built-in help, selected-GPU details, and resolved
    config inspection.
  - Reuses the central decision reason reference in selected-GPU details so
    the dashboard and `reason-reference` command do not drift.
  - Keeps keyboard behavior in a testable dashboard state machine instead of
    hiding control logic inside the curses loop.
  - Can filter recent event display to the selected GPU.
  - Shows status age/stale state and utilization-target health so a stuck
    controller or quota risk is visible.
- `diagnostics`
  - Read-only Python, monitoring-backend, optional NVML, optional `nvidia-smi`,
    and PyTorch CUDA environment checks.
  - Returns actionable suggestions with failed checks so `doctor` and `report`
    can guide setup fixes without starting workers.
- `events`
  - Append-only JSONL runtime event logging.
  - Records lifecycle, decision changes, and owned worker start/stop events.
  - Supports read-side filtering by event type and GPU index for CLI readers.
  - Skips corrupt or partial JSONL lines on read so status tools keep working
    after interrupted writes or manual log edits.
- `history`
  - Builds read-only summaries from recent events for action/reason breakdowns,
    per-GPU worker churn, thermal events, and quota debugging.
- `explain`
  - Combines status health, per-GPU decisions, reason guidance, and recent
    history into read-only operator findings.
- `cli`
  - Argument parsing and commands.
  - Loads TOML config files and applies CLI overrides.
  - Validates resolved configs before long-running commands start.
  - Reports corrupt or partial `status.json` files as invalid status for
    `status`, `dashboard --once`, and `report` instead of raising tracebacks.
  - Rejects unknown TOML config keys so spelling mistakes fail before workers
    can start.
  - Generates deployment helpers such as a systemd user service unit without
    writing system state.
  - Provides `preflight` as a no-worker startup readiness report that combines
    config validation, optional diagnostics, runtime state directory readiness,
    pidfile readiness, visible GPU selection, policy decisions, and risk
    summaries.
  - Exposes a documented config-reference schema for humans and tooling.
  - Generates shell completion scripts for bash, zsh, and fish.
  - Generates Markdown and roff command manuals from shared command metadata.

## Scheduling Priority

1. Pause file releases all holders.
2. Runtime-disabled GPUs release holders and skip new workers.
3. New non-holder processes receive a short process-grace yield window.
4. Thermal limits release holders before new compute load is considered.
5. If a GPU has been below `idle-util` for the full `idle-window`, intervene.
6. If a busy non-holder process is above the memory threshold and the GPU is not
   in low-util emergency, release/yield.
7. If GPU utilization is below target, hold or continue holding.
8. Otherwise stay in standby.

External processes are never control targets. The controller may stop only
workers that it started itself.

## Adaptive Duty Cycle

Compute load is not fixed. The policy computes duty cycle from three signals:

- current GPU utilization gap
- per-GPU rolling history gap
- rolling machine-average utilization gap over `machine_window`

The controller still reports the instantaneous machine average, but the target
decision can use the rolling policy average. This is meant to match hourly
utilization quotas where one short dip should be treated differently from a
full-window deficit.

Emergency low-util GPUs receive a stronger minimum duty signal. `memory-only`
mode forces duty to zero; `compute-only` mode forces memory allocation to zero.
This keeps the controller useful for both "reserve memory" and "raise hourly
utilization" workloads.

The bounds and weights are part of the resolved config:

- `min_duty_cycle`
- `max_duty_cycle`
- `current_gap_weight`
- `history_gap_weight`
- `machine_gap_weight`
- `compute_burst_seconds`
- `compute_burst_jitter`
- `worker_update_duty_threshold`

The three weights must sum to 1.0. Validation warns when they do not, and
rejects negative weights or invalid duty bounds.
Each worker compute cycle uses `compute_burst_seconds` as the base active
window and applies symmetric `compute_burst_jitter` before computing the sleep
window needed to preserve the requested duty cycle.
When an owned worker is already running, the controller does not restart it for
duty-cycle changes smaller than `worker_update_duty_threshold`, reducing churn
from normal utilization noise.

If `max_held_gpus` is configured, the fleet policy caps concurrent `hold` and
`assist` decisions. The cap is applied after per-GPU decisions are computed, so
busy-process yielding and pause semantics remain local and explicit. When more
GPUs want holders than the cap allows, emergency decisions, higher duty cycles,
lower recent utilization, and already-running holders are preferred.

The controller also tracks non-holder GPU PIDs per GPU. When a new PID appears,
it records `process_grace_start` and passes that GPU to policy as a temporary
process-grace yield. This protects jobs during startup, before their memory use
crosses `busy_process_mem_threshold`. `process_grace_window = 0` disables this
behavior.

The policy also supports explicit protected process patterns. If a visible
non-holder process name matches `protected_process_patterns`, the GPU yields
with reason `protected_process` before busy-process and low-util emergency
rules are considered. This gives operators a precise way to protect known
training or serving commands that may use little memory but should not share
with holder work. Status payloads annotate each visible process with
`protected_process` and `protected_process_pattern`, so TUI details and metrics
can explain the match without reimplementing policy matching.

The controller also tracks thermal blocks per GPU. If NVML reports
`temperature_celsius >= max_gpu_temp`, the per-GPU decision becomes
`thermal_limit` and releases the owned worker or remains in standby. The block
stays active until the GPU cools to `thermal_resume_temp`, preventing a worker
from restarting every time temperature fluctuates around the release limit. This
check runs before low-util emergency intervention so utilization quotas never
override thermal protection. `max_gpu_temp = 0` disables this guard in
CLI/config parsing; `thermal_resume_temp = 0` disables hysteresis and resumes
once the temperature is below `max_gpu_temp`.

## Worker Modes

- `hold`
  - Used on idle/lightly-used GPUs.
  - Allocates the configured memory target and runs compute load.
- `memory-only`
  - Allocates memory without compute load.
  - If current free memory after reserve is zero, the scheduler releases or
    remains in standby with `insufficient_free_memory`.
- `compute-only`
  - Runs compute load without a large memory allocation.
- `assist`
  - Used when a GPU has a large process but is still low-util for the emergency
    window.
  - Allocates smaller `assist-mem` and runs compute load.
  - Memory allocation is capped by current free memory after reserve.
- `release`
  - Stop the holder process for the GPU.
- `standby`
  - Do nothing.

## Runtime State

The controller writes:

- `~/.gpu-holder/status.json`
- `~/.gpu-holder/events.jsonl`
- `~/.gpu-holder/events.jsonl.1` and later backups when event logs rotate
- `~/.gpu-holder/gpu-holder.pid`
- `~/.gpu-holder/gpu-holder.log` when launched with `start`
- `~/.gpu-holder/pause` can be created to pause all workers; plain text means an
  indefinite pause, while JSON with `pause_until` auto-expires
- `~/.gpu-holder/disabled_gpus.json` can disable individual GPUs at runtime;
  entries may be indefinite or auto-expiring

The dashboard can create/remove the pause file and toggle per-GPU runtime
disable using the same shared runtime state helpers as the CLI, so terminal
control and shell commands have identical semantics. It can also write bounded
temporary pause and per-GPU disable states using dashboard-configured durations.

Per-GPU disable updates are written to a temporary file in `state_dir` and then
atomically replaced. A failed or interrupted write keeps the previous
`disabled_gpus.json` content intact. Legacy list-form state files remain
readable, while temporary disables are stored with per-GPU `disabled_until`
timestamps and removed automatically after expiry. Status payloads also copy
each active GPU's expiration to `gpu_disabled_until` so CLI/TUI/metrics readers
do not need to understand the state-file schema.

`status.json` uses the same temporary-file and same-directory replace pattern,
so a failed status update preserves the previous complete status payload.

`events.jsonl` is size bounded by config. The controller rotates only its own
event log files inside `state_dir`; external process logs are never modified.

Controller pid files are conservative for both foreground `guard` and daemon
`start`:

- A pid file for an existing `gpu-holder guard` process prevents duplicate
  controllers.
- A pid file for a live non-holder process is never overwritten.
- A pid file for a dead process is treated as stale and can be removed by
  `guard` or `start` before the controller runs.
- Foreground `guard` creates the pid file with its current PID before entering
  the controller loop, so another foreground launch sees an existing controller
  instead of racing through an empty pidfile window.
- Daemon `start` records the spawned child PID without overwriting a pid file
  that appears in the spawn window; on conflict it terminates only the child it
  just created.
- On shutdown, the controller removes the pid file only if it still contains
  the current process PID.

`status.json` includes a fleet summary:

- GPU count
- current machine average utilization
- machine policy-window size, sample count, coverage seconds, and coverage
  ratio, so readers can distinguish a warmed-up quota window from startup data
- utilization target forecast, including the remaining-window average needed
  to hit the configured target and whether that is still reachable
- action counts by policy decision
- owned worker count and active worker startup backoff count
- per-GPU snapshots, processes, decisions, rolling history averages, and machine
  averages used by the policy
- per-GPU owned worker details and active worker startup backoff timers
- per-GPU active process grace timers
- per-GPU temperature, whether the configured thermal limit is currently
  exceeded, and whether thermal hysteresis is still blocking the GPU
- derived status file age and stale state for CLI/TUI readers

Machine-readable outputs:

- `gpu-holder plan --json`
- `gpu-holder preflight --json`
- `gpu-holder config --json`
- `gpu-holder config --explain --json`
- `gpu-holder config-reference --json`
- `gpu-holder profile-reference --json`
- `gpu-holder reason-reference --json`
- `gpu-holder completion bash`
- `gpu-holder manual --format man`
- `gpu-holder init-config --path gpu-holder.toml`
- `gpu-holder status --json`
- `gpu-holder status --check`
- `gpu-holder status --check --require-target`
- `gpu-holder status --check --require-forecast`
- `gpu-holder status --watch --json` as compact JSON Lines
- `gpu-holder metrics [--output path]` as Prometheus text or an atomically
  replaced node_exporter textfile
- `gpu-holder alerts [--json]` for Prometheus alerting rules built from the
  exported metrics
- `gpu-holder grafana-dashboard` for an importable Grafana dashboard JSON built
  from the exported metrics
- `gpu-holder monitoring-bundle --output-dir <dir>` to write alert rules,
  dashboard JSON, and import notes into one directory
- `gpu-holder events --json`
- `gpu-holder events --follow --json` as compact JSON Lines
- `gpu-holder events --event-type decision --gpu 0 --json`
- `gpu-holder history --since 1h --json`
- `gpu-holder explain --json`
- `gpu-holder doctor --json`
- `gpu-holder report --json` for issue-ready support bundles
- `gpu-holder report --redact --json` for public issue reports with local
  paths and process identity fields hidden
- `gpu-holder service --config /path/to/gpu-holder.toml` for systemd user unit
  generation

Support reports include a compact `summary` section with diagnostics status,
status presence, event type counts, machine utilization, policy-window health,
utilization-target health, quota forecast, action counts, decision reason
counts, worker count, holder PID count, external process count, and disabled GPU
count.

`status --json` and `status --watch --json` preserve machine-readable output on
missing or invalid status files by emitting `ok=false` with `status_error`.
`status --check --json` also includes a failing `status_check` object.
`status --check --require-target` extends that check to the configured
utilization target by comparing `config.target_util` with the machine
policy/window average stored in `status.json`.
`status --check --require-forecast` instead checks the remaining-window quota
forecast and fails only when the forecast is unrecoverable or complete-missed.

## Worker Startup

Worker processes report readiness back to the controller after CUDA device
selection and memory allocation. If startup fails, the controller records a
`worker_start_failed` event and does not mark that worker as running. This keeps
one bad GPU, OOM, missing PyTorch install, or driver failure from taking down the
whole controller loop.

Readiness is bounded by `worker_start_timeout`. If a child process remains alive
but never reports readiness, the controller stops that owned child, records a
startup failure, and applies the normal restart backoff instead of treating the
worker as healthy.

Repeated startup failures are rate limited per GPU by
`worker_restart_backoff`. While the backoff is active, the controller records a
single `worker_start_backoff` event and skips additional worker creation
attempts for that GPU. Release/standby decisions clear the backoff.

After startup, the controller checks only workers it owns. If one exits, it logs
`worker_exit`, removes it from the owned-worker set, and subsequent policy
decisions may start a replacement.
