# Changelog

All notable changes to `gpu-holder` will be documented in this file.

The format follows Keep a Changelog, and this project uses semantic versioning
while it remains in alpha.

## [Unreleased]

### Added

- Smart policy engine for target machine utilization with per-GPU low-util
  emergency handling.
- Holder modes: `balanced`, `memory-only`, `compute-only`, and internal
  `assist`.
- CUDA worker programs: `matmul`, `conv`, `fft`, `elementwise`, `mixed`, and
  `random`.
- `--program` also accepts comma-separated custom rotation lists such as
  `matmul,conv,fft`.
- Worker compute bursts are configurable with `--compute-burst-seconds` and
  `--compute-burst-jitter` for less mechanical load shaping.
- `--worker-update-duty-threshold` reduces owned-worker restart churn from
  small adaptive duty-cycle changes.
- Terminal dashboard with pause/resume, per-GPU disable, help, config, details,
  events, and selected-GPU event filtering.
- Dashboard header now shows utilization-target health and target gap alongside
  status age/stale state.
- Dashboard can show offline tuning advice for the resolved runtime config via
  the `a` key or `dashboard --once --advice`.
- Dashboard selected-GPU details now include decision reason meaning and
  operator guidance from the shared reason reference.
- Non-interactive `dashboard --once` snapshots for SSH logs and issue reports.
- Script-friendly `status --check` with non-zero exit codes for missing or
  stale status files.
- Systemd user service unit generation via `gpu-holder service`.
- Daemon lifecycle commands: `start`, `stop`, `status`, `events`, `pause`,
  `resume`, `disable-gpu`, and `enable-gpu`.
- TOML config loading, resolved config validation, `init-config`, and
  `config-reference`.
- Side-effect-free default config preview via `init-config --stdout`.
- Strict config-file key validation so misspelled TOML settings fail before
  workers can start.
- Startup `preflight` reports that validate config, optional diagnostics,
  visible GPU selection, policy decisions, and risk summaries without workers.
- `preflight` now checks runtime `state_dir` readiness, including existing
  non-directory paths, inaccessible directories, non-writable parents, and
  parent path components blocked by files.
- `preflight` now includes a read-only pidfile readiness check for existing
  controllers, live non-holder pidfile conflicts, invalid pidfiles, and stale
  pidfiles.
- Configurable worker readiness timeout; unready live workers are stopped and
  handled as startup failures instead of being marked healthy.
- Policy decision reason reference with operator guidance.
- Shell completion generation for bash, zsh, and fish.
- Generated Markdown/manual-page command reference.
- Source-tree module entrypoint via `python -m gpu_holder` plus top-level
  `--version`.
- Publishable package metadata, real project URLs, and CI package build
  verification.
- Event log rotation, event filtering, support reports, status stale detection,
  and actionable `doctor` diagnostics.
- Support reports now include a compact summary with target health, action
  counts, decision reason counts, event type counts, workers, and external
  process counts.
- `report --redact` hides local paths, PIDs, process names, and PID lists for
  public issue reports while preserving diagnostic structure.
- Event readers tolerate corrupt or partial JSONL lines.
- Status readers report corrupt or partial `status.json` files as invalid
  status, and support reports include the status read error.
- `status.json` is written with the shared atomic same-directory replace helper,
  preserving the previous status if replacement fails.
- `status --json`, `status --check --json`, and `status --watch --json` keep
  missing or invalid status-file errors machine-readable; `--check --json`
  includes a failing `status_check` object for both cases.
- `status --check --require-target` fails when the recorded policy/window
  average utilization is below the configured `target_util`, and status JSON
  includes `utilization_target_health`.
- `simulate --trace` reports missing or invalid trace files as clean CLI errors,
  with machine-readable `ok=false` payloads in JSON mode and path-qualified
  schema errors.
- `simulate --scenario all` runs every built-in offline scenario as one
  regression suite.
- `tune` scores the resolved config against offline scenarios and emits
  pass/warn/fail recommendations without touching live GPUs.
- `tune --strict` exits non-zero for warnings as well as failures, making the
  offline advisor usable as a CI or release gate.
- `tune --compare-profiles` scores every built-in runtime profile and
  recommends a starting profile using the same offline advisor.
- `recipes` lists scenario-oriented config recipes and prints copyable TOML for
  `first-run`, `strict-quota`, `busy-shared`, and `compute-only` deployments.
- NVML monitor deduplicates process rows by PID across compute and graphics
  process APIs before policy sees them.
- The monitor can fall back to `nvidia-smi` when NVML Python bindings are not
  installed, reducing runtime dependency friction on managed machines.
- `doctor` now reports the required monitoring backend separately from optional
  NVML and `nvidia-smi` checks.
- Thermal safety with release limit and resume hysteresis.
- Holder memory requests are capped by current free memory after reserve; a
  memory-only holder now releases/stands by instead of starting with zero
  allocatable memory.
- Per-GPU runtime disable state is written with atomic same-directory replace,
  preserving the previous state if an update is interrupted.
- `disable-gpu --for <duration>` temporarily disables a GPU and auto-expires
  without requiring a later `enable-gpu`.
- Status text, TUI labels/details, and Prometheus metrics expose temporary
  per-GPU disable expiration times.
- The TUI dashboard `D` key temporarily disables the selected GPU for the
  configurable `--temporary-disable-duration` window.
- The TUI dashboard `P` key temporarily pauses all holder workers for the
  configurable `--temporary-pause-duration` window.
- TUI help now shows the configured temporary pause/disable durations instead
  of hard-coded key descriptions.
- `protected_process_patterns` / `--protected-process` can explicitly protect
  known user workloads by process-name pattern, forcing the holder to yield
  even before the busy memory threshold is crossed.
- `preflight` now reports decision reason counts and highlights protected
  process matches before workers are started.
- Status payloads, TUI GPU details, and Prometheus metrics now mark which
  visible processes matched protected process patterns.
- Source-tree tests can run with plain `python -m pytest` from the project root.
- Local testing docs consistently use plain `python -m pytest -q` now that
  pytest config includes the source tree import path.

### Safety

- External GPU processes are read-only scheduling signals. `gpu-holder` starts
  and stops only worker processes that it owns.
- Foreground `guard` now applies the same conservative pidfile checks as
  daemon `start`, avoiding accidental duplicate controllers or overwriting a
  live non-holder pidfile.
- Foreground `guard` pre-reserves its pidfile before the controller loop starts,
  reducing duplicate-controller races between near-simultaneous tmux launches.
- Controller shutdown removes the pidfile only when it still points to the
  current process, avoiding accidental cleanup of a replaced pidfile.
- Daemon `start` no longer overwrites a pidfile that appears after spawning its
  child process; on conflict it terminates only that just-created child.

### Fixed

- `nvidia-smi` fallback monitoring now recognizes hidden namespace worker PIDs
  as owned holders when their GPU index and memory footprint match the
  controller's worker plan, avoiding false `process_grace` releases.
- Fleet policy now keeps holders running while the rolling policy window is
  below target, even if instantaneous utilization temporarily exceeds the
  target. This prevents high-load matmul holders from oscillating between
  active and idle states.

## [0.1.0] - Unreleased

- Initial alpha release target.
