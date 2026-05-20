from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable

from . import __version__
from .advisor import profile_comparison, tuning_advice
from .alerts import alert_rules_payload, render_prometheus_alert_rules
from .config import (
    CONFIG_PROFILES,
    CONFIG_RECIPES,
    DEFAULT_STATE_DIR,
    GuardConfig,
    config_payload,
    config_reference,
    config_template,
    load_config_file,
    profile_defaults,
    profile_reference,
    recipe_reference,
    recipe_template,
    validate_config,
    validate_config_keys,
)
from .completion import generate_completion
from .controller import GuardController, read_status, read_status_result
from .diagnostics import run_diagnostics
from .events import filter_events, read_events, read_events_since
from .explain import explain_runtime, format_explanation
from .grafana import grafana_dashboard_payload
from .history import summarize_events
from .manual import generate_manual
from .metrics import prometheus_metrics
from .models import GpuHistory, GpuProcess, GpuSnapshot, HolderDecision
from .monitoring_bundle import write_monitoring_bundle
from .policy import decide_for_fleet
from .reasons import reason_reference
from .service import generate_systemd_user_service
from .simulator import (
    ALL_SCENARIOS,
    DEFAULT_SCENARIO,
    built_in_trace,
    load_trace,
    scenario_choices,
    simulate_policy,
    simulate_scenarios,
)
from .state import (
    disable_gpu,
    enable_gpu,
    parse_duration_seconds,
    pause_guard,
    resume_guard,
    read_disabled_gpus,
    write_text_atomically,
)
from .status_health import (
    enrich_status_health,
    format_policy_window_health,
    format_status_age,
    format_utilization_target_forecast,
    policy_window_health,
    status_check,
    utilization_target_forecast,
    utilization_target_health,
)
from .units import parse_memory_spec


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "guard":
        return _cmd_guard(args)
    if args.command == "plan":
        return _cmd_plan(args)
    if args.command == "preflight":
        return _cmd_preflight(args)
    if args.command == "start":
        return _cmd_start(args)
    if args.command == "config":
        return _cmd_config(args)
    if args.command == "config-reference":
        return _cmd_config_reference(args)
    if args.command == "profile-reference":
        return _cmd_profile_reference(args)
    if args.command == "recipes":
        return _cmd_recipes(args)
    if args.command == "reason-reference":
        return _cmd_reason_reference(args)
    if args.command == "init-config":
        return _cmd_init_config(args)
    if args.command == "simulate":
        return _cmd_simulate(args)
    if args.command == "tune":
        return _cmd_tune(args)
    if args.command == "stop":
        return _cmd_stop(args)
    if args.command == "status":
        return _cmd_status(args)
    if args.command == "metrics":
        return _cmd_metrics(args)
    if args.command == "alerts":
        return _cmd_alerts(args)
    if args.command == "grafana-dashboard":
        return _cmd_grafana_dashboard(args)
    if args.command == "monitoring-bundle":
        return _cmd_monitoring_bundle(args)
    if args.command == "events":
        return _cmd_events(args)
    if args.command == "history":
        return _cmd_history(args)
    if args.command == "explain":
        return _cmd_explain(args)
    if args.command == "dashboard":
        return _cmd_dashboard(args)
    if args.command == "pause":
        return _cmd_pause(args)
    if args.command == "resume":
        return _cmd_resume(args)
    if args.command == "disable-gpu":
        return _cmd_disable_gpu(args)
    if args.command == "enable-gpu":
        return _cmd_enable_gpu(args)
    if args.command == "doctor":
        return _cmd_doctor(args)
    if args.command == "report":
        return _cmd_report(args)
    if args.command == "service":
        return _cmd_service(args)
    if args.command == "completion":
        return _cmd_completion(args)
    if args.command == "manual":
        return _cmd_manual(args)
    parser.print_help()
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpu-holder",
        description="Keep shared NVIDIA GPU utilization high while yielding to real workloads.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    _add_guard_like(subparsers.add_parser("guard", help="run the foreground controller"))
    _add_guard_like(subparsers.add_parser("plan", help="print policy decisions without workers"))
    preflight_parser = subparsers.add_parser(
        "preflight",
        help="validate startup readiness without workers",
    )
    _add_guard_like(preflight_parser)
    preflight_parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="skip runtime diagnostics and only inspect config/snapshots/policy",
    )
    _add_guard_like(subparsers.add_parser("start", help="start a lightweight background daemon"))
    config_parser = subparsers.add_parser("config", help="validate and print the resolved guard config")
    _add_guard_like(config_parser)
    config_parser.add_argument(
        "--explain",
        action="store_true",
        help="include profile/default/config/CLI precedence details",
    )
    config_reference_parser = subparsers.add_parser(
        "config-reference",
        help="print documented guard config fields",
    )
    config_reference_parser.add_argument("--json", action="store_true")
    profile_reference_parser = subparsers.add_parser(
        "profile-reference",
        help="print documented runtime profiles",
    )
    profile_reference_parser.add_argument("--json", action="store_true")
    recipes_parser = subparsers.add_parser(
        "recipes",
        help="print scenario-oriented config recipes",
    )
    recipes_parser.add_argument(
        "--name",
        choices=tuple(CONFIG_RECIPES),
        default=None,
        help="print a TOML template for one recipe",
    )
    recipes_parser.add_argument("--json", action="store_true")
    reason_reference_parser = subparsers.add_parser(
        "reason-reference",
        help="print documented policy decision reasons",
    )
    reason_reference_parser.add_argument("--json", action="store_true")
    init_config_parser = subparsers.add_parser("init-config", help="write a default TOML config file")
    init_config_parser.add_argument("--path", default="gpu-holder.toml")
    init_config_parser.add_argument("--profile", choices=tuple(CONFIG_PROFILES), default="balanced")
    init_config_parser.add_argument(
        "--minimal",
        action="store_true",
        help="write a compact profile-friendly config with only commented overrides",
    )
    init_config_parser.add_argument("--force", action="store_true", help="overwrite an existing file")
    init_config_parser.add_argument("--stdout", action="store_true", help="print the default config and do not write a file")
    simulate_parser = subparsers.add_parser("simulate", help="simulate policy decisions from a trace")
    _add_guard_like(simulate_parser)
    simulate_parser.add_argument("--trace", default=None, help="JSON trace file; uses a built-in scenario when omitted")
    simulate_parser.add_argument(
        "--scenario",
        choices=scenario_choices(),
        default=DEFAULT_SCENARIO,
        help="built-in scenario used when --trace is omitted; use 'all' to run every scenario",
    )
    tune_parser = subparsers.add_parser(
        "tune",
        help="score config against offline policy scenarios",
    )
    _add_guard_like(tune_parser)
    tune_parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero on warnings as well as failures",
    )
    tune_parser.add_argument(
        "--compare-profiles",
        action="store_true",
        help="also score all built-in profiles and recommend one",
    )
    stop_parser = subparsers.add_parser("stop", help="stop the background daemon")
    stop_parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    status_parser = subparsers.add_parser("status", help="show the latest status file")
    status_parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    status_parser.add_argument("--json", action="store_true")
    status_parser.add_argument("--check", action="store_true", help="exit non-zero if status is missing or stale")
    status_parser.add_argument(
        "--require-target",
        action="store_true",
        help="with --check, fail if the policy average utilization is below target_util",
    )
    status_parser.add_argument(
        "--require-forecast",
        action="store_true",
        help="with --check, fail if the utilization target forecast is unrecoverable",
    )
    status_parser.add_argument("--watch", action="store_true", help="keep rendering status until interrupted")
    status_parser.add_argument("--interval", type=float, default=2.0, help="watch refresh interval in seconds")
    status_parser.add_argument("--count", type=int, default=None, help="stop after N watch refreshes")
    status_parser.add_argument("--no-clear", action="store_true", help="do not clear the terminal while watching")
    status_parser.add_argument(
        "--stale-after",
        type=float,
        default=15.0,
        help="mark status as stale after this many seconds; use 0 to disable",
    )
    metrics_parser = subparsers.add_parser("metrics", help="export latest status as Prometheus text")
    metrics_parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    metrics_parser.add_argument(
        "--stale-after",
        type=float,
        default=15.0,
        help="mark status as stale after this many seconds; use 0 to disable",
    )
    metrics_parser.add_argument(
        "--output",
        default=None,
        help="write Prometheus text to this path atomically instead of stdout",
    )
    alerts_parser = subparsers.add_parser("alerts", help="print Prometheus alert rules")
    alerts_parser.add_argument("--json", action="store_true")
    alerts_parser.add_argument("--group-name", default="gpu-holder")
    grafana_parser = subparsers.add_parser("grafana-dashboard", help="print a Grafana dashboard JSON")
    grafana_parser.add_argument("--title", default="gpu-holder")
    grafana_parser.add_argument("--uid", default="gpu-holder")
    grafana_parser.add_argument("--datasource", default="${DS_PROMETHEUS}")
    bundle_parser = subparsers.add_parser(
        "monitoring-bundle",
        help="write Prometheus alerts and Grafana dashboard files",
    )
    bundle_parser.add_argument("--output-dir", default="gpu-holder-monitoring")
    bundle_parser.add_argument("--group-name", default="gpu-holder")
    bundle_parser.add_argument("--dashboard-title", default="gpu-holder")
    bundle_parser.add_argument("--dashboard-uid", default="gpu-holder")
    bundle_parser.add_argument("--datasource", default="${DS_PROMETHEUS}")
    bundle_parser.add_argument("--json", action="store_true")
    events_parser = subparsers.add_parser("events", help="show recent controller events")
    events_parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    events_parser.add_argument("--limit", type=int, default=20)
    events_parser.add_argument("--json", action="store_true")
    events_parser.add_argument(
        "--event-type",
        action="append",
        default=None,
        help="filter events by type; can be repeated or comma-separated",
    )
    events_parser.add_argument(
        "--gpu",
        default=None,
        help="filter events by GPU index list, e.g. 0 or 0,2",
    )
    events_parser.add_argument("--follow", action="store_true", help="follow events like tail -f")
    events_parser.add_argument("--interval", type=float, default=1.0, help="follow polling interval in seconds")
    events_parser.add_argument("--count", type=int, default=None, help="stop after N follow polls")
    history_parser = subparsers.add_parser("history", help="summarize recent controller event history")
    history_parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    history_parser.add_argument("--limit", type=int, default=5000, help="maximum recent events to inspect")
    history_parser.add_argument("--since", default=None, help="summarize events newer than this duration, e.g. 1h")
    history_parser.add_argument("--gpu", default=None, help="filter by GPU index list, e.g. 0 or 0,2")
    history_parser.add_argument("--json", action="store_true")
    explain_parser = subparsers.add_parser("explain", help="explain current guard state and suggested actions")
    explain_parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    explain_parser.add_argument("--limit", type=int, default=5000, help="recent events to inspect")
    explain_parser.add_argument(
        "--stale-after",
        type=float,
        default=15.0,
        help="mark status as stale after this many seconds; use 0 to disable",
    )
    explain_parser.add_argument("--json", action="store_true")
    dashboard_parser = subparsers.add_parser("dashboard", help="open the terminal status dashboard")
    dashboard_parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    dashboard_parser.add_argument("--refresh-interval", type=float, default=2.0)
    dashboard_parser.add_argument(
        "--temporary-disable-duration",
        default="30m",
        help="duration used by the dashboard D key, e.g. 30s, 10m, 2h",
    )
    dashboard_parser.add_argument(
        "--temporary-pause-duration",
        default="10m",
        help="duration used by the dashboard P key, e.g. 30s, 10m, 2h",
    )
    dashboard_parser.add_argument("--once", action="store_true", help="render one text snapshot and exit")
    dashboard_parser.add_argument("--events", action="store_true", help="include recent events with --once")
    dashboard_parser.add_argument("--history", action="store_true", help="include recent history summary with --once")
    dashboard_parser.add_argument("--explain", action="store_true", help="include current explanation with --once")
    dashboard_parser.add_argument("--advice", action="store_true", help="include offline tuning advice with --once")
    dashboard_parser.add_argument("--history-limit", type=int, default=5000, help="recent events to inspect for dashboard history")
    dashboard_parser.add_argument("--config", action="store_true", help="show resolved config with --once")
    dashboard_parser.add_argument("--details", type=int, default=None, help="show one GPU detail with --once")
    dashboard_parser.add_argument("--event-gpu", type=int, default=None, help="filter --once events to one GPU")
    pause_parser = subparsers.add_parser("pause", help="pause all holder workers via pause file")
    pause_parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    pause_parser.add_argument(
        "--for",
        dest="duration",
        default=None,
        help="auto-resume after this duration, e.g. 30s, 10m, 2h",
    )
    resume_parser = subparsers.add_parser("resume", help="resume holder workers by removing pause file")
    resume_parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    disable_parser = subparsers.add_parser("disable-gpu", help="release and skip one GPU at runtime")
    disable_parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    disable_parser.add_argument(
        "--for",
        dest="duration",
        default=None,
        help="auto-enable this GPU after the duration, e.g. 30s, 10m, 2h",
    )
    disable_parser.add_argument("gpu_index", type=int)
    enable_parser = subparsers.add_parser("enable-gpu", help="allow holder workers on one GPU again")
    enable_parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    enable_parser.add_argument("gpu_index", type=int)
    doctor_parser = subparsers.add_parser("doctor", help="diagnose NVML/PyTorch runtime readiness")
    doctor_parser.add_argument("--json", action="store_true")
    report_parser = subparsers.add_parser("report", help="build a read-only support report")
    report_parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    report_parser.add_argument("--limit", type=int, default=20, help="number of recent events to include")
    report_parser.add_argument("--no-diagnostics", action="store_true", help="skip runtime diagnostics")
    report_parser.add_argument("--redact", action="store_true", help="redact local paths and process identity fields")
    report_parser.add_argument("--json", action="store_true")
    service_parser = subparsers.add_parser("service", help="print a systemd user service unit")
    service_parser.add_argument("--config", default=None)
    service_parser.add_argument("--state-dir", default=None)
    service_parser.add_argument("--working-directory", default=None)
    service_parser.add_argument("--executable", default="gpu-holder")
    service_parser.add_argument("--restart-sec", type=int, default=10)
    service_parser.add_argument("--description", default="gpu-holder guard")
    service_parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=None,
        help="additional arguments appended after `guard`",
    )
    service_parser.add_argument("guard_args", nargs=argparse.REMAINDER)
    completion_parser = subparsers.add_parser("completion", help="print shell completion script")
    completion_parser.add_argument("shell", choices=("bash", "zsh", "fish"))
    manual_parser = subparsers.add_parser("manual", help="print generated command manual")
    manual_parser.add_argument("--format", choices=("markdown", "man"), default="markdown")
    return parser


def _add_guard_like(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="TOML config file with a [guard] table")
    parser.add_argument("--profile", choices=tuple(CONFIG_PROFILES), default=None)
    parser.add_argument("--gpus", default=None, help="GPU list, e.g. all or 0,1,3")
    parser.add_argument("--target-util", type=int, default=None)
    parser.add_argument("--idle-util", type=int, default=None)
    parser.add_argument("--idle-window", type=float, default=None)
    parser.add_argument("--machine-window", type=float, default=None)
    parser.add_argument("--mem", default=None)
    parser.add_argument("--reserve", default=None)
    parser.add_argument("--busy-process-mem-threshold", default=None)
    parser.add_argument("--assist-mem", default=None)
    parser.add_argument(
        "--max-gpu-temp",
        type=int,
        default=None,
        help="release holders at or above this GPU temperature in Celsius; use 0 to disable",
    )
    parser.add_argument(
        "--thermal-resume-temp",
        type=int,
        default=None,
        help="resume holders after cooling to this Celsius temperature; use 0 for no hysteresis",
    )
    parser.add_argument("--sample-interval", type=float, default=None)
    parser.add_argument("--min-duty-cycle", type=float, default=None)
    parser.add_argument("--max-duty-cycle", type=float, default=None)
    parser.add_argument("--current-gap-weight", type=float, default=None)
    parser.add_argument("--history-gap-weight", type=float, default=None)
    parser.add_argument("--machine-gap-weight", type=float, default=None)
    parser.add_argument("--event-log-max-size", default=None, help="rotate events.jsonl after this size, e.g. 10MiB")
    parser.add_argument("--event-log-backup-count", type=int, default=None)
    parser.add_argument("--worker-restart-backoff", type=float, default=None)
    parser.add_argument("--worker-start-timeout", type=float, default=None)
    parser.add_argument("--worker-update-duty-threshold", type=float, default=None)
    parser.add_argument("--process-grace-window", type=float, default=None)
    parser.add_argument(
        "--protected-process",
        dest="protected_process_patterns",
        action="append",
        default=None,
        help="process name pattern that always yields the GPU; repeat or comma-separate",
    )
    parser.add_argument("--max-held-gpus", type=int, default=None)
    parser.add_argument(
        "--program",
        default=None,
        help="mixed, random, a single program, or a comma-list such as matmul,conv,fft",
    )
    parser.add_argument(
        "--hold-mode",
        choices=("balanced", "memory-only", "compute-only"),
        default=None,
        help="balanced allocates memory and compute; memory-only only reserves memory; compute-only avoids memory hold",
    )
    parser.add_argument("--compute-burst-seconds", type=float, default=None)
    parser.add_argument("--compute-burst-jitter", type=float, default=None)
    parser.add_argument("--state-dir", default=None)
    parser.add_argument("--pause-file", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tui", action="store_true")
    parser.add_argument("--fake", action="store_true", help="use fake snapshots for planning")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")


def _cmd_guard(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    if not _print_config_errors(config):
        return 2
    config.state_dir.mkdir(parents=True, exist_ok=True)
    pidfile_status = _prepare_pid_file_for_guard(config.pid_file)
    if pidfile_status != 0:
        return pidfile_status
    GuardController(config).run()
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    if not _print_config_errors(config):
        return 2
    snapshots = _filter_snapshots(_fake_snapshots() if args.fake else _read_snapshots(config), config.gpus)
    decisions = _decisions_for_snapshots(snapshots, config)
    if args.json:
        payload = {
            "snapshots": [_snapshot_payload(snapshot) for snapshot in snapshots],
            "decisions": [asdict(decision) for decision in decisions],
        }
        print(json.dumps(payload, indent=2))
        return 0
    for decision in decisions:
        print(
            f"gpu={decision.gpu_index} action={decision.action.value} "
            f"reason={decision.reason} mem={decision.memory_bytes} duty={decision.duty_cycle:.2f} "
            f"program={decision.program} mode={decision.hold_mode}"
        )
    return 0


def _cmd_preflight(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    payload = build_preflight_report(
        config=config,
        fake=bool(args.fake),
        include_diagnostics=not bool(args.no_diagnostics),
    )
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_preflight_report(payload)
    return 0 if payload["ok"] else 2


def _cmd_start(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    if not _print_config_errors(config):
        return 2
    config.state_dir.mkdir(parents=True, exist_ok=True)
    pidfile_status = _prepare_pid_file_for_start(config.pid_file)
    if pidfile_status != 0:
        return pidfile_status
    command = [sys.executable, "-m", "gpu_holder.cli", "guard", *_guard_args_for_child(args, config)]
    log = config.log_file.open("ab")
    env = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = (
        source_root
        if not env.get("PYTHONPATH")
        else f"{source_root}{os.pathsep}{env['PYTHONPATH']}"
    )
    process = subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    pidfile_status = _record_started_process_pid(config.pid_file, process.pid, process)
    if pidfile_status != 0:
        return pidfile_status
    print(f"started gpu-holder pid={process.pid} log={config.log_file}")
    return 0


def _prepare_pid_file_for_start(pid_file: Path) -> int:
    return _prepare_pid_file_for_controller(pid_file, current_pid=None)


def _prepare_pid_file_for_guard(pid_file: Path) -> int:
    current_pid = os.getpid()
    while True:
        try:
            descriptor = os.open(pid_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            status = _prepare_pid_file_for_controller(pid_file, current_pid=current_pid)
            if status != 0:
                return status
            if _pid_file_contains(pid_file, current_pid):
                return 0
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"{current_pid}\n")
        return 0


def _record_started_process_pid(
    pid_file: Path,
    pid: int,
    process: subprocess.Popen,
) -> int:
    while True:
        try:
            descriptor = os.open(pid_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            if _pid_file_contains(pid_file, pid):
                return 0
            status = _prepare_pid_file_for_controller(pid_file, current_pid=pid)
            if status != 0:
                _terminate_started_process(process)
                return status
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(str(pid))
        return 0


def _terminate_started_process(process: subprocess.Popen) -> None:
    try:
        process.terminate()
    except OSError:
        return
    try:
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return


def _prepare_pid_file_for_controller(pid_file: Path, *, current_pid: int | None) -> int:
    if not pid_file.exists():
        return 0
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except ValueError:
        pid_file.unlink(missing_ok=True)
        print(f"removed invalid stale pid file: {pid_file}", file=sys.stderr)
        return 0
    if current_pid is not None and pid == current_pid:
        return 0
    if _is_gpu_holder_guard_process(pid):
        print(f"gpu-holder already running pid={pid}", file=sys.stderr)
        return 1
    if _process_exists(pid):
        print(
            f"refusing to reuse pid file {pid_file}: pid={pid} belongs to another process",
            file=sys.stderr,
        )
        return 2
    pid_file.unlink(missing_ok=True)
    print(f"removed stale pid file: {pid_file}", file=sys.stderr)
    return 0


def _pid_file_contains(pid_file: Path, pid: int) -> bool:
    try:
        return int(pid_file.read_text(encoding="utf-8").strip()) == int(pid)
    except (OSError, ValueError):
        return False


def _cmd_config(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    errors, warnings = validate_config(config)
    payload = {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "config": config_payload(config),
    }
    if args.explain:
        payload["explain"] = _config_explain_from_args(args, config)
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0 if not errors else 2
    print(f"ok={payload['ok']}")
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
    if warnings:
        for warning in warnings:
            print(f"warning: {warning}", file=sys.stderr)
    config_data = payload["config"]
    if isinstance(config_data, dict):
        for key, value in config_data.items():
            print(f"{key}={value}")
    if args.explain:
        explain = payload.get("explain")
        if isinstance(explain, dict):
            print_config_explain(explain)
    return 0 if not errors else 2


def _cmd_config_reference(args: argparse.Namespace) -> int:
    payload = {
        "schema_version": 1,
        "fields": config_reference(),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_config_reference(payload)
    return 0


def _cmd_profile_reference(args: argparse.Namespace) -> int:
    payload = {
        "schema_version": 1,
        "profiles": profile_reference(),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_profile_reference(payload)
    return 0


def _cmd_recipes(args: argparse.Namespace) -> int:
    payload: dict[str, object] = {
        "schema_version": 1,
        "recipes": recipe_reference(),
    }
    if args.name:
        recipe = _recipe_payload(args.name)
        payload["recipe"] = recipe
        payload["toml"] = recipe_template(args.name)
        if args.json:
            print(json.dumps(payload, indent=2))
            return 0
        print(str(payload["toml"]), end="")
        return 0
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_recipe_reference(payload)
    return 0


def _recipe_payload(name: str) -> dict[str, object]:
    for recipe in recipe_reference():
        if recipe.get("name") == name:
            return recipe
    supported = ", ".join(CONFIG_RECIPES)
    raise ValueError(f"unsupported recipe: {name}; supported: {supported}")


def _cmd_reason_reference(args: argparse.Namespace) -> int:
    payload = {
        "schema_version": 1,
        "reasons": reason_reference(),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_reason_reference(payload)
    return 0


def _cmd_init_config(args: argparse.Namespace) -> int:
    minimal = bool(args.minimal or args.profile != "balanced")
    payload = config_template(profile=args.profile, minimal=minimal)
    if args.stdout:
        print(payload, end="")
        return 0
    path = Path(args.path).expanduser()
    if path.exists() and not args.force:
        print(f"config already exists: {path}; use --force to overwrite", file=sys.stderr)
        return 2
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    print(f"wrote config: {path}")
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    if not _print_config_errors(config):
        return 2
    try:
        if args.trace:
            frames = load_trace(args.trace)
            payload = simulate_policy(frames, config)
            payload["scenario"] = None
            payload["trace"] = args.trace
        elif args.scenario == ALL_SCENARIOS:
            payload = simulate_scenarios(config)
            payload["scenario"] = ALL_SCENARIOS
            payload["trace"] = None
        else:
            frames = built_in_trace(args.scenario)
            payload = simulate_policy(frames, config)
            payload["scenario"] = args.scenario
            payload["trace"] = None
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        error = f"trace error: {exc}"
        if args.json:
            print(json.dumps({"ok": False, "error": error}, indent=2))
        else:
            print(error, file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    if payload.get("scenario") == ALL_SCENARIOS:
        print_simulation_suite(payload)
        return 0
    print_simulation(payload)
    return 0


def _cmd_tune(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    errors, warnings = validate_config(config)
    for warning in warnings:
        print(f"config warning: {warning}", file=sys.stderr)
    if errors:
        payload = {
            "schema_version": 1,
            "ok": False,
            "grade": "fail",
            "score": 0,
            "errors": errors,
            "warnings": warnings,
            "config": config_payload(config),
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            for error in errors:
                print(f"config error: {error}", file=sys.stderr)
        return 2
    payload = tuning_advice(config)
    payload["warnings"] = warnings
    payload["strict"] = bool(args.strict)
    payload["strict_ok"] = bool(payload["ok"] and (not args.strict or payload["grade"] == "pass"))
    if args.compare_profiles:
        payload["profile_comparison"] = profile_comparison()
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0 if payload["strict_ok"] else 2
    print_tuning_advice(payload)
    return 0 if payload["strict_ok"] else 2


def print_simulation(payload: dict[str, object]) -> None:
    summary = payload["summary"]
    if isinstance(summary, dict):
        print(
            f"steps={summary.get('steps')} gpu_count={summary.get('gpu_count')} "
            f"avg_machine={float(summary.get('average_machine_utilization', 0.0)):.1f}% "
            f"actions={summary.get('action_counts')}"
        )
    timeline = payload.get("timeline", [])
    if isinstance(timeline, list):
        for item in timeline:
            if not isinstance(item, dict):
                continue
            print(
                f"step={item.get('step')} timestamp={item.get('timestamp')} "
                f"machine_avg={float(item.get('machine_average', 0.0)):.1f}% "
                f"running={item.get('running_gpu_indices')}"
            )
            decisions = item.get("decisions", [])
            if not isinstance(decisions, list):
                continue
            for decision in decisions:
                if not isinstance(decision, dict):
                    continue
                print(
                    f"  gpu={decision.get('gpu_index')} action={decision.get('action')} "
                    f"reason={decision.get('reason')} duty={float(decision.get('duty_cycle', 0.0)):.2f} "
                    f"mode={decision.get('hold_mode')}"
                )


def print_tuning_advice(payload: dict[str, object]) -> None:
    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    print(
        f"gpu-holder tune grade={payload.get('grade')} score={payload.get('score')} "
        f"ok={payload.get('ok')} strict_ok={payload.get('strict_ok')}"
    )
    print(
        f"checks={summary.get('checks')} passed={summary.get('passed')} "
        f"warnings={summary.get('warnings')} failures={summary.get('failures')} "
        f"scenarios={summary.get('scenario_count')}"
    )
    checks = payload.get("checks", [])
    if isinstance(checks, list):
        for check in checks:
            if not isinstance(check, dict):
                continue
            print(
                f"check={check.get('name')} status={check.get('status')} "
                f"message={check.get('message')}"
            )
    recommendations = payload.get("recommendations", [])
    if isinstance(recommendations, list) and recommendations:
        print("recommendations:")
        for recommendation in recommendations:
            print(f"recommendation: {recommendation}")
    comparison = payload.get("profile_comparison")
    if isinstance(comparison, dict):
        print("Profile comparison")
        print(f"recommended_profile={comparison.get('recommended_profile')}")
        profiles = comparison.get("profiles", [])
        if isinstance(profiles, list):
            for profile in profiles:
                if not isinstance(profile, dict):
                    continue
                print(
                    f"profile={profile.get('profile')} grade={profile.get('grade')} "
                    f"score={profile.get('score')} warnings={profile.get('warnings')} "
                    f"failures={profile.get('failures')} target={profile.get('target_util')} "
                    f"mem={profile.get('mem')} program={profile.get('program')} "
                    f"mode={profile.get('hold_mode')} max_held={profile.get('max_held_gpus')} "
                    f"description={profile.get('description')}"
                )


def print_simulation_suite(payload: dict[str, object]) -> None:
    summary = payload.get("summary", {})
    if isinstance(summary, dict):
        print(
            f"scenarios={summary.get('scenario_count')} steps={summary.get('steps')} "
            f"gpu_decisions={summary.get('gpu_decisions')} "
            f"avg_machine={float(summary.get('average_machine_utilization', 0.0)):.1f}% "
            f"actions={summary.get('action_counts')}"
        )
    scenarios = payload.get("scenarios", [])
    if not isinstance(scenarios, list):
        return
    for item in scenarios:
        if not isinstance(item, dict):
            continue
        item_summary = item.get("summary", {})
        if not isinstance(item_summary, dict):
            continue
        print(
            f"scenario={item.get('scenario')} steps={item_summary.get('steps')} "
            f"gpu_count={item_summary.get('gpu_count')} "
            f"avg_machine={float(item_summary.get('average_machine_utilization', 0.0)):.1f}% "
            f"actions={item_summary.get('action_counts')}"
        )


def _cmd_stop(args: argparse.Namespace) -> int:
    pid_file = Path(args.state_dir).expanduser() / "gpu-holder.pid"
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except FileNotFoundError:
        print("gpu-holder is not running")
        return 0
    if not _is_gpu_holder_guard_process(pid):
        print(
            f"refusing to stop pid={pid}: process does not look like gpu-holder guard",
            file=sys.stderr,
        )
        return 2
    os.kill(pid, 15)
    print(f"sent SIGTERM to pid={pid}")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    status_file = Path(args.state_dir).expanduser() / "status.json"
    if args.watch:
        return watch_status(
            status_file=status_file,
            json_output=args.json,
            interval=args.interval,
            count=args.count,
            clear=not args.no_clear,
            stale_after=args.stale_after,
        )
    payload, status_error = read_status_result(status_file)
    if payload is None:
        error_payload = _status_error_payload(status_error or "no status file", check=args.check)
        if args.json:
            print(json.dumps(error_payload, indent=2))
        else:
            print(error_payload["status_error"])
        return 2 if args.check and status_error != "no status file" else 1
    payload = enrich_status_health(payload, stale_after=args.stale_after)
    check_result = (
        status_check(
            payload,
            stale_after=args.stale_after,
            require_target=bool(args.require_target),
            require_forecast=bool(args.require_forecast),
        )
        if args.check
        else None
    )
    if check_result is not None:
        payload["status_check"] = check_result
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_status(payload, stale_after=args.stale_after)
        if check_result is not None:
            if check_result["ok"]:
                print("status_check=ok")
            else:
                print("status_check=failed")
                for error in check_result["errors"]:
                    print(f"status_check_error: {error}")
    if check_result is not None and not check_result["ok"]:
        return 2
    return 0


def _cmd_metrics(args: argparse.Namespace) -> int:
    payload, status_error = read_status_result(Path(args.state_dir).expanduser() / "status.json")
    rendered = prometheus_metrics(
        payload,
        status_error=status_error,
        stale_after=args.stale_after,
    )
    if args.output:
        output_path = write_text_atomically(args.output, rendered)
        print(f"wrote metrics: {output_path}")
    else:
        print(rendered, end="")
    return 0 if payload is not None else 1


def _cmd_alerts(args: argparse.Namespace) -> int:
    if args.json:
        print(json.dumps(alert_rules_payload(group_name=args.group_name), indent=2))
    else:
        print(render_prometheus_alert_rules(group_name=args.group_name), end="")
    return 0


def _cmd_grafana_dashboard(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            grafana_dashboard_payload(
                title=args.title,
                uid=args.uid,
                datasource=args.datasource,
            ),
            indent=2,
        )
    )
    return 0


def _cmd_monitoring_bundle(args: argparse.Namespace) -> int:
    payload = write_monitoring_bundle(
        Path(args.output_dir),
        group_name=args.group_name,
        dashboard_title=args.dashboard_title,
        dashboard_uid=args.dashboard_uid,
        datasource=args.datasource,
    )
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"wrote monitoring bundle: {payload['output_dir']}")
        files = payload.get("files", {})
        if isinstance(files, dict):
            for label in ("alerts", "dashboard", "readme"):
                print(f"{label}: {files.get(label)}")
    return 0


def _cmd_events(args: argparse.Namespace) -> int:
    event_log = Path(args.state_dir).expanduser() / "events.jsonl"
    if args.follow:
        return follow_events(
            event_log=event_log,
            limit=args.limit,
            json_output=args.json,
            interval=args.interval,
            count=args.count,
            event_types=_parse_event_types(args.event_type),
            gpu_indices=_parse_gpu_filter(args.gpu),
        )
    events = read_events(
        event_log,
        limit=args.limit,
        event_types=_parse_event_types(args.event_type),
        gpu_indices=_parse_gpu_filter(args.gpu),
    )
    if args.json:
        print(json.dumps(events, indent=2))
        return 0
    if not events:
        print("no events")
        return 1
    for event in events:
        extras = " ".join(
            f"{key}={value}"
            for key, value in event.items()
            if key not in {"timestamp", "type"}
        )
        print(f"timestamp={event.get('timestamp')} type={event.get('type')} {extras}".rstrip())
    return 0


def _cmd_history(args: argparse.Namespace) -> int:
    since_seconds = None
    if args.since is not None:
        try:
            since_seconds = parse_duration_seconds(args.since)
        except ValueError as exc:
            print(f"invalid --since: {exc}", file=sys.stderr)
            return 2
    event_log = Path(args.state_dir).expanduser() / "events.jsonl"
    events = read_events(event_log, limit=args.limit)
    payload = summarize_events(
        events,
        since_seconds=since_seconds,
        gpu_indices=_parse_gpu_filter(args.gpu),
    )
    payload["source"] = str(event_log)
    payload["limit"] = args.limit
    payload["since_seconds"] = since_seconds
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    if not payload["ok"]:
        print("no history")
        return 1
    print_history(payload)
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    state_dir = Path(args.state_dir).expanduser()
    status_payload, status_error = read_status_result(state_dir / "status.json")
    events = read_events(state_dir / "events.jsonl", limit=args.limit)
    payload = explain_runtime(
        status=status_payload,
        status_error=status_error,
        events=events,
        stale_after=args.stale_after,
    )
    payload["source"] = {
        "state_dir": str(state_dir),
        "event_limit": args.limit,
        "stale_after": args.stale_after,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("\n".join(format_explanation(payload)))
    return 0 if payload.get("status") is not None else 1


def watch_status(
    *,
    status_file: Path,
    json_output: bool,
    interval: float,
    count: int | None = None,
    clear: bool = True,
    stale_after: float | None = 15.0,
    sink: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    iterations = 0
    printed = False
    while count is None or iterations < count:
        payload, status_error = read_status_result(status_file)
        if payload is None:
            error_payload = _status_error_payload(status_error or "no status file", check=False)
            if json_output:
                sink(json.dumps(error_payload, separators=(",", ":")))
            else:
                sink(str(error_payload["status_error"]))
        elif json_output:
            payload = enrich_status_health(payload, stale_after=stale_after)
            sink(json.dumps(payload, separators=(",", ":")))
            printed = True
        else:
            if clear:
                sink("\033[2J\033[H")
            print_status(payload, stale_after=stale_after, sink=sink)
            printed = True
        iterations += 1
        if count is not None and iterations >= count:
            break
        sleep(max(0.0, float(interval)))
    return 0 if printed else 1


def _status_error_payload(status_error: str, *, check: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "ok": False,
        "status_error": status_error,
    }
    if check:
        payload["status_check"] = {
            "ok": False,
            "errors": [status_error],
        }
    return payload


def follow_events(
    *,
    event_log: Path,
    limit: int,
    json_output: bool,
    interval: float,
    count: int | None = None,
    sink: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
    after_poll: Callable[[int], None] | None = None,
    event_types: set[str] | None = None,
    gpu_indices: set[int] | None = None,
) -> int:
    printed = False
    offset = 0
    initial_events = read_events(
        event_log,
        limit=limit,
        event_types=event_types,
        gpu_indices=gpu_indices,
    )
    for event in initial_events:
        _print_event(event, json_output=json_output, sink=sink)
        printed = True
    try:
        offset = event_log.stat().st_size
    except FileNotFoundError:
        offset = 0
    iterations = 0
    while count is None or iterations < count:
        events, offset = read_events_since(event_log, offset=offset)
        events = filter_events(events, event_types=event_types, gpu_indices=gpu_indices)
        for event in events:
            _print_event(event, json_output=json_output, sink=sink)
            printed = True
        if after_poll is not None:
            after_poll(iterations)
        iterations += 1
        if count is not None and iterations >= count:
            break
        sleep(max(0.0, float(interval)))
    return 0 if printed else 1


def _cmd_dashboard(args: argparse.Namespace) -> int:
    from .tui import format_dashboard_snapshot, run_status_dashboard

    state_dir = Path(args.state_dir).expanduser()
    status_file = state_dir / "status.json"
    event_log = state_dir / "events.jsonl"
    try:
        temporary_disable_seconds = parse_duration_seconds(args.temporary_disable_duration)
    except ValueError as exc:
        print(f"invalid --temporary-disable-duration: {exc}", file=sys.stderr)
        return 2
    try:
        temporary_pause_seconds = parse_duration_seconds(args.temporary_pause_duration)
    except ValueError as exc:
        print(f"invalid --temporary-pause-duration: {exc}", file=sys.stderr)
        return 2
    if args.once:
        payload, status_error = read_status_result(status_file)
        if payload is None:
            print(status_error or "no status file")
            return 1
        event_limit = args.history_limit if args.history or args.explain else 8
        events = (
            read_events(event_log, limit=event_limit)
            if args.events or args.history or args.explain or args.event_gpu is not None
            else []
        )
        lines = format_dashboard_snapshot(
            payload,
            events=events,
            show_events=bool(args.events or args.event_gpu is not None),
            show_history=bool(args.history),
            show_explain=bool(args.explain),
            show_advice=bool(args.advice),
            show_config=bool(args.config),
            show_detail=args.details is not None,
            selected_gpu_index=args.details,
            event_filter_gpu_index=args.event_gpu,
        )
        print("\n".join(lines))
        return 0
    run_status_dashboard(
        status_reader=lambda: read_status(status_file),
        event_reader=lambda: read_events(event_log, limit=args.history_limit),
        state_dir=state_dir,
        refresh_interval=args.refresh_interval,
        temporary_disable_seconds=temporary_disable_seconds,
        temporary_pause_seconds=temporary_pause_seconds,
    )
    return 0


def _cmd_pause(args: argparse.Namespace) -> int:
    duration_seconds = None
    if args.duration is not None:
        try:
            duration_seconds = parse_duration_seconds(args.duration)
        except ValueError as exc:
            print(f"invalid --for: {exc}", file=sys.stderr)
            return 2
    now = time.time()
    pause_file = pause_guard(args.state_dir, duration_seconds=duration_seconds, now=now)
    if duration_seconds is None:
        print(f"paused: {pause_file}")
    else:
        pause_until = now + duration_seconds
        print(f"paused: {pause_file} until={_format_timestamp(pause_until)}")
    return 0


def _cmd_resume(args: argparse.Namespace) -> int:
    pause_file = resume_guard(args.state_dir)
    if pause_file.exists():
        print(f"paused: {pause_file}")
    else:
        print(f"resumed: removed {pause_file}")
    return 0


def _cmd_disable_gpu(args: argparse.Namespace) -> int:
    duration_seconds = None
    if args.duration is not None:
        try:
            duration_seconds = parse_duration_seconds(args.duration)
        except ValueError as exc:
            print(f"invalid --for: {exc}", file=sys.stderr)
            return 2
    now = time.time()
    path = disable_gpu(
        args.state_dir,
        args.gpu_index,
        duration_seconds=duration_seconds,
        now=now,
    )
    disabled = sorted(read_disabled_gpus(args.state_dir))
    if duration_seconds is None:
        print(f"disabled gpu={args.gpu_index} state={path} disabled={disabled}")
    else:
        disabled_until = now + duration_seconds
        print(
            f"disabled gpu={args.gpu_index} state={path} "
            f"until={_format_timestamp(disabled_until)} disabled={disabled}"
        )
    return 0


def _cmd_enable_gpu(args: argparse.Namespace) -> int:
    path = enable_gpu(args.state_dir, args.gpu_index)
    disabled = sorted(read_disabled_gpus(args.state_dir))
    print(f"enabled gpu={args.gpu_index} state={path} disabled={disabled}")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    payload = run_diagnostics()
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"gpu-holder doctor ok={payload['ok']}")
        for check in payload["checks"]:
            print(f"{check['name']}: {'ok' if check['ok'] else 'fail'} - {check['detail']}")
            suggestion = check.get("suggestion")
            if suggestion:
                print(f"  suggestion: {suggestion}")
    return 0 if payload["ok"] else 1


def _cmd_report(args: argparse.Namespace) -> int:
    payload = build_support_report(
        state_dir=Path(args.state_dir).expanduser(),
        event_limit=args.limit,
        include_diagnostics=not args.no_diagnostics,
        redact=bool(args.redact),
    )
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_report(payload)
    return 0


def _cmd_service(args: argparse.Namespace) -> int:
    raw_extra_args = args.extra_args if args.extra_args is not None else args.guard_args
    extra_args = tuple(str(item) for item in raw_extra_args if str(item) != "--")
    print(
        generate_systemd_user_service(
            executable=args.executable,
            config_path=args.config,
            state_dir=args.state_dir,
            working_directory=args.working_directory,
            restart_seconds=args.restart_sec,
            extra_args=extra_args,
            description=args.description,
        ),
        end="",
    )
    return 0


def _cmd_completion(args: argparse.Namespace) -> int:
    print(generate_completion(args.shell), end="")
    return 0


def _cmd_manual(args: argparse.Namespace) -> int:
    print(generate_manual(args.format), end="")
    return 0


def build_preflight_report(
    *,
    config: GuardConfig,
    fake: bool = False,
    include_diagnostics: bool = True,
) -> dict[str, object]:
    errors, warnings = validate_config(config)
    visible_snapshots: list[GpuSnapshot] = []
    selected_snapshots: list[GpuSnapshot] = []
    decisions = []
    missing_selected_gpus: list[int] = []
    risks: list[dict[str, object]] = []
    diagnostics = run_diagnostics() if include_diagnostics else None
    state_dir_summary = _state_dir_summary(config.state_dir)
    _extend_preflight_with_state_dir_check(
        state_dir_summary=state_dir_summary,
        errors=errors,
        risks=risks,
    )
    pid_file_summary = _pid_file_summary(config.pid_file)
    _extend_preflight_with_pid_file_check(
        pid_file_summary=pid_file_summary,
        errors=errors,
        warnings=warnings,
        risks=risks,
    )

    if not errors:
        try:
            visible_snapshots = _fake_snapshots() if fake else _read_snapshots(config)
        except Exception as exc:  # pragma: no cover - exercised by real runtime failures.
            errors.append(f"failed to read GPU snapshots: {exc}")
            risks.append(
                {
                    "severity": "error",
                    "code": "snapshot_read_failed",
                    "message": str(exc),
                }
            )
        selected_snapshots = _filter_snapshots(visible_snapshots, config.gpus)
        missing_selected_gpus = _missing_selected_gpus(visible_snapshots, config.gpus)
        if missing_selected_gpus:
            missing_text = ",".join(str(index) for index in missing_selected_gpus)
            errors.append(f"selected GPU indices are not visible: {missing_text}")
            risks.append(
                {
                    "severity": "error",
                    "code": "selected_gpu_missing",
                    "message": f"selected GPU indices are not visible: {missing_text}",
                    "gpu_indices": missing_selected_gpus,
                }
            )
        if visible_snapshots and not selected_snapshots:
            errors.append("no GPUs selected after applying the gpus filter")
            risks.append(
                {
                    "severity": "error",
                    "code": "no_selected_gpus",
                    "message": "no GPUs selected after applying the gpus filter",
                }
            )
        if not visible_snapshots:
            errors.append("no visible NVIDIA GPUs were found")
            risks.append(
                {
                    "severity": "error",
                    "code": "no_visible_gpus",
                    "message": "no visible NVIDIA GPUs were found",
                }
            )
        if selected_snapshots and not missing_selected_gpus:
            decisions = _decisions_for_snapshots(selected_snapshots, config)

    action_counts = _action_counts(decisions)
    reason_counts = _reason_counts(decisions)
    holders_would_start = sum(
        action_counts.get(action, 0)
        for action in (
            "hold",
            "assist",
        )
    )
    if holders_would_start:
        risks.append(
            {
                "severity": "info",
                "code": "holders_would_start",
                "message": f"{holders_would_start} holder worker(s) would be requested now",
                "count": holders_would_start,
            }
        )
    protected_process_count = reason_counts.get("protected_process", 0)
    if protected_process_count:
        risks.append(
            {
                "severity": "info",
                "code": "protected_process_matches",
                "message": (
                    f"{protected_process_count} GPU(s) would yield because a "
                    "visible process matches protected_process_patterns"
                ),
                "count": protected_process_count,
            }
        )

    diagnostics_ok = True
    if isinstance(diagnostics, dict):
        diagnostics_ok = bool(diagnostics.get("ok"))

    ok = not errors and diagnostics_ok
    return {
        "schema_version": 1,
        "ok": ok,
        "fake": fake,
        "errors": errors,
        "warnings": warnings,
        "risks": risks,
        "diagnostics": diagnostics,
        "missing_selected_gpus": missing_selected_gpus,
        "summary": {
            "visible_gpu_count": len(visible_snapshots),
            "selected_gpu_count": len(selected_snapshots),
            "action_counts": action_counts,
            "reason_counts": reason_counts,
            "holders_would_start": holders_would_start,
        },
        "state_dir": state_dir_summary,
        "pid_file": pid_file_summary,
        "config": config_payload(config),
        "snapshots": [_snapshot_payload(snapshot) for snapshot in selected_snapshots],
        "decisions": [asdict(decision) for decision in decisions],
    }


def _extend_preflight_with_state_dir_check(
    *,
    state_dir_summary: dict[str, object],
    errors: list[str],
    risks: list[dict[str, object]],
) -> None:
    path = str(state_dir_summary.get("path"))
    if state_dir_summary.get("exists"):
        if not state_dir_summary.get("is_dir"):
            message = f"state_dir exists but is not a directory: {path}"
            errors.append(message)
            risks.append(
                {
                    "severity": "error",
                    "code": "state_dir_not_directory",
                    "message": message,
                    "path": path,
                }
            )
            return
        if not state_dir_summary.get("readable") or not state_dir_summary.get("writable"):
            message = f"state_dir is not readable and writable: {path}"
            errors.append(message)
            risks.append(
                {
                    "severity": "error",
                    "code": "state_dir_not_accessible",
                    "message": message,
                    "path": path,
                }
            )
        return

    blocked_by_file = state_dir_summary.get("blocked_by_file")
    if blocked_by_file:
        message = f"state_dir path is blocked by a non-directory parent: {blocked_by_file}"
        errors.append(message)
        risks.append(
            {
                "severity": "error",
                "code": "state_dir_parent_not_directory",
                "message": message,
                "path": path,
                "blocked_by_file": blocked_by_file,
            }
        )
        return

    parent = str(state_dir_summary.get("nearest_existing_parent"))
    if not state_dir_summary.get("parent_writable"):
        message = f"state_dir parent is not writable: {parent}"
        errors.append(message)
        risks.append(
            {
                "severity": "error",
                "code": "state_dir_parent_not_writable",
                "message": message,
                "path": path,
                "parent": parent,
            }
        )


def _extend_preflight_with_pid_file_check(
    *,
    pid_file_summary: dict[str, object],
    errors: list[str],
    warnings: list[str],
    risks: list[dict[str, object]],
) -> None:
    if not pid_file_summary.get("exists"):
        return
    path = str(pid_file_summary.get("path"))
    if not pid_file_summary.get("valid"):
        warnings.append(f"invalid pid file will be removed by guard/start: {path}")
        risks.append(
            {
                "severity": "warning",
                "code": "pid_file_invalid",
                "message": f"invalid pid file will be removed by guard/start: {path}",
                "path": path,
            }
        )
        return
    pid = pid_file_summary.get("pid")
    if not isinstance(pid, int):
        return
    if _is_gpu_holder_guard_process(pid):
        message = f"gpu-holder already appears to be running: {pid}"
        errors.append(message)
        risks.append(
            {
                "severity": "error",
                "code": "pid_file_existing_controller",
                "message": message,
                "pid": pid,
                "path": path,
            }
        )
        return
    if _process_exists(pid):
        message = f"pid file points to a live non-holder process: {pid}"
        errors.append(message)
        risks.append(
            {
                "severity": "error",
                "code": "pid_file_live_non_holder",
                "message": message,
                "pid": pid,
                "path": path,
            }
        )
        return
    message = f"stale pid file will be removed by guard/start: {pid}"
    warnings.append(message)
    risks.append(
        {
            "severity": "warning",
            "code": "pid_file_stale",
            "message": message,
            "pid": pid,
            "path": path,
        }
    )


def print_preflight_report(
    payload: dict[str, object],
    *,
    sink: Callable[[str], None] = print,
) -> None:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    assert isinstance(summary, dict)
    sink(f"gpu-holder preflight ok={payload.get('ok')}")
    sink(
        f"selected_gpus={summary.get('selected_gpu_count')} "
        f"visible_gpus={summary.get('visible_gpu_count')} "
        f"actions={summary.get('action_counts')} "
        f"reasons={summary.get('reason_counts')} "
        f"holders_would_start={summary.get('holders_would_start')}"
    )
    for warning in payload.get("warnings", []):
        sink(f"warning: {warning}")
    for error in payload.get("errors", []):
        sink(f"error: {error}")
    risks = payload.get("risks", [])
    if isinstance(risks, list):
        for risk in risks:
            if not isinstance(risk, dict):
                continue
            sink(
                f"risk[{risk.get('severity')}]: "
                f"{risk.get('code')} - {risk.get('message')}"
            )
    decisions = payload.get("decisions", [])
    if not isinstance(decisions, list):
        return
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        sink(
            f"gpu={decision.get('gpu_index')} action={decision.get('action')} "
            f"reason={decision.get('reason')} mem={decision.get('memory_bytes')} "
            f"duty={float(decision.get('duty_cycle', 0.0)):.2f} "
            f"program={decision.get('program')} mode={decision.get('hold_mode')}"
        )


def build_support_report(
    *,
    state_dir: Path,
    event_limit: int = 20,
    include_diagnostics: bool = True,
    redact: bool = False,
) -> dict[str, object]:
    status_file = state_dir / "status.json"
    event_log = state_dir / "events.jsonl"
    pid_file = state_dir / "gpu-holder.pid"
    log_file = state_dir / "gpu-holder.log"
    disabled_gpus_file = state_dir / "disabled_gpus.json"
    status_payload, status_error = read_status_result(status_file)
    events = read_events(event_log, limit=event_limit)
    diagnostics = run_diagnostics() if include_diagnostics else None
    payload = {
        "schema_version": 1,
        "generated_at": time.time(),
        "gpu_holder_version": __version__,
        "state_dir": str(state_dir),
        "redacted": bool(redact),
        "files": {
            "status_file": _file_summary(status_file),
            "event_log": _file_summary(event_log),
            "pid_file": _file_summary(pid_file),
            "log_file": _file_summary(log_file),
            "disabled_gpus_file": _file_summary(disabled_gpus_file),
        },
        "pid_file": _pid_file_summary(pid_file),
        "disabled_gpus": sorted(read_disabled_gpus(state_dir)),
        "diagnostics": diagnostics,
        "status": status_payload,
        "status_error": status_error,
        "events": events,
    }
    payload["summary"] = _support_report_summary(
        status=status_payload,
        status_error=status_error,
        diagnostics=diagnostics,
        events=events,
    )
    return _redact_support_report(payload) if redact else payload


def _support_report_summary(
    *,
    status: dict[str, object] | None,
    status_error: str | None,
    diagnostics: object,
    events: list[dict[str, object]],
) -> dict[str, object]:
    machine = status.get("machine") if isinstance(status, dict) else None
    if not isinstance(machine, dict):
        machine = {}
    gpus = status.get("gpus") if isinstance(status, dict) else None
    if not isinstance(gpus, list):
        gpus = []
    holder_pids = status.get("holder_pids") if isinstance(status, dict) else None
    disabled_gpus = status.get("disabled_gpus") if isinstance(status, dict) else None
    diagnostics_ok = diagnostics.get("ok") if isinstance(diagnostics, dict) else None
    return {
        "diagnostics_ok": diagnostics_ok,
        "status_present": isinstance(status, dict),
        "status_error": status_error,
        "event_count": len(events),
        "event_type_counts": _event_type_counts(events),
        "gpu_count": len(gpus),
        "machine_average_utilization": machine.get("average_utilization"),
        "machine_policy_average_utilization": machine.get("policy_average_utilization"),
        "policy_window_health": (
            policy_window_health(status) if isinstance(status, dict) else None
        ),
        "utilization_target_forecast": (
            utilization_target_forecast(status) if isinstance(status, dict) else None
        ),
        "utilization_target_health": (
            utilization_target_health(status) if isinstance(status, dict) else None
        ),
        "action_counts": machine.get("action_counts", {}),
        "decision_reason_counts": _decision_reason_counts(gpus),
        "owned_worker_count": _owned_worker_count(gpus),
        "holder_pid_count": len(holder_pids) if isinstance(holder_pids, list) else 0,
        "external_process_count": _external_process_count(gpus),
        "disabled_gpu_count": len(disabled_gpus) if isinstance(disabled_gpus, list) else 0,
    }


def _event_type_counts(events: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        event_type = event.get("type")
        if not isinstance(event_type, str):
            continue
        counts[event_type] = counts.get(event_type, 0) + 1
    return counts


def _decision_reason_counts(gpus: list[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for gpu in gpus:
        if not isinstance(gpu, dict):
            continue
        decision = gpu.get("decision")
        if not isinstance(decision, dict):
            continue
        reason = decision.get("reason")
        if not isinstance(reason, str):
            continue
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _owned_worker_count(gpus: list[object]) -> int:
    count = 0
    for gpu in gpus:
        if not isinstance(gpu, dict):
            continue
        worker = gpu.get("worker")
        if isinstance(worker, dict) and worker:
            count += 1
    return count


def _external_process_count(gpus: list[object]) -> int:
    count = 0
    for gpu in gpus:
        if not isinstance(gpu, dict):
            continue
        processes = gpu.get("processes")
        if not isinstance(processes, list):
            continue
        for process in processes:
            if isinstance(process, dict) and not bool(process.get("is_holder")):
                count += 1
    return count


def _redact_support_report(payload: dict[str, object]) -> dict[str, object]:
    redacted = dict(payload)
    redacted["state_dir"] = "<redacted>"
    files = redacted.get("files")
    if isinstance(files, dict):
        redacted["files"] = {
            key: _redact_path_fields(value)
            for key, value in files.items()
        }
    redacted["pid_file"] = _redact_pid_file(redacted.get("pid_file"))
    status = redacted.get("status")
    if isinstance(status, dict):
        redacted["status"] = _redact_status_payload(status)
    events = redacted.get("events")
    if isinstance(events, list):
        redacted["events"] = [_redact_event(event) for event in events]
    return redacted


def _redact_path_fields(value: object) -> object:
    if not isinstance(value, dict):
        return value
    result = dict(value)
    if "path" in result:
        result["path"] = "<redacted>"
    return result


def _redact_pid_file(value: object) -> object:
    if not isinstance(value, dict):
        return value
    result = _redact_path_fields(value)
    if isinstance(result, dict) and result.get("pid") is not None:
        result["pid"] = "<redacted>"
    return result


def _redact_status_payload(status: dict[str, object]) -> dict[str, object]:
    result = dict(status)
    if isinstance(result.get("holder_pids"), list):
        result["holder_pids"] = ["<redacted>" for _item in result["holder_pids"]]
    config = result.get("config")
    if isinstance(config, dict):
        result["config"] = _redact_config_paths(config)
    gpus = result.get("gpus")
    if isinstance(gpus, list):
        result["gpus"] = [_redact_status_gpu(gpu) for gpu in gpus]
    return result


def _redact_config_paths(config: dict[str, object]) -> dict[str, object]:
    result = dict(config)
    for key in (
        "state_dir",
        "pause_file",
        "pid_file",
        "status_file",
        "event_log_file",
        "log_file",
    ):
        if key in result and result[key] is not None:
            result[key] = "<redacted>"
    return result


def _redact_status_gpu(gpu: object) -> object:
    if not isinstance(gpu, dict):
        return gpu
    result = dict(gpu)
    processes = result.get("processes")
    if isinstance(processes, list):
        result["processes"] = [_redact_process(process) for process in processes]
    worker = result.get("worker")
    if isinstance(worker, dict):
        result["worker"] = _redact_pid_fields(worker)
    return result


def _redact_process(process: object) -> object:
    if not isinstance(process, dict):
        return process
    result = _redact_pid_fields(process)
    if isinstance(result, dict) and "name" in result:
        result["name"] = "<redacted>"
    return result


def _redact_event(event: object) -> object:
    if not isinstance(event, dict):
        return event
    return _redact_pid_fields(event)


def _redact_pid_fields(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    if "pid" in result and result["pid"] is not None:
        result["pid"] = "<redacted>"
    if isinstance(result.get("pids"), list):
        result["pids"] = ["<redacted>" for _item in result["pids"]]
    return result


def print_report(payload: dict[str, object], *, sink: Callable[[str], None] = print) -> None:
    diagnostics = payload.get("diagnostics")
    diagnostics_ok = None
    if isinstance(diagnostics, dict):
        diagnostics_ok = diagnostics.get("ok")
    status = payload.get("status")
    has_status = isinstance(status, dict)
    events = payload.get("events")
    event_count = len(events) if isinstance(events, list) else 0
    status_error = payload.get("status_error")
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        summary = {}
    sink("gpu-holder report")
    sink(f"version={payload.get('gpu_holder_version')} state_dir={payload.get('state_dir')}")
    sink(f"diagnostics_ok={diagnostics_ok} status={'present' if has_status else 'missing'} events={event_count}")
    if summary:
        sink(
            "summary "
            f"gpu_count={summary.get('gpu_count')} "
            f"policy_avg={summary.get('machine_policy_average_utilization')} "
            f"{format_policy_window_health(summary.get('policy_window_health'))} "
            f"{format_utilization_target_forecast(summary.get('utilization_target_forecast'))} "
            f"target={_summary_target_status(summary)} "
            f"external_processes={summary.get('external_process_count')} "
            f"workers={summary.get('owned_worker_count')} "
            f"reasons={summary.get('decision_reason_counts')} "
            f"event_types={summary.get('event_type_counts')}"
        )
    if status_error:
        sink(f"status_error={status_error}")
    if has_status:
        machine = status.get("machine") or {}
        if isinstance(machine, dict):
            sink(
                "machine "
                f"gpu_count={machine.get('gpu_count')} "
                f"avg={machine.get('average_utilization')} "
                f"actions={machine.get('action_counts')}"
            )


def _summary_target_status(summary: dict[str, object]) -> object:
    health = summary.get("utilization_target_health")
    if not isinstance(health, dict):
        return None
    return health.get("status")


def print_config_reference(
    payload: dict[str, object],
    *,
    sink: Callable[[str], None] = print,
) -> None:
    sink("gpu-holder config reference")
    fields = payload.get("fields", [])
    if not isinstance(fields, list):
        return
    categories = []
    for field in fields:
        if not isinstance(field, dict):
            continue
        category = str(field.get("category"))
        if category not in categories:
            categories.append(category)
    for category in categories:
        sink("")
        sink(f"[{category}]")
        for field in fields:
            if not isinstance(field, dict) or field.get("category") != category:
                continue
            cli_flag = field.get("cli_flag") or "-"
            default = field.get("default")
            example = field.get("example")
            suffix = f" example={example}" if example is not None else ""
            sink(
                f"{field.get('key')} ({field.get('value_type')}) "
                f"default={default} cli={cli_flag}{suffix}"
            )
            sink(f"  {field.get('description')}")


def print_profile_reference(
    payload: dict[str, object],
    *,
    sink: Callable[[str], None] = print,
) -> None:
    sink("gpu-holder profile reference")
    profiles = payload.get("profiles", [])
    if not isinstance(profiles, list):
        return
    for profile in profiles:
        if not isinstance(profile, dict):
            continue
        sink("")
        sink(f"{profile.get('name')}")
        sink(f"  {profile.get('description')}")
        values = profile.get("values")
        if not isinstance(values, dict) or not values:
            sink("  values: default")
            continue
        sink("  values:")
        for key in sorted(values):
            sink(f"    {key}={values[key]}")


def print_recipe_reference(
    payload: dict[str, object],
    *,
    sink: Callable[[str], None] = print,
) -> None:
    sink("gpu-holder recipes")
    recipes = payload.get("recipes", [])
    if not isinstance(recipes, list):
        return
    for recipe in recipes:
        if not isinstance(recipe, dict):
            continue
        sink("")
        sink(f"{recipe.get('name')} profile={recipe.get('profile')}")
        sink(f"  {recipe.get('description')}")
        values = recipe.get("values")
        if not isinstance(values, dict) or not values:
            continue
        sink("  values:")
        for key in sorted(values):
            sink(f"    {key}={_format_reference_value(values[key])}")
    sink("")
    sink("Print TOML with: gpu-holder recipes --name NAME")


def _format_reference_value(value: object, *, quote_strings: bool = False) -> str:
    if isinstance(value, str):
        if quote_strings:
            return json.dumps(value)
        return value
    if isinstance(value, list | tuple):
        return (
            "["
            + ", ".join(_format_reference_value(item, quote_strings=True) for item in value)
            + "]"
        )
    return str(value)


def print_config_explain(
    payload: dict[str, object],
    *,
    sink: Callable[[str], None] = print,
) -> None:
    sink("config explain")
    sink(f"  profile={payload.get('profile')}")
    sink(f"  precedence={payload.get('precedence')}")
    for key in ("file_overrides", "cli_overrides", "profile_defaults_applied"):
        value = payload.get(key)
        sink(f"  {key}={_format_list_or_map(value)}")
    overridden = payload.get("profile_defaults_overridden")
    if isinstance(overridden, dict) and overridden:
        sink("  profile_defaults_overridden:")
        for key in sorted(overridden):
            item = overridden[key]
            if not isinstance(item, dict):
                continue
            sink(
                f"    {key}: profile={item.get('profile_value')} "
                f"source={item.get('source')} effective={item.get('effective_value')}"
            )


def print_reason_reference(
    payload: dict[str, object],
    *,
    sink: Callable[[str], None] = print,
) -> None:
    sink("gpu-holder reason reference")
    reasons = payload.get("reasons", [])
    if not isinstance(reasons, list):
        return
    for reason in reasons:
        if not isinstance(reason, dict):
            continue
        actions = reason.get("actions")
        action_text = ",".join(str(action) for action in actions) if isinstance(actions, list) else "-"
        sink("")
        sink(f"{reason.get('reason')} actions={action_text} priority={reason.get('priority')}")
        sink(f"  meaning: {reason.get('meaning')}")
        sink(f"  guidance: {reason.get('operator_guidance')}")


def print_history(
    payload: dict[str, object],
    *,
    sink: Callable[[str], None] = print,
) -> None:
    sink(
        "history "
        f"events={payload.get('event_count')} "
        f"decisions={payload.get('decision_count')} "
        f"duration={_format_duration(payload.get('duration_seconds'))} "
        f"first={_format_optional_timestamp(payload.get('first_timestamp'))} "
        f"last={_format_optional_timestamp(payload.get('last_timestamp'))}"
    )
    sink(f"event_types={_format_count_map(payload.get('event_type_counts'))}")
    sink(f"actions={_format_count_map(payload.get('action_counts'))}")
    sink(f"reasons={_format_count_map(payload.get('reason_counts'))}")
    gpus = payload.get("gpus")
    if not isinstance(gpus, dict):
        return
    for gpu_key in sorted(gpus, key=lambda item: int(item) if str(item).isdigit() else str(item)):
        gpu = gpus.get(gpu_key)
        if not isinstance(gpu, dict):
            continue
        sink(
            f"gpu={gpu.get('gpu_index')} "
            f"events={gpu.get('event_count')} "
            f"decisions={gpu.get('decision_count')} "
            f"last_action={gpu.get('last_action')} "
            f"last_reason={gpu.get('last_reason')} "
            f"actions={_format_count_map(gpu.get('action_counts'))} "
            f"reasons={_format_count_map(gpu.get('reason_counts'))} "
            f"workers=start:{gpu.get('worker_starts')} stop:{gpu.get('worker_stops')} "
            f"fail:{gpu.get('worker_start_failures')} backoff:{gpu.get('worker_start_backoffs')} "
            f"thermal=start:{gpu.get('thermal_limit_starts')} end:{gpu.get('thermal_limit_ends')}"
        )


def _format_timestamp(timestamp: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))


def _format_optional_timestamp(value: object) -> str:
    if value is None:
        return "None"
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return str(value)
    return _format_timestamp(timestamp)


def _format_duration(value: object) -> str:
    if value is None:
        return "-"
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return str(value)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _format_count_map(value: object) -> str:
    if not isinstance(value, dict) or not value:
        return "-"
    return ",".join(f"{key}:{value[key]}" for key in sorted(value))


def _format_list_or_map(value: object) -> str:
    if isinstance(value, dict):
        if not value:
            return "-"
        return ",".join(f"{key}:{value[key]}" for key in sorted(value))
    if isinstance(value, list):
        return ",".join(str(item) for item in value) if value else "-"
    return str(value)


def print_status(
    payload: dict[str, object],
    *,
    stale_after: float | None = 15.0,
    now: float | None = None,
    sink: Callable[[str], None] = print,
) -> None:
    payload = enrich_status_health(payload, stale_after=stale_after, now=now)
    machine = payload.get("machine") or {}
    if not isinstance(machine, dict):
        machine = {}
    target_health = payload.get("utilization_target_health") or {}
    if not isinstance(target_health, dict):
        target_health = {}
    sink(
        f"timestamp={payload.get('timestamp')} paused={payload.get('paused')} "
        f"pause_until={_format_optional_timestamp(payload.get('pause_until'))} "
        f"age={format_status_age(payload.get('status_age_seconds'))} "
        f"stale={payload.get('status_stale')} "
        f"disabled_gpus={payload.get('disabled_gpus')} "
        f"machine_avg={machine.get('average_utilization')} "
        f"policy_avg={machine.get('policy_average_utilization')} "
        f"target_health={target_health.get('status')} "
        f"target_gap={target_health.get('gap')} "
        f"{format_policy_window_health(payload.get('policy_window_health'))} "
        f"{format_utilization_target_forecast(payload.get('utilization_target_forecast'))} "
        f"actions={machine.get('action_counts')}"
    )
    for gpu in payload.get("gpus", []):
        if not isinstance(gpu, dict):
            continue
        decision = gpu.get("decision") or {}
        if not isinstance(decision, dict):
            decision = {}
        worker = gpu.get("worker") or {}
        if not isinstance(worker, dict):
            worker = {}
        backoff = gpu.get("worker_restart_backoff") or {}
        if not isinstance(backoff, dict):
            backoff = {}
        process_grace = gpu.get("process_grace") or {}
        if not isinstance(process_grace, dict):
            process_grace = {}
        gpu_disabled = bool(gpu.get("gpu_disabled"))
        worker_text = "-"
        if gpu_disabled:
            disabled_until = gpu.get("gpu_disabled_until")
            if disabled_until is None:
                worker_text = "disabled"
            else:
                worker_text = f"disabled_until={_format_optional_timestamp(disabled_until)}"
        elif gpu.get("thermal_blocked"):
            worker_text = "thermal"
        elif worker:
            worker_text = (
                f"pid={worker.get('pid')} mode={worker.get('hold_mode')} "
                f"duty={worker.get('duty_cycle')}"
            )
        elif backoff:
            remaining = backoff.get("remaining_seconds")
            if isinstance(remaining, float):
                worker_text = f"backoff={remaining:.0f}s"
            else:
                worker_text = f"backoff={remaining}s"
        elif process_grace:
            remaining = process_grace.get("remaining_seconds")
            if isinstance(remaining, float):
                worker_text = f"process_grace={remaining:.0f}s"
            else:
                worker_text = f"process_grace={remaining}s"
        sink(
            f"gpu={gpu.get('index')} util={gpu.get('utilization')}% "
            f"temp={_format_temp(gpu.get('temperature_celsius'))} "
            f"mem={gpu.get('memory_used_human')}/{gpu.get('memory_total_human')} "
            f"action={decision.get('action')} reason={decision.get('reason')} "
            f"hist={decision.get('history_average')} worker={worker_text}"
        )


def _format_temp(value: object) -> str:
    if value is None:
        return "-"
    return f"{value}C"


def _print_event(
    event: dict[str, object],
    *,
    json_output: bool,
    sink: Callable[[str], None],
) -> None:
    if json_output:
        sink(json.dumps(event, separators=(",", ":")))
        return
    extras = " ".join(
        f"{key}={value}"
        for key, value in event.items()
        if key not in {"timestamp", "type"}
    )
    sink(f"timestamp={event.get('timestamp')} type={event.get('type')} {extras}".rstrip())


def _file_summary(path: Path) -> dict[str, object]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {
            "path": str(path),
            "exists": False,
        }
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "modified_at": stat.st_mtime,
    }


def _state_dir_summary(path: Path) -> dict[str, object]:
    target = path.expanduser()
    try:
        exists = target.exists()
    except OSError as exc:
        return {
            "path": str(target),
            "exists": False,
            "error": str(exc),
            "nearest_existing_parent": None,
            "parent_writable": False,
        }
    if exists:
        is_dir = target.is_dir()
        return {
            "path": str(target),
            "exists": True,
            "is_dir": is_dir,
            "readable": is_dir and os.access(target, os.R_OK | os.X_OK),
            "writable": is_dir and os.access(target, os.W_OK | os.X_OK),
        }

    parent, blocked_by_file = _nearest_existing_parent(target.parent)
    return {
        "path": str(target),
        "exists": False,
        "is_dir": False,
        "nearest_existing_parent": str(parent) if parent is not None else None,
        "blocked_by_file": str(blocked_by_file) if blocked_by_file is not None else None,
        "parent_writable": (
            blocked_by_file is None
            and parent is not None
            and os.access(parent, os.W_OK | os.X_OK)
        ),
    }


def _nearest_existing_parent(path: Path) -> tuple[Path | None, Path | None]:
    current = path.expanduser()
    while True:
        if current.exists():
            if current.is_dir():
                return current, None
            return current.parent, current
        if current.parent == current:
            return None, None
        current = current.parent


def _pid_file_summary(path: Path) -> dict[str, object]:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return {
            "path": str(path),
            "exists": False,
            "pid": None,
            "valid": False,
        }
    except OSError as exc:
        return {
            "path": str(path),
            "exists": False,
            "pid": None,
            "valid": False,
            "error": str(exc),
        }
    try:
        pid = int(raw)
    except ValueError:
        return {
            "path": str(path),
            "exists": True,
            "pid": None,
            "valid": False,
            "raw": raw,
        }
    return {
        "path": str(path),
        "exists": True,
        "pid": pid,
        "valid": True,
    }


def _print_config_errors(config: GuardConfig) -> bool:
    errors, warnings = validate_config(config)
    for warning in warnings:
        print(f"config warning: {warning}", file=sys.stderr)
    for error in errors:
        print(f"config error: {error}", file=sys.stderr)
    return not errors


def _config_from_args(args: argparse.Namespace) -> GuardConfig:
    file_config, source_errors = _load_file_config_from_args(args)
    default = GuardConfig()
    profile = str(_raw_config_value(args, file_config, "profile", default.profile))
    try:
        profiled_config = {**profile_defaults(profile), **file_config}
    except ValueError as exc:
        source_errors = (*source_errors, str(exc))
        profiled_config = dict(file_config)
    try:
        return GuardConfig(
            profile=str(_config_value(args, profiled_config, "profile", default.profile)),
            gpus=_parse_gpus(_config_value(args, profiled_config, "gpus", default.gpus)),
            target_util=int(_config_value(args, profiled_config, "target_util", default.target_util)),
            idle_util=int(_config_value(args, profiled_config, "idle_util", default.idle_util)),
            idle_window=float(_config_value(args, profiled_config, "idle_window", default.idle_window)),
            machine_window=float(
                _config_value(args, profiled_config, "machine_window", default.machine_window)
            ),
            mem=parse_memory_spec(str(_config_value(args, profiled_config, "mem", default.mem.raw))),
            reserve=parse_memory_spec(
                str(_config_value(args, profiled_config, "reserve", default.reserve.raw))
            ),
            busy_process_mem_threshold=parse_memory_spec(
                str(
                    _config_value(
                        args,
                        profiled_config,
                        "busy_process_mem_threshold",
                        default.busy_process_mem_threshold.raw,
                    )
                )
            ),
            assist_mem=parse_memory_spec(
                str(_config_value(args, profiled_config, "assist_mem", default.assist_mem.raw))
            ),
            max_gpu_temp=_optional_positive_int(
                _config_value(args, profiled_config, "max_gpu_temp", default.max_gpu_temp)
            ),
            thermal_resume_temp=_optional_positive_int(
                _config_value(args, profiled_config, "thermal_resume_temp", default.thermal_resume_temp)
            ),
            sample_interval=float(
                _config_value(args, profiled_config, "sample_interval", default.sample_interval)
            ),
            min_duty_cycle=float(
                _config_value(args, profiled_config, "min_duty_cycle", default.min_duty_cycle)
            ),
            max_duty_cycle=float(
                _config_value(args, profiled_config, "max_duty_cycle", default.max_duty_cycle)
            ),
            current_gap_weight=float(
                _config_value(args, profiled_config, "current_gap_weight", default.current_gap_weight)
            ),
            history_gap_weight=float(
                _config_value(args, profiled_config, "history_gap_weight", default.history_gap_weight)
            ),
            machine_gap_weight=float(
                _config_value(args, profiled_config, "machine_gap_weight", default.machine_gap_weight)
            ),
            event_log_max_bytes=_parse_byte_size(
                _config_value(args, profiled_config, "event_log_max_size", default.event_log_max_bytes)
            ),
            event_log_backup_count=int(
                _config_value(args, profiled_config, "event_log_backup_count", default.event_log_backup_count)
            ),
            worker_restart_backoff=float(
                _config_value(args, profiled_config, "worker_restart_backoff", default.worker_restart_backoff)
            ),
            worker_start_timeout=float(
                _config_value(args, profiled_config, "worker_start_timeout", default.worker_start_timeout)
            ),
            worker_update_duty_threshold=float(
                _config_value(
                    args,
                    profiled_config,
                    "worker_update_duty_threshold",
                    default.worker_update_duty_threshold,
                )
            ),
            process_grace_window=float(
                _config_value(args, profiled_config, "process_grace_window", default.process_grace_window)
            ),
            protected_process_patterns=_parse_patterns(
                _config_value(
                    args,
                    profiled_config,
                    "protected_process_patterns",
                    default.protected_process_patterns,
                )
            ),
            max_held_gpus=_optional_int(
                _config_value(args, profiled_config, "max_held_gpus", default.max_held_gpus)
            ),
            program=str(_config_value(args, profiled_config, "program", default.program)),
            hold_mode=str(_config_value(args, profiled_config, "hold_mode", default.hold_mode)),
            compute_burst_seconds=float(
                _config_value(args, profiled_config, "compute_burst_seconds", default.compute_burst_seconds)
            ),
            compute_burst_jitter=float(
                _config_value(args, profiled_config, "compute_burst_jitter", default.compute_burst_jitter)
            ),
            state_dir=Path(
                str(_config_value(args, profiled_config, "state_dir", default.state_dir))
            ).expanduser(),
            pause_file=_optional_path(_config_value(args, profiled_config, "pause_file", None)),
            dry_run=bool(args.dry_run),
            tui=bool(args.tui),
            source_errors=source_errors,
        )
    except (argparse.ArgumentTypeError, TypeError, ValueError) as exc:
        return GuardConfig(source_errors=(*source_errors, f"invalid config value: {exc}"))


def _config_explain_from_args(args: argparse.Namespace, config: GuardConfig) -> dict[str, object]:
    file_config, source_errors = _load_file_config_from_args(args)
    default = GuardConfig()
    profile = str(_raw_config_value(args, file_config, "profile", default.profile))
    try:
        profile_values = profile_defaults(profile)
    except ValueError:
        profile_values = {}
    effective = config_payload(config)
    file_keys = sorted(_normalized_file_keys(file_config))
    cli_keys = sorted(_cli_config_keys(args))
    overridden: dict[str, dict[str, object]] = {}
    applied: dict[str, object] = {}
    for key, profile_value in sorted(profile_values.items()):
        source = None
        if key in cli_keys:
            source = "cli"
        elif key in file_keys:
            source = "config"
        if source is None:
            applied[key] = profile_value
            continue
        overridden[key] = {
            "profile_value": profile_value,
            "source": source,
            "effective_value": effective.get(_payload_key_for_config_key(key)),
        }
    profile_description = None
    for item in profile_reference():
        if item.get("name") == profile:
            profile_description = item.get("description")
            break
    return {
        "profile": profile,
        "profile_description": profile_description,
        "source_errors": list(source_errors),
        "precedence": "defaults -> profile -> config file -> CLI",
        "file_overrides": file_keys,
        "cli_overrides": cli_keys,
        "profile_values": profile_values,
        "profile_defaults_applied": applied,
        "profile_defaults_overridden": overridden,
    }


def _load_file_config_from_args(args: argparse.Namespace) -> tuple[dict[str, object], tuple[str, ...]]:
    if not args.config:
        return {}, ()
    try:
        file_config = load_config_file(args.config)
    except (OSError, ValueError) as exc:
        return {}, (f"failed to load config file {args.config}: {exc}",)
    return file_config, tuple(validate_config_keys(file_config))


def _normalized_file_keys(file_config: dict[str, object]) -> set[str]:
    return {str(key).replace("-", "_") for key in file_config}


def _cli_config_keys(args: argparse.Namespace) -> set[str]:
    keys: set[str] = set()
    for field in config_reference():
        key = field.get("key")
        if isinstance(key, str) and getattr(args, key, None) is not None:
            keys.add(key)
    return keys


def _payload_key_for_config_key(key: str) -> str:
    if key == "event_log_max_size":
        return "event_log_max_bytes"
    return key


def _config_value(
    args: argparse.Namespace,
    file_config: dict[str, object],
    key: str,
    default: object,
) -> object:
    cli_value = getattr(args, key, None)
    if cli_value is not None:
        return cli_value
    return file_config.get(key.replace("_", "-"), file_config.get(key, default))


def _raw_config_value(
    args: argparse.Namespace,
    file_config: dict[str, object],
    key: str,
    default: object,
) -> object:
    cli_value = getattr(args, key, None)
    if cli_value is not None:
        return cli_value
    return file_config.get(key.replace("_", "-"), file_config.get(key, default))


def _optional_path(value: object) -> Path | None:
    if value is None:
        return None
    return Path(str(value)).expanduser()


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_positive_int(value: object) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed == 0:
        return None
    return parsed


def _parse_gpus(raw: str) -> tuple[int, ...] | str:
    if isinstance(raw, (list, tuple)):
        return tuple(int(item) for item in raw)
    text = str(raw).strip().lower()
    if text == "all":
        return "all"
    if not text:
        raise argparse.ArgumentTypeError("gpus cannot be empty")
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def _parse_patterns(raw: object) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, (list, tuple)):
        values = list(raw)
    else:
        values = [raw]
    patterns: list[str] = []
    for value in values:
        for item in str(value).split(","):
            pattern = item.strip()
            if pattern:
                patterns.append(pattern)
    return tuple(patterns)


def _parse_event_types(raw_values: list[str] | None) -> set[str] | None:
    if not raw_values:
        return None
    values: set[str] = set()
    for raw in raw_values:
        values.update(item.strip() for item in str(raw).split(",") if item.strip())
    return values or None


def _parse_gpu_filter(raw: str | None) -> set[int] | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    return {int(item.strip()) for item in text.split(",") if item.strip()}


def _read_snapshots(config: GuardConfig) -> list[GpuSnapshot]:
    from .monitor import NvmlMonitor

    monitor = NvmlMonitor()
    try:
        if config.gpus == "all":
            indices = None
        else:
            indices = tuple(config.gpus)
        return monitor.snapshots(indices)
    finally:
        monitor.close()


def _decisions_for_snapshots(
    snapshots: list[GpuSnapshot],
    config: GuardConfig,
) -> list[HolderDecision]:
    histories: dict[int, GpuHistory] = {}
    for snapshot in snapshots:
        history = GpuHistory(window_seconds=config.idle_window)
        for offset in range(int(config.idle_window) + 1):
            history.add(snapshot.timestamp - config.idle_window + offset, snapshot.utilization)
        histories[snapshot.index] = history
    return decide_for_fleet(
        snapshots,
        histories,
        config,
        running_gpu_indices=set(),
        paused=False,
    )


def _fake_snapshots() -> list[GpuSnapshot]:
    now = time.time()
    total = 80 * 1024**3
    return [
        GpuSnapshot(
            index=0,
            name="Fake GPU idle",
            memory_total_bytes=total,
            memory_used_bytes=2 * 1024**3,
            utilization=12,
            temperature_celsius=45,
            timestamp=now,
        ),
        GpuSnapshot(
            index=1,
            name="Fake GPU busy-idle",
            memory_total_bytes=total,
            memory_used_bytes=24 * 1024**3,
            utilization=15,
            temperature_celsius=55,
            processes=(GpuProcess(pid=1234, used_memory_bytes=20 * 1024**3, name="train.py"),),
            timestamp=now,
        ),
        GpuSnapshot(
            index=2,
            name="Fake GPU busy",
            memory_total_bytes=total,
            memory_used_bytes=24 * 1024**3,
            utilization=80,
            temperature_celsius=72,
            processes=(GpuProcess(pid=5678, used_memory_bytes=20 * 1024**3, name="train.py"),),
            timestamp=now,
        ),
    ]


def _snapshot_payload(snapshot: GpuSnapshot) -> dict[str, object]:
    return {
        "index": snapshot.index,
        "name": snapshot.name,
        "memory_total": snapshot.memory_total_bytes,
        "memory_used": snapshot.memory_used_bytes,
        "utilization": snapshot.utilization,
        "temperature_celsius": snapshot.temperature_celsius,
        "timestamp": snapshot.timestamp,
        "processes": [asdict(process) for process in snapshot.processes],
    }


def _filter_snapshots(snapshots: list[GpuSnapshot], gpus: tuple[int, ...] | str) -> list[GpuSnapshot]:
    if gpus == "all":
        return snapshots
    selected = set(gpus)
    return [snapshot for snapshot in snapshots if snapshot.index in selected]


def _missing_selected_gpus(
    snapshots: list[GpuSnapshot],
    gpus: tuple[int, ...] | str,
) -> list[int]:
    if gpus == "all":
        return []
    visible = {snapshot.index for snapshot in snapshots}
    return sorted(index for index in gpus if index not in visible)


def _action_counts(decisions: list[HolderDecision]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for decision in decisions:
        action = getattr(decision, "action", None)
        value = getattr(action, "value", str(action))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _reason_counts(decisions: list[HolderDecision]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for decision in decisions:
        counts[decision.reason] = counts.get(decision.reason, 0) + 1
    return counts


def _guard_args_for_child(args: argparse.Namespace, config: GuardConfig) -> list[str]:
    result = [
        "--profile",
        config.profile,
        "--gpus",
        _format_gpus(config.gpus),
        "--target-util",
        str(config.target_util),
        "--idle-util",
        str(config.idle_util),
        "--idle-window",
        str(config.idle_window),
        "--machine-window",
        str(config.machine_window),
        "--mem",
        config.mem.raw,
        "--reserve",
        config.reserve.raw,
        "--busy-process-mem-threshold",
        config.busy_process_mem_threshold.raw,
        "--assist-mem",
        config.assist_mem.raw,
        "--max-gpu-temp",
        str(config.max_gpu_temp or 0),
        "--thermal-resume-temp",
        str(config.thermal_resume_temp or 0),
        "--sample-interval",
        str(config.sample_interval),
        "--min-duty-cycle",
        str(config.min_duty_cycle),
        "--max-duty-cycle",
        str(config.max_duty_cycle),
        "--current-gap-weight",
        str(config.current_gap_weight),
        "--history-gap-weight",
        str(config.history_gap_weight),
        "--machine-gap-weight",
        str(config.machine_gap_weight),
        "--event-log-max-size",
        str(config.event_log_max_bytes),
        "--event-log-backup-count",
        str(config.event_log_backup_count),
        "--worker-restart-backoff",
        str(config.worker_restart_backoff),
        "--worker-start-timeout",
        str(config.worker_start_timeout),
        "--worker-update-duty-threshold",
        str(config.worker_update_duty_threshold),
        "--process-grace-window",
        str(config.process_grace_window),
        "--program",
        config.program,
        "--hold-mode",
        config.hold_mode,
        "--compute-burst-seconds",
        str(config.compute_burst_seconds),
        "--compute-burst-jitter",
        str(config.compute_burst_jitter),
        "--state-dir",
        str(config.state_dir),
    ]
    for pattern in config.protected_process_patterns:
        result.extend(["--protected-process", pattern])
    if config.max_held_gpus is not None:
        result.extend(["--max-held-gpus", str(config.max_held_gpus)])
    if config.pause_file:
        result.extend(["--pause-file", str(config.pause_file)])
    if config.dry_run:
        result.append("--dry-run")
    if config.tui:
        result.append("--tui")
    return result


def _format_gpus(gpus: tuple[int, ...] | str) -> str:
    if gpus == "all":
        return "all"
    return ",".join(str(index) for index in gpus)


def _parse_byte_size(raw: object) -> int:
    if isinstance(raw, int):
        return raw
    text = str(raw).strip()
    if text.isdigit():
        return int(text)
    spec = parse_memory_spec(text)
    if spec.bytes_value is None:
        raise argparse.ArgumentTypeError("byte size must use an absolute unit, not a percent")
    return spec.bytes_value


def _is_gpu_holder_guard_process(pid: int) -> bool:
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except OSError:
        return False
    parts = [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]
    joined = " ".join(parts)
    return "gpu_holder.cli" in joined and " guard" in f" {joined} "


def _process_exists(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
