from __future__ import annotations

import curses
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

from .advisor import config_from_status_config, tuning_advice
from .config import GuardConfig, config_reference
from .explain import explain_runtime, format_explanation
from .history import summarize_events
from .models import GpuSnapshot, HolderDecision
from .reasons import reason_reference
from .state import disable_gpu, enable_gpu, pause_guard, read_disabled_gpus, resume_guard
from .status_health import (
    enrich_status_health,
    format_policy_window_health,
    format_status_age,
    format_utilization_target_forecast,
)
from .units import format_bytes


DEFAULT_TEMPORARY_DISABLE_SECONDS = 30.0 * 60.0
DEFAULT_TEMPORARY_PAUSE_SECONDS = 10.0 * 60.0


def _format_duration(value: object) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return str(value)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def dashboard_help_lines(
    *,
    temporary_pause_seconds: float = DEFAULT_TEMPORARY_PAUSE_SECONDS,
    temporary_disable_seconds: float = DEFAULT_TEMPORARY_DISABLE_SECONDS,
) -> list[str]:
    return [
        "gpu-holder dashboard keys",
        "",
        "q / Esc      quit the dashboard",
        "p            pause all holder workers",
        f"P            temporarily pause all holder workers for {_format_duration(temporary_pause_seconds)}",
        "r            resume holder workers",
        "e            show or hide recent events",
        "y            show or hide recent history summary",
        "x            show or hide current explanation and guidance",
        "a            show or hide offline tuning advice",
        "f            filter events/history to the selected GPU",
        "c            show or hide resolved config",
        "j / Down     select next GPU",
        "k / Up       select previous GPU",
        "i / Enter    show or hide details for the selected GPU",
        "d            disable or enable holder workers on the selected GPU",
        f"D            temporarily disable selected GPU for {_format_duration(temporary_disable_seconds)}",
        "h / ?        show or hide this help",
    ]


DASHBOARD_HELP_LINES = dashboard_help_lines()


@dataclass
class DashboardState:
    show_events: bool = False
    show_help: bool = False
    show_detail: bool = False
    show_config: bool = False
    show_history: bool = False
    show_explain: bool = False
    show_advice: bool = False
    filter_events_to_selected_gpu: bool = False
    message: str = ""
    selected_gpu_index: int | None = None
    temporary_disable_seconds: float = DEFAULT_TEMPORARY_DISABLE_SECONDS
    temporary_pause_seconds: float = DEFAULT_TEMPORARY_PAUSE_SECONDS

    @property
    def event_filter_gpu_index(self) -> int | None:
        return self.selected_gpu_index if self.filter_events_to_selected_gpu else None


def render_once(
    *,
    snapshots: list[GpuSnapshot],
    decisions: list[HolderDecision],
    config: GuardConfig,
) -> None:
    decision_by_gpu = {decision.gpu_index: decision for decision in decisions}
    print("\033[2J\033[H", end="")
    print("gpu-holder")
    print(
        f"target={config.target_util}% idle={config.idle_util}%/{config.idle_window:.0f}s "
        f"mem={config.mem.raw} program={config.program} mode={config.hold_mode}"
    )
    print()
    print("GPU  UTIL  TEMP   MEMORY              ACTION   REASON")
    print("---  ----  -----  ------------------  -------  -------------------------------")
    for snapshot in snapshots:
        decision = decision_by_gpu.get(snapshot.index)
        action = decision.action.value if decision is not None else "-"
        reason = decision.reason if decision is not None else "-"
        memory = f"{format_bytes(snapshot.memory_used_bytes)}/{format_bytes(snapshot.memory_total_bytes)}"
        print(
            f"{snapshot.index:>3}  {snapshot.utilization:>3}%  "
            f"{format_temperature(snapshot.temperature_celsius):>5}  "
            f"{memory:<18}  {action:<7}  {reason}"
        )


def format_dashboard_snapshot(
    payload: dict[str, object],
    *,
    events: list[dict[str, Any]] | None = None,
    show_events: bool = False,
    show_config: bool = False,
    show_history: bool = False,
    show_explain: bool = False,
    show_advice: bool = False,
    show_detail: bool = False,
    selected_gpu_index: int | None = None,
    event_filter_gpu_index: int | None = None,
    now: float | None = None,
) -> list[str]:
    payload = enrich_status_health(payload, now=now)
    lines = ["gpu-holder dashboard"]
    if show_config:
        lines.extend(format_config_lines(payload.get("config")))
        return lines
    if show_advice:
        lines.extend(format_advice_lines(payload.get("config")))
        return lines

    machine = payload.get("machine") or {}
    if not isinstance(machine, dict):
        machine = {}
    lines.append(
        f"timestamp={payload.get('timestamp')} "
        f"age={format_status_age(payload.get('status_age_seconds'))} "
        f"stale={payload.get('status_stale')} "
        f"paused={payload.get('paused')} "
        f"machine_avg={machine.get('average_utilization')} "
        f"policy_avg={machine.get('policy_average_utilization')} "
        f"{format_policy_window_health(payload.get('policy_window_health'))} "
        f"{format_utilization_target_forecast(payload.get('utilization_target_forecast'))} "
        f"{format_target_health(payload.get('utilization_target_health'))}"
    )
    lines.append("GPU snapshot")
    gpus = payload.get("gpus", [])
    if isinstance(gpus, list):
        for gpu in gpus:
            if not isinstance(gpu, dict):
                continue
            decision = gpu.get("decision") or {}
            if not isinstance(decision, dict):
                decision = {}
            processes = gpu.get("processes") or []
            process_count = len(processes) if isinstance(processes, list) else 0
            lines.append(
                f"gpu={gpu.get('index')} "
                f"util={gpu.get('utilization')}% "
                f"temp={format_temperature(gpu.get('temperature_celsius'))} "
                f"hist={_decision_value(decision, 'history_average')} "
                f"mem={gpu.get('memory_used_human')}/{gpu.get('memory_total_human')} "
                f"action={decision.get('action')} "
                f"reason={decision.get('reason')} "
                f"worker={format_worker_label(gpu)} "
                f"processes={process_count}"
            )
        if show_detail:
            lines.extend(format_gpu_detail_lines(selected_gpu_payload(gpus, selected_gpu_index)))

    if show_events:
        filtered_events = filter_dashboard_events(events or [], gpu_index=event_filter_gpu_index)
        title = (
            "Recent events"
            if event_filter_gpu_index is None
            else f"Recent events for gpu={event_filter_gpu_index}"
        )
        lines.append(title)
        if not filtered_events:
            lines.append("  none")
        for event in filtered_events:
            lines.append(format_event_line(event))
    if show_history:
        lines.extend(format_history_lines(events or [], gpu_index=event_filter_gpu_index))
    if show_explain:
        lines.extend(format_explain_lines(payload, events or [], now=now))
    return lines


def run_status_dashboard(
    *,
    status_reader,
    event_reader,
    state_dir: Path,
    refresh_interval: float = 2.0,
    temporary_disable_seconds: float = DEFAULT_TEMPORARY_DISABLE_SECONDS,
    temporary_pause_seconds: float = DEFAULT_TEMPORARY_PAUSE_SECONDS,
) -> None:
    curses.wrapper(
        _dashboard_main,
        status_reader,
        event_reader,
        state_dir,
        refresh_interval,
        temporary_disable_seconds,
        temporary_pause_seconds,
    )


def _dashboard_main(
    stdscr,
    status_reader,
    event_reader,
    state_dir: Path,
    refresh_interval: float,
    temporary_disable_seconds: float,
    temporary_pause_seconds: float,
) -> None:
    curses.curs_set(0)
    stdscr.nodelay(True)
    state = DashboardState(
        temporary_disable_seconds=temporary_disable_seconds,
        temporary_pause_seconds=temporary_pause_seconds,
    )
    while True:
        payload = status_reader()
        key = stdscr.getch()
        if not handle_dashboard_key(key, payload, state, state_dir):
            break
        events = event_reader() if state.show_events or state.show_history or state.show_explain else []
        _draw_status(
            stdscr,
            payload,
            events=events,
            show_events=state.show_events,
            show_history=state.show_history,
            show_explain=state.show_explain,
            show_advice=state.show_advice,
            show_help=state.show_help,
            show_detail=state.show_detail,
            show_config=state.show_config,
            temporary_pause_seconds=state.temporary_pause_seconds,
            temporary_disable_seconds=state.temporary_disable_seconds,
            event_filter_gpu_index=state.event_filter_gpu_index,
            message=state.message,
            selected_gpu_index=state.selected_gpu_index,
        )
        stdscr.refresh()
        time.sleep(max(0.2, float(refresh_interval)))


def handle_dashboard_key(
    key: int,
    payload: dict[str, object] | None,
    state: DashboardState,
    state_dir: Path,
) -> bool:
    if key in {ord("q"), ord("Q"), 27}:
        return False
    if key in {ord("h"), ord("H"), ord("?")}:
        state.show_help = not state.show_help
        state.message = "help shown" if state.show_help else "help hidden"
    elif key in {ord("i"), ord("I"), 10, 13}:
        if state.selected_gpu_index is None:
            state.selected_gpu_index = move_selection(payload, state.selected_gpu_index, 1)
        state.show_detail = not state.show_detail
        state.message = "details shown" if state.show_detail else "details hidden"
    elif key in {ord("c"), ord("C")}:
        state.show_config = not state.show_config
        state.message = "config shown" if state.show_config else "config hidden"
    elif key == ord("p"):
        pause_file = pause_guard(state_dir)
        state.message = f"paused via {pause_file}"
    elif key == ord("P"):
        now = time.time()
        pause_file = pause_guard(
            state_dir,
            duration_seconds=state.temporary_pause_seconds,
            now=now,
        )
        duration = _format_duration(state.temporary_pause_seconds)
        state.message = (
            f"paused for {duration} "
            f"until={_format_optional_timestamp(now + state.temporary_pause_seconds)} via {pause_file}"
        )
    elif key in {ord("r"), ord("R")}:
        pause_file = resume_guard(state_dir)
        state.message = f"resumed via {pause_file}"
    elif key in {ord("e"), ord("E")}:
        state.show_events = not state.show_events
        state.message = "events shown" if state.show_events else "events hidden"
    elif key in {ord("y"), ord("Y")}:
        state.show_history = not state.show_history
        state.message = "history shown" if state.show_history else "history hidden"
    elif key in {ord("x"), ord("X")}:
        state.show_explain = not state.show_explain
        state.message = "explain shown" if state.show_explain else "explain hidden"
    elif key in {ord("a"), ord("A")}:
        state.show_advice = not state.show_advice
        state.message = "advice shown" if state.show_advice else "advice hidden"
    elif key in {ord("f"), ord("F")}:
        if state.selected_gpu_index is None:
            state.selected_gpu_index = move_selection(payload, state.selected_gpu_index, 1)
        state.filter_events_to_selected_gpu = not state.filter_events_to_selected_gpu
        state.show_events = True
        state.message = (
            "events/history filtered to selected GPU"
            if state.filter_events_to_selected_gpu
            else "event/history filter cleared"
        )
    elif key in {curses.KEY_DOWN, ord("j"), ord("J")}:
        state.selected_gpu_index = move_selection(payload, state.selected_gpu_index, 1)
    elif key in {curses.KEY_UP, ord("k"), ord("K")}:
        state.selected_gpu_index = move_selection(payload, state.selected_gpu_index, -1)
    elif key == ord("d") and state.selected_gpu_index is not None:
        disabled = read_disabled_gpus(state_dir)
        if state.selected_gpu_index in disabled:
            path = enable_gpu(state_dir, state.selected_gpu_index)
            state.message = f"enabled gpu={state.selected_gpu_index} via {path}"
        else:
            path = disable_gpu(state_dir, state.selected_gpu_index)
            state.message = f"disabled gpu={state.selected_gpu_index} via {path}"
    elif key == ord("D") and state.selected_gpu_index is not None:
        now = time.time()
        path = disable_gpu(
            state_dir,
            state.selected_gpu_index,
            duration_seconds=state.temporary_disable_seconds,
            now=now,
        )
        duration = _format_duration(state.temporary_disable_seconds)
        state.message = (
            f"disabled gpu={state.selected_gpu_index} for {duration} "
            f"until={_format_optional_timestamp(now + state.temporary_disable_seconds)} via {path}"
        )
    return True


def _draw_status(
    stdscr,
    payload: dict[str, object] | None,
    *,
    events: list[dict[str, Any]] | None = None,
    show_events: bool = False,
    show_history: bool = False,
    show_explain: bool = False,
    show_advice: bool = False,
    show_help: bool = False,
    show_detail: bool = False,
    show_config: bool = False,
    temporary_pause_seconds: float = DEFAULT_TEMPORARY_PAUSE_SECONDS,
    temporary_disable_seconds: float = DEFAULT_TEMPORARY_DISABLE_SECONDS,
    event_filter_gpu_index: int | None = None,
    message: str = "",
    selected_gpu_index: int | None = None,
) -> None:
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    _add_line(stdscr, 0, 0, "gpu-holder dashboard", width, curses.A_BOLD)
    _add_line(
        stdscr,
        1,
        0,
        "q: quit   h/?: help   c/a: config/advice   p/P/r: pause/temp/resume   e/y/x/f: events/history/explain/filter   j/k: select   i: details   d/D: disable/temp",
        width,
    )
    if message:
        _add_line(stdscr, 2, 0, message, width, curses.A_BOLD)
    if show_help:
        _draw_help(
            stdscr,
            4,
            width,
            temporary_pause_seconds=temporary_pause_seconds,
            temporary_disable_seconds=temporary_disable_seconds,
        )
        return
    if payload is None:
        _add_line(stdscr, 4, 0, "No status file yet. Start `gpu-holder guard` first.", width)
        return
    payload = enrich_status_health(payload)
    if show_config:
        _draw_config(stdscr, 4, height, width, payload.get("config"))
        return
    if show_advice:
        _draw_advice(stdscr, 4, height, width, payload.get("config"))
        return

    paused = payload.get("paused")
    timestamp = payload.get("timestamp")
    machine = payload.get("machine") or {}
    if not isinstance(machine, dict):
        machine = {}
    avg = machine.get("average_utilization")
    policy_avg = machine.get("policy_average_utilization")
    _add_line(
        stdscr,
        3,
        0,
        f"timestamp={timestamp} age={format_status_age(payload.get('status_age_seconds'))} "
        f"stale={payload.get('status_stale')} paused={paused} "
        f"machine_avg={avg} policy_avg={policy_avg} "
        f"{format_policy_window_health(payload.get('policy_window_health'))} "
        f"{format_utilization_target_forecast(payload.get('utilization_target_forecast'))} "
        f"{format_target_health(payload.get('utilization_target_health'))}",
        width,
    )
    _add_line(
        stdscr,
        5,
        0,
        "GPU  UTIL  TEMP   HIST  MEMORY              ACTION   WORKER             REASON                         PROCS",
        width,
        curses.A_BOLD,
    )
    gpus = payload.get("gpus", [])
    if not isinstance(gpus, list):
        return
    selected_gpu = selected_gpu_payload(gpus, selected_gpu_index)
    row = 6
    bottom_panel_start = max(8, height - 8) if show_events or show_history or show_explain else height
    for gpu in gpus:
        if row >= bottom_panel_start - 1:
            _add_line(stdscr, row, 0, "...", width)
            break
        if not isinstance(gpu, dict):
            continue
        decision = gpu.get("decision") or {}
        if not isinstance(decision, dict):
            decision = {}
        processes = gpu.get("processes") or []
        process_count = len(processes) if isinstance(processes, list) else 0
        worker_label = format_worker_label(gpu)
        text = (
            f"{_selection_marker(gpu, selected_gpu_index)}{str(gpu.get('index')):>2}  "
            f"{str(gpu.get('utilization')):>3}%  "
            f"{format_temperature(gpu.get('temperature_celsius')):>5}  "
            f"{_decision_value(decision, 'history_average'):>4}  "
            f"{gpu.get('memory_used_human')}/{gpu.get('memory_total_human'):<10}  "
            f"{str(decision.get('action')):<7}  "
            f"{worker_label:<17}  "
            f"{str(decision.get('reason')):<30}  "
            f"{process_count}"
        )
        attr = _action_attr(str(decision.get("action")))
        _add_line(stdscr, row, 0, text, width, attr)
        row += 1
    if show_detail:
        row = _draw_gpu_detail(stdscr, row + 1, bottom_panel_start, width, selected_gpu)
    if show_explain:
        _draw_explain(
            stdscr,
            bottom_panel_start,
            width,
            payload,
            events or [],
        )
        return
    if show_history:
        _draw_history(
            stdscr,
            bottom_panel_start,
            width,
            events or [],
            filter_gpu_index=event_filter_gpu_index,
        )
        return
    if show_events:
        _draw_events(
            stdscr,
            bottom_panel_start,
            width,
            events or [],
            filter_gpu_index=event_filter_gpu_index,
        )


def _draw_help(
    stdscr,
    start_row: int,
    width: int,
    *,
    temporary_pause_seconds: float = DEFAULT_TEMPORARY_PAUSE_SECONDS,
    temporary_disable_seconds: float = DEFAULT_TEMPORARY_DISABLE_SECONDS,
) -> None:
    lines = dashboard_help_lines(
        temporary_pause_seconds=temporary_pause_seconds,
        temporary_disable_seconds=temporary_disable_seconds,
    )
    for offset, line in enumerate(lines):
        attr = curses.A_BOLD if offset == 0 else 0
        _add_line(stdscr, start_row + offset, 0, line, width, attr)


def _draw_config(stdscr, start_row: int, stop_row: int, width: int, raw_config: object) -> None:
    lines = format_config_lines(raw_config)
    for offset, line in enumerate(lines):
        row = start_row + offset
        if row >= stop_row - 1:
            _add_line(stdscr, row, 0, "...", width)
            return
        attr = curses.A_BOLD if line.startswith("[") or offset == 0 else 0
        _add_line(stdscr, row, 0, line, width, attr)


def _draw_advice(stdscr, start_row: int, stop_row: int, width: int, raw_config: object) -> None:
    lines = format_advice_lines(raw_config)
    for offset, line in enumerate(lines):
        row = start_row + offset
        if row >= stop_row - 1:
            _add_line(stdscr, row, 0, "...", width)
            return
        attr = curses.A_BOLD if offset == 0 else 0
        _add_line(stdscr, row, 0, line, width, attr)


def _draw_gpu_detail(
    stdscr,
    start_row: int,
    stop_row: int,
    width: int,
    gpu: dict[str, Any] | None,
) -> int:
    if start_row >= stop_row:
        return start_row
    lines = format_gpu_detail_lines(gpu)
    for offset, line in enumerate(lines):
        row = start_row + offset
        if row >= stop_row:
            _add_line(stdscr, row - 1, 0, "...", width)
            return row
        attr = curses.A_BOLD if offset == 0 else 0
        _add_line(stdscr, row, 0, line, width, attr)
    return start_row + len(lines)


def _draw_events(
    stdscr,
    start_row: int,
    width: int,
    events: list[dict[str, Any]],
    *,
    filter_gpu_index: int | None = None,
) -> None:
    height, _ = stdscr.getmaxyx()
    if start_row >= height:
        return
    visible_events = filter_dashboard_events(events, gpu_index=filter_gpu_index)
    title = "Recent events" if filter_gpu_index is None else f"Recent events for gpu={filter_gpu_index}"
    _add_line(stdscr, start_row, 0, title, width, curses.A_BOLD)
    row = start_row + 1
    for event in visible_events:
        if row >= height - 1:
            break
        text = format_event_line(event)
        _add_line(stdscr, row, 0, text, width)
        row += 1


def _draw_history(
    stdscr,
    start_row: int,
    width: int,
    events: list[dict[str, Any]],
    *,
    filter_gpu_index: int | None = None,
) -> None:
    height, _ = stdscr.getmaxyx()
    if start_row >= height:
        return
    lines = format_history_lines(events, gpu_index=filter_gpu_index)
    for offset, line in enumerate(lines):
        row = start_row + offset
        if row >= height - 1:
            break
        attr = curses.A_BOLD if offset == 0 else 0
        _add_line(stdscr, row, 0, line, width, attr)


def _draw_explain(
    stdscr,
    start_row: int,
    width: int,
    payload: dict[str, object],
    events: list[dict[str, Any]],
) -> None:
    height, _ = stdscr.getmaxyx()
    if start_row >= height:
        return
    lines = format_explain_lines(payload, events)
    for offset, line in enumerate(lines):
        row = start_row + offset
        if row >= height - 1:
            break
        attr = curses.A_BOLD if offset == 0 else 0
        _add_line(stdscr, row, 0, line, width, attr)


def format_explain_lines(
    payload: dict[str, object],
    events: list[dict[str, Any]],
    *,
    now: float | None = None,
) -> list[str]:
    explanation = explain_runtime(
        status=payload,
        status_error=None,
        events=events,
        now=now,
    )
    lines = format_explanation(explanation)
    if not lines:
        return ["Current explanation", "  none"]
    return ["Current explanation", *[f"  {line}" for line in lines]]


def format_advice_lines(raw_config: object) -> list[str]:
    try:
        config = config_from_status_config(raw_config)
        advice = tuning_advice(config)
    except (TypeError, ValueError) as exc:
        return ["Tuning advice", f"  unavailable: {exc}"]
    summary = advice.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    lines = [
        "Tuning advice",
        (
            f"  grade={advice.get('grade')} score={advice.get('score')} "
            f"warnings={summary.get('warnings')} failures={summary.get('failures')} "
            f"scenarios={summary.get('scenario_count')}"
        ),
    ]
    checks = advice.get("checks", [])
    if isinstance(checks, list):
        for check in checks:
            if not isinstance(check, dict):
                continue
            lines.append(
                f"  check={check.get('name')} status={check.get('status')} "
                f"message={check.get('message')}"
            )
    recommendations = advice.get("recommendations", [])
    if isinstance(recommendations, list) and recommendations:
        lines.append("  recommendations:")
        for recommendation in recommendations:
            lines.append(f"  recommendation: {recommendation}")
    return lines


def format_history_lines(
    events: list[dict[str, Any]],
    *,
    gpu_index: int | None = None,
) -> list[str]:
    gpu_indices = None if gpu_index is None else {int(gpu_index)}
    summary = summarize_events(events, gpu_indices=gpu_indices)
    title = "Recent history" if gpu_index is None else f"Recent history for gpu={gpu_index}"
    if not summary["ok"]:
        return [title, "  none"]
    lines = [
        title,
        (
            f"  events={summary.get('event_count')} "
            f"decisions={summary.get('decision_count')} "
            f"actions={_format_count_map(summary.get('action_counts'))}"
        ),
        f"  reasons={_format_count_map(summary.get('reason_counts'))}",
    ]
    gpus = summary.get("gpus")
    if isinstance(gpus, dict):
        for gpu_key in sorted(gpus, key=lambda item: int(item) if str(item).isdigit() else str(item)):
            gpu = gpus.get(gpu_key)
            if not isinstance(gpu, dict):
                continue
            lines.append(
                f"  gpu={gpu.get('gpu_index')} "
                f"last={gpu.get('last_action')}/{gpu.get('last_reason')} "
                f"workers=start:{gpu.get('worker_starts')} stop:{gpu.get('worker_stops')} "
                f"fail:{gpu.get('worker_start_failures')} "
                f"thermal=start:{gpu.get('thermal_limit_starts')} end:{gpu.get('thermal_limit_ends')}"
            )
    return lines


def format_event_line(event: dict[str, Any]) -> str:
    parts = [f"type={event.get('type')}"]
    for key in ("gpu_index", "action", "reason", "pid", "hold_mode", "program"):
        if key in event:
            parts.append(f"{key}={event[key]}")
    return " ".join(parts)


def filter_dashboard_events(
    events: list[dict[str, Any]],
    *,
    gpu_index: int | None,
) -> list[dict[str, Any]]:
    if gpu_index is None:
        return events
    filtered: list[dict[str, Any]] = []
    for event in events:
        try:
            event_gpu_index = int(event["gpu_index"])
        except (KeyError, TypeError, ValueError):
            continue
        if event_gpu_index == gpu_index:
            filtered.append(event)
    return filtered


def format_worker_label(gpu: dict[str, Any]) -> str:
    if gpu.get("gpu_disabled"):
        disabled_until = gpu.get("gpu_disabled_until")
        if disabled_until is not None:
            return f"disabled until {_format_optional_timestamp(disabled_until)}"
        return "disabled"
    if gpu.get("thermal_blocked"):
        return "thermal"
    worker = gpu.get("worker") or {}
    if isinstance(worker, dict) and worker:
        pid = worker.get("pid")
        mode = worker.get("hold_mode")
        duty = worker.get("duty_cycle")
        if isinstance(duty, float):
            return f"pid={pid} {mode} {duty:.0%}"
        return f"pid={pid} {mode}"
    backoff = gpu.get("worker_restart_backoff") or {}
    if isinstance(backoff, dict) and backoff:
        remaining = backoff.get("remaining_seconds")
        if isinstance(remaining, float):
            return f"backoff {remaining:.0f}s"
        return f"backoff {remaining}s"
    process_grace = gpu.get("process_grace") or {}
    if isinstance(process_grace, dict) and process_grace:
        remaining = process_grace.get("remaining_seconds")
        if isinstance(remaining, float):
            return f"grace {remaining:.0f}s"
        return f"grace {remaining}s"
    return "-"


def format_target_health(raw_health: object) -> str:
    if not isinstance(raw_health, dict):
        return "target=unknown gap=-"
    status = raw_health.get("status") or "unknown"
    gap = raw_health.get("gap")
    if isinstance(gap, (int, float)):
        gap_text = f"{float(gap):.1f}"
    else:
        gap_text = "-"
    return f"target={status} gap={gap_text}"


def format_config_lines(raw_config: object) -> list[str]:
    if not isinstance(raw_config, dict) or not raw_config:
        return ["Resolved config", "No config payload in status.json."]
    lines = ["Resolved config"]
    fields = config_reference()
    categories: list[str] = []
    for field in fields:
        category = str(field["category"])
        if category not in categories:
            categories.append(category)
    rendered_keys: set[str] = set()
    for category in categories:
        category_lines: list[str] = []
        for field in fields:
            if field["category"] != category:
                continue
            key = str(field["key"])
            if key not in raw_config:
                continue
            rendered_keys.add(key)
            category_lines.append(f"  {key}={raw_config.get(key)}")
        if category_lines:
            lines.append(f"[{category}]")
            lines.extend(category_lines)
    extras = sorted(str(key) for key in raw_config if str(key) not in rendered_keys)
    if extras:
        lines.append("[derived]")
        lines.extend(f"  {key}={raw_config.get(key)}" for key in extras)
    return lines


def selected_gpu_payload(gpus: list[object], selected_gpu_index: int | None) -> dict[str, Any] | None:
    if selected_gpu_index is None:
        return None
    for gpu in gpus:
        if isinstance(gpu, dict) and gpu.get("index") == selected_gpu_index:
            return gpu
    return None


def format_gpu_detail_lines(gpu: dict[str, Any] | None) -> list[str]:
    if gpu is None:
        return ["Selected GPU", "No GPU selected. Use j/k to select a row."]
    decision = gpu.get("decision") or {}
    if not isinstance(decision, dict):
        decision = {}
    processes = gpu.get("processes") or []
    process_lines: list[str] = []
    if isinstance(processes, list) and processes:
        for process in processes[:5]:
            if not isinstance(process, dict):
                continue
            process_lines.append(
                "  "
                f"pid={process.get('pid')} "
                f"name={process.get('name') or '-'} "
                f"mem={_format_process_memory(process.get('used_memory_bytes'))} "
                f"holder={process.get('is_holder')} "
                f"protected={process.get('protected_process', False)} "
                f"pattern={process.get('protected_process_pattern') or '-'}"
            )
    if not process_lines:
        process_lines = ["  none"]
    elif isinstance(processes, list) and len(processes) > 5:
        process_lines.append(f"  ... {len(processes) - 5} more")
    return [
        f"Selected GPU {gpu.get('index')}: {gpu.get('name')}",
        (
            f"  util={gpu.get('utilization')}% "
            f"temp={format_temperature(gpu.get('temperature_celsius'))} "
            f"thermal_blocked={bool(gpu.get('thermal_blocked'))}"
        ),
        f"  gpu_disabled={bool(gpu.get('gpu_disabled'))} until={_format_optional_timestamp(gpu.get('gpu_disabled_until'))}",
        f"  memory={gpu.get('memory_used_human')}/{gpu.get('memory_total_human')}",
        (
            f"  decision action={decision.get('action')} "
            f"reason={decision.get('reason')} "
            f"duty={decision.get('duty_cycle')} "
            f"hold_mode={decision.get('hold_mode')}"
        ),
        *format_reason_help_lines(decision.get("reason")),
        f"  worker={format_worker_label(gpu)}",
        "  processes:",
        *process_lines,
    ]


def format_reason_help_lines(reason: object) -> list[str]:
    reason_text = str(reason or "-")
    for item in reason_reference():
        if item.get("reason") != reason_text:
            continue
        return [
            f"  reason meaning={item.get('meaning')}",
            f"  reason guidance={item.get('operator_guidance')}",
        ]
    return [
        f"  reason meaning=No reference entry for {reason_text}.",
        "  reason guidance=Run `gpu-holder reason-reference` to inspect known reasons.",
    ]


def _format_process_memory(value: object) -> str:
    if isinstance(value, int):
        return format_bytes(value)
    return "-"


def _format_optional_timestamp(value: object) -> str:
    if value is None:
        return "-"
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return str(value)
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))


def format_temperature(value: object) -> str:
    if value is None:
        return "-"
    return f"{value}C"


def move_selection(
    payload: dict[str, Any] | None,
    current_gpu_index: int | None,
    delta: int,
) -> int | None:
    if payload is None:
        return current_gpu_index
    gpus = payload.get("gpus", [])
    if not isinstance(gpus, list):
        return current_gpu_index
    indices = [int(gpu["index"]) for gpu in gpus if isinstance(gpu, dict) and "index" in gpu]
    if not indices:
        return current_gpu_index
    if current_gpu_index not in indices:
        return indices[0]
    position = indices.index(current_gpu_index) + int(delta)
    position = max(0, min(len(indices) - 1, position))
    return indices[position]


def _selection_marker(gpu: dict[str, Any], selected_gpu_index: int | None) -> str:
    return ">" if gpu.get("index") == selected_gpu_index else " "


def _action_attr(action: str) -> int:
    if action in {"hold", "assist"}:
        return curses.A_BOLD
    if action == "release":
        return curses.A_DIM
    return 0


def _decision_value(decision: dict[str, object], key: str) -> str:
    value = decision.get(key)
    if isinstance(value, float):
        return f"{value:.0f}%"
    if isinstance(value, int):
        return f"{value}%"
    return "-"


def _format_count_map(value: object) -> str:
    if not isinstance(value, dict) or not value:
        return "-"
    return ",".join(f"{key}:{value[key]}" for key in sorted(value))


def _add_line(stdscr, y: int, x: int, text: str, width: int, attr: int = 0) -> None:
    if y < 0:
        return
    stdscr.addnstr(y, x, text, max(0, width - x - 1), attr)
