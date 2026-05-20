from __future__ import annotations

from typing import Any
import time

from .advisor import config_from_status_config, tuning_advice
from .config import config_reference
from .explain import explain_runtime, format_explanation
from .history import summarize_events
from .reasons import reason_reference
from .status_health import (
    enrich_status_health,
    format_policy_window_health,
    format_status_age,
    format_utilization_target_forecast,
)
from .units import format_bytes


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
    lines = ["gpu-holder status snapshot"]
    if show_config:
        lines.extend(format_config_lines(payload.get("config")))
        return lines
    if show_advice:
        lines.extend(format_advice_lines(payload.get("config")))
        return lines

    machine = payload.get("machine") if isinstance(payload.get("machine"), dict) else {}
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
            decision = gpu.get("decision") if isinstance(gpu.get("decision"), dict) else {}
            processes = gpu.get("processes") if isinstance(gpu.get("processes"), list) else []
            lines.append(
                f"gpu={gpu.get('index')} "
                f"util={gpu.get('utilization')}% "
                f"temp={format_temperature(gpu.get('temperature_celsius'))} "
                f"hist={_decision_value(decision, 'history_average')} "
                f"mem={gpu.get('memory_used_human')}/{gpu.get('memory_total_human')} "
                f"action={decision.get('action')} "
                f"reason={decision.get('reason')} "
                f"worker={format_worker_label(gpu)} "
                f"processes={len(processes)}"
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


def format_explain_lines(
    payload: dict[str, object],
    events: list[dict[str, Any]],
    *,
    now: float | None = None,
) -> list[str]:
    explanation = explain_runtime(status=payload, status_error=None, events=events, now=now)
    return ["Current explanation", *format_explanation(explanation)]


def format_advice_lines(raw_config: object) -> list[str]:
    if not isinstance(raw_config, dict):
        return ["Tuning advice", "No config payload in status.json."]
    try:
        config = config_from_status_config(raw_config)
    except (TypeError, ValueError) as exc:
        return ["Tuning advice", f"Unable to parse config: {exc}"]
    advice = tuning_advice(config)
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
            if isinstance(check, dict):
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
    gap_text = f"{float(gap):.1f}" if isinstance(gap, (int, float)) else "-"
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
        return ["Selected GPU", "No GPU selected. Use --details <index>."]
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
        if item.get("reason") == reason_text:
            return [
                f"  reason meaning={item.get('meaning')}",
                f"  reason guidance={item.get('operator_guidance')}",
            ]
    return [
        f"  reason meaning=No reference entry for {reason_text}.",
        "  reason guidance=Run `gpu-holder reason-reference` to inspect known reasons.",
    ]


def format_temperature(value: object) -> str:
    if value is None:
        return "-"
    return f"{value}C"


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
