from __future__ import annotations

from typing import Any

from .status_health import enrich_status_health


def prometheus_metrics(
    payload: dict[str, object] | None,
    *,
    status_error: str | None = None,
    stale_after: float | None = 15.0,
    now: float | None = None,
) -> str:
    lines: list[str] = []
    _add_metric(
        lines,
        "gpu_holder_status_available",
        "Whether a valid gpu-holder status payload was available.",
        "gauge",
        1 if payload is not None else 0,
    )
    if payload is None:
        _add_metric(
            lines,
            "gpu_holder_status_error",
            "Status read error by error label.",
            "gauge",
            1,
            {"error": status_error or "missing"},
        )
        return "\n".join(lines) + "\n"

    status = enrich_status_health(payload, stale_after=stale_after, now=now)
    machine = _dict_value(status.get("machine"))
    target_health = _dict_value(status.get("utilization_target_health"))
    target_forecast = _dict_value(status.get("utilization_target_forecast"))
    window_health = _dict_value(status.get("policy_window_health"))
    config = _dict_value(status.get("config"))

    _add_metric(lines, "gpu_holder_status_timestamp_seconds", "Status timestamp.", "gauge", status.get("timestamp"))
    _add_metric(
        lines,
        "gpu_holder_status_age_seconds",
        "Age of the status payload.",
        "gauge",
        status.get("status_age_seconds"),
    )
    _add_metric(lines, "gpu_holder_status_stale", "Whether the status payload is stale.", "gauge", bool(status.get("status_stale")))
    _add_metric(lines, "gpu_holder_paused", "Whether gpu-holder is paused.", "gauge", bool(status.get("paused")))
    _add_metric(lines, "gpu_holder_target_utilization_percent", "Configured target utilization.", "gauge", target_health.get("target_util"))
    _add_metric(
        lines,
        "gpu_holder_target_gap_percent",
        "Policy average minus target utilization.",
        "gauge",
        target_health.get("gap"),
    )
    _add_metric(
        lines,
        "gpu_holder_target_required_remaining_average_percent",
        "Average utilization required over the remaining policy window to reach target.",
        "gauge",
        target_forecast.get("required_remaining_average"),
    )
    _add_metric(
        lines,
        "gpu_holder_target_remaining_window_seconds",
        "Remaining seconds in the policy window.",
        "gauge",
        target_forecast.get("remaining_seconds"),
    )
    _add_metric(
        lines,
        "gpu_holder_target_forecast_ok",
        "Whether the policy window target is still reachable under the forecast.",
        "gauge",
        bool(target_forecast.get("ok")),
    )
    _add_metric(
        lines,
        "gpu_holder_target_forecast_at_risk",
        "Whether the target forecast requires more than 100 percent utilization over the remaining window.",
        "gauge",
        target_forecast.get("status") == "at_risk",
    )
    _add_metric(lines, "gpu_holder_gpu_count", "Number of GPUs in the status payload.", "gauge", machine.get("gpu_count"))
    _add_metric(lines, "gpu_holder_machine_average_utilization_percent", "Current machine-average GPU utilization.", "gauge", machine.get("average_utilization"))
    _add_metric(lines, "gpu_holder_machine_policy_average_utilization_percent", "Policy/window machine-average GPU utilization.", "gauge", machine.get("policy_average_utilization"))
    _add_metric(lines, "gpu_holder_machine_policy_window_seconds", "Configured policy machine window size.", "gauge", window_health.get("window_seconds"))
    _add_metric(lines, "gpu_holder_machine_policy_window_sample_count", "Machine policy window sample count.", "gauge", window_health.get("sample_count"))
    _add_metric(lines, "gpu_holder_machine_policy_window_coverage_seconds", "Observed coverage of the policy machine window.", "gauge", window_health.get("coverage_seconds"))
    _add_metric(lines, "gpu_holder_machine_policy_window_coverage_ratio", "Observed coverage ratio of the policy machine window.", "gauge", window_health.get("coverage_ratio"))
    _add_metric(lines, "gpu_holder_machine_policy_window_ready", "Whether the policy machine window has enough coverage.", "gauge", bool(window_health.get("ok")))
    _add_metric(lines, "gpu_holder_owned_worker_count", "Owned holder worker count.", "gauge", machine.get("owned_worker_count"))
    _add_metric(lines, "gpu_holder_worker_backoff_count", "Active worker startup backoff count.", "gauge", machine.get("worker_backoff_count"))
    _add_metric(lines, "gpu_holder_process_grace_count", "Active process grace count.", "gauge", machine.get("process_grace_count"))
    _add_metric(lines, "gpu_holder_thermal_limit_count", "Thermal-limited GPU count.", "gauge", machine.get("thermal_limit_count"))
    _add_action_metrics(lines, machine.get("action_counts"))

    profile = str(config.get("profile", ""))
    gpus = status.get("gpus")
    reference_time = _reference_time(status, now=now)
    if isinstance(gpus, list):
        for gpu in gpus:
            if isinstance(gpu, dict):
                _add_gpu_metrics(lines, gpu, profile=profile, reference_time=reference_time)
    return "\n".join(lines) + "\n"


def _add_action_metrics(lines: list[str], raw_counts: object) -> None:
    if not isinstance(raw_counts, dict):
        return
    for action, count in sorted(raw_counts.items()):
        _add_metric(
            lines,
            "gpu_holder_machine_action_count",
            "Machine decision action counts in the latest status payload.",
            "gauge",
            count,
            {"action": str(action)},
        )


def _add_gpu_metrics(
    lines: list[str],
    gpu: dict[str, Any],
    *,
    profile: str,
    reference_time: float | None,
) -> None:
    gpu_index = str(gpu.get("index"))
    labels = {"gpu": gpu_index, "name": str(gpu.get("name", "")), "profile": profile}
    _add_metric(lines, "gpu_holder_gpu_utilization_percent", "Per-GPU utilization.", "gauge", gpu.get("utilization"), labels)
    _add_metric(lines, "gpu_holder_gpu_temperature_celsius", "Per-GPU temperature.", "gauge", gpu.get("temperature_celsius"), labels)
    _add_metric(lines, "gpu_holder_gpu_memory_used_bytes", "Per-GPU used memory.", "gauge", gpu.get("memory_used"), labels)
    _add_metric(lines, "gpu_holder_gpu_memory_total_bytes", "Per-GPU total memory.", "gauge", gpu.get("memory_total"), labels)
    _add_metric(lines, "gpu_holder_gpu_disabled", "Whether this GPU is runtime disabled.", "gauge", bool(gpu.get("gpu_disabled")), labels)
    disabled_until = _number(gpu.get("gpu_disabled_until"))
    if disabled_until is not None:
        _add_metric(
            lines,
            "gpu_holder_gpu_disabled_until_seconds",
            "Unix timestamp when the temporary runtime disable expires.",
            "gauge",
            disabled_until,
            labels,
        )
        if reference_time is not None:
            _add_metric(
                lines,
                "gpu_holder_gpu_disabled_remaining_seconds",
                "Seconds remaining before the temporary runtime disable expires.",
                "gauge",
                max(0.0, disabled_until - reference_time),
                labels,
            )
    _add_metric(lines, "gpu_holder_gpu_thermal_blocked", "Whether this GPU is thermally blocked.", "gauge", bool(gpu.get("thermal_blocked")), labels)
    processes = gpu.get("processes")
    process_count = len(processes) if isinstance(processes, list) else 0
    external_process_count = 0
    protected_process_count = 0
    if isinstance(processes, list):
        external_process_count = sum(
            1 for process in processes if isinstance(process, dict) and not bool(process.get("is_holder"))
        )
        protected_process_count = sum(
            1 for process in processes if isinstance(process, dict) and bool(process.get("protected_process"))
        )
    _add_metric(lines, "gpu_holder_gpu_process_count", "Visible GPU process count.", "gauge", process_count, labels)
    _add_metric(lines, "gpu_holder_gpu_external_process_count", "Visible non-holder GPU process count.", "gauge", external_process_count, labels)
    _add_metric(
        lines,
        "gpu_holder_gpu_protected_process_count",
        "Visible protected non-holder GPU process count.",
        "gauge",
        protected_process_count,
        labels,
    )

    worker = gpu.get("worker")
    worker_alive = isinstance(worker, dict) and bool(worker.get("alive", worker))
    _add_metric(lines, "gpu_holder_gpu_worker_alive", "Whether an owned holder worker is alive for this GPU.", "gauge", worker_alive, labels)
    decision = gpu.get("decision")
    if isinstance(decision, dict):
        decision_labels = {
            **labels,
            "action": str(decision.get("action", "")),
            "reason": str(decision.get("reason", "")),
            "mode": str(decision.get("hold_mode", "")),
            "program": str(decision.get("program", "")),
        }
        _add_metric(lines, "gpu_holder_gpu_decision", "Latest per-GPU decision labels.", "gauge", 1, decision_labels)
        _add_metric(lines, "gpu_holder_gpu_decision_duty_cycle", "Latest requested decision duty cycle.", "gauge", decision.get("duty_cycle"), labels)
        _add_metric(lines, "gpu_holder_gpu_decision_memory_bytes", "Latest requested decision memory.", "gauge", decision.get("memory_bytes"), labels)


def _add_metric(
    lines: list[str],
    name: str,
    help_text: str,
    metric_type: str,
    value: object,
    labels: dict[str, str] | None = None,
) -> None:
    number = _number(value)
    if number is None:
        return
    if f"# HELP {name} " not in "\n".join(lines):
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {metric_type}")
    lines.append(f"{name}{_format_labels(labels)} {_format_number(number)}")


def _format_labels(labels: dict[str, str] | None) -> str:
    if not labels:
        return ""
    rendered = ",".join(f'{key}="{_escape_label(str(value))}"' for key, value in sorted(labels.items()))
    return "{" + rendered + "}"


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.6g}"


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _reference_time(status: dict[str, object], *, now: float | None) -> float | None:
    if now is not None:
        return float(now)
    timestamp = _number(status.get("timestamp"))
    age = _number(status.get("status_age_seconds"))
    if timestamp is None or age is None:
        return None
    return timestamp + age


def _dict_value(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}
