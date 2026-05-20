from __future__ import annotations

from collections import Counter
from typing import Any

from .history import summarize_events
from .reasons import reason_reference
from .status_health import (
    enrich_status_health,
    format_policy_window_health,
    format_status_age,
    format_utilization_target_forecast,
)


def explain_runtime(
    *,
    status: dict[str, object] | None,
    status_error: str | None,
    events: list[dict[str, Any]],
    stale_after: float | None = 15.0,
    now: float | None = None,
) -> dict[str, object]:
    findings: list[dict[str, object]] = []
    if status is None:
        findings.append(
            _finding(
                "critical",
                "status_missing",
                status_error or "No status file is available.",
                "Start `gpu-holder guard` or `gpu-holder start`, then run `gpu-holder status`.",
            )
        )
        return _payload(findings=findings, status=None, history=summarize_events(events))

    enriched = enrich_status_health(status, stale_after=stale_after, now=now)
    _add_status_findings(findings, enriched)
    _add_gpu_findings(findings, enriched)
    history = summarize_events(events)
    _add_history_findings(findings, history)
    return _payload(findings=findings, status=enriched, history=history)


def format_explanation(payload: dict[str, object]) -> list[str]:
    status = payload.get("status")
    if not isinstance(status, dict):
        status = {}
    target_health = status.get("utilization_target_health")
    if not isinstance(target_health, dict):
        target_health = {}
    window_health = status.get("policy_window_health")
    target_forecast = status.get("utilization_target_forecast")
    history = payload.get("history")
    if not isinstance(history, dict):
        history = {}
    lines = [
        f"gpu-holder explain ok={payload.get('ok')}",
        (
            f"status_age={format_status_age(status.get('status_age_seconds'))} "
            f"stale={status.get('status_stale')} paused={status.get('paused')} "
            f"{format_policy_window_health(window_health)} "
            f"{format_utilization_target_forecast(target_forecast)} "
            f"target={target_health.get('status')} gap={target_health.get('gap')} "
            f"history_events={history.get('event_count', 0)}"
        ),
    ]
    findings = payload.get("findings")
    if not isinstance(findings, list) or not findings:
        lines.append("finding[info]: healthy - No immediate operator action is suggested.")
        return lines
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        lines.append(
            f"finding[{finding.get('severity')}]: "
            f"{finding.get('code')} - {finding.get('message')}"
        )
        guidance = finding.get("guidance")
        if guidance:
            lines.append(f"  guidance: {guidance}")
    return lines


def _add_status_findings(findings: list[dict[str, object]], status: dict[str, object]) -> None:
    if status.get("status_stale"):
        findings.append(
            _finding(
                "warning",
                "status_stale",
                f"Status is stale; last update was {format_status_age(status.get('status_age_seconds'))} ago.",
                "Check whether the guard process is still running with `gpu-holder status` or restart it.",
            )
        )
    if status.get("paused"):
        pause_until = status.get("pause_until")
        suffix = f" until {pause_until}" if pause_until is not None else ""
        findings.append(
            _finding(
                "warning",
                "paused",
                f"The guard is paused{suffix}.",
                _guidance_for_reason("paused"),
            )
        )
    target_health = status.get("utilization_target_health")
    window_health = status.get("policy_window_health")
    target_forecast = status.get("utilization_target_forecast")
    if isinstance(target_forecast, dict) and target_forecast.get("status") == "at_risk":
        findings.append(
            _finding(
                "warning",
                "quota_forecast_at_risk",
                (
                    "The remaining policy window would need more than 100% average "
                    "utilization to reach the target."
                ),
                "Reduce blockers, raise allowed holder coverage, or treat this window as unrecoverable before the reclaim check.",
            )
        )
    if isinstance(window_health, dict) and window_health.get("status") == "warming_up":
        findings.append(
            _finding(
                "info",
                "policy_window_warming_up",
                (
                    "Policy/window average is still warming up "
                    f"({format_policy_window_health(window_health)})."
                ),
                "Let the guard run long enough to fill the machine window before treating the policy average as a full quota-window signal.",
            )
        )
    if isinstance(target_health, dict) and target_health.get("status") == "below_target":
        guidance = "Inspect per-GPU findings, recent history, and whether holders are paused, capped, thermal-blocked, or yielding to workloads."
        if isinstance(window_health, dict) and window_health.get("status") == "warming_up":
            guidance = (
                "The policy window is still warming up; inspect current blockers, "
                "but avoid treating this as a full-window quota miss yet."
            )
        findings.append(
            _finding(
                "warning",
                "below_target",
                (
                    "Policy/window average utilization is below target "
                    f"by {abs(float(target_health.get('gap', 0.0))):.1f}%."
                ),
                guidance,
            )
        )
    disabled_gpus = status.get("disabled_gpus")
    if isinstance(disabled_gpus, list) and disabled_gpus:
        findings.append(
            _finding(
                "info",
                "gpu_disabled",
                f"Runtime-disabled GPUs: {disabled_gpus}.",
                _guidance_for_reason("gpu_disabled"),
            )
        )


def _add_gpu_findings(findings: list[dict[str, object]], status: dict[str, object]) -> None:
    gpus = status.get("gpus")
    if not isinstance(gpus, list):
        return
    reason_counts: Counter[str] = Counter()
    worker_backoff_gpus: list[int] = []
    external_process_count = 0
    for gpu in gpus:
        if not isinstance(gpu, dict):
            continue
        gpu_index = _gpu_index(gpu)
        decision = gpu.get("decision")
        if isinstance(decision, dict):
            reason = decision.get("reason")
            if isinstance(reason, str):
                reason_counts[reason] += 1
        backoff = gpu.get("worker_restart_backoff")
        if isinstance(backoff, dict) and backoff and gpu_index is not None:
            worker_backoff_gpus.append(gpu_index)
        processes = gpu.get("processes")
        if isinstance(processes, list):
            external_process_count += sum(
                1
                for process in processes
                if isinstance(process, dict) and not bool(process.get("is_holder"))
            )
    for reason in (
        "thermal_limit",
        "protected_process",
        "busy_process",
        "insufficient_free_memory",
        "process_grace",
    ):
        count = reason_counts.get(reason, 0)
        if count:
            findings.append(
                _finding(
                    _severity_for_reason(reason),
                    reason,
                    f"{count} GPU(s) currently have decision reason `{reason}`.",
                    _guidance_for_reason(reason),
                )
            )
    if worker_backoff_gpus:
        findings.append(
            _finding(
                "warning",
                "worker_start_backoff",
                f"Worker start backoff is active on GPU(s): {worker_backoff_gpus}.",
                "Run `gpu-holder doctor`; inspect recent `worker_start_failed` events and CUDA/PyTorch availability.",
            )
        )
    if external_process_count:
        findings.append(
            _finding(
                "info",
                "external_processes_present",
                f"{external_process_count} non-holder GPU process(es) are visible.",
                "This is expected on shared machines; gpu-holder treats them as read-only scheduling signals.",
            )
        )


def _add_history_findings(findings: list[dict[str, object]], history: dict[str, object]) -> None:
    if not history.get("ok"):
        findings.append(
            _finding(
                "info",
                "history_empty",
                "No recent events were found.",
                "Let the controller run for a few intervals, or inspect the configured state directory.",
            )
        )
        return
    event_counts = history.get("event_type_counts")
    if not isinstance(event_counts, dict):
        return
    failed = int(event_counts.get("worker_start_failed", 0) or 0)
    backoff = int(event_counts.get("worker_start_backoff", 0) or 0)
    if failed or backoff:
        findings.append(
            _finding(
                "warning",
                "worker_start_failures",
                f"Recent history includes worker_start_failed={failed}, worker_start_backoff={backoff}.",
                "Run `gpu-holder doctor` and check whether PyTorch can allocate CUDA tensors.",
            )
        )


def _payload(
    *,
    findings: list[dict[str, object]],
    status: dict[str, object] | None,
    history: dict[str, object],
) -> dict[str, object]:
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda item: severity_order.get(str(item.get("severity")), 99))
    return {
        "ok": not any(item.get("severity") == "critical" for item in findings),
        "status": status,
        "history": history,
        "findings": findings,
        "finding_count": len(findings),
    }


def _finding(severity: str, code: str, message: str, guidance: str) -> dict[str, object]:
    return {
        "severity": severity,
        "code": code,
        "message": message,
        "guidance": guidance,
    }


def _guidance_for_reason(reason: str) -> str:
    for item in reason_reference():
        if item.get("reason") == reason:
            return str(item.get("operator_guidance"))
    return "Inspect `gpu-holder reason-reference` and recent events for more context."


def _severity_for_reason(reason: str) -> str:
    if reason in {"thermal_limit", "insufficient_free_memory"}:
        return "warning"
    return "info"


def _gpu_index(gpu: dict[str, object]) -> int | None:
    try:
        return int(gpu["index"])
    except (KeyError, TypeError, ValueError):
        return None
