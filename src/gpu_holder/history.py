from __future__ import annotations

from collections import Counter
from typing import Any


def summarize_events(
    events: list[dict[str, Any]],
    *,
    since_seconds: float | None = None,
    now: float | None = None,
    gpu_indices: set[int] | None = None,
) -> dict[str, Any]:
    selected = select_events(events, since_seconds=since_seconds, now=now, gpu_indices=gpu_indices)
    timestamps = [_float_value(event.get("timestamp")) for event in selected]
    timestamps = [timestamp for timestamp in timestamps if timestamp is not None]
    first_timestamp = min(timestamps) if timestamps else None
    last_timestamp = max(timestamps) if timestamps else None
    per_gpu: dict[int, dict[str, Any]] = {}
    event_type_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()

    for event in selected:
        event_type = str(event.get("type", "unknown"))
        event_type_counts[event_type] += 1
        gpu_index = _gpu_index(event)
        if gpu_index is not None:
            gpu_summary = per_gpu.setdefault(gpu_index, _empty_gpu_summary(gpu_index))
            gpu_summary["event_count"] += 1
        else:
            gpu_summary = None

        if event_type == "decision":
            action = str(event.get("action", "unknown"))
            reason = str(event.get("reason", "unknown"))
            action_counts[action] += 1
            reason_counts[reason] += 1
            if gpu_summary is not None:
                gpu_summary["decision_count"] += 1
                gpu_summary["action_counts"][action] += 1
                gpu_summary["reason_counts"][reason] += 1
                gpu_summary["last_action"] = action
                gpu_summary["last_reason"] = reason
                gpu_summary["last_decision_timestamp"] = event.get("timestamp")
            continue

        if gpu_summary is not None:
            _count_gpu_event(gpu_summary, event_type)

    return {
        "ok": bool(selected),
        "event_count": len(selected),
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "duration_seconds": (
            None
            if first_timestamp is None or last_timestamp is None
            else max(0.0, last_timestamp - first_timestamp)
        ),
        "event_type_counts": dict(sorted(event_type_counts.items())),
        "decision_count": sum(action_counts.values()),
        "action_counts": dict(sorted(action_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "gpu_indices": sorted(per_gpu),
        "gpus": {
            str(gpu_index): _finalize_gpu_summary(summary)
            for gpu_index, summary in sorted(per_gpu.items())
        },
    }


def select_events(
    events: list[dict[str, Any]],
    *,
    since_seconds: float | None = None,
    now: float | None = None,
    gpu_indices: set[int] | None = None,
) -> list[dict[str, Any]]:
    if since_seconds is None and gpu_indices is None:
        return list(events)
    latest = _latest_timestamp(events) if now is None else float(now)
    threshold = None if since_seconds is None else latest - float(since_seconds)
    selected: list[dict[str, Any]] = []
    for event in events:
        if threshold is not None:
            timestamp = _float_value(event.get("timestamp"))
            if timestamp is None or timestamp < threshold:
                continue
        if gpu_indices is not None:
            gpu_index = _gpu_index(event)
            if gpu_index not in gpu_indices:
                continue
        selected.append(event)
    return selected


def _empty_gpu_summary(gpu_index: int) -> dict[str, Any]:
    return {
        "gpu_index": gpu_index,
        "event_count": 0,
        "decision_count": 0,
        "action_counts": Counter(),
        "reason_counts": Counter(),
        "last_action": None,
        "last_reason": None,
        "last_decision_timestamp": None,
        "worker_starts": 0,
        "worker_stops": 0,
        "worker_start_failures": 0,
        "worker_start_backoffs": 0,
        "worker_exits": 0,
        "thermal_limit_starts": 0,
        "thermal_limit_ends": 0,
        "process_grace_starts": 0,
    }


def _count_gpu_event(summary: dict[str, Any], event_type: str) -> None:
    key_by_type = {
        "worker_start": "worker_starts",
        "worker_stop": "worker_stops",
        "worker_start_failed": "worker_start_failures",
        "worker_start_backoff": "worker_start_backoffs",
        "worker_exit": "worker_exits",
        "thermal_limit_start": "thermal_limit_starts",
        "thermal_limit_end": "thermal_limit_ends",
        "process_grace_start": "process_grace_starts",
    }
    key = key_by_type.get(event_type)
    if key is not None:
        summary[key] += 1


def _finalize_gpu_summary(summary: dict[str, Any]) -> dict[str, Any]:
    finalized = dict(summary)
    finalized["action_counts"] = dict(sorted(summary["action_counts"].items()))
    finalized["reason_counts"] = dict(sorted(summary["reason_counts"].items()))
    return finalized


def _latest_timestamp(events: list[dict[str, Any]]) -> float:
    timestamps = [_float_value(event.get("timestamp")) for event in events]
    timestamps = [timestamp for timestamp in timestamps if timestamp is not None]
    return max(timestamps, default=0.0)


def _float_value(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _gpu_index(event: dict[str, Any]) -> int | None:
    try:
        return int(event["gpu_index"])
    except (KeyError, TypeError, ValueError):
        return None
