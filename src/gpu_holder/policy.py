from __future__ import annotations

import time
from typing import Protocol

from .models import Decision, GpuProcess, GpuSnapshot


class PolicyConfig(Protocol):
    target_util: float
    risk_util: float
    mem: float
    reserve: str
    busy_process_mem_threshold: str
    assist_mem: str
    min_duty_cycle: float
    max_duty_cycle: float
    process_grace_window: float


def decide(
    snapshots: list[GpuSnapshot],
    config: PolicyConfig,
    *,
    external_process_first_seen: dict[int, dict[tuple[int, ...], float]] | None = None,
    now: float | None = None,
) -> list[Decision]:
    risk = config.risk_util * 100
    decisions: list[Decision] = []
    busy_threshold = parse_bytes(config.busy_process_mem_threshold)
    external_process_first_seen = external_process_first_seen or {}
    now = time.monotonic() if now is None else float(now)
    for gpu in snapshots:
        holder_running = any(process.is_holder for process in gpu.processes)
        external_processes = [process for process in gpu.processes if not process.is_holder]
        external_busy = any(process.used_memory >= busy_threshold for process in external_processes)
        external_stale = external_processes and all(is_stale_process(process) for process in external_processes)
        if external_processes:
            external_signature = process_signature(external_processes)
            first_seen = external_process_first_seen.get(gpu.index, {}).get(external_signature, now)
            in_grace = now - first_seen < max(0.0, config.process_grace_window)
            if external_busy and in_grace:
                decisions.append(make_release(gpu, "busy_process_grace"))
                continue
            if external_busy and gpu.utilization < risk:
                decisions.append(make_hold(gpu, config, reason="busy_process_idle", assist=True))
                continue
            if external_stale and not holder_running and gpu.utilization < risk:
                decisions.append(make_hold(gpu, config, reason="external_process_idle", assist=True))
                continue
            reason = "busy_process" if external_busy else "external_process"
            decisions.append(make_release(gpu, reason))
            continue
        if gpu.utilization < risk:
            decisions.append(make_hold(gpu, config, reason="below_risk", assist=False))
        elif holder_running:
            decisions.append(make_hold(gpu, config, reason="holder_running", assist=False))
        else:
            decisions.append(make_release(gpu, "risk_clear"))
    return decisions


def make_hold(gpu: GpuSnapshot, config: PolicyConfig, *, reason: str, assist: bool) -> Decision:
    spec = config.assist_mem if assist else config.mem
    reserve = parse_bytes(config.reserve)
    memory = (
        resolve_memory_spec(str(spec), free=gpu.memory_free, reserve=reserve)
        if assist
        else resolve_memory_ratio(config.mem, total=gpu.memory_total, free=gpu.memory_free, reserve=reserve)
    )
    duty = max(config.min_duty_cycle, min(config.max_duty_cycle, config.target_util))
    return Decision(
        gpu_index=gpu.index,
        action="hold",
        reason=reason,
        memory_bytes=memory,
        duty_cycle=duty,
        hold_mode="assist" if assist else "balanced",
    )


def make_release(gpu: GpuSnapshot, reason: str) -> Decision:
    return Decision(
        gpu_index=gpu.index,
        action="release",
        reason=reason,
        memory_bytes=0,
        duty_cycle=0,
        hold_mode="-",
    )


def is_stale_process(process: GpuProcess) -> bool:
    return process.name.strip() == "[Not Found]"


def process_signature(processes: list[GpuProcess]) -> tuple[int, ...]:
    return tuple(sorted(process.pid for process in processes))


def resolve_memory_ratio(ratio: float, *, total: int, free: int, reserve: int) -> int:
    requested = int(total * parse_ratio(ratio))
    return max(0, min(requested, free - reserve))


def resolve_memory_spec(spec: str, *, free: int, reserve: int) -> int:
    requested = parse_bytes(spec)
    return max(0, min(requested, free - reserve))


def parse_ratio(raw: object) -> float:
    if isinstance(raw, (float, int)):
        value = float(raw)
    else:
        text = str(raw).strip()
        if text.endswith("%"):
            value = float(text[:-1]) / 100
        else:
            value = float(text)
    if 1 < value <= 100:
        value = value / 100
    if not 0 <= value <= 1:
        raise ValueError("ratio must be a float from 0 to 1")
    return value


def parse_bytes(raw: str) -> int:
    text = str(raw).strip()
    units = {
        "b": 1,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
    }
    lowered = text.lower()
    for unit, scale in sorted(units.items(), key=lambda item: -len(item[0])):
        if lowered.endswith(unit):
            return int(float(lowered[: -len(unit)]) * scale)
    if lowered.isdigit():
        return int(lowered)
    raise ValueError(f"invalid byte size: {raw}")
