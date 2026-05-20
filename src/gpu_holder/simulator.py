from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .config import GuardConfig
from .models import DecisionAction, GpuHistory, GpuProcess, GpuSnapshot, HolderDecision, MachineHistory
from .policy import average_utilization, decide_for_fleet
from .units import parse_memory_spec


DEFAULT_TOTAL_MEMORY = 80 * 1024**3
DEFAULT_SCENARIO = "mixed"
ALL_SCENARIOS = "all"
SCENARIO_NAMES = ("mixed", "idle", "busy", "thermal", "quota-risk")


def simulate_policy(
    frames: list[list[GpuSnapshot]],
    config: GuardConfig,
    *,
    paused: bool = False,
) -> dict[str, object]:
    histories: dict[int, GpuHistory] = {}
    machine_history = MachineHistory(window_seconds=config.machine_window)
    running_gpu_indices: set[int] = set()
    timeline: list[dict[str, object]] = []
    action_counts: dict[str, int] = {}
    machine_averages: list[float] = []

    for step_index, snapshots in enumerate(frames):
        for snapshot in snapshots:
            history = histories.setdefault(
                snapshot.index,
                GpuHistory(window_seconds=config.idle_window),
            )
            history.add(snapshot.timestamp, snapshot.utilization)
        machine_average = average_utilization(snapshots)
        machine_timestamp = snapshots[0].timestamp if snapshots else float(step_index)
        machine_history.add(machine_timestamp, machine_average)
        machine_policy_average = machine_history.average()

        decisions = decide_for_fleet(
            snapshots,
            histories,
            config,
            running_gpu_indices=running_gpu_indices,
            paused=paused,
            machine_policy_average=machine_policy_average,
        )
        for decision in decisions:
            action_counts[decision.action.value] = action_counts.get(decision.action.value, 0) + 1
        running_gpu_indices = _next_running_set(running_gpu_indices, decisions)
        machine_averages.append(machine_average)
        timeline.append(
            {
                "step": step_index,
                "timestamp": snapshots[0].timestamp if snapshots else 0.0,
                "machine_average": machine_average,
                "machine_policy_average": machine_policy_average,
                "running_gpu_indices": sorted(running_gpu_indices),
                "decisions": [asdict(decision) for decision in decisions],
            }
        )

    return {
        "summary": {
            "steps": len(frames),
            "gpu_count": max((len(frame) for frame in frames), default=0),
            "average_machine_utilization": (
                sum(machine_averages) / len(machine_averages) if machine_averages else 0.0
            ),
            "min_machine_utilization": min(machine_averages) if machine_averages else 0.0,
            "max_machine_utilization": max(machine_averages) if machine_averages else 0.0,
            "action_counts": action_counts,
            "final_running_gpu_indices": sorted(running_gpu_indices),
        },
        "timeline": timeline,
    }


def load_trace(path: str | Path) -> list[list[GpuSnapshot]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        frames_payload = payload.get("frames")
    else:
        frames_payload = payload
    if not isinstance(frames_payload, list):
        raise ValueError("trace must be a list of frames or an object with frames")
    return [_parse_frame(frame, step_index=index) for index, frame in enumerate(frames_payload)]


def scenario_names() -> tuple[str, ...]:
    return SCENARIO_NAMES


def scenario_choices() -> tuple[str, ...]:
    return (ALL_SCENARIOS, *SCENARIO_NAMES)


def simulate_scenarios(
    config: GuardConfig,
    *,
    scenarios: tuple[str, ...] = SCENARIO_NAMES,
) -> dict[str, object]:
    results: list[dict[str, object]] = []
    action_counts: dict[str, int] = {}
    total_steps = 0
    total_gpu_decisions = 0
    weighted_utilization = 0.0

    for scenario in scenarios:
        payload = simulate_policy(built_in_trace(scenario), config)
        payload["scenario"] = scenario
        payload["trace"] = None
        results.append(payload)

        summary = payload["summary"]
        if not isinstance(summary, dict):
            continue
        steps = int(summary.get("steps", 0))
        gpu_count = int(summary.get("gpu_count", 0))
        total_steps += steps
        total_gpu_decisions += steps * gpu_count
        weighted_utilization += float(summary.get("average_machine_utilization", 0.0)) * steps

        summary_actions = summary.get("action_counts", {})
        if not isinstance(summary_actions, dict):
            continue
        for action, count in summary_actions.items():
            action_counts[str(action)] = action_counts.get(str(action), 0) + int(count)

    return {
        "summary": {
            "scenario_count": len(results),
            "scenarios": [str(scenario) for scenario in scenarios],
            "steps": total_steps,
            "gpu_decisions": total_gpu_decisions,
            "average_machine_utilization": (
                weighted_utilization / total_steps if total_steps else 0.0
            ),
            "action_counts": action_counts,
        },
        "scenarios": results,
    }


def built_in_trace(scenario: str = DEFAULT_SCENARIO) -> list[list[GpuSnapshot]]:
    if scenario == "mixed":
        return _mixed_trace()
    if scenario == "idle":
        return _idle_trace()
    if scenario == "busy":
        return _busy_trace()
    if scenario == "thermal":
        return _thermal_trace()
    if scenario == "quota-risk":
        return _quota_risk_trace()
    available = ", ".join(SCENARIO_NAMES)
    raise ValueError(f"unknown built-in scenario: {scenario}; available: {available}")


def _mixed_trace() -> list[list[GpuSnapshot]]:
    total = DEFAULT_TOTAL_MEMORY
    busy = GpuProcess(pid=1234, used_memory_bytes=20 * 1024**3, name="train.py")
    frames: list[list[GpuSnapshot]] = []
    for timestamp, util0, util1, processes1 in (
        (0.0, 15, 15, (busy,)),
        (30.0, 12, 15, (busy,)),
        (60.0, 10, 15, (busy,)),
        (90.0, 70, 85, (busy,)),
    ):
        frames.append(
            [
                GpuSnapshot(
                    index=0,
                    name="sim-gpu-0",
                    memory_total_bytes=total,
                    memory_used_bytes=2 * 1024**3,
                    utilization=util0,
                    timestamp=timestamp,
                ),
                GpuSnapshot(
                    index=1,
                    name="sim-gpu-1",
                    memory_total_bytes=total,
                    memory_used_bytes=24 * 1024**3 if processes1 else 2 * 1024**3,
                    utilization=util1,
                    processes=processes1,
                    timestamp=timestamp,
                ),
            ]
        )
    return frames


def _idle_trace() -> list[list[GpuSnapshot]]:
    return [
        _frame(timestamp=timestamp, utilizations=(8, 12))
        for timestamp in (0.0, 30.0, 60.0, 90.0)
    ]


def _busy_trace() -> list[list[GpuSnapshot]]:
    busy0 = GpuProcess(pid=2000, used_memory_bytes=24 * 1024**3, name="train.py")
    busy1 = GpuProcess(pid=2001, used_memory_bytes=18 * 1024**3, name="serve.py")
    return [
        _frame(
            timestamp=timestamp,
            utilizations=(75, 82),
            processes=(busy0, busy1),
            used_memory=(28 * 1024**3, 22 * 1024**3),
        )
        for timestamp in (0.0, 30.0, 60.0, 90.0)
    ]


def _thermal_trace() -> list[list[GpuSnapshot]]:
    return [
        _frame(
            timestamp=timestamp,
            utilizations=(20, 25),
            temperatures=(88, 70),
        )
        for timestamp in (0.0, 30.0, 60.0, 90.0)
    ]


def _quota_risk_trace() -> list[list[GpuSnapshot]]:
    return [
        _frame(timestamp=timestamp, utilizations=(10, 15))
        for timestamp in (0.0, 300.0, 600.0, 900.0, 1200.0, 1500.0, 1800.0)
    ]


def _frame(
    *,
    timestamp: float,
    utilizations: tuple[int, int],
    processes: tuple[GpuProcess, GpuProcess] | None = None,
    used_memory: tuple[int, int] | None = None,
    temperatures: tuple[int | None, int | None] = (None, None),
) -> list[GpuSnapshot]:
    total = DEFAULT_TOTAL_MEMORY
    process_tuple = processes or (
        GpuProcess(pid=0, used_memory_bytes=0, name="", is_holder=False),
        GpuProcess(pid=0, used_memory_bytes=0, name="", is_holder=False),
    )
    memory_tuple = used_memory or (2 * 1024**3, 2 * 1024**3)
    return [
        GpuSnapshot(
            index=0,
            name="sim-gpu-0",
            memory_total_bytes=total,
            memory_used_bytes=memory_tuple[0],
            utilization=utilizations[0],
            temperature_celsius=temperatures[0],
            processes=() if process_tuple[0].pid == 0 else (process_tuple[0],),
            timestamp=timestamp,
        ),
        GpuSnapshot(
            index=1,
            name="sim-gpu-1",
            memory_total_bytes=total,
            memory_used_bytes=memory_tuple[1],
            utilization=utilizations[1],
            temperature_celsius=temperatures[1],
            processes=() if process_tuple[1].pid == 0 else (process_tuple[1],),
            timestamp=timestamp,
        ),
    ]


def _next_running_set(
    running_gpu_indices: set[int],
    decisions: list[HolderDecision],
) -> set[int]:
    next_running = set(running_gpu_indices)
    for decision in decisions:
        if decision.action in {DecisionAction.HOLD, DecisionAction.ASSIST}:
            next_running.add(decision.gpu_index)
        elif decision.action in {DecisionAction.RELEASE, DecisionAction.STANDBY}:
            next_running.discard(decision.gpu_index)
    return next_running


def _parse_frame(frame: object, *, step_index: int) -> list[GpuSnapshot]:
    context = f"frames[{step_index}]"
    if not isinstance(frame, dict):
        raise ValueError(f"{context} must be an object")
    timestamp = _float_value(frame.get("timestamp", step_index), context=f"{context}.timestamp")
    gpus = frame.get("gpus")
    if not isinstance(gpus, list):
        raise ValueError(f"{context}.gpus must be a list")
    return [
        _parse_gpu(gpu, timestamp=timestamp, context=f"{context}.gpus[{gpu_index}]")
        for gpu_index, gpu in enumerate(gpus)
    ]


def _parse_gpu(gpu: object, *, timestamp: float, context: str) -> GpuSnapshot:
    if not isinstance(gpu, dict):
        raise ValueError(f"{context} must be an object")
    gpu_index = _int_value(_required(gpu, "index", context), context=f"{context}.index")
    total = _memory_value(
        gpu.get("memory_total", DEFAULT_TOTAL_MEMORY),
        DEFAULT_TOTAL_MEMORY,
        context=f"{context}.memory_total",
    )
    used = _memory_value(gpu.get("memory_used", 0), total, context=f"{context}.memory_used")
    raw_processes = gpu.get("processes", [])
    if not isinstance(raw_processes, list):
        raise ValueError(f"{context}.processes must be a list")
    processes = tuple(
        _parse_process(item, total_memory=total, context=f"{context}.processes[{process_index}]")
        for process_index, item in enumerate(raw_processes)
    )
    return GpuSnapshot(
        index=gpu_index,
        name=str(gpu.get("name", f"gpu-{gpu_index}")),
        memory_total_bytes=total,
        memory_used_bytes=used,
        utilization=_int_value(
            _required(gpu, "utilization", context),
            context=f"{context}.utilization",
        ),
        temperature_celsius=_temperature_value(gpu, context=context),
        processes=processes,
        timestamp=timestamp,
    )


def _parse_process(process: object, *, total_memory: int, context: str) -> GpuProcess:
    if not isinstance(process, dict):
        raise ValueError(f"{context} must be an object")
    return GpuProcess(
        pid=_int_value(_required(process, "pid", context), context=f"{context}.pid"),
        used_memory_bytes=_memory_value(
            process.get("used_memory", 0),
            total_memory,
            context=f"{context}.used_memory",
        ),
        name=str(process.get("name", "")),
        is_holder=bool(process.get("is_holder", False)),
    )


def _memory_value(raw: Any, total_memory: int, *, context: str) -> int:
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    try:
        return parse_memory_spec(str(raw)).resolve(total_memory)
    except ValueError as exc:
        raise ValueError(f"{context} is invalid: {exc}") from exc


def _temperature_value(gpu: dict[str, object], *, context: str) -> int | None:
    raw = gpu.get("temperature_celsius", gpu.get("temperature"))
    if raw is None:
        return None
    return _int_value(raw, context=f"{context}.temperature")


def _required(payload: dict[str, object], key: str, context: str) -> object:
    if key not in payload:
        raise ValueError(f"{context}.{key} is required")
    return payload[key]


def _int_value(raw: object, *, context: str) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be an integer") from exc


def _float_value(raw: object, *, context: str) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be a number") from exc
