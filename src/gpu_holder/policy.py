from __future__ import annotations

from .config import GuardConfig
from .models import DecisionAction, GpuHistory, GpuSnapshot, HolderDecision
from .process_matching import has_protected_process


def decide_for_gpu(
    snapshot: GpuSnapshot,
    history: GpuHistory,
    config: GuardConfig,
    *,
    holder_running: bool,
    paused: bool = False,
    disabled: bool = False,
    thermal_blocked: bool = False,
    process_grace: bool = False,
    machine_needs_help: bool = True,
    machine_average: float = 0.0,
    machine_policy_average: float | None = None,
) -> HolderDecision:
    history_average = history.average()
    effective_machine_average = (
        machine_average if machine_policy_average is None else machine_policy_average
    )
    if paused:
        return HolderDecision(
            gpu_index=snapshot.index,
            action=DecisionAction.RELEASE if holder_running else DecisionAction.STANDBY,
            reason="paused",
            machine_average=machine_average,
            machine_policy_average=effective_machine_average,
            history_average=history_average,
        )

    if disabled:
        return HolderDecision(
            gpu_index=snapshot.index,
            action=DecisionAction.RELEASE if holder_running else DecisionAction.STANDBY,
            reason="gpu_disabled",
            machine_average=machine_average,
            machine_policy_average=effective_machine_average,
            history_average=history_average,
        )

    if thermal_blocked or _thermal_limit_exceeded(snapshot, config):
        return HolderDecision(
            gpu_index=snapshot.index,
            action=DecisionAction.RELEASE if holder_running else DecisionAction.STANDBY,
            reason="thermal_limit",
            machine_average=machine_average,
            machine_policy_average=effective_machine_average,
            history_average=history_average,
        )

    if process_grace:
        return HolderDecision(
            gpu_index=snapshot.index,
            action=DecisionAction.RELEASE if holder_running else DecisionAction.STANDBY,
            reason="process_grace",
            machine_average=machine_average,
            machine_policy_average=effective_machine_average,
            history_average=history_average,
        )

    if _has_protected_process(snapshot, config):
        return HolderDecision(
            gpu_index=snapshot.index,
            action=DecisionAction.RELEASE if holder_running else DecisionAction.STANDBY,
            reason="protected_process",
            machine_average=machine_average,
            machine_policy_average=effective_machine_average,
            history_average=history_average,
        )

    reserve_bytes = config.reserve.resolve(snapshot.memory_total_bytes)
    busy_threshold = config.busy_process_mem_threshold.resolve(snapshot.memory_total_bytes)
    busy_processes = [
        process
        for process in snapshot.non_holder_processes()
        if process.used_memory_bytes >= busy_threshold
    ]
    low_util_emergency = history.full_window_below(
        now=snapshot.timestamp,
        threshold=config.idle_util,
    )

    if low_util_emergency:
        if busy_processes:
            memory_bytes = min(
                config.assist_mem.resolve(snapshot.memory_total_bytes, reserve_bytes=reserve_bytes),
                _free_memory(snapshot, reserve_bytes=reserve_bytes),
            )
            return HolderDecision(
                gpu_index=snapshot.index,
                action=DecisionAction.ASSIST,
                reason="low_util_emergency_with_busy_process",
                memory_bytes=max(0, memory_bytes),
                duty_cycle=_adaptive_duty_cycle(
                    current_util=snapshot.utilization,
                    history_average=history_average,
                    machine_average=effective_machine_average,
                    config=config,
                    emergency=True,
                ),
                program=config.program,
                hold_mode="assist",
                machine_average=machine_average,
                machine_policy_average=effective_machine_average,
                history_average=history_average,
            )
        memory_bytes = _memory_for_mode(
            mode=config.hold_mode,
            snapshot=snapshot,
            config=config,
            reserve_bytes=reserve_bytes,
        )
        if _memory_only_without_memory(config.hold_mode, memory_bytes):
            return _insufficient_free_memory_decision(
                snapshot=snapshot,
                holder_running=holder_running,
                machine_average=machine_average,
                machine_policy_average=effective_machine_average,
                history_average=history_average,
            )
        return HolderDecision(
            gpu_index=snapshot.index,
            action=DecisionAction.HOLD,
            reason="low_util_emergency",
            memory_bytes=memory_bytes,
            duty_cycle=_duty_for_mode(
                mode=config.hold_mode,
                current_util=snapshot.utilization,
                history_average=history_average,
                machine_average=effective_machine_average,
                config=config,
                emergency=True,
            ),
            program=config.program,
            hold_mode=config.hold_mode,
            machine_average=machine_average,
            machine_policy_average=effective_machine_average,
            history_average=history_average,
        )

    if busy_processes:
        return HolderDecision(
            gpu_index=snapshot.index,
            action=DecisionAction.RELEASE if holder_running else DecisionAction.STANDBY,
            reason="busy_process",
            machine_average=machine_average,
            machine_policy_average=effective_machine_average,
            history_average=history_average,
        )

    if machine_needs_help and snapshot.utilization < config.target_util:
        memory_bytes = _memory_for_mode(
            mode=config.hold_mode,
            snapshot=snapshot,
            config=config,
            reserve_bytes=reserve_bytes,
        )
        if _memory_only_without_memory(config.hold_mode, memory_bytes):
            return _insufficient_free_memory_decision(
                snapshot=snapshot,
                holder_running=holder_running,
                machine_average=machine_average,
                machine_policy_average=effective_machine_average,
                history_average=history_average,
            )
        return HolderDecision(
            gpu_index=snapshot.index,
            action=DecisionAction.HOLD,
            reason="below_target",
            memory_bytes=memory_bytes,
            duty_cycle=_duty_for_mode(
                mode=config.hold_mode,
                current_util=snapshot.utilization,
                history_average=history_average,
                machine_average=effective_machine_average,
                config=config,
                emergency=False,
            ),
            program=config.program,
            hold_mode=config.hold_mode,
            machine_average=machine_average,
            machine_policy_average=effective_machine_average,
            history_average=history_average,
        )

    return HolderDecision(
        gpu_index=snapshot.index,
        action=DecisionAction.RELEASE if holder_running else DecisionAction.STANDBY,
        reason="machine_target_met" if not machine_needs_help else "target_met",
        machine_average=machine_average,
        machine_policy_average=effective_machine_average,
        history_average=history_average,
    )


def decide_for_fleet(
    snapshots: list[GpuSnapshot],
    histories: dict[int, GpuHistory],
    config: GuardConfig,
    *,
    running_gpu_indices: set[int],
    paused: bool = False,
    process_grace_gpu_indices: set[int] | None = None,
    disabled_gpu_indices: set[int] | None = None,
    thermal_blocked_gpu_indices: set[int] | None = None,
    machine_policy_average: float | None = None,
) -> list[HolderDecision]:
    machine_average = average_utilization(snapshots)
    effective_machine_average = (
        machine_average if machine_policy_average is None else machine_policy_average
    )
    machine_needs_help = effective_machine_average < float(config.target_util)
    decisions: list[HolderDecision] = []
    for snapshot in snapshots:
        history = histories[snapshot.index]
        decisions.append(
            decide_for_gpu(
                snapshot,
                history,
                config,
                holder_running=snapshot.index in running_gpu_indices,
                paused=paused,
                disabled=(
                    disabled_gpu_indices is not None
                    and snapshot.index in disabled_gpu_indices
                ),
                thermal_blocked=(
                    thermal_blocked_gpu_indices is not None
                    and snapshot.index in thermal_blocked_gpu_indices
                ),
                process_grace=(
                    process_grace_gpu_indices is not None
                    and snapshot.index in process_grace_gpu_indices
                ),
                machine_needs_help=machine_needs_help,
                machine_average=machine_average,
                machine_policy_average=effective_machine_average,
            )
        )
    return _apply_max_held_gpus(
        decisions,
        config=config,
        running_gpu_indices=running_gpu_indices,
    )


def average_utilization(snapshots: list[GpuSnapshot]) -> float:
    if not snapshots:
        return 0.0
    return sum(snapshot.utilization for snapshot in snapshots) / len(snapshots)


def _thermal_limit_exceeded(snapshot: GpuSnapshot, config: GuardConfig) -> bool:
    if config.max_gpu_temp is None or snapshot.temperature_celsius is None:
        return False
    return int(snapshot.temperature_celsius) >= int(config.max_gpu_temp)


def _has_protected_process(snapshot: GpuSnapshot, config: GuardConfig) -> bool:
    return has_protected_process(
        snapshot.non_holder_processes(),
        config.protected_process_patterns,
    )


def _apply_max_held_gpus(
    decisions: list[HolderDecision],
    *,
    config: GuardConfig,
    running_gpu_indices: set[int],
) -> list[HolderDecision]:
    if config.max_held_gpus is None:
        return decisions
    limit = int(config.max_held_gpus)
    holding_actions = {DecisionAction.HOLD, DecisionAction.ASSIST}
    candidates = [decision for decision in decisions if decision.action in holding_actions]
    if len(candidates) <= limit:
        return decisions
    keep = {
        decision.gpu_index
        for decision in sorted(
            candidates,
            key=lambda decision: _holder_priority(decision, running_gpu_indices),
            reverse=True,
        )[:limit]
    }
    capped: list[HolderDecision] = []
    for decision in decisions:
        if decision.action not in holding_actions or decision.gpu_index in keep:
            capped.append(decision)
            continue
        capped.append(
            HolderDecision(
                gpu_index=decision.gpu_index,
                action=(
                    DecisionAction.RELEASE
                    if decision.gpu_index in running_gpu_indices
                    else DecisionAction.STANDBY
                ),
                reason="max_held_gpus_exceeded",
                machine_average=decision.machine_average,
                machine_policy_average=decision.machine_policy_average,
                history_average=decision.history_average,
                program=decision.program,
                hold_mode=decision.hold_mode,
            )
        )
    return capped


def _holder_priority(
    decision: HolderDecision,
    running_gpu_indices: set[int],
) -> tuple[float, float, float, float]:
    emergency_rank = 1.0 if decision.reason.startswith("low_util_emergency") else 0.0
    running_rank = 1.0 if decision.gpu_index in running_gpu_indices else 0.0
    history_gap = _normalized_gap(max(1.0, decision.machine_average), decision.history_average)
    return (
        emergency_rank,
        decision.duty_cycle,
        history_gap,
        running_rank,
    )


def _free_memory(snapshot: GpuSnapshot, *, reserve_bytes: int) -> int:
    return max(0, snapshot.memory_total_bytes - snapshot.memory_used_bytes - reserve_bytes)


def _adaptive_duty_cycle(
    *,
    current_util: int,
    history_average: float,
    machine_average: float,
    config: GuardConfig,
    emergency: bool,
) -> float:
    current_gap = _normalized_gap(config.target_util, current_util)
    history_gap = _normalized_gap(config.target_util, history_average)
    machine_gap = _normalized_gap(config.target_util, machine_average)
    weighted = (
        current_gap * float(config.current_gap_weight)
        + history_gap * float(config.history_gap_weight)
        + machine_gap * float(config.machine_gap_weight)
    )
    if emergency:
        weighted = max(weighted, _normalized_gap(config.idle_util, history_average))
    if weighted <= 0:
        return 0.0
    duty = max(config.min_duty_cycle, min(1.0, weighted))
    return min(config.max_duty_cycle, duty)


def _normalized_gap(target: float, value: float) -> float:
    target = max(1.0, float(target))
    return min(1.0, max(0.0, (target - float(value)) / target))


def _memory_for_mode(
    *,
    mode: str,
    snapshot: GpuSnapshot,
    config: GuardConfig,
    reserve_bytes: int,
) -> int:
    if mode == "compute-only":
        return 0
    requested = config.mem.resolve(snapshot.memory_total_bytes, reserve_bytes=reserve_bytes)
    return min(requested, _free_memory(snapshot, reserve_bytes=reserve_bytes))


def _memory_only_without_memory(mode: str, memory_bytes: int) -> bool:
    return mode == "memory-only" and memory_bytes <= 0


def _insufficient_free_memory_decision(
    *,
    snapshot: GpuSnapshot,
    holder_running: bool,
    machine_average: float,
    machine_policy_average: float,
    history_average: float,
) -> HolderDecision:
    return HolderDecision(
        gpu_index=snapshot.index,
        action=DecisionAction.RELEASE if holder_running else DecisionAction.STANDBY,
        reason="insufficient_free_memory",
        machine_average=machine_average,
        machine_policy_average=machine_policy_average,
        history_average=history_average,
    )


def _duty_for_mode(
    *,
    mode: str,
    current_util: int,
    history_average: float,
    machine_average: float,
    config: GuardConfig,
    emergency: bool,
) -> float:
    if mode == "memory-only":
        return 0.0
    return _adaptive_duty_cycle(
        current_util=current_util,
        history_average=history_average,
        machine_average=machine_average,
        config=config,
        emergency=emergency,
    )
