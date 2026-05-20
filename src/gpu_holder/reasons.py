from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ReasonInfo:
    reason: str
    actions: tuple[str, ...]
    priority: int
    meaning: str
    operator_guidance: str


REASONS = [
    ReasonInfo(
        reason="paused",
        actions=("release", "standby"),
        priority=10,
        meaning="The global pause file exists, so all holder workers should be released.",
        operator_guidance="Run `gpu-holder resume` or remove the configured pause file to allow holders again.",
    ),
    ReasonInfo(
        reason="gpu_disabled",
        actions=("release", "standby"),
        priority=20,
        meaning="This GPU is disabled in runtime state and is skipped by the scheduler.",
        operator_guidance="Run `gpu-holder enable-gpu <index>` or use dashboard `d` on the selected GPU.",
    ),
    ReasonInfo(
        reason="thermal_limit",
        actions=("release", "standby"),
        priority=30,
        meaning="The GPU is at or above the configured thermal limit, or thermal hysteresis is still active.",
        operator_guidance="Wait for the GPU to cool, raise limits cautiously, or disable with `--max-gpu-temp 0` if appropriate.",
    ),
    ReasonInfo(
        reason="process_grace",
        actions=("release", "standby"),
        priority=40,
        meaning="A new non-holder GPU process appeared and receives a startup grace window.",
        operator_guidance="Wait for the grace window to expire, or set `process_grace_window = 0` if startup yielding is not needed.",
    ),
    ReasonInfo(
        reason="protected_process",
        actions=("release", "standby"),
        priority=45,
        meaning="A non-holder process name matched protected_process_patterns, so this GPU is explicitly protected.",
        operator_guidance="This is intentional yielding. Edit `protected_process_patterns` or `--protected-process` if the match is too broad.",
    ),
    ReasonInfo(
        reason="low_util_emergency_with_busy_process",
        actions=("assist",),
        priority=50,
        meaning="The GPU stayed below idle_util for the full idle_window even though a large non-holder process is present.",
        operator_guidance="Inspect the user workload; gpu-holder uses smaller assist memory to reduce OOM risk while raising utilization.",
    ),
    ReasonInfo(
        reason="low_util_emergency",
        actions=("hold",),
        priority=60,
        meaning="The GPU stayed below idle_util for the full idle_window, so emergency utilization support is active.",
        operator_guidance="Adjust idle_util, idle_window, hold_mode, or max_held_gpus if this is too aggressive.",
    ),
    ReasonInfo(
        reason="busy_process",
        actions=("release", "standby"),
        priority=70,
        meaning="A non-holder process is using at least busy_process_mem_threshold on this GPU.",
        operator_guidance="This is normal yielding behavior. Tune busy_process_mem_threshold only if jobs are misclassified.",
    ),
    ReasonInfo(
        reason="insufficient_free_memory",
        actions=("release", "standby"),
        priority=75,
        meaning="The selected memory-only holder would allocate zero bytes after current usage and reserve memory are respected.",
        operator_guidance="Free memory on the GPU, lower mem/reserve, or use balanced/compute-only mode if utilization support is needed.",
    ),
    ReasonInfo(
        reason="below_target",
        actions=("hold",),
        priority=80,
        meaning="The machine policy average is below target_util and this GPU is below target_util.",
        operator_guidance="This is normal holder behavior. Tune target_util, duty weights, or hold_mode if needed.",
    ),
    ReasonInfo(
        reason="machine_target_met",
        actions=("release", "standby"),
        priority=90,
        meaning="The rolling machine policy average already meets target_util.",
        operator_guidance="No action needed; gpu-holder is backing off because the fleet target is satisfied.",
    ),
    ReasonInfo(
        reason="target_met",
        actions=("release", "standby"),
        priority=100,
        meaning="The machine still needs help, but this specific GPU is already at or above target_util.",
        operator_guidance="No action needed; other lower-util GPUs may be preferred.",
    ),
    ReasonInfo(
        reason="max_held_gpus_exceeded",
        actions=("release", "standby"),
        priority=110,
        meaning="More GPUs wanted holder workers than max_held_gpus allows, so this GPU was capped.",
        operator_guidance="Raise max_held_gpus or inspect policy priority if too few GPUs are being held.",
    ),
]


def reason_reference() -> list[dict[str, object]]:
    return [
        {
            **asdict(reason),
            "actions": list(reason.actions),
        }
        for reason in REASONS
    ]
