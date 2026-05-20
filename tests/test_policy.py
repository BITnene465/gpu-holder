from __future__ import annotations

from gpu_holder.config import GuardConfig
from gpu_holder.models import DecisionAction, GpuHistory, GpuProcess, GpuSnapshot
from gpu_holder.policy import average_utilization, decide_for_fleet, decide_for_gpu


GIB = 1024**3


def snapshot(
    *,
    utilization: int,
    used_gib: int = 2,
    temperature_celsius: int | None = None,
    processes: tuple[GpuProcess, ...] = (),
    timestamp: float = 100.0,
    index: int = 0,
) -> GpuSnapshot:
    return GpuSnapshot(
        index=index,
        name="test-gpu",
        memory_total_bytes=80 * GIB,
        memory_used_bytes=used_gib * GIB,
        utilization=utilization,
        temperature_celsius=temperature_celsius,
        processes=processes,
        timestamp=timestamp,
    )


def low_history(*, value: int = 10, now: float = 100.0) -> GpuHistory:
    history = GpuHistory(window_seconds=60)
    for offset in range(61):
        history.add(now - 60 + offset, value)
    return history


def short_low_history(*, value: int = 10, now: float = 100.0) -> GpuHistory:
    history = GpuHistory(window_seconds=60)
    for offset in range(30):
        history.add(now - 30 + offset, value)
    return history


def test_default_idle_gpu_holds_twenty_percent_memory() -> None:
    config = GuardConfig()
    snap = snapshot(utilization=20)

    decision = decide_for_gpu(snap, short_low_history(), config, holder_running=False)

    assert decision.action == DecisionAction.HOLD
    assert decision.reason == "below_target"
    assert decision.memory_bytes == 16 * GIB
    assert decision.duty_cycle > 0
    assert decision.hold_mode == "balanced"
    assert decision.history_average == 10


def test_compute_only_mode_is_carried_to_decision() -> None:
    config = GuardConfig(hold_mode="compute-only")
    snap = snapshot(utilization=20)

    decision = decide_for_gpu(snap, short_low_history(), config, holder_running=False)

    assert decision.action == DecisionAction.HOLD
    assert decision.hold_mode == "compute-only"
    assert decision.memory_bytes == 0


def test_memory_only_mode_has_no_compute_duty() -> None:
    config = GuardConfig(hold_mode="memory-only")
    snap = snapshot(utilization=20)

    decision = decide_for_gpu(snap, short_low_history(), config, holder_running=False)

    assert decision.action == DecisionAction.HOLD
    assert decision.hold_mode == "memory-only"
    assert decision.memory_bytes == 16 * GIB
    assert decision.duty_cycle == 0.0


def test_hold_memory_is_capped_by_current_free_memory_after_reserve() -> None:
    config = GuardConfig()
    snap = snapshot(utilization=20, used_gib=77)

    decision = decide_for_gpu(snap, short_low_history(), config, holder_running=False)

    assert decision.action == DecisionAction.HOLD
    assert decision.memory_bytes == 1 * GIB


def test_memory_only_standby_when_no_free_memory_after_reserve() -> None:
    config = GuardConfig(hold_mode="memory-only")
    snap = snapshot(utilization=20, used_gib=79)

    decision = decide_for_gpu(snap, short_low_history(), config, holder_running=False)

    assert decision.action == DecisionAction.STANDBY
    assert decision.reason == "insufficient_free_memory"
    assert decision.memory_bytes == 0


def test_memory_only_releases_running_holder_when_no_free_memory_after_reserve() -> None:
    config = GuardConfig(hold_mode="memory-only")
    snap = snapshot(utilization=20, used_gib=79)

    decision = decide_for_gpu(snap, short_low_history(), config, holder_running=True)

    assert decision.action == DecisionAction.RELEASE
    assert decision.reason == "insufficient_free_memory"


def test_busy_process_yields_when_utilization_is_not_emergency() -> None:
    config = GuardConfig()
    busy = GpuProcess(pid=1234, used_memory_bytes=12 * GIB, name="train.py")
    snap = snapshot(utilization=70, used_gib=20, processes=(busy,))

    decision = decide_for_gpu(snap, short_low_history(), config, holder_running=True)

    assert decision.action == DecisionAction.RELEASE
    assert decision.reason == "busy_process"


def test_busy_process_standby_when_no_holder_running() -> None:
    config = GuardConfig()
    busy = GpuProcess(pid=1234, used_memory_bytes=12 * GIB, name="train.py")
    snap = snapshot(utilization=70, used_gib=20, processes=(busy,))

    decision = decide_for_gpu(snap, short_low_history(), config, holder_running=False)

    assert decision.action == DecisionAction.STANDBY
    assert decision.reason == "busy_process"


def test_sixty_seconds_below_idle_util_forces_hold() -> None:
    config = GuardConfig()
    snap = snapshot(utilization=20)

    decision = decide_for_gpu(snap, low_history(), config, holder_running=False)

    assert decision.action == DecisionAction.HOLD
    assert decision.reason == "low_util_emergency"


def test_low_util_emergency_overrides_busy_process_with_assist_mode() -> None:
    config = GuardConfig()
    busy = GpuProcess(pid=1234, used_memory_bytes=12 * GIB, name="train.py")
    snap = snapshot(utilization=20, used_gib=20, processes=(busy,))

    decision = decide_for_gpu(snap, low_history(), config, holder_running=False)

    assert decision.action == DecisionAction.ASSIST
    assert decision.reason == "low_util_emergency_with_busy_process"
    assert decision.memory_bytes == 512 * 1024**2
    assert decision.hold_mode == "assist"


def test_pause_releases_only_holder() -> None:
    config = GuardConfig()
    busy = GpuProcess(pid=1234, used_memory_bytes=12 * GIB, name="train.py")
    snap = snapshot(utilization=20, used_gib=20, processes=(busy,))

    decision = decide_for_gpu(snap, low_history(), config, holder_running=True, paused=True)

    assert decision.action == DecisionAction.RELEASE
    assert decision.reason == "paused"


def test_process_grace_yields_even_during_low_util_emergency() -> None:
    config = GuardConfig()
    snap = snapshot(utilization=10)

    decision = decide_for_gpu(
        snap,
        low_history(value=10),
        config,
        holder_running=True,
        process_grace=True,
    )

    assert decision.action == DecisionAction.RELEASE
    assert decision.reason == "process_grace"


def test_protected_process_yields_even_during_low_util_emergency() -> None:
    config = GuardConfig(protected_process_patterns=("python*train.py",))
    process = GpuProcess(pid=1234, used_memory_bytes=1 * GIB, name="Python train.py")
    snap = snapshot(utilization=10, processes=(process,))

    decision = decide_for_gpu(
        snap,
        low_history(value=10),
        config,
        holder_running=True,
    )

    assert decision.action == DecisionAction.RELEASE
    assert decision.reason == "protected_process"


def test_protected_process_ignores_holder_processes() -> None:
    config = GuardConfig(protected_process_patterns=("gpu-holder",))
    process = GpuProcess(
        pid=1234,
        used_memory_bytes=1 * GIB,
        name="gpu-holder",
        is_holder=True,
    )
    snap = snapshot(utilization=10, processes=(process,))

    decision = decide_for_gpu(snap, low_history(value=10), config, holder_running=True)

    assert decision.reason == "low_util_emergency"


def test_disabled_gpu_releases_even_during_low_util_emergency() -> None:
    config = GuardConfig()
    snap = snapshot(utilization=10)

    decision = decide_for_gpu(
        snap,
        low_history(value=10),
        config,
        holder_running=True,
        disabled=True,
    )

    assert decision.action == DecisionAction.RELEASE
    assert decision.reason == "gpu_disabled"


def test_thermal_limit_releases_even_during_low_util_emergency() -> None:
    config = GuardConfig(max_gpu_temp=80)
    snap = snapshot(utilization=10, temperature_celsius=82)

    decision = decide_for_gpu(
        snap,
        low_history(value=10),
        config,
        holder_running=True,
    )

    assert decision.action == DecisionAction.RELEASE
    assert decision.reason == "thermal_limit"


def test_thermal_limit_standby_when_holder_is_not_running() -> None:
    config = GuardConfig(max_gpu_temp=80)
    snap = snapshot(utilization=10, temperature_celsius=80)

    decision = decide_for_gpu(
        snap,
        low_history(value=10),
        config,
        holder_running=False,
    )

    assert decision.action == DecisionAction.STANDBY
    assert decision.reason == "thermal_limit"


def test_thermal_block_releases_until_controller_allows_resume() -> None:
    config = GuardConfig(max_gpu_temp=85, thermal_resume_temp=80)
    snap = snapshot(utilization=10, temperature_celsius=83)

    decision = decide_for_gpu(
        snap,
        low_history(value=10),
        config,
        holder_running=True,
        thermal_blocked=True,
    )

    assert decision.action == DecisionAction.RELEASE
    assert decision.reason == "thermal_limit"


def test_fleet_target_met_does_not_start_non_emergency_holder() -> None:
    config = GuardConfig(target_util=75)
    snapshots = [
        snapshot(index=0, utilization=60),
        snapshot(index=1, utilization=95),
    ]
    histories = {
        0: short_low_history(value=60),
        1: short_low_history(value=95),
    }

    decisions = decide_for_fleet(
        snapshots,
        histories,
        config,
        running_gpu_indices=set(),
    )

    assert average_utilization(snapshots) >= 75
    assert decisions[0].action == DecisionAction.STANDBY
    assert decisions[0].reason == "machine_target_met"


def test_fleet_uses_machine_policy_average_to_decide_if_help_is_needed() -> None:
    config = GuardConfig(target_util=75)
    snapshots = [
        snapshot(index=0, utilization=85),
        snapshot(index=1, utilization=85),
    ]
    histories = {
        0: short_low_history(value=85),
        1: short_low_history(value=85),
    }

    decisions = decide_for_fleet(
        snapshots,
        histories,
        config,
        running_gpu_indices=set(),
        machine_policy_average=60.0,
    )

    assert average_utilization(snapshots) >= 75
    assert decisions[0].action == DecisionAction.STANDBY
    assert decisions[0].reason == "target_met"
    assert decisions[0].machine_average == 85
    assert decisions[0].machine_policy_average == 60.0


def test_fleet_releases_when_machine_policy_average_is_above_target() -> None:
    config = GuardConfig(target_util=75)
    snapshots = [
        snapshot(index=0, utilization=50),
        snapshot(index=1, utilization=50),
    ]
    histories = {
        0: short_low_history(value=50),
        1: short_low_history(value=50),
    }

    decisions = decide_for_fleet(
        snapshots,
        histories,
        config,
        running_gpu_indices={0},
        machine_policy_average=80.0,
    )

    assert average_utilization(snapshots) < 75
    assert decisions[0].action == DecisionAction.RELEASE
    assert decisions[0].reason == "machine_target_met"
    assert decisions[0].machine_policy_average == 80.0


def test_fleet_target_met_still_allows_low_util_emergency() -> None:
    config = GuardConfig(target_util=75)
    snapshots = [
        snapshot(index=0, utilization=10),
        snapshot(index=1, utilization=100),
        snapshot(index=2, utilization=100),
        snapshot(index=3, utilization=100),
    ]
    histories = {
        0: low_history(value=10),
        1: short_low_history(value=100),
        2: short_low_history(value=100),
        3: short_low_history(value=100),
    }

    decisions = decide_for_fleet(
        snapshots,
        histories,
        config,
        running_gpu_indices=set(),
    )

    assert average_utilization(snapshots) >= 75
    assert decisions[0].action == DecisionAction.HOLD
    assert decisions[0].reason == "low_util_emergency"
    assert decisions[0].machine_average == average_utilization(snapshots)


def test_adaptive_duty_increases_when_machine_average_is_lower() -> None:
    config = GuardConfig(target_util=75)
    histories_low = {
        0: short_low_history(value=50),
        1: short_low_history(value=20),
    }
    low_machine = [
        snapshot(index=0, utilization=50),
        snapshot(index=1, utilization=20),
    ]
    high_machine = [
        snapshot(index=0, utilization=50),
        snapshot(index=1, utilization=90),
    ]
    histories_high = {
        0: short_low_history(value=50),
        1: short_low_history(value=90),
    }

    low_decision = decide_for_fleet(
        low_machine,
        histories_low,
        config,
        running_gpu_indices=set(),
    )[0]
    high_decision = decide_for_fleet(
        high_machine,
        histories_high,
        config,
        running_gpu_indices=set(),
    )[0]

    assert low_decision.duty_cycle > high_decision.duty_cycle
    assert low_decision.machine_average < high_decision.machine_average


def test_max_held_gpus_limits_new_holder_count_by_priority() -> None:
    config = GuardConfig(target_util=75, max_held_gpus=2)
    snapshots = [
        snapshot(index=0, utilization=35),
        snapshot(index=1, utilization=5),
        snapshot(index=2, utilization=25),
        snapshot(index=3, utilization=10),
    ]
    histories = {
        0: short_low_history(value=35),
        1: short_low_history(value=5),
        2: short_low_history(value=25),
        3: short_low_history(value=10),
    }

    decisions = decide_for_fleet(
        snapshots,
        histories,
        config,
        running_gpu_indices=set(),
    )

    held = [decision.gpu_index for decision in decisions if decision.action == DecisionAction.HOLD]
    standby = [decision for decision in decisions if decision.action == DecisionAction.STANDBY]
    assert held == [1, 3]
    assert {decision.gpu_index for decision in standby} == {0, 2}
    assert {decision.reason for decision in standby} == {"max_held_gpus_exceeded"}


def test_max_held_gpus_preserves_existing_holder_when_priority_ties() -> None:
    config = GuardConfig(target_util=75, max_held_gpus=1)
    snapshots = [
        snapshot(index=0, utilization=20),
        snapshot(index=1, utilization=20),
    ]
    histories = {
        0: short_low_history(value=20),
        1: short_low_history(value=20),
    }

    decisions = decide_for_fleet(
        snapshots,
        histories,
        config,
        running_gpu_indices={1},
    )

    assert decisions[0].action == DecisionAction.STANDBY
    assert decisions[0].reason == "max_held_gpus_exceeded"
    assert decisions[1].action == DecisionAction.HOLD


def test_max_held_gpus_zero_releases_running_holders() -> None:
    config = GuardConfig(target_util=75, max_held_gpus=0)
    snapshots = [
        snapshot(index=0, utilization=5),
        snapshot(index=1, utilization=10),
    ]
    histories = {
        0: low_history(value=5),
        1: low_history(value=10),
    }

    decisions = decide_for_fleet(
        snapshots,
        histories,
        config,
        running_gpu_indices={0},
    )

    assert decisions[0].action == DecisionAction.RELEASE
    assert decisions[0].reason == "max_held_gpus_exceeded"
    assert decisions[1].action == DecisionAction.STANDBY
    assert decisions[1].reason == "max_held_gpus_exceeded"


def test_fleet_process_grace_applies_to_selected_gpu_only() -> None:
    config = GuardConfig(target_util=75)
    snapshots = [
        snapshot(index=0, utilization=10),
        snapshot(index=1, utilization=10),
    ]
    histories = {
        0: low_history(value=10),
        1: low_history(value=10),
    }

    decisions = decide_for_fleet(
        snapshots,
        histories,
        config,
        running_gpu_indices={0, 1},
        process_grace_gpu_indices={1},
    )

    assert decisions[0].action == DecisionAction.HOLD
    assert decisions[0].reason == "low_util_emergency"
    assert decisions[1].action == DecisionAction.RELEASE
    assert decisions[1].reason == "process_grace"


def test_fleet_disabled_gpus_apply_to_selected_gpu_only() -> None:
    config = GuardConfig(target_util=75)
    snapshots = [
        snapshot(index=0, utilization=10),
        snapshot(index=1, utilization=10),
    ]
    histories = {
        0: low_history(value=10),
        1: low_history(value=10),
    }

    decisions = decide_for_fleet(
        snapshots,
        histories,
        config,
        running_gpu_indices={0, 1},
        disabled_gpu_indices={1},
    )

    assert decisions[0].action == DecisionAction.HOLD
    assert decisions[0].reason == "low_util_emergency"
    assert decisions[1].action == DecisionAction.RELEASE
    assert decisions[1].reason == "gpu_disabled"
