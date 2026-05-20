from __future__ import annotations

import json
import os
from pathlib import Path
import time

import gpu_holder.controller as controller_mod
import pytest
from gpu_holder.controller import GuardController, build_status_payload
from gpu_holder.config import GuardConfig, config_payload
from gpu_holder.events import read_events, write_event
from gpu_holder.models import DecisionAction, GpuProcess, GpuSnapshot, HolderDecision
from gpu_holder.state import pause_guard


def test_status_payload_preserves_external_processes_without_control_actions() -> None:
    external = GpuProcess(pid=1234, used_memory_bytes=12 * 1024**3, name="train.py")
    snapshot = GpuSnapshot(
        index=0,
        name="test",
        memory_total_bytes=80 * 1024**3,
        memory_used_bytes=20 * 1024**3,
        utilization=70,
        temperature_celsius=86,
        processes=(external,),
        timestamp=1.0,
    )
    decision = HolderDecision(
        gpu_index=0,
        action=DecisionAction.RELEASE,
        reason="busy_process",
    )

    payload = build_status_payload(
        snapshots=[snapshot],
        decisions=[decision],
        holder_pids=set(),
        paused=False,
        workers={},
        worker_backoffs={},
        config={
            "target_util": 75,
            "max_gpu_temp": 85,
            "protected_process_patterns": ["train.py"],
        },
    )

    gpu = payload["gpus"][0]
    assert payload["machine"]["average_utilization"] == 70
    assert payload["machine"]["action_counts"] == {"release": 1}
    assert payload["machine"]["owned_worker_count"] == 0
    assert payload["machine"]["thermal_limit_count"] == 1
    assert payload["config"]["target_util"] == 75
    assert gpu["temperature_celsius"] == 86
    assert gpu["thermal_limit"] is True
    assert gpu["processes"][0]["pid"] == 1234
    assert gpu["processes"][0]["protected_process"] is True
    assert gpu["processes"][0]["protected_process_pattern"] == "train.py"
    assert gpu["decision"]["action"] == "release"
    assert gpu["worker"] is None
    assert "kill" not in str(payload).lower()


def test_status_payload_does_not_mark_holder_as_protected_process() -> None:
    holder = GpuProcess(
        pid=1234,
        used_memory_bytes=1024,
        name="gpu-holder",
        is_holder=True,
    )
    snapshot = GpuSnapshot(
        index=0,
        name="test",
        memory_total_bytes=80 * 1024**3,
        memory_used_bytes=20 * 1024**3,
        utilization=70,
        processes=(holder,),
        timestamp=1.0,
    )
    decision = HolderDecision(
        gpu_index=0,
        action=DecisionAction.STANDBY,
        reason="target_met",
    )

    payload = build_status_payload(
        snapshots=[snapshot],
        decisions=[decision],
        holder_pids={1234},
        paused=False,
        config={"protected_process_patterns": ["gpu-holder"]},
    )

    process = payload["gpus"][0]["processes"][0]
    assert process["protected_process"] is False
    assert process["protected_process_pattern"] is None


def test_status_payload_includes_owned_worker_and_backoff_state() -> None:
    snapshot = GpuSnapshot(
        index=0,
        name="test",
        memory_total_bytes=80 * 1024**3,
        memory_used_bytes=20 * 1024**3,
        utilization=20,
        timestamp=1.0,
    )
    decision = HolderDecision(
        gpu_index=0,
        action=DecisionAction.HOLD,
        reason="below_target",
        memory_bytes=4 * 1024**3,
        duty_cycle=0.4,
    )

    payload = build_status_payload(
        snapshots=[snapshot],
        decisions=[decision],
        holder_pids={4321},
        paused=False,
        workers={
            0: {
                "gpu_index": 0,
                "pid": 4321,
                "alive": True,
                "exitcode": None,
                "memory_bytes": 4 * 1024**3,
                "memory_human": "4.00GiB",
                "duty_cycle": 0.4,
                "program": "mixed",
                "hold_mode": "balanced",
            }
        },
        worker_backoffs={},
        config={},
    )

    gpu = payload["gpus"][0]
    assert payload["holder_pids"] == [4321]
    assert payload["machine"]["owned_worker_count"] == 1
    assert gpu["worker"]["pid"] == 4321
    assert gpu["worker"]["memory_human"] == "4.00GiB"
    assert gpu["worker_restart_backoff"] is None


def test_controller_status_includes_resolved_config(tmp_path) -> None:
    controller = GuardController(GuardConfig(state_dir=tmp_path, target_util=82))
    snapshot = GpuSnapshot(
        index=0,
        name="test",
        memory_total_bytes=80 * 1024**3,
        memory_used_bytes=0,
        utilization=0,
        timestamp=1.0,
    )
    decision = HolderDecision(
        gpu_index=0,
        action=DecisionAction.HOLD,
        reason="below_target",
    )

    controller.write_status(snapshots=[snapshot], decisions=[decision])

    payload = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert payload["config"] == config_payload(controller.config)


def test_controller_status_write_keeps_previous_status_when_replace_fails(
    tmp_path,
    monkeypatch,
) -> None:
    status_file = tmp_path / "status.json"
    status_file.write_text('{"previous": true}\n', encoding="utf-8")
    controller = GuardController(GuardConfig(state_dir=tmp_path))
    original_replace = Path.replace
    replace_calls: list[Path] = []

    def fail_status_replace(self: Path, target: str | Path) -> Path:
        if Path(target) == status_file and self.name.startswith(f".{status_file.name}."):
            replace_calls.append(self)
            raise RuntimeError("replace failed")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_status_replace)

    with pytest.raises(RuntimeError, match="replace failed"):
        controller.write_status(snapshots=[], decisions=[])

    assert replace_calls
    assert json.loads(status_file.read_text(encoding="utf-8")) == {"previous": True}
    assert list(tmp_path.glob(f".{status_file.name}.*.tmp")) == []


def test_controller_pidfile_cleanup_removes_only_current_process_pid(tmp_path) -> None:
    controller = GuardController(GuardConfig(state_dir=tmp_path))
    controller.config.pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")

    controller._remove_pid_file_if_current_process()

    assert not controller.config.pid_file.exists()


def test_controller_pidfile_cleanup_preserves_replaced_pidfile(tmp_path) -> None:
    controller = GuardController(GuardConfig(state_dir=tmp_path))
    controller.config.pid_file.write_text(f"{os.getpid() + 1}\n", encoding="utf-8")

    controller._remove_pid_file_if_current_process()

    assert controller.config.pid_file.read_text(encoding="utf-8") == f"{os.getpid() + 1}\n"


def test_controller_status_includes_machine_window_average(tmp_path) -> None:
    controller = GuardController(GuardConfig(state_dir=tmp_path, machine_window=3600.0))
    snapshots = [
        GpuSnapshot(
            index=0,
            name="test-0",
            memory_total_bytes=80 * 1024**3,
            memory_used_bytes=0,
            utilization=40,
            timestamp=1.0,
        ),
        GpuSnapshot(
            index=1,
            name="test-1",
            memory_total_bytes=80 * 1024**3,
            memory_used_bytes=0,
            utilization=80,
            timestamp=1.0,
        ),
    ]

    decisions = controller.decide(snapshots)
    controller.write_status(snapshots=snapshots, decisions=decisions)

    payload = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert payload["machine"]["average_utilization"] == 60.0
    assert payload["machine"]["window_average_utilization"] == 60.0
    assert payload["machine"]["policy_average_utilization"] == 60.0
    assert payload["machine"]["policy_window_seconds"] == 3600.0
    assert payload["machine"]["policy_window_sample_count"] == 1
    assert payload["machine"]["policy_window_coverage_seconds"] == 0.0
    assert payload["machine"]["policy_window_coverage_ratio"] == 0.0


def test_controller_ignores_expired_temporary_pause(tmp_path) -> None:
    pause_file = pause_guard(tmp_path, duration_seconds=1, now=1)
    controller = GuardController(GuardConfig(state_dir=tmp_path))
    snapshot = GpuSnapshot(
        index=0,
        name="test",
        memory_total_bytes=80 * 1024**3,
        memory_used_bytes=0,
        utilization=0,
        timestamp=time.time(),
    )

    decisions = controller.decide([snapshot])
    controller.write_status(snapshots=[snapshot], decisions=decisions)

    payload = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert decisions[0].reason != "paused"
    assert payload["paused"] is False
    assert payload["pause_until"] is None
    assert not pause_file.exists()


def test_controller_status_reports_worker_restart_backoff(tmp_path, monkeypatch) -> None:
    class FailingWorker:
        pid = None

        def __init__(
            self,
            *,
            gpu_index: int,
            memory_bytes: int,
            duty_cycle: float,
            program: str,
            hold_mode: str,
            burst_seconds: float = 0.20,
            burst_jitter: float = 0.0,
        ) -> None:
            self.gpu_index = gpu_index
            self.memory_bytes = memory_bytes
            self.duty_cycle = duty_cycle
            self.program = program
            self.hold_mode = hold_mode
            self.burst_seconds = burst_seconds
            self.burst_jitter = burst_jitter

        def start(self, timeout: float = 10.0) -> None:
            raise RuntimeError("cuda unavailable")

        def is_alive(self) -> bool:
            return False

        def stop(self) -> None:
            pass

    monkeypatch.setattr(controller_mod, "WorkerProcess", FailingWorker)
    controller = GuardController(GuardConfig(state_dir=tmp_path, worker_restart_backoff=60.0))
    snapshot = GpuSnapshot(
        index=0,
        name="test",
        memory_total_bytes=80 * 1024**3,
        memory_used_bytes=0,
        utilization=0,
        timestamp=time.time(),
    )
    decision = HolderDecision(
        gpu_index=0,
        action=DecisionAction.HOLD,
        reason="below_target",
        memory_bytes=1024,
        duty_cycle=0.5,
    )

    controller.apply([decision])
    controller.write_status(snapshots=[snapshot], decisions=[decision])

    payload = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    gpu = payload["gpus"][0]
    assert payload["machine"]["worker_backoff_count"] == 1
    assert gpu["worker"] is None
    assert gpu["worker_restart_backoff"]["active"] is True
    assert gpu["worker_restart_backoff"]["remaining_seconds"] > 0


def test_controller_new_external_process_triggers_grace_release(tmp_path) -> None:
    class LiveWorker:
        pid = 4321
        memory_bytes = 1024
        duty_cycle = 0.5
        program = "mixed"
        hold_mode = "balanced"
        exitcode = None

        def is_alive(self) -> bool:
            return True

        def stop(self) -> None:
            pass

    controller = GuardController(GuardConfig(state_dir=tmp_path, process_grace_window=30.0))
    controller._workers[0] = LiveWorker()
    external = GpuProcess(pid=1234, used_memory_bytes=512 * 1024**2, name="train.py")
    snap = GpuSnapshot(
        index=0,
        name="test",
        memory_total_bytes=80 * 1024**3,
        memory_used_bytes=2 * 1024**3,
        utilization=10,
        processes=(external,),
        timestamp=time.time(),
    )

    decisions = controller.decide([snap])
    controller.record_decisions(decisions)
    controller.write_status(snapshots=[snap], decisions=decisions)

    assert decisions[0].action == DecisionAction.RELEASE
    assert decisions[0].reason == "process_grace"
    events = read_events(tmp_path / "events.jsonl")
    assert [event["type"] for event in events] == ["process_grace_start", "decision"]
    assert events[0]["gpu_index"] == 0
    assert events[0]["pids"] == [1234]
    payload = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert payload["machine"]["process_grace_count"] == 1
    assert payload["gpus"][0]["process_grace"]["active"] is True


def test_controller_process_grace_window_can_be_disabled(tmp_path) -> None:
    controller = GuardController(GuardConfig(state_dir=tmp_path, process_grace_window=0.0))
    external = GpuProcess(pid=1234, used_memory_bytes=512 * 1024**2, name="train.py")
    snap = GpuSnapshot(
        index=0,
        name="test",
        memory_total_bytes=80 * 1024**3,
        memory_used_bytes=2 * 1024**3,
        utilization=10,
        processes=(external,),
        timestamp=time.time(),
    )

    decisions = controller.decide([snap])

    assert decisions[0].reason != "process_grace"
    assert not (tmp_path / "events.jsonl").exists()


def test_controller_keeps_thermal_block_until_resume_temperature(tmp_path) -> None:
    controller = GuardController(
        GuardConfig(
            state_dir=tmp_path,
            max_gpu_temp=85,
            thermal_resume_temp=80,
        )
    )

    hot = GpuSnapshot(
        index=0,
        name="test",
        memory_total_bytes=80 * 1024**3,
        memory_used_bytes=2 * 1024**3,
        utilization=10,
        temperature_celsius=86,
        timestamp=100.0,
    )
    warm = GpuSnapshot(
        index=0,
        name="test",
        memory_total_bytes=80 * 1024**3,
        memory_used_bytes=2 * 1024**3,
        utilization=10,
        temperature_celsius=83,
        timestamp=162.0,
    )
    cooled = GpuSnapshot(
        index=0,
        name="test",
        memory_total_bytes=80 * 1024**3,
        memory_used_bytes=2 * 1024**3,
        utilization=10,
        temperature_celsius=80,
        timestamp=222.0,
    )

    hot_decision = controller.decide([hot])[0]
    warm_decision = controller.decide([warm])[0]
    cooled_decision = controller.decide([cooled])[0]
    controller.write_status(snapshots=[warm], decisions=[warm_decision])

    assert hot_decision.reason == "thermal_limit"
    assert warm_decision.reason == "thermal_limit"
    assert cooled_decision.action == DecisionAction.HOLD
    assert cooled_decision.reason == "low_util_emergency"
    events = read_events(tmp_path / "events.jsonl")
    assert [event["type"] for event in events] == ["thermal_limit_start", "thermal_limit_end"]
    payload = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert payload["machine"]["thermal_limit_count"] == 1
    assert payload["gpus"][0]["thermal_limit"] is False
    assert payload["gpus"][0]["thermal_blocked"] is True


def test_controller_disabled_gpu_releases_holder_and_status_marks_gpu(tmp_path) -> None:
    from gpu_holder.state import disable_gpu

    class LiveWorker:
        pid = 4321
        memory_bytes = 1024
        duty_cycle = 0.5
        program = "mixed"
        hold_mode = "balanced"
        exitcode = None

        def is_alive(self) -> bool:
            return True

        def stop(self) -> None:
            pass

    disable_gpu(tmp_path, 0)
    controller = GuardController(GuardConfig(state_dir=tmp_path))
    controller._workers[0] = LiveWorker()
    snap = GpuSnapshot(
        index=0,
        name="test",
        memory_total_bytes=80 * 1024**3,
        memory_used_bytes=2 * 1024**3,
        utilization=10,
        timestamp=time.time(),
    )

    decisions = controller.decide([snap])
    controller.write_status(snapshots=[snap], decisions=decisions)

    assert decisions[0].action == DecisionAction.RELEASE
    assert decisions[0].reason == "gpu_disabled"
    payload = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert payload["disabled_gpus"] == [0]
    assert payload["gpus"][0]["gpu_disabled"] is True


def test_controller_status_includes_temporary_disabled_gpu_expirations(tmp_path) -> None:
    from gpu_holder.state import disable_gpu

    disable_gpu(tmp_path, 0, duration_seconds=600)
    controller = GuardController(GuardConfig(state_dir=tmp_path))
    snap = GpuSnapshot(
        index=0,
        name="test",
        memory_total_bytes=80 * 1024**3,
        memory_used_bytes=2 * 1024**3,
        utilization=10,
        timestamp=time.time(),
    )

    decisions = controller.decide([snap])
    controller.write_status(snapshots=[snap], decisions=decisions)

    payload = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert payload["disabled_gpus"] == [0]
    assert payload["disabled_gpu_expirations"]["0"] > time.time()
    assert payload["gpus"][0]["gpu_disabled_until"] == payload["disabled_gpu_expirations"]["0"]


def test_event_log_reads_latest_events(tmp_path) -> None:
    path = tmp_path / "events.jsonl"

    write_event(path, "controller_start", pid=1)
    write_event(path, "decision", gpu_index=0, action="hold")
    write_event(path, "worker_stop", gpu_index=0, pid=2)

    events = read_events(path, limit=2)

    assert [event["type"] for event in events] == ["decision", "worker_stop"]
    assert events[-1]["pid"] == 2


def test_event_log_rotates_when_size_limit_is_reached(tmp_path) -> None:
    path = tmp_path / "events.jsonl"

    for index in range(8):
        write_event(path, "decision", max_bytes=160, backup_count=2, gpu_index=index)

    assert path.exists()
    assert (tmp_path / "events.jsonl.1").exists()
    assert (tmp_path / "events.jsonl.2").exists()
    assert not (tmp_path / "events.jsonl.3").exists()
    assert read_events(path)[-1]["gpu_index"] == 7


def test_controller_events_use_configured_rotation(tmp_path) -> None:
    controller = GuardController(
        GuardConfig(
            state_dir=tmp_path,
            event_log_max_bytes=120,
            event_log_backup_count=1,
        )
    )

    for index in range(4):
        controller.record_decisions(
            [
                HolderDecision(
                    gpu_index=0,
                    action=DecisionAction.HOLD,
                    reason=f"reason_{index}",
                )
            ]
        )

    assert (tmp_path / "events.jsonl").exists()
    assert (tmp_path / "events.jsonl.1").exists()
    assert not (tmp_path / "events.jsonl.2").exists()


def test_controller_records_decision_changes_only(tmp_path) -> None:
    controller = GuardController(GuardConfig(state_dir=tmp_path))
    first = HolderDecision(
        gpu_index=0,
        action=DecisionAction.HOLD,
        reason="below_target",
        memory_bytes=1024,
        duty_cycle=0.25,
    )
    changed = HolderDecision(
        gpu_index=0,
        action=DecisionAction.RELEASE,
        reason="busy_process",
    )

    controller.record_decisions([first])
    controller.record_decisions([first])
    controller.record_decisions([changed])

    events = read_events(tmp_path / "events.jsonl")
    assert [event["type"] for event in events] == ["decision", "decision"]
    assert events[0]["action"] == "hold"
    assert events[1]["action"] == "release"


def test_controller_logs_worker_start_failure_without_crashing(tmp_path, monkeypatch) -> None:
    seen_timeouts = []

    class FailingWorker:
        pid = None

        def __init__(
            self,
            *,
            gpu_index: int,
            memory_bytes: int,
            duty_cycle: float,
            program: str,
            hold_mode: str,
            burst_seconds: float = 0.20,
            burst_jitter: float = 0.0,
        ) -> None:
            self.gpu_index = gpu_index
            self.memory_bytes = memory_bytes
            self.duty_cycle = duty_cycle
            self.program = program
            self.hold_mode = hold_mode
            self.burst_seconds = burst_seconds
            self.burst_jitter = burst_jitter

        def start(self, timeout: float = 10.0) -> None:
            seen_timeouts.append(timeout)
            raise RuntimeError("cuda oom")

        def stop(self) -> None:
            raise AssertionError("failed workers must not be stopped as running workers")

    monkeypatch.setattr(controller_mod, "WorkerProcess", FailingWorker)
    controller = GuardController(GuardConfig(state_dir=tmp_path, worker_start_timeout=3.5))
    decision = HolderDecision(
        gpu_index=0,
        action=DecisionAction.HOLD,
        reason="below_target",
        memory_bytes=1024,
        duty_cycle=0.5,
    )

    controller.apply([decision])

    assert controller._workers == {}
    event = read_events(tmp_path / "events.jsonl")[-1]
    assert event["type"] == "worker_start_failed"
    assert event["gpu_index"] == 0
    assert event["error_type"] == "RuntimeError"
    assert event["error"] == "cuda oom"
    assert event["retry_after_seconds"] == 30.0
    assert seen_timeouts == [3.5]


def test_controller_backs_off_after_worker_start_failure(tmp_path, monkeypatch) -> None:
    starts = []

    class FailingWorker:
        pid = None

        def __init__(
            self,
            *,
            gpu_index: int,
            memory_bytes: int,
            duty_cycle: float,
            program: str,
            hold_mode: str,
            burst_seconds: float = 0.20,
            burst_jitter: float = 0.0,
        ) -> None:
            self.gpu_index = gpu_index
            self.memory_bytes = memory_bytes
            self.duty_cycle = duty_cycle
            self.program = program
            self.hold_mode = hold_mode
            self.burst_seconds = burst_seconds
            self.burst_jitter = burst_jitter

        def start(self, timeout: float = 10.0) -> None:
            starts.append(self.gpu_index)
            raise RuntimeError("cuda unavailable")

        def is_alive(self) -> bool:
            return False

        def stop(self) -> None:
            pass

    monkeypatch.setattr(controller_mod, "WorkerProcess", FailingWorker)
    controller = GuardController(GuardConfig(state_dir=tmp_path, worker_restart_backoff=60.0))
    decision = HolderDecision(
        gpu_index=0,
        action=DecisionAction.HOLD,
        reason="below_target",
        memory_bytes=1024,
        duty_cycle=0.5,
    )

    controller.apply([decision])
    controller.apply([decision])
    controller.apply([decision])

    events = read_events(tmp_path / "events.jsonl")
    assert starts == [0]
    assert [event["type"] for event in events] == [
        "worker_start_failed",
        "worker_start_backoff",
    ]
    assert events[0]["retry_after_seconds"] == 60.0
    assert events[1]["remaining_seconds"] > 0


def test_controller_clears_worker_start_backoff_on_release(tmp_path, monkeypatch) -> None:
    starts = []

    class FailingWorker:
        pid = None

        def __init__(
            self,
            *,
            gpu_index: int,
            memory_bytes: int,
            duty_cycle: float,
            program: str,
            hold_mode: str,
            burst_seconds: float = 0.20,
            burst_jitter: float = 0.0,
        ) -> None:
            self.gpu_index = gpu_index
            self.memory_bytes = memory_bytes
            self.duty_cycle = duty_cycle
            self.program = program
            self.hold_mode = hold_mode
            self.burst_seconds = burst_seconds
            self.burst_jitter = burst_jitter

        def start(self, timeout: float = 10.0) -> None:
            starts.append(self.gpu_index)
            raise RuntimeError("cuda unavailable")

        def is_alive(self) -> bool:
            return False

        def stop(self) -> None:
            pass

    monkeypatch.setattr(controller_mod, "WorkerProcess", FailingWorker)
    controller = GuardController(GuardConfig(state_dir=tmp_path, worker_restart_backoff=60.0))
    hold = HolderDecision(
        gpu_index=0,
        action=DecisionAction.HOLD,
        reason="below_target",
        memory_bytes=1024,
        duty_cycle=0.5,
    )
    release = HolderDecision(
        gpu_index=0,
        action=DecisionAction.RELEASE,
        reason="busy_process",
    )

    controller.apply([hold])
    controller.apply([release])
    controller.apply([hold])

    assert starts == [0, 0]


def test_controller_reaps_dead_owned_workers(tmp_path) -> None:
    class DeadWorker:
        gpu_index = 0
        memory_bytes = 1024
        duty_cycle = 0.5
        program = "mixed"
        hold_mode = "balanced"
        pid = 4321
        exitcode = 7

        def is_alive(self) -> bool:
            return False

        def stop(self) -> None:
            raise AssertionError("dead workers should be removed without stop")

    controller = GuardController(GuardConfig(state_dir=tmp_path))
    controller._workers[0] = DeadWorker()

    controller.reap_workers()

    assert controller._workers == {}
    event = read_events(tmp_path / "events.jsonl")[-1]
    assert event["type"] == "worker_exit"
    assert event["gpu_index"] == 0
    assert event["pid"] == 4321
    assert event["exitcode"] == 7


def test_controller_keeps_worker_when_duty_change_is_within_threshold(tmp_path) -> None:
    class LiveWorker:
        gpu_index = 0
        memory_bytes = 1024
        duty_cycle = 0.50
        program = "mixed"
        hold_mode = "balanced"
        burst_seconds = 0.20
        burst_jitter = 0.20
        pid = 4321
        exitcode = None
        stopped = False

        def is_alive(self) -> bool:
            return True

        def stop(self) -> None:
            self.stopped = True

    worker = LiveWorker()
    controller = GuardController(GuardConfig(state_dir=tmp_path, worker_update_duty_threshold=0.05))
    controller._workers[0] = worker
    decision = HolderDecision(
        gpu_index=0,
        action=DecisionAction.HOLD,
        reason="below_target",
        memory_bytes=1024,
        duty_cycle=0.53,
    )

    controller.apply([decision])

    assert controller._workers[0] is worker
    assert worker.stopped is False


def test_controller_restarts_worker_when_duty_change_exceeds_threshold(
    tmp_path,
    monkeypatch,
) -> None:
    class ExistingWorker:
        gpu_index = 0
        memory_bytes = 1024
        duty_cycle = 0.50
        program = "mixed"
        hold_mode = "balanced"
        burst_seconds = 0.20
        burst_jitter = 0.20
        pid = 4321
        exitcode = None
        stopped = False

        def is_alive(self) -> bool:
            return True

        def stop(self) -> None:
            self.stopped = True

    class ReplacementWorker:
        pid = 9876

        def __init__(
            self,
            *,
            gpu_index: int,
            memory_bytes: int,
            duty_cycle: float,
            program: str,
            hold_mode: str,
            burst_seconds: float = 0.20,
            burst_jitter: float = 0.20,
        ) -> None:
            self.gpu_index = gpu_index
            self.memory_bytes = memory_bytes
            self.duty_cycle = duty_cycle
            self.program = program
            self.hold_mode = hold_mode
            self.burst_seconds = burst_seconds
            self.burst_jitter = burst_jitter

        def start(self, timeout: float = 10.0) -> None:
            pass

        def is_alive(self) -> bool:
            return True

        def stop(self) -> None:
            pass

    existing = ExistingWorker()
    monkeypatch.setattr(controller_mod, "WorkerProcess", ReplacementWorker)
    controller = GuardController(GuardConfig(state_dir=tmp_path, worker_update_duty_threshold=0.05))
    controller._workers[0] = existing
    decision = HolderDecision(
        gpu_index=0,
        action=DecisionAction.HOLD,
        reason="below_target",
        memory_bytes=1024,
        duty_cycle=0.60,
    )

    controller.apply([decision])

    assert existing.stopped is True
    assert isinstance(controller._workers[0], ReplacementWorker)
    assert controller._workers[0].duty_cycle == 0.60


def test_controller_restarts_after_dead_worker_is_reaped(tmp_path, monkeypatch) -> None:
    class DeadWorker:
        gpu_index = 0
        memory_bytes = 1024
        duty_cycle = 0.5
        program = "mixed"
        hold_mode = "balanced"
        pid = 4321
        exitcode = 1

        def is_alive(self) -> bool:
            return False

        def stop(self) -> None:
            raise AssertionError("dead workers should be removed without stop")

    class LiveWorker:
        pid = 9876

        def __init__(
            self,
            *,
            gpu_index: int,
            memory_bytes: int,
            duty_cycle: float,
            program: str,
            hold_mode: str,
            burst_seconds: float = 0.20,
            burst_jitter: float = 0.0,
        ) -> None:
            self.gpu_index = gpu_index
            self.memory_bytes = memory_bytes
            self.duty_cycle = duty_cycle
            self.program = program
            self.hold_mode = hold_mode
            self.burst_seconds = burst_seconds
            self.burst_jitter = burst_jitter

        def start(self, timeout: float = 10.0) -> None:
            pass

        def is_alive(self) -> bool:
            return True

        def stop(self) -> None:
            pass

    monkeypatch.setattr(controller_mod, "WorkerProcess", LiveWorker)
    controller = GuardController(GuardConfig(state_dir=tmp_path))
    controller._workers[0] = DeadWorker()
    decision = HolderDecision(
        gpu_index=0,
        action=DecisionAction.HOLD,
        reason="below_target",
        memory_bytes=1024,
        duty_cycle=0.5,
    )

    controller.apply([decision])

    assert controller._workers[0].pid == 9876
    events = read_events(tmp_path / "events.jsonl")
    assert [event["type"] for event in events] == ["worker_exit", "worker_start"]
