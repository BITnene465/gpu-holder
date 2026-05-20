from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import time
from typing import Iterable

from .config import GuardConfig, config_payload
from .events import write_event
from .models import DecisionAction, GpuHistory, GpuProcess, GpuSnapshot, HolderDecision, MachineHistory
from .monitor import NvmlMonitor
from .policy import average_utilization, decide_for_fleet
from .process_matching import matched_protected_process_pattern, normalize_process_patterns
from .state import read_disabled_gpu_state, read_pause_state_file, write_text_atomically
from .units import format_bytes
from .worker import WorkerProcess


class GuardController:
    def __init__(self, config: GuardConfig) -> None:
        self.config = config
        self._workers: dict[int, WorkerProcess] = {}
        self._histories: dict[int, GpuHistory] = {}
        self._machine_history = MachineHistory(window_seconds=self.config.machine_window)
        self._last_decision_signatures: dict[int, tuple[object, ...]] = {}
        self._worker_restart_blocked_until: dict[int, float] = {}
        self._worker_restart_backoff_logged_until: dict[int, float] = {}
        self._seen_non_holder_pids_by_gpu: dict[int, set[int]] = {}
        self._process_grace_until: dict[int, float] = {}
        self._thermal_blocked_gpu_indices: set[int] = set()
        self._stop = False

    def run(self) -> None:
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        self.config.pid_file.write_text(str(os.getpid()), encoding="utf-8")
        self._event("controller_start", pid=os.getpid(), dry_run=self.config.dry_run)
        signal.signal(signal.SIGTERM, self._request_stop)
        signal.signal(signal.SIGINT, self._request_stop)
        monitor = NvmlMonitor()
        try:
            indices = _resolve_gpu_indices(self.config.gpus, gpu_count=monitor.gpu_count())
            while not self._stop:
                self.reap_workers()
                monitor.update_holder_pids(self._holder_pids())
                snapshots = monitor.snapshots(indices)
                decisions = self.decide(snapshots)
                self.record_decisions(decisions)
                if not self.config.dry_run:
                    self.apply(decisions)
                self.write_status(snapshots=snapshots, decisions=decisions)
                if self.config.tui:
                    from .tui import render_once

                    render_once(snapshots=snapshots, decisions=decisions, config=self.config)
                time.sleep(float(self.config.sample_interval))
        finally:
            self.release_all()
            monitor.close()
            self._remove_pid_file_if_current_process()
            self._event("controller_stop", pid=os.getpid())

    def _remove_pid_file_if_current_process(self) -> None:
        try:
            pid = int(self.config.pid_file.read_text(encoding="utf-8").strip())
        except (FileNotFoundError, ValueError):
            return
        if pid != os.getpid():
            return
        try:
            self.config.pid_file.unlink()
        except FileNotFoundError:
            pass

    def decide(self, snapshots: Iterable[GpuSnapshot]) -> list[HolderDecision]:
        self.reap_workers()
        pause_state = read_pause_state_file(self.config.resolved_pause_file)
        paused = pause_state.active
        snapshot_list = list(snapshots)
        disabled_gpu_state = read_disabled_gpu_state(self.config.state_dir)
        for snapshot in snapshot_list:
            history = self._histories.setdefault(
                snapshot.index,
                GpuHistory(window_seconds=self.config.idle_window),
            )
            history.add(snapshot.timestamp, snapshot.utilization)
        machine_average = average_utilization(snapshot_list)
        machine_timestamp = max((snapshot.timestamp for snapshot in snapshot_list), default=time.time())
        self._machine_history.add(machine_timestamp, machine_average)
        machine_policy_average = self._machine_history.average()
        thermal_blocked_gpu_indices = self._update_thermal_blocks(snapshot_list)
        process_grace_gpu_indices = self._update_process_grace(snapshot_list)
        return decide_for_fleet(
            snapshot_list,
            self._histories,
            self.config,
            running_gpu_indices=set(self._workers),
            paused=paused,
            process_grace_gpu_indices=process_grace_gpu_indices,
            disabled_gpu_indices=disabled_gpu_state.active,
            thermal_blocked_gpu_indices=thermal_blocked_gpu_indices,
            machine_policy_average=machine_policy_average,
        )

    def apply(self, decisions: Iterable[HolderDecision]) -> None:
        self.reap_workers()
        for decision in decisions:
            if decision.action in {DecisionAction.RELEASE, DecisionAction.STANDBY}:
                self._clear_worker_restart_backoff(decision.gpu_index)
                self._stop_worker(decision.gpu_index)
                continue
            current = self._workers.get(decision.gpu_index)
            if current is not None:
                same_shape = (
                    current.memory_bytes == decision.memory_bytes
                    and abs(current.duty_cycle - decision.duty_cycle)
                    < self.config.worker_update_duty_threshold
                    and current.program == decision.program
                    and current.hold_mode == decision.hold_mode
                    and abs(current.burst_seconds - self.config.compute_burst_seconds) < 0.001
                    and abs(current.burst_jitter - self.config.compute_burst_jitter) < 0.001
                )
                if same_shape:
                    continue
                self._stop_worker(decision.gpu_index)
            if self._worker_start_blocked(decision.gpu_index):
                self._event_worker_start_backoff(decision)
                continue
            worker = WorkerProcess(
                gpu_index=decision.gpu_index,
                memory_bytes=decision.memory_bytes,
                duty_cycle=decision.duty_cycle,
                program=decision.program,
                hold_mode=decision.hold_mode,
                burst_seconds=self.config.compute_burst_seconds,
                burst_jitter=self.config.compute_burst_jitter,
            )
            try:
                worker.start(timeout=self.config.worker_start_timeout)
            except Exception as exc:
                retry_after = float(self.config.worker_restart_backoff)
                blocked_until = time.monotonic() + retry_after
                self._worker_restart_blocked_until[decision.gpu_index] = blocked_until
                self._worker_restart_backoff_logged_until.pop(decision.gpu_index, None)
                self._event(
                    "worker_start_failed",
                    gpu_index=decision.gpu_index,
                    memory_bytes=decision.memory_bytes,
                    duty_cycle=decision.duty_cycle,
                    program=decision.program,
                    hold_mode=decision.hold_mode,
                    burst_seconds=self.config.compute_burst_seconds,
                    burst_jitter=self.config.compute_burst_jitter,
                    reason=decision.reason,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    retry_after_seconds=retry_after,
                )
                continue
            self._workers[decision.gpu_index] = worker
            self._clear_worker_restart_backoff(decision.gpu_index)
            self._event(
                "worker_start",
                gpu_index=decision.gpu_index,
                pid=worker.pid,
                memory_bytes=decision.memory_bytes,
                duty_cycle=decision.duty_cycle,
                program=decision.program,
                hold_mode=decision.hold_mode,
                burst_seconds=self.config.compute_burst_seconds,
                burst_jitter=self.config.compute_burst_jitter,
                reason=decision.reason,
            )

    def release_all(self) -> None:
        for gpu_index in list(self._workers):
            self._stop_worker(gpu_index)

    def reap_workers(self) -> None:
        for gpu_index, worker in list(self._workers.items()):
            if worker.is_alive():
                continue
            self._workers.pop(gpu_index, None)
            self._event(
                "worker_exit",
                gpu_index=gpu_index,
                pid=worker.pid,
                exitcode=worker.exitcode,
            )

    def record_decisions(self, decisions: Iterable[HolderDecision]) -> None:
        for decision in decisions:
            signature = _decision_signature(decision)
            previous = self._last_decision_signatures.get(decision.gpu_index)
            if previous == signature:
                continue
            self._last_decision_signatures[decision.gpu_index] = signature
            self._event("decision", **asdict(decision))

    def write_status(
        self,
        *,
        snapshots: Iterable[GpuSnapshot],
        decisions: Iterable[HolderDecision],
    ) -> None:
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        pause_state = read_pause_state_file(self.config.resolved_pause_file)
        disabled_gpu_state = read_disabled_gpu_state(self.config.state_dir)
        payload = build_status_payload(
            snapshots=list(snapshots),
            decisions=list(decisions),
            holder_pids=self._holder_pids(),
            paused=pause_state.active,
            pause_until=pause_state.pause_until,
            workers=self._worker_statuses(),
            worker_backoffs=self._worker_backoffs(),
            process_graces=self._process_graces(),
            config=config_payload(self.config),
            machine_window_average=self._machine_history.average(),
            machine_window_seconds=self.config.machine_window,
            machine_window_sample_count=self._machine_history.sample_count(),
            machine_window_coverage_seconds=self._machine_history.coverage_seconds(),
            disabled_gpus=disabled_gpu_state.active,
            disabled_gpu_expirations=disabled_gpu_state.disabled_until_by_gpu,
        )
        write_text_atomically(self.config.status_file, json.dumps(payload, indent=2))

    def _holder_pids(self) -> set[int]:
        return {
            worker.pid
            for worker in self._workers.values()
            if worker.pid is not None and worker.is_alive()
        }

    def _worker_statuses(self) -> dict[int, dict[str, object]]:
        statuses: dict[int, dict[str, object]] = {}
        for gpu_index, worker in self._workers.items():
            statuses[gpu_index] = {
                "gpu_index": gpu_index,
                "pid": getattr(worker, "pid", None),
                "alive": worker.is_alive(),
                "exitcode": getattr(worker, "exitcode", None),
                "memory_bytes": getattr(worker, "memory_bytes", 0),
                "memory_human": format_bytes(getattr(worker, "memory_bytes", 0)),
                "duty_cycle": getattr(worker, "duty_cycle", 0.0),
                "program": getattr(worker, "program", ""),
                "hold_mode": getattr(worker, "hold_mode", ""),
                "burst_seconds": getattr(worker, "burst_seconds", 0.0),
                "burst_jitter": getattr(worker, "burst_jitter", 0.0),
            }
        return statuses

    def _worker_backoffs(self) -> dict[int, dict[str, object]]:
        now = time.monotonic()
        backoffs: dict[int, dict[str, object]] = {}
        for gpu_index, blocked_until in self._worker_restart_blocked_until.items():
            remaining = max(0.0, float(blocked_until) - now)
            if remaining <= 0:
                continue
            backoffs[gpu_index] = {
                "gpu_index": gpu_index,
                "active": True,
                "remaining_seconds": remaining,
            }
        return backoffs

    def _process_graces(self) -> dict[int, dict[str, object]]:
        now = time.time()
        graces: dict[int, dict[str, object]] = {}
        for gpu_index, active_until in self._process_grace_until.items():
            remaining = max(0.0, float(active_until) - now)
            if remaining <= 0:
                continue
            graces[gpu_index] = {
                "gpu_index": gpu_index,
                "active": True,
                "remaining_seconds": remaining,
            }
        return graces

    def _update_process_grace(self, snapshots: list[GpuSnapshot]) -> set[int]:
        now = time.time()
        window = float(self.config.process_grace_window)
        if window <= 0:
            self._seen_non_holder_pids_by_gpu = {
                snapshot.index: {
                    process.pid
                    for process in snapshot.non_holder_processes()
                }
                for snapshot in snapshots
            }
            self._process_grace_until.clear()
            return set()

        active: set[int] = set()
        for snapshot in snapshots:
            current_pids = {process.pid for process in snapshot.non_holder_processes()}
            previous_pids = self._seen_non_holder_pids_by_gpu.get(snapshot.index, set())
            new_pids = current_pids - previous_pids
            self._seen_non_holder_pids_by_gpu[snapshot.index] = current_pids
            if new_pids:
                active_until = now + window
                self._process_grace_until[snapshot.index] = max(
                    active_until,
                    self._process_grace_until.get(snapshot.index, 0.0),
                )
                self._event(
                    "process_grace_start",
                    gpu_index=snapshot.index,
                    pids=sorted(new_pids),
                    duration_seconds=window,
                )
            active_until = self._process_grace_until.get(snapshot.index)
            if active_until is not None and now < active_until:
                active.add(snapshot.index)

        active_snapshot_indices = {snapshot.index for snapshot in snapshots}
        for gpu_index, active_until in list(self._process_grace_until.items()):
            if now >= active_until or gpu_index not in active_snapshot_indices:
                self._process_grace_until.pop(gpu_index, None)
        return active

    def _update_thermal_blocks(self, snapshots: list[GpuSnapshot]) -> set[int]:
        max_temp = self.config.max_gpu_temp
        if max_temp is None:
            self._thermal_blocked_gpu_indices.clear()
            return set()

        active: set[int] = set()
        seen_indices = {snapshot.index for snapshot in snapshots}
        for snapshot in snapshots:
            temperature = snapshot.temperature_celsius
            was_blocked = snapshot.index in self._thermal_blocked_gpu_indices
            if temperature is None:
                if was_blocked:
                    active.add(snapshot.index)
                continue
            if temperature >= max_temp:
                active.add(snapshot.index)
                if not was_blocked:
                    self._event(
                        "thermal_limit_start",
                        gpu_index=snapshot.index,
                        temperature_celsius=temperature,
                        max_gpu_temp=max_temp,
                    )
                continue
            if not was_blocked:
                continue
            resume_temp = self.config.thermal_resume_temp
            should_resume = temperature < max_temp if resume_temp is None else temperature <= resume_temp
            if should_resume:
                self._event(
                    "thermal_limit_end",
                    gpu_index=snapshot.index,
                    temperature_celsius=temperature,
                    max_gpu_temp=max_temp,
                    thermal_resume_temp=resume_temp,
                )
            else:
                active.add(snapshot.index)

        self._thermal_blocked_gpu_indices = active & seen_indices
        return set(self._thermal_blocked_gpu_indices)

    def _stop_worker(self, gpu_index: int) -> None:
        worker = self._workers.pop(gpu_index, None)
        if worker is not None:
            self._event("worker_stop", gpu_index=gpu_index, pid=worker.pid)
            worker.stop()

    def _request_stop(self, signum: int, frame: object) -> None:
        del signum, frame
        self._stop = True

    def _worker_start_blocked(self, gpu_index: int) -> bool:
        blocked_until = self._worker_restart_blocked_until.get(gpu_index)
        if blocked_until is None:
            return False
        if time.monotonic() < blocked_until:
            return True
        self._clear_worker_restart_backoff(gpu_index)
        return False

    def _event_worker_start_backoff(self, decision: HolderDecision) -> None:
        blocked_until = self._worker_restart_blocked_until.get(decision.gpu_index)
        if blocked_until is None:
            return
        if self._worker_restart_backoff_logged_until.get(decision.gpu_index) == blocked_until:
            return
        self._worker_restart_backoff_logged_until[decision.gpu_index] = blocked_until
        self._event(
            "worker_start_backoff",
            gpu_index=decision.gpu_index,
            memory_bytes=decision.memory_bytes,
            duty_cycle=decision.duty_cycle,
            program=decision.program,
            hold_mode=decision.hold_mode,
            reason=decision.reason,
            remaining_seconds=max(0.0, blocked_until - time.monotonic()),
        )

    def _clear_worker_restart_backoff(self, gpu_index: int) -> None:
        self._worker_restart_blocked_until.pop(gpu_index, None)
        self._worker_restart_backoff_logged_until.pop(gpu_index, None)

    def _event(self, event_type: str, **payload: object) -> None:
        write_event(
            self.config.event_log_file,
            event_type,
            max_bytes=self.config.event_log_max_bytes,
            backup_count=self.config.event_log_backup_count,
            **payload,
        )


def build_status_payload(
    *,
    snapshots: list[GpuSnapshot],
    decisions: list[HolderDecision],
    holder_pids: set[int],
    paused: bool,
    pause_until: float | None = None,
    workers: dict[int, dict[str, object]] | None = None,
    worker_backoffs: dict[int, dict[str, object]] | None = None,
    process_graces: dict[int, dict[str, object]] | None = None,
    config: dict[str, object] | None = None,
    machine_window_average: float | None = None,
    machine_window_seconds: float | None = None,
    machine_window_sample_count: int | None = None,
    machine_window_coverage_seconds: float | None = None,
    disabled_gpus: set[int] | None = None,
    disabled_gpu_expirations: dict[int, float] | None = None,
) -> dict[str, object]:
    by_gpu = {decision.gpu_index: decision for decision in decisions}
    worker_by_gpu = workers or {}
    backoff_by_gpu = worker_backoffs or {}
    grace_by_gpu = process_graces or {}
    disabled_gpu_set = disabled_gpus or set()
    disabled_until_by_gpu = disabled_gpu_expirations or {}
    config_payload_data = config or {}
    max_gpu_temp = _configured_max_gpu_temp(config_payload_data)
    protected_patterns = _configured_protected_process_patterns(config_payload_data)
    machine_average = average_utilization(snapshots)
    policy_average = machine_average if machine_window_average is None else machine_window_average
    policy_window_seconds = _positive_float_or_none(machine_window_seconds)
    policy_window_sample_count = (
        None if machine_window_sample_count is None else max(0, int(machine_window_sample_count))
    )
    policy_window_coverage_seconds = _nonnegative_float_or_none(
        machine_window_coverage_seconds
    )
    policy_window_coverage_ratio = _coverage_ratio(
        coverage_seconds=policy_window_coverage_seconds,
        window_seconds=policy_window_seconds,
    )
    action_counts: dict[str, int] = {}
    for decision in decisions:
        action_counts[decision.action.value] = action_counts.get(decision.action.value, 0) + 1
    thermal_limit_count = sum(
        1 for snapshot in snapshots if _thermal_blocked(snapshot, by_gpu, max_gpu_temp)
    )
    return {
        "timestamp": time.time(),
        "paused": bool(paused),
        "pause_until": pause_until,
        "disabled_gpus": sorted(disabled_gpu_set),
        "disabled_gpu_expirations": {
            str(gpu_index): disabled_until_by_gpu[gpu_index]
            for gpu_index in sorted(disabled_until_by_gpu)
        },
        "holder_pids": sorted(holder_pids),
        "config": config_payload_data,
        "machine": {
            "gpu_count": len(snapshots),
            "average_utilization": machine_average,
            "window_average_utilization": policy_average,
            "policy_average_utilization": policy_average,
            "policy_window_seconds": policy_window_seconds,
            "policy_window_sample_count": policy_window_sample_count,
            "policy_window_coverage_seconds": policy_window_coverage_seconds,
            "policy_window_coverage_ratio": policy_window_coverage_ratio,
            "action_counts": action_counts,
            "owned_worker_count": len(worker_by_gpu),
            "worker_backoff_count": len(backoff_by_gpu),
            "process_grace_count": len(grace_by_gpu),
            "thermal_limit_count": thermal_limit_count,
        },
        "gpus": [
            {
                "index": snapshot.index,
                "name": snapshot.name,
                "utilization": snapshot.utilization,
                "temperature_celsius": snapshot.temperature_celsius,
                "thermal_limit": _thermal_limit_active(snapshot.temperature_celsius, max_gpu_temp),
                "thermal_blocked": _thermal_blocked(snapshot, by_gpu, max_gpu_temp),
                "memory_total": snapshot.memory_total_bytes,
                "memory_used": snapshot.memory_used_bytes,
                "memory_total_human": format_bytes(snapshot.memory_total_bytes),
                "memory_used_human": format_bytes(snapshot.memory_used_bytes),
                "processes": [
                    _process_payload(process, protected_patterns=protected_patterns)
                    for process in snapshot.processes
                ],
                "decision": asdict(by_gpu[snapshot.index]) if snapshot.index in by_gpu else None,
                "gpu_disabled": snapshot.index in disabled_gpu_set,
                "gpu_disabled_until": disabled_until_by_gpu.get(snapshot.index),
                "worker": worker_by_gpu.get(snapshot.index),
                "worker_restart_backoff": backoff_by_gpu.get(snapshot.index),
                "process_grace": grace_by_gpu.get(snapshot.index),
            }
            for snapshot in snapshots
        ],
    }


def _configured_max_gpu_temp(config: dict[str, object]) -> int | None:
    raw = config.get("max_gpu_temp")
    if raw is None:
        return None
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _configured_protected_process_patterns(config: dict[str, object]) -> tuple[str, ...]:
    return normalize_process_patterns(config.get("protected_process_patterns"))


def _positive_float_or_none(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _nonnegative_float_or_none(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, parsed)


def _coverage_ratio(
    *,
    coverage_seconds: float | None,
    window_seconds: float | None,
) -> float | None:
    if coverage_seconds is None or window_seconds is None or window_seconds <= 0:
        return None
    return min(1.0, max(0.0, coverage_seconds / window_seconds))


def _process_payload(
    process: GpuProcess,
    *,
    protected_patterns: tuple[str, ...],
) -> dict[str, object]:
    payload = asdict(process)
    pattern = matched_protected_process_pattern(process, protected_patterns)
    payload["protected_process"] = pattern is not None
    payload["protected_process_pattern"] = pattern
    return payload


def _thermal_limit_active(temperature: int | None, max_gpu_temp: int | None) -> bool:
    return temperature is not None and max_gpu_temp is not None and temperature >= max_gpu_temp


def _thermal_blocked(
    snapshot: GpuSnapshot,
    decisions: dict[int, HolderDecision],
    max_gpu_temp: int | None,
) -> bool:
    decision = decisions.get(snapshot.index)
    if decision is not None and decision.reason == "thermal_limit":
        return True
    return _thermal_limit_active(snapshot.temperature_celsius, max_gpu_temp)


def read_status(path: Path) -> dict[str, object] | None:
    payload, _error = read_status_result(path)
    return payload


def read_status_result(path: Path) -> tuple[dict[str, object] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "no status file"
    except json.JSONDecodeError as exc:
        return None, f"invalid status file: {exc.msg} at line {exc.lineno} column {exc.colno}"
    if not isinstance(payload, dict):
        return None, "invalid status file: expected JSON object"
    return payload, None


def _decision_signature(decision: HolderDecision) -> tuple[object, ...]:
    return (
        decision.action.value,
        decision.reason,
        decision.memory_bytes,
        round(decision.duty_cycle, 2),
        decision.program,
        decision.hold_mode,
    )


def _resolve_gpu_indices(raw: tuple[int, ...] | str, *, gpu_count: int) -> tuple[int, ...]:
    if raw == "all":
        return tuple(range(int(gpu_count)))
    return tuple(int(index) for index in raw)
