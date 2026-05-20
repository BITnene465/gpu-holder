from __future__ import annotations

import sys
from types import SimpleNamespace

import gpu_holder.monitor as monitor_mod
from gpu_holder.monitor import NvmlMonitor


class FakeNvml:
    NVML_TEMPERATURE_GPU = 0

    def __init__(self) -> None:
        self.initialized = False
        self.shutdown = False

    def nvmlInit(self) -> None:
        self.initialized = True

    def nvmlShutdown(self) -> None:
        self.shutdown = True

    def nvmlDeviceGetCount(self) -> int:
        return 1

    def nvmlDeviceGetHandleByIndex(self, index: int) -> str:
        return f"handle-{index}"

    def nvmlDeviceGetName(self, handle: object) -> bytes:
        return b"Fake GPU"

    def nvmlDeviceGetMemoryInfo(self, handle: object) -> object:
        return SimpleNamespace(total=80 * 1024**3, used=4 * 1024**3)

    def nvmlDeviceGetUtilizationRates(self, handle: object) -> object:
        return SimpleNamespace(gpu=42)

    def nvmlDeviceGetTemperature(self, handle: object, kind: int) -> int:
        raise RuntimeError("temperature unavailable")

    def nvmlDeviceGetComputeRunningProcesses(self, handle: object) -> list[object]:
        return [
            SimpleNamespace(pid=1234, usedGpuMemory=1024),
            SimpleNamespace(pid=2222, usedGpuMemory=None),
        ]

    def nvmlDeviceGetGraphicsRunningProcesses(self, handle: object) -> list[object]:
        return [
            SimpleNamespace(pid=1234, usedGpuMemory=2048),
            SimpleNamespace(pid=3333),
        ]


def test_nvml_monitor_deduplicates_processes_and_marks_holders(monkeypatch) -> None:
    fake_nvml = FakeNvml()
    monkeypatch.setitem(__import__("sys").modules, "pynvml", fake_nvml)
    monkeypatch.setattr(monitor_mod, "_process_name", lambda pid: f"proc-{pid}")

    monitor = NvmlMonitor(holder_pids={1234})
    try:
        snapshot = monitor.snapshots((0,))[0]
    finally:
        monitor.close()

    processes = {process.pid: process for process in snapshot.processes}
    assert fake_nvml.initialized is True
    assert fake_nvml.shutdown is True
    assert snapshot.name == "Fake GPU"
    assert snapshot.utilization == 42
    assert snapshot.temperature_celsius is None
    assert len(snapshot.processes) == 3
    assert sorted(processes) == [1234, 2222, 3333]
    assert processes[1234].used_memory_bytes == 2048
    assert processes[1234].is_holder is True
    assert processes[1234].name == "proc-1234"
    assert processes[2222].used_memory_bytes == 0
    assert processes[3333].used_memory_bytes == 0


def test_nvml_monitor_updates_holder_pids(monkeypatch) -> None:
    fake_nvml = FakeNvml()
    monkeypatch.setitem(__import__("sys").modules, "pynvml", fake_nvml)
    monkeypatch.setattr(monitor_mod, "_process_name", lambda pid: "")

    monitor = NvmlMonitor(holder_pids=set())
    try:
        monitor.update_holder_pids({3333})
        snapshot = monitor.snapshots((0,))[0]
    finally:
        monitor.close()

    processes = {process.pid: process for process in snapshot.processes}
    assert processes[3333].is_holder is True
    assert processes[1234].is_holder is False


def test_nvidia_smi_fallback_reads_snapshots_and_processes(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", None)
    monkeypatch.setattr(monitor_mod, "_process_name", lambda pid: f"proc-{pid}")

    def fake_run_nvidia_smi(args: list[str]) -> str:
        query = args[0]
        if query.startswith("--query-gpu="):
            return "\n".join(
                [
                    "0, GPU-0000, Fake GPU 0, 81920, 1024, 5, 41",
                    "1, GPU-1111, Fake GPU 1, 81920, 2048, 77, 52",
                ]
            )
        if query.startswith("--query-compute-apps="):
            return "\n".join(
                [
                    "GPU-1111, 2222, 3072",
                    "GPU-1111, 2222, 4096",
                    "GPU-1111, [Not Found], 8192",
                ]
            )
        raise AssertionError(args)

    monkeypatch.setattr(monitor_mod, "_run_nvidia_smi", fake_run_nvidia_smi)

    monitor = NvmlMonitor(holder_pids={2222})
    try:
        assert monitor.gpu_count() == 2
        snapshot = monitor.snapshots((1,))[0]
    finally:
        monitor.close()

    assert snapshot.index == 1
    assert snapshot.name == "Fake GPU 1"
    assert snapshot.memory_total_bytes == 81920 * 1024**2
    assert snapshot.memory_used_bytes == 2048 * 1024**2
    assert snapshot.utilization == 77
    assert snapshot.temperature_celsius == 52
    assert len(snapshot.processes) == 1
    process = snapshot.processes[0]
    assert process.pid == 2222
    assert process.used_memory_bytes == 4096 * 1024**2
    assert process.name == "proc-2222"
    assert process.is_holder is True
