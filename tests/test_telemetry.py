from __future__ import annotations

import subprocess

import pytest

import gpu_holder.telemetry as telemetry_mod
from gpu_holder.telemetry import check_nvidia_smi
from gpu_holder.telemetry import mib
from gpu_holder.telemetry import read_snapshots
from gpu_holder.telemetry import run_csv
from gpu_holder.telemetry import select_gpus


class FakeWorker:
    def __init__(self, *, pid: int | None, memory_bytes: int) -> None:
        self.pid = pid
        self.memory_bytes = memory_bytes


def test_read_snapshots_parses_gpu_and_compute_process_rows(monkeypatch) -> None:
    def fake_run_csv(command: list[str]) -> list[list[str]]:
        query = command[1]
        if query.startswith("--query-gpu="):
            return [
                ["0", "GPU-0", "NVIDIA A", "42", "81920", "1024", "80896", "45"],
                ["1", "GPU-1", "NVIDIA B", "0", "81920", "0", "81920", "[N/A]"],
            ]
        if query.startswith("--query-compute-apps="):
            return [
                ["GPU-0", "123", "5120", "python"],
                ["GPU-1", "456", "1024", "gpu-holder"],
                ["GPU-unknown", "999", "1", "ignored"],
            ]
        raise AssertionError(command)

    monkeypatch.setattr(telemetry_mod, "run_csv", fake_run_csv)

    snapshots = read_snapshots({1: FakeWorker(pid=456, memory_bytes=1024 * 1024**2)})

    assert [gpu.index for gpu in snapshots] == [0, 1]
    assert snapshots[0].memory_total == 81920 * 1024**2
    assert snapshots[0].temperature == 45
    assert snapshots[1].temperature is None
    assert snapshots[0].processes[0].is_holder is False
    assert snapshots[1].processes[0].is_holder is True


def test_select_gpus_filters_by_requested_indices() -> None:
    snapshots = [
        telemetry_mod.GpuSnapshot(0, "GPU-0", "A", 0, 1, 0, 1, None, []),
        telemetry_mod.GpuSnapshot(1, "GPU-1", "B", 0, 1, 0, 1, None, []),
    ]

    assert select_gpus(snapshots, "all") == snapshots
    assert [gpu.index for gpu in select_gpus(snapshots, (1,))] == [1]


def test_run_csv_parses_rows_and_wraps_subprocess_errors(monkeypatch) -> None:
    class Result:
        stdout = "0, GPU-0\n\n1, GPU-1\n"

    monkeypatch.setattr(telemetry_mod.subprocess, "run", lambda *args, **kwargs: Result())

    assert run_csv(["nvidia-smi"]) == [["0", "GPU-0"], ["1", "GPU-1"]]

    def fail_run(*args: object, **kwargs: object) -> object:
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=10)

    monkeypatch.setattr(telemetry_mod.subprocess, "run", fail_run)

    with pytest.raises(RuntimeError, match="failed to run nvidia-smi"):
        run_csv(["nvidia-smi"])


def test_check_nvidia_smi_reports_gpu_count(monkeypatch) -> None:
    monkeypatch.setattr(telemetry_mod, "run_csv", lambda command: [["0"], ["1"]])

    assert check_nvidia_smi() == {"name": "nvidia-smi", "ok": True, "detail": "gpu_count=2"}


def test_mib_parses_nvidia_smi_memory_values() -> None:
    assert mib("[N/A]") == 0
    assert mib("") == 0
    assert mib("1024") == 1024 * 1024**2

