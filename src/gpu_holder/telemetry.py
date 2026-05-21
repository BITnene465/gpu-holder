from __future__ import annotations

import subprocess
from typing import Protocol

from .models import GpuProcess, GpuSnapshot


class SnapshotWorker(Protocol):
    memory_bytes: int

    @property
    def pid(self) -> int | None: ...


def read_snapshots(workers: dict[int, SnapshotWorker]) -> list[GpuSnapshot]:
    rows = run_csv(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,utilization.gpu,memory.total,memory.used,memory.free,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus: list[GpuSnapshot] = []
    for row in rows:
        if len(row) < 8:
            continue
        gpus.append(
            GpuSnapshot(
                index=int(row[0]),
                uuid=row[1],
                name=row[2],
                utilization=int(float(row[3] or 0)),
                memory_total=mib(row[4]),
                memory_used=mib(row[5]),
                memory_free=mib(row[6]),
                temperature=None if row[7] in {"", "[N/A]"} else int(float(row[7])),
                processes=[],
            )
        )
    by_uuid = {gpu.uuid: gpu for gpu in gpus}
    worker_pids = {worker.pid for worker in workers.values() if worker.pid}
    worker_mem = {gpu: worker.memory_bytes for gpu, worker in workers.items()}
    for row in run_process_rows():
        if len(row) < 4 or row[0] not in by_uuid:
            continue
        gpu = by_uuid[row[0]]
        pid = int(float(row[1]))
        used = mib(row[2])
        is_holder = pid in worker_pids or abs(used - worker_mem.get(gpu.index, -1)) < 1024**3
        gpu.processes.append(GpuProcess(pid=pid, used_memory=used, name=row[3], is_holder=is_holder))
    return gpus


def select_gpus(snapshots: list[GpuSnapshot], gpus: tuple[int, ...] | str) -> list[GpuSnapshot]:
    if gpus == "all":
        return snapshots
    wanted = set(gpus)
    return [gpu for gpu in snapshots if gpu.index in wanted]


def check_nvidia_smi() -> dict[str, object]:
    try:
        count = len(run_csv(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"]))
    except RuntimeError as exc:
        return {"name": "nvidia-smi", "ok": False, "detail": str(exc)}
    return {"name": "nvidia-smi", "ok": count > 0, "detail": f"gpu_count={count}"}


def run_process_rows() -> list[list[str]]:
    try:
        return run_csv(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
                "--format=csv,noheader,nounits",
            ]
        )
    except RuntimeError:
        return []


def run_csv(command: list[str]) -> list[list[str]]:
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"failed to run {' '.join(command)}: {exc}") from exc
    rows = []
    for line in result.stdout.splitlines():
        if line.strip():
            rows.append([part.strip() for part in line.split(",")])
    return rows


def mib(raw: str) -> int:
    if raw in {"", "[N/A]"}:
        return 0
    return int(float(raw)) * 1024**2

