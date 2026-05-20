from __future__ import annotations

import csv
import subprocess
import time
from io import StringIO

from .models import GpuProcess, GpuSnapshot


class NvmlMonitor:
    def __init__(self, holder_pids: set[int] | None = None) -> None:
        try:
            import pynvml  # type: ignore
        except ImportError:
            self._nvml = None
            self._fallback = NvidiaSmiMonitor(holder_pids=holder_pids)
            return
        self._nvml = pynvml
        self._fallback = None
        self._holder_pids = holder_pids or set()
        self._holder_memory_by_gpu: dict[int, int] = {}
        self._nvml.nvmlInit()

    def close(self) -> None:
        if self._fallback is not None:
            self._fallback.close()
            return
        self._nvml.nvmlShutdown()

    def gpu_count(self) -> int:
        if self._fallback is not None:
            return self._fallback.gpu_count()
        return int(self._nvml.nvmlDeviceGetCount())

    def snapshots(self, indices: tuple[int, ...] | None = None) -> list[GpuSnapshot]:
        if self._fallback is not None:
            return self._fallback.snapshots(indices)
        selected = indices if indices is not None else tuple(range(self.gpu_count()))
        now = time.time()
        return [self._snapshot(index, now=now) for index in selected]

    def update_holder_pids(
        self,
        pids: set[int],
        holder_memory_by_gpu: dict[int, int] | None = None,
    ) -> None:
        if self._fallback is not None:
            self._fallback.update_holder_pids(pids, holder_memory_by_gpu=holder_memory_by_gpu)
            return
        self._holder_pids = set(pids)
        self._holder_memory_by_gpu = dict(holder_memory_by_gpu or {})

    def _snapshot(self, index: int, *, now: float) -> GpuSnapshot:
        handle = self._nvml.nvmlDeviceGetHandleByIndex(int(index))
        name = self._decode(self._nvml.nvmlDeviceGetName(handle))
        memory = self._nvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = self._nvml.nvmlDeviceGetUtilizationRates(handle)
        temperature = self._temperature(handle)
        processes = self._processes(handle)
        return GpuSnapshot(
            index=int(index),
            name=name,
            memory_total_bytes=int(memory.total),
            memory_used_bytes=int(memory.used),
            utilization=int(utilization.gpu),
            temperature_celsius=temperature,
            processes=tuple(processes),
            timestamp=now,
        )

    def _temperature(self, handle: object) -> int | None:
        getter = getattr(self._nvml, "nvmlDeviceGetTemperature", None)
        temp_kind = getattr(self._nvml, "NVML_TEMPERATURE_GPU", None)
        if getter is None or temp_kind is None:
            return None
        try:
            return int(getter(handle, temp_kind))
        except Exception:
            return None

    def _processes(self, handle: object) -> list[GpuProcess]:
        processes_by_pid: dict[int, GpuProcess] = {}
        for getter_name in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
            getter = getattr(self._nvml, getter_name, None)
            if getter is None:
                continue
            try:
                raw_processes = getter(handle)
            except Exception:
                continue
            for raw in raw_processes:
                pid = int(raw.pid)
                used = _process_used_memory(raw)
                previous = processes_by_pid.get(pid)
                processes_by_pid[pid] = GpuProcess(
                    pid=pid,
                    used_memory_bytes=max(
                        used,
                        previous.used_memory_bytes if previous is not None else 0,
                    ),
                    name=previous.name if previous is not None else _process_name(pid),
                    is_holder=pid in self._holder_pids,
                )
        return [processes_by_pid[pid] for pid in sorted(processes_by_pid)]

    @staticmethod
    def _decode(value: object) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)


class NvidiaSmiMonitor:
    def __init__(self, holder_pids: set[int] | None = None) -> None:
        self._holder_pids = holder_pids or set()
        self._holder_memory_by_gpu: dict[int, int] = {}
        self._gpu_rows()

    def close(self) -> None:
        return

    def gpu_count(self) -> int:
        return len(self._gpu_rows())

    def snapshots(self, indices: tuple[int, ...] | None = None) -> list[GpuSnapshot]:
        selected = set(indices) if indices is not None else None
        gpu_rows = self._gpu_rows()
        index_by_uuid = {
            row["uuid"]: index
            for row in gpu_rows
            if (index := _parse_int(row["index"])) is not None
        }
        process_map = self._processes_by_uuid(index_by_uuid=index_by_uuid)
        now = time.time()
        snapshots = []
        for row in gpu_rows:
            index = _parse_int(row["index"])
            if index is None:
                continue
            if selected is not None and index not in selected:
                continue
            uuid = row["uuid"]
            snapshots.append(
                GpuSnapshot(
                    index=index,
                    name=row["name"],
                    memory_total_bytes=_mib_to_bytes(row["memory_total"]),
                    memory_used_bytes=_mib_to_bytes(row["memory_used"]),
                    utilization=_parse_int(row["utilization"]) or 0,
                    temperature_celsius=_parse_int(row["temperature"]),
                    processes=tuple(process_map.get(uuid, ())),
                    timestamp=now,
                )
            )
        return snapshots

    def update_holder_pids(
        self,
        pids: set[int],
        holder_memory_by_gpu: dict[int, int] | None = None,
    ) -> None:
        self._holder_pids = set(pids)
        self._holder_memory_by_gpu = dict(holder_memory_by_gpu or {})

    def _gpu_rows(self) -> list[dict[str, str]]:
        output = _run_nvidia_smi(
            [
                "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ]
        )
        rows = []
        for fields in _csv_rows(output):
            if len(fields) != 7:
                continue
            rows.append(
                {
                    "index": fields[0],
                    "uuid": fields[1],
                    "name": fields[2],
                    "memory_total": fields[3],
                    "memory_used": fields[4],
                    "utilization": fields[5],
                    "temperature": fields[6],
                }
            )
        return rows

    def _processes_by_uuid(
        self,
        *,
        index_by_uuid: dict[str, int],
    ) -> dict[str, tuple[GpuProcess, ...]]:
        try:
            output = _run_nvidia_smi(
                [
                    "--query-compute-apps=gpu_uuid,pid,used_memory",
                    "--format=csv,noheader,nounits",
                ]
            )
        except RuntimeError:
            return {}
        processes_by_uuid: dict[str, dict[int, GpuProcess]] = {}
        for fields in _csv_rows(output):
            if len(fields) != 3:
                continue
            uuid, pid_text, used_text = fields
            pid = _parse_int(pid_text)
            if pid is None:
                continue
            gpu_index = index_by_uuid.get(uuid)
            existing = processes_by_uuid.setdefault(uuid, {})
            previous = existing.get(pid)
            used = _mib_to_bytes(used_text)
            process_name = previous.name if previous is not None else _process_name(pid)
            existing[pid] = GpuProcess(
                pid=pid,
                used_memory_bytes=max(
                    used,
                    previous.used_memory_bytes if previous is not None else 0,
                ),
                name=process_name,
                is_holder=(
                    pid in self._holder_pids
                    or _matches_hidden_holder_process(
                        gpu_index=gpu_index,
                        pid=pid,
                        used_memory_bytes=used,
                        process_name=process_name,
                        holder_memory_by_gpu=self._holder_memory_by_gpu,
                    )
                ),
            )
        return {
            uuid: tuple(processes[pid] for pid in sorted(processes))
            for uuid, processes in processes_by_uuid.items()
        }


def _process_name(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/comm", encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def _process_used_memory(raw: object) -> int:
    try:
        return max(0, int(getattr(raw, "usedGpuMemory", 0) or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _matches_hidden_holder_process(
    *,
    gpu_index: int | None,
    pid: int,
    used_memory_bytes: int,
    process_name: str,
    holder_memory_by_gpu: dict[int, int],
) -> bool:
    if gpu_index is None or gpu_index not in holder_memory_by_gpu:
        return False
    if pid in holder_memory_by_gpu:
        return False
    if process_name:
        return False
    expected = max(0, int(holder_memory_by_gpu[gpu_index]))
    if expected <= 0:
        return False
    lower_bound = max(0, expected - 1024**3)
    upper_bound = expected + 2 * 1024**3
    return lower_bound <= int(used_memory_bytes) <= upper_bound


def _run_nvidia_smi(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"nvidia-smi query failed: {exc}") from exc
    return result.stdout


def _csv_rows(text: str) -> list[list[str]]:
    return [
        [value.strip() for value in row]
        for row in csv.reader(StringIO(text))
        if row and any(value.strip() for value in row)
    ]


def _parse_int(value: str) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _mib_to_bytes(value: str) -> int:
    parsed = _parse_int(value)
    if parsed is None:
        return 0
    return max(0, parsed) * 1024**2
