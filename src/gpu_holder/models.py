from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GpuProcess:
    pid: int
    used_memory: int
    name: str
    is_holder: bool = False


@dataclass
class GpuSnapshot:
    index: int
    uuid: str
    name: str
    utilization: int
    memory_total: int
    memory_used: int
    memory_free: int
    temperature: int | None
    processes: list[GpuProcess]


@dataclass
class Decision:
    gpu_index: int
    action: str
    reason: str
    memory_bytes: int
    duty_cycle: float
    hold_mode: str
